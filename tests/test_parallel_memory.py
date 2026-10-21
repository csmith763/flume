"""Memory regression checks and an opt-in RSS benchmark for parallel System runs.

Run the deterministic regression with:
    python -m unittest tests.test_parallel_memory

Run the RSS benchmark with:
    FLUME_RUN_MEMORY_BENCHMARK=1 python -m unittest tests.test_parallel_memory
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from flume.base_classes.analysis import Analysis
from flume.base_classes.state import State
from flume.base_classes.system import System


class DesignSource(Analysis):
    def __init__(self, obj_name, sub_analyses=None, **kwargs):
        self.default_parameters = {"n": 4}
        super().__init__(obj_name, sub_analyses=sub_analyses or [], **kwargs)
        self.variables = {
            "x": State(value=np.ones(self.parameters["n"]), desc="design", source=self)
        }

    def _analyze(self):
        self.outputs = {
            "x": State(
                value=np.array(self.variables["x"].value),
                desc="design passthrough",
                source=self,
            )
        }

    def _analyze_adjoint(self):
        self.variables["x"].set_deriv_value(
            self.variables["x"].deriv + self.outputs["x"].deriv
        )


class LinearBranch(Analysis):
    def __init__(self, obj_name, sub_analyses=None, **kwargs):
        self.default_parameters = {"weight": 1.0}
        super().__init__(obj_name, sub_analyses=sub_analyses or [], **kwargs)
        self.weight = self.parameters["weight"]
        self.variables = {
            "x": State(value=np.ones(4), desc="branch input", source=self)
        }

    def _analyze(self):
        self.outputs = {
            "y": State(
                value=float(self.weight * np.sum(self.variables["x"].value)),
                desc="branch output",
                source=self,
            )
        }

    def _analyze_adjoint(self):
        self.variables["x"].set_deriv_value(
            self.variables["x"].deriv
            + self.weight * self.outputs["y"].deriv * np.ones(4)
        )


class SumSink(Analysis):
    def __init__(self, obj_name, sub_analyses=None, **kwargs):
        self.default_parameters = {}
        super().__init__(obj_name, sub_analyses=sub_analyses or [], **kwargs)
        self.variables = {
            branch.obj_name: State(value=0.0, desc="branch output", source=self)
            for branch in self.sub_analyses
        }

    def _analyze(self):
        self.outputs = {
            "value": State(
                value=float(sum(var.value for var in self.variables.values())),
                desc="sink output",
                source=self,
            )
        }

    def _analyze_adjoint(self):
        for var in self.variables.values():
            var.set_deriv_value(var.deriv + self.outputs["value"].deriv)


def _build_system(parallel):
    source = DesignSource("design", n=4)
    branches = [
        LinearBranch("branch_a", [source], weight=1.0),
        LinearBranch("branch_b", [source], weight=2.0),
    ]
    objective = SumSink("objective", branches)
    constraint = SumSink("constraint", branches)

    log_dir = tempfile.TemporaryDirectory(prefix="flume_parallel_memory_")
    system = System(
        sys_name="parallel_memory",
        top_level_analysis_list=[objective, constraint],
        log_prefix=log_dir.name,
        parallel_execution=parallel,
        parallel_max_workers=4,
    )
    system.declare_design_vars(global_var_name={"design.x": {}})
    system.declare_objective(global_obj_name="objective.value")
    system.declare_constraints(
        global_con_name={"constraint.value": {"rhs": 1.0}}
    )
    return system, source, log_dir


def _run_iteration(system, source, x):
    source.set_var_values(variables={"x": x})
    system.reset_analysis_flags()
    system.execute()
    system.execute_adjoint()


def _unique_states(system):
    seen = set()
    states = []
    for analysis in system.full_analysis_list:
        for state in list(analysis.outputs.values()) + list(
            analysis.variables.values()
        ):
            if id(state) not in seen:
                seen.add(id(state))
                states.append(state)
    return states


def _current_rss_bytes():
    rss_kib = subprocess.check_output(
        ["ps", "-o", "rss=", "-p", str(os.getpid())], text=True
    )
    return int(rss_kib.strip()) * 1024


def _rss_worker():
    system, source, log_dir = _build_system(parallel=True)
    x = np.ones(4)

    for _ in range(5):
        _run_iteration(system, source, x)

    checkpoints = []
    completed = 0
    for count in (20, 80, 200):
        for _ in range(count):
            _run_iteration(system, source, x)
        completed += count
        checkpoints.append(
            {
                "iterations": completed,
                "rss_bytes": _current_rss_bytes(),
            }
        )

    print(json.dumps({"warmup_iterations": 5, "checkpoints": checkpoints}))
    log_dir.cleanup()


class TestParallelMemory(unittest.TestCase):
    def test_adjoint_derivative_storage_stays_bounded(self):
        parallel_system, parallel_source, parallel_log_dir = _build_system(
            parallel=True
        )
        serial_system, serial_source, serial_log_dir = _build_system(parallel=False)
        x = np.array([0.5, 1.0, 1.5, 2.0])
        try:
            _run_iteration(parallel_system, parallel_source, x)
            _run_iteration(serial_system, serial_source, x)

            objective_sweep = ("objective.value", 0)
            constraint_sweep = ("constraint.value", 0)
            np.testing.assert_allclose(
                parallel_system.design_derivs[objective_sweep]["design.x"],
                serial_system.design_derivs[objective_sweep]["design.x"],
            )
            np.testing.assert_allclose(
                parallel_system.design_derivs[constraint_sweep]["design.x"],
                serial_system.design_derivs[constraint_sweep]["design.x"],
            )
            self.assertEqual(
                parallel_system.obj_analysis.outputs["value"].value,
                serial_system.obj_analysis.outputs["value"].value,
            )

            baseline_slot_count = sum(
                len(state._deriv_store) for state in _unique_states(parallel_system)
            )
            self.assertGreater(baseline_slot_count, 0)

            for _ in range(40):
                _run_iteration(parallel_system, parallel_source, x)
                current_slot_count = sum(
                    len(state._deriv_store)
                    for state in _unique_states(parallel_system)
                )
                self.assertEqual(
                    current_slot_count,
                    baseline_slot_count,
                    "Parallel adjoint derivative storage grew across iterations.",
                )
        finally:
            parallel_log_dir.cleanup()
            serial_log_dir.cleanup()

    @unittest.skipUnless(
        os.environ.get("FLUME_RUN_MEMORY_BENCHMARK") == "1",
        "Set FLUME_RUN_MEMORY_BENCHMARK=1 to run the RSS growth benchmark.",
    )
    def test_parallel_rss_growth_checkpoints(self):
        if shutil.which("ps") is None:
            self.skipTest("RSS measurement requires the ps process utility.")

        completed = subprocess.run(
            [sys.executable, "-m", "tests.test_parallel_memory", "--rss-worker"],
            check=True,
            capture_output=True,
            text=True,
        )
        report = json.loads(completed.stdout.strip().splitlines()[-1])
        checkpoints = report["checkpoints"]

        self.assertEqual(
            [point["iterations"] for point in checkpoints], [20, 100, 300]
        )
        self.assertTrue(all(point["rss_bytes"] > 0 for point in checkpoints))

        baseline = checkpoints[0]["rss_bytes"]
        measurements = ", ".join(
            f'{point["iterations"]} iterations: '
            f'{point["rss_bytes"] / (1024 * 1024):.2f} MiB '
            f'({(point["rss_bytes"] - baseline) / (1024 * 1024):+.2f} MiB '
            "vs. 20 iterations)"
            for point in checkpoints
        )
        print(f"\n[parallel RSS] after 5 warm-ups; {measurements}")


if __name__ == "__main__":
    if sys.argv[1:] == ["--rss-worker"]:
        _rss_worker()
    else:
        unittest.main()
