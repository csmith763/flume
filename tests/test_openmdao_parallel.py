"""Serial/parallel System tests using OpenMDAO groups."""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import time
import unittest
import multiprocessing
from pathlib import Path

import numpy as np
import openmdao.api as om

from flume.base_classes.analysis import Analysis
from flume.base_classes.om_wrapper import OpenMDAOGroupAnalysis
from flume.base_classes.state import State
from flume.base_classes.system import System
from flume.interfaces.scipy_interface import FlumeScipyInterface


class OpenMDAOQuadraticComponent(om.ExplicitComponent):
    """Quadratic scalar discipline with a dense solve in each evaluation."""

    def initialize(self):
        self.options.declare("size", types=int)
        self.options.declare("seed", types=int)
        self.options.declare("output_name", types=str)

    def setup(self):
        size = self.options["size"]
        rng = np.random.default_rng(self.options["seed"])
        factor = rng.standard_normal((size, size)) / np.sqrt(size)
        self._matrix = factor.T @ factor + np.eye(size)
        self._rhs_x = rng.standard_normal(size)
        self._rhs_y = rng.standard_normal(size)
        self._output_name = self.options["output_name"]

        self.add_input("x", val=0.0)
        self.add_input("y", val=0.0)
        self.add_output(self._output_name, val=0.0)
        self.declare_partials(self._output_name, ["x", "y"])

    def compute(self, inputs, outputs):
        x = inputs["x"][0]
        y = inputs["y"][0]
        residual = (x - 1.0) * self._rhs_x + (y - 1.0) * self._rhs_y
        solution = np.linalg.solve(self._matrix, residual)
        rosenbrock = (1.0 - x) ** 2 + 100.0 * (y - x**2) ** 2
        outputs[self._output_name] = rosenbrock + 0.005 * (residual @ solution)

    def compute_partials(self, inputs, partials):
        x = inputs["x"][0]
        y = inputs["y"][0]
        residual = (x - 1.0) * self._rhs_x + (y - 1.0) * self._rhs_y
        response = np.linalg.solve(self._matrix, residual)
        penalty_scale = 0.01
        partials[self._output_name, "x"] = (
            2.0 * (x - 1.0)
            - 400.0 * x * (y - x**2)
            + penalty_scale * (self._rhs_x @ response)
        )
        partials[self._output_name, "y"] = (
            200.0 * (y - x**2)
            + penalty_scale * (self._rhs_y @ response)
        )


class OpenMDAOQuadraticGroup(om.Group):
    def initialize(self):
        self.options.declare("size", types=int)
        self.options.declare("seed", types=int)
        self.options.declare("output_name", types=str)

    def setup(self):
        self.add_subsystem(
            "quadratic",
            OpenMDAOQuadraticComponent(
                size=self.options["size"],
                seed=self.options["seed"],
                output_name=self.options["output_name"],
            ),
            promotes_inputs=["x", "y"],
            promotes_outputs=[self.options["output_name"]],
        )


class DesignSource(Analysis):
    def __init__(self, obj_name, sub_analyses=None):
        self.default_parameters = {}
        super().__init__(obj_name, sub_analyses=sub_analyses or [])
        self.variables = {
            "x": State(value=0.0, desc="design variable", source=self),
            "y": State(value=0.0, desc="design variable", source=self),
        }

    def _analyze(self):
        self.outputs = {
            "x": State(value=self.variables["x"].value, desc="design value", source=self),
            "y": State(value=self.variables["y"].value, desc="design value", source=self),
        }

    def _analyze_adjoint(self):
        for name in ("x", "y"):
            self.variables[name].set_deriv_value(
                self.variables[name].deriv + self.outputs[name].deriv
            )


class SumObjective(Analysis):
    def __init__(self, obj_name, branches, output_names):
        self.default_parameters = {}
        super().__init__(obj_name, sub_analyses=branches)
        self.variables = {
            name: State(value=0.0, desc=name, source=self) for name in output_names
        }

    def _analyze(self):
        self.outputs = {
            "f": State(
                value=float(sum(state.value for state in self.variables.values())),
                desc="objective",
                source=self,
            )
        }

    def _analyze_adjoint(self):
        for state in self.variables.values():
            state.set_deriv_value(state.deriv + self.outputs["f"].deriv)


class TestOpenMDAOParallelExecution(unittest.TestCase):
    N_BRANCHES = 4
    SOLVE_SIZE = 800
    N_TIMING_REPS = 3

    def _build_system(self, parallel, run_type):
        mode = "parallel" if parallel else "serial"
        log_dir = (
            Path(__file__).resolve().parent
            / "openmdao_parallel_logs"
            / f"{run_type}_{mode}"
        )
        log_dir.mkdir(parents=True, exist_ok=True)
        source = DesignSource("design")
        branch_names = [f"branch_{i}" for i in range(self.N_BRANCHES)]
        output_names = [f"y_{i}" for i in range(self.N_BRANCHES)]
        branches = []

        for i, (branch_name, output_name) in enumerate(
            zip(branch_names, output_names)
        ):
            group = OpenMDAOQuadraticGroup(
                size=self.SOLVE_SIZE,
                seed=17 + i,
                output_name=output_name,
            )
            branches.append(
                OpenMDAOGroupAnalysis(
                    om_group=group,
                    group_variable_names=[f"{branch_name}.x", f"{branch_name}.y"],
                    group_output_names=[f"{branch_name}.{output_name}"],
                    obj_name=branch_name,
                    sub_analyses=[source],
                )
            )

        objective = SumObjective("objective", branches, output_names)
        system = System(
            sys_name="openmdao_parallel",
            top_level_analysis_list=[objective],
            log_prefix=str(log_dir),
            parallel_execution=parallel,
            parallel_max_workers=self.N_BRANCHES,
            track_node_timing=True,
        )
        system.declare_design_vars(
            global_var_name={
                "design.x": {"lb": -2.0, "ub": 2.0},
                "design.y": {"lb": -1.0, "ub": 3.0},
            }
        )
        system.declare_objective(global_obj_name="objective.f")
        return system, source, log_dir

    @staticmethod
    def _run_iteration(system, source, x, y):
        source.set_var_values(variables={"x": float(x), "y": float(y)})
        system.reset_analysis_flags()
        system.execute()
        system.execute_adjoint()

    def _time_forward_and_adjoint(self, system, source):
        self._run_iteration(system, source, -1.2, 1.0)
        forward_times = []
        adjoint_times = []
        for iteration in range(self.N_TIMING_REPS):
            source.set_var_values(variables={"x": -1.2, "y": 1.0})
            system.reset_analysis_flags()

            start = time.perf_counter()
            system.execute()
            forward_end = time.perf_counter()
            system.execute_adjoint()
            adjoint_end = time.perf_counter()
            forward_times.append(forward_end - start)
            adjoint_times.append(adjoint_end - forward_end)
            system.log_information(iter_number=iteration)
            system.profile_iteration(iter_number=iteration)

        system.finalize_profile_log()
        objective_grad = system.design_derivs[("objective.f", 0)]
        return (
            float(np.median(forward_times)),
            float(np.median(adjoint_times)),
            system.obj_analysis.outputs["f"].value,
            np.array(
                [
                    objective_grad["design.x"],
                    objective_grad["design.y"],
                ]
            ),
        )

    def _optimize(self, parallel):
        system, source, _ = self._build_system(parallel, "optimization")
        interface = FlumeScipyInterface(system)
        initial_point = interface.set_initial_point(
            initial_global_vars={"design.x": -1.2, "design.y": 1.0}
        )
        start = time.perf_counter()
        solution, result = interface.optimize_system(
            x0=initial_point, method="SLSQP", maxit=30
        )
        elapsed = time.perf_counter() - start
        self._run_iteration(system, source, solution[0], solution[1])
        gradient = system.design_derivs[("objective.f", 0)]
        return (
            np.array(solution),
            float(result.fun),
            elapsed,
            bool(result.success),
            np.array([gradient["design.x"], gradient["design.y"]]),
            int(result.nit),
        )

    def test_forward_and_adjoint_match_and_speed_up(self):
        serial, serial_source, _ = self._build_system(
            parallel=False, run_type="execution"
        )
        parallel, parallel_source, _ = self._build_system(
            parallel=True, run_type="execution"
        )
        serial_fwd, serial_adj, serial_obj, serial_grad = (
            self._time_forward_and_adjoint(serial, serial_source)
        )
        parallel_fwd, parallel_adj, parallel_obj, parallel_grad = (
            self._time_forward_and_adjoint(parallel, parallel_source)
        )

        self.assertAlmostEqual(serial_obj, parallel_obj, places=11)
        np.testing.assert_allclose(
            serial_grad, parallel_grad, rtol=1e-10, atol=1e-12
        )

        fwd_speedup = serial_fwd / parallel_fwd
        adj_speedup = serial_adj / parallel_adj
        print(
            "\n[OpenMDAO System execution] "
            f"forward={fwd_speedup:.2f}x, adjoint={adj_speedup:.2f}x "
            f"(median of {self.N_TIMING_REPS} timed iterations)"
        )

        if multiprocessing.cpu_count() < 2:
            self.skipTest("Only one CPU is available; speedup is not measurable.")
        self.assertGreater(fwd_speedup, 1.05)
        self.assertGreater(adj_speedup, 1.05)

    def test_optimization_matches_and_reports_speedup(self):
        (
            serial_x,
            serial_f,
            serial_time,
            serial_success,
            serial_grad,
            serial_nit,
        ) = self._optimize(False)
        (
            parallel_x,
            parallel_f,
            parallel_time,
            parallel_success,
            parallel_grad,
            parallel_nit,
        ) = self._optimize(True)

        self.assertTrue(serial_success)
        self.assertTrue(parallel_success)
        np.testing.assert_allclose(serial_x, parallel_x, rtol=1e-5, atol=1e-6)
        self.assertAlmostEqual(serial_f, parallel_f, places=10)
        np.testing.assert_allclose(serial_x, [1.0, 1.0], atol=1e-5)
        np.testing.assert_allclose(parallel_x, [1.0, 1.0], atol=1e-5)
        self.assertLess(np.linalg.norm(serial_grad), 5e-4)
        self.assertLess(np.linalg.norm(parallel_grad), 5e-4)
        self.assertGreater(serial_nit, 10)
        self.assertGreater(parallel_nit, 10)

        speedup = serial_time / parallel_time
        print(
            "\n[OpenMDAO optimization] "
            f"serial={serial_time:.3f}s, parallel={parallel_time:.3f}s, "
            f"speedup={speedup:.2f}x, iterations={serial_nit}/{parallel_nit}"
        )
        if multiprocessing.cpu_count() >= 2:
            self.assertGreater(speedup, 1.05)


if __name__ == "__main__":
    unittest.main()
