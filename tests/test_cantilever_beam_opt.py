from examples.cantilever_beam.independents import Independents
from examples.cantilever_beam.compliance import Compliance
from examples.cantilever_beam.inertia import MomentofInertia
from examples.cantilever_beam.linear_solve import LinearStaticSolve
from examples.cantilever_beam.volume import BeamVolume
import numpy as np
import unittest
from flume.base_classes.system import System
from flume.interfaces.pyoptsparse_interface import FlumePyOptSparseInterface
from icecream import ic


class TestCantileverBeamComplianceMinimization(unittest.TestCase):
    """
    Tests the implementation of the parallel execution for the forward/adjoint passes for a System. Compares the results from the serial execution and the parallel execution for the example problem of the Cantilever beam compliance minimization subject to a volume constraint.
    """

    def setUp(self):

        # Set the parameter values
        L = 1.0
        b = 0.1
        E = 1.0

        volume_constraint = 0.01
        nelems = 50

        # Compute the number of nodes
        nnodes = nelems + 1

        # Construct the force vector (point load in the vertical direction at the final node)
        force_vector = np.zeros(2 * nnodes)
        force_vector[-2] = -1.0

        # Construct the Independents object
        indeps = Independents(obj_name="indeps", sub_analyses=[], nelems=nelems)

        # Construct the MomentofInertia object
        inertia = MomentofInertia(
            obj_name="inertia", sub_analyses=[indeps], b=b, nelems=nelems
        )

        # Construct the LinearStaticSolve object
        linear_solve = LinearStaticSolve(
            obj_name="linear_solve",
            sub_analyses=[inertia],
            nelems=nelems,
            E=E,
            L=L,
            f=force_vector,
        )

        # Construct the Compliance object
        compliance = Compliance(
            obj_name="compliance",
            sub_analyses=[linear_solve],
            nelems=nelems,
            f=force_vector,
        )

        # Construct the BeamVolume object
        volume = BeamVolume(
            obj_name="volume", sub_analyses=[indeps], nelems=nelems, b=b, L=L
        )

        # Construct the System for the optimization problem
        self.serial_sys = System(
            sys_name="BeamThicknessOptimization",
            top_level_analysis_list=[compliance, volume],
            log_name="flume.log",
            log_prefix="tests/test_logs/cantilever_beam",
            parallel_execution=False,
        )

        # Declare the objective function for the beam
        obj_scale = 1e-5
        self.obj_scale = obj_scale
        self.serial_sys.declare_objective(
            global_obj_name="compliance.c", obj_scale=obj_scale
        )

        # Declare the constraint for the beam
        self.serial_sys.declare_constraints(
            global_con_name={
                "volume.V": {"direction": "both", "rhs": volume_constraint}
            }
        )

        # Declare the design variables
        self.serial_sys.declare_design_vars(
            global_var_name={"indeps.h_dv": {"lb": 1e-2, "ub": 10.0}}
        )

        # Construct the FlumeScipyInterface
        self.serial_interface = FlumePyOptSparseInterface(
            flume_sys=self.serial_sys, callback=None
        )

        # Setupt the System for the parallel execution
        self.parallel_sys = System(
            sys_name="BeamThicknessOptimization",
            top_level_analysis_list=[compliance, volume],
            log_name="flume.log",
            log_prefix="tests/test_logs/cantilever_beam_parallel",
            parallel_execution=True,
        )

        # Declare the objective function for the beam
        self.parallel_sys.declare_objective(
            global_obj_name="compliance.c", obj_scale=obj_scale
        )

        # Declare the constraint for the beam
        self.parallel_sys.declare_constraints(
            global_con_name={
                "volume.V": {"direction": "both", "rhs": volume_constraint}
            }
        )

        # Declare the design variables
        self.parallel_sys.declare_design_vars(
            global_var_name={"indeps.h_dv": {"lb": 1e-2, "ub": 10.0}}
        )

        # Construct the FlumeScipyInterface
        self.parallel_interface = FlumePyOptSparseInterface(
            flume_sys=self.parallel_sys, callback=None
        )

        # Set the initial point for the optimization
        self.h0 = np.random.uniform(low=0.05, high=0.15, size=nelems)

        return

    def test_optimized_results(self):
        """
        Perform the optimization for both serial and parallel systems, and compare the expected result to the true solution.
        """

        # Perform the optimization
        x0 = {"indeps.h_dv": self.h0}

        serial_sol = self.serial_interface.optimize_system(
            x0dict=x0,
            opt_prob_name="Serial_Beam",
            optimizer="SNOPT",
            options=None,
            history_filename="history.hst",
        )

        parallel_sol = self.parallel_interface.optimize_system(
            x0dict=x0,
            opt_prob_name="Parallel_Beam",
            optimizer="SNOPT",
            options=None,
            history_filename="history.hst",
        )

        # Compare the compliance values to the expected compliance value
        cstar = 23762.153677294387
        serial_c = serial_sol.fStar / self.obj_scale
        parallel_c = parallel_sol.fStar / self.obj_scale
        err_tol = 1e-6

        with self.subTest("Serial answer:"):
            # Compute the relative error and check
            serial_rel_err = abs(cstar - serial_c) / cstar

            self.assertLessEqual(
                serial_rel_err,
                err_tol,
                "The relative error for the compliance value (SERIAL execution) is not below the tolerance.",
            )

        with self.subTest("Parallel answer:"):
            # Compute the relative error and check
            parallel_rel_err = abs(cstar - parallel_c) / cstar

            self.assertLessEqual(
                parallel_rel_err,
                err_tol,
                "The relative error for the compliance value (PARALLEL execution) is not below the tolerance.",
            )

        return
