from flume.base_classes.system import System
from flume.interfaces.scipy_interface import FlumeScipyInterface
from icecream import ic
import numpy as np
import unittest
from examples.rosenbrock.rosenbrock_problem_classes import (
    Rosenbrock,
    RosenbrockConstraint,
    RosenbrockDVs,
)
from examples.thomson_problem.thomson_problem_classes import (
    ParticlePositions,
    ParticleConstraints,
    PotentialEnergy,
)
from math import pi


class TestUnconstrainedRosenbrock(unittest.TestCase):
    """
    Tests the implementation of the optimization of the unconstrained Rosenbrock function using the FlumeScipyInterface.
    """

    # Execution mode for the System (overridden by the parallel subclass below)
    parallel_execution = False
    dir_suffix = ""

    def setUp(self):
        # Construct the analysis object for the Rosenbrock function
        a = 1.0
        b = 100.0

        rosenbrock = Rosenbrock(obj_name="rosenbrock", sub_analyses=[], a=a, b=b)

        rosenbrock.set_var_values(variables={"x": 0.0, "y": 0.0})

        # Construct the system
        sys = System(
            sys_name="rosen_sys",
            top_level_analysis_list=[rosenbrock],
            log_name="flume.log",
            log_prefix="tests/test_logs/rosenbrock_unconstrained" + self.dir_suffix,
            parallel_execution=self.parallel_execution,
        )

        # Declare the design variables for the system
        sys.declare_design_vars(
            global_var_name={
                "rosenbrock.x": {"lb": -2.0, "ub": 2.0},
                "rosenbrock.y": {"lb": -2.0, "ub": 2.0},
            }
        )

        sys.declare_objective(global_obj_name="rosenbrock.f")

        # Save the system as an attribute
        self.flume_sys = sys

        return

    def test_optimization_scipy(self):
        """
        Tests the implementation of the SciPy interface for the unconstrained Rosenbrock problem.
        """

        # Construct the interface between Flume and SciPy
        interface = FlumeScipyInterface(flume_sys=self.flume_sys)

        # Set a random starting point
        x0 = np.random.uniform(low=-5.0, high=5.0, size=2)

        # Optimize the problem with SciPy minimize
        x, res = interface.optimize_system(x0=x0)

        # Set the expected optimal values
        xstar = np.array([1.0, 1.0])

        # Check that the values match
        np.testing.assert_allclose(
            actual=x,
            desired=xstar,
            rtol=1e-3,
            err_msg="The computed optimal values do not match the expected solution for the constrained Rosenbrock function.",
            verbose=True,
        )

        return


class TestConstrainedRosenbrock(unittest.TestCase):
    """
    Tests the implementation of the optimization of the constrained Rosenbrock function using the FlumeScipyInterface.
    """

    # Execution mode for the System (overridden by the parallel subclass below)
    parallel_execution = False
    dir_suffix = ""

    def setUp(self):

        # Construct the design variables object
        rosenbrock_dvs = RosenbrockDVs(obj_name="dvs", sub_analyses=[])

        # Construct the analysis object for the Rosenbrock function
        a = 1.0
        b = 100.0

        rosenbrock = Rosenbrock(
            obj_name="rosenbrock", sub_analyses=[rosenbrock_dvs], a=a, b=b
        )

        # Construct the analysis object for the constraint on the design variables
        rosenbrock_con = RosenbrockConstraint(
            obj_name="con", sub_analyses=[rosenbrock_dvs]
        )

        # Construct the system
        sys = System(
            sys_name="rosen_sys_con",
            top_level_analysis_list=[rosenbrock, rosenbrock_con],
            log_name="flume.log",
            log_prefix="tests/test_logs/rosenbrock_constrained" + self.dir_suffix,
            parallel_execution=self.parallel_execution,
        )

        # Declare the design variables for the system
        sys.declare_design_vars(
            global_var_name={
                "dvs.x_dv": {"lb": -1.5, "ub": 1.5},
                "dvs.y_dv": {"lb": -1.5, "ub": 1.5},
            }
        )

        # Declare the objective
        sys.declare_objective(global_obj_name="rosenbrock.f")

        # Declare the constraint
        sys.declare_constraints(
            global_con_name={"con.g": {"direction": "leq", "rhs": 1.0}}
        )

        # Store the system as an attribute
        self.flume_sys = sys

        return

    def test_optimization_SLSQP(self):
        """
        Tests that the solution for the constrained Rosenbrock optimization matches the expected solution.
        """

        # Construct the Scipy interface
        interface = FlumeScipyInterface(flume_sys=self.flume_sys)

        # Set a random starting point
        x0 = np.random.uniform(low=-5.0, high=5.0, size=2)

        # Optimize the problem with SciPy minimize
        x, res = interface.optimize_system(x0=x0, method="SLSQP")

        # Set the expected optimal values
        xstar = np.array([0.786, 0.618])

        # Check that the values match
        np.testing.assert_allclose(
            actual=x,
            desired=xstar,
            rtol=1e-3,
            err_msg="The computed optimal values do not match the expected solution for the constrained Rosenbrock function.",
            verbose=True,
        )

        return


class TestThomsonProblem(unittest.TestCase):
    """
    Tests the optimization of the Thomson problem using the FlumeSciPy interface. Here, the check is that the objective function value at the optimized point matches the value for the known, exact solutions within a relative error tolerance of 1e-3.
    """

    # Execution mode for the System (overridden by the parallel subclass below)
    parallel_execution = False
    dir_suffix = ""

    def construct_system(self, n_p):

        # Construct the analysis objects for the system
        positions = ParticlePositions(obj_name="positions", sub_analyses=[], n_p=n_p)

        energy = PotentialEnergy(obj_name="energy", sub_analyses=[positions], n_p=n_p)

        cons = ParticleConstraints(obj_name="cons", sub_analyses=[positions], n_p=n_p)

        # Construct the system
        sys = System(
            sys_name="thomson_problem",
            top_level_analysis_list=[energy, cons],
            log_name=f"flume_{n_p}.log",
            log_prefix="tests/test_logs/thomson_problem" + self.dir_suffix,
            parallel_execution=self.parallel_execution,
        )

        # Declare the design variables for the system
        sys.declare_design_vars(
            global_var_name={
                "positions.theta": {"lb": -pi, "ub": pi},
                "positions.phi": {"lb": 0.0, "ub": 2 * pi},
            }
        )

        # Declare the objective
        sys.declare_objective(global_obj_name="energy.f")

        # Declare the constraints
        sys.declare_constraints(
            global_con_name={"cons.c": {"direction": "both", "rhs": 0.0}}
        )

        return sys

    def optimize_system(self, n_p: int, maxit: int = 100):
        """
        Using the number of particles provided with n_p, optimizes the system using the FlumeScipyInterface and returns the objective function value.
        """

        flume_sys = self.construct_system(n_p=n_p)

        # Construct the Scipy interface
        interface = FlumeScipyInterface(flume_sys=flume_sys)

        # Set random positions for x, y, z to start
        theta0 = np.random.uniform(size=n_p)
        phi0 = np.random.uniform(size=n_p)

        # Set the initial guess using theta0 and phi0
        init_global_vars = {"positions.theta": theta0, "positions.phi": phi0}
        initial_point = interface.set_initial_point(
            initial_global_vars=init_global_vars
        )

        # Optimize the problem with SciPy minimize (SLSQP does not work with this implementation because the subproblem is singular)
        x, res = interface.optimize_system(
            x0=initial_point, method="trust-constr", maxit=maxit
        )

        # Check that the potential energy at the final point matches the expected value
        obj_val = res.fun

        return obj_val

    def test_optimization_2np(self):
        """
        Tests that the solution for the thomson problem with N particles matches the expected solution.
        """

        # Construct the system
        n_p = 2

        # Optimize the system and get the objective value
        obj_val = self.optimize_system(n_p=n_p)
        obj_star = 0.5

        # Compute the relative error
        rel_error = abs(obj_val - obj_star) / obj_star

        # Perform the check
        self.assertLessEqual(
            a=rel_error,
            b=1e-3,
            msg=f"The optimal value of the objective function does not match the expected value for {n_p} particles within the relative error tolerance 1e-3.",
        )

        return

    def test_optimization_3np(self):
        """
        Tests that the solution for the thomson problem with N particles matches the expected solution.
        """

        # Construct the system
        n_p = 3
        maxit = 150

        # Optimize the system and get the objective value
        obj_val = self.optimize_system(n_p=n_p, maxit=maxit)
        obj_star = 1.732050808

        # Compute the relative error
        rel_error = abs(obj_val - obj_star) / obj_star

        # Perform the check
        self.assertLessEqual(
            a=rel_error,
            b=1e-3,
            msg=f"The optimal value of the objective function does not match the expected value for {n_p} particles within the relative error tolerance 1e-3.",
        )

        return

    def test_optimization_12np(self):
        """
        Tests that the solution for the thomson problem with N particles matches the expected solution.
        """

        # Construct the system
        n_p = 12
        maxit = 400

        # Optimize the system and get the objective value
        obj_val = self.optimize_system(n_p=n_p, maxit=maxit)
        obj_star = 49.165253058

        # Compute the relative error
        rel_error = abs(obj_val - obj_star) / obj_star

        # Perform the check
        self.assertLessEqual(
            a=rel_error,
            b=5e-3,
            msg=f"The optimal value of the objective function does not match the expected value for {n_p} particles within the relative error tolerance 5e-3.",
        )

        return


# ---------------------------------------------------------------------------
# Parallel-execution variants. These reuse the exact same problems and test
# methods as the serial classes above, but run the System with
# parallel_execution=True and write their output files to '<dir>_parallel'.
# ---------------------------------------------------------------------------


class TestUnconstrainedRosenbrockParallel(TestUnconstrainedRosenbrock):
    """Parallel-execution variant of TestUnconstrainedRosenbrock."""

    parallel_execution = True
    dir_suffix = "_parallel"


class TestConstrainedRosenbrockParallel(TestConstrainedRosenbrock):
    """Parallel-execution variant of TestConstrainedRosenbrock."""

    parallel_execution = True
    dir_suffix = "_parallel"


class TestThomsonProblemParallel(TestThomsonProblem):
    """Parallel-execution variant of TestThomsonProblem."""

    parallel_execution = True
    dir_suffix = "_parallel"


if __name__ == "__main__":
    unittest.main()
