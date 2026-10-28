import os
import time
from flume.base_classes.system import System
import numpy as np
from icecream import ic
from pyoptsparse import Optimization, Optimizer, Solution

try:
    import openmdao.api as om

    failures = (
        om.AnalysisError,
        ValueError,
        FloatingPointError,
        np.linalg.LinAlgError,
    )
except:
    failures = (
        ValueError,
        FloatingPointError,
        np.linalg.LinAlgError,
    )


class FlumePyOptSparseInterface:

    def __init__(
        self,
        flume_sys: System,
        callback=None,
        update=None,
        rejection_log_filename: str = None,
    ):
        """
        Creates an interface that is used to link an instance of a Flume System to an optimization problem constructed and driven through pyOptSparse. To perform the numeric optimization, the user should call the `optimize_system` method after constructing this object.

        Parameters
        ----------
        flume_sys : System
            Instance of a Flume System that represents the problem to be solved with the interface
        callback : callable function, default None
            This is a callable function that gets executed during every iteration of the _objConFun method during optimization. See below for the structure of the function
        update : callable function, default None
            This is a callable function that gets executed at the start of every iteration of the evalObjCon method during optimization. Nominally, this is used to do parameter updates, such as for a continuation strategy
        rejection_log_filename : str, default None
            Name of a file that records every rejected trial point (function evaluation number, exception type and message, and the design variables). The full filepath is `os.path.join(self.flume_sys.log_prefix, rejection_log_filename)`. If not provided, rejections are only printed.

        Note
        ----
        The callback function is an arbitrary function that the user writes, and it will be called at the end of every _objconGradFun execution. It is required that the function is setup such that it takes in two arguments:

        def user_callback(xdict, it_number):
            ...

        Here, the parameters are:
            xdict : dictionary that specifies the current design variable values for the optimization problem
            it_number : current iteration number for the system

        The update function is also a function that the user can optionally provide, and it will be called at the start of every evalObjCon execution. This method is primarily used for updating parameters within the Flume System (e.g. for continuation strategies in topology optimization), and it is structured as follows:

        def user_update(it_number):
            ...

        where it_number is the current iteration number for the system
        """

        # Store the flume system as an attribute
        self.flume_sys = flume_sys

        # Store the callback function
        self.callback = callback

        # Store the update function
        self.update = update

        # Initialize the iteration counter
        self.it_counter = 0

        # Initialize the counter for all function evaluations, including rejected ones
        self.eval_counter = 0

        # Store the path of the rejection log, if requested
        self.rejection_log_path = None
        if rejection_log_filename is not None:
            self.rejection_log_path = os.path.join(
                self.flume_sys.log_prefix, rejection_log_filename
            )

        # Initialize the most recent successful function values, which are returned (with fail=True) when a trial point is rejected; None until the first successful evaluation
        self._last_good_funcs = None

    def _set_system_variables(self, xdict: dict):
        """
        Private method that is used to set the design variables values provided in the input dictionary into the various Flume Analysis objects.

        Parameters
        ----------
        xdict : dict
            Dictionary that specifies the current design point for the optimization problem. The keys are the global variable names, and the values are the numeric values for the design variables in the System
        """

        # Since system variables are being set, all analysis objects must be recomputed
        self.flume_sys.reset_analysis_flags()

        # Loop through the design variables for the system and set the values for their components
        for var in self.flume_sys.design_vars_info:
            # Get the numeric values from the design variables dictionary
            x_val = xdict[var]

            # Extract the local name of the variable
            local_name = self.flume_sys.design_vars_info[var]["local_name"]

            # Declare the variable as a design variable for Flume
            self.flume_sys.design_vars_info[var]["instance"].declare_design_vars(
                variables=[local_name]
            )

            # Map any arays of size 1 to floats for Flume compatibility
            if isinstance(x_val, np.ndarray) and x_val.size == 1:
                x_val = float(x_val.item())

            # Set the variable value for the Flume analysis object
            self.flume_sys.design_vars_info[var]["instance"].set_var_values(
                variables={local_name: x_val}
            )

        return

    def _objconfun(self, xdict: dict):
        """
        Method which is responsible for computing the objective and constraint function values at the current design point. Uses Flume internally to compute the quantities of interest. Internally, this method also first triggers the execution of the user's provided `update` function (if any) and Flume's logging procedure after the objective and constraints have been evaluated.

        Parameters
        ----------
        xdict : dict
            Dictionary that specifies the current design point for the optimization problem

        Returns
        -------
        funcs : dict
            Dictionary whose keys correspond to the local names for the objective and constraint functions in the System and whose values are the numeric values for those quantities
        fail : bool
            Boolean flag that indicates whether the analysis failed (returns False if everything was evaluated correctly)
        """

        # Call the update function, if it was provided
        if self.update is not None:
            self.update(it_num=self.it_counter)

        # If the Flume system does not have an FOI attribute, set it
        if not hasattr(self.flume_sys, "foi"):
            self.flume_sys.declare_foi(global_foi_name=[])

        # Check to make sure that the objective analysis info has been set, otherwise raise an error
        if not hasattr(self.flume_sys, "obj_analysis"):
            raise RuntimeError(
                f"The objective information for the system named '{self.flume_sys.sys_name}' has not yet been declared, so _objconfun can not be executed. Ensure that the function 'declare_objective' has been called."
            )

        # Update the counter for all function evaluations
        self.eval_counter += 1

        # Set the variable values for the various analyses
        self._set_system_variables(xdict)

        # Execute the required analyses for the System (performs analyses for objective and constraint Analysis objects, and then scaling/assignment happens below)
        try:
            self.flume_sys.execute()
        except failures as e:
            print(f"Rejected trial point: {e}")
            ic(xdict)
            self._log_rejection(xdict, e)
            if self._last_good_funcs is None:
                raise  # failure at the initial point: no fallback values exist
            return {k: np.copy(v) for k, v in self._last_good_funcs.items()}, True

        # Extract the objective function output
        self.obj_name = self.flume_sys.obj_local_name
        obj = self.flume_sys.obj_analysis.outputs[self.obj_name].value

        # Store the objective function value in the funcs dictionary
        funcs = {}
        funcs[self.flume_sys.obj_local_name] = obj * self.flume_sys.obj_scale

        # Evaluate the constraint functions
        for con in self.flume_sys.con_info:

            # Extract the output for the constraint
            con_name = self.flume_sys.con_info[con]["local_name"]

            con_val = self.flume_sys.con_info[con]["instance"].outputs[con_name].value

            # Store the constraint value in the funcs dictionary
            funcs[con_name] = con_val

        # Call the logger function
        self.flume_sys.log_information(iter_number=self.it_counter)

        # Set the failure flag
        fail = False

        # Update the iteration counter
        self.it_counter += 1

        self._last_good_funcs = {k: np.copy(v) for k, v in funcs.items()}
        return funcs, fail

    def _log_rejection(self, xdict: dict, exc: Exception):
        """
        Append a rejected trial point to the rejection log (no-op if no log was requested). The entry records the function evaluation number, the exception type and message (which names the failed node under the MPI backend), and the design variables.

        Parameters
        ----------
        xdict : dict
            Dictionary that specifies the rejected design point
        exc : Exception
            The exception that caused the rejection
        """
        if self.rejection_log_path is None:
            return

        lines = [
            f"=== Rejected evaluation {self.eval_counter} "
            f"(after {self.it_counter} successful) at "
            f"{time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"{type(exc).__name__}: {exc}",
            "Design variables:",
        ]
        for var, val in xdict.items():
            val_str = np.array2string(
                np.atleast_1d(val), precision=6, max_line_width=200
            )
            lines.append(f"  {var} = {val_str}")

        # Logging must never mask the rejection itself
        try:
            with open(self.rejection_log_path, "a") as f:
                f.write("\n".join(lines) + "\n\n")
        except OSError as log_err:
            print(f"Could not write the rejection log: {log_err}")

        return

    def _objconGradFun(self, xdict: dict, funcs: dict):
        """
        Method which is responsible for computing the design sensitivities for the optimization problem. Uses Flume to compute the derivatives using the adjoint method. Internally, this also executes the user's provided `callback` function (if any) and Flume's profiling procedure after the derivatives are evaluated.

        Parameters
        ----------
        xdict : dict
            Dictionary that specifies the current design point for the optimization problem
        funcs : dict
            Dictionary that specifies the objective and constraint values at the current design point.

        Returns
        -------
        grad_vals : dict
            A dictionary that specifies the analytic derivatives, computed using the adjoint method. Structured as a dictionary with format as grad_vals = {output_name: {var_name: ndarray}}. For vector-valued constraints, the innery array has shape (con_size, n_vars).
        """

        # Check to make sure that the objective analysis info has been set, otherwise raise an error
        if not hasattr(self.flume_sys, "obj_analysis"):
            raise RuntimeError(
                f"The objective information for the system named '{self.flume_sys.sys_name}' has not yet been declared, so evalObjCon can not be executed. Ensure that the function 'declare_objective' has been called."
            )

        # Initialize the dictionary that will store the gradient information
        grad_vals = {}

        # Execute the adjoint procedure for the System
        self.flume_sys.execute_adjoint(debug_print=False)
        design_derivs = self.flume_sys.design_derivs

        # --- Objective: a single scalar sweep (global_obj_name, 0) ---
        # Extract the name and sweep ID of the objective
        obj_local = self.flume_sys.obj_local_name
        obj_sweep_id = (self.flume_sys.global_obj_name, 0)

        grad_vals[obj_local] = {}
        # Loop through the design variables, extract, and assign the derivative information
        for var in design_derivs[obj_sweep_id]:
            grad_vals[obj_local][var] = (
                design_derivs[obj_sweep_id][var] * self.flume_sys.obj_scale
            )

        # --- Constraints: assemble each Jacobian by its size, indexing components ---
        # Loop through each constraint in the system
        for con in self.flume_sys.con_info:
            # Get the local constraint name and the size of the constraint vector
            local_name = self.flume_sys.con_info[con]["local_name"]
            con_size = self.flume_sys.con_info[con]["size"]

            grad_vals[local_name] = {}
            # Loop through the design variables
            for var in self.flume_sys.design_vars_info:
                # Get the portion of the constraint Jacobian for the current constraint
                rows = [
                    np.atleast_1d(np.asarray(design_derivs[(con, i)][var], dtype=float))
                    for i in range(con_size)
                ]
                grad_vals[local_name][var] = np.vstack(rows)  # shape (con_size, n_vars)

        # Add the profiling information for the current iteration
        self.flume_sys.profile_iteration(self.it_counter - 1)

        # Call the callback function, if it was provided
        if self.callback is not None:
            self.callback(xdict, self.it_counter - 1)

        return grad_vals

    def _addVarGroups(self, x0dict: dict, optProb: Optimization):
        """
        Private method used to add the variable groups to the Optimization object. Returns nothing, but internally adds the variables specified in the Flume System to the pyOptSparse optimization problem.

        Parameters
        ----------
        x0dict : dict
            Dictionary that specifies the initial point for the optimization
        optProb : Optimization
            Instance of the Optimization class, which specifies the structure of the optimization problem of interest
        """

        # Add the design variables to the problem
        for var in self.flume_sys.design_vars_info:

            # Extract the lower bound value, if it exists
            if "lb" in self.flume_sys.design_vars_info[var]:
                lb = self.flume_sys.design_vars_info[var]["lb"]
            else:
                lb = -1e30

            # Extract the upper bound value, if it exists
            if "ub" in self.flume_sys.design_vars_info[var]:
                ub = self.flume_sys.design_vars_info[var]["ub"]
            else:
                ub = 1e30

            # Extract the scale, if it exists (the optimizer sees x_opt = x * scale)
            if "scale" in self.flume_sys.design_vars_info[var]:
                scale = self.flume_sys.design_vars_info[var]["scale"]
            else:
                scale = 1.0

            # Extract the starting value
            x0 = x0dict[var]

            # Get the number of design variables in the group
            if isinstance(x0, np.ndarray):
                var_size = x0.size
            else:
                var_size = 1

            # Add the variable group to the pyOptSparse problem
            optProb.addVarGroup(
                name=var,
                nVars=var_size,
                varType="c",
                value=x0,
                lower=np.ones_like(x0) * lb,
                upper=np.ones_like(x0) * ub,
                scale=scale,
            )

        return

    def _addConGroups(self, optProb: Optimization):
        """
        Private method used to add the constraint groups to the Optimization object. Returns nothing, but internally adds all of the constraints specified in the Flume System to the pyOptSparse optimization problem.

        Parameters
        ----------
        optProb : Optimization
            Instance of the Optimization class, which specifies the structure of the optimization problem of interest

        Note
        ----
        If a constraint specified in the Flume System has `rhs != 0.0`, then that value for the right-hand side of the constraint is used as the scale for the pyOptSparse constraint groups. This means the optimizer will see constraints as `con(x) / abs(rhs_val)`. A 'scale' given when declaring the constraint overrides this, and the optimizer then sees `con(x) * scale`.
        """

        # With the multi-rank MPI backend the constraint analyses run on their owner ranks, so a single distributed forward pass provides every constraint value on rank zero (rank zero never analyzes nodes itself)
        mpi_multi_rank = (
            self.flume_sys.execution_backend == "mpi" and self.flume_sys._mpi_size > 1
        )
        if mpi_multi_rank and self.flume_sys.con_info:
            self.flume_sys.execute(debug_print=False)

        # Loop through each constraint in the Flume System instance
        for con in self.flume_sys.con_info:
            # Extract the local name of the constraint
            con_name = self.flume_sys.con_info[con]["local_name"]

            # Perform the analysis to get the size for the constraint (serial and thread backends; the MPI pass above already populated the outputs)
            if not mpi_multi_rank:
                self.flume_sys.con_info[con]["instance"].analyze(debug_print=False)

            con_val = self.flume_sys.con_info[con]["instance"].outputs[con_name].value
            if isinstance(con_val, np.ndarray):
                con_size = con_val.size
            else:
                con_size = 1

            # Store the size in the constraints info dictionary
            self.flume_sys.con_info[con]["size"] = con_size

            # Extract the direction of the constraint
            direction = self.flume_sys.con_info[con]["direction"]
            rhs_val = self.flume_sys.con_info[con]["rhs"]

            # Get the values to scale the constraint and the value for the lower bound (note that the optimizer sees constraints as con_opt = con * scale)
            bound_val = rhs_val
            if rhs_val == 0.0:
                scale = 1.0
            else:
                scale = 1 / abs(rhs_val)

            # A user-specified scale overrides the one derived from the rhs
            if "scale" in self.flume_sys.con_info[con]:
                scale = self.flume_sys.con_info[con]["scale"]

            # Greater than inequality constraints
            if direction == "geq":
                optProb.addConGroup(
                    name=con_name,
                    nCon=con_size,
                    lower=bound_val,
                    upper=None,
                    scale=scale,
                )
            # Less than inequality constraints
            elif direction == "leq":
                optProb.addConGroup(
                    name=con_name,
                    nCon=con_size,
                    lower=None,
                    upper=bound_val,
                    scale=scale,
                )
            # Equality constraints
            elif direction == "both":
                optProb.addConGroup(
                    name=con_name,
                    nCon=con_size,
                    lower=bound_val,
                    upper=bound_val,
                    scale=scale,
                )

            else:
                raise RuntimeError(
                    "Constraint direction must be 'geq', 'leq', or 'both'."
                )

        return

    def _construct_optimization_problem(
        self, opt_prob_name: str, x0dict: dict, optimizer: str, options: dict = None
    ) -> tuple[Optimizer, Optimization]:
        """
        Private method used to construct the pyOptSparse Optimization and Optimizer objects for the numerical design optimization.

        Parameters
        ----------
        opt_prob_name : str
            Name to give to the optimization problem
        x0dict : dict
            Dictionary that specifies the initial point for the optimization
        optimizer : str
            The name of the optimizer to use for optimization, which defaults to SLSQP. Must be one of 'SLSQP', 'IPOPT', 'SNOPT', 'NLPQLP', 'NSGA2', 'PSQP', or 'CONMIN'.
        options : dict
            Dictionary of options that is provided to the optimizer. These were either specified by the user or are the default options for a given optimizer

        Returns
        -------
        opt : Optimizer
            Instance of a class that inherits from the Optimizer abstract class. Specifically, this will be an instance of the SLSQP, IPOPT, SNOPT, NLPQLP, NSSGA2, PSQP, or CONMIN class, depending on the input for the optimizer to use
        optProb : Optimization
            Instance of the Optimization class, which specifies the structure of the optimization problem of interest
        """

        # Keep pyOptSparse's optimizer protocol local to rank zero. Its MPI
        # optimizers use optProb.comm for their own worker loops, which would
        # conflict with Flume's independent DAG-worker protocol on COMM_WORLD.
        optimizer_comm = None
        if self.flume_sys.execution_backend == "mpi" and self.flume_sys._mpi_size > 1:
            from mpi4py import MPI

            optimizer_comm = MPI.COMM_SELF

        # Construct the Optimization problem
        optProb = Optimization(
            name=opt_prob_name,
            objFun=self._objconfun,
            comm=optimizer_comm,
        )

        # Add the design variable groups to the problem
        self._addVarGroups(x0dict=x0dict, optProb=optProb)

        # Add the constraints to the problem
        self._addConGroups(optProb=optProb)

        # Add the objective to the problem
        obj_name = self.flume_sys.obj_local_name
        optProb.addObj(name=obj_name)

        # Set the optimizer and optimizer options
        if optimizer.lower() == "SLSQP".lower():
            from pyoptsparse import SLSQP

            opt = SLSQP(options=options)
        elif optimizer.lower() == "PSQP".lower():
            from pyoptsparse import PSQP

            opt = PSQP(options=options)
        elif optimizer.lower() == "NSGA2".lower():
            from pyoptsparse import NSGA2

            opt = NSGA2(options=options)
        elif optimizer.lower() == "SNOPT".lower():
            from pyoptsparse import SNOPT

            opt = SNOPT(options=options)
        elif optimizer.lower() == "IPOPT".lower():
            from pyoptsparse import IPOPT

            opt = IPOPT(options=options)
        elif optimizer.lower() == "NLPQLP".lower():
            from pyoptsparse import NLPQLP

            opt = NLPQLP(options=options)
        elif optimizer.lower() == "CONMIN".lower():
            from pyoptsparse import CONMIN

            opt = CONMIN(options=options)
        else:
            raise RuntimeError(
                f"Optimizer by the name of '{optimizer} ' is not supported."
            )

        return opt, optProb

    def optimize_system(
        self,
        x0dict: dict,
        opt_prob_name: str,
        optimizer: str = "SLSQP",
        options: dict = None,
        history_filename: str = None,
        maxit: int = None,
    ) -> Solution | None:
        """
        Performs optimization on the Flume System, interfaced with pyOptSparse, using the optimizer specified by the user. If using a proprietary optimizer (i.e. SNOPT or NLPQLP), it is assumed that the user already has access to the optimizer. Additionally, it is assumed IPOPT is already installed if the user wants to use this optimizer.

        Parameters
        ----------
        x0dict : dict
            Dictionary that provides the initial point from which the optimization will progress. Structured such that the keys are the global variable names, and the values are the numeric values to use
        opt_prob_name : str
            Name that is provided to the optimization problem constructed with pyOptSparse
        optimizer : str
            The name of the optimizer to use for optimization, which defaults to SLSQP. Must be one of 'SLSQP', 'IPOPT', 'SNOPT', 'NLPQLP', 'NSGA2', 'PSQP', or 'CONMIN'.
        options : dict
            Dictionary of options that is provided to the optimizer. If not provided, defaults to None, and the default options for the specified optimizer are used (see pyOptSparse documentation for details)
        history_filename : str
            Name to use for the pyOptSparse History file. The full filepath is set as `os.path.join(self.flume_sys.log_prefix, history_filename)`. If not provided, no History file is written.

        Returns
        -------
        sol : Solution or None
            Rank zero returns the pyOptSparse Solution. In MPI mode, non-root
            ranks serve Flume DAG work and return None after rank zero finishes.
        """

        mpi_worker = (
            self.flume_sys.execution_backend == "mpi"
            and self.flume_sys._mpi_size > 1
            and self.flume_sys._mpi_rank != self.flume_sys.mpi_root
        )
        if mpi_worker:
            self.flume_sys.serve_mpi_workers()
            return None

        is_mpi_root = (
            self.flume_sys.execution_backend == "mpi"
            and self.flume_sys._mpi_size > 1
            and self.flume_sys._mpi_rank == self.flume_sys.mpi_root
        )

        try:
            # Construct the optimization problem
            opt, optProb = self._construct_optimization_problem(
                opt_prob_name=opt_prob_name,
                x0dict=x0dict,
                optimizer=optimizer,
                options=options,
            )

            # Update the options with the output directory for the Flume system
            if options is None:
                self._update_options_output_directory(optimizer, opt, maxit)

            # Set the filepath for the History file, if provided by the user
            if history_filename is not None:
                storeHistory = os.path.join(self.flume_sys.log_prefix, history_filename)
            else:
                storeHistory = None

            # Perform the optimization
            sol = opt(
                optProb,
                sens=self._objconGradFun,
                storeHistory=storeHistory,
                sensMode="pgc",
            )

            return sol
        finally:
            if is_mpi_root:
                self.flume_sys.shutdown_mpi_workers()

    def _update_options_output_directory(
        self, optimizer: str, opt: Optimizer, maxit: int = None
    ):
        """
        Private method used to get the default options for a given optimizer name. Uses the default options specified in the pyOptSparse documentation, but overwrites the output file locations so that they are stored in the directory associated with the Flume System.

        Parameters
        ----------
        optimizer : str
            Optimizer name used for the optimization
        opt : Optimizer
            Instance of the Optimizer class, which is used to drive the optimization
        """

        # Update the output file names with the Flume output directory for each optimizer
        if optimizer.lower() == "slsqp":
            options = opt.getOptions()
            options["IFILE"] = os.path.join(self.flume_sys.log_prefix, "SLSQP.out")
        elif optimizer.lower() == "psqp":
            options = opt.getOptions()
            options["IFILE"] = os.path.join(self.flume_sys.log_prefix, "PSQP.out")
        elif optimizer.lower() == "nsga2":
            options = opt.getOptions()
            options["PrintOut"] = (
                0  # note that there is no way to specify where the output files go besides the current directory, so turning them off entirely by default
            )
        elif optimizer.lower() == "snopt":
            options = opt.getOptions()
            options["Print file"] = os.path.join(
                self.flume_sys.log_prefix, "SNOPT_print.out"
            )
            options["Summary file"] = os.path.join(
                self.flume_sys.log_prefix, "SNOPT_summary.out"
            )
            options["Verify level"] = 0
            options["Difference interval"] = 1e-8

            if maxit is not None:
                options["Major iterations limit"] = maxit

        elif optimizer.lower() == "ipopt":
            options = opt.getOptions()
            options["output_file"] = os.path.join(
                self.flume_sys.log_prefix, "IPOPT.out"
            )
        elif optimizer.lower() == "nlpqlp":
            options = opt.getOptions()
            options["iFile"] = os.path.join(self.flume_sys.log_prefix, "NLPQLP.out")
        elif optimizer.lower() == "conmin":
            options = opt.getOptions()
            options["IFILE"] = os.path.join(self.flume_sys.log_prefix, "CONMIN.out")
        else:
            raise NotImplementedError(
                f"Have not implemented default options for optimizer '{optimizer}' yet."
            )

        # Set the updated options
        opt.options = options

        return
