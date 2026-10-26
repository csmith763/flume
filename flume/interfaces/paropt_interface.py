import os
from flume.base_classes.system import System
import numpy as np
from icecream import ic

try:
    from paropt import ParOpt
    from mpi4py import MPI

    class ParOptProb(ParOpt.Problem):
        def __init__(self, comm, prob) -> None:
            self.prob = prob
            super(ParOptProb, self).__init__(comm, nvars=prob.ndvs, ncon=prob.ncon)
            return

        def getVarsAndBounds(self, x, lb, ub):
            return self.prob.getVarsAndBounds(x, lb, ub)

        def evalObjCon(self, x):
            return self.prob.evalObjCon(x)

        def evalObjConGradient(self, x, g, A):
            return self.prob.evalObjConGradient(x, g, A)

except ImportError:
    pass


class FlumeParOptInterface:

    def __init__(self, flume_sys: System, callback=None, update=None):
        """
        Creates an interface that is used to link an instance of a Flume System to a ParOpt optimization problem. Note that at construction of the interface, each Analysis for the constraints is performed to determine the size for each constraint in the System.

        Parameters
        ----------
        flume_sys : System
            Instance of a Flume System that represents the problem to be solved with ParOpt
        callback : callable function, default None
            This is a callable function that gets executed at the end of every iteration of the evalObjConGradient method during optimization. See below for the structure of the function
        update : callable function, default None
            This is a callable function that gets executed at the start of every iteration of the evalObjCon method during optimization. Nominally, this is used to do parameter updates, such as for a continuation strategy

        Note
        ----
        The callback function is an arbitrary function that the user writes, and it will be called at the end of every evalObjConGradient execution. It is required that the function is setup such that it takes in two arguments:

        def user_callback(x, it_number):
            ...

        Here, the parameters are:
            x : current design variable info for the problem
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

        # Store the string that declares the optimizer
        self.optimizer = "paropt"

        # Initialize the iteration counter
        self.it_counter = 0

        # Set the number of constraints. With the multi-rank MPI backend the constraint analyses can only be distributed once the worker ranks are serving, so the sizing is deferred to optimize_system on rank zero (avoiding a full local analysis on every rank here)
        self.ncon = None
        if not self._is_mpi_multi_rank():
            self._set_constraint_sizes()

        # Set the number of design variables and the indices for the ParOpt PVec x
        self._set_ndvs()

        return

    def _is_mpi_multi_rank(self) -> bool:
        """
        Returns True when the System uses the MPI backend with more than one rank, in which case only rank zero runs ParOpt and the other ranks serve Flume DAG work.
        """
        return (
            self.flume_sys.execution_backend == "mpi" and self.flume_sys._mpi_size > 1
        )

    def _set_constraint_sizes(self):
        """
        Determines the size of each constraint in the System, storing it in the constraint info dictionary, and sets the total number of ParOpt constraints. For the multi-rank MPI backend this must be called on rank zero while the workers are serving, as it executes the System to obtain the constraint values.
        """

        if not hasattr(self.flume_sys, "con_info"):
            self.ncon = 0
            return

        # With the multi-rank MPI backend, a single distributed forward pass provides every constraint value on rank zero
        if self._is_mpi_multi_rank():
            self.flume_sys.execute(debug_print=False)

        ncon = 0
        for con in self.flume_sys.con_info:

            # Perform the analysis for the current constraint locally (serial and thread backends; the MPI pass above already populated the outputs)
            if not self._is_mpi_multi_rank():
                self.flume_sys.con_info[con]["instance"].analyze(debug_print=False)

            # Get the size of the constraint
            con_local_name = self.flume_sys.con_info[con]["local_name"]

            con_val = (
                self.flume_sys.con_info[con]["instance"].outputs[con_local_name].value
            )

            if isinstance(con_val, np.ndarray):
                con_size = con_val.size
            else:
                con_size = 1

            # Store the constraint size
            self.flume_sys.con_info[con]["size"] = con_size

            if self.flume_sys.con_info[con]["direction"] == "both":
                ncon += 2 * con_size
            else:
                ncon += con_size

        self.ncon = ncon

        return

    def _set_ndvs(self):
        """
        Gets the number of design variables for the system of interest.
        """

        ndvs = 0
        self.indices = {}
        # Loop through each of the keys in the design_vars_info dictionary
        for var in self.flume_sys.design_vars_info:

            # Extract the local name for the current variable
            local_name = self.flume_sys.design_vars_info[var]["local_name"]

            # Extract variable value for the current variable
            var_i = self.flume_sys.design_vars_info[var]["instance"].get_var_values(
                variables=[local_name]
            )[local_name]

            # Extract the number of design variables associated with the current variable
            if isinstance(var_i, float):
                ndvs_i = 1
            elif isinstance(var_i, np.ndarray):
                ndvs_i = np.size(var_i)

            # Store the start and end indices for the current design variable, which is used to set/extract the variables from the x PVec
            self.indices[var] = {"start": ndvs, "end": ndvs + ndvs_i}

            # Add the number of dvs to the total number of design variables
            ndvs += ndvs_i

        # Set the attribute for the number of dvs
        self.ndvs = ndvs

        return

    def getVarsAndBounds(self, x, lb, ub):
        """
        Get the variable values and set the bounds for the optimization problem.

        Parameters
        ----------
        x : ParOptVec
            Design variable vector at the current iteration
        lb : ParOptVec
            Vector containing the lower bounds to apply to the design variables in x
        ub : ParOptVec
            Vector containing the ubber bounds to apply to the design variables in x
        """

        # Check to make sure that the design variable info has been set, otherwise raise an error
        if not hasattr(self.flume_sys, "design_vars_info"):
            raise RuntimeError(
                f"The design variables have not yet been declared for the system named '{self.flume_sys.sys_name},' so the bounds cannot be set. Ensure that the function 'declare_design_vars' has been called."
            )

        # Extract the design variable and bound values for each variable in the system
        for var in self.flume_sys.design_vars_info:
            # Extract the local name for the current variable
            local_name = self.flume_sys.design_vars_info[var]["local_name"]

            # Extract variable value for the current variable
            var_i = self.flume_sys.design_vars_info[var]["instance"].get_var_values(
                variables=[local_name]
            )[local_name]

            # Set the value in the x PVec
            start = self.indices[var]["start"]
            end = self.indices[var]["end"]
            x[start:end] = var_i

            # Set the lower bound value for the current variable, if it exists
            if "lb" in self.flume_sys.design_vars_info[var]:
                lb[start:end] = self.flume_sys.design_vars_info[var]["lb"]
            else:
                lb[start:end] = -1e30

            # Set the upper bound value for the current variable
            if "ub" in self.flume_sys.design_vars_info[var]:
                ub[start:end] = self.flume_sys.design_vars_info[var]["ub"]
            else:
                ub[start:end] = 1e30

        return

    def set_system_variables(self, x):
        """
        Sets the variable values for the analysis objects contained within the System.

        Parameters
        ----------
        x : ParOptVec
            Design variable vector at the current iteration
        """

        # Since system variables are being set, all analysis objects must be recomputed
        self.flume_sys.reset_analysis_flags()

        # Loop through the design variables for the system and set for their components
        for var in self.flume_sys.design_vars_info:
            # Get the indices for the current variable
            start = self.indices[var]["start"]
            end = self.indices[var]["end"]

            # Extract the variable value
            if end - start == 1:
                x_i = x[start:end].item()
            else:
                x_i = x[start:end]

            # Extract the local name of the variable
            local_name = self.flume_sys.design_vars_info[var]["local_name"]

            # Declare the variable as a design variable for Flume
            self.flume_sys.design_vars_info[var]["instance"].declare_design_vars(
                variables=[local_name]
            )

            # Set the variable value for the Flume analysis object
            self.flume_sys.design_vars_info[var]["instance"].set_var_values(
                variables={local_name: x_i}
            )

        return

    def evalObjCon(self, x):
        """
        Evaluates the objective and constraints for the problem using the current design variables.

        Parameters
        ----------
        x : ParOptVec
            Design variable vector at the current iteration

        Returns
        -------
        fail : int
            Returns 0 if the method evaluates successfully
        obj : float
            Value of the objective function
        con_list : list
            List of the constraint values at the current design point
        """

        # print("\nCALLING EVALOBJCON")

        # Call the update function, if it was provided
        if self.update is not None:
            self.update(it_num=self.it_counter)

        # If the Flume system does not have an FOI attribute, set it
        if not hasattr(self.flume_sys, "foi"):
            self.flume_sys.declare_foi(global_foi_name=[])

        # Check to make sure that the objective analysis info has been set, otherwise raise an error
        if not hasattr(self.flume_sys, "obj_analysis"):
            raise RuntimeError(
                f"The objective information for the system named '{self.flume_sys.sys_name}' has not yet been declared, so evalObjCon can not be executed. Ensure that the function 'declare_objective' has been called."
            )

        # Set the variable values for the various analyses
        self.set_system_variables(x)

        # Execute the required analyses for the System (performs analyses for objective and constraint Analysis objects, and then scaling/assignment happens below)
        self.flume_sys.execute()

        # Extract the objective function output
        self.obj_name = self.flume_sys.obj_local_name
        obj = (
            self.flume_sys.obj_analysis.outputs[self.obj_name].value
            * self.flume_sys.obj_scale
        )

        # Loop through each of the constraints and perform their respective analyses
        con_list = []
        for con in self.flume_sys.con_info:
            # Extract the output for the constraint
            con_name = self.flume_sys.con_info[con]["local_name"]

            con_val = self.flume_sys.con_info[con]["instance"].outputs[con_name].value

            # Get the constraint size
            con_size = self.flume_sys.con_info[con]["size"]

            # Extract the direction and rhs values
            direction = self.flume_sys.con_info[con]["direction"]
            rhs = self.flume_sys.con_info[con]["rhs"]

            # Using the direction information about the constraint, set the normalized constraint value
            if direction == "geq":
                # If rhs is not 0.0, scale the constraint
                if rhs != 0.0:
                    con_val = con_val / rhs - 1.0

                # If the rhs is < 0, flip the sign for the constraint
                if rhs < 0.0:
                    con_val *= -1.0

                # Append the constraint value to the constraints list
                if con_size > 1:
                    con_list.extend(con_val.tolist())
                else:
                    con_list.append(con_val)

            elif direction == "leq":
                # If rhs is not 0.0, scale the constraint
                if rhs != 0.0:
                    con_val = 1.0 - con_val / rhs
                else:
                    # This step is necessary only for 'leq' to convert constraint to proper form, c(x) >= 0.0
                    con_val *= -1.0

                # If the rhs is < 0, flip the sign for the constraint
                if rhs < 0.0:
                    con_val *= -1.0

                # Append the constraint value to the constraints list
                if con_size > 1:
                    con_list.extend(con_val.tolist())
                else:
                    con_list.append(con_val)

            elif direction == "both":
                # If rhs is not 0.0, scale the constraints
                if rhs != 0.0:
                    # Greater than inequality part
                    con_val_geq = con_val / rhs - 1.0

                    # If the rhs is < 0, flip the sign for the constraint
                    if rhs < 0.0:
                        con_val_geq *= -1.0

                    if con_size > 1:
                        con_list.extend(con_val_geq.tolist())
                    else:
                        con_list.append(con_val_geq)

                    # Less than inequality part
                    con_val_leq = 1.0 - con_val / rhs

                    # If the rhs is < 0, flip the sign for the constraint
                    if rhs < 0.0:
                        con_val_leq *= -1.0

                    if con_size > 1:
                        con_list.extend(con_val_leq.tolist())
                    else:
                        con_list.append(con_val_leq)
                else:
                    # Greater than inequality part
                    con_val_geq = con_val

                    if con_size > 1:
                        con_list.extend(con_val_geq.tolist())
                    else:
                        con_list.append(con_val_geq)

                    # Less than inequality part
                    con_val_leq = -con_val

                    if con_size > 1:
                        con_list.extend(con_val_leq.tolist())
                    else:
                        con_list.append(con_val_leq)

            else:
                raise RuntimeError(
                    "Constraint direction must be 'geq', 'leq', or 'both'."
                )

        # Call the logger function
        self.flume_sys.log_information(iter_number=self.it_counter)

        # Set the failure flag
        fail = 0

        # Update the iteration counter
        self.it_counter += 1

        # Store the objective and constraint info
        self.obj_val = obj
        self.con_vals = con_list

        return fail, obj, con_list

    def evalObjConGradient(self, x, g, A):
        """
        Evaluates the objective and constraint gradients for the system.

        Parameters
        ----------
        x : ParOptVec
            Design variable vector at the current iteration
        g : ParOptVec
            Gradient of the objective function at x
        A : ParOptVec
            Gradients of the constraints evaluated at x. For vector-valued constraints, each element of the constraint array occupies its own row in the constraint Jacobian

        Returns
        -------
        fail : int
            Returns 0 if the method evaluates successfully
        """

        # Check to make sure that the objective analysis info has been set, otherwise raise an error
        if not hasattr(self.flume_sys, "obj_analysis"):
            raise RuntimeError(
                f"The objective information for the system named '{self.flume_sys.sys_name}' has not yet been declared, so evalObjCon can not be executed. Ensure that the function 'declare_objective' has been called."
            )

        # Execute the adjoint procedure for the entire System. This computes the
        # design sensitivities for the objective and every constraint sweep at once
        # and stores them in self.flume_sys.design_derivs, keyed by sweep ID.
        self.flume_sys.execute_adjoint(debug_print=False)
        design_derivs = self.flume_sys.design_derivs

        # --- Objective gradient ------------------------------------------------
        # The objective sweep ID is (global_obj_name, 0)
        obj_sweep_id = (self.flume_sys.global_obj_name, 0)
        for var in self.flume_sys.design_vars_info:
            # Get the indices for the current variable
            start = self.indices[var]["start"]
            end = self.indices[var]["end"]

            # Extract the derivative for the current design variable and apply the objective scale
            gradx_i = design_derivs[obj_sweep_id][var]
            g[start:end] = gradx_i * self.flume_sys.obj_scale

        # --- Constraint gradients ----------------------------------------------
        con_index = 0
        # Loop through the constraints in the system
        for con in self.flume_sys.con_info:
            # Get the size of the constraint
            con_size = self.flume_sys.con_info[con]["size"]

            # Get the constraint direction and rhs value
            direction = self.flume_sys.con_info[con]["direction"]
            rhs = self.flume_sys.con_info[con]["rhs"]

            # Loop through by the size of the constraint (each entry in the array is effectively a separate constraint) and evaluate the contributions
            for i in range(con_size):
                # The sweep ID for the ith component of the current constraint
                con_sweep_id = (con, i)

                # Loop through the variables in the system
                for var in self.flume_sys.design_vars_info:
                    # Get the indices for the current variable
                    start = self.indices[var]["start"]
                    end = self.indices[var]["end"]

                    # Extract the derivative value for the current constraint component and variable combination
                    gradc_i = design_derivs[con_sweep_id][var]

                    # Copy the gradient value if it is an array (to ensure the stored derivatives are not overwritten by the in-place scaling below)
                    if isinstance(gradc_i, np.ndarray):
                        gradc_i = gradc_i.copy()

                    # Assign the constraint gradient, accounting for the scaling, when necessary
                    if direction == "geq":
                        # If rhs is not 0.0, apply the scale to the constraint
                        if rhs != 0.0:
                            gradc_i /= rhs

                        if rhs < 0.0:
                            gradc_i *= -1.0

                        A[con_index + i][start:end] = gradc_i

                    elif direction == "leq":
                        # If rhs is not 0.0, scale the constraint
                        if rhs != 0.0:
                            gradc_i /= -rhs
                        else:
                            gradc_i *= -1.0

                        if rhs < 0.0:
                            gradc_i *= -1.0

                        A[con_index + i][start:end] = gradc_i

                    elif direction == "both":
                        # If rhs is not 0.0, scale the constraints
                        if rhs != 0.0:
                            # Greater than inequality part
                            gradc_i_geq = gradc_i.copy() / rhs

                            if rhs < 0.0:
                                gradc_i_geq *= -1.0

                            # Less than inequality part
                            gradc_i_leq = -gradc_i.copy() / rhs

                            if rhs < 0.0:
                                gradc_i_leq *= -1.0

                        else:
                            gradc_i_geq = gradc_i.copy()
                            gradc_i_leq = -1.0 * gradc_i.copy()

                        # Assign the constraints
                        # Greater than inequality part
                        A[con_index + i][start:end] = gradc_i_geq

                        # Less than inequality part
                        A[con_index + con_size + i][start:end] = gradc_i_leq

            # Update the constraint index
            if direction == "both":
                con_index += 2 * con_size
            else:
                con_index += con_size

        # Add the profiling information for the current iteration
        self.flume_sys.profile_iteration(self.it_counter - 1)

        # Call the callback function, if it was provided
        if self.callback is not None:
            self.callback(x, self.it_counter - 1)

        return 0

    def optimize_system(
        self,
        algorithm: str = "tr",
        options: dict = None,
        check_gradients: bool = False,
    ) -> tuple[np.ndarray, float, list, ParOpt.Optimizer]:
        """
        Performs optimization on the Flume System using ParOpt. Assumes that the user has set the expected initial values for the Flume Analysis instances prior to calling this method (otherwise the defaults for the Flume Analyses will be used). With the MPI backend, call this on every rank: rank zero runs ParOpt (on MPI.COMM_SELF) while the other ranks serve the Flume DAG work it schedules.

        Parameters
        ----------
        algorithm : str
            String that specifies the algorithm to use for the optimization. Should be 'tr' (trust-region) or 'mma' (method of moving asymptotes)
        options : dict
            Dictionary of options to use for ParOpt. If not provided, the default ParOpt options will be used for the specified algorithm (see ParOpt documentation for details)
        check_gradients: bool
            When provided, executes the ParOpt finite difference gradient check for 3 iterations. Returns nothing when this is True. Defaults to False

        Returns
        -------
        Assuming check_gradients is False, the following are returned:

        x_opt : np.ndarray
            Array of design variables at the final point
        fstar : float
            Objective function value at the final point
        con_star : list
            List of constraint function values at the final point
        opt : ParOpt.Optimizer
            Instance of ParOpt's Optimizer class (in case the user wants to extract any additional information)

        If check_gradients is True, nothing is returned

        In MPI mode, non-root ranks serve Flume DAG work and return (None, None, None, None) after rank zero finishes (or nothing if check_gradients is True).
        """

        # Check that the algorithm is acceptable
        if algorithm not in ["tr", "mma"]:
            raise ValueError(
                f"The input for algorithm of '{algorithm} ' is not acceptable. Must be 'tr' or 'mma'."
            )

        # With the MPI backend, ParOpt runs on rank zero only; every other rank serves the DAG work that rank zero schedules until the optimization completes
        is_mpi_root = self._is_mpi_multi_rank() and (
            self.flume_sys._mpi_rank == self.flume_sys.mpi_root
        )
        if self._is_mpi_multi_rank() and not is_mpi_root:
            self.flume_sys.serve_mpi_workers()
            if check_gradients:
                return
            return None, None, None, None

        try:
            # Determine the constraint sizes, if they were deferred at construction (multi-rank MPI backend)
            if self.ncon is None:
                self._set_constraint_sizes()

            # Construct the ParOpt problem
            paroptprob = self.construct_paropt_problem()

            # Get the default options if the user did not provide them
            if options is None:
                options = self.get_paropt_default_options(
                    algorithm=algorithm, output_prefix=self.flume_sys.log_prefix
                )

            # Check the gradients for the system, if specified
            if check_gradients:
                print("Performing ParOpt Gradient check:")
                for i in range(3):
                    paroptprob.checkGradients(1e-6)
                return

            # Perform the optimization
            opt = ParOpt.Optimizer(paroptprob, options)

            opt.optimize()

            # Extract the optimized point
            x, z, zw, zl, zu = opt.getOptimizedPoint()

            # Write the optimized point to the json file
            x_opt = np.array(x)

            # Get the final value of the objective function and constraints
            fstar = self.obj_val
            con_star = self.con_vals

            return x_opt, fstar, con_star, opt
        finally:
            # Release the MPI workers from their service loop
            if is_mpi_root:
                self.flume_sys.shutdown_mpi_workers()

    def construct_paropt_problem(self):
        """
        Function that creates the ParOpt problem that will be used for optimization

        Returns
        -------
        paroptprob : ParOptProb
            Returns an instance of the ParOptProb class
        """

        # Construct the ParOptProblem for the Flume system. ParOpt is kept on MPI.COMM_SELF (rank zero only under the MPI backend) so its communication does not conflict with Flume's DAG-worker protocol on the System communicator
        paroptprob = ParOptProb(MPI.COMM_SELF, prob=self)

        return paroptprob

    def get_paropt_default_options(self, output_prefix, algorithm="tr", maxit=1000):
        """
        Get the default options for paropt.

        Parameters
        ----------
        output_prefix : str
            A string that specifies the directory where the output data should be stored
        algorithm : str
            String that specifies the method to use, defaults to 'tr' which is trust-region
        maxit : int
            Maximum number of iterations

        Returns
        -------
        options : dict
            A dictionary containing the options that can be passed to ParOpt
        """

        # Make the output directory if necessary
        if not os.path.isdir(output_prefix):
            os.makedirs(output_prefix, exist_ok=True)

        # Define the dictionary of default options
        options = {
            "algorithm": algorithm,
            "tr_init_size": 0.05,
            "tr_min_size": 1e-6,
            "tr_max_size": 10.0,
            "tr_eta": 0.25,
            "tr_infeas_tol": 1e-6,
            "tr_l1_tol": 1e-5,
            "tr_linfty_tol": 1e-5,
            "tr_adaptive_gamma_update": True,
            "tr_max_iterations": maxit,
            "mma_max_iterations": maxit,
            "mma_init_asymptote_offset": 0.2,
            "max_major_iters": 100,
            "penalty_gamma": 1e3,
            "qn_subspace_size": 10,
            "qn_type": "bfgs",
            "abs_res_tol": 1e-8,
            "starting_point_strategy": "affine_step",
            # "barrier_strategy": "mehrotra_predictor_corrector",
            "barrier_strategy": "mehrotra",
            "use_line_search": False,
            # "mma_constraints_delta": True,
            "mma_move_limit": 0.1,
            "output_file": os.path.join(output_prefix, "paropt.out"),
            "tr_output_file": os.path.join(output_prefix, "paropt.tr"),
            "mma_output_file": os.path.join(output_prefix, "paropt.mma"),
        }

        return options
