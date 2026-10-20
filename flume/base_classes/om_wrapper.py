import numpy as np
import openmdao.api as om
from flume.base_classes.analysis import Analysis
from flume.base_classes.state import State
from icecream import ic


class OpenMDAOComponentAnalysis(Analysis):

    def __init__(
        self,
        om_component: om.ExplicitComponent,
        obj_name: str,
        sub_analyses: list = [],
        **kwargs,
    ):

        # Default parameters
        self.default_parameters = {}

        # Store the OpenMDAO component
        self.om_component = om_component

        # Construct an OpenMDAO problem that solely contains the input component in the model
        self.prob = om.Problem()
        self.prob.model.add_subsystem(obj_name, subsys=self.om_component)

        # Perform the setup for the problem so that the component's inputs and outputs are accessibl
        self.prob.setup()
        self.prob.final_setup()

        # Perform the base class object initialization
        super().__init__(obj_name=obj_name, sub_analyses=sub_analyses, **kwargs)

        # Get the input and output names
        om_inputs = self.prob.model.list_inputs(out_stream=None, val=True, desc=True)
        om_outputs = self.prob.model.list_outputs(out_stream=None, val=True, desc=True)

        # Initialize the dictionary that maps Flume names to OpenMDAO names
        self._var_map = {}
        self._var_rev_map = {}

        # Initialize the list that stores all OpenMDAO variable names
        self._om_var_names = []

        # Setup the variables dictionary for the Flume object (maps local Flume names to OpenMDAO model name)
        self.variables = {}

        # Maps each local variable name to its native OpenMDAO shape, used to
        # restore the flat Flume representation before setting values in OpenMDAO
        self._var_shape = {}

        for var in om_inputs:
            # Get the variable name and variable info from the OpenMDAO Component
            var_name = var[0]
            var_info = var[1]

            # Append the variable name to the list
            self._om_var_names.append(var_name)

            # Get the local variable name (split from global_name.local_name)
            local_name = var_name.split(".")[-1]

            # Record the native OpenMDAO shape so the flat Flume representation can
            # be restored to what OpenMDAO expects when the value is set in _analyze
            # (e.g. a (1, 5) OpenMDAO variable is stored flat as (5,)).
            self._var_shape[local_name] = np.shape(var_info["val"])

            # Convert to Flume's native representation, flattening arrays to 1-D. The
            # optimizer interfaces flatten every design variable into a 1-D vector, so
            # the State must be scalar or 1-D to round-trip; otherwise a native (1, N)
            # variable would reject the flat (N,) slice in Analysis.set_var_values.
            value = self._to_native(var_info["val"])
            if isinstance(value, np.ndarray):
                value = value.reshape(-1)
            desc = var_info["desc"]

            # Add the entry to the var map
            self._var_map[local_name] = var_name

            # Construct the State object
            var_state = State(value=value, desc=desc, source=self)

            # Add the State object to the variables dictionary
            self.variables[local_name] = var_state

            # Add the entry to the reverse map (OpenMDAO -> Flume)
            self._var_rev_map[var_name] = local_name

        # Setup the dictionary that maps Flume output names to OpenMDAO output names
        self._out_map = {}
        self._out_rev_map = {}

        # Initialize the list that stores all OpenMDAO output names
        self._om_out_names = []

        # Loop through the outputs and populate the output map dictionary and list
        for out in om_outputs:
            # Get the OpenMDAO output name
            out_name = out[0]
            out_info = out[1]

            # Append the output name to the list
            self._om_out_names.append(out_name)

            # Get the local output name
            local_name = out_name.split(".")[-1]

            # Add the entry to the output name map
            self._out_map[local_name] = {}
            self._out_map[local_name]["om_name"] = out_name
            self._out_map[local_name]["desc"] = out_info["desc"]

            # Add the entry to the reverse map (OpenMDAO -> Flume)
            self._out_rev_map[out_name] = local_name

        return

    @staticmethod
    def _to_native(val):
        """
        Convert an OpenMDAO value into Flume's native representation.

        OpenMDAO stores every variable/output as a NumPy array (a scalar is a shape-(1,) array). Flume's convention, however, is that scalar quantities are Python scalars (e.g. States are built with value=0.0), and the optimizer interfaces rely on this..

        This helper returns a Python scalar for size-1 quantities and an independent NumPy array copy otherwise. Returning a copy (or an immutable scalar) is also required because prob.get_val hands back a live reference into OpenMDAO's internal vectors, which are overwritten in place on the next run_model.
        """
        arr = np.asarray(val)
        if arr.size == 1:
            # .item() preserves the scalar's own type (float or complex)
            return arr.item()
        return np.array(arr)

    def _analyze(self):
        """
        Private analyze method, which calls run_model method to compute the Component outputs using the variable (input) values, and then assigns them into the States.
        """

        # Loop through the variables and set the values in the State object in the OpenMDAO model
        for var in self.variables:
            # Get the value of the variable State
            var_val = self.variables[var].value

            # Get the name of the OpenMDAO variable
            om_name = self._var_map[var]

            # Set the variable value in the OpenMDAO model, restoring the native
            # OpenMDAO shape from Flume's flat representation
            self.prob.set_val(
                name=om_name, val=np.reshape(var_val, self._var_shape[var])
            )

        # Execute run_model for the problem
        self.prob.run_model()

        # Extract the outputs and assign in the outputs dictionary
        self.outputs = {}

        for out in self._out_map:
            # Get the OpenMDAO output name
            om_out_name = self._out_map[out]["om_name"]
            desc = self._out_map[out]["desc"]

            # Get the value for the OpenMDAO output, converting to Flume's native
            # representation
            out_val = self._to_native(self.prob.get_val(name=om_out_name))

            # Construct the State object and assign it in the dictionary
            self.outputs[out] = State(value=out_val, desc=desc, source=self)

        return

    def _analyze_adjoint(self):
        """
        Private adjoint Analysis, which calls the compute_totals method to compute the total derivatives of each output wrt each input, and then accumulates across outputs for updating the variables' derivatives.
        """

        # Execute the compute_totals method for the OpenMDAO model to compute the total derivatives through the Component
        derivs = self.prob.compute_totals(of=self._om_out_names, wrt=self._om_var_names)

        # Loop through all outputs
        for out in self.outputs:
            # Get the adjoint value for the output
            outb = self.outputs[out].deriv

            # Get the OpenMDAO output name
            om_out_name = self._out_map[out]["om_name"]

            # Loop through all variables and accumulate the contributions to each variable's derivative
            for var in self.variables:

                # Get the existing derivative value
                varb = self.variables[var].deriv

                # Get the partial derivative for the current (output, variable) combination
                om_var_name = self._var_map[var]

                dout_dvar = derivs[
                    (om_out_name, om_var_name)
                ]  # Note this is a 2d Jacobian shaped (n_out, n_var)

                # Reverse-mode accumulation x_bar += J^T @ f_bar
                J = np.atleast_2d(dout_dvar)  # (n_out, n_var)
                fbar = np.atleast_1d(outb)  # (n_out,)
                contrib = J.T @ fbar  # (n_var,)

                # Match the variable's native representation (scalar vs array)
                if not isinstance(self.variables[var].value, np.ndarray):
                    contrib = contrib.item()

                varb = varb + contrib

                # Assign into the dictionary
                self.variables[var].set_deriv_value(varb)

        return


class OpenMDAOGroupAnalysis(Analysis):

    def __init__(
        self,
        om_group: om.Group,
        group_variable_names: list,
        group_output_names: list,
        obj_name: str,
        sub_analyses: list = [],
        **kwargs,
    ):

        # Default parameters
        self.default_parameters = {}

        # Store the OpenMDAO group
        self.om_group = om_group

        # Store the group variables and group outputs
        self.group_var_names = group_variable_names
        self.group_out_names = group_output_names

        # Construct an OpenMDAO problem that contains the group in the modle
        self.prob = om.Problem()
        self.prob.model.add_subsystem(name=obj_name, subsys=self.om_group)

        # Perform the setup for the problem so that the component's inputs and outputs are accessibl
        self.prob.setup(force_alloc_complex=True)
        self.prob.final_setup()

        # Perform the base class object initialization
        super().__init__(obj_name=obj_name, sub_analyses=sub_analyses, **kwargs)

        # Get the input and output names
        om_inputs = self.prob.model.list_inputs(out_stream=None, val=True, desc=True)
        om_outputs = self.prob.model.list_outputs(out_stream=None, val=True, desc=True)

        # Set up the variables dictionary for the Flume object, which creates States for each of the OpenMDAO variables defined in group_variables
        self._var_map = {}
        self.variables = {}

        # Maps each local variable name to its native OpenMDAO shape, used to
        # restore the flat Flume representation before setting values in OpenMDAO
        self._var_shape = {}

        # Initialize the list that stores all OpenMDAO variable names
        self._om_var_names = []

        for var in om_inputs:
            # Get the promoted name of the variable
            var_info = var[1]
            prom_var_name = var_info["prom_name"]

            # ic(prom_var_name)

            if prom_var_name in self.group_var_names:
                # Append the variable name to the list
                self._om_var_names.append(prom_var_name)

                # Get the variable name
                local_name = prom_var_name.split(".", maxsplit=1)[-1]

                # Add the entry to the map between the local variable names and the OpenMDAO varaible name
                self._var_map[local_name] = prom_var_name

                # Record the native OpenMDAO shape so the flat Flume representation
                # can be restored to what OpenMDAO expects when the value is set in
                # _analyze (e.g. a (1, 5) variable is stored flat as (5,)).
                self._var_shape[local_name] = np.shape(var_info["val"])

                # Get the value and description, flattening arrays to 1-D so the DV
                # round-trips through the optimizer interfaces' flat vector (a native
                # (1, N) variable would otherwise reject the flat (N,) slice in
                # Analysis.set_var_values).
                value = self._to_native(var_info["val"])
                if isinstance(value, np.ndarray):
                    value = value.reshape(-1)
                desc = var_info["desc"]

                # Construct the State object
                self.variables[local_name] = State(value=value, desc=desc, source=self)

        # Setup the dictionary that maps Flume output names to OpenMDAO output names
        self._out_map = {}
        self._out_rev_map = {}

        # Initialize the list that stores all OpenMDAO output names
        self._om_out_names = []

        # Loop through the outputs and populate the output map dictionary and list
        for out in om_outputs:

            # Get the promoted name of the output
            out_info = out[1]
            prom_out_name = out_info["prom_name"]

            # ic(prom_out_name)

            if prom_out_name in self.group_out_names:

                # Append the output name to the list
                self._om_out_names.append(prom_out_name)

                # Get the local output name
                local_name = prom_out_name.split(".", maxsplit=1)[-1]

                # Add the entry to the output name map
                self._out_map[local_name] = {}
                self._out_map[local_name]["om_name"] = prom_out_name
                self._out_map[local_name]["desc"] = out_info["desc"]

                # Add the entry to the reverse map (OpenMDAO -> Flume)
                self._out_rev_map[prom_out_name] = local_name

        return

    @staticmethod
    def _to_native(val):
        """
        Convert an OpenMDAO value into Flume's native representation.

        OpenMDAO stores every variable/output as a NumPy array (a scalar is a shape-(1,) array). Flume's convention, however, is that scalar quantities are Python scalars (e.g. States are built with value=0.0), and the optimizer interfaces rely on this..

        This helper returns a Python scalar for size-1 quantities and an independent NumPy array copy otherwise. Returning a copy (or an immutable scalar) is also required because prob.get_val hands back a live reference into OpenMDAO's internal vectors, which are overwritten in place on the next run_model.
        """
        arr = np.asarray(val)
        if arr.size == 1:
            # .item() preserves the scalar's own type (float or complex)
            return arr.item()
        return np.array(arr)

    def _analyze(self):
        """
        Private analyze method, which calls the run_model method to compute the outputs from the Group using the current input values for the variables. Then, assigns them into the States.
        """

        # Loop through the variables and set the values in the State object in the OpenMDAO model
        for var in self.variables:
            # Get the value of the variable State
            var_val = self.variables[var].value

            # Get the name of the OpenMDAO variable
            om_name = self._var_map[var]

            # Set the variable value in the OpenMDAO model, restoring the native
            # OpenMDAO shape from Flume's flat representation
            self.prob.set_val(
                name=om_name, val=np.reshape(var_val, self._var_shape[var])
            )

        # Execute run_model for the problem
        self.prob.run_model()

        # Extract the outputs and assign in the outputs dictionary
        self.outputs = {}

        for out in self._out_map:
            # Get the OpenMDAO output name
            om_out_name = self._out_map[out]["om_name"]
            desc = self._out_map[out]["desc"]

            # Get the value for the OpenMDAO output, converting to Flume's native
            # representation
            out_val = self._to_native(self.prob.get_val(name=om_out_name))

            # Construct the State object and assign it in the dictionary
            self.outputs[out] = State(value=out_val, desc=desc, source=self)

        # Update the flag for the total derivative computation to reflect that the design point has changed
        self.om_totals_computed = False

        return

    def _analyze_adjoint(self):
        """
        Private adjoint Analysis, which calls the compute_totals method to compute the total derivatives of each output wrt each input, and then accumulates across outputs for updating the variables' derivatives.
        """

        # Execute the compute_totals method for the OpenMDAO model to compute the total derivatives through the Component (if they have not been already computed for the current design point)
        if not self.om_totals_computed:
            self.derivs = self.prob.compute_totals(
                of=self._om_out_names, wrt=self._om_var_names
            )

            # Update the flag to denote that the totals have been computed for the current design point
            self.om_totals_computed = True

        # Loop through all outputs
        for out in self.outputs:
            # Get the adjoint value for the output
            outb = self.outputs[out].deriv

            # Get the OpenMDAO output name
            om_out_name = self._out_map[out]["om_name"]

            # Loop through all variables and accumulate the contributions to each variable's derivative
            for var in self.variables:

                # Get the existing derivative value
                varb = self.variables[var].deriv

                # Get the partial derivative for the current (output, variable) combination
                om_var_name = self._var_map[var]

                dout_dvar = self.derivs[
                    (om_out_name, om_var_name)
                ]  # Note this is a 2d Jacobian shaped (n_out, n_var)

                # Reverse-mode accumulation x_bar += J^T @ f_bar
                J = np.atleast_2d(dout_dvar)  # (n_out, n_var)
                fbar = np.atleast_1d(outb)  # (n_out,)
                contrib = J.T @ fbar  # (n_var,)

                # Match the variable's native representation (scalar vs array)
                if not isinstance(self.variables[var].value, np.ndarray):
                    contrib = contrib.item()

                varb = varb + contrib

                # Assign into the dictionary
                self.variables[var].set_deriv_value(varb)

        return
