from flume.base_classes.analysis import Analysis
import graphviz as gv
from icecream import ic
from typing import List
from flume.interfaces.utils import Logger
import os
import numpy as np


class System:
    """
    This class is used to wrap multiple Analysis objects into a system, which can then be utilized to perform optimization using one of Flume's optimizer interfaces.
    """

    def __init__(
        self,
        sys_name: str,
        top_level_analysis_list: List[Analysis],
        log_name: str = "flume.log",
        log_prefix: str = ".",
        execution_backend: str = "serial",
        parallel_max_workers: int = 4,
        track_node_timing: bool = False,
        mpi_comm=None,
    ):
        """
        Defines a class that wraps multiple analysis objects into a single system, which can then be utilized to perform optimization with one of Flume's optimizer interfaces after declaring design varaibles, an objective function, and (optionally) constraints.

        Parameters
        ----------
        sys_name : str
            Name that is assigned to the System
        top_level_analysis_list: list
            A list of instances of Analysis objects that will be utilized within the optimization problem. Here, the objects that should be provided are the analyses that define the objective function and any constraint functions so that these can be declared for the optimization formulation. Other Analysis classes, such as those used as sub-analyses, do not need to be provided here, as they should be provided when creating the objects in the top-level analysis list
        log_name : str
            String that defines the name to use for the log file, defaults to 'flume.log'
        log_prefix : str
            String that defines the output directory where the log file and other files are saved, defaults to the current directory '.'
        parallel_execution : bool
            Boolean operator that specifies whether the DAG should be executed in parallel for the forward and adjoint analyses. Defaults to False.
        parallel_max_workers : int
            Only applied when `parallel_execution` is True. Spawns this many threads for performing the forward/adjoint analyses in parallel.
        execution_backend : str, optional
            Execution backend to use: ``"serial"``, ``"thread"``, or ``"mpi"``. If omitted, ``parallel_execution`` selects the serial or thread backend for backwards compatibility. MPI execution is scheduled by rank zero; non-root ranks must call ``serve_mpi_workers()``.
        mpi_comm : MPI communicator, optional
            Communicator to use with the MPI backend. Defaults to ``mpi4py.MPI.COMM_WORLD``.
        track_node_timing : bool
            Boolean value that specifies whether the `profile.log` file should track the execution time for the forward and adjoint analyses for each node in the DAG.
        """

        # Store the name for the system
        self.sys_name = sys_name

        # Store the list of analyses
        self.top_level_analysis_list = top_level_analysis_list

        # Store the information for the logging
        self.log_name = log_name
        self.log_prefix = log_prefix

        os.makedirs(self.log_prefix, exist_ok=True)

        # Checks for the execution_backend input
        if execution_backend not in ("serial", "thread", "mpi"):
            raise ValueError(
                "execution_backend must be one of 'serial', 'thread', or 'mpi'."
            )
        elif execution_backend == "thread" and parallel_max_workers <= 1:
            raise RuntimeError(
                "User specified mode as 'thread', but input for 'parallel_max_workers' <= 1. This should be strictly greater than one to leverage thread-based parallelism."
            )

        # `parallel_execution` remains available as a compatibility attribute; MPI is selected and dispatched independently.
        self.execution_backend = execution_backend
        self.parallel_execution = execution_backend == "thread"
        self.parallel_max_workers = parallel_max_workers

        # Store the MPI information (only used if execution backend is MPI)
        self.mpi_comm = None
        self.mpi_root = 0
        self._mpi_rank = 0
        self._mpi_size = 1

        # Update the comm, rank, and size info if MPI is the backend
        if self.execution_backend == "mpi":
            try:
                from mpi4py import MPI
            except ImportError as exc:
                raise ImportError(
                    "execution_backend='mpi' requires mpi4py and an MPI runtime."
                ) from exc

            self.mpi_comm = MPI.COMM_WORLD if mpi_comm is None else mpi_comm
            self._mpi_rank = self.mpi_comm.Get_rank()
            self._mpi_size = self.mpi_comm.Get_size()

        # Store the attribute for whether per-node timing should be tracked. When True, each node's individual analysis time is measured/stored (enabling the summed-CPU and efficiency profile columns and the optional per-node detail). When False, only the System-level wall-clock time is recorded.
        self.track_node_timing = track_node_timing

        # Configure the file path for the log file
        logger_path = self.log_prefix
        if self.execution_backend == "mpi" and self._mpi_rank != self.mpi_root:
            logger_path = os.path.join(self.log_prefix, f"mpi_rank_{self._mpi_rank}")
            os.makedirs(logger_path, exist_ok=True)
        self.outputs_log = Logger(log_path=logger_path, log_name=self.log_name)
        self.profile_log = Logger(log_path=logger_path, log_name="profile.log")

        # Assemble the list of all analysis objects
        self.full_analysis_list = self.assemble_full_analysis_list()

        # Initialize the constraints dictionary (default assumes no constraints)
        self.con_info = {}

        return

    def reset_analysis_flags(self):
        """
        Resets all of the analysis flags for all analyses within the system to be False. This is done when variable values are updated, as all systems must be analyzed again to propagate changes in design variable values.
        """

        # Loop through each top-level analysis, resetting each analysis in the stack
        for top_level in self.top_level_analysis_list:

            # If the stack does not exist, make it
            if not hasattr(top_level, "stack"):
                # Assemble the stack
                top_level.stack = top_level._make_stack()

            # Loop through each object within the current top-level's analysis stack and set the analyzed attribute to be False
            for analysis in top_level.stack:
                analysis.analyzed = False

        return

    def assemble_full_analysis_list(self):
        """
        Assembles the full list of analyses that comprise the overall system architecture.
        """

        # Initialize the analysis list
        full_analysis_list = []

        # Loop through each top-level analysis, assemble the stack, and append the analyses to the total analysis list
        for top_level in self.top_level_analysis_list:
            # Assemble the stack
            stack = top_level._make_stack()

            # Add the entries in the current stack to the list if they are not there already
            for analysis in stack:
                if analysis not in full_analysis_list:
                    full_analysis_list.append(analysis)
                else:
                    continue

        return full_analysis_list

    def build_dag(self):
        """
        Method to construct the directed acyclic graph for the System. Utilizes the nodes defined as the top-level Analysis objects and their sub-analyses to construct the DAG strucutre. Creates attributes which define the following:

        self.dag_nodes : a set for all unique nodes in the DAG
        self.dag_dependents : a dictionary which specifies the children/dependents for each node in the DAG
        self.dag_remaining : a dictionary that specifies the number of remaining nodes that need to be analzed before a given node can be analyzed (i.e. number of unanalyzed ancestors for each node)
        self.dag_sink_ancestors : a dictionary that defines the reverse mapping, where each node is mapped to its ancestors (opposite of self.dag_dependents)
        """

        from collections import defaultdict

        # Initialize a set to store all of the unique Analysis objects in the DAG
        nodes = set()

        # Initialize a dictionary of lists, which will store all of the dependencies for each Analysis object
        dependents = defaultdict(list)

        # Initialize a dictionary, which will store information about how many dependencies remain for each node in the DAG. Used as a trigger for when an Analysis object can begin its execution (after all dependencies have been analyzed)
        remaining = {}

        # Define the local function to use to trace through the stack and build the DAG
        def visit(n):
            # Skip the node if it is already in the set of nodes
            if n in nodes:
                return

            # Add the node to the set
            nodes.add(n)

            # Set the number of dependencies for the current node in the 'remaining' dictionary
            remaining[n] = len(n.sub_analyses)

            # Loop through the sub-analyses for the current node, add them to the dependents dictionary, and visit each sub-analysis
            for sub in n.sub_analyses:
                dependents[sub].append(n)
                visit(sub)

        # Invoke the 'visit' function for all top-level Analysis objects (nodes) to construct the DAG
        for t in self.top_level_analysis_list:
            visit(t)

        # Store the information as attributes
        self.dag_nodes = nodes
        self.dag_dependents = dependents
        self.dag_remaining = remaining

        # Map each top-level analysis (i.e. a sink with no dependent Analyses) to its ancestor set (i.e. its dependencies )
        self.dag_sink_ancestors = {}
        for sink in self.top_level_analysis_list:
            # Define a set for the Analysis objects seen in the adjoint pass
            seen = set()

            # Append the sink (i.e. the top-level Analysis object) to the stack
            stack = [sink]

            while stack:
                # Pop the node off the stack
                n = stack.pop()

                # If it is already in the set of seen objects, continue
                if n in seen:
                    continue

                # Otherwise, add the node to the set
                seen.add(n)

                # Extend the stack by adding its sub-analyses into the stack
                stack.extend(n.sub_analyses)

            # Assign the ancestors for the current sink as all of the distinct Analysis objects in the "seen" set
            self.dag_sink_ancestors[sink] = seen

        return

    def execute(self, mode: str = "real", debug_print: bool = False):
        """
        Executes all top-level Analysis objects for the System using the
        configured serial, thread, or MPI backend. MPI execution is root-only;
        non-root ranks must be running ``serve_mpi_workers()``.
        """
        import time

        # Start the wall-clock timer for the entire forward pass (always measured)
        wall_start = time.perf_counter()

        # Get the boolean attribute for whether parallel execution should be used
        parallel_execution = self.parallel_execution

        # Whether to track per-node (and summed-CPU) timing
        track = self.track_node_timing

        # Summed per-node CPU time for the forward pass. Only meaningful when per-node timing is tracked; otherwise left as NaN.
        self.dag_total_time = 0.0 if track else float("nan")

        if self.execution_backend == "mpi":
            self._execute_mpi(mode=mode, debug_print=debug_print)
            self.dag_forward_wall = time.perf_counter() - wall_start
            return

        # Serial path for the System, which sequentially executes the objective and all constraint Analysis objects
        if not parallel_execution:
            # Perform the analysis for the objective function
            self.obj_analysis.analyze(mode=mode, debug_print=debug_print)

            if track:
                self.dag_total_time += self.obj_analysis.forward_total

            # Perform the analysis for all constraint functions
            for con in self.con_info:
                con_instance = self.con_info[con]["instance"]
                con_instance.analyze(debug_print=debug_print)

                if track:
                    self.dag_total_time += con_instance.forward_total

        else:
            from concurrent.futures import ThreadPoolExecutor
            import queue

            # Build the information that defines the DAG if it does not already exist
            if not hasattr(self, "dag_nodes"):
                self.build_dag()

            # Get the nodes and dependents attributes (static across calls)
            nodes = self.dag_nodes
            dependents = self.dag_dependents

            # Create a copy of the dag_remaining dictionary for use in the execution. The original is preserved so it can be accessed for subsequent executions (the copy is modified by decrementing the counters)
            remaining = self.dag_remaining.copy()

            # Get the maximum number of workers to use
            max_parallel_workers = self.parallel_max_workers

            # Define the local function that defines the procedures to execute when an Analysis object/node is to be analyzed
            def perform_node_analysis(analysis_obj: Analysis):

                # Construct/get the connections
                analysis_obj._connect()

                # Call the private analyze method, timing it per-node only if requested
                if track:
                    start = time.perf_counter()
                    analysis_obj._analyze()
                    analysis_obj.analyzed = True
                    end = time.perf_counter()

                    # Store the time it took to perform this Analysis
                    analysis_obj.analysis_time = end - start
                else:
                    start = end = None
                    analysis_obj._analyze()
                    analysis_obj.analyzed = True

                if debug_print:
                    import threading

                    tname = threading.current_thread().name
                    tid = threading.get_ident()

                    if track:
                        print(
                            f"[{tname} tid={tid}] {analysis_obj.obj_name}: "
                            f"start={start:.6f} end={end:.6f}"
                        )
                    else:
                        print(f"[{tname} tid={tid}] {analysis_obj.obj_name}")

            # Serially initialize all Analysis objects
            for n in nodes:
                n._initialize_analysis(mode=mode)

            # Construct the initial set of Analysis objects that are ready to be executed (i.e. anything that is a source for the DAG)
            initial_ready = [n for n in remaining if remaining[n] == 0]

            # Construct the ThreadPoolExecutor and execute all of the Analyses in the System
            with ThreadPoolExecutor(max_workers=max_parallel_workers) as pool:
                # Completed nodes are handed back to this scheduler thread via a thread-safe queue This makes the scheduling O(number_of_nodes) instead of O(number_of_nodes**2)
                done_q = queue.Queue()

                def _submit(n):
                    fut = pool.submit(perform_node_analysis, n)
                    fut.add_done_callback(lambda f, node=n: done_q.put((node, f)))

                # Every node in the DAG is executed exactly once
                pending = len(nodes)

                # Kick off the source nodes (those with no unmet dependencies)
                for n in initial_ready:
                    _submit(n)

                # Drain completions one at a time; O(1) bookkeeping per completion
                while pending:
                    node, future = done_q.get()

                    # Extract the data from the future (triggers the end of the perform_node_analysis and propagates any worker exception)
                    future.result()
                    pending -= 1

                    # Loop through the dependents for the current node
                    for dep in dependents[node]:
                        # Decrement the counter for the dependent
                        remaining[dep] -= 1

                        # Trigger the execution of the next Analysis object once all of its dependencies have finished executing
                        if remaining[dep] == 0:
                            _submit(dep)

            # Sum the per-node forward times for the DAG (only if tracking is on)
            if track:
                for n in nodes:
                    self.dag_total_time += n.analysis_time

        # Record the wall-clock time for the entire forward pass
        self.dag_forward_wall = time.perf_counter() - wall_start

        # Display the timing for the forward pass through the DAG, if requested
        if debug_print:
            print(
                f"DAG forward pass: wall={self.dag_forward_wall:.6f} s, "
                f"cpu-sum={self.dag_total_time} s"
            )

        return

    def _execute_mpi(self, mode: str, debug_print: bool = False):
        """
        Execution function when backend is set to MPI. Executes the forward DAG on rank zero with rank-local MPI workers.
        """

        # Ensure that the current rank is the MPI root
        self._mpi_require_root()

        # Get the MPI topology for the DAG
        nodes, node_ids, signature = self._mpi_topology()

        # Initialize the MPI workers to execute the DAG in forward mode
        self._mpi_begin_workers("forward", signature, mode=mode)

        # Loop through nodes and initialize the analyses
        for node in nodes:
            node._initialize_analysis(mode=mode)

        # Copy the dag_remaining dictionary
        remaining = self.dag_remaining.copy()

        # Get the list of nodes that are ready to be executed (i.e. they have no dependents)
        ready = sorted(node_ids[node] for node in nodes if remaining[node] == 0)

        # Set the size of pending nodes and initialize all other information
        pending = len(nodes)
        busy = {}
        failure = None
        track = self.track_node_timing

        # Get the rank owners for the MPI process
        rank_owners = self._mpi_node_owners(len(nodes))
        local_rank_only = self._mpi_size == 1

        # Define a function to run the node
        def run_local(node_id):
            # Get the node for its ID and its connections
            node = nodes[node_id]
            node._connect()

            # Copy the State information
            inputs = {
                name: self._mpi_copy(state.value)
                for name, state in node.variables.items()
            }

            # Perform the forward analysis for the node
            result = self._mpi_run_forward_node(node, inputs, track=track)

            # Apply the outputs and set the time it took to complete the analysis
            self._mpi_apply_outputs(node, result["outputs"])
            if track:
                node.analysis_time = result["elapsed"]
            return

        # If the MPI size is one, perform the forward analysis on the only rank
        if local_rank_only:
            while ready:
                # Get the current node ID
                node_id = ready.pop(0)

                # Run the analysis for the node
                run_local(node_id)

                # Decrement the pending counter
                pending -= 1
                node = nodes[node_id]

                # Loop through the dependents for the current node
                for dependent in self.dag_dependents[node]:
                    # Decrement the remaining counter
                    remaining[dependent] -= 1

                    # If there are no remaining dependents, update the list that dneotes the nodes that are ready to execute
                    if remaining[dependent] == 0:
                        ready.append(node_ids[dependent])

                # Sort the nodes by ID
                ready.sort()
        # MPI-enabled branch
        else:
            while pending:
                # Loop through the worker ranks in MPI size
                for worker_rank in range(1, self._mpi_size):
                    if worker_rank in busy or not ready or failure is not None:
                        continue

                    # Get the next node ID
                    node_id = next(
                        (
                            candidate
                            for candidate in ready
                            if rank_owners[candidate] == worker_rank
                        ),
                        None,
                    )

                    if node_id is None:
                        continue

                    # Remove the node ID from the 'ready' list
                    ready.remove(node_id)

                    # Extract the node with its node ID and connections info
                    node = nodes[node_id]
                    node._connect()

                    # Get the inputs and outputs, using MPI copying
                    inputs = {
                        name: self._mpi_copy(state.value)
                        for name, state in node.variables.items()
                    }
                    dependency_outputs = {
                        node_ids[sub]: self._mpi_export_outputs(sub)
                        for sub in node.sub_analyses
                    }

                    # Trigger the execution for the forward task
                    self.mpi_comm.send(
                        {
                            "kind": "forward_task",
                            "node_id": node_id,
                            "inputs": inputs,
                            "dependency_outputs": dependency_outputs,
                            "mode": mode,
                            "track": track,
                        },
                        dest=worker_rank,
                        tag=self._MPI_COMMAND_TAG,
                    )

                    # Mark that the current node is now busy
                    busy[worker_rank] = node_id

                # Check if the busy dictionary is empty
                if not busy:
                    if failure is not None:
                        break
                    if ready:
                        raise RuntimeError(
                            "MPI DAG scheduler deadlocked: ready nodes have no available owner rank."
                        )
                    break

                # Get the results for the current node and MPI rank
                result, worker_rank = self._mpi_receive_result()

                # Get the expected node ID by popping it out of the busy dictionary
                expected_node_id = busy.pop(worker_rank)

                # Raise a RuntimeError if the expected node ID does not match the results node ID
                if result.get("node_id") != expected_node_id:
                    failure = RuntimeError(
                        f"MPI worker {worker_rank} returned node "
                        f"{result.get('node_id')} while executing "
                        f"{expected_node_id}."
                    )

                # If the result is okay, proceed by updating the information
                elif result.get("ok"):
                    # Get the node for its ID
                    node = nodes[expected_node_id]

                    # Apply the outputs and update the analysis time
                    self._mpi_apply_outputs(node, result["outputs"])
                    if track:
                        node.analysis_time = result["elapsed"]

                    # Decrement the pending counter
                    pending -= 1

                    # Loop through the dependents for the node
                    for dependent in self.dag_dependents[node]:
                        # Decrement the counter for the dependents
                        remaining[dependent] -= 1

                        # If the dependent has no more remaining dependencies, update that it is ready
                        if remaining[dependent] == 0:
                            ready.append(node_ids[dependent])

                    # Sort the ready list by node ID
                    ready.sort()

                # Triger failure information
                elif failure is None:
                    message = (
                        f"MPI forward analysis failed on rank {worker_rank} "
                        f"for node {nodes[expected_node_id].obj_name}: "
                        f"{result.get('exception_message', 'Unknown worker error')}\n"
                        f"{result.get('error', 'No worker traceback available.')}"
                    )
                    failure = self._mpi_rebuild_exception(result, message)

            # Raise the failure if it is not None
            if failure is not None:
                raise failure

        # Update the total DAG time for forward execution if tracking is turned on
        if track:
            self.dag_total_time = sum(
                getattr(node, "analysis_time", 0.0) for node in nodes
            )

        if debug_print:
            print(
                f"MPI DAG forward pass completed on {self._mpi_size} rank(s); "
                f"nodes={len(nodes)}"
            )

    # Define tags for MPI communication information
    _MPI_COMMAND_TAG = 27101  # command tag: rank 0 -> worker process
    _MPI_RESULT_TAG = 27102  # result tag: worker process -> rank 0

    @staticmethod
    def _mpi_rebuild_exception(result, message):
        """
        Rebuild a worker's forward-analysis exception on rank zero with its original type, so callers see the same exception class as with the serial backend (e.g. an optimizer interface that rejects a trial point on ValueError or LinAlgError). Falls back to RuntimeError when the class cannot be imported or constructed from a message.
        """

        # If a recoverable analysis error is detected in the result, return that error to rank0
        if result.get("recoverable_analysis_error"):
            from openmdao.core.analysis_error import AnalysisError

            return AnalysisError(message)

        # Catch the exception class coming from the result
        class_info = result.get("exception_class")
        if class_info is not None:
            try:
                import importlib

                # Extract the module and qualifying name for the class
                module_name, qualname = class_info
                exc_class = importlib.import_module(module_name)

                # Get the exception class
                for attr in qualname.split("."):
                    exc_class = getattr(exc_class, attr)

                # Return the exception class with the message if it is actually an aception class
                if isinstance(exc_class, type) and issubclass(exc_class, Exception):
                    return exc_class(message)

            except Exception:
                pass

        return RuntimeError(message)

    @staticmethod
    def _mpi_copy(value):
        """
        Copy array/scalar State data before handing it to mpi4py.
        """
        import copy

        return copy.deepcopy(value)

    def _mpi_require_root(self):
        """
        Private helper that makes sure the current MPI rank is the root, otherwise raises a RuntimeError.
        """

        if self._mpi_rank != self.mpi_root:
            raise RuntimeError(
                f"MPI System.execute()/execute_adjoint() may only be called on rank {self.mpi_root}; rank {self._mpi_rank} must call serve_mpi_workers()."
            )

        return

    def _mpi_topology(self):
        """
        Return deterministic node ordering, IDs, and a cross-rank signature.
        """

        # Build the DAG for the System
        self.build_dag()

        # Initialize a list for the ordered DAG nodes and a set for the nodes that have been seen already
        ordered = []
        seen = set()

        # Define the visit function, which adds nodes into the 'ordered' list and the 'seen' set
        def visit(node):
            # If node has already been seen, return
            if node in seen:
                return

            # Otherwise, add to the set
            seen.add(node)

            # Trace through the sub-analyses for the current node, executing the 'visit' function on each
            for sub_analysis in node.sub_analyses:
                visit(sub_analysis)

            # Add the node to the 'ordered' list
            ordered.append(node)

        # Visit all top-level analyses in the list
        for top_level in self.top_level_analysis_list:
            visit(top_level)

        # Set the node IDs in a dictionary by enumerating through the ordered list
        node_ids = {node: index for index, node in enumerate(ordered)}

        # Define a signature for each node in the 'ordered' list, which stores a variety of information for each node in the DAG
        signature = tuple(
            (
                index,  # index number
                node.obj_name,  # object name for the node
                node.__class__.__module__,  # module for the node's class
                node.__class__.__qualname__,  # qualiified name for the node's class
                tuple(
                    node_ids[sub] for sub in node.sub_analyses
                ),  # tuple of node IDs for the node's sub-analyses
                tuple(node.variables),  # tuple of node variable names
            )
            for index, node in enumerate(ordered)
        )

        # Store the information in MPI-specific attributes
        self._mpi_nodes = ordered
        self._mpi_node_ids = node_ids
        self._mpi_topology_signature = signature

        return ordered, node_ids, signature

    def _mpi_node_owners(self, n_nodes):
        """
        Private helper to the rank owners for MPI.
        """

        # If the MPI size is one, return the root
        if self._mpi_size == 1:
            return {node_id: self.mpi_root for node_id in range(n_nodes)}
        # Otherwise, increment the node IDs based on the MPI size
        return {
            node_id: 1 + (node_id % (self._mpi_size - 1)) for node_id in range(n_nodes)
        }

    def _mpi_begin_workers(self, phase, signature, **kwargs):
        """
        Private method which starts the MPI workers for forward execution of the DAG.
        """

        # If the MPI size is 1, return
        if self._mpi_size == 1:
            return

        # Sweep through the MPI size and send info to each of the MPI workers
        for worker_rank in range(1, self._mpi_size):
            # Send information from the root to a worker proces
            self.mpi_comm.send(
                {
                    "kind": "begin",
                    "phase": phase,
                    "signature": signature,
                    **kwargs,
                },
                dest=worker_rank,
                tag=self._MPI_COMMAND_TAG,
            )

        # Loop through the workers, receive results, and store errors (if any)
        errors = []
        for _ in range(self._mpi_size - 1):
            # Receive the result for the current worker process
            result, worker_rank = self._mpi_receive_result()

            # If the result did not terminate successfully, append the error information
            if not result.get("ok"):
                errors.append(
                    f"rank {worker_rank}: "
                    f"{result.get('error', 'worker initialization failed')}"
                )

        # If any errors were detected, raise a RuntimeError
        if errors:
            raise RuntimeError(
                f"MPI worker initialization failed for {phase}: " + "; ".join(errors)
            )

        return

    def _mpi_receive_result(self):
        """
        Method which receives the results using MPI.
        """
        from mpi4py import MPI

        # Get the MPI status
        status = MPI.Status()

        # Receive the result using MPI comm info, and return the info
        result = self.mpi_comm.recv(
            source=MPI.ANY_SOURCE,
            tag=self._MPI_RESULT_TAG,
            status=status,
        )
        return result, status.Get_source()

    def _mpi_apply_outputs(self, node, output_values):
        """
        Method which loops thorugh and copies the values from output States
        """

        from flume.base_classes.state import State

        if not hasattr(node, "outputs"):
            node.outputs = {}

        # Loop through outputs in the output_values dictionary
        for name, output in output_values.items():
            # Copy the value using MPI
            value = self._mpi_copy(output["value"])

            # If the name is in the current node's outputs, set the value
            if name in node.outputs:
                node.outputs[name].value = value

            # Otherwise, create a new output for the node
            else:
                node.outputs[name] = State(
                    value=value,
                    desc=output["desc"],
                    source=node,
                )

        return

    def _mpi_export_outputs(self, node):
        """
        Exports output information for a node using MPI copy. Returns a dictionary that shares the same name: value & desc information for the outputs of the input node.
        """

        return {
            name: {
                "value": self._mpi_copy(state.value),
                "desc": state.desc,
            }
            for name, state in node.outputs.items()
        }

    def _mpi_run_forward_node(self, node, inputs, track, dependency_outputs=None):
        """
        Private method that executes the forward analysis for a node in the graph.
        """
        import time

        # Loop through the dependent output values for the current node
        for dependency_id, output_values in (dependency_outputs or {}).items():
            # Apply the output values for the curent node
            self._mpi_apply_outputs(self._mpi_nodes[dependency_id], output_values)

        # Get the connections for the node
        node._connect()

        # Loop through the inputs and assign the values into the State objects for the current node using MPI copy
        for name, value in inputs.items():
            node.variables[name].value = self._mpi_copy(value)

        # Time and perform the analysis
        start = time.perf_counter()
        node._analyze()
        elapsed = time.perf_counter() - start

        # Update the analyzed attribute
        node.analyzed = True

        # Set the dictionary that will be returned
        forward_node_outputs = {
            "outputs": {
                name: {
                    "value": self._mpi_copy(state.value),
                    "desc": state.desc,
                }
                for name, state in node.outputs.items()
            },
            "elapsed": elapsed if track else 0.0,
        }

        return forward_node_outputs

    def serve_mpi_workers(self):
        """
        Serve root-scheduled MPI DAG tasks on a non-root rank.

        Construct the same ``System`` on every rank. Rank zero runs the optimizer; every other rank calls this method and remains in the worker loop until root calls :meth:`shutdown_mpi_workers`.
        """

        # Raise an error if MPI is not the backend for the system
        if self.execution_backend != "mpi":
            raise RuntimeError("serve_mpi_workers() requires execution_backend='mpi'.")

        # Raise an error if the current MPI rank is not the root
        if self._mpi_rank == self.mpi_root:
            raise RuntimeError(
                "Rank zero runs the optimizer and must not call " "serve_mpi_workers()."
            )

        # Initialize the nodes and signature info
        nodes = None
        signature = None
        while True:
            # Get the command and its type
            command = self.mpi_comm.recv(
                source=self.mpi_root, tag=self._MPI_COMMAND_TAG
            )

            kind = command.get("kind")

            # If command kind is stop
            if kind == "stop":
                # Send info
                self.mpi_comm.send(
                    {"ok": True, "kind": "stopped"},
                    dest=self.mpi_root,
                    tag=self._MPI_RESULT_TAG,
                )
                return

            # If kind is begin, start the execution
            if kind == "begin":
                try:
                    # Get the MPI topology and command from the signature
                    nodes, _, signature = self._mpi_topology()
                    if signature != command["signature"]:
                        raise RuntimeError(
                            "DAG topology differs from rank zero; all ranks must construct the same analyses in the same order."
                        )

                    # Initialize forward analyses
                    if command["phase"] == "forward":
                        for node in nodes:
                            node._initialize_analysis(mode=command["mode"])

                    # Pass in adjoint mode
                    elif command["phase"] == "adjoint":
                        pass

                    # Raise an error that an known MPI phase was detected
                    else:
                        raise ValueError(f"Unknown MPI phase {command['phase']!r}.")

                    # Set the response
                    response = {"ok": True, "kind": "ready"}

                # Set the response if there was an exception, reporting the error back to the top rank
                except Exception:
                    import traceback

                    response = {
                        "ok": False,
                        "kind": "ready",
                        "error": traceback.format_exc(),
                    }

                # Send the info from the current result tag to the root
                self.mpi_comm.send(
                    response,
                    dest=self.mpi_root,
                    tag=self._MPI_RESULT_TAG,
                )
                continue

            # If kind is forward_task, execute the forward analysis
            if kind == "forward_task":
                # Get the node ID
                node_id = command["node_id"]

                # Try executing the forward node analysis with MPI
                try:
                    result = self._mpi_run_forward_node(
                        nodes[node_id],
                        command["inputs"],
                        track=command["track"],
                        dependency_outputs=command["dependency_outputs"],
                    )
                    response = {"ok": True, "node_id": node_id, **result}
                # If an exception is raised, record the exception info in the response
                except Exception as exc:
                    import traceback

                    exception_mro = type(exc).__mro__
                    response = {
                        "ok": False,
                        "node_id": node_id,
                        "exception_message": str(exc),
                        "exception_class": (
                            type(exc).__module__,
                            type(exc).__qualname__,
                        ),
                        "recoverable_analysis_error": any(
                            cls.__module__ == "openmdao.core.analysis_error"
                            and cls.__name__ == "AnalysisError"
                            for cls in exception_mro
                        ),
                        "error": traceback.format_exc(),
                    }

                # Send the response from the forward execution to the root
                self.mpi_comm.send(
                    response,
                    dest=self.mpi_root,
                    tag=self._MPI_RESULT_TAG,
                )
                continue

            # If kind is adjoint task, execute the adjoint analysis
            if kind == "adjoint_task":

                # Try executing the adjoint analysis with MPI
                try:
                    response = self._mpi_run_adjoint_node(
                        nodes[command["node_id"]],
                        command["node_id"],
                        command["sweep_id"],
                        command["output_seeds"],
                        command["variable_keys"],
                        command["track"],
                    )
                    # Update the response
                    response.update(
                        {
                            "ok": True,
                            "node_id": command["node_id"],
                            "sweep_id": command["sweep_id"],
                        }
                    )
                # If an exception is raised, record the exception info in the response
                except Exception:
                    import traceback

                    response = {
                        "ok": False,
                        "node_id": command["node_id"],
                        "sweep_id": command["sweep_id"],
                        "error": traceback.format_exc(),
                    }

                # Send the response from the adjoint execution back to the root
                self.mpi_comm.send(
                    response,
                    dest=self.mpi_root,
                    tag=self._MPI_RESULT_TAG,
                )
                continue

            # If kind is call, run a provided function with the current node's live state (for callback functions)
            if kind == "call":
                try:
                    if nodes is None:
                        nodes, _, _ = self._mpi_topology()
                    value = command["func"](
                        nodes[command["node_id"]],
                        *command["args"],
                        **command["kwargs"],
                    )
                    response = {"ok": True, "kind": "call_result", "value": value}
                except Exception:
                    import traceback

                    response = {
                        "ok": False,
                        "kind": "call_result",
                        "error": traceback.format_exc(),
                    }

                # Send the result from the call back to the root
                self.mpi_comm.send(
                    response,
                    dest=self.mpi_root,
                    tag=self._MPI_RESULT_TAG,
                )
                continue

            # Raise runtime error if this is reached, unknown command is trying to be passed
            raise RuntimeError(f"Unknown MPI worker command {kind!r}.")

        return

    def shutdown_mpi_workers(self):
        """
        Stop all non-root MPI workers after the optimizer has completed."""
        # Raise an error if the execution backend is not MPI
        if self.execution_backend != "mpi":
            raise RuntimeError(
                "shutdown_mpi_workers() requires execution_backend='mpi'."
            )

        # Ensure that the current rank is the MPI root
        self._mpi_require_root()

        # Send the information depending on the MPI size
        if self._mpi_size == 1:
            return
        for worker_rank in range(1, self._mpi_size):
            self.mpi_comm.send(
                {"kind": "stop"},
                dest=worker_rank,
                tag=self._MPI_COMMAND_TAG,
            )

        # Detect and store any errors
        errors = []
        for _ in range(self._mpi_size - 1):
            result, worker_rank = self._mpi_receive_result()
            if result.get("kind") != "stopped" or not result.get("ok"):
                errors.append(f"rank {worker_rank}: {result!r}")

        # Raise a RuntimeError if the MPI workers did not shut down correctly
        if errors:
            raise RuntimeError(
                "MPI workers did not shut down cleanly: " + "; ".join(errors)
            )

        return

    def call_on_owner(self, node, func, *args, **kwargs):
        """
        Evaluate ``func(node, *args, **kwargs)`` against the live state of ``node``.

        With the MPI backend only the rank that owns a node executes its ``_analyze``, so any internal state that is not a Flume output (e.g. an OpenMDAO problem's intermediate variables or a cached trajectory) is current only on that rank. This method runs ``func`` on the owner rank and returns its result to rank zero. For the serial and thread backends it simply calls ``func`` locally.

        The result reflects the owner's most recent forward evaluation, so call this after ``execute()`` at the design point of interest.

        Parameters
        ----------
        node : Analysis
            Analysis object in this System's DAG whose state should be read
        func : callable
            Module-level function (it is pickled by reference for MPI) taking the
            node as its first argument and returning picklable data
        *args, **kwargs
            Additional picklable arguments forwarded to ``func``

        Returns
        -------
        The value returned by ``func`` on the owner rank.
        """
        if self.execution_backend != "mpi" or self._mpi_size == 1:
            return func(node, *args, **kwargs)

        self._mpi_require_root()
        if (
            getattr(self, "_mpi_node_ids", None) is None
            or node not in self._mpi_node_ids
        ):
            self._mpi_topology()
        node_id = self._mpi_node_ids[node]
        owner = self._mpi_node_owners(len(self._mpi_nodes))[node_id]

        self.mpi_comm.send(
            {
                "kind": "call",
                "node_id": node_id,
                "func": func,
                "args": args,
                "kwargs": kwargs,
            },
            dest=owner,
            tag=self._MPI_COMMAND_TAG,
        )
        result, worker_rank = self._mpi_receive_result()
        if result.get("kind") != "call_result" or worker_rank != owner:
            raise RuntimeError(
                f"Unexpected MPI reply from rank {worker_rank} to call_on_owner "
                f"for node '{node.obj_name}': {result!r}"
            )
        if not result.get("ok"):
            raise RuntimeError(
                f"call_on_owner failed on rank {owner} for node "
                f"'{node.obj_name}':\n{result.get('error')}"
            )
        return result["value"]

    def _collect_sinks_info(self):
        """
        Helper method which is used to collect the information for all sinks in the DAG (i.e. top-level Analysis objects) and the reverse mapping.

        Returns
        -------
        sinks_info : list
            List of tuples for each sink/top-level Analysis object in the DAG. Each tuple stores the (sweep_id, sink_analysis_object, output_names, seed_value)
        sinks_of_info : list
            Dictionary that is used to reverse map each sweep ID name to its corresponding Analysis object instance
        """

        # Initialize the list that is returned. Each entry in the list is a tuple that contains (sweep_id, sink_analysis_obj, output_names, seed)
        sinks_info = []

        # Define the dictionary, which specifies the reverse mapping between the sweep ID and the sink analysis object
        sinks_of_info = {}

        # Get the sink information for the objective
        obj_sweep_id = (self.global_obj_name, 0)
        obj_sink_tuple = (
            obj_sweep_id,  # sweep ID (global_obj_name, index)
            self.obj_analysis,  # objective Analysis object
            self.obj_local_name,  # objective local name
            1.0,  # seed value
        )
        sinks_info.append(obj_sink_tuple)

        sinks_of_info[obj_sweep_id] = self.obj_analysis

        # Loop through each object in the top-level Analysis list, which corresponds to all sinks in the DAG
        for global_con_key in self.con_info:
            # Extract the information that goes into the tuple for the current sink
            instance = self.con_info[global_con_key]["instance"]
            out_local_name = self.con_info[global_con_key]["local_name"]

            # Get the output value and size, which is used for setting seed information
            con_val = instance.outputs[out_local_name].value

            # Set the seed info and sweep ID name depending on whether the constraint is a scalar or vector
            if isinstance(con_val, np.ndarray):
                # Loop over each entry in the constraint vector and populate info
                for i in range(con_val.size):
                    # Initialize a zero vector that is the shape of the con_val
                    seed = np.zeros(con_val.shape)

                    # Set the ith entry to 1
                    seed.flat[i] = 1.0

                    # Set the name
                    sweep_id_name = (global_con_key, i)

                    # Set the tuple for the current constraint
                    sinks_info.append((sweep_id_name, instance, out_local_name, seed))

                    # Add the entry to the dictionary which maps sweep ID to analysis object
                    sinks_of_info[sweep_id_name] = instance
            else:
                # Constraint is a scalar
                seed = 1.0
                sweep_id_name = (global_con_key, 0)

                # Set the tuple for the current constraint
                sinks_info.append((sweep_id_name, instance, out_local_name, seed))

                # Add the entry to the dictionary which maps sweep ID to analysis object
                sinks_of_info[sweep_id_name] = instance

        return sinks_info, sinks_of_info

    def _extract_dv_derivs(self, sweep_id=None):
        """
        Helper method which is used to extract the design derivatives for the current adjoint execution.
        """

        # Initialize the dictionary that will store the design derivatives for the current sweep ID value
        sweep_id_derivs = {}

        # Loop through the design variables, and extract the derivative value for the input sweep ID
        for var in self.design_vars_info:
            # Get the local name for the current variable
            local_name = self.design_vars_info[var]["local_name"]
            instance = self.design_vars_info[var]["instance"]

            # Extract the derivative value for the current variable
            deriv_val = instance.variables[local_name].get_deriv(sweep_id)

            # Store the derivative value in the dictionary object
            sweep_id_derivs[var] = deriv_val

        return sweep_id_derivs

    def execute_adjoint(self, debug_print: bool = False):
        """
        Execute objective and constraint adjoints using the configured backend.
        MPI execution is root-only; non-root ranks must be running
        ``serve_mpi_workers()``.
        """

        # Get the information for the sweeps that are required (i.e. the total number of quantities of interest, one objective and each constraint)
        sinks_info, sinks_of_info = self._collect_sinks_info()

        import time

        # Start the wall-clock timer for the entire adjoint pass (always measured)
        wall_start = time.perf_counter()

        # Whether to track summed-CPU timing
        track = self.track_node_timing

        # Summed adjoint CPU time (only populated where per-node timing is available)
        self.dag_adjoint_time = 0.0 if track else float("nan")

        # Initialize the dictionary that stores the derivatives of interest
        self.design_derivs = {}

        # Extract the parallel execution attribute
        parallel_execution = self.parallel_execution

        if self.execution_backend == "mpi":
            self._execute_adjoint_mpi(debug_print=debug_print)
            self.dag_adjoint_wall = time.perf_counter() - wall_start
            return

        # Serial path for the adjoint analysis of the System, which sequentially executes the objective and all constraint Analysis objects (and serially traces through the stack)
        if not parallel_execution:
            # Loop through the sweeps, and execute all of the adjoint paths serially
            for sweep_id, sink_object, output_names, seed in sinks_info:
                # Here, _current_sweep is None, which triggers the original, serial execution
                sink_object._add_output_seed(outputs=[output_names], seed=seed)

                # Call the analzye_adjoint method
                sink_object.analyze_adjoint(debug_print=debug_print)

                # Accumulate the summed adjoint CPU time, if tracking is enabled
                if track:
                    self.dag_adjoint_time += sink_object.adjoint_total

                # Extract the derivatives and store them into the dictionary of design derivatives
                self.design_derivs[sweep_id] = self._extract_dv_derivs(sweep_id=None)

            # Record the wall-clock time for the adjoint pass
            self.dag_adjoint_wall = time.perf_counter() - wall_start

            return

        # Parallel path for the adjoint analysis (parallelizes over the top-level Analysis objects and within each sweep)
        else:
            from concurrent.futures import ThreadPoolExecutor
            import queue
            from flume.base_classes.state import (
                _current_sweep,
                _current_writer,
                sweep_context,
            )
            import threading

            # Construct the DAG info if not done already
            if not hasattr(self, "dag_nodes"):
                self.build_dag()

            # Extract the DAG info
            nodes = self.dag_nodes
            dependents = self.dag_dependents
            sink_ancestors = self.dag_sink_ancestors

            # Serial pre-pass: for every (node, sweep) in each sink's subgraph, create the canonical (zeroed) slot and the node's private per-writer contribution buffer for each of its variables and outputs. Parallel phase only mutates existing values
            adjoint_nodes = set()
            for sweep_id, sink_object, output_names, seed in sinks_info:
                # Loop through all nodes in the ancestors list for each sink object
                for node in sink_ancestors[sink_object]:

                    # Add the node to the set
                    adjoint_nodes.add(node)

                    # Loop through all output/variable State objects for the node
                    for state in list(node.outputs.values()) + list(
                        node.variables.values()
                    ):
                        # Canonical slot for this sweep (zeroed)
                        state.ensure_sweep_slot(sweep_id)

                        # This node's private contribution buffer for this sweep
                        state.prepare_writer_slot(sweep_id, id(node))

            # Create one adjoint lock per Analysis object. This guards _analyze_adjoint so the same node object cannot execute its adjoint concurrently from two sweeps (which would race on any instance attributes the method mutates)
            for node in adjoint_nodes:
                # Create the adjoint locks for each node in the adjoint nodes set
                node._adjoint_lock = threading.Lock()

            # Set each sink's seed in its corresonding slot (done serially)
            for sweep_id, sink_object, output_names, seed in sinks_info:
                # Use the context manager with the current sweep ID to set the output seed for each sweep ID
                with sweep_context(sweep_id):
                    sink_object._add_output_seed(outputs=[output_names], seed=seed)

            # Construct the transpose of the "remaining" dictionary created for the forward DAG, which defines the nodes that need to be executed before information can be passed upstream in the adjoint pass
            remaining_adj = {}
            for sweep_id, sink_object, _, _ in sinks_info:
                # Get the ancestors for the current sink object/top-level Analysis object
                anc = sink_ancestors[sink_object]

                # For each sweep ID, constructs the dictionary of dependencies to trace through the DAG in reverse
                remaining_adj[sweep_id] = {
                    n: sum(1 for d in dependents[n] if d in anc) for n in anc
                }

            # Define the function used to perform the adjoint analysis for a given node
            # Per-node adjoint timing (only when tracking is enabled). A single node
            # may run its adjoint once per sweep, so times are accumulated. The lock
            # guards the shared summed-CPU accumulator and each node's adjoint_time
            # against concurrent updates from different sweeps.
            if track:
                self.dag_adjoint_time = 0.0
                for node in adjoint_nodes:
                    node.adjoint_time = 0.0
                _adjoint_time_lock = threading.Lock()

            def perform_node_adjoint(n, sweep_id):
                # Seed reduction (lock-free, sweep-isolated): fill this node's private read-buffer for each of its outputs with the total incoming adjoint (canonical seed + sum of all consumer contribution buffers)
                for out_state in n.outputs.values():
                    out_state.reduce_seed(sweep_id, id(n))

                # Run the user adjoint with the sweep and writer context set, so that all derivative reads/writes route to this node's private (sweep, id(n)) buffers
                token_s = _current_sweep.set(sweep_id)
                token_w = _current_writer.set(id(n))
                try:
                    with n._adjoint_lock:
                        if track:
                            _adj_start = time.perf_counter()
                            n._analyze_adjoint()
                            n.adjoint_analyzed = True
                            _adj_elapsed = time.perf_counter() - _adj_start
                        else:
                            n._analyze_adjoint()
                            n.adjoint_analyzed = True
                finally:
                    _current_writer.reset(token_w)
                    _current_sweep.reset(token_s)

                # Accumulate the per-node and summed adjoint CPU time (thread-safe)
                if track:
                    with _adjoint_time_lock:
                        n.adjoint_time += _adj_elapsed
                        self.dag_adjoint_time += _adj_elapsed

            # Setup the scheduler, which is responsible for scheduling all adjoint analyses (in parallel over each quantity of interest, and parralel within a given sweep)
            with ThreadPoolExecutor(max_workers=self.parallel_max_workers) as pool:
                # Completed (node, sweep) tasks are handed back to this scheduler thread via a thread-safe queue,
                done_q = queue.Queue()

                def _submit_adj(n, sweep_id):
                    fut = pool.submit(perform_node_adjoint, n, sweep_id)
                    fut.add_done_callback(
                        lambda f, node=n, sid=sweep_id: done_q.put((node, sid, f))
                    )

                # Total number of (node, sweep) adjoint tasks; each runs exactly once
                pending = sum(len(remaining_adj[sid]) for sid in remaining_adj)

                # Kick off each sweep's roots (the sinks: objective and constraints)
                for sweep_id, sink_object, _, _ in sinks_info:
                    # Loop through all nodes and values in the remaining adjoint dictionary
                    for n, r in remaining_adj[sweep_id].items():
                        # Submit the perform_node_adjoint task if no dependents remain
                        if r == 0:
                            _submit_adj(n, sweep_id)

                # Execute all adjoint analyses until all derivatives computed across the entire DAG for each quantity of interest
                while pending:
                    # Extract the node and sweep ID of the next completed task
                    n, sweep_id, fut = done_q.get()

                    # Extract the data from the future (triggers the end of perform_node_adjoint and propagates any worker exception)
                    fut.result()
                    pending -= 1

                    # Get the ancestors for the current top-level Analysis object/sink
                    anc = sink_ancestors[sinks_of_info[sweep_id]]

                    # Loop through the sub-analyses for the current node
                    for sub in n.sub_analyses:
                        # Continue if the sub analysis is not in the ancestors list
                        if sub not in anc:
                            continue

                        # Decrement the counter for the adjoint tracker
                        remaining_adj[sweep_id][sub] -= 1

                        # Trigger the adjoint for the sub-analysis once all of its
                        # dependents in this sweep have completed
                        if remaining_adj[sweep_id][sub] == 0:
                            _submit_adj(sub, sweep_id)

            # Extract the design derivatives for each sweep. Before extracting, reduce each design-variable State's per-writer contribution buffers into its canonical slot so that _extract_dv_derivs returns the total accumulated gradient for that sweep
            for sweep_id, sink, _, _ in sinks_info:
                for var in self.design_vars_info:
                    instance = self.design_vars_info[var]["instance"]
                    local_name = self.design_vars_info[var]["local_name"]
                    instance.variables[local_name].reduce_to_canonical(sweep_id)

                self.design_derivs[sweep_id] = self._extract_dv_derivs(
                    sweep_id=sweep_id
                )

            # Per-node adjoint CPU time is accumulated in perform_node_adjoint when
            # track_node_timing is enabled (self.dag_adjoint_time). When tracking is
            # disabled, leave it as the NaN sentinel set at the top of this method.

        # Record the wall-clock time for the adjoint pass
        self.dag_adjoint_wall = time.perf_counter() - wall_start

        return

    def _execute_adjoint_mpi(self, debug_print: bool = False):
        """Run reverse-DAG sweeps and reduce derivative contributions on root."""
        self._mpi_require_root()
        nodes, node_ids, signature = self._mpi_topology()
        self._mpi_begin_workers("adjoint", signature)
        for node in nodes:
            node._connect()

        sinks_info, _ = self._collect_sinks_info()
        dependents = self.dag_dependents
        sink_ancestors = {}
        for _, sink, _, _ in sinks_info:
            ancestors = set()
            stack = [sink]
            while stack:
                node = stack.pop()
                if node in ancestors:
                    continue
                ancestors.add(node)
                stack.extend(node.sub_analyses)
            sink_ancestors[sink] = ancestors
        sink_seeds = {}
        remaining = {}
        ancestors_by_sweep = {}

        for sweep_id, sink, output_name, seed in sinks_info:
            output_state = sink.outputs[output_name]
            state_key = self._mpi_state_key(output_state, node_ids)
            sink_seeds.setdefault((sweep_id, state_key), None)
            sink_seeds[(sweep_id, state_key)] = self._mpi_add(
                sink_seeds[(sweep_id, state_key)], seed
            )
            ancestors = sink_ancestors[sink]
            ancestors_by_sweep[sweep_id] = ancestors
            remaining[sweep_id] = {
                node: sum(1 for dependent in dependents[node] if dependent in ancestors)
                for node in ancestors
            }

        contributions = {}
        rank_owners = self._mpi_node_owners(len(nodes))
        track = self.track_node_timing
        if track:
            for node in nodes:
                node.adjoint_time = 0.0
            self.dag_adjoint_time = 0.0

        ready = []
        for sweep_id, _, _, _ in sinks_info:
            for node, count in remaining[sweep_id].items():
                if count == 0:
                    ready.append((sweep_id, node_ids[node]))
        ready.sort(key=lambda item: (self._mpi_sweep_sort_key(item[0]), item[1]))

        pending = sum(len(sweep_nodes) for sweep_nodes in remaining.values())
        failure = None

        def run_task(sweep_id, node_id):
            node = nodes[node_id]
            output_seeds = {}
            for name, state in node.outputs.items():
                key = self._mpi_state_key(state, node_ids)
                seed = sink_seeds.get((sweep_id, key))
                seed = self._mpi_add(seed, contributions.get((sweep_id, key)))
                output_seeds[name] = (
                    self._mpi_copy(seed)
                    if seed is not None
                    else self._mpi_zero(state.value)
                )
            variable_keys = {
                name: self._mpi_state_key(state, node_ids)
                for name, state in node.variables.items()
            }
            return self._mpi_run_adjoint_node(
                node,
                node_id,
                sweep_id,
                output_seeds,
                variable_keys,
                track,
            )

        def complete_node(sweep_id, node_id, result):
            nonlocal pending, failure
            node = nodes[node_id]
            if not result.get("ok", True):
                if failure is None:
                    failure = RuntimeError(
                        f"MPI adjoint failed on a worker for "
                        f"{node.obj_name}, sweep {sweep_id}:\n"
                        f"{result.get('error', 'Unknown worker error')}"
                    )
                return
            for state_key, value in result["contributions"].items():
                key = (sweep_id, state_key)
                contributions[key] = self._mpi_add(contributions.get(key), value)
            if track:
                elapsed = result["elapsed"]
                node.adjoint_time += elapsed
                self.dag_adjoint_time += elapsed
            pending -= 1
            ancestors = ancestors_by_sweep[sweep_id]
            for sub_analysis in node.sub_analyses:
                if sub_analysis not in ancestors:
                    continue
                remaining[sweep_id][sub_analysis] -= 1
                if remaining[sweep_id][sub_analysis] == 0:
                    ready.append((sweep_id, node_ids[sub_analysis]))
            ready.sort(key=lambda item: (self._mpi_sweep_sort_key(item[0]), item[1]))

        if self._mpi_size == 1:
            while ready:
                sweep_id, node_id = ready.pop(0)
                result = run_task(sweep_id, node_id)
                complete_node(sweep_id, node_id, result)
                if failure is not None:
                    raise failure
        else:
            busy = {}
            while pending:
                for worker_rank in range(1, self._mpi_size):
                    if worker_rank in busy or not ready or failure is not None:
                        continue
                    task_position = next(
                        (
                            position
                            for position, (_, node_id) in enumerate(ready)
                            if rank_owners[node_id] == worker_rank
                        ),
                        None,
                    )
                    if task_position is None:
                        continue
                    sweep_id, node_id = ready.pop(task_position)
                    node = nodes[node_id]
                    output_seeds = {}
                    for name, state in node.outputs.items():
                        state_key = self._mpi_state_key(state, node_ids)
                        seed = sink_seeds.get((sweep_id, state_key))
                        seed = self._mpi_add(
                            seed, contributions.get((sweep_id, state_key))
                        )
                        output_seeds[name] = (
                            self._mpi_copy(seed)
                            if seed is not None
                            else self._mpi_zero(state.value)
                        )
                    variable_keys = {
                        name: self._mpi_state_key(state, node_ids)
                        for name, state in node.variables.items()
                    }
                    self.mpi_comm.send(
                        {
                            "kind": "adjoint_task",
                            "node_id": node_id,
                            "sweep_id": sweep_id,
                            "output_seeds": output_seeds,
                            "variable_keys": variable_keys,
                            "track": track,
                        },
                        dest=worker_rank,
                        tag=self._MPI_COMMAND_TAG,
                    )
                    busy[worker_rank] = (sweep_id, node_id)

                if not busy:
                    if failure is not None:
                        break
                    if ready:
                        raise RuntimeError(
                            "MPI adjoint scheduler deadlocked: ready tasks "
                            "have no available owner rank."
                        )
                    break

                result, worker_rank = self._mpi_receive_result()
                expected_sweep, expected_node_id = busy.pop(worker_rank)
                if (
                    result.get("node_id") != expected_node_id
                    or result.get("sweep_id") != expected_sweep
                ):
                    if failure is None:
                        failure = RuntimeError(
                            f"MPI worker {worker_rank} returned an unexpected "
                            "adjoint task result."
                        )
                else:
                    complete_node(expected_sweep, expected_node_id, result)

            if failure is not None:
                raise failure

        self.design_derivs = {}
        for sweep_id, _, _, _ in sinks_info:
            sweep_derivs = {}
            for global_name, metadata in self.design_vars_info.items():
                state = metadata["instance"].variables[metadata["local_name"]]
                state_key = self._mpi_state_key(state, node_ids)
                value = contributions.get((sweep_id, state_key))
                if value is None:
                    value = self._mpi_zero(state.value)
                state._deriv_store[sweep_id] = self._mpi_copy(value)
                sweep_derivs[global_name] = self._mpi_copy(value)
            self.design_derivs[sweep_id] = sweep_derivs

        if debug_print:
            print(
                f"MPI DAG adjoint pass completed on {self._mpi_size} rank(s); "
                f"sweeps={len(sinks_info)}"
            )

    @staticmethod
    def _mpi_sweep_sort_key(sweep_id):
        return repr(sweep_id)

    @staticmethod
    def _mpi_zero(value):
        return np.zeros(value.shape) if isinstance(value, np.ndarray) else 0.0

    @staticmethod
    def _mpi_add(left, right):
        if left is None:
            return None if right is None else System._mpi_copy(right)
        if right is None:
            return System._mpi_copy(left)
        return left + right

    def _mpi_state_key(self, state, node_ids):
        source = state.source
        if source not in node_ids:
            raise RuntimeError(
                f"State {state.desc!r} has a source outside this System's DAG."
            )
        for name, source_state in source.outputs.items():
            if source_state is state:
                return ("output", node_ids[source], name)
        for name, source_state in source.variables.items():
            if source_state is state:
                return ("variable", node_ids[source], name)
        raise RuntimeError(
            f"Could not identify State {state.desc!r} on its source analysis "
            f"{source.obj_name!r}."
        )

    def _mpi_run_adjoint_node(
        self, node, node_id, sweep_id, output_seeds, variable_keys, track
    ):
        import time
        from flume.base_classes.state import _current_sweep, _current_writer

        states = {
            id(state): state
            for state in list(node.outputs.values()) + list(node.variables.values())
        }
        for state in states.values():
            state.ensure_sweep_slot(sweep_id)
            state.prepare_writer_slot(sweep_id, node_id)
        for name, seed in output_seeds.items():
            node.outputs[name]._deriv_store[(sweep_id, node_id)] = self._mpi_copy(seed)

        token_sweep = _current_sweep.set(sweep_id)
        token_writer = _current_writer.set(node_id)
        start = time.perf_counter()
        try:
            node._analyze_adjoint()
            node.adjoint_analyzed = True
        finally:
            elapsed = time.perf_counter() - start
            _current_writer.reset(token_writer)
            _current_sweep.reset(token_sweep)

        contributions = {}
        for name, state_key in variable_keys.items():
            if state_key in contributions:
                continue
            state = node.variables[name]
            contributions[state_key] = self._mpi_copy(
                state._deriv_store[(sweep_id, node_id)]
            )
        return {
            "contributions": contributions,
            "elapsed": elapsed if track else 0.0,
        }

    def graph_network(
        self,
        filename: str = None,
        output_directory: str = None,
        interactive: bool = False,
        format: str = "pdf",
        consolidated_graph: bool = False,
    ):
        """
        Construct the visualization of the network associated with the Flume system using graphviz.

        Parameters
        ----------
        filename : str
            Name to use for the file that is created
        output_directory : str
            String that defines the directory where the file should be saved
        interactive : bool
            Boolean value that indicates whether the graph should be output in interactive mode. *This is an experimental feature at the moment*
        """

        # Create the graph, according to the interactive boolean argument
        if interactive:
            # Make the graph with graphviz in interactive mode
            graph = self._static_graph_network()

            # Render as an svg for interactive graph
            int_filename = filename + "_interactive"
            graph.render(
                filename=int_filename,
                directory=output_directory,
                format="svg",
                cleanup=True,
            )

            # Edit the SVG file to enable interactive features
            svg_filepath = output_directory + "/" + int_filename + ".svg"
            # ic(svg_filepath)

            # FIXME: Rewriting this adds ns0 to the output, need to fix
            self._enable_interactive_graph(svg_filepath=svg_filepath)

            # Embed the interactive svg into an HTML with the interactive capabilities
            self._create_interactive_html(
                output_directory=output_directory, svg_filepath=svg_filepath
            )

        elif consolidated_graph:
            # Make the consolidated version of the graph visualization
            graph = self._consolidate_static_graph_network(
                node_fillcolor="#8CD17D",
                top_level_fillcolor="#5FB7EA",
                opacity=0.5,
                penwidth=3,
            )

            # Render the graph
            graph.render(
                filename=filename,
                directory=output_directory,
                cleanup=True,
                format=format,
            )

        else:
            # Make the graph with graphviz
            graph = self._static_graph_network()

            # Render the graph
            graph.render(
                filename=filename,
                directory=output_directory,
                cleanup=True,
                format=format,
            )

        return graph

    def _static_graph_network(self):
        """
        Private method that is used to greate the graphviz visual in static form, which is ultimately returned by this method.
        """

        # graph = nx.Graph()
        graph = gv.Digraph(
            name=f"{self.sys_name.upper()}",
            graph_attr={"rankdir": "LR", "ranksep": "0.7"},
            node_attr={"shape": "box", "fontname": "Helvetica"},
        )

        # Initialize an empty list to store the systems added to the graph as nodes
        self.nodes = []
        self.edges = {}

        # Loop through and add nodes to the graph (outer loop is top-level analyses, inner loop is individual sub-analyses)
        for analysis in self.top_level_analysis_list:
            # Check if the object has a stack already, otherwise assemble
            if hasattr(analysis, "stack"):
                stack = analysis.stack
                pass
            else:
                stack = analysis._make_stack()

            # Check if the object has been connected already, otherwise perform the analysis to establish connections map
            if analysis.connected:
                pass
            else:
                analysis.analyze()

            # Loop through sub analyses in the current stack and add the nodes if they do not already exist
            for i, sub in enumerate(stack):
                # Add node if it is not in the system network already
                if sub not in self.nodes:
                    self.nodes.append(sub)

                    # Set the color depending on whether the analysis is top-level
                    if sub in self.top_level_analysis_list:
                        # Extract the names of the outputs
                        out_labels = list(sub.outputs.keys())
                        out_str = ", ".join(out_labels)

                        # Add the node
                        graph.node(
                            sub.obj_name,
                            f"{sub.obj_name}\nOutputs: {out_str}",
                            color="red",
                        )
                    else:
                        graph.node(sub.obj_name, f"{sub.obj_name}")
                else:
                    pass

                # Add edge if not the first entry in the stack
                if i == 0:
                    pass
                else:
                    # Extract the States that are connected between the objects, if there are any connections
                    if hasattr(sub, "connects"):
                        connect_labels = list(sub.connects.keys())

                        # Loop through the keys in the connections dictionary
                        for out in connect_labels:

                            # Add the entry to the edges dictionary for the current output, if necessary
                            if out not in self.edges:
                                self.edges[out] = []

                            # Check if the edge already exists in the edges dictionary
                            if (sub.connects[out].obj_name, sub.obj_name) in self.edges[
                                out
                            ]:
                                pass
                            else:
                                # Add the edge label to the edges dictionary
                                self.edges[out].append(
                                    (sub.connects[out].obj_name, sub.obj_name)
                                )

                                # Add the edge to the graph and the connection label
                                graph.edge(
                                    sub.connects[out].obj_name,
                                    sub.obj_name,
                                    label=f"{out}",
                                )

        return graph

    def _consolidate_static_graph_network(
        self,
        node_fillcolor: str = None,
        top_level_fillcolor: str = None,
        opacity: float = 1.0,
        penwidth: float = 2.0,
    ):
        """
        Private method that builds a decluttered version of the static graphviz visual.

        Compared to ``_static_graph_network``, this version:
          * Groups all variables flowing between the same (source, target) pair
            of analyses into a single edge (no edge labels).
          * Variable names are stored as a tooltip (visible in SVG/HTML output).
          * Enables ``concentrate=True`` so graphviz merges shared edge segments.
          * Renders nodes with rounded corners.
          * Optionally fills nodes with a hex color code.

        Parameters
        ----------
        node_fillcolor : str, optional
            Hex color code (e.g. ``"#AED6F1"``) used to fill non-top-level
            analysis nodes. If ``None``, no fill is applied.
        top_level_fillcolor : str, optional
            Hex color code (e.g. ``"#F1948A"``) used to fill top-level analysis
            nodes. If ``None``, defaults to ``node_fillcolor`` when that is set,
            otherwise no fill is applied.
        opacity : float, optional
            Fill opacity from 0.0 (transparent) to 1.0 (fully opaque). Applied
            to both ``node_fillcolor`` and ``top_level_fillcolor`` by appending
            an alpha byte to the hex color. Defaults to 1.0.

        Returns
        -------
        graph : graphviz.Digraph
            The consolidated graphviz digraph.
        """

        # Helper: append alpha byte to a 6-digit hex color (#RRGGBB → #RRGGBBAA)
        def _apply_opacity(color):
            if color is None:
                return None
            alpha = format(round(opacity * 255), "02X")
            return (
                color.rstrip().rstrip(")") + alpha if color.startswith("#") else color
            )

        # Helper: return fully-opaque version of a hex color (strips any alpha byte)
        def _border_color(color):
            if color is None:
                return None
            return "#" + color.lstrip("#")[:6]

        node_fillcolor = _apply_opacity(node_fillcolor)
        top_level_fillcolor = _apply_opacity(top_level_fillcolor)

        # Determine base node style
        base_node_attrs = {
            "shape": "box",
            "style": "rounded",
            "fontname": "Helvetica",
            "penwidth": str(penwidth),
        }
        if node_fillcolor is not None:
            base_node_attrs["style"] = "rounded,filled"
            base_node_attrs["fillcolor"] = node_fillcolor
            base_node_attrs["color"] = _border_color(node_fillcolor)

        graph = gv.Digraph(
            name=f"{self.sys_name.upper()}",
            graph_attr={"rankdir": "LR", "ranksep": "0.7", "concentrate": "true"},
            node_attr=base_node_attrs,
        )

        self.nodes = []
        self.edges = {}

        # Resolve top-level fill: explicit arg > fallback to node_fillcolor > none
        _top_fill = (
            top_level_fillcolor if top_level_fillcolor is not None else node_fillcolor
        )

        for analysis in self.top_level_analysis_list:
            stack = (
                analysis.stack if hasattr(analysis, "stack") else analysis._make_stack()
            )

            if not analysis.connected:
                analysis.analyze()

            for i, sub in enumerate(stack):
                if sub not in self.nodes:
                    self.nodes.append(sub)

                    if sub in self.top_level_analysis_list:
                        out_str = ", ".join(sub.outputs.keys())
                        top_attrs = {}
                        if _top_fill is not None:
                            top_attrs["style"] = "rounded,filled"
                            top_attrs["fillcolor"] = _top_fill
                            top_attrs["color"] = _border_color(_top_fill)
                        label = f"<<B>{sub.obj_name}</B><BR/>>"  # <I>Outputs: {out_str}</I>>"
                        graph.node(
                            sub.obj_name,
                            label,
                            **top_attrs,
                        )
                    else:
                        graph.node(sub.obj_name, f"{sub.obj_name}")

                if i == 0 or not hasattr(sub, "connects"):
                    continue

                # Group variables by (source, target) pair
                grouped = {}
                for out, source in sub.connects.items():
                    grouped.setdefault((source.obj_name, sub.obj_name), []).append(out)

                for edge_key, labels in grouped.items():
                    if edge_key in self.edges:
                        continue
                    self.edges[edge_key] = labels
                    tooltip = ", ".join(labels)
                    graph.edge(*edge_key, tooltip=tooltip, labeltooltip=tooltip)

        return graph

    def _enable_interactive_graph(self, svg_filepath):
        """
        Privat method that is used to enable interactive capabilities with the file provided with svg_filepath

        Parameters
        ----------
        svg_filepath : str
            String that provides a filepath to an SVG file, which will be modified to make it interactive
        """

        # Import xml
        import xml.etree.ElementTree as ET

        # Parse the graph
        ET.register_namespace("", "http://www.w3.org/2000/svg")
        tree = ET.parse(svg_filepath)
        root = tree.getroot()

        # Set the id for the svg
        root.set("id", "my-svg")

        # Find node groups and edit
        for g in root.iter("{http://www.w3.org/2000/svg}g"):
            # Get the class type for the attribute
            attrib_class = g.attrib["class"]
            # print(attrib_class)

            # Replace the classes for graphs/nodes
            if attrib_class == "node":
                g.set("class", "graph-node")

            # title = g.find("{http://www.w3.org/2000/svg}title")
            # print(g.attrib)
            # g.get()
            # print(g)
            # print(title.text)

        # Rewrite the file
        tree.write(svg_filepath)

        return

    def _create_interactive_html(self, output_directory, svg_filepath):
        """
        Privat emethod that creates an interactive HTML using the file located at output_directory/svg_filepath

        Parameters
        ----------
        output_directory : str
            String that specifies the location of the output directory for the HTML
        svg_filepath : str
            String that specifies the location of the SVG file that is to be converted to interactive mode
        """

        from flume.base_classes.system_html import _write_html_file

        _write_html_file(output_directory=output_directory, svg_filepath=svg_filepath)

        return

    def _find_analysis_object(self, instance_name, var_name):
        """
        Searches through the full analysis list for the system and finds the instance of the analysis object associated with the argument for the instance_name.
        """

        # Loop through the full analysis list
        for analysis in self.full_analysis_list:
            # Check if the current object name matches the provided instance name and return if so
            if analysis.obj_name == instance_name:
                return analysis

        # Raise a RuntimeError if this point is reached, as the instance name did not find a match
        raise RuntimeError(
            f"No instance found for object named '{instance_name}'! Verify the definition for the State named '{instance_name}.{var_name}'"
        )

    def declare_objective(self, global_obj_name, obj_scale=1.0):
        """
        Sets the objective function for the optimization problem according to the provided global output name. This output should be associated with one of the top-level analyses for the system (i.e. included in the top_level_analysis_list).

        Parameters
        ----------
        global_obj_name : str
            A string that specifies the global name for the value that is to be used as the objective function (global name meaning 'object_name.local_output_name')
        obj_scale : float
            Float value that is *multiplied* by the value of the objective function, which should scale the objective function value to O(1). This is used by the optimizer interfaces internally to scale the objective.
        """

        # Using the provided objective name, store the associated analysis object and the local variable name
        self.global_obj_name = global_obj_name
        obj_analysis_name, self.obj_local_name = global_obj_name.split(".", maxsplit=1)

        # Store the objective scale
        self.obj_scale = obj_scale

        # Find the instance for the objective analysis object
        self.obj_analysis = self._find_analysis_object(
            instance_name=obj_analysis_name, var_name=self.obj_local_name
        )

        return

    def declare_constraints(self, global_con_name: dict):
        """
        Sets the constraints for the optimization problem according to the provided global output names.

        Parameters
        ----------
        global_con_name_dict : dict
            This is a dictionary of dictionaries. The keys correspond to the global output names for the constraints. The inner dictionary specifies additional information about the structure of the constriant, including the following:

            * 'rhs' (float) - specifies the right hand side of the constraint equation. Internally, the optimizer interfaces will use this to convert the constraint to an equivalent form that normalizes it, If the 'rhs' value is 0.0, no scaling is applied
            * 'direction' (str) - string that is either 'geq' (>=), 'leq' (<=), or 'both' (=). Defaults to 'geq' in the event that a direction is not provided.

        Example
        -------

        Here, the constraints are defined as follows:
            x <= 1.0
            y >= 1.0
            z = 2.0

        Thus, the argument here is given as: global_con_name = {
                                                    "block1.x":{"direction":"leq", "rhs":1.0},
                                                    "block2.y":{"rhs":1.0}
                                                    "block3.z":{"direction": "both", "rhs": 2.0}
                                                    }
        """

        # Loop through the keys in the dictionary and add them to the constraints for the system
        for key in global_con_name.keys():
            # Add the key to the con_info dictionary
            self.con_info[key] = {}

            # Split the string
            con_analysis_name, con_local_name = key.split(".", maxsplit=1)

            # Find the analysis object for the current constraint
            con_analysis = self._find_analysis_object(
                con_analysis_name, var_name=con_local_name
            )

            # Add the analysis object and local constraint name for the con_info dictionary
            self.con_info[key]["instance"] = con_analysis
            self.con_info[key]["local_name"] = con_local_name
            self.con_info[key]["rhs"] = global_con_name[key]["rhs"]

            if "direction" not in global_con_name[key].keys():
                self.con_info[key]["direction"] = "geq"
            else:
                if global_con_name[key]["direction"] not in ["geq", "leq", "both"]:
                    raise RuntimeError(
                        f"The value for 'direction' for the constraint {key} must be 'geq', 'leq', or 'both' and not {self.con_info[key]['direction']}"
                    )
                else:
                    self.con_info[key]["direction"] = global_con_name[key]["direction"]

        return

    def declare_design_vars(self, global_var_name: dict):
        """
        Sets the design variables for the system according to the provided global variable names.

        Parameters
        ----------
        global_var_name : dict
            This is a dictionary of dictionaries. Keys for the first dictionary correspond to the global variable names that should be added to the set of design variables, and the values contain information about the variable bounds. If the inner dictionary is empty, then no bounds are provided. Otherwise, the values for the variables lower bound 'lb' and upper bound 'ub' will be added, if included.

        Example
        -------

        Here, the bounds are defined as follows:
            1.0 <= x <= 2.0
            y has no bounds
            0.0 <= z

        Thus, the argument here is: global_var_name = {"block1.x":{"lb":1.0, "lb":2.0}, "block2.y":{}, "block3.z":{"lb":0.0}}
        """

        # Initialize the design variables dictionary
        self.design_vars_info = {}

        # Loop through the keys in the dictionary and add them to the design variables for the system
        for key in global_var_name:
            # Add the key to the design_vars_info dictionary
            self.design_vars_info[key] = {}

            # Split the string for the current variable
            var_analysis_name, var_local_name = key.split(".", maxsplit=1)

            # Find the analysis object instance for the current design variable
            var_analysis = self._find_analysis_object(var_analysis_name, var_local_name)

            # Add the analysis object and local variable name to the dictionary
            self.design_vars_info[key]["instance"] = var_analysis
            self.design_vars_info[key]["local_name"] = var_local_name

            # Add the bounds, if they were specified
            if global_var_name[
                key
            ]:  # This is evaluated as True if the variable has at leaast one bound specified

                # Add the lower bound, if specified
                if "lb" in global_var_name[key].keys():
                    self.design_vars_info[key]["lb"] = global_var_name[key]["lb"]

                # Add the upper bound, if specified
                if "ub" in global_var_name[key].keys():
                    self.design_vars_info[key]["ub"] = global_var_name[key]["ub"]

                # Add the scale, if specified
                if "scale" in global_var_name[key].keys():
                    self.design_vars_info[key]["scale"] = global_var_name[key]["scale"]

            else:
                continue

        return

    def declare_foi(self, global_foi_name: list):
        """
        Sets the functions of interest that are to be tracked by the logger at each iteration. Here, the names must be provided according to their global names. If already declared, the objective and constraints are added by default, but other outputs can be included.
        """

        # Initialize the foi dictionary
        foi = {}

        # Add the objective if it has already been declared
        if hasattr(self, "obj_analysis"):
            foi["obj"] = {}

            foi["obj"]["instance"] = self.obj_analysis
            foi["obj"]["local_name"] = self.obj_local_name

        else:
            raise RuntimeError("No objective has been declared!")

        # Add the constraints if it has already been declared
        if hasattr(self, "con_info"):
            foi["cons"] = {}

            # For each constraint in self.con_info, add it to the foi dictionary
            for key in self.con_info:
                foi["cons"][key] = {}

                foi["cons"][key]["instance"] = self.con_info[key]["instance"]
                foi["cons"][key]["local_name"] = self.con_info[key]["local_name"]

        # Add the names in the global_foi_name input list to the foi dictionary
        foi["other"] = {}

        for name in global_foi_name:
            # Split the name for the global foi
            foi_analysis_name, foi_local_name = name.split(".", maxsplit=1)

            # Find the analysis object associated with the global_foi_name
            foi_analysis = self._find_analysis_object(
                instance_name=foi_analysis_name, var_name=foi_local_name
            )

            # Add the analysis object and local state name to the foi dictionary
            foi["other"][name] = {}
            foi["other"][name]["instance"] = foi_analysis
            foi["other"][name]["local_name"] = foi_local_name

        # Store the foi dictionary
        self.foi = foi

        return

    def _compute_log_columns(self):
        """
        Computes column headers and widths for use in log_information. Each column
        width is set to the maximum of the header label length (plus 2 for padding)
        and 20 (to accommodate values formatted with %20.10e).

        Returns
        -------
        columns : list of tuple
            Each entry is (header_str, width, category, key, index) where category
            is 'obj', 'con', or 'other', key is the dictionary key, and index is
            the array index (or None for scalars).
        """

        columns = []

        # Objective column
        obj_header = f"obj: {self.obj_local_name}"
        columns.append(("obj", None, None, obj_header))

        # Constraint columns
        for con in self.foi["cons"].keys():
            con_val = (
                self.foi["cons"][con]["instance"]
                .outputs[self.foi["cons"][con]["local_name"]]
                .value
            )

            if isinstance(con_val, np.ndarray):
                for i in range(con_val.size):
                    con_header = f"con: {self.foi['cons'][con]['local_name']}[{i}]"
                    columns.append(("con", con, i, con_header))
            else:
                con_header = f"con: {self.foi['cons'][con]['local_name']}"
                columns.append(("con", con, None, con_header))

        # Other FOI columns
        for other in self.foi["other"].keys():
            other_val = (
                self.foi["other"][other]["instance"]
                .outputs[self.foi["other"][other]["local_name"]]
                .value
            )

            if isinstance(other_val, np.ndarray):
                for i in range(other_val.size):
                    other_header = (
                        f"other: {self.foi['other'][other]['local_name']}[{i}]"
                    )
                    columns.append(("other", other, i, other_header))
            else:
                other_header = f"other: {self.foi['other'][other]['local_name']}"
                columns.append(("other", other, None, other_header))

        # Compute widths: at least 20 (for numeric formatting), or header length + 2 for padding
        col_info = []
        for category, key, index, header in columns:
            width = max(len(header) + 2, 20)
            col_info.append(
                {
                    "category": category,
                    "key": key,
                    "index": index,
                    "header": header,
                    "width": width,
                }
            )

        return col_info

    def log_information(self, iter_number):
        """
        Helper function that is used to log the values for the objective function, constraints, and other functions of interest at each iteration. Internally, this will update the log file for the System with this information at every iteration.

        Parameters
        ----------
        iter_number : int
            Current iteration number
        """

        # Check that the system has an FOI attribute, otherwise generate it (only needed if the user does not declare additional FOI to track)
        if not hasattr(self, "foi"):
            self.declare_foi(global_foi_name=[])

        # Compute column layout (recompute every header cycle in case FOI structure changes)
        if iter_number % 10 == 0 or not hasattr(self, "_log_columns"):
            self._log_columns = self._compute_log_columns()

        # Log the header names if the current iter number is divisible by 10
        if iter_number % 10 == 0:
            self.outputs_log.log("\n%5s" % "iter", end="")
            for col in self._log_columns:
                fmt = f"%{col['width']}s"
                self.outputs_log.log(fmt % col["header"], end="")

        # Log the values for the current iteration
        self.outputs_log.log("\n%5d" % iter_number, end="")

        for col in self._log_columns:
            width = col["width"]
            category = col["category"]
            key = col["key"]
            index = col["index"]

            # Retrieve the value based on category
            if category == "obj":
                val = (
                    self.foi["obj"]["instance"]
                    .outputs[self.foi["obj"]["local_name"]]
                    .value
                )
            elif category == "con":
                val = (
                    self.foi["cons"][key]["instance"]
                    .outputs[self.foi["cons"][key]["local_name"]]
                    .value
                )
                if isinstance(val, np.ndarray):
                    val = val[index]
            else:  # "other"
                val = (
                    self.foi["other"][key]["instance"]
                    .outputs[self.foi["other"][key]["local_name"]]
                    .value
                )
                if isinstance(val, np.ndarray):
                    val = val[index]

            # Format the value
            if not isinstance(val, str):
                val_str = "%20.10e" % val
            else:
                val_str = val

            fmt = f"%{width}s"
            self.outputs_log.log(fmt % val_str, end="")

        return

    def profile_iteration(self, iter_number):
        """
        Helper function that is used to display the time taken for each analyze and analyze_adjoint method at the current iteration. Internally, this updates a profile log file that stores the timing information for the System at each iteration.

        Parameters
        ----------
        iter_number : int
            Current iteration number
        """
        import time

        # Record the wall-clock start time on the first profiled iteration. This marks
        # the beginning of the optimization run and is used by finalize_profile_log()
        # to report the total elapsed run time at the end of profile.log.
        if not hasattr(self, "_profile_start_time"):
            self._profile_start_time = time.perf_counter()

        # Gather the System-level timing metrics captured by execute()/execute_adjoint().
        # Missing attributes (e.g. a pass that has not run yet) are reported as NaN.
        fwd_wall = getattr(self, "dag_forward_wall", float("nan"))
        fwd_cpu = getattr(self, "dag_total_time", float("nan"))
        adj_wall = getattr(self, "dag_adjoint_wall", float("nan"))
        adj_cpu = getattr(self, "dag_adjoint_time", float("nan"))

        def _eff(cpu, wall):
            # Summed node CPU time / wall time ~ effective concurrency (1.0 == serial).
            # NaN when either value is unavailable (cpu-sum requires track_node_timing).
            if wall == wall and wall > 0 and cpu == cpu:
                return cpu / wall
            return float("nan")

        # Log the header and the execution mode every 10 iterations
        if iter_number % 10 == 0:
            mode = (
                "parallel"
                if self.execution_backend == "thread"
                else self.execution_backend
            )
            self.profile_log.log(
                "\n# mode: %s, max_workers: %s, track_node_timing: %s"
                % (
                    mode,
                    getattr(self, "parallel_max_workers", "-"),
                    getattr(self, "track_node_timing", False),
                ),
                end="",
            )
            self.profile_log.log(
                "\n%5s%14s%14s%12s%14s%14s%12s"
                % (
                    "iter",
                    "fwd_wall",
                    "fwd_cpu",
                    "fwd_eff",
                    "adj_wall",
                    "adj_cpu",
                    "adj_eff",
                ),
                end="",
            )

        # Log the System-level timing row for the current iteration
        self.profile_log.log(
            "\n%5d%14.6f%14.6f%12.2f%14.6f%14.6f%12.2f"
            % (
                iter_number,
                fwd_wall,
                fwd_cpu,
                _eff(fwd_cpu, fwd_wall),
                adj_wall,
                adj_cpu,
                _eff(adj_cpu, adj_wall),
            ),
            end="",
        )

        # Optionally log a per-node forward/adjoint-time breakdown when per-node
        # timing is tracked. Uses the unique DAG nodes so shared sub-analyses are not
        # double counted (unlike a per-top-level breakdown).
        if getattr(self, "track_node_timing", False) and hasattr(self, "dag_nodes"):
            for n in sorted(
                self.dag_nodes,
                key=lambda a: getattr(a, "analysis_time", 0.0),
                reverse=True,
            ):
                self.profile_log.log(
                    "\n    %-28s fwd=%12.6f  adj=%12.6f"
                    % (
                        n.obj_name,
                        getattr(n, "analysis_time", float("nan")),
                        getattr(n, "adjoint_time", float("nan")),
                    ),
                    end="",
                )

        return

    def finalize_profile_log(self):
        """
        Writes a final summary line to profile.log recording the total wall-clock run
        time for the entire optimization. The run start is captured lazily on the first
        call to profile_iteration(), so this should be called once after the
        optimization loop completes.

        If profile_iteration() was never called (e.g. no iterations ran), the total
        run time is reported as NaN.
        """
        import time

        # Compute the total elapsed run time since the first profiled iteration
        if hasattr(self, "_profile_start_time"):
            total_run_time = time.perf_counter() - self._profile_start_time
        else:
            total_run_time = float("nan")

        # Store the value as an attribute for programmatic access
        self.total_run_time = total_run_time

        # Append the summary line to the end of the profile log
        self.profile_log.log(
            "\n# total optimization run time: %.6f s" % total_run_time,
            end="",
        )

        return
