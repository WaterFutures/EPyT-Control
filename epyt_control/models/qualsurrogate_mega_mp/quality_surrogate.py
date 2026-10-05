"""
This file contains a class
(:class:`~epyt_control.models.qualsurrogate_mega_mp.quality_surrogate.QualitySurrogateModel`)
for a high-level and another class
(:class:`~epyt_control.models.qualsurrogate_mega_mp.quality_surrogate.QualitySurrogateModelEx`)
for a low-level interface to the water quality surrogate.
"""
import torch
import numpy as np
from epyt_flow.simulation import EpanetConstants, NetworkTopology, ScadaData

from .utils import get_edge_attribute, make_edge_index, flow_to_velocity
from .model.advection_model_mp import AdvectionModelMP
from .model.advection_layer import AdvectionLayer
from .modules import AdvectionModuleGridSampleDynamic, compute_backward_transit_times_fast


class QualitySurrogateModelEx():
    """
    A surrogate model for chemical concentrations in a water distribution system.

    Parameters
    ----------
    edge_index: `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
        Indices of all links/edges.
    interpolation: `str`, optional
        Interpolation mode.
        Must be one of the following:

            - "nearest"
            - "bilinear"
            - "bicubic"

        The default is "bilinear".
    max_msg_passing_rounds: `int`, optional
        Maximum number of message-passing rounds.

        The default is 300.
    verbose: `bool`, optional
        If True, progress of computation will be shown.

        The default is False.
    mixing_at_nodes: `bool`, optional
        If True, mixing at nodes will be calculated.

        The default is True.
    iterate: `bool`, optional
        The default is False.
    adaptive_steps: `bool`, optional
        The default is False.
    device: `str`, optional
        Name of the device being used by PyTorch.

        The default is "cpu".
    """
    def __init__(self, edge_index: np.ndarray, interpolation: str = 'bilinear',
                 max_msg_passing_rounds: int = 300, verbose: bool = False,
                 mixing_at_nodes: bool = True, iterate: bool = False,
                 adaptive_steps: bool = False, device: str = 'cpu'):
        if not isinstance(edge_index, np.ndarray):
            raise TypeError("'edge_index' must be an instance of 'numpy.ndarray' " +
                            f"but not of '{type(edge_index)}'")
        if interpolation not in ["nearest", "bilinear", "bicubic"]:
            raise ValueError("Invalid value of 'interpolation'. " +
                             "Must be on of the following: 'nearest', 'bilinear', 'bicubic'")
        if not isinstance(max_msg_passing_rounds, int):
            raise TypeError("'max_msg_passing_rounds' must be an instance of 'int' " +
                            f"but not of '{type(max_msg_passing_rounds)}'")
        if max_msg_passing_rounds <= 0:
            raise ValueError("'max_msg_passing_rounds' must be > 0")
        if not isinstance(verbose, bool):
            raise TypeError("'verbose' must be an instance of 'bool' " +
                            f"but not of '{type(verbose)}'")
        if not isinstance(mixing_at_nodes, bool):
            raise TypeError("'mixing_at_nodes' must be an instance of 'bool' " +
                            f"but not of '{type(mixing_at_nodes)}'")
        if not isinstance(iterate, bool):
            raise TypeError("'iterate' must be an instance of 'bool' " +
                            f"but not of '{type(iterate)}'")
        if not isinstance(adaptive_steps, bool):
            raise TypeError("'adaptive_steps' must be an instance of 'bool' " +
                            f"but not of '{type(adaptive_steps)}'")
        if not isinstance(device, str):
            raise TypeError("'device' must be an instance of 'str' " +
                            f"but not of '{type(device)}'")

        self._edge_index = edge_index
        self._iterate = iterate
        self._device = device
        
        advection_op = AdvectionModuleGridSampleDynamic(interpolation_mode=interpolation)

        layer = AdvectionLayer(advection_op, mixing_at_nodes)
        self._model = AdvectionModelMP(
            layer, max_msg_passing_rounds=max_msg_passing_rounds,
            progress=verbose, adaptive_steps=adaptive_steps
        )
    
        self._model.to(self._device)

    def _prepare_model_inputs(self, initial_state, flow_field, control_inputs: np.ndarray,
                              control_indices: list[int], edge_lengths, edge_capacities,
                              output_times, dt, nsteps):
        _, n_edges = self._edge_index.shape
        flow_field_graph = flow_field[:n_edges]

        traversal_times, selfloop_mask, xs_map = compute_backward_transit_times_fast(
            edge_lengths, flow_field_graph, dt
        )
        traversal_times = np.nan_to_num(traversal_times, nan=nsteps*dt)
        flow_field_graph = torch.as_tensor(flow_field_graph, dtype=torch.get_default_dtype(), device=self._device)

        edge_capacities = torch.as_tensor(edge_capacities, dtype=torch.get_default_dtype(), device=self._device)
        initial_state = torch.as_tensor(initial_state, dtype=torch.get_default_dtype(), device=self._device)
        traversal_times = torch.as_tensor(traversal_times, dtype=torch.get_default_dtype(), device=self._device)
        xs_map = torch.as_tensor(xs_map, dtype=torch.get_default_dtype(), device=self._device)
        selfloop_mask = torch.as_tensor(selfloop_mask, dtype=torch.get_default_dtype(), device=self._device)
        edge_index = torch.as_tensor(self._edge_index, device=self._device)

        if control_indices is not None:
            boundary_index = torch.tensor(control_indices, device=self._device)
            boundary_input = torch.tensor(control_inputs, dtype=torch.get_default_dtype(), device=self._device)
            if boundary_input.ndim < 2:
                boundary_input = boundary_input.unsqueeze(0)
            if boundary_input.ndim < 3:
                boundary_input = boundary_input.unsqueeze(-1)
        else:
            boundary_index = boundary_input = None
        
        if edge_capacities.ndim < 2 and edge_capacities.ndim > 0:
            edge_capacities = edge_capacities.unsqueeze(1)
        if initial_state.ndim < 2:
            initial_state = initial_state.unsqueeze(-1)
        if initial_state.ndim < 3:
            initial_state = initial_state.unsqueeze(-1)

        delay_steps = traversal_times.clamp(max=max(output_times))
        delay_steps = delay_steps / dt
        
        if self._iterate:
            n_steps = nsteps
            nsteps = 1
        else:
            n_steps = 1

        return {
            'x' : initial_state,
            'edge_index' : edge_index,
            'flows' : flow_field_graph * edge_capacities,
            'delay_steps' : delay_steps,
            'xs_map' : xs_map,
            'sl_mask' : selfloop_mask,
            'Tau' : nsteps,
            'n_steps' : n_steps,
            'boundary_values' : boundary_input,
            'boundary_index' : boundary_index,
        }

    def predict_ex(self, initial_state: np.ndarray, flow_field, control_inputs: np.ndarray,
                   control_indices: list[int], edge_lengths: np.ndarray, edge_diameters: np.ndarray,
                   output_times: list[int], dt: int, nsteps: int):
        model_inputs = self._prepare_model_inputs(
            initial_state, flow_field, control_inputs, control_indices, edge_lengths,
            edge_diameters, output_times, dt, nsteps
        )

        pred, edge_passes, _, agg_time, aggs_all = self._model(**model_inputs)
    
        pred[:, :model_inputs['x'].shape[1]] = model_inputs['x']
    
        return pred.cpu(), edge_passes, _, agg_time, aggs_all

    def predict_conc(self, initial_state: np.ndarray, flow_field: np.ndarray,
                     control_inputs: np.ndarray, control_indices: list[int],
                     edge_lengths: np.ndarray, edge_diameters: np.ndarray, output_times: list[int],
                     dt: int, nsteps: int):
        """
        Predicts the concentration at every node over time.

        Parameters
        ----------
        initial_state : `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
            Initial concentration at nodes.
        flow_field : `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
            Flow velocities in m/s.
        edge_lengths : `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
            Lengths of all edges/links in meter.
        edge_diameters : `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
            Diatemers of all edges/links in meter.
        output_times : `list[int]`
            Output times.
        dt : `int`
            Hydraulic time step.
        nsteps : `int`
            Number of steps to "simulate".

        Returns
        -------
        `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
            Node concentrations (1. axis) over time (2. axis).
        """
        if not isinstance(initial_state, np.ndarray):
            raise TypeError("'initial_state' must be an instance of 'numpy.ndarray' " +
                            f"but not of '{type(initial_state)}'")
        if not isinstance(flow_field, np.ndarray):
            raise TypeError("'flow_field' must be an instance of 'numpy.ndarray' " +
                            f"but not of '{type(flow_field)}'")
        if not isinstance(edge_lengths, np.ndarray):
            raise TypeError("'edge_lengths' must be an instance of 'numpy.ndarray' " +
                            f"but not of '{type(edge_lengths)}'")
        if not isinstance(edge_diameters, np.ndarray):
            raise TypeError("'edge_diameters' must be an instance of 'numpy.ndarray' " +
                            f"but not of '{type(edge_diameters)}'")
        if not isinstance(output_times, list):
            raise TypeError("'output_times' must be an instance of 'list' " +
                            f"but not of '{type(output_times)}'") 
        if not isinstance(dt, int):
            raise TypeError(f"'dt' must be an instance of 'int' but not of '{type(dt)}'")
        if dt <= 0:
            raise ValueError("'dt' must be > 0")
        if not isinstance(nsteps, int):
            raise TypeError(f"'nsteps' must be an instance of 'int' but not of '{type(nsteps)}'")
        if nsteps <= 0:
            raise ValueError("'nsteps' must be > 0")

        return np.squeeze(self.predict_ex(initial_state, flow_field, control_inputs,
                                          control_indices, edge_lengths, edge_diameters,
                                          output_times, dt, nsteps)[0].numpy().T)



class QualitySurrogateModel(QualitySurrogateModelEx):
    """
    A surrogate model for chemical concentrations in a water distribution system.
    This class provides an easy-to-use high-level interface -- for a more technical interface,
    please take a look at
    :class:`~epyt_control.models.qualsurrogate_mega_mp.quality_surrogate.QualitySurrogateModelEx`.

    Parameters
    ----------
    network_topo: epyt_flow.topology.NetworkTopology
        Topology of the water network.
    """
    def __init__(self, network_topo: NetworkTopology):
        if not isinstance(network_topo, NetworkTopology):
            raise TypeError("'network_topo' must be an instance of "+
                            f"'epyt_flow.topology.NetworkTopology' but not of '{type(network_topo)}'")
        if len(network_topo.get_all_tanks()):
            raise ValueError("Tanks are not supported.")

        self._network_topo = network_topo
        if self._network_topo.flow_units != EpanetConstants.EN_CMH:
            self._network_topo = \
                self._network_topo.convert_units(flow_units=EpanetConstants.EN_CMH,
                                                 pressure_units=network_topo.pressure_units)

        self._edge_lengths = get_edge_attribute(self._network_topo, 'length')
        self._edge_diameters = get_edge_attribute(self._network_topo, 'diameter')/10/100

        super().__init__(edge_index=make_edge_index(self._network_topo),
                         interpolation='bilinear',
                         max_msg_passing_rounds=1000)

    def predict(self, source_conc: dict[str, list[float]], hyd_scada_data: ScadaData,
                initial_state: np.ndarray = None) -> np.ndarray:
        """
        Predicts the concentration at all nodes over time based on a given source node
        concentration pattern and hydraulic data (i.e., flow rates).

        Parameters
        ----------
        source_conc: `dict[str, list[float]]`
            Concentration at the source nodes over time --
            key: node ID, value: concentration over time
        hyd_scada_data: `epyt_flow.simulation.scada_data.ScadaData`
            Hydraulic data -- i.e., flow rate at every link.
        initial_state: `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_, optional
            Initial concentration at nodes.
            If None, a concentration of zero will be used everywhere.

            The default is None.

        Returns
        -------
        `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
            Concentrations at every node (2. axis) over time (1. axis).
        """
        if not isinstance(source_conc, dict):
            raise TypeError("'source_conc' must be an instance of 'dict' " +
                            f"but not of '{type(source_conc)}'")
        if any([n_id not in self._network_topo.get_all_nodes() for n_id in source_conc.keys()]):
            raise ValueError("Invalid node ID in 'source_conc'")
        if not isinstance(hyd_scada_data, ScadaData):
            raise TypeError("'hyd_scada_data' must be an instance of " +
                            f"'epyt_flow.simulation.ScadaData' but not of '{type(hyd_scada_data)}'")
        if initial_state is not None:
            if not isinstance(initial_state, np.ndarray):
                raise TypeError("'initial_state' must be an instance of 'numpy.ndarray' " +
                                f"but not of '{type(initial_state)}'")
            if initial_state.shape != (self._network_topo.get_number_of_nodes()):
                raise ValueError("Invalid shape of 'initial_state' -- must be one dimensional, " +
                                 f"i.e., ({self._network_topo.get_number_of_nodes()})")

        topo = hyd_scada_data.network_topo

        if topo.flow_units != EpanetConstants.EN_CMH:
            hyd_scada_data = hyd_scada_data.convert_units(flow_unit=EpanetConstants.EN_CMH)

        sensor_readings_time = hyd_scada_data.sensor_readings_time
        sim_duration = sensor_readings_time[-1]
        hyd_step = int(sensor_readings_time[1] - sensor_readings_time[0])

        nsteps = int(sim_duration / hyd_step)
        output_times = [nsteps * hyd_step]
        
        edge_flows = hyd_scada_data.get_data_flows()
        flow_field = flow_to_velocity(topo, edge_flows).T

        boundary_index = []
        boundary_values = []
        for node_id, node_conc_pattern in source_conc.items():
            boundary_index.append(self._network_topo.get_all_nodes().index(node_id))
            boundary_values.append(np.concatenate((np.array([0]), node_conc_pattern[:-1])).T)
        boundary_values = np.array(boundary_values)

        if initial_state is None:
            initial_state = np.zeros(topo.get_number_of_nodes())

        conc_pred = self.predict_conc(initial_state=initial_state,
                                      flow_field=flow_field,
                                      control_indices=boundary_index,
                                      control_inputs=boundary_values,
                                      edge_lengths=self._edge_lengths,
                                      edge_diameters=self._edge_diameters,
                                      output_times=output_times,
                                      nsteps=nsteps,
                                      dt=hyd_step)
        return conc_pred
