"""
This example demonstrates how to use the low-level interface for setting up and using the
water quality surrogate model for predicting chlorine (CL2) concentrations at every node in
the Hanoi network based on a setpoint source at the reservoir.

Note that there also exist a high-level interface for easier usage if no full control
over all parameters of the surrogate is needed.
"""
import numpy as np
from epyt_control.models import QualitySurrogateModelEx, make_edge_index, get_edge_attribute, flow_to_velocity
from epyt_flow.utils import to_seconds, plot_timeseries_prediction, plot_timeseries_data
from epyt_flow.data.networks import load_hanoi

from sim_utils import run_simulation


def generate_injection_pattern(num_points, min_modes=3, max_modes=30, seed=None):
    rs = np.random.RandomState(seed)
    
    t = np.linspace(0, 2 * np.pi, num_points)
    signal = np.zeros(num_points)
    num_modes = rs.randint(min_modes, max_modes)
    decay_power = rs.uniform(0.5, 1.5)
    
    for k in range(1, num_modes + 1):
        amplitude_scale = 1.0 / (k**decay_power)
        a_k = rs.normal() * amplitude_scale
        b_k = rs.normal() * amplitude_scale
        signal += a_k * np.cos(k * t) + b_k * np.sin(k * t)
        
    return (signal - signal.min()) / (signal.max() - signal.min())


if __name__ == "__main__":
    # Load Hanoi network and specify general scenario parameters
    f_inp_in = load_hanoi().f_inp_in
    f_msx_in = "simplecl2.msx"  # Simple chlorine dynamics. Note that this .msx file does not contain any topological information and can therefore be used for any network!
    sources_at = ['1']  # Chlorine source at the reservoir!

    N_SECONDS = to_seconds(hours=12)
    HYDRAULIC_STEP = 60
    QUALITY_TIMESTEP = 1
    PATTERN_STEP = HYDRAULIC_STEP
    nsteps = int(N_SECONDS / HYDRAULIC_STEP)
    times = np.linspace(0, N_SECONDS, nsteps)

    # Create chlorine concentration pattern to be used as a setpoint at the reservoir
    injection_pattern = generate_injection_pattern((N_SECONDS // HYDRAULIC_STEP) + 1, seed=42)
    plot_timeseries_data(injection_pattern.reshape(1, -1),
                         x_axis_label="Time steps",
                         y_axis_label="CL2 concentration for source node")

    # Run hydraulic simulation to get the flow rates and ground truth chlorine concentrations at every node
    sim_kwargs = {
        'duration' : N_SECONDS,
        'hydraulic_step' : HYDRAULIC_STEP,
        'quality_step' : QUALITY_TIMESTEP,
        'f_msx_in' : f_msx_in,  # If set to None, EPANET is used instead of EPANET-MSX for simulating the water quality dynamics
        'source_conc' : {n_id: injection_pattern for n_id in sources_at}
    }

    scada_data = run_simulation(f_inp_in, **sim_kwargs)  # Uses EPyT-Flow for running EPANET and EPANET-MSX simulations

    # Extrac ground truth concentration
    if sim_kwargs["f_msx_in"] is None:
        conc = scada_data.get_data_nodes_quality()
    else:
        conc = scada_data.get_data_bulk_species_node_concentration()

    # Create parameters for the surrogate model
    topo = scada_data.network_topo
    initial_state = np.zeros(topo.get_number_of_nodes())    # Zero concentration everywhere in the beginning

    output_times = [nsteps * HYDRAULIC_STEP]
    edge_index = make_edge_index(topo)
    edge_lengths = get_edge_attribute(topo, 'length')
    edge_diameters = get_edge_attribute(topo, 'diameter')/10/100  # Make sure to get the units right! -- i.e., mm to meters

    # Create surrogate model
    surrogate = QualitySurrogateModelEx(edge_index=edge_index,
                                        interpolation='bilinear',
                                        max_msg_passing_rounds=1000)

    # Build boundary conditions based on the source pattern
    boundary_indices =  [topo.get_all_nodes().index(n) for n in sim_kwargs["source_conc"].keys()]
    boundary_values = np.concatenate((np.array([0]), injection_pattern[:-1])).reshape(1, -1)
    #boundary_values = conc[:, boundary_indices].T  # Alternatively, we could use the results form the ground truth simulation

    # Build flow field
    edge_flows = scada_data.get_data_flows()
    flow_field = flow_to_velocity(topo, edge_flows).T

    pred, edge_passes, _, agg_time, aggs_all = surrogate.predict_ex(initial_state=initial_state,     # Initial concentation at all nodes
                                                                    control_indices=boundary_indices,  # Indices of source nodes
                                                                    control_inputs=boundary_values,  # Concentation of those source nodes over time
                                                                    flow_field=flow_field,           # Flow velocity at all links
                                                                    edge_lengths=edge_lengths,       # Length of all links in m
                                                                    edge_diameters=edge_diameters,   # Diameter of all links in m
                                                                    output_times=output_times,
                                                                    nsteps=nsteps,
                                                                    dt=HYDRAULIC_STEP)

    # Alternatively, you can call .predict_conc if you are interest in the node-wise concentration over time
    conc_pred = surrogate.predict_conc(initial_state=initial_state,
                                       control_indices=boundary_indices,
                                       control_inputs=boundary_values,
                                       flow_field=flow_field,
                                       edge_lengths=edge_lengths,
                                       edge_diameters=edge_diameters,
                                       output_times=output_times,
                                       nsteps=nsteps,
                                       dt=HYDRAULIC_STEP)
    print(conc_pred.shape)  # 1. dimension: time; 2. dimension: nodes

    # Evaluate results
    print(f"MAE: {np.mean(np.abs(conc_pred - conc))}")

    # Plot ground truth concentration vs. prediction from the surrogate model at six random nodes
    for idx in np.random.RandomState(32).choice(range(scada_data.network_topo.get_number_of_nodes()), size=6):
        plot_timeseries_prediction(conc[:, idx], conc_pred[:, idx],
                                x_axis_label="Time steps", y_axis_label="CL2 concentration")
