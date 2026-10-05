"""
This example demonstrates how to use the water quality surrogate model for predicting chlorine (CL2) concentrations
at every node in the Hanoi network based on a setpoint source at the reservoir.
"""
import numpy as np
from epyt_control.models import QualitySurrogateModel
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
    sources_at = ['1']      # Chlorine source at the reservoir

    N_SECONDS = to_seconds(hours=12)    # 12hr scenario
    HYDRAULIC_STEP = 60   # Hydraulic time step = 60s (1min)
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

    # Build water quality surrogate model
    surrogate = QualitySurrogateModel(scada_data.network_topo)

    # Predict the CL2 concentration of our single species at all nodes, based on the given source and the hydraulic simulation (i.e., flow rates)
    conc_pred = surrogate.predict({n_id: injection_pattern for n_id in sources_at}, scada_data)
    print(conc_pred.shape)  # 1. dimension: time; 2. dimension: nodes

    # Evaluation
    print(f"MAE: {np.mean(np.abs(conc_pred - conc))}")

    # Plot ground truth concentration vs. prediction from the surrogate model at six random nodes
    for idx in np.random.RandomState(32).choice(range(scada_data.network_topo.get_number_of_nodes()), size=6):
        plot_timeseries_prediction(conc[:, idx], conc_pred[:, idx],
                                   x_axis_label="Time steps", y_axis_label="CL2 concentration")
