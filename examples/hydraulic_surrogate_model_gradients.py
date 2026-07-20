"""
This example demonstrates how to compute gradients of different quantities
in the context of a hydraulic surrogat model.
"""
import numpy as np
import torch
from epyt_control.models import PIGNNModel, device
from epyt_flow.simulation import ScenarioSimulator, EpanetConstants
from epyt_flow.data.networks import load_anytown, to_seconds


def create_test_data_anytown():
    with ScenarioSimulator(scenario_config=load_anytown()) as sim:
        # Place pressure, flow, and demand sensors everywhere
        sim.place_demand_sensors_everywhere()
        sim.place_pressure_sensors_everywhere()
        sim.place_flow_sensors_everywhere()

        # Extract the location of the reservoirs -- recall that we need the heads at
        # all reservoirs as inputs for the hydraulic surrogate model
        reservoir_idx = [idx - 1 for idx in sim.epanet_api.get_all_reservoirs_idx()]  # NOTE: EPANET starts counting at 1!

        # Make sure the right units are used -- i.e., m^3/s for flows and m for pressure
        sim.set_general_parameters(flow_units_id=EpanetConstants.EN_CMS,
                                   pressure_units_id=EpanetConstants.EN_METERS,
                                   simulation_duration=to_seconds(days=7))  # Generate data for 7 days

        # Run the simulation
        scada_data = sim.run_simulation()

        # Extract simulation results 
        topo = scada_data.network_topo
        demands = scada_data.get_data_demands()
        pressures = scada_data.get_data_pressures()

        # Heads(m) = pressure(m) + elevation(m)
        elevs_np = np.array([float(topo.get_node_info(node_id).get("elevation", 0.0))
                             for node_id in topo.get_all_nodes()], dtype=np.float32)
        elevs_tile = elevs_np[None, :].repeat(pressures.shape[0], axis=0)

        heads = (pressures[:, :len(topo.get_all_nodes())] + elevs_tile[:, :len(topo.get_all_nodes())])

        return heads, reservoir_idx, demands


if __name__ == "__main__":
    # Create and load a pre-trained hydrauic surrogate model for Anytown
    model = PIGNNModel.from_network("anytown", load_pretrained_model=True)
    #model.load_model("anytown_pignn.pt")  # Alternatively, you can load your own trained weights

    # Create test data
    heads, reservoir_idx, demands = create_test_data_anytown()  # First axis in 'heads' and 'demands' encode time!

    # Compute gradients for the first time step only!
    r = model.compute_gradients_from_forward_pass(reservoir_heads=heads[0, reservoir_idx].reshape(1, -1),
                                                  demands=demands[0, :].reshape(1, -1))

    # Build a custom loss function and compute gradients w.r.t. the demands through this loss function
    # to the inputs of the surrogate (e.g., demands or other parameters)
    hyd_pred = model.predict(reservoir_heads=heads[0, reservoir_idx].reshape(1, -1),
                             demands=demands[0, :].reshape(1, -1)) 

    heads_true = np.random.uniform(0, 1, size=hyd_pred["heads_pred"][0, :].shape).astype(np.float32)  # Generate random ground truth
    heads_true = torch.tensor(heads_true, dtype=torch.float32).to(device)
    loss = lambda hpred: (hpred - heads_true).sum().view(-1, 1)  # Compare predicted pressure heads to some imaginery ground truth (here, random heads)

    g = model.compute_gradients(reservoir_heads=heads[0, reservoir_idx].reshape(1, -1),
                                demands=demands[0, :].reshape(1, -1),
                                gradient_output="heads",   # We are targeting the predicted pressure heads, not the flow rates!
                                gradient_input="demands",  # We compute gradients w.r.t. the demands
                                output_func=loss)
    print(g["grads"].shape)  # Get gradients' shape -- note that they are returned as a torch.Tensor
