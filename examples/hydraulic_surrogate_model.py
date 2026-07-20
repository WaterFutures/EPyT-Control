"""
This example demonstrates how to load a pre-trained hydraulic surrogate model
and use it for predicting flows and pressure heads.
"""
import numpy as np
from epyt_control.models import PIGNNModel
from epyt_flow.simulation import ScenarioSimulator, EpanetConstants
from epyt_flow.data.networks import load_anytown
from epyt_flow.utils import plot_timeseries_data, to_seconds


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
        flows = scada_data.get_data_flows()

        # Heads(m) = pressure(m) + elevation(m)
        elevs_np = np.array([float(topo.get_node_info(node_id).get("elevation", 0.0))
                             for node_id in topo.get_all_nodes()], dtype=np.float32)
        elevs_tile = elevs_np[None, :].repeat(pressures.shape[0], axis=0)

        heads = (pressures[:, :len(topo.get_all_nodes())] + elevs_tile[:, :len(topo.get_all_nodes())])

        return heads, reservoir_idx, demands, flows


if __name__ == "__main__":
    # Create and load a pre-trained hydrauic surrogate model for Anytown
    model = PIGNNModel.from_network("anytown")#, load_pretrained_model=True)
    model.load_model("anytown_pignn.pt")  # Alternatively, you can load your own pre-trained weights

    # Create test data
    heads, reservoir_idx, demands, flows = create_test_data_anytown()  # First axis in 'heads', 'demands', and 'flows' encode time!

    # Use the hydraulic surrogate model to predict flow rates and heads
    hyd_pred = model.predict(reservoir_heads=heads[:, reservoir_idx],
                             demands=demands)
    heads_pred = hyd_pred["heads_pred"].detach().cpu().numpy().squeeze()  # Predicted pressure heads at every node
    flows_pred = hyd_pred["flows_pred"].detach().cpu().numpy().squeeze()  # Predicted flow rates at every link

    # Compare predictions to ground truth
    plot_timeseries_data(np.abs(heads_pred - heads),
                         x_axis_label="Time",
                         y_axis_label="Absolute pressure head error")
    plot_timeseries_data(np.abs(flows_pred - flows),  # TODO: two links per pipe - how to reduce to one pipe?
                         x_axis_label="Time",
                         y_axis_label="Absolute flow rate error")
