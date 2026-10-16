import numpy as np
from epyt_flow.simulation import ScenarioSimulator, EpanetConstants
from epyt_flow.data.networks import load_anytown
from epyt_flow.utils import to_seconds


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
