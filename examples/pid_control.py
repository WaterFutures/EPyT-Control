"""
Example of using a PID controller for controlling the chlorine injection in a simple scenario.
"""
import numpy as np
from epyt_flow.data.benchmarks import load_leakdb_scenarios
from epyt_flow.simulation import ScenarioSimulator, EpanetConstants, ModelUncertainty, \
    ScenarioConfig, ScadaData, SensorConfig
from epyt_flow.uncertainty import RelativeUniformUncertainty, AbsoluteGaussianUncertainty
from epyt_flow.utils import to_seconds, plot_timeseries_data

from epyt_control.envs import HydraulicControlEnv
from epyt_control.envs.actions import ChemicalInjectionAction
from epyt_control.controllers import PidController


def create_scenario():
    # Create a scenario based the LeakDB Hanoi
    [scenario_config] = load_leakdb_scenarios(scenarios_id=list(range(1)), use_net1=False)
    with ScenarioSimulator(scenario_config=scenario_config) as sim:
        # Set simulation duration to 20 days
        sim.set_general_parameters(simulation_duration=to_seconds(days=20))

        # Enable chlorine simulation and place a chlorine injection pump at the reservoir
        sim.enable_chemical_analysis()

        reservoid_node_id, = sim.epanet_api.getNodeReservoirNameID()
        sim.add_quality_source(node_id=reservoid_node_id,
                               pattern=np.array([1.]),
                               source_type=EpanetConstants.EN_MASS,
                               pattern_id="my-chl-injection")

        # Set initial concentration and simple (constant) reactions
        zeroNodes = [0] * sim.epanet_api.getNodeCount()
        sim.epanet_api.setNodeInitialQuality(zeroNodes)
        sim.epanet_api.setLinkBulkReactionCoeff([-.5] * sim.epanet_api.getLinkCount())
        sim.epanet_api.setLinkWallReactionCoeff([-.01] * sim.epanet_api.getLinkCount())

        # Set flow and chlorine sensors everywhere
        sim.sensor_config = SensorConfig.create_empty_sensor_config(sim.sensor_config)
        sim.set_flow_sensors(sim.sensor_config.links)
        sim.set_node_quality_sensors(sim.sensor_config.nodes)

        # Specify uncertainties -- similar to the one already implemented in LeakDB
        my_uncertainties = {"global_pipe_length_uncertainty": RelativeUniformUncertainty(low=0, high=0.25),
                            "global_pipe_roughness_uncertainty": RelativeUniformUncertainty(low=0, high=0.25),
                            "global_base_demand_uncertainty": RelativeUniformUncertainty(low=0, high=0.25),
                            "global_demand_pattern_uncertainty": AbsoluteGaussianUncertainty(mean=0, scale=.2)}
        sim.set_model_uncertainty(ModelUncertainty(**my_uncertainties))

        # Export scenario
        sim.save_to_epanet_file("cl_injection_scenario.inp")
        sim.get_scenario_config().save_to_file("cl_injection_scenario")


class SimpleChlorineInjectionEnv(HydraulicControlEnv):
    """
    A simple environment for controlling the chlorine injection.
    """
    def __init__(self):
        # Load scenario and set autoreset=False
        scenario_config_file_in = "cl_injection_scenario.epytflow_scenario_config"

        super().__init__(scenario_config=ScenarioConfig.load_from_file(scenario_config_file_in),
                         chemical_injection_actions=[ChemicalInjectionAction(node_id="1",
                                                                             pattern_id="my-chl-injection",
                                                                             source_type_id=EpanetConstants.EN_MASS,
                                                                             upper_bound=15000.)],
                         autoreset=False,
                         reload_scenario_when_reset=False)

    def _compute_reward_function(self, scada_data: ScadaData) -> float:
        """
        Computes the current reward based on the current sensors readings (i.e. SCADA data).
        The reward is zero iff all chlorine bounds are satisfied and negative otherwise.

        Parameters
        ----------
        :class:`epyt_flow.simulation.ScadaData`
            Current sensor readings.

        Returns
        -------
        `float`
            Current reward.
        """
        # Sum up (negative) residuals for out of bounds Cl concentrations at nodes -- i.e.
        # reward of zero means everythings is okay, while a negative reward
        # denotes Cl concentration bound violations
        reward = 0.

        # Regulation Limits
        upper_cl_bound = 2.  # (mg/l)
        lower_cl_bound = .3  # (mg/l)

        new_sensor_config = SensorConfig.create_empty_sensor_config(scada_data.sensor_config)
        new_sensor_config.quality_node_sensors = scada_data.sensor_config.nodes
        old_sensor_config = scada_data.sensor_config
        scada_data.change_sensor_config(new_sensor_config)

        nodes_quality = scada_data.get_data_nodes_quality()

        upper_bound_violation_idx = nodes_quality > upper_cl_bound
        reward += -1. * np.sum(nodes_quality[upper_bound_violation_idx] - upper_cl_bound)

        lower_bound_violation_idx = nodes_quality < lower_cl_bound
        reward += np.sum(nodes_quality[lower_bound_violation_idx] - lower_cl_bound)

        scada_data.change_sensor_config(old_sensor_config)
        return reward


if __name__ == "__main__":
    # Create and load environment
    create_scenario()

    with SimpleChlorineInjectionEnv() as env:
        # Use a simple PID controller for controlling the chlorine (Cl) injection
        # Note that a reward of zero indicates that Cl bounds at all nodes are satisfied!
        # Also, note that a better performance couod be achieved by properly tuning
        # the gain coefficients.
        pid_control = PidController(proportional_gain=30., integral_gain=10.,
                                    derivative_gain=0.,
                                    target_value=0.,
                                    action_lower_bound=float(env.action_space.low),
                                    action_upper_bound=float(env.action_space.high))

        # Evaluate the PID controller -- note that autorest=False
        env.reset()
        reward = 0

        rewards = []
        actions = []
        while True:
            # Compute Cl injection action
            act = [pid_control.step(reward)]

            # Execute Cl injection and observe a reward
            _, reward, terminated, _, _ = env.step(act)
            if terminated is True:
                break

            # Show observed reward and chosen action
            rewards.append(reward)
            actions.append(*act)

        # Show reward and actions over time
        plot_timeseries_data(np.array(rewards).reshape(1, -1),
                             y_axis_label="Reward",
                             x_axis_label="Time steps (30min)")
        plot_timeseries_data(np.array(actions).reshape(1, -1),
                             y_axis_label="Cl injection $mg$",
                             x_axis_label="Time steps (30min)")
