"""
This file contains test for EPANET-MSX control environments.
"""
import os
from pathlib import Path
import random
import numpy as np
from epyt_flow.data.benchmarks import load_leakdb_scenarios
from epyt_flow.simulation import ScenarioSimulator, EpanetConstants, ModelUncertainty, \
    ScenarioConfig, ScadaData, SensorConfig
from epyt_flow.simulation.events import SpeciesInjectionEvent
from epyt_flow.uncertainty import RelativeUniformUncertainty, AbsoluteGaussianUncertainty
from epyt_flow.utils import to_seconds
from epyt_control.envs import AdvancedQualityControlEnv, MultiConfigAdvancedQualityControlEnv
from epyt_control.envs.actions import SpeciesInjectionAction
from gymnasium.utils.env_checker import check_env

from .utils import get_temp_folder


def create_scenario(scenario_id: int):
    # Extract .inp file
    scenario_config, = load_leakdb_scenarios(scenarios_id=[scenario_id], use_net1=False,
                                             download_dir=get_temp_folder())
    f_inp_out = os.path.join(get_temp_folder(), f"cl2_msx-injection_scenario-{scenario_id}.inp")

    with ScenarioSimulator(scenario_config=scenario_config) as scenario:
        Path(get_temp_folder()).mkdir(exist_ok=True)
        scenario.save_to_epanet_file(f_inp_out)

    # Create complete scenario
    with ScenarioSimulator(f_inp_in=f_inp_out, f_msx_in="as3_cl2.msx") as sim:
        # Set simulation duration to 21 days -- see EPANET-MSX bug
        sim.set_general_parameters(simulation_duration=to_seconds(days=21))

        # Place a chlorine injection pump at the reservoirs and tanks
        for node_id in sim.epanet_api.get_all_reservoirs_id() + sim.epanet_api.get_all_tanks_id():
            print(node_id)
            sim.add_species_injection_source(species_id="CL2", node_id=node_id,
                                             pattern=np.array([1000]),
                                             source_type=EpanetConstants.EN_MASS,
                                             pattern_id=f"cl2-injection-at-node_{node_id}")

        # Set flow and chlorine sensors everywhere
        sim.sensor_config = SensorConfig.create_empty_sensor_config(sim.sensor_config)
        sim.set_flow_sensors(sim.sensor_config.links)

        # Export .inp and .msx files
        Path(get_temp_folder()).mkdir(exist_ok=True)
        sim.save_to_epanet_file(inp_file_path=f_inp_out,
                                msx_file_path=os.path.join(get_temp_folder(),
                                                           f"cl2_msx-injection_scenario-{scenario_id}.msx"))

        # Random aresenic contaminations
        n_contamination_events = random.randint(0, 5)
        for _ in range(n_contamination_events):
            contamination_node_id, = random.sample(sim.sensor_config.nodes, k=1)
            contamination_strength = random.randint(1000, 5000)
            start_day = random.randint(1, 20)
            end_day = start_day + random.randint(1, 21-start_day)

            sim.add_system_event(SpeciesInjectionEvent(species_id="AsIII",
                                                       node_id=contamination_node_id,
                                                       profile=np.array([contamination_strength]),
                                                       source_type=EpanetConstants.EN_MASS,
                                                       start_time=to_seconds(days=start_day),
                                                       end_time=to_seconds(days=end_day)))

        # Specify uncertainties -- similar to the one already implemented in LeakDB
        my_uncertainties = {"global_pipe_length_uncertainty": RelativeUniformUncertainty(low=0, high=0.25),
                            "global_pipe_roughness_uncertainty": RelativeUniformUncertainty(low=0, high=0.25),
                            "global_base_demand_uncertainty": RelativeUniformUncertainty(low=0, high=0.25),
                            "global_demand_pattern_uncertainty": AbsoluteGaussianUncertainty(mean=0, scale=.2)}
        sim.set_model_uncertainty(ModelUncertainty(**my_uncertainties))

        # Export scenario
        sim.get_scenario_config().save_to_file(os.path.join(get_temp_folder(),
                                                            f"cl2_msx-injection_scenario-{scenario_id}"))


class MyAdvancedQualityControlEnv(AdvancedQualityControlEnv):
    """
    A simple environment for controlling the chlorine injection in an EPANET-MSX scenario.
    """
    def __init__(self, scenario_id: int):
        # Create scenario
        create_scenario(scenario_id)

        # Load scenario and set autoreset=True
        scenario_config_file_in = os.path.join(get_temp_folder(),
                                               f"cl2_msx-injection_scenario-{scenario_id}.epytflow_scenario_config")
        super().__init__(scenario_config=ScenarioConfig.load_from_file(scenario_config_file_in),
                         action_space=[SpeciesInjectionAction(species_id="CL2", node_id="1",
                                                              pattern_id="cl2-injection-at-node_1",
                                                              source_type_id=EpanetConstants.EN_MASS,
                                                              upper_bound=10.)],
                         rerun_hydraulics_when_reset=False)

        self.__sensor_config_reward = None

    def step(self, action: np.ndarray):
        # Scaling of Cl injection
        return super().step(action * 1000)

    def _compute_reward_function(self, scada_data: ScadaData) -> float:
        """
        Computes the current reward based on the current sensors readings (i.e. SCADA data).

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

        if self.__sensor_config_reward is None:
            self.__sensor_config_reward = SensorConfig.create_empty_sensor_config(scada_data.sensor_config)   # TODO: Move to constructor
            self.__sensor_config_reward.bulk_species_node_sensors = {"CL2": scada_data.sensor_config.nodes}
        scada_data.change_sensor_config(self.__sensor_config_reward)

        nodes_quality = scada_data.get_data_bulk_species_node_concentration({"CL2": scada_data.sensor_config.nodes})

        upper_bound_violation_idx = nodes_quality > upper_cl_bound
        reward += -1. * np.sum(nodes_quality[upper_bound_violation_idx] - upper_cl_bound)

        lower_bound_violation_idx = nodes_quality < lower_cl_bound
        reward += np.sum(nodes_quality[lower_bound_violation_idx] - lower_cl_bound)

        return float(reward)


class MyMultiConfigAdvancedQualityControlEnv(MultiConfigAdvancedQualityControlEnv):
    """
    A simple environment for controlling the chlorine injection in a list of EPANET-MSX scenarios.
    """
    def __init__(self, scenario_id: int):
        # Create scenario
        create_scenario(scenario_id)

        # Load scenario and set autoreset=True
        scenario_config_file_in = os.path.join(get_temp_folder(),
                                               f"cl2_msx-injection_scenario-{scenario_id}.epytflow_scenario_config")
        super().__init__(scenario_configs=[ScenarioConfig.load_from_file(scenario_config_file_in)],
                         action_space=[SpeciesInjectionAction(species_id="CL2", node_id="1",
                                                              pattern_id="cl2-injection-at-node_1",
                                                              source_type_id=EpanetConstants.EN_MASS,
                                                              upper_bound=10.)],
                         rerun_hydraulics_when_reset=False)

        self.__sensor_config_reward = None

    def step(self, action: np.ndarray):
        # Scaling of Cl injection
        return super().step(action * 1000)

    def _compute_reward_function(self, scada_data: ScadaData) -> float:
        """
        Computes the current reward based on the current sensors readings (i.e. SCADA data).

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

        if self.__sensor_config_reward is None:
            self.__sensor_config_reward = SensorConfig.create_empty_sensor_config(scada_data.sensor_config)   # TODO: Move to constructor
            self.__sensor_config_reward.bulk_species_node_sensors = {"CL2": scada_data.sensor_config.nodes}
        scada_data.change_sensor_config(self.__sensor_config_reward)

        nodes_quality = scada_data.get_data_bulk_species_node_concentration({"CL2": scada_data.sensor_config.nodes})

        upper_bound_violation_idx = nodes_quality > upper_cl_bound
        reward += -1. * np.sum(nodes_quality[upper_bound_violation_idx] - upper_cl_bound)

        lower_bound_violation_idx = nodes_quality < lower_cl_bound
        reward += np.sum(nodes_quality[lower_bound_violation_idx] - lower_cl_bound)

        return float(reward)


def test_msx_env():
    with MyAdvancedQualityControlEnv(scenario_id=1) as env:
        check_env(env)


def test_msx_multiconfig_env():
    with MyMultiConfigAdvancedQualityControlEnv(scenario_id=1) as env:
        check_env(env)


def test_msx():
    with MyAdvancedQualityControlEnv(scenario_id=1) as env:
        check_env(env)

        obs, info = env.reset()

        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)

        for _ in range(100):
            obs, reward, terminated, _, _ = env.step(env.action_space.sample())

            assert isinstance(obs, np.ndarray)
            assert isinstance(reward, float)
            assert terminated is False


def test_multiconfig_msx():
    with MyMultiConfigAdvancedQualityControlEnv(scenario_id=1) as env:
        check_env(env)

        obs, info = env.reset()

        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)

        for _ in range(100):
            obs, reward, terminated, _, _ = env.step(env.action_space.sample())

            assert isinstance(obs, np.ndarray)
            assert isinstance(reward, float)
            assert terminated is False
