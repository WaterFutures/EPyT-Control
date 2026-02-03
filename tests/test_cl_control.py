"""
This file contains test for (EPANET) chlorine injection control environments.
"""
import os
import time
import numpy as np
from epyt_flow.data.benchmarks import load_leakdb_scenarios
from epyt_flow.simulation import ScenarioSimulator, EpanetConstants, ModelUncertainty, \
    ScenarioConfig, ScadaData, SensorConfig
from epyt_flow.utils import to_seconds
from epyt_flow.uncertainty import RelativeUniformUncertainty, AbsoluteGaussianUncertainty
from epyt_control.envs import HydraulicControlEnv, MultiConfigHydraulicControlEnv
from epyt_control.envs.actions import ChemicalInjectionAction
from gymnasium.utils.env_checker import check_env

from .utils import get_temp_folder


def create_scenario() -> ScenarioConfig:
    # Create scenario based on the LeakDB Hanoi
    [scenario_config] = load_leakdb_scenarios(scenarios_id=list(range(1)), use_net1=False,
                                              download_dir=get_temp_folder())
    with ScenarioSimulator(scenario_config=scenario_config) as sim:
        sim.set_general_parameters(simulation_duration=to_seconds(days=2))

        # Enable chlorine simulation and place a chlorine injection pump at the reservoir
        sim.enable_chemical_analysis()

        reservoid_node_id, = sim.epanet_api.get_all_reservoirs_id()
        sim.add_quality_source(node_id=reservoid_node_id,
                               pattern=np.array([1.]),
                               source_type=EpanetConstants.EN_MASS,
                               pattern_id="my-chl-injection")

        # Set initial concentration and simple (constant) reactions
        for node_idx in sim.epanet_api.get_all_nodes_idx():
            sim.epanet_api.set_node_init_quality(node_idx, 0)
        for link_idx in sim.epanet_api.get_all_links_idx():
            sim.epanet_api.setlinkvalue(link_idx, EpanetConstants.EN_KBULK, -.5)
            sim.epanet_api.setlinkvalue(link_idx, EpanetConstants.EN_KWALL, -.01)

        # Set flow and chlorine sensors everywhere
        sim.sensor_config = SensorConfig.create_empty_sensor_config(sim.sensor_config)
        sim.set_flow_sensors(sim.sensor_config.links)

        # Specify uncertainties -- similar to the one already implemented in LeakDB
        my_uncertainties = {"global_pipe_length_uncertainty": RelativeUniformUncertainty(low=0, high=0.25),
                            "global_pipe_roughness_uncertainty": RelativeUniformUncertainty(low=0, high=0.25),
                            "global_base_demand_uncertainty": RelativeUniformUncertainty(low=0, high=0.25),
                            "global_demand_pattern_uncertainty": AbsoluteGaussianUncertainty(mean=0, scale=.2)}
        sim.set_model_uncertainty(ModelUncertainty(**my_uncertainties))

        sim.save_to_epanet_file(os.path.join(get_temp_folder(), f"SimpleChlorineInjectionEnv-{time.time()}.inp"))
        return sim.get_scenario_config()


class SimpleChlorineInjectionEnv(HydraulicControlEnv):
    """
    A simple environment for controlling the chlorine injection.
    """
    def __init__(self, autoreset: bool, reload_scenario_when_reset: bool):
        scenario_config = create_scenario()

        chlorine_injection_action_space = ChemicalInjectionAction(node_id="1",
                                                                  pattern_id="my-chl-injection",
                                                                  source_type_id=EpanetConstants.EN_MASS,
                                                                  upper_bound=10000.)

        super().__init__(scenario_config=scenario_config,
                         chemical_injection_actions=[chlorine_injection_action_space],
                         autoreset=autoreset,
                         reload_scenario_when_reset=reload_scenario_when_reset)

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
        new_sensor_config = scada_data.sensor_config
        new_sensor_config.quality_node_sensors = scada_data.sensor_config.nodes
        scada_data.change_sensor_config(new_sensor_config)

        # Regulation Limits
        upper_cl_bound = 2.  # (mg/l)
        lower_cl_bound = .3  # (mg/l)

        # Sum up (negative) residuals for out of bounds Cl concentrations at nodes -- i.e.
        # reward of zero means everythings is okay, while a negative reward
        # denotes Cl concentration bound violations
        reward = 0

        nodes_quality = scada_data.get_data_nodes_quality()

        upper_bound_violation_idx = nodes_quality > upper_cl_bound
        reward += -1. * np.sum(nodes_quality[upper_bound_violation_idx] - upper_cl_bound)

        lower_bound_violation_idx = nodes_quality < lower_cl_bound
        reward += np.sum(nodes_quality[lower_bound_violation_idx] - lower_cl_bound)

        return float(reward)


class SimpleChlorineInjectionMultiConfigEnv(MultiConfigHydraulicControlEnv):
    """
    A simple environment for controlling the chlorine injection --
    can handle multiple scenario configs.
    """
    def __init__(self, reload_scenario_when_reset: bool):
        scenario_config = create_scenario()

        chlorine_injection_action_space = ChemicalInjectionAction(node_id="1",
                                                                  pattern_id="my-chl-injection",
                                                                  source_type_id=EpanetConstants.EN_MASS,
                                                                  upper_bound=10000.)

        super().__init__(scenario_configs=[scenario_config],
                         chemical_injection_actions=[chlorine_injection_action_space],
                         reload_scenario_when_reset=reload_scenario_when_reset)

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
        new_sensor_config = scada_data.sensor_config
        new_sensor_config.quality_node_sensors = scada_data.sensor_config.nodes
        scada_data.change_sensor_config(new_sensor_config)

        # Regulation Limits
        upper_cl_bound = 2.  # (mg/l)
        lower_cl_bound = .3  # (mg/l)

        # Sum up (negative) residuals for out of bounds Cl concentrations at nodes -- i.e.
        # reward of zero means everythings is okay, while a negative reward
        # denotes Cl concentration bound violations
        reward = 0

        nodes_quality = scada_data.get_data_nodes_quality()

        upper_bound_violation_idx = nodes_quality > upper_cl_bound
        reward += -1. * np.sum(nodes_quality[upper_bound_violation_idx] - upper_cl_bound)

        lower_bound_violation_idx = nodes_quality < lower_cl_bound
        reward += np.sum(nodes_quality[lower_bound_violation_idx] - lower_cl_bound)

        return float(reward)


def test_cl_env():
    with SimpleChlorineInjectionEnv(autoreset=True, reload_scenario_when_reset=False) as env:
        check_env(env)


def test_cl_multiconfig_env():
    with SimpleChlorineInjectionMultiConfigEnv(reload_scenario_when_reset=False) as env:
        check_env(env)


def test_cl_env_1():
    with SimpleChlorineInjectionEnv(autoreset=True, reload_scenario_when_reset=False) as env:
        check_env(env)

        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)

        for _ in range(20):
            act = env.action_space.sample()
            obs, reward, terminated, _, _ = env.step(act)
            assert isinstance(obs, np.ndarray)
            assert isinstance(reward, float)
            assert terminated is False


def test_cl_env_2():
    with SimpleChlorineInjectionEnv(autoreset=False, reload_scenario_when_reset=False) as env:
        check_env(env)

        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)

        terminated_at_some_point = False
        for _ in range(100):
            act = env.action_space.sample()
            obs, reward, terminated, _, _ = env.step(act)

            if terminated is True:
                terminated_at_some_point = True
                break

            assert isinstance(obs, np.ndarray)
            assert isinstance(reward, float)

        assert terminated_at_some_point is True


def test_cl_env_reload():
    with SimpleChlorineInjectionEnv(autoreset=True, reload_scenario_when_reset=True) as env:
        check_env(env)

        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)

        for _ in range(20):
            act = env.action_space.sample()
            obs, reward, terminated, _, _ = env.step(act)
            assert isinstance(obs, np.ndarray)
            assert isinstance(reward, float)
            assert terminated is False


def test_cl_env_multiconfig():
    with SimpleChlorineInjectionMultiConfigEnv(reload_scenario_when_reset=False) as env:
        check_env(env)

        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)

        for _ in range(20):
            act = env.action_space.sample()
            obs, reward, terminated, _, _ = env.step(act)
            assert isinstance(obs, np.ndarray)
            assert isinstance(reward, float)
            assert terminated is False
