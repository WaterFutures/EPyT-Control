"""
This file contains test for pump speed control environments.
"""
import os
import numpy as np
import pandas as pd
from epyt_flow.simulation import ScenarioSimulator, ScenarioConfig, ScadaData
from epyt_flow.utils import to_seconds
from epyt_control.envs import HydraulicControlEnv, MultiConfigHydraulicControlEnv
from epyt_control.envs.actions import PumpSpeedAction
from gymnasium.utils.env_checker import check_env
from gymnasium.spaces import Box, Tuple
from gymnasium.spaces.utils import flatten_space


def create_scenario(f_inp_in: str) -> tuple[ScenarioConfig, list[str]]:
    """
    Creates a new scenario for a given .inp file.
    Note that pressure sensors are placed at every junction.
    """
    with ScenarioSimulator(f_inp_in=f_inp_in) as scenario:
        scenario.set_general_parameters(simulation_duration=to_seconds(hours=12))

        # Sensors = input to the agent (control strategy)
        # Place pressure sensors at all junctions
        junctions = scenario.sensor_config.nodes
        for tank_id in scenario.sensor_config.tanks:
            junctions.remove(tank_id)
        scenario.set_pressure_sensors(sensor_locations=junctions)

        # Place pump efficiency sensors at every pump
        scenario.place_pump_efficiency_sensors_everywhere()

        # Place flow sensors at every pump and tank connection
        topo = scenario.get_topology()
        tank_connections = []
        for tank in topo.get_all_tanks():
            for link, _ in topo.get_adjacent_links(tank):
                tank_connections.append(link)

        flow_sensors = tank_connections + scenario.sensor_config.pumps
        scenario.set_flow_sensors(flow_sensors)

        # Return the scenario config and tank connections
        return scenario.get_scenario_config(), tank_connections


class ContinuousPumpControlEnv(HydraulicControlEnv):
    """
    Class implementing a continous pump speed environment --
    i.e. a continous action space for the pump speed.
    """
    def __init__(self, autoreset: bool, reload_scenario_when_reset: bool):
        f_inp_in = os.path.join("examples", "Anytown.inp")
        scenario_config, tank_connections = create_scenario(f_inp_in)

        self._tank_connections = tank_connections
        self._network_constraints = {"min_pressure": 28.1227832,
                                     "max_pressure": 70,
                                     "max_pump_efficiencies": pd.Series({"b1": .65,
                                                                         "b2": .65,
                                                                         "b3": .65})}
        self._objective_weights = {"pressure_violation": .9,
                                   "abs_tank_flow": .02,
                                   "pump_efficiency": .08}

        super().__init__(scenario_config=scenario_config,
                         pumps_speed_actions=[PumpSpeedAction(pump_id=p_id,
                                                              speed_upper_bound=4.0)
                                              for p_id in scenario_config.sensor_config.pumps],
                         autoreset=autoreset,
                         reload_scenario_when_reset=reload_scenario_when_reset)

    def _compute_reward_function(self, scada_data: ScadaData) -> float:
        # Compute different objectives and final reward
        pressure_data = scada_data.get_data_pressures()
        tanks_flow_data = scada_data.get_data_flows(sensor_locations=self._tank_connections)
        pumps_flow_data = scada_data.get_data_flows(sensor_locations=scada_data.sensor_config.pumps)
        pump_efficiency = scada_data.get_data_pumps_efficiency()

        pressure_violations = np.logical_or(
            pressure_data > self._network_constraints["max_pressure"],
            pressure_data < self._network_constraints["min_pressure"]
        ).any(axis=0).sum()
        n_sensors = pressure_data.shape[1]
        pressure_obj = float(1 - pressure_violations / n_sensors)

        total_abs_tank_flow = np.abs(tanks_flow_data).sum(axis=None)
        total_pump_flow = pumps_flow_data.sum(axis=None)
        tank_obj = float(total_pump_flow / (total_pump_flow + total_abs_tank_flow))

        pump_efficiencies = pd.Series(
            pump_efficiency.mean(axis=0),
            index=scada_data.sensor_config.pumps
        )
        max_pump_efficiencies = self._network_constraints["max_pump_efficiencies"]
        normalized_pump_efficiencies = pump_efficiencies / max_pump_efficiencies
        pump_efficiency_obj = normalized_pump_efficiencies.mean()

        reward = self._objective_weights["pressure_violation"] * pressure_obj + \
            self._objective_weights["abs_tank_flow"] * tank_obj + \
            self._objective_weights["pump_efficiency"] * pump_efficiency_obj

        return reward


class ContinuousPumpControlMultiConfigEnv(MultiConfigHydraulicControlEnv):
    """
    Class implementing a continous pump speed environment that can handle
    multiple scenario configs -- i.e. a continous action space for the pump speed.
    """
    def __init__(self, reload_scenario_when_reset: bool):
        f_inp_in = os.path.join("examples", "Anytown.inp")
        scenario_config, tank_connections = create_scenario(f_inp_in)

        self._tank_connections = tank_connections
        self._network_constraints = {"min_pressure": 28.1227832,
                                     "max_pressure": 70,
                                     "max_pump_efficiencies": pd.Series({"b1": .65,
                                                                         "b2": .65,
                                                                         "b3": .65})}
        self._objective_weights = {"pressure_violation": .9,
                                   "abs_tank_flow": .02,
                                   "pump_efficiency": .08}

        super().__init__(scenario_configs=[scenario_config],
                         pumps_speed_actions=[PumpSpeedAction(pump_id=p_id,
                                                              speed_upper_bound=4.0)
                                              for p_id in scenario_config.sensor_config.pumps],
                         reload_scenario_when_reset=reload_scenario_when_reset)

    def _compute_reward_function(self, scada_data: ScadaData) -> float:
        # Compute different objectives and final reward
        pressure_data = scada_data.get_data_pressures()
        tanks_flow_data = scada_data.get_data_flows(sensor_locations=self._tank_connections)
        pumps_flow_data = scada_data.get_data_flows(sensor_locations=scada_data.sensor_config.pumps)
        pump_efficiency = scada_data.get_data_pumps_efficiency()

        pressure_violations = np.logical_or(
            pressure_data > self._network_constraints["max_pressure"],
            pressure_data < self._network_constraints["min_pressure"]
        ).any(axis=0).sum()
        n_sensors = pressure_data.shape[1]
        pressure_obj = float(1 - pressure_violations / n_sensors)

        total_abs_tank_flow = np.abs(tanks_flow_data).sum(axis=None)
        total_pump_flow = pumps_flow_data.sum(axis=None)
        tank_obj = float(total_pump_flow / (total_pump_flow + total_abs_tank_flow))

        pump_efficiencies = pd.Series(
            pump_efficiency.mean(axis=0),
            index=scada_data.sensor_config.pumps
        )
        max_pump_efficiencies = self._network_constraints["max_pump_efficiencies"]
        normalized_pump_efficiencies = pump_efficiencies / max_pump_efficiencies
        pump_efficiency_obj = normalized_pump_efficiencies.mean()

        reward = self._objective_weights["pressure_violation"] * pressure_obj + \
            self._objective_weights["abs_tank_flow"] * tank_obj + \
            self._objective_weights["pump_efficiency"] * pump_efficiency_obj

        return reward


def test_env_api():
    with ContinuousPumpControlEnv(autoreset=False,
                                  reload_scenario_when_reset=False) as env:
        check_env(env)


def test_multiconig_env_api():
    with ContinuousPumpControlMultiConfigEnv(reload_scenario_when_reset=False) as env:
        check_env(env)


def test_autoreset1():
    with ContinuousPumpControlEnv(autoreset=False,
                                  reload_scenario_when_reset=False) as env:
        check_env(env)

        # Run some iterations -- note that autorest=True
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)

        terminated_at_some_point = False
        for _ in range(500):
            act = env.action_space.sample()
            obs, reward, terminated, _, _ = env.step(act)

            if terminated is True:
                terminated_at_some_point = True
                break

            assert isinstance(obs, np.ndarray)
            assert isinstance(reward, float)

        assert terminated_at_some_point is True


def test_autoreset2():
    with ContinuousPumpControlEnv(autoreset=True,
                                  reload_scenario_when_reset=False) as env:
        check_env(env)

        # Run some iterations -- note that autorest=True
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)

        for _ in range(100):
            act = env.action_space.sample()
            obs, reward, terminated, _, _ = env.step(act)

            assert isinstance(obs, np.ndarray)
            assert isinstance(reward, float)
            assert terminated is False


def test_reload():
    with ContinuousPumpControlEnv(autoreset=True,
                                  reload_scenario_when_reset=True) as env:
        check_env(env)

        # Run some iterations -- note that autorest=True
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)

        for _ in range(100):
            act = env.action_space.sample()
            obs, reward, terminated, _, _ = env.step(act)

            assert isinstance(obs, np.ndarray)
            assert isinstance(reward, float)
            assert terminated is False


def test_sensor_ordering():
    with ContinuousPumpControlEnv(
        autoreset=False, reload_scenario_when_reset=True
    ) as env:
        sensor_config = env._scenario_config.sensor_config
        pressure_obs_space = [Box(low=0, high=float("inf"))] * len(
            sensor_config.pressure_sensors
        )
        flow_obs_space = [Box(low=float("-inf"), high=float("inf"))] * len(
            sensor_config.flow_sensors
        )
        pump_efficiency_obs_space = [Box(low=0, high=float("inf"))] * len(
            sensor_config.pump_efficiency_sensors
        )
        obs_space_list = pressure_obs_space + flow_obs_space + pump_efficiency_obs_space
        correctly_ordered_obs_space = flatten_space(Tuple(obs_space_list))
        assert env.observation_space==correctly_ordered_obs_space

def test_multiconfig():
    with ContinuousPumpControlMultiConfigEnv(reload_scenario_when_reset=True) as env:
        check_env(env)

        # Run some iterations -- note that autorest=True
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)

        for _ in range(100):
            act = env.action_space.sample()
            obs, reward, terminated, _, _ = env.step(act)

            assert isinstance(obs, np.ndarray)
            assert isinstance(reward, float)
            assert terminated is False
