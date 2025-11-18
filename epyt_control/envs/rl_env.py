"""
This module contains a base class for reinforcement learning (RL) environments.
"""
from abc import abstractmethod
import os
import uuid
from copy import deepcopy
from typing import Optional, Any, Union
import numpy as np
from epyt_flow.simulation import ScadaData, ScenarioConfig, ScenarioSimulator
from epyt_flow.gym import ScenarioControlEnv
from epyt_flow.utils import get_temp_folder
from gymnasium import Env
from gymnasium.spaces import Space, Box, Discrete, Tuple
from gymnasium.spaces.utils import flatten_space

from .actions import Action


class RlEnv(ScenarioControlEnv, Env):
    """
    Base class for reinforcement learning environments.

    Parameters
    ----------
    scenario_config : `epyt_flow.simulation.ScenarioConfig <https://epyt-flow.readthedocs.io/en/stable/epyt_flow.simulation.html#epyt_flow.simulation.scenario_config.ScenarioConfig>`_
        Config of the scenario.
    gym_action_space : `gymnasium.spaces.Space <https://gymnasium.farama.org/api/spaces/#gymnasium.spaces.Space>`_
        Gymnasium action space.
    action_space : list[:class:`~epyt_control.actions.actions.Action`]
        List of all action spaces -- one space for each element that can be controlled by the agent.
    reload_scenario_when_reset : `bool`, optional
        If True, the scenario (incl. the .inp and .msx file) is reloaded from the hard disk.
        If False, only the simulation is reset.

        The default is True.
    hyd_file_in : `str`, optional
        Path to an EPANET .hyd file containing the simulated hydraulics.
        Can only be used in conjunction with 'hyd_scada_in' in the case of an EPANET-MSX scenario.
        If set, hydraulics will not be simulated but taken from the specified file.

        The default is None.
    hyd_scada_in : `epyt_flow.simulation.ScadaData <https://epyt-flow.readthedocs.io/en/stable/epyt_flow.simulation.scada.html#epyt_flow.simulation.scada.scada_data.ScadaData>`_, optional
        ScadaData instance containing the simulated hydraulics -- must match the hydraulics
        from 'hyd_file_in'. Can only be used in conjunction with 'hyd_file_in'.

        The default is None.
    frozen_sensor_config : `bool`, optional
        If True, only the sensor readings from the observation space will be stored when running the
        simulation -- note that this implies that the reward function can only use the observations.
        This can lead to a significant speed-up of the simulation.

        The default is False.
    """
    def __init__(self, scenario_config: ScenarioConfig, gym_action_space: Space,
                 action_space: list[Action], reload_scenario_when_reset: bool = True,
                 hyd_file_in: str = None, hyd_scada_in: ScadaData = None,
                 frozen_sensor_config: bool = False, **kwds):
        if not isinstance(gym_action_space, Space):
            raise TypeError("'gym_action_space' must be an instance of 'gymnasium.spaces.Space' " +
                            f"but not of '{type(gym_action_space)}'")
        if not isinstance(action_space, list):
            raise TypeError("'action_spaces' must be an instance of " +
                            "'list[epyt_control.actions.Action]' " +
                            f"but not of '{type(action_space)}'")
        if any(not isinstance(a_s, Action) for a_s in action_space):
            raise TypeError("Every item in 'action_spaces' must be an instance of " +
                            "'epyt_control.actions.Action'")
        if not isinstance(reload_scenario_when_reset, bool):
            raise TypeError("'reload_scenario_when_reset' must be an instance of 'bool' " +
                            f"but not of '{type(reload_scenario_when_reset)}'")
        if not isinstance(frozen_sensor_config, bool):
            raise TypeError("'frozen_sensor_config' must be an instance of 'bool' " +
                            f"but not of '{type(frozen_sensor_config)}'")

        if (hyd_file_in is not None and hyd_scada_in is None) or \
                (hyd_file_in is None and hyd_scada_in is not None):
            raise ValueError("")
        if hyd_file_in is not None:
            if not isinstance(hyd_file_in, str):
                raise TypeError("'hyd_file_in' must be an instance of 'str' " +
                                f"but not of '{type(hyd_file_in)}'")
        if hyd_scada_in is not None:
            if not isinstance(hyd_scada_in, ScadaData):
                raise TypeError("'hyd_scada_in' must be an instance of " +
                                "'epyt_flow.simulation.ScadaData' but not of " +
                                f"'{type(hyd_scada_in)}'")

        self._hyd_file_in = hyd_file_in
        self._hyd_scada_in = hyd_scada_in

        super().__init__(scenario_config=scenario_config, **kwds)

        self._observation_space = self._get_observation_space()
        self._action_space = action_space
        self._gym_action_space = gym_action_space
        self._reload_scenario_when_reset = reload_scenario_when_reset
        self._frozen_sensor_config = frozen_sensor_config

    def _get_observation_space(self) -> Space:
        obs_space = []
        sensor_config = self._scenario_config.sensor_config

        obs_space += [Box(low=0, high=float("inf"))] * len(sensor_config.pressure_sensors)
        obs_space += [Box(low=float("-inf"), high=float("inf"))] * len(sensor_config.flow_sensors)
        obs_space += [Box(low=0, high=float("inf"))] * len(sensor_config.demand_sensors)
        obs_space += [Box(low=0, high=float("inf"))] * len(sensor_config.quality_node_sensors)
        obs_space += [Box(low=0, high=float("inf"))] * len(sensor_config.quality_link_sensors)
        obs_space += [Discrete(2, start=2)] * len(sensor_config.valve_state_sensors)
        obs_space += [Discrete(2, start=2)] * len(sensor_config.pump_state_sensors)
        obs_space += [Box(low=0, high=float("inf"))] * len(sensor_config.pump_efficiency_sensors)
        obs_space += [Box(low=0, high=float("inf"))] * \
            len(sensor_config.pump_energyconsumption_sensors)
        obs_space += [Box(low=0, high=float("inf"))] * len(sensor_config.tank_volume_sensors)
        for species_id in sensor_config.surface_species_sensors:
            obs_space += [Box(low=0, high=float("inf"))] * \
                len(sensor_config.surface_species_sensors[species_id])
        for species_id in sensor_config.bulk_species_node_sensors:
            obs_space += [Box(low=0, high=float("inf"))] * \
                len(sensor_config.bulk_species_node_sensors[species_id])
        for species_id in sensor_config.bulk_species_link_sensors:
            obs_space += [Box(low=0, high=float("inf"))] * \
                len(sensor_config.bulk_species_link_sensors[species_id])

        return flatten_space(Tuple(obs_space))

    @property
    def observation_space(self) -> Space:
        """
        Returns the observation space of this environment.

        Returns
        -------
        `gymnasium.spaces.Space <https://gymnasium.farama.org/api/spaces/#gymnasium.spaces.Space>`_
            Gymnasium (observation) space instance.
        """
        return self._observation_space

    @property
    def action_space(self) -> Space:
        """
        Returns the action space of this environment.

        Returns
        -------
        `gymnasium.spaces.Space <https://gymnasium.farama.org/api/spaces/#gymnasium.spaces.Space>`_
            Gymnasium (action) space instance.
        """
        return self._gym_action_space

    def _next_sim_itr(self) -> Union[tuple[ScadaData, bool], ScadaData]:
        try:
            next(self._sim_generator)
            scada_data, terminated = self._sim_generator.send(False)

            if self._scenario_sim.f_msx_in is not None:
                cur_time = int(scada_data.sensor_readings_time[0])
                cur_hyd_scada_data = self._hydraulic_scada_data.\
                    extract_time_window(cur_time, cur_time)
                scada_data.join(cur_hyd_scada_data)

            if self.autoreset is True:
                return scada_data
            else:
                return scada_data, terminated
        except StopIteration:
            if self.autoreset is True:
                _, info = self.reset()
                return info["scada_data"]
            else:
                return None, True

    def reset(self, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
              ) -> tuple[np.ndarray, dict]:
        """
        Resets this environment to an initial internal state, returning an
        initial observation and info.

        Parameters
        ----------
        seed : `int`, optional
            The seed that is used to initialize the environment's PRNG.

            The default is None.
        options : `dict[str, Any]`, optional
            Additional information to specify how the environment is reset
            (optional, depending on the specific environment).

            The default is None.

        Returns
        -------
        tuple[`numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_, dict]
            Observation (`numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_),
            {"scada_data": `ScadaData <https://epyt-flow.readthedocs.io/en/stable/epyt_flow.simulation.scada.html#epyt_flow.simulation.scada.scada_data.ScadaData>`_}
            (`epyt_flow.simulation.ScadaData <https://epyt-flow.readthedocs.io/en/stable/epyt_flow.simulation.scada.html#epyt_flow.simulation.scada.scada_data.ScadaData>`_ as additional info).
        """
        Env.reset(self, seed=seed)

        if self._reload_scenario_when_reset is True:
            scada_data = super().reset()
        else:
            if self._scenario_sim is None:
                self._scenario_sim = ScenarioSimulator(scenario_config=self._scenario_config)
            else:
                # Abort current simulation if any is runing
                try:
                    next(self._sim_generator)
                    self._sim_generator.send(True)
                except StopIteration:
                    pass

            if self._scenario_sim.f_msx_in is not None:
                if self._hyd_file_in is not None:
                    hyd_export = self._hyd_file_in
                    self._hydraulic_scada_data = self._hyd_scada_in
                else:
                    hyd_export = os.path.join(get_temp_folder(),
                                              f"epytflow_env_MSX_{uuid.uuid4()}.hyd")
                    sim = self._scenario_sim.run_hydraulic_simulation
                    self._hydraulic_scada_data = sim(hyd_export=hyd_export,
                                                     frozen_sensor_config=self._frozen_sensor_config,
                                                     reapply_uncertainties=self.reapply_uncertainties_at_reset)

                gen = self._scenario_sim.run_advanced_quality_simulation_as_generator
                self._sim_generator = gen(hyd_export, support_abort=True,
                                          frozen_sensor_config=self._frozen_sensor_config,
                                          reapply_uncertainties=self.reapply_uncertainties_at_reset)
            else:
                gen = self._scenario_sim.run_hydraulic_simulation_as_generator
                self._sim_generator = gen(support_abort=True,
                                          frozen_sensor_config=self._frozen_sensor_config,
                                          reapply_uncertainties=self.reapply_uncertainties_at_reset)

            scada_data = self._next_sim_itr()

        r = scada_data
        if isinstance(r, tuple):
            r, _ = r
        r = self._get_observation(r)

        return r, {"scada_data": scada_data}

    def _get_observation(self, scada_data: ScadaData) -> np.ndarray:
        if scada_data is not None:
            return scada_data.get_data().flatten().astype(np.float32)
        else:
            return None

    @abstractmethod
    def _compute_reward_function(self, scada_data: ScadaData) -> float:
        """
        Computes the current reward based on the current sensors readings (i.e. SCADA data).

        Parameters
        ----------
        scada_data :`epyt_flow.simulation.ScadaData <https://epyt-flow.readthedocs.io/en/stable/epyt_flow.simulation.scada.html#epyt_flow.simulation.scada.scada_data.ScadaData>`_
            Current sensor readings.

        Returns
        -------
        `float`
            Current reward.
        """
        raise NotImplementedError()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Performs the next step by applying an action and observing the next
        state together with a reward.

        Parameters
        ----------
        action : `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
            Actions to be executed.

        Returns
        -------
        tuple[`numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_, float, bool, bool, dict]
            Observation (`numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_), reward, terminated, False (truncated), {"scada_data": `ScadaData <https://epyt-flow.readthedocs.io/en/stable/epyt_flow.simulation.scada.html#epyt_flow.simulation.scada.scada_data.ScadaData>`_}
            (`epyt_flow.simulation.ScadaData <https://epyt-flow.readthedocs.io/en/stable/epyt_flow.simulation.scada.html#epyt_flow.simulation.scada.scada_data.ScadaData>`_ as additional info).
        """
        # Apply actions
        for action_value, action in zip(action, self._action_space):
            action.apply(self, action_value)

        # Run one simulation step and observe the sensor readings (SCADA data)
        if self.autoreset is False:
            current_scada_data, terminated = self._next_sim_itr()
        else:
            terminated = False
            current_scada_data = self._next_sim_itr()

        if isinstance(current_scada_data, tuple):
            current_scada_data, _ = current_scada_data

        if current_scada_data is not None:
            obs = self._get_observation(current_scada_data)

            # Calculate reward
            current_reward = self._compute_reward_function(deepcopy(current_scada_data))
        else:
            obs = None
            current_reward = None

        # Return observation and reward
        return obs, current_reward, terminated, False, {"scada_data": current_scada_data}
