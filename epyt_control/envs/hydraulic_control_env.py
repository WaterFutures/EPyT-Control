"""
This module contains a base class for EPANET control environments --
i.e. controlling hydraulic actuators such as pumps and valves or single chemical (no EPANET-MSX support!).
"""
from typing import Optional, Any
import warnings
import numpy as np
from epyt_flow.simulation import ScenarioConfig
from gymnasium.spaces import Dict
from gymnasium.spaces.utils import flatten_space

from .rl_env import RlEnv
from .actions.pump_speed_actions import PumpSpeedAction
from .actions.quality_actions import ChemicalInjectionAction
from .actions.actuator_state_actions import PumpStateAction, ValveStateAction


class EpanetControlEnv(RlEnv):
    """
    Base class for hydraulic control environments
    (incl. basic quality that can be simulated with EPANET only).

    Parameters
    ----------
    scenario_config : `epyt_flow.simulation.ScenarioConfig <https://epyt-flow.readthedocs.io/en/stable/epyt_flow.simulation.html#epyt_flow.simulation.scenario_config.ScenarioConfig>`_
        Configuration of the scenario.
    pumps_speed_actions : list[:class:`~epyt_control.actions.pump_speed_actions.PumpSpeedAction`], optional
        List of pumps where the speed has to be controlled.

        The default is None.
    pumps_state_actions : list[:class:`~epyt_control.actions.actuator_state_actions.PumpStateAction`], optional
        Lisst of pumps where the state has to be controlled.

        The default is None.
    valves_state_actions : list[:class:`~epyt_control.actions.actuator_state_actions.ValveStateAction`], optional
        List of valves that have to be controlled.

        The default is None.
    chemical_injection_actions : list[:class:`~epyt_control.actions.quality_actions.ChemicalInjectionAction`], optional
        List chemical injection actions -- i.e. places in the network where the
        injection of the chemical has to be controlled.

        The default is None.
    """
    def __init__(self, scenario_config: ScenarioConfig,
                 pumps_speed_actions: Optional[list[PumpSpeedAction]] = None,
                 pumps_state_actions: Optional[list[PumpStateAction]] = None,
                 valves_state_actions: Optional[list[ValveStateAction]] = None,
                 chemical_injection_actions: Optional[list[ChemicalInjectionAction]] = None,
                 **kwds):
        if pumps_speed_actions is not None:
            if not isinstance(pumps_speed_actions, list):
                raise TypeError("'pumps_speed_actions' must be an instance of " +
                                "'list[PumpSpeedAction]' but not of " +
                                f"'{type(pumps_speed_actions)}'")
            if any(not isinstance(pump_speed_action, PumpSpeedAction)
                   for pump_speed_action in pumps_speed_actions):
                raise TypeError("All items in 'pumps_speed_actions' must be an instance of " +
                                "'PumpSpeedAction'")
        if pumps_state_actions is not None:
            if not isinstance(pumps_state_actions, list):
                raise TypeError("'pumps_state_actions' must be an instance of " +
                                "'list[PumpStateAction]' but not of " +
                                f"'{type(pumps_state_actions)}'")
            if any(not isinstance(pump_state_action, PumpStateAction)
                   for pump_state_action in pumps_state_actions):
                raise TypeError("All items in 'pumps_state_actions' must be an instance of " +
                                "'PumpStateAction'")
        if valves_state_actions is not None:
            if not isinstance(valves_state_actions, list):
                raise TypeError("'valves_state_actions' must be an instance of " +
                                "'list[ValveAction]' but not of " +
                                f"'{type(valves_state_actions)}'")
            if any(not isinstance(valve_state_action, ValveStateAction)
                   for valve_state_action in valves_state_actions):
                raise TypeError("All items in 'valves_state_actions' must " +
                                "be an instance of 'ValveStateAction'")
        if chemical_injection_actions is not None:
            if not isinstance(chemical_injection_actions, list):
                raise TypeError("'chemical_injection_actions' must be an instance of " +
                                "'list[ChemicalInjectionAction]' but not of " +
                                f"'{type(chemical_injection_actions)}'")
            if any(not isinstance(chemical_injection_action, ChemicalInjectionAction)
                   for chemical_injection_action in chemical_injection_actions):
                raise TypeError("All items in 'chemical_injection_actions' " +
                                "must be an instance of 'ChemicalInjectionAction'")

        self._pumps_speed_actions = pumps_speed_actions
        self._pumps_state_actions = pumps_state_actions
        self._valves_state_actions = valves_state_actions
        self._chemical_injection_actions = chemical_injection_actions

        action_space = {}
        my_actions = []
        if self._pumps_speed_actions is not None:
            my_actions += self._pumps_speed_actions
            action_space |= {f"{action_space.pump_id}-speed": action_space.to_gym_action_space()
                             for action_space in self._pumps_speed_actions}
        if self._pumps_state_actions is not None:
            my_actions += self._pumps_state_actions
            action_space |= {f"{action_space.pump_id}-state": action_space.to_gym_action_space()
                             for action_space in self._pumps_state_actions}
        if self._valves_state_actions is not None:
            my_actions += self._valves_state_actions
            action_space |= {f"{action_space.valve_id}-state": action_space.to_gym_action_space()
                             for action_space in self._valves_state_actions}
        if self._valves_state_actions is not None:
            my_actions += self._valves_state_actions
            action_space |= {f"{action_space.valve_id}-state": action_space.to_gym_action_space()
                             for action_space in self._valves_state_actions}
        if self._chemical_injection_actions is not None:
            my_actions += self._chemical_injection_actions
            action_space |= {f"{action_space.node_id}-chem": action_space.to_gym_action_space()
                             for action_space in self._chemical_injection_actions}

        gym_action_space = flatten_space(Dict(action_space))

        super().__init__(scenario_config=scenario_config, gym_action_space=gym_action_space,
                         action_space=my_actions, **kwds)


HydraulicControlEnv = EpanetControlEnv


class MultiConfigEpanetControlEnv(EpanetControlEnv):
    """
    Base class for hydraulic control environments (incl. basic quality that can be simulated
    with EPANET only) that can handle multiple scenario configurations -- those scenarios are
    utilized in a round-robin scheduling scheme (i.e. autorest=True).

    Parameters
    ----------
    scenario_configs : list[`epyt_flow.simulation.ScenarioConfig <https://epyt-flow.readthedocs.io/en/stable/epyt_flow.simulation.html#epyt_flow.simulation.scenario_config.ScenarioConfig>`_]
        List of all scenario configurations that are used in this environment.
    pumps_speed_actions : list[:class:`~epyt_control.actions.pump_speed_actions.PumpSpeedAction`], optional
        List of pumps where the speed has to be controlled.

        The default is None.
    pumps_state_actions : list[:class:`~epyt_control.actions.actuator_state_actions.PumpStateAction`], optional
        Lisst of pumps where the state has to be controlled.

        The default is None.
    valves_state_actions : list[:class:`~epyt_control.actions.actuator_state_actions.ValveStateAction`], optional
        List of valves that have to be controlled.

        The default is None.
    chemical_injection_actions : list[:class:`~epyt_control.actions.quality_actions.ChemicalInjectionAction`], optional
        List chemical injection actions -- i.e. places in the network where the
        injection of the chemical has to be controlled.

        The default is None.
    reload_scenario_when_reset : `bool`, optional
        If True, the scenario (incl. the .inp and .msx file) is reloaded from the hard disk.
        If False, only the simulation is reset.

        The default is True.
    """
    def __init__(self, scenario_configs: list[ScenarioConfig],
                 pumps_speed_actions: Optional[list[PumpSpeedAction]] = None,
                 pumps_state_actions: Optional[list[PumpStateAction]] = None,
                 valves_state_actions: Optional[list[ValveStateAction]] = None,
                 chemical_injection_actions: Optional[list[ChemicalInjectionAction]] = None,
                 reload_scenario_when_reset: bool = True):
        if not isinstance(scenario_configs, list):
            raise TypeError("'scenario_configs' must be an instance of " +
                            "epyt_flow.simulation.ScenarioConfig but " +
                            f"not of '{type(scenario_configs)}'")
        if any(not isinstance(scenario_config, ScenarioConfig)
               for scenario_config in scenario_configs):
            raise TypeError("All items in 'scenario_config' must be instances of " +
                            "epyt_flow.simulation.ScenarioConfig")

        if len(scenario_configs) > 10:
            warnings.warn("You are using many scenarios. You might face issues w.r.t. " +
                          "memory consumption as well as with the maximum number of open files " +
                          "allowed by the operating system.", UserWarning)

        self._scenario_configs = scenario_configs
        self._scenario_sims = [None] * len(scenario_configs)
        self._current_scenario_idx = 0

        super().__init__(self._scenario_configs[self._current_scenario_idx], pumps_speed_actions,
                         pumps_state_actions, valves_state_actions, chemical_injection_actions,
                         autoreset=True,
                         reload_scenario_when_reset=reload_scenario_when_reset)

    def reset(self, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
              ) -> tuple[np.ndarray, dict]:
        # Back up current simulation
        self._scenario_sims[self._current_scenario_idx] = self._scenario_sim

        # Move on to next scenario
        self._current_scenario_idx = (self._current_scenario_idx + 1) % len(self._scenario_configs)
        self._scenario_config = self._scenario_configs[self._current_scenario_idx]
        self._scenario_sim = self._scenario_sims[self._current_scenario_idx]

        return super().reset(seed, options)


MultiConfigHydraulicControlEnv = MultiConfigEpanetControlEnv
