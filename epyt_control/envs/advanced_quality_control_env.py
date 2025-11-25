"""
This module contains a base class for an EPANET-MSX control environment --
i.e. controlling the injection and reaction of one or multiple species in an
EPANET-MSX scenario (no control over pumps, valves, etc.).
"""
import os
import uuid
from typing import Optional, Any
import warnings
import numpy as np
from epyt_flow.simulation import ScenarioConfig, ScenarioSimulator, ScadaData
from epyt_flow.utils import get_temp_folder
from gymnasium.spaces import Dict
from gymnasium.spaces.utils import flatten_space
from gymnasium import Env

from .actions.quality_actions import SpeciesInjectionAction
from .rl_env import RlEnv


class EpanetMsxControlEnv(RlEnv):
    """
    Base class for advanced quality control scenarios -- i.e. EPANET-MSX control scenarios.

    Parameters
    ----------
    scenario_config : `epyt_flow.simulation.ScenarioConfig <https://epyt-flow.readthedocs.io/en/stable/epyt_flow.simulation.html#epyt_flow.simulation.scenario_config.ScenarioConfig>`_
        Configuration of the scenario.
    action_space : list[:class:`~epyt_control.actions.quality_actions.SpeciesInjectionAction`]
        The action spaces (i.e. list of species injections) that have to be controlled by the agent.
    rerun_hydraulics_when_reset : `bool`, optional
        If True, the hydraulic simulation is going to be re-run when the environment is reset,
        otherwise the hydraulics from the initial run are re-used and the scenario will
        also not be reloaded -- i.e. reload_scenario_when_reset=False.

        The default is True.
    hyd_file_in : `str`, optional
        Path to an EPANET .hyd file containing the simulated hydraulics.
        Can only be used in conjunction with 'hyd_scada_in'.
        If set, hydraulics will not be simulated but taken from the specified file.

        The default is None.
    hyd_scada_in : `epyt_flow.simulation.ScadaData <https://epyt-flow.readthedocs.io/en/stable/epyt_flow.simulation.scada.html#epyt_flow.simulation.scada.scada_data.ScadaData>`_, optional
        ScadaData instance containing the simulated hydraulics -- must match the hydraulics
        from 'hyd_file_in'. Can only be used in conjunction with 'hyd_file_in'.

        The default is None.
    """
    def __init__(self, scenario_config: ScenarioConfig,
                 action_space: list[SpeciesInjectionAction],
                 rerun_hydraulics_when_reset: bool = True,
                 hyd_file_in: str = None, hyd_scada_in: ScadaData = None,**kwds):
        if not isinstance(action_space, list):
            raise TypeError("'action_space' must be an instance of " +
                            "`list[SpeciesInjectionActionSpace]` " +
                            f"but not of '{type(action_space)}'")
        if any(not isinstance(action_desc, SpeciesInjectionAction)
               for action_desc in action_space):
            raise TypeError("All items in 'action_space' must be an instance of " +
                            "'epyt_control.actions.quality_actions.SpeciesInjectionAction'")
        if len(action_space) == 0:
            raise ValueError("Empty action space")
        if not isinstance(rerun_hydraulics_when_reset, bool):
            raise TypeError("'rerun_hydraulics_when_reset' must be an instance of 'bool' " +
                            f"but not of '{type(rerun_hydraulics_when_reset)}'")
        if "reload_scenario_when_reset" in kwds:
            if kwds["reload_scenario_when_reset"] is True and rerun_hydraulics_when_reset is False:
                raise ValueError("'rerun_hydraulics_when_reset' must be True " +
                                 "if 'reload_scenario_when_reset=True'")
        else:
            if rerun_hydraulics_when_reset is False:
                kwds["reload_scenario_when_reset"] = False

        if hyd_scada_in is not None and hyd_file_in is not None:
            if rerun_hydraulics_when_reset is True:
                raise ValueError("'rerun_hydraulics_when_reset' must be False " +
                                 "if pre-computed hydraulics are provided")

        self._rerun_hydraulics_when_reset = rerun_hydraulics_when_reset
        self._hyd_export = os.path.join(get_temp_folder(),
                                        f"epytcontrol_env_MSX_{uuid.uuid4()}.hyd")
        gym_action_space = flatten_space(Dict({f"{action_space.species_id}-{action_space.node_id}":
                                               action_space.to_gym_action_space()
                                               for action_space in action_space}))

        super().__init__(scenario_config=scenario_config, gym_action_space=gym_action_space,
                         action_space=action_space, hyd_scada_in=hyd_scada_in,
                         hyd_file_in=hyd_file_in, **kwds)

    def reset(self, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
              ) -> tuple[np.ndarray, dict]:
        Env.reset(self, seed=seed)

        if self._rerun_hydraulics_when_reset is True:
            scada_data = super().reset()
        else:
            if self._scenario_sim is None or self._reload_scenario_when_reset:
                self._scenario_sim = ScenarioSimulator(
                    scenario_config=self._scenario_config)

                # Run hydraulic simulation first if necessary
                if self._hyd_file_in is not None:
                    self._hyd_export = self._hyd_file_in
                    self._hydraulic_scada_data = self._hyd_scada_in
                else:
                    sim = self._scenario_sim.run_hydraulic_simulation
                    self._hydraulic_scada_data = sim(hyd_export=self._hyd_export,
                                                     reapply_uncertainties=self.reapply_uncertainties_at_reset)
            else:
                # Abort current simulation if any is runing
                try:
                    next(self._sim_generator)
                    self._sim_generator.send(True)
                except StopIteration:
                    pass

            # Run advanced quality analysis (EPANET-MSX) on top of the computed hydraulics
            gen = self._scenario_sim.run_advanced_quality_simulation_as_generator
            self._sim_generator = gen(self._hyd_export, support_abort=True,
                                      reapply_uncertainties=self.reapply_uncertainties_at_reset)

            scada_data = self._next_sim_itr()

        r = scada_data
        if isinstance(r, tuple):
            r, _ = r

        cur_time = int(r.sensor_readings_time[0])
        hyd_scada = self._hydraulic_scada_data.extract_time_window(start_time=cur_time,
                                                                   end_time=cur_time)
        r.join(hyd_scada)

        r = self._get_observation(r)

        return r, {"scada_data": scada_data}


AdvancedQualityControlEnv = EpanetMsxControlEnv


class MultiConfigEpanetMsxControlEnv(EpanetMsxControlEnv):
    """
    Base class for advanced quality control scenarios (i.e. EPANET-MSX control scenarios) that can
    handle multiple scenario configurations -- those scenarios are utilized in a round-robin
    scheduling scheme (i.e. autorest=True).

    Note that all scenarios must share the same action and observation space.

    Parameters
    ----------
    scenario_configs : list[`epyt_flow.simulation.ScenarioConfig <https://epyt-flow.readthedocs.io/en/stable/epyt_flow.simulation.html#epyt_flow.simulation.scenario_config.ScenarioConfig>`_]
        Configuration of the scenario. Note that all sceanrios must share the same action and
        observation space.
    action_space : list[:class:`~epyt_control.actions.quality_actions.SpeciesInjectionAction`]
        The action spaces (i.e. list of species injections) that have to be controlled by the agent.
        Must be the same for all scenarios specified in 'scenario_configs'.
    rerun_hydraulics_when_reset : `bool`, optional
        If True, the hydraulic simulation is going to be re-run when the environment is reset,
        otherwise the hydraulics from the initial run are re-used and the scenario will
        also not be reloaded -- i.e. reload_scenario_when_reset=False.
    precomputed_hydraulics : list[tuple[str, `epyt_flow.simulation.ScadaData <https://epyt-flow.readthedocs.io/en/stable/epyt_flow.simulation.scada.html#epyt_flow.simulation.scada.scada_data.ScadaData>`_]], optional
        Pre-computed hydraulics -- i.e., for each scenario in 'scenario_configs', a tuple of a
        path to an EPANET generatd .hyd file and a ScadaData instance -- that are used instead
        of re-running the hydraulic simulation.
        If used, 'rerun_hydraulics_when_reset' must be False.

        The default is None.
    """
    def __init__(self, scenario_configs: list[ScenarioConfig],
                 action_space: list[SpeciesInjectionAction],
                 rerun_hydraulics_when_reset: bool = True,
                 precomputed_hydraulics: list[tuple[str, ScadaData]] = None, **kwds):
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
        self._hyd_exports = [os.path.join(get_temp_folder(),
                                          f"epytcontrol_env_MSX_{uuid.uuid4()}.hyd")
                             for _ in range(len(scenario_configs))]
        self._hydraulic_scada_datas = [None] * len(scenario_configs)
        self._current_scenario_idx = 0
        self._use_precomputed_hydraulics = False

        if precomputed_hydraulics is not None:
            def __raise_type_error():
                raise TypeError("'precomputed_hydraulics' must be an instance of " +
                                "'list[tuple[str, epyt_flow.simulation.ScadaData]]'")

            if not isinstance(precomputed_hydraulics, list):
                __raise_type_error()
            if any(not isinstance(hyd, tuple) for hyd in precomputed_hydraulics):
                __raise_type_error()
            if any(not isinstance(hyd[0], str) or not isinstance(hyd[1], ScadaData)
                   for hyd in precomputed_hydraulics):
                __raise_type_error()
            if len(precomputed_hydraulics) != len(scenario_configs):
                raise ValueError("Length of 'precomputed_hydraulics' must be equal to the " +
                                 "number of scenarios in 'scenario_configs'")
            if rerun_hydraulics_when_reset is True:
                raise ValueError("'rerun_hydraulics_when_reset' msut be False if " +
                                 "pre-computed hydraulics are used")

            self._hyd_exports = []
            self._hydraulic_scada_datas = []
            for hyd_file_in, scada_hyd_in in precomputed_hydraulics:
                self._hyd_exports.append(hyd_file_in)
                self._hydraulic_scada_datas.append(scada_hyd_in)

            self._use_precomputed_hydraulics = True

        super().__init__(self._scenario_configs[self._current_scenario_idx],
                         action_space, rerun_hydraulics_when_reset,
                         autoreset=True, **kwds)
        self._hyd_export = self._hyd_exports[self._current_scenario_idx]
        self._hydraulic_scada_data = self._hydraulic_scada_datas[self._current_scenario_idx]

    def reset(self, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
              ) -> tuple[np.ndarray, dict]:
        # Back up current simulation
        self._scenario_sims[self._current_scenario_idx] = self._scenario_sim
        self._hydraulic_scada_datas[self._current_scenario_idx] = self._hydraulic_scada_data

        # Move on to next scenario
        self._current_scenario_idx = (self._current_scenario_idx + 1) % len(self._scenario_configs)
        self._scenario_config = self._scenario_configs[self._current_scenario_idx]
        self._scenario_sim = self._scenario_sims[self._current_scenario_idx]
        self._hyd_export = self._hyd_exports[self._current_scenario_idx]
        self._hydraulic_scada_data = self._hydraulic_scada_datas[self._current_scenario_idx]

        return super().reset(seed, options)


MultiConfigAdvancedQualityControlEnv = MultiConfigEpanetMsxControlEnv
