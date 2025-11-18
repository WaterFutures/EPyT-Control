.. _tut.create_env:

*******************
Custom Environments
*******************

EPyT-Control also allows the user to easily create their own control environments
for developing and testing control strategies and methods.

Creating a custom control environment requires two steps:

1. :ref:`Creating a scenario <create_scenario>`
2. :ref:`Creating an environment <create_environment>`

.. _create_scenario:

Creating a Scenario
+++++++++++++++++++

Specifying the (EPANET or EPANET-MSX) scenario that will be used in the control environment.
The specification is needed as an
`epyt_flow.simulation.ScenarioConfig <https://epyt-flow.readthedocs.io/en/stable/epyt_flow.simulation.html#epyt_flow.simulation.scenario_config.ScenarioConfig>`_
instance.

Note that the observation space (i.e. input to the agent/control strategy) is automatically derived
from the specified sensor configuration.


.. _create_environment:

Creating an Environment
+++++++++++++++++++++++

Creating a custom environment requires deriving a child class from
:class:`~epyt_control.envs.hydraulic_control_env.HydraulicControlEnv`
(if you are dealing with an EPANET scenario) or
:class:`~epyt_control.envs.advanced_quality_control_env.AdvancedQualityControlEnv`
(if you are dealing with an EPANET-MSX scenario).

.. note::

    Note that ``EpanetControlEnv`` is a synonym for
    :class:`~epyt_control.envs.hydraulic_control_env.HydraulicControlEnv` and
    ``EpanetMsxControlEnv`` is a synonym for
    :class:`~epyt_control.envs.advanced_quality_control_env.AdvancedQualityControlEnv`.

In this child class, you have to overwrite and implement the :func:`~epyt_control.envs.rl_env.RlEnv._compute_reward_function`
function. This function gets as an input the system state as a
`epyt_flow.simulation.ScadaData <https://epyt-flow.readthedocs.io/en/stable/epyt_flow.simulation.scada.html#epyt_flow.simulation.scada.scada_data.ScadaData>`_
instance, and must return a reward -- this function can make arbitrary changes to the given
`epyt_flow.simulation.ScadaData <https://epyt-flow.readthedocs.io/en/stable/epyt_flow.simulation.scada.html#epyt_flow.simulation.scada.scada_data.ScadaData>`_
instance because it will be discarded after this function is called.

The `__init__` function of the parent class requires the scenario configuration (as a
`epyt_flow.simulation.ScenarioConfig <https://epyt-flow.readthedocs.io/en/stable/epyt_flow.simulation.html#epyt_flow.simulation.scenario_config.ScenarioConfig>`_
instance) describing the scenario for which a control strategy is required.

.. note::

    The observation space (i.e. input to the agent/control strategy) is automatically derived from
    the sensor configuration specified in the scenario configuration.


Furthermore, the action space has to be specied as well. For that,
:class:`~epyt_control.envs.hydraulic_control_env.HydraulicControlEnv` and
:class:`~epyt_control.envs.advanced_quality_control_env.AdvancedQualityControlEnv`
provide arguments where a list of all actions (per action type) specify the action space --
please see the following tables for an overview of all supported types of actions.

Possible actions in an EPANET scenario (i.e. an :class:`~epyt_control.envs.hydraulic_control_env.HydraulicControlEnv` instance):

+-----------------------------------------------------------------------------+-------------------------------+
| Implementation                                                              | Description                   |
+=============================================================================+===============================+
| :class:`~epyt_control.envs.actions.actuator_state_actions.ValveStateAction` | Opening/Closing a valve.      |
+-----------------------------------------------------------------------------+-------------------------------+
| :class:`~epyt_control.envs.actions.actuator_state_actions.PumpStateAction`  | Starting/Stopping a pump.     |
+-----------------------------------------------------------------------------+-------------------------------+
| :class:`~epyt_control.envs.actions.pump_speed_actions.PumpSpeedAction`      | Setting the speed of a pump.  |
+-----------------------------------------------------------------------------+-------------------------------+
| :class:`~epyt_control.envs.actions.quality_actions.ChemicalInjectionAction` | Injecting a chemical.         |
+-----------------------------------------------------------------------------+-------------------------------+

Possible actions in an EPANET-MSX scenario (i.e. an :class:`~epyt_control.envs.advanced_quality_control_env.AdvancedQualityControlEnv` instance):

+----------------------------------------------------------------------------+--------------------------------+
| Implementation                                                             | Description                    |
+============================================================================+================================+
| :class:`~epyt_control.envs.actions.quality_actions.SpeciesInjectionAction` | Injecting a specific species.  |
+----------------------------------------------------------------------------+--------------------------------+


Multi-Config Environments
-------------------------

The environments :class:`~epyt_control.envs.hydraulic_control_env.HydraulicControlEnv` and
:class:`~epyt_control.envs.advanced_quality_control_env.AdvancedQualityControlEnv` can only handle
a single EPANET or EPANET-MSX scenario. 

However, the corresponding equivalents
:class:`~epyt_control.envs.hydraulic_control_env.MultiConfigHydraulicControlEnv`
(also available as ``MultiConfigEpanetControlEnv``) and
:class:`~epyt_control.envs.advanced_quality_control_env.MultiConfigAdvancedQualityControlEnv`
(also available as ``MultiConfigEpanetMsxControlEnv``)
support an arbitrary number of scenarios that are processed in a Round-robin scheduling scheme -- i.e.
the environment switches to the next scenario whenever the current scenario is finished.


Example
+++++++

Example of creating an EPANET-MSX environment for controlling the chlorine (CL2) injection
in the Hanoi network (given as "Hanoi.inp"), where we place a chlorine injection pump at
the reservoir (node "1"). The dynamics of chlorine are described in "cl2.msx" which is given as
well.
The objective is to make sure that the chlorine concentration stays within a pre-defined bound.

First, we have to create a new scenario, specify the CL2 source (will be used for controlling the
CL2 injection in the environment), and specify a sensor configuration from which the
observation space will be derived automatically:

.. code-block:: python

    with ScenarioSimulator(f_inp_in="Hanoi.inp", f_msx_in="cl2.msx") as scenario:
        # Set simulation duration to 21 days -- see EPANET-MSX bug
        scenario.set_general_parameters(simulation_duration=to_seconds(days=21))

        # Place a chlorine injection pump at the reservoirs (node "1")
        scenario.add_species_injection_source(species_id="CL2",
                                              node_id="1",
                                              pattern=np.array([1]),
                                              source_type=EpanetConstants.EN_MASS,
                                              pattern_id=f"cl2-injection-at-node_1")

        # Place flow sensors everywhere
        scenario.sensor_config = SensorConfig.create_empty_sensor_config(sim.sensor_config)
        scenario.set_flow_sensors(scenario.sensor_config.links)

        # Export .inp and .msx files
        scenario.save_to_epanet_file(inp_file_path="hanoi-cl2.inp",
                                     msx_file_path="hanoi-cl2.msx")

        # Export scenario
        scenario.get_scenario_config().save_to_file("hanoi-cl2")

Second, we create the environment -- there is only one action (CL2 injection at the reservoir)
and we decide not to re-run the hydraulic simulation when the environment is reset:

.. code-block:: python

    class MyEnv(AdvancedQualityControlEnv):
        def __init__(self, scenario_config_file_in: str):
            cl_injection_action = SpeciesInjectionAction(species_id="CL2",
                                                         node_id="1",
                                                         pattern_id="cl2-injection-at-node_1",
                                                         source_type_id=EpanetConstants.EN_MASS,
                                                         upper_bound=10000.)

            scenario_config = ScenarioConfig.load_from_file(scenario_config_file_in)
            super().__init__(scenario_config=scenario_config,
                             action_space=[cl_injection_action],
                             autoreset=True,
                             rerun_hydraulics_when_reset=False)

            self.__sensor_config_reward = None

        def _compute_reward_function(self, scada_data: ScadaData) -> float:
            # Regulation Limits
            lower_cl_bound = .3  # (mg/l)
            upper_cl_bound = 2.  # (mg/l)

            # Change the sensor configuration to measure the CL2 concentration at every node
            if self.__sensor_config_reward is None:
                self.__sensor_config_reward = SensorConfig.create_empty_sensor_config(scada_data.sensor_config)
                self.__sensor_config_reward.bulk_species_node_sensors = {"CL2": scada_data.sensor_config.nodes}
            scada_data.change_sensor_config(self.__sensor_config_reward)

            nodes_quality = scada_data.get_data_bulk_species_node_concentration({"CL2": scada_data.sensor_config.nodes})

            # Sum up (negative) residuals for out-of-bounds Cl concentrations at nodes -- i.e.
            # reward of zero means everything is okay, while a negative reward
            # denotes Cl concentration bound violations
            reward = 0.

            upper_bound_violation_idx = nodes_quality > upper_cl_bound
            reward += -1. * np.sum(nodes_quality[upper_bound_violation_idx] - upper_cl_bound)

            lower_bound_violation_idx = nodes_quality < lower_cl_bound
            reward += np.sum(nodes_quality[lower_bound_violation_idx] - lower_cl_bound)

            return reward
