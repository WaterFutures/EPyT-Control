from epyt_flow.simulation import ScenarioSimulator, EpanetConstants, ScadaData


def run_simulation(f_inp_in: str, f_msx_in: str = None, duration: int=None,
                   hydraulic_step: int = None, quality_step: int = None,
                   source_conc: dict[str, list[float]] = {},
                   verbose: bool = True, **kwargs) -> ScadaData:

    with ScenarioSimulator(f_inp_in=f_inp_in, f_msx_in=f_msx_in) as sim:
            sim.set_general_parameters(
                simulation_duration=duration,
                hydraulic_time_step=hydraulic_step,
                reporting_time_step=hydraulic_step,
                quality_time_step=quality_step,
                flow_units_id=EpanetConstants.EN_CMH,
                **kwargs
            )

            sim.enable_chemical_analysis()
            sim.epanet_api.settimeparam(EpanetConstants.EN_PATTERNSTEP, hydraulic_step)

            if f_msx_in is not None:
                sim.epanet_api.set_msx_time_step(quality_step)

            for node_id, pattern in source_conc.items():
                if f_msx_in is not None:
                    sim.add_species_injection_source(
                        'CL2', 
                        node_id=node_id,
                        pattern=pattern,
                        source_type=EpanetConstants.EN_SETPOINT
                    )
                else:
                    sim.add_quality_source(
                        node_id=node_id,
                        pattern=pattern,
                        source_type=EpanetConstants.EN_CONCEN
                    )

            # Place quality and flow sensor everywhere
            sim.set_flow_sensors(sensor_locations=sim.sensor_config.links)

            if f_msx_in is not None:
                sim.set_bulk_species_node_sensors(sensor_info={
                        "CL2": sim.sensor_config.nodes,
                })
            else:
                sim.set_node_quality_sensors(sensor_locations=sim.sensor_config.nodes)

            # Run simulation and store results
            scada_data = sim.run_simulation(verbose=verbose, frozen_sensor_config=True)

            return scada_data
