"""
This module contains classes for modeling chemical/species injection action spaces
-- i.e. a chemical (EPANET) or species (EPANET-MSX) injection that has to be
controlled by the agent.
"""
from typing import Optional
from epyt_flow.gym import ScenarioControlEnv
from epyt_flow.simulation import EpanetConstants
from gymnasium.spaces import Space, Box

from .actions import Action


class ChemicalInjectionAction(Action):
    """
    Action for controlling the injection of a chemical -- only for EPANET control scenarios.

    Parameters
    ----------
    node_id : `str`
        ID of the node at which the injection is going to happen.
    pattern_id : `str`
        ID of the pattern that is used for the injection.
    source_type_id : `int`
        Type of the injection source -- must be one of
        the following EPANET constants:

            - EN_CONCEN     = 0
            - EN_MASS       = 1
            - EN_SETPOINT   = 2
            - EN_FLOWPACED  = 3

        Description:

            - E_CONCEN Sets the concentration of external inflow entering a node
            - EN_MASS Injects a given mass/minute into a node
            - EN_SETPOINT Sets the concentration leaving a node to a given value
            - EN_FLOWPACED Adds a given value to the concentration leaving a node
    upper_bound : `float`, optional
        Upper bound on the amount that can be injected by the agent.
        If None, there is no upper bound on the amount.

        The default is None.
    """
    def __init__(self, node_id: str, pattern_id: str,
                 source_type_id: int, upper_bound: Optional[float] = None,
                 **kwds):
        if not isinstance(node_id, str):
            raise TypeError(f"'node_id' must be an instance of 'str' but not of '{type(node_id)}'")
        if not isinstance(pattern_id, str):
            raise TypeError("'pattern_id' must be an instance of 'str' " +
                            f"but not of '{type(pattern_id)}'")
        if not isinstance(source_type_id, int):
            raise TypeError("'source_type_id' must be an instance of 'int' " +
                            f"but not of '{type(source_type_id)}'")
        if source_type_id not in [EpanetConstants.EN_MASS, EpanetConstants.EN_CONCEN,
                                  EpanetConstants.EN_SETPOINT, EpanetConstants.EN_FLOWPACED]:
            raise ValueError("Invalid 'source_type_id'")
        if upper_bound is not None:
            if not isinstance(upper_bound, float):
                raise TypeError("'upper_bound' must be an instance of 'float' " +
                                f"but not of '{type(upper_bound)}'")
            if upper_bound <= 0:
                raise ValueError("'upper_bound' must be positive")

        self._node_id = node_id
        self._pattern_id = pattern_id
        self._source_type_id = source_type_id
        self._upper_bound = upper_bound

        super().__init__(**kwds)

    @property
    def node_id(self) -> str:
        """
        Return the ID of the node at which the injection is going to happen.

        Returns
        -------
        `str`
            ID of the node.
        """
        return self._node_id

    @property
    def pattern_id(self) -> str:
        """
        Returns the ID of the pattern that is used for the injection.

        Returns
        -------
        `str`
            ID of the pattern.
        """
        return self._pattern_id

    @property
    def source_type_id(self) -> int:
        """
        Returns the type (i.e. ID) of the injection source -- will be one of
        the following EPANET toolkit constants:

            - EN_CONCEN     = 0
            - EN_MASS       = 1
            - EN_SETPOINT   = 2
            - EN_FLOWPACED  = 3

        Description:

            - E_CONCEN Sets the concentration of external inflow entering a node
            - EN_MASS Injects a given mass/minute into a node
            - EN_SETPOINT Sets the concentration leaving a node to a given value
            - EN_FLOWPACED Adds a given value to the concentration leaving a node

        Returns
        -------
        `int`
            Type (ID) of the injection source.
        """
        return self._source_type_id

    @property
    def upper_bound(self) -> float:
        """
        Returns the upper bound on the amount that can be injected by the agent.

        Returns
        -------
        `float`
            Upper bound.
        """
        return self._upper_bound

    def __eq__(self, other) -> bool:
        return super().__eq__(other) and self._node_id == other.node_id and \
            self._pattern_id == other.pattern_id and \
            self._source_type_id == other.source_type_id and \
            self._upper_bound == other.upper_bound

    def __str__(self) -> str:
        return f"Node ID: {self._node_id} " +\
            f"Pattern ID: {self._pattern_id} Source type ID: {self._source_type_id} " +\
            f"Upper bound: {self._upper_bound}"

    def to_gym_action_space(self) -> Space:
        return Box(low=0, high=self._upper_bound if self._upper_bound is not None
                   else float("inf"))

    def apply(self, env: ScenarioControlEnv, action_value: float) -> None:
        env.set_node_quality_source_value(self._node_id, self._pattern_id, action_value)


class SpeciesInjectionAction(ChemicalInjectionAction):
    """
    Action for controlling the injection of a given species --
    only for EPANET-MSX control scenarios.

    Parameters
    ----------
    species_id : `str`
        ID of the species that is going to be injected by the agent.
    node_id : `str`
        ID of the node at which the injection is going to happen.
    pattern_id : `str`
        ID of the pattern that is used for the injection.
    source_type_id : `int`
        Types of the injection source -- must be one of
        the following EPANET toolkit constants:

            - EN_CONCEN     = 0
            - EN_MASS       = 1
            - EN_SETPOINT   = 2
            - EN_FLOWPACED  = 3

        Description:

            - E_CONCEN Sets the concentration of external inflow entering a node
            - EN_MASS Injects a given mass/minute into a node
            - EN_SETPOINT Sets the concentration leaving a node to a given value
            - EN_FLOWPACED Adds a given value to the concentration leaving a node
    upper_bound : `float`, optional
        Upper bound on the amount that can be injected by the agent.
        If None, there is no upper bound on the amount.

        The default is None.
    """
    def __init__(self, species_id: str, node_id: str, pattern_id: str,
                 source_type_id: int, upper_bound: Optional[float] = None,
                 **kwds):
        if not isinstance(species_id, str):
            raise TypeError("'species_id' must be an instance of 'str' " +
                            f"but not of '{type(species_id)}'")

        self._species_id = species_id

        super().__init__(node_id=node_id, pattern_id=pattern_id,
                         source_type_id=source_type_id, upper_bound=upper_bound, **kwds)

    @property
    def species_id(self) -> str:
        """
        Returns the ID of the species that is going to be injected.

        Returns
        -------
        `str`
            ID of the species.
        """
        return self._species_id

    def __eq__(self, other) -> bool:
        return super().__eq__(other) and self._species_id == other.species_id

    def __str__(self) -> str:
        return super().__str__() + f"Species ID: {self._species_id}"

    def apply(self, env: ScenarioControlEnv, action_value: float) -> None:
        env.set_node_species_source_value(self._species_id, self._node_id, self._source_type_id,
                                          self._pattern_id, action_value)
