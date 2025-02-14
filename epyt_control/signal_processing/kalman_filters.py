"""
This module contains implementations of various Kalman filters.
"""
from typing import Optional
from abc import ABC, abstractmethod
import numpy as np


class KalmanFilterBase(ABC):
    """
    Base class for Kalman filters.

    Parameters
    ----------
    state_dim : `int`
        Dimensionality of states.
    obs_dim : `int`
        Dimensionality of observations.
    init_state : `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
        Initial state.
    """
    def __init__(self, state_dim: int, obs_dim: int, init_state: np.ndarray):
        if not isinstance(state_dim, int):
            raise TypeError("'state_dim' must be an instance of 'int' " +
                            f"but not of '{type(state_dim)}'")
        if state_dim <= 0:
            raise ValueError("'state_dim' must be > 0")
        if not isinstance(obs_dim, int):
            raise TypeError("'obs_dim' must be an instance of 'int' " +
                            f"but not of '{type(obs_dim)}'")
        if obs_dim <= 0:
            raise ValueError("'obs_dim' must be > 0")
        if not isinstance(init_state, np.ndarray):
            raise TypeError("'init_state' must be an instance of 'numpy.ndarray' " +
                            f"but not of '{type(init_state)}'")
        if init_state.shape != (state_dim,):
            raise ValueError("'init_state' must be of shap (state_dim,) -- " +
                             f"i.e. {(state_dim,)}. But found {init_state.shape}")

        self._state_dim = state_dim
        self._obs_dim = obs_dim
        self._x = init_state
        self._init_state = np.copy(init_state)

    @property
    def state_dim(self) -> int:
        """
        Returns the dimensionality of states.

        Returns
        -------
        `int`
            Dimensionality.
        """
        return self._state_dim

    @property
    def obs_dim(self) -> int:
        """
        Returns the dimensionality of observations.

        Returns
        -------
        `int`
            Dimensionality.
        """
        return self._obs_dim

    @property
    def init_state(self) -> np.ndarray:
        """
        Returns the initial state.

        Returns
        -------
        `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
            Initial state.
        """
        return np.copy(self._init_state)

    def reset(self) -> None:
        """
        Resets the filter to the initial state.
        """
        self._x = np.copy(self._init_state)

    def __str__(self) -> str:
        return f"state_dim: {self._state_dim} obs_state: {self._obs_dim} x: {self._x} " +\
            f"init_state: {self._init_state}"

    def __eq__(self, other) -> bool:
        return self._state_dim == other.state_dim and self._obs_dim == other.obs_dim and \
            self._init_state == other.init_state

    @abstractmethod
    def step(self, observation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Predicts the current state (incl. it's uncertainty) based on a given current observation.
        Also, updates all other internal states of the Kalman filter.
        """
        raise NotImplementedError()


class KalmanFilter(KalmanFilterBase):
    """
    Class implementing the multivariate Kalman filter.

    Parameters
    ----------
    state_dim : `int`
        Dimensionality of states.
    obs_dim : `int`
        Dimensionality of observations.
    init_state : `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
        Initial state.
    measurement_func : `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
        Measurement function -- i.e. matrix that is converting a state into an observation.
    state_transition_func : `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
        State transition function -- i.e. matrix moving from a given state to the next state.
    init_state_uncertainty_cov : `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_, optional
        Covariance matrix of the initial state uncertainty.
        If None, the identity matrix will be used.

        The default is None.
    measurement_uncertainty_cov : `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_, optional
        Covariance matrix of the measurement/observation uncertainty.
        If None, the identity matrix will be used.

        The default is None.
    system_uncertainty_cov : `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_, optional
        Covariance matrix of the system uncertainty.
        If None, the identity matrix will be used.

        The default is None.
    """
    def __init__(self, state_dim: int, obs_dim: int, init_state: np.ndarray,
                 measurement_func: np.ndarray, state_transition_func: np.ndarray,
                 init_state_uncertainty_cov: Optional[np.ndarray] = None,
                 measurement_uncertainty_cov: Optional[np.ndarray] = None,
                 system_uncertainty_cov: Optional[np.ndarray] = None):
        if not isinstance(measurement_func, np.ndarray):
            raise ValueError("'measurement_func' must be an instance of 'numpy.ndarray' " +
                             f"but not of '{type(measurement_func)}'")
        if measurement_func.shape != (obs_dim, state_dim):
            raise ValueError("'measurement_func' must be of shape (obs_dim, state_dim) -- " +
                             f"i.e. {(obs_dim, state_dim)}. But found {measurement_func.shape}")
        if not isinstance(state_transition_func, np.ndarray):
            raise ValueError("'state_transition_func' must be an instance of 'numpy.ndarray' " +
                             f"but not of '{type(state_transition_func)}'")
        if state_transition_func.shape != (state_dim, state_dim):
            raise ValueError("'state_transition_func' must be of shape (state_dim, state_dim) -- " +
                             f"i.e. {(state_dim, state_dim)}. " +
                             f"But found {state_transition_func.shape}")
        if init_state_uncertainty_cov is not None:
            if not isinstance(init_state_uncertainty_cov, np.ndarray):
                raise ValueError("'init_state_uncertainty_cov' must be an instance of " +
                                 f"'numpy.ndarray' but not of '{type(init_state_uncertainty_cov)}'")
            if init_state_uncertainty_cov.shape != (state_dim, state_dim):
                raise ValueError("'init_state_uncertainty_cov' must be of shape " +
                                 f"(state_dim, state_dim) -- i.e. {(state_dim, state_dim)}. " +
                                 f"But found {init_state_uncertainty_cov.shape}")
        if measurement_uncertainty_cov is not None:
            if not isinstance(measurement_uncertainty_cov, np.ndarray):
                raise ValueError("'measurement_uncertainty_cov' must be an instance of " +
                                 "'numpy.ndarray' but not of " +
                                 f"'{type(measurement_uncertainty_cov)}'")
            if measurement_uncertainty_cov.shape != (obs_dim, obs_dim):
                raise ValueError("'measurement_uncertainty_cov' must be of shape " +
                                 f"(obs_dim, obs_dim) -- i.e. {(obs_dim, obs_dim)}. " +
                                 f"But found {measurement_uncertainty_cov.shape}")
        if system_uncertainty_cov is not None:
            if not isinstance(system_uncertainty_cov, np.ndarray):
                raise ValueError("'system_uncertainty_cov' must be an instance of " +
                                 f"'numpy.ndarray' but not of '{type(system_uncertainty_cov)}'")
            if system_uncertainty_cov.shape != (state_dim, state_dim):
                raise ValueError("'system_uncertainty_cov' must be of shape " +
                                 f"(state_dim, state_dim) -- i.e. {(state_dim, state_dim)}. " +
                                 f"But found {system_uncertainty_cov.shape}")

        super().__init__(state_dim=state_dim, obs_dim=obs_dim, init_state=init_state)

        self._H = measurement_func
        self._F = state_transition_func
        self._I = np.eye(state_dim)

        if init_state_uncertainty_cov is None:
            self._P = self._I
        else:
            self._P = init_state_uncertainty_cov

        if measurement_uncertainty_cov is None:
            self._R = np.eye(obs_dim)
        else:
            self._R = measurement_uncertainty_cov

        if system_uncertainty_cov is None:
            self._Q = self._I
        else:
            self._Q = system_uncertainty_cov

        self._init_state_uncertainty_cov = np.copy(self._P)

    @property
    def measurement_func(self) -> np.ndarray:
        """
        Returns the measurement function -- i.e. matrix for converting a state into an observation.

        Returns
        -------
        `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
            Measurement function/matrix.
        """
        return np.copy(self._H)

    @property
    def state_transition_func(self) -> np.ndarray:
        """
        Returns the state transition function -- i.e. matrix for moving from a given state to
        the next state.

        Returns
        -------
        `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
            State transition matrix.
        """
        return np.copy(self._F)

    @property
    def measurement_uncertainty_cov(self) -> np.ndarray:
        """
        Returns the covariance matrix of the measurement/observation uncertainty.

        Returns
        -------
        `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
            Covariance matrix.
        """
        return np.copy(self._R)

    @property
    def system_uncertainty_cov(self) -> np.ndarray:
        """
        Returns the covariance matrix of the system uncertainty.

        Returns
        -------
        `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
            Covariance matrix.
        """
        return np.copy(self._Q)

    @property
    def init_state_uncertainty_cov(self) -> np.ndarray:
        """
        Returns the covariance matrix of the initial state uncertainty.

        Returns
        -------
        `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
            Covariance matrix.
        """
        return np.copy(self._init_state_uncertainty_cov)

    def __eq__(self, other) -> bool:
        return super().__eq__(other) and \
            np.all(self._H == other.measurement_func) and \
            np.all(self._F == other.state_transition_func) and \
            np.all(self._R == other.measurement_uncertainty_cov) and \
            np.all(self._Q == other.system_uncertainty_cov) and \
            np.all(self._init_state == other.init_state) and \
            np.all(self._init_state_uncertainty_cov == other.init_state_uncertainty_cov)

    def __str__(self) -> str:
        return super().__str__() +\
            f" init_state_uncertainty_cov: {self._init_state_uncertainty_cov} " +\
            f"measurement_func: {self._H} state_transition_func: {self._H} " +\
            f"measurement_uncertainty_cov: {self._R} system_uncertainty_cov: {self._Q}"

    def reset(self) -> None:
        super().reset()

        self._P = np.copy(self._init_state_uncertainty_cov)

    def step(self, observation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Predicts the current state (incl. it's uncertainty) based on a given current observation.
        Also, updates all other internal states of the Kalman filter.

        Parameters
        ----------
        observation : `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
            Current observation.

        Returns
        -------
        tuple[`numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_, `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_]
            Tuple of predicted system state and uncertainty covariance matrix.
        """
        if not isinstance(observation, np.ndarray):
            raise TypeError("'observation' must be an instance of 'numpy.ndarray' " +
                            f"but not of '{type(observation)}'")
        if observation.shape != (self._obs_dim,):
            raise ValueError("'observation' must be of shap (obs_dim,) -- " +
                             f"i.e. {(self._obs_dim,)}. But found {observation.shape}")
        # Predict
        self._x = np.dot(self._F, self._x)
        self._P = np.dot(self._F, self._P).dot(self._F.T) + self._Q

        # Update
        y = observation - np.dot(self._H, self._x)
        S = np.dot(self._H, self._P).dot(self._H.T) + self._R
        K = np.dot(self._P, self._H.T).dot(np.linalg.inv(S))
        self._x = self._x + np.dot(K, y)
        self._P = (self._I - np.dot(K, self._H)).dot(self._P)

        return np.copy(self._x), np.copy(self._P)
