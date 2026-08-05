"""
This module contains implementations of various Kalman filters.
"""
from typing import Optional, Callable
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
            raise ValueError("'init_state' must be of shape (state_dim,) -- " +
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


class TimeVaryingKalmanFilter(KalmanFilter):
    """
    Implementation of the time varying Kalman filter -- i.e. transition matrix,
    system uncertainty, and measurement uncertainty depend on time.

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
    state_transition_func : Callable[[int], `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_]
        Function mapping time (integer) to the time dependent state transition function --
        i.e. matrix moving from a given state to the next state.
    init_state_uncertainty_cov : `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_, optional
        Covariance matrix of the initial state uncertainty.
        If None, the identity matrix will be used.

        The default is None.
    measurement_uncertainty_cov : Callable[[int], `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_], optional
        Function mapping time (integer) to the time dependent covariance matrix of the
        measurement/observation uncertainty.
        If None, the identity matrix will be used in all time steps.

        The default is None.
    system_uncertainty_cov : Callable[[int], `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_], optional
        Function mapping time (integer) to the time dependent covariance matrix of the
        system uncertainty.
        If None, the identity matrix will be used in all time steps.

        The default is None.
    """
    def __init__(self, state_dim: int, obs_dim: int, init_state: np.ndarray,
                 measurement_func:  np.ndarray,
                 state_transition_func: Callable[[int], np.ndarray],
                 init_state_uncertainty_cov: Optional[np.ndarray] = None,
                 measurement_uncertainty_cov: Optional[Callable[[int], np.ndarray]] = None,
                 system_uncertainty_cov: Optional[Callable[[int], np.ndarray]] = None) -> None:
        if not callable(state_transition_func):
            raise TypeError("'state_transition_func' must be a function mapping time (int) " +
                            "to the time dependent state transition function/matrix")

        self._t = 0
        self._get_state_transition_func = state_transition_func

        if measurement_uncertainty_cov is None:
            self._get_measurement_uncertainty_cov = lambda _: np.eye(obs_dim)
        else:
            if not callable(measurement_uncertainty_cov):
                raise TypeError("'measurement_uncertainty_cov' must be a function mapping " +
                                "time (int) to the time dependent covariance matrix of " +
                                "the measurement uncertainty.")

            self._get_measurement_uncertainty_cov = measurement_uncertainty_cov

        if system_uncertainty_cov is None:
            self._get_system_uncertainty_cov = lambda _: np.eye(state_dim)
        else:
            if not callable(system_uncertainty_cov):
                raise TypeError("'system_uncertainty_cov' must be a function mapping time (int) " +
                                "to the time dependent covariance matrix of " +
                                "the system uncertainty.")

            self._get_system_uncertainty_cov = system_uncertainty_cov

        super().__init__(state_dim=state_dim, obs_dim=obs_dim, init_state=init_state,
                         measurement_func=measurement_func(0),
                         state_transition_func=self._get_state_transition_func(0),
                         init_state_uncertainty_cov=init_state_uncertainty_cov,
                         measurement_uncertainty_cov=self._get_measurement_uncertainty_cov(0),
                         system_uncertainty_cov=self._get_system_uncertainty_cov(0))

    def reset(self) -> None:
        self._t = 0

        super().reset()

    def step(self, observation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self._F = self._get_state_transition_func(self._t)
        self._R = self._get_measurement_uncertainty_cov(self._t)
        self._Q = self._get_system_uncertainty_cov(self._t)
        self._t += 1

        return super().step(observation)


class ExtendedKalmanFilter(KalmanFilterBase):
    """
    Class implementing the extended multivariate Kalman filter.

    Parameters
    ----------
    state_dim : `int`
        Dimensionality of states.
    obs_dim : `int`
        Dimensionality of observations.
    init_state : `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
        Initial state.
    measurement_func : Callable[[`numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_], `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_]
        Measurement function -- i.e. a function for mapping a system state to an observation.
    measurement_func_grad : Callable[[`numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_], `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_]
        Gradient/Jacobian (w.r.t. a system state) of the measurement function.
    state_transition_func : Callable[[`numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_], `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_]
        State transition function -- i.e. a function evolving a system state to the next time step.
    state_transition_func_grad : Callable[[`numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_], `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_]
        Gradient/Jacobian (w.r.t. a system state) of the state transition function.
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
                 measurement_func: Callable[[np.ndarray], np.ndarray],
                 measurement_func_grad: Callable[[np.ndarray], np.ndarray],
                 state_transition_func: Callable[[np.ndarray], np.ndarray],
                 state_transition_func_grad: Callable[[np.ndarray], np.ndarray],
                 init_state_uncertainty_cov: Optional[np.ndarray] = None,
                 measurement_uncertainty_cov: Optional[np.ndarray] = None,
                 system_uncertainty_cov: Optional[np.ndarray] = None):
        super().__init__(state_dim=state_dim, obs_dim=obs_dim, init_state=init_state)

        if not callable(measurement_func):
            raise TypeError("'measurement_func' must be callable -- i.e. mapping a given " +
                            "system state (numpy.ndarray) to an observation (numpy.ndarray)")
        if not callable(measurement_func_grad):
            raise TypeError("'measurement_func_grad' must be callable -- i.e. computing the " +
                            "gradient/Jacobian ((numpy.ndarray)) of 'measurement_func' " +
                            "w.r.t. to a given system state")
        if not callable(state_transition_func):
            raise TypeError("'state_transition_func' must be callable -- i.e. evolving a given " +
                            "system state (numpy.ndarray) for one time step")
        if not callable(state_transition_func_grad):
            raise TypeError("'state_transition_func_grad' must be callable -- i.e. computing the " +
                            "gradient/Jacobian (numpy.ndarray) of 'state_transition_func' " +
                            "w.r.t. to a given system state")

        self._measurement_func = measurement_func
        self._measurement_func_grad = measurement_func_grad
        self._state_transition_func = state_transition_func
        self._state_transition_func_grad = state_transition_func_grad

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
    def measurement_func(self) -> Callable[[np.ndarray], np.ndarray]:
        """
        Returns the measurement function -- i.e. a function for mapping a
        system state to an observation.

        Returns
        -------
        Callable[[`numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_], `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_]
            Measurement function.
        """
        return self._measurement_func

    @property
    def measurement_func_grad(self) -> Callable[[np.ndarray], np.ndarray]:
        """
        Returns the gradient/Jacobian (w.r.t. a system state) of the measurement function.

        Returns
        -------
        Callable[[`numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_], `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_]
            Gradient/Jacobian of the the measurement function.
        """
        return self._measurement_func_grad

    @property
    def state_transition_func(self) -> Callable[[np.ndarray], np.ndarray]:
        """
        Returns the state transition function -- i.e. a function evolving a
        system state to the next time step.

        Returns
        -------
        Callable[[`numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_], `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_]
            State transition function.
        """
        return self._state_transition_func

    @property
    def state_transition_func_grad(self) -> Callable[[np.ndarray], np.ndarray]:
        """
        Returns the gradient/Jacobian (w.r.t. a system state) of the state transition function.

        Returns
        -------
        Callable[[`numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_], `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_]
            Gradient of the state transition function.
        """
        return self._state_transition_func_grad

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
            self._measurement_func == other.measurement_func and \
            self._measurement_func_grad == other.measurement_func_grad and \
            self._state_transition_func == other.state_transition_func and \
            self._state_transition_func_grad == other.state_transition_func_grad and \
            np.all(self._R == other.measurement_uncertainty_cov) and \
            np.all(self._Q == other.system_uncertainty_cov) and \
            np.all(self._init_state == other.init_state) and \
            np.all(self._init_state_uncertainty_cov == other.init_state_uncertainty_cov)

    def __str__(self) -> str:
        return super().__str__() +\
            f" init_state_uncertainty_cov: {self._init_state_uncertainty_cov} " +\
            f"measurement_func: {self._measurement_func} " +\
            f"measurement_func_grad: {self._measurement_func_grad} " +\
            f"state_transition_func: {self._state_transition_func} " +\
            f"state_transition_func_grad: {self._state_transition_func_grad} " +\
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
            raise ValueError("'observation' must be of shape (obs_dim,) -- " +
                             f"i.e. {(self._obs_dim,)}. But found {observation.shape}")
        # Predict
        F = self._state_transition_func_grad(self._x)
        self._x = self._state_transition_func(self._x)
        self._P = np.dot(F, self._P).dot(F.T) + self._Q

        # Update
        H = self._measurement_func_grad(self._x)
        y = observation - self._measurement_func(self._x)
        S = np.dot(H, self._P).dot(H.T) + self._R
        K = np.dot(self._P, H.T).dot(np.linalg.inv(S))
        self._x = self._x + np.dot(K, y)
        self._P = (self._I - np.dot(K, H)).dot(self._P)

        return np.copy(self._x), np.copy(self._P)


class TimeVaryingExtendedKalmanFilter(ExtendedKalmanFilter):
    """
    Implementing the timevarying extended multivariate Kalman filter -- i.e.
    state transition and measurement functions may depend on time.

    Parameters
    ----------
    state_dim : `int`
        Dimensionality of states.
    obs_dim : `int`
        Dimensionality of observations.
    init_state : `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
        Initial state.
    get_state_transition_func : `Callable[[int], Callable[[numpy.ndarray], numpy.ndarray]]`
        A function that maps time to the curresponding state transition function
        (a function evolving a given system state to the next time step).
    get_state_transition_func_grad : `Callable[[int], Callable[[numpy.ndarray], numpy.ndarray]]`
        A function that maps time to the curresponding function for computing the gradient/Jacobian
        (w.r.t. a system state) of the state transition function.
    get_measurement_func_grad : `Callable[[int], Callable[[numpy.ndarray], numpy.ndarray]]`
        A function that maps time to the curresponding function for computing the gradient/Jacobian
        (w.r.t. a system state) of the measurement function.
    get_measurement_func : `Callable[[int], Callable[[numpy.ndarray], numpy.ndarray]]`
        A function that maps time to the curresponding measurement function
        (a function for mapping a system state to an observation).
    """
    def __init__(self, state_dim: int, obs_dim: int, init_state: np.ndarray,
                 get_state_transition_func: Callable[[int], Callable[[np.ndarray], np.ndarray]],
                 get_state_transition_func_grad: Callable[[int], Callable[[np.ndarray], np.ndarray]],
                 get_measurement_func_grad: Callable[[int], Callable[[np.ndarray], np.ndarray]],
                 get_measurement_func: Callable[[int], Callable[[np.ndarray], np.ndarray]], **kwds):
        if not callable(get_state_transition_func):
            raise TypeError("'get_state_transition_func' must be callable -- i.e. mapping " +
                            "time to the curresponding state transition function")
        if not callable(get_state_transition_func_grad):
            raise TypeError("'get_state_transition_func_grad' must be callable -- i.e. mapping " +
                            "time to the curresponding function for computing the " +
                            "gradient/Jacobian of the state trasition function")
        if not callable(get_measurement_func_grad):
            raise TypeError("'get_measurement_func_grad' must be callable -- i.e. mapping " +
                            "time to the curresponding function for computing the " +
                            "gradient/Jacobian of the measurement function")
        if not callable(get_measurement_func):
            raise TypeError("'get_measurement_func' must be callable -- i.e. mapping " +
                            "time to the curresponding measurement function")

        self._get_state_transition_func = get_state_transition_func
        self._get_state_transition_func_grad = get_state_transition_func_grad
        self._get_measurement_func_grad = get_measurement_func_grad
        self._get_measurement_func = get_measurement_func

        self._t = 0

        super().__init__(state_dim=state_dim, obs_dim=obs_dim, init_state=init_state,
                         state_transition_func=self._get_state_transition_func(self._t),
                         state_transition_func_grad=self._get_state_transition_func_grad(self._t),
                         measurement_func=self._get_measurement_func(self._t),
                         measurement_func_grad=self._get_measurement_func_grad(self._t), **kwds)

    def reset(self) -> None:
        self._t = 0

        super().reset()

    def step(self, observation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self._state_transition_func_grad = self._get_state_transition_func_grad(self._t)
        self._state_transition_func = self._get_state_transition_func(self._t)
        self._measurement_func_grad = self._get_measurement_func_grad(self._t)
        self._measurement_func = self._get_measurement_func(self._t)
        self._t += 1

        return super().step(observation)
    

class EnsembleKalmanFilter(KalmanFilterBase):
    """
    Class implementing the Ensemble Kalman Filter (EnKF).

    https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/Appendix-E-Ensemble-Kalman-Filters.ipynb
    https://towardsdatascience.com/addressing-the-butterfly-effect-data-assimilation-using-ensemble-kalman-filter-9883d0e1197b/
    https://www.math.umd.edu/~slud/RITF17/enkf-tutorial.pdf


    Parameters
    ----------
    state_dim : `int`
        Dimensionality of states.
    obs_dim : `int`
        Dimensionality of observations.
    init_state : `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
        Initial state.
    measurement_func : `Callable[[numpy.ndarray], numpy.ndarray]`
        Measurement function -- i.e. a function mapping a single state vector to an observation.
    state_transition_func : `Callable[[numpy.ndarray], numpy.ndarray]`
        State transition function -- i.e. a function avoiding a single state vector to the next time step.
    ensemble_size : `int`, optional
        Number of ensemble members used for representing the state distribution. 

        The default is 100.
    seed : `int`, optional
        It can be set for reproducibility. 

        The default is None.
    init_state_uncertainty_cov : `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_, optional
        Covariance matrix of the initial state uncertainty. - used for sampling the initial ensemble around `init_state`
        If None, the identity matrix will be used.

        The default is None.
    measurement_uncertainty_cov : `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_, optional
        Covariance matrix of the measurement/observation uncertainty.
        If None, the identity matrix will be used.

        The default is None.
    system_uncertainty_cov : `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_, optional
        Covariance matrix of the system uncertainty (process noise) -- added to every ensemble member after the state transition.
        If None, the identity matrix will be used.

        The default is None.

    """
    def __init__(self, state_dim: int, obs_dim: int, init_state: np.ndarray,
                 measurement_func: Callable[[np.ndarray], np.ndarray],
                 state_transition_func: Callable[[np.ndarray], np.ndarray],
                 ensemble_size: int = 100,
                 seed: int = None,
                 init_state_uncertainty_cov: Optional[np.ndarray] = None,
                 measurement_uncertainty_cov: Optional[np.ndarray] = None,
                 system_uncertainty_cov: Optional[np.ndarray] = None):
        super().__init__(state_dim=state_dim, obs_dim=obs_dim, init_state=init_state)


        # TODO


        if not callable(measurement_func):
            raise TypeError("'measurement_func' must be callable -- i.e. mapping a given " +
                            "system state (numpy.ndarray) to an observation (numpy.ndarray)")
        
        if not callable(state_transition_func):
            raise TypeError("'state_transition_func' must be callable -- i.e. evolving a given " +
                            "system state (numpy.ndarray) for one time step")
        
        if not isinstance(ensemble_size, int):
            raise TypeError("'ensemble_size' must be an instance of 'int' " + 
                            f"but not of '{type(ensemble_size)}'")
        
        if ensemble_size <= 0:
            raise ValueError("'ensemble_size' must be > 0")
        
        if init_state_uncertainty_cov is not None:
            if not isinstance(init_state_uncertainty_cov, np.ndarray):
                raise TypeError("'init_state_uncertainty_cov' must be an instance of " + 
                                "'numpy.ndarray' but not of" +
                                f"'{type(init_state_uncertainty_cov)}'")
            if init_state_uncertainty_cov.shape != (state_dim, state_dim):
                raise ValueError("'init_state_uncertainty_cov' must be of shape " + 
                                 f"(state_dim, state_dim) -- i.e. {(state_dim, state_dim)}." + 
                                 f"But found {init_state_uncertainty_cov.shape}")
            
        if measurement_uncertainty_cov is not None:
            if not isinstance(measurement_uncertainty_cov, np.ndarray):
                raise TypeError("'measurement_uncertainty_cov' must be an instance of " + 
                                "'numpy.ndarray' but not of" +
                                f"'{type(measurement_uncertainty_cov)}'")
            if measurement_uncertainty_cov.shape != (obs_dim, obs_dim):
                raise ValueError("'measurement_uncertainty_cov' must be of shape " + 
                                 f"(obs_dim, obs_dim) -- i.e. {(obs_dim, obs_dim)}." + 
                                 f"But found {measurement_uncertainty_cov.shape}")
            
        if system_uncertainty_cov is not None:
            if not isinstance(system_uncertainty_cov, np.ndarray):
                raise TypeError("'system_uncertainty_cov' must be an instance of " + 
                                "'numpy.ndarray' but not of" +
                                f"'{type(system_uncertainty_cov)}'")
            if system_uncertainty_cov.shape != (state_dim, state_dim):
                raise ValueError("'system_uncertainty_cov' must be of shape " + 
                                 f"(state_dim, state_dim) -- i.e. {(state_dim, state_dim)}." + 
                                 f"But found {system_uncertainty_cov.shape}")
            
        self._measurement_func = measurement_func
        self._state_transition_func = state_transition_func
        self._N = ensemble_size
        
        if init_state_uncertainty_cov is None:
            self._P = np.eye(state_dim)
        else: 
            self._P = init_state_uncertainty_cov

        if measurement_uncertainty_cov is None: 
            self._R = np.eye(obs_dim)
        else: 
            self._R = measurement_uncertainty_cov
        
        if system_uncertainty_cov is None: 
            self._Q = np.eye(state_dim)

        else: 
            self._Q = system_uncertainty_cov

        self._init_state_uncertainty_cov = np.copy(self._P)

        if seed is None:
            self._rng = np.random.default_rng()
        else: 
            self._rng = np.random.default_rng(seed)

        # draw the initial ensemble by sampling around the initial state
        self._ensemble = self._x + self._rng.multivariate_normal(
            mean = np.zeros(self._state_dim), cov=self._P, size=self._N
        )


    @property
    def measurement_func(self) -> Callable[[np.ndarray], np.ndarray]:
        """
        Returns the measurement function -- i.e. a function for mapping a
        system state to an observation.

        Returns
        -------
        Callable[[`numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_], `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_]
            Measurement function.
        """
        return self._measurement_func
    
    @property
    def state_transition_func(self) -> Callable[[np.ndarray], np.ndarray]:
        """
        Returns the state transition function -- i.e. a function evolving a
        system state to the next time step.

        Returns
        -------
        Callable[[`numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_], `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_]
            State transition function.
        """
        return self._state_transition_func
    
    @property
    def ensemble_size(self) -> int:
        """
        Returns the number of ensemble members.

        Returns
        -------
        `int`
            Ensemble size. 
        """
        return self._N
    
    @property
    def ensemble(self) -> np.ndarray:
        """
        Returns the current ensemble -- shape (ensemble_size, state_dim).

        Returns
        -------
        `numpy.ndarray`
            Ensemble members. 
        """
        return np.copy(self._ensemble)
    
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
            self._measurement_func == other.measurement_func and \
            self._state_transition_func == other.state_transition_func and \
            self._N == other.ensemble_size and \
            np.all(self._R == other.measurement_uncertainty_cov) and \
            np.all(self._Q == other.system_uncertainty_cov) and \
            np.all(self._init_state_uncertainty_cov == other.init_state_uncertainty_cov)

    def __str__(self) -> str:
        return super().__str__() +\
            f" init_state_uncertainty_cov: {self._init_state_uncertainty_cov} " +\
            f"measurement_func: {self._measurement_func} " +\
            f"state_transition_func: {self._state_transition_func} " +\
            f"ensemble_size: {self._N} " +\
            f"measurement_uncertainty_cov: {self._R} system_uncertainty_cov: {self._Q}"


        
    def reset(self) -> None:
        super().reset()

        self._P = np.copy(self._init_state_uncertainty_cov)

        # re-draw the ensemble around the (reset) initial state
        self._ensemble = self._x + self._rng.multivariate_normal(
            mean=np.zeros(self._state_dim), cov=self._P, size=self._N)

    def step(self, observation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Predicts the current state (incl. its uncertainty) based on a given current observation.
        Also, updates all other internal states of the Kalman filter.

        Parameters
        ----------
        observation : `numpy.ndarray`
            Current observation. 

        Returns
        -------
        tuple[`numpy.ndarray`, `numpy.ndarray`]
            Tuple of predicted system state and uncertainty covariance matrix.
        """
        if not isinstance(observation, np.ndarray):
            raise TypeError("'observation' must be an instance of 'numpy.ndarray' " +
                            f"but not of '{type(observation)}'")
        if observation.shape != (self._obs_dim,):
            raise ValueError("'observation' must be of shape (obs_dim,) -- " +
                                f"i.e. {(self._obs_dim,)}. But found {observation.shape}")

        # predict step: propagate every ensemble member through the state transition function
        # add process noise ~ N(0, Q)
        self._ensemble = np.array([self._state_transition_func(member) 
                                   for member in self._ensemble])
        self._ensemble += self._rng.multivariate_normal(
            mean=np.zeros(self._state_dim), cov=self._Q, size=self._N
        )

        # prior (forecast) mean and covariance 
        self._x = np.mean(self._ensemble, axis=0)
        self._P = np.cov(self._ensemble, rowvar=False)

        # map every ensemble into observation space: z_i = h(x_i) 
        Z = np.array([self._measurement_func(member) for member in self._ensemble])
        z_mean = np.mean(Z, axis=0)

        # perturbation
        perturbed_obs = observation + self._rng.multivariate_normal(
            mean=np.zeros(self._obs_dim), cov=self._R, size=self._N
        )

        # estimate cross-covariance 
        X_dev = self._ensemble - self._x
        Z_dev = Z - z_mean
        P_xz = X_dev.T.dot(Z_dev) / (self._N - 1)
        P_zz = Z_dev.T.dot(Z_dev) / (self._N - 1) + self._R

        # Kalman gain 
        K = P_xz.dot(np.linalg.inv(P_zz))

        # update ensemble members
        innovations = perturbed_obs - Z
        self._ensemble = self._ensemble + innovations.dot(K.T)

        # posterior
        self._x = np.mean(self._ensemble, axis=0)
        self._P = np.cov(self._ensemble, rowvar=False)

        return np.copy(self._x), np.copy(self._P)

