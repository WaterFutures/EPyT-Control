import numpy as np 
from epyt_control.signal_processing.state_estimation import kalman_filters
from epyt_control.signal_processing.state_estimation import EnsembleKalmanFilter


def test_ensemble_kalman_filter():
    rng = np.random.default_rng(42)

    state_dim = 2
    obs_dim = 1


    # simple linear system 
    def state_transition_func(x: np.ndarray) -> np.ndarray:
        return x 

    def measurement_func(x: np.ndarray) -> np.ndarray:
        return np.array([x[0]])

    true_state = np.array([3.0, -1.0])
    init_state = np.array([0.0, 0.0])

    ekf = EnsembleKalmanFilter(
        state_dim=state_dim,
        obs_dim=obs_dim,
        init_state=init_state,
        measurement_func=measurement_func,
        state_transition_func=state_transition_func,
        ensemble_size=200,
        init_state_uncertainty_cov=np.eye(state_dim) * 1.0,
        measurement_uncertainty_cov=np.eye(obs_dim) * 0.01,
        system_uncertainty_cov=np.eye(state_dim) * 1e-4,
    )

    # shape and type checks
    assert ekf.state_dim == state_dim
    assert ekf.obs_dim == obs_dim
    assert ekf.ensemble_size == 200
    assert ekf.ensemble.shape == (200, state_dim)
    assert np.allclose(np.mean(ekf.ensemble, axis=0), init_state, atol=0.5)

    # run several steps 
    last_x, last_P = None, None
    for _ in range(50):
        noisy_obs = np.array([true_state[0]]) + rng.normal(scale=0.05, size=obs_dim)
        x, P = ekf.step(noisy_obs)

        assert isinstance(x, np.ndarray) and x.shape == (state_dim,)
        assert isinstance(P, np.ndarray) and P.shape == (state_dim, state_dim)
        assert np.all(np.isfinite(x))
        assert np.all(np.isfinite(P))

        last_x, last_P = x, P

    # state estimate for the observed component should be close to the true value 
    assert abs(last_x[0] - true_state[0]) < 0.2 

    # uncertainty should shrink 
    assert last_P[0,0] < ekf.init_state_uncertainty_cov[0,0]

    # ensemble mean should match returned state estimate 
    assert np.allclose(np.mean(ekf.ensemble, axis=0), last_x, atol=1e-8)

    # reset() should restore ensemble around the original init_state
    ekf.reset()
    assert np.allclose(ekf.init_state, init_state)
    assert ekf.ensemble.shape == (200, state_dim)
    assert np.allclose(np.mean(ekf.ensemble, axis=0), init_state, atol=0.5)

    # input validation 
    try:
        ekf.step(np.array([1.0, 2.0])) # wrong 
        assert False, "expected ValueError for wrong observation shape"
    except ValueError:
        pass

    try:
        ekf.step([1.0]) # not a numpy array
        assert False, "expected TypeError for non-ndarray observation"
    except TypeError:
        pass

if __name__ == "__main__":
    test_ensemble_kalman_filter()



