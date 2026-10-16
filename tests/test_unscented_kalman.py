import numpy as np
from epyt_control.signal_processing.state_estimation import kalman_filters
from epyt_control.signal_processing.state_estimation import UnscentedKalmanFilter


def test_unscented_kalman_filter():
    rng = np.random.default_rng(42)  # only used to generate synthetic noisy observations

    state_dim = 2
    obs_dim = 1

    # simple linear system: x_{k+1} = x_k (identity), z_k = x_k[0]
    def state_transition_func(x: np.ndarray) -> np.ndarray:
        return x  # static state 

    def measurement_func(x: np.ndarray) -> np.ndarray:
        return np.array([x[0]])

    true_state = np.array([3.0, -1.0])
    init_state = np.array([0.0, 0.0])

    ukf = UnscentedKalmanFilter(
        state_dim=state_dim,
        obs_dim=obs_dim,
        init_state=init_state,
        measurement_func=measurement_func,
        state_transition_func=state_transition_func,
        alpha=0.1,
        beta=2.0,
        init_state_uncertainty_cov=np.eye(state_dim) * 1.0,
        measurement_uncertainty_cov=np.eye(obs_dim) * 0.01,
        system_uncertainty_cov=np.eye(state_dim) * 1e-4,
    )

    # basic shape / type / parameter checks right after construction 
    assert ukf.state_dim == state_dim
    assert ukf.obs_dim == obs_dim
    assert ukf.alpha == 0.1
    assert ukf.beta == 2.0
    assert ukf.kappa == 3 - state_dim  # default heuristic
    assert np.allclose(ukf.init_state, init_state)

    # determinism check: same input state/cov must give identical sigma points
    sigmas_a = ukf._generate_sigma_points(init_state, ukf.init_state_uncertainty_cov)
    sigmas_b = ukf._generate_sigma_points(init_state, ukf.init_state_uncertainty_cov)
    assert sigmas_a.shape == (2 * state_dim + 1, state_dim)
    assert np.array_equal(sigmas_a, sigmas_b)

    #run several steps feeding noisy observations of true_state[0] 
    last_x, last_P = None, None
    for _ in range(50):
        noisy_obs = np.array([true_state[0]]) + rng.normal(scale=0.05, size=obs_dim)
        x, P = ukf.step(noisy_obs)

        assert isinstance(x, np.ndarray) and x.shape == (state_dim,)
        assert isinstance(P, np.ndarray) and P.shape == (state_dim, state_dim)
        assert np.all(np.isfinite(x))
        assert np.all(np.isfinite(P))

        last_x, last_P = x, P

    #state estimate for the observed component should converge close to the truth
    assert abs(last_x[0] - true_state[0]) < 0.2

    # Uncertainty should shrink from the initial covariance after repeated updates
    assert last_P[0, 0] < ukf.init_state_uncertainty_cov[0, 0]

    # reproducibility check: two independently constructed filters fed the
    # exact same observation sequence must produce identical results (no RNG
    # inside the filter itself)
    rng_repeat = np.random.default_rng(42)
    ukf_repeat = UnscentedKalmanFilter(
        state_dim=state_dim,
        obs_dim=obs_dim,
        init_state=init_state,
        measurement_func=measurement_func,
        state_transition_func=state_transition_func,
        alpha=0.1,
        beta=2.0,
        init_state_uncertainty_cov=np.eye(state_dim) * 1.0,
        measurement_uncertainty_cov=np.eye(obs_dim) * 0.01,
        system_uncertainty_cov=np.eye(state_dim) * 1e-4,
    )
    x_repeat, P_repeat = None, None
    for _ in range(50):
        noisy_obs = np.array([true_state[0]]) + rng_repeat.normal(scale=0.05, size=obs_dim)
        x_repeat, P_repeat = ukf_repeat.step(noisy_obs)

    assert np.allclose(x_repeat, last_x)
    assert np.allclose(P_repeat, last_P)

    # reset() should restore state/covariance to the original init_state 
    ukf.reset()
    assert np.allclose(ukf.init_state, init_state)
    assert np.allclose(ukf._x, init_state)
    assert np.allclose(ukf._P, ukf.init_state_uncertainty_cov)

    # input validation 
    try:
        ukf.step(np.array([1.0, 2.0]))  # wrong obs_dim
        assert False, "expected ValueError for wrong observation shape"
    except ValueError:
        pass

    try:
        ukf.step([1.0])  # not a numpy array
        assert False, "expected TypeError for non-ndarray observation"
    except TypeError:
        pass

    print("All UnscentedKalmanFilter tests passed.")


if __name__ == "__main__":
    test_unscented_kalman_filter()