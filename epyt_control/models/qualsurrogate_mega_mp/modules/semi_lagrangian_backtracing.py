import torch
import numpy as np
from numba import njit, prange


def compute_backward_transit_times(l, v, dt):
    """
    Computes the backward transit time for each edge given a time series
    of flow velocities using PyTorch.
    
    Parameters:
    l, shape: [E]
        Edge lengths
    v, shape: [E, T]
        Tensor of velocity values
    dt, float: 
        Time step
    
    Returns:
    Tensor of backward transit times, shape: [E, T]
    """
    B, T = v.shape[:2]  # Number of time steps
    time_indices = torch.arange(T, device=v.device, dtype=torch.float32) * dt  # Time array

    l = l.unsqueeze(1).unsqueeze(1)

    if True: # This is one way to compute the local distance traveled, but its inefficient
        # Compute distances traveled for all backward time intervals
        v_matrix = torch.tril(v[:,None].repeat(1, T, 1))  # Upper triangular velocity matrix
        dx_matrix = v_matrix * dt  # Distance matrix
        # Compute cumulative distances by summing backward
        x_traveled = torch.flip(torch.cumsum(torch.flip(dx_matrix, dims=[-1]), dim=-1), dims=[-1])

    # Find where the traveled distance reaches the pipe length
    exit_mask = x_traveled.abs() >= l  # Boolean mask where travel distance >= l
    
    # Find first time each row reaches x = l
    first_reach_idx = (exit_mask.flip([-1]).cumsum(-1).flip([-1]).float().argmin(-1) - 1).clamp(0)# [Fixes]
    first_reach_idx_full = torch.stack((
        torch.arange(B, device=v.device).repeat_interleave(T), 
        torch.arange(T, device=v.device).repeat(B), 
        first_reach_idx.view(-1)
    ), dim=0)
    
    # Interpolate for sub-time-step accuracy
    dx_last = dx_matrix[(*first_reach_idx_full,)].view(B, -1)  # Distance at last step
    excess = ((x_traveled.abs() - l))[(*first_reach_idx_full,)].view(B, -1) # [Fixes]
    frac = excess.abs() / dx_last.abs()  # Fractional correction
    # Assert: frac has to be 0 <= |frac| <= 1.
    # assert (dx_last.abs() >= excess.abs())[exit_mask.any(-1)].all()
    frac = torch.nan_to_num(frac, posinf=0, neginf=0)
    
    # Special Case (selfloops): 
    # First outflow of a node A, then flow direction inverts, then inflow into node A)
    # This happens at intervals where the velocity integral becomes zero.
    # We find this by checking where the sign of the integral changes
    backflow_mask = torch.nn.functional.pad(
        x_traveled.sign()[...,1:] != x_traveled.sign()[...,:-1], (0,1)
    )
    backflow_mask*= ~torch.eye(T, dtype=bool) # Remove diagonal
    backflow_reach_idx = backflow_mask.flip([-1]).cumsum(-1).flip([-1]).float().argmin(-1) - 1
    backflow_reach_idx = backflow_reach_idx.clamp(0, T-1)
    backflow_reach_idx_full = torch.stack((
        torch.arange(B, device=v.device).repeat_interleave(T), 
        torch.arange(T, device=v.device).repeat(B),
        backflow_reach_idx.view(-1)
    ), dim=0)
    bf_dx_last = dx_matrix[(*backflow_reach_idx_full,)].view(B, -1)
    bf_excess = x_traveled[(*backflow_reach_idx_full,)].view(B, -1)
    bf_frac = bf_excess.abs() / bf_dx_last.abs()
    # Assert: bf_frac has to be 0 <= |frac| <= 1.
    assert (bf_dx_last.abs() >= bf_excess.abs())[backflow_mask.any(-1)].all()
    bf_frac = torch.nan_to_num(bf_frac, posinf=0, neginf=0)

    selfloop_mask = (first_reach_idx + (1 - frac)) < (backflow_reach_idx + (1 - bf_frac))
    first_reach_idx = torch.maximum(first_reach_idx, backflow_reach_idx)
    frac = torch.where(selfloop_mask, bf_frac, frac)
    Tb = time_indices[torch.arange(T, device=v.device) - first_reach_idx] + dt * (1 - frac)
    Tb = (torch.arange(T, device=v.device) - first_reach_idx + (1 - frac)) * dt
    
    # Set values to NaN where the pipe isn't fully traversed
    #Tb[~exit_mask.any(dim=1)] => float('nan')
    integral_sign = x_traveled[(*first_reach_idx_full,)].view(B, -1).sign()
    integral_sign[selfloop_mask] = x_traveled[(*backflow_reach_idx_full,)].view(B, -1)[selfloop_mask].sign()

    return (Tb.clamp(min=-(T*dt), max=(T*dt))) * integral_sign, selfloop_mask #* v_sign#.clamp(0) * v_sign

# The above function to compute the laplacian backward tracing is fast but has
# memory complexity of O(E x T^2), which is prohibitive for long time series
# and graphs with a lot of edges. The below code computes the backward tracing
# in parallel for each edge and is comparably fast.

@njit(parallel=True, fastmath=True, cache=True)
def compute_deltas_cpu(cumulative, l, delta_ts, delta_xs=None, start_t=None):
    B, T = cumulative.shape
    if start_t is None:
        start_t = 0
        
    for b in prange(B):
        if l[b] != 0:
            for t in range(start_t, T):
                sp = (cumulative[b, t] - cumulative[b, t-1])
                target = (cumulative[b, t] - l[b])
                target2 = (cumulative[b, t] + l[b])
                
                for x in range(t, -1, -1):

                    if cumulative[b, x] <= target:
                        if x < t:
                            c1, c2 = cumulative[b, x], cumulative[b, x+1]
                            if c2 != c1:
                                alpha = np.abs((target - c1) / (c2 - c1))
                                interpolated_x = x + alpha
                            else:
                                interpolated_x = x
                        else:
                            interpolated_x = x
                        delta_ts[b, t] = t - interpolated_x # + 1
                        break
                    elif cumulative[b, x] >= target2:
                        if x < t:
                            c1, c2 = cumulative[b, x], cumulative[b, x + 1]
                            if c2 != c1:
                                alpha = (target2 - c1) / (c2 - c1)
                                interpolated_x = x + alpha
                            else:
                                interpolated_x = x
                        else:
                            interpolated_x = x
                        delta_ts[b, t] = -t + interpolated_x
                        break
                else:
                    if delta_xs is not None:
                        dx = (cumulative[b, t] - cumulative[b, x])
                        delta_xs[b, t] = dx / l[b]
        else:
            for t in range(start_t,  T):
                for x in range(t, 0, -1):
                    c1, c2 = cumulative[b, x-1:x+1] - cumulative[b, t]
                    
                    # This case is because sign includes the zero, if omitted 
                    # the loop breaks at the first iteration 
                    # (maybe we can skip the first interation completely?)
                    if x == t:
                        if c1 == c2:
                            delta_ts[b, t] = 1 # if c1 > 0 else -1
                            break
                        continue

                    # print(c1, c2, (c2 - c1), cumulative[b, t])
                    if c2 == 0:
                        #delta_ts[b, t] = -2
                        #break
                        continue
                    
                    if np.sign(c1) != np.sign(c2):
                        alpha = c2 / (c2 - c1)
                        interpolated_x = x - alpha
                        #print(t, '\t', np.sign(c2) * (t - interpolated_x)
                        #      )
                        if c2 > 0:
                            sign = 1
                        elif c2 == 0:
                            sign = np.sign(c1)
                        else:
                            sign = -1
                        delta_ts[b, t] = sign * (t - interpolated_x)
                        break


def compute_backward_transit_times_fast(l, v, dt, start_t=None, return_sl=True):
    """
    Computes the backward transit time for each edge given a time series
    of flow velocities using njit.
    
    Parameters:
    l, shape: [E]
        Edge lengths
    v, shape: [E, T]
        Tensor of velocity values
    dt, float: 
        Time step
    
    Returns:
    Tensor of backward transit times, shape: [E, T]
    """
    v = np.asarray(v, dtype=np.float32)
    l = np.asarray(l, dtype=np.float32)
    B, T = v.shape

    cumulative = np.zeros((B, T+1), dtype=np.float32)
    cumulative[:, 1:] = np.cumsum(v * dt, axis=1)

    # Important: If we initialize delta_ts with NaNs, then we assume that
    # Information is traceable until the end of the domain. Wherever it is not
    # (e.g. no path from 0 to L) selfloops won't be found as information will
    # not be available anyways. Alternatively: initialize with T+1
    
    delta_ts = np.full((B, T+1), np.nan, dtype=np.float32) # Initialize with NaN or with T+1
    delta_xs = np.full((B, T+1), np.nan, dtype=np.float32)
    compute_deltas_cpu(cumulative, l, delta_ts, delta_xs=delta_xs, start_t=start_t)

    if not return_sl:
        if start_t is not None:
            delta_ts = delta_ts[:,start_t-1:start_t+1]
        return delta_ts[:, 1:] * dt

    delta_ts_zeros = np.full((B, T+1), np.nan, dtype=np.float32)
    compute_deltas_cpu(cumulative, l * 0, delta_ts_zeros, start_t=start_t)
    
    sl_mask = np.abs(delta_ts_zeros) < np.nan_to_num(np.abs(delta_ts), nan=np.inf)
    
    transit_steps = np.where(sl_mask, delta_ts_zeros, delta_ts)
    transit_times = transit_steps * dt

    if start_t is not None:
        transit_times = transit_times[:,start_t-1:start_t+1]
        sl_mask = sl_mask[:,start_t-1:start_t+1]
        delta_xs = delta_xs[:,start_t-1:start_t+1]
    
    return transit_times[:, 1:], sl_mask[:, 1:], delta_xs[:, 1:]