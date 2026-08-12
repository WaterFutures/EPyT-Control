"""
This module provides an implementation of the physics informed graph neural network for a
hydraulic surrogate model as proposed by Ashraf el al. (2024 - 2025).
"""

from __future__ import annotations

import datetime
import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import networkx as nx
import numpy as np
import torch
from torch.nn import L1Loss, Module, ModuleList
from torch.optim import Adam, lr_scheduler
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn.dense.linear import Linear
from torch_scatter import scatter
from tqdm import tqdm


from epyt_flow.simulation import ScenarioSimulator, ScadaData, EpanetConstants
from epyt_flow.utils import download_if_necessary, get_temp_folder
from epyt_flow.data.networks import (
    load_anytown,  load_hanoi, load_ltown_a,
)


# Map of built-in EPytFlow network names -> loader functions
EPYTFLOW_NETWORKS_PIGNNHYDSURROGATE: Dict[str, Any] = {}
try:
    EPYTFLOW_NETWORKS_PIGNNHYDSURROGATE = {
        "anytown": load_anytown,
        "hanoi": load_hanoi, "ltown_a": load_ltown_a,
    }
except NameError:
    pass


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# GNN Layers
# ---------------------------------------------------------------------------

class MLP(torch.nn.Sequential):
    """MLP with optional activation and dropout between hidden layers."""

    def __init__(self, dims: List[int], bias: bool = True, dropout: float = 0.0, activ=None):
        layers: list = []
        for i in range(1, len(dims)):
            layers.append(Linear(dims[i - 1], dims[i], bias=bias))
            if i < len(dims) - 1:
                if activ is not None:
                    layers.append(activ)
                layers.append(torch.nn.Dropout(dropout))
        super().__init__(*layers)


class SGNN_Layer(torch.nn.Module):
    """Graph Neural Network layer using edge-based message passing with residual."""

    def __init__(self, edge_dim, aggr="max", bias=False, **kwargs):
        super().__init__()
        self.aggr = aggr
        edge_dims = [3 * edge_dim, 2 * edge_dim, edge_dim]
        self.mlp_edges = MLP(edge_dims, bias=bias, activ=torch.nn.ReLU())

    def forward(self, g, edge_index, z):
        sndr = g[edge_index[0, :], :]
        rcvr = g[edge_index[1, :], :]
        m_e = torch.relu(z + self.mlp_edges(torch.cat((sndr, rcvr, z), dim=-1)))
        g = scatter(m_e, dim=0, index=edge_index[1:2, :].T, reduce=self.aggr, out=torch.zeros_like(g))
        return g, m_e


# ---------------------------------------------------------------------------
# Physics helpers
# ---------------------------------------------------------------------------

def _construct_heads_pp(h, q, r, edge_index, m_n_prv, m_e_pump, pump_ccs,
                        reservoir_mask, zeta=1e-24, verbose=True):
    """Reconstruct heads using the physics-informed algorithm."""
    q_relu = torch.relu(q)
    q_x = torch.pow(q_relu.double(), 1.852)
    l = (q_x * r.double()).float()

    if (m_e_pump == 1).sum() != 0:
        pump_ccs_orig = pump_ccs.clone()
        pcc = pump_ccs_orig[m_e_pump[:, 0] == 1, :]
        l_pumps = -1 * pcc[:, 3:4] ** 2 * (
            pcc[:, 0:1] - pcc[:, 1:2] * (q_relu[m_e_pump[:, 0] == 1, :] / pcc[:, 3:4]) ** pcc[:, 2:3]
        )
        l[m_e_pump[:, 0] == 1, :] = l_pumps

    J = 0
    h_updated = torch.zeros_like(h)
    h_star = h.clone()

    while not torch.equal(h, h_updated):
        h_updated = h.clone()
        sndr = h[edge_index[0, :], :]
        msg = sndr - l
        h_max = scatter(msg, dim=0, index=edge_index[1, :], reduce="max", out=torch.zeros_like(h))
        h = torch.maximum(h, h_max)
        h = torch.minimum(h, m_n_prv)
        h[reservoir_mask[:, 0], :] = h_star[reservoir_mask[:, 0], :]
        J += 1
        if J > 1000:
            if verbose is True:
                print("Warning: Maximum iterations reached in head reconstruction.")
            break

    l_hat = torch.pow(q.abs(), 1.852) * torch.sign(q) * r
    return h, l_hat, J


def _compute_net_flows_pp(h, r, edge_index, d, m_e_prv, m_e_pump, pump_ccs, q_hat=None, zeta=1e-32):
    """Compute flows/demands from heads using hydraulic principles."""
    sndr = h[edge_index[0, :], :]
    rcvr = h[edge_index[1, :], :]
    h_l = sndr - rcvr
    h_lr = h_l.double() / r.double()
    h_lr[h_lr == 0] = zeta
    q = (torch.pow(h_lr.abs(), 1 / 1.852) * torch.sign(h_lr)).float()

    if (m_e_pump == 1).sum() != 0:
        pcc = pump_ccs[m_e_pump[:, 0] == 1, :]
        q_x = ((h_l[m_e_pump[:, 0] == 1, :] + pcc[:, 3:4] ** 2 * pcc[:, 0:1]) * pcc[:, 3:4] ** pcc[:, 2:3]) / (
            pcc[:, 3:4] ** 2 * pcc[:, 1:2]
        )
        q_pumps = torch.pow(q_x.relu() + zeta, 1 / pcc[:, 2:3])
        q[m_e_pump[:, 0] == 1, :] = q_pumps
        q[m_e_pump[:, 0] == -1, :] = q_pumps * -1
        q[m_e_pump[:, 0] == 2, :] = 0.0
        q[m_e_pump[:, 0] == -2, :] = 0.0

    if (m_e_prv > 0).sum() != 0:
        q[m_e_prv[:, 0] != 0, :] = 0.0
        q_sum = scatter(q, dim=0, index=edge_index[1:2, :].T, reduce="add", out=torch.zeros_like(h))
        q_sum_in = -1 * (q_sum[edge_index[1, :], :] - d[edge_index[1, :], :])
        q[m_e_prv[:, 0] != 0, :] = q_sum_in[m_e_prv[:, 0] != 0, :]

    d_out = scatter(q, dim=0, index=edge_index[1:2, :].T, reduce="add", out=torch.zeros_like(h))
    return d_out, q, h_l


# ---------------------------------------------------------------------------
# PI_GNN Model
# ---------------------------------------------------------------------------

class _PI_GNN(Module):
    """
    Physics-informed GNN for state simulation in water distribution systems.
    """

    def __init__(self, M_n: int = 3, out_dim: int = 1, M_e=2, M_l=128, aggr="max", dia=13,
                 I=5, bias=False, n_iter=7, n_epochs=3000):
        super().__init__()
        self.I = I
        self.n_iter = n_iter
        self.n_epochs = n_epochs
        self.M_l = M_l
        self.dia = dia

        self.node = Linear(M_n, M_l, bias=bias)
        self.edge = Linear(M_e, M_l, bias=bias)
        self.z_latent = Linear(3 * M_l, M_l, bias=bias)
        self.flows_latent = Linear(2 * M_l, out_dim, bias=bias)

        # GNN layers
        self.gcn_aggrs: ModuleList = ModuleList()
        for _ in range(I):
            self.gcn_aggrs.append(SGNN_Layer(M_l, aggr=aggr, bias=bias))

    # ------------------------------------------------------------------ forward
    def forward(self, data, r_iter=5, zeta=1e-24, epoch=300, in_demands=None):
        _dev = next(self.parameters()).device
        data = data.to(_dev)
        x = data.x
        self.edge_index = data.edge_index
        self.r = data.edge_attr[:, 0:1]
        self.batch_size = data.num_graphs
        self.n_nodes = int(data.num_nodes / self.batch_size)
        self.n_edges = int(data.num_edges / self.batch_size)

        self.prv_mask_nodes = x[:, 2:3]
        self.pump_mask_nodes = x[:, 3:4]
        self.prv_mask_edges = data.edge_attr[:, 2:3]
        self.pump_mask_edges = data.edge_attr[:, 3:4]
        self.pump_curve_coefs = data.edge_attr[:, 4:8]
        self.edge_direct_mask = data.edge_attr[:, 1:2]

        self.pump_mask_edges_dir = self.pump_mask_edges[self.edge_direct_mask[:, 0] == 1, :]

        if in_demands is None:
            self.d_star = x[:, 1:2]
        else:
            self.d_star = in_demands

        self.h_star = x[:, 0:1].clone()
        self.reservoir_mask = x[:, 4:5].bool()

        self._normalize_hydraulics()

        # Initialise
        self.d_hat_nrm = torch.zeros_like(self.d_star)
        self.q_hat_nrm = torch.zeros_like(self.r)
        self.l_hat_nrm = torch.zeros_like(self.r)
        self.d_tilde_nrm = torch.zeros_like(self.d_star)
        self.q_tilde_nrm = torch.zeros_like(self.r)
        self.l_tilde_nrm = torch.zeros_like(self.r)
        self.h_tilde_nrm = self.h_star_nrm.clone()
        self.q_hat_dir = torch.zeros_like(self.r[self.edge_direct_mask[:, 0] == 0, :])

        self.epoch = epoch
        K = self.n_iter + (np.random.randint(0, r_iter) if self.training else r_iter)

        for k in range(K):
            # f1
            g = self.node(torch.cat((self.d_hat_nrm, self.d_star_nrm, self.reservoir_mask.float()),
                                    dim=-1))
            z = self.edge(torch.cat((self.q_tilde_nrm, self.q_hat_nrm), dim=-1))
            for gcn in self.gcn_aggrs:
                g, z = gcn(g, self.edge_index, z)

            sndr_g = g[self.edge_index[0, :], :]
            rcvr_g = g[self.edge_index[1, :], :]
            z_bar = self.z_latent(torch.cat((sndr_g, rcvr_g, z), dim=-1))

            delta_q = self.flows_latent(
                torch.cat((z_bar[self.edge_direct_mask[:, 0] == 0, :],
                           z_bar[self.edge_direct_mask[:, 0] == 1, :]), dim=-1)
            )
            self.q_hat_dir = self.q_hat_dir + delta_q

            # Pump flow modifications
            self.q_hat_dir[self.pump_mask_edges_dir[:, 0] == 1, :] = torch.minimum(
                self.q_hat_dir[self.pump_mask_edges_dir[:, 0] == 1, :],
                torch.zeros_like(self.q_hat_dir[self.pump_mask_edges_dir[:, 0] == 1, :]),
            )
            self.q_hat_dir[self.pump_mask_edges_dir[:, 0] == 2, :] = 0

            self.q_hat_nrm[self.edge_direct_mask[:, 0] == 0, :] = self.q_hat_dir
            self.q_hat_nrm[self.edge_direct_mask[:, 0] == 1, :] = self.q_hat_dir * -1

            self.d_hat_nrm = scatter(
                self.q_hat_nrm, dim=0, index=self.edge_index[1:2, :].T, reduce="add"
            )

            # f2 – physics-informed head reconstruction
            if ((k + 1) * self.I) >= self.dia:
                self.h_tilde_nrm, self.l_hat_nrm, _ = _construct_heads_pp(
                    h=self.h_star_nrm.clone(), q=self.q_hat_nrm.clone(), r=self.r_nrm,
                    edge_index=self.edge_index, m_n_prv=self.prv_mask_nodes_nrm,
                    m_e_pump=self.pump_mask_edges, pump_ccs=self.pump_curve_coefs_nrm,
                    zeta=zeta, reservoir_mask=self.reservoir_mask,
                )
                self.d_tilde_nrm, self.q_tilde_nrm, self.l_tilde_nrm = _compute_net_flows_pp(
                    h=self.h_tilde_nrm, r=self.r_nrm, edge_index=self.edge_index,
                    d=self.d_star_nrm, m_e_prv=self.prv_mask_edges, m_e_pump=self.pump_mask_edges,
                    pump_ccs=self.pump_curve_coefs_nrm,
                    q_hat=self.q_hat_nrm, zeta=zeta,
                )
                self.q_hat_nrm[self.pump_mask_edges[:, 0] == 1, :] = \
                    self.q_tilde_nrm[self.pump_mask_edges[:, 0] == 1, :]
                self.q_hat_nrm[self.pump_mask_edges[:, 0] == -1, :] = \
                    self.q_tilde_nrm[self.pump_mask_edges[:, 0] == -1, :]
                self.d_hat_nrm = scatter(
                    self.q_hat_nrm, dim=0, index=self.edge_index[1:2, :].T, reduce="add"
                )

        self._denormalize_hydraulics()

        return self.h_tilde

    # ----------------------------------------------------------- loss
    def loss(self, rho=0.1, delta=0.1):
        l1 = L1Loss(reduction="mean")
        non_res = ~self.reservoir_mask[:, 0]
        self.loss_d_hat = l1(self.d_hat_nrm[non_res, :], self.d_star_nrm[non_res, :])
        self.loss_d_tilde = l1(self.d_tilde_nrm[non_res, :], self.d_star_nrm[non_res, :])
        self.loss_q = l1(self.q_hat_nrm, self.q_tilde_nrm)
        return self.loss_d_hat + rho * self.loss_d_tilde + delta * self.loss_q

    # -------------------------------------------------------- normalization
    def _normalize_hydraulics(self):
        B, N, E = self.batch_size, self.n_nodes, self.n_edges

        # ---- Per-graph demand scale [B, 1] ----
        d_star_bg = torch.stack(self.d_star.split(N))   # [B, N, 1]
        d_max_bg  = d_star_bg.sum(dim=1)                # [B, 1]

        # ---- Per-graph resistance scale [B, 1] ----
        r_bg     = torch.stack(self.r.split(E))         # [B, E, 1]
        r_max_bg = 3.0 * r_bg.std(dim=1)               # [B, 1]

        # ---- Per-graph head scale [B, 1] ----
        d_max_n  = d_max_bg.repeat_interleave(N, dim=0)  # [B*N, 1]
        r_max_n  = r_max_bg.repeat_interleave(N, dim=0)  # [B*N, 1]
        h_ratio  = self.h_star / (torch.pow(d_max_n, 1.852) * r_max_n + 1e-12)
        h_star_max_bg = torch.stack(h_ratio.split(N)).max(dim=1).values  # [B, 1]

        # Combined head scale per graph [B, 1]
        scale_h_bg = torch.pow(d_max_bg, 1.852) * r_max_bg * h_star_max_bg  # [B, 1]

        # ---- Replicate scales to match batched node/edge tensors ----
        d_max_n   = d_max_bg.repeat_interleave(N, dim=0)              # [B*N, 1]
        d_max_e   = d_max_bg.repeat_interleave(E, dim=0)              # [B*E, 1]
        scale_h_n = scale_h_bg.repeat_interleave(N, dim=0)            # [B*N, 1]
        scale_h_e = scale_h_bg.repeat_interleave(E, dim=0)            # [B*E, 1]
        r_scale_e = (h_star_max_bg * r_max_bg + 1e-12).repeat_interleave(E, dim=0)  # [B*E, 1]

        # Store node-aligned and edge-aligned scales for _denormalize_hydraulics
        self._d_max_n   = d_max_n
        self._d_max_e   = d_max_e
        self._scale_h_n = scale_h_n

        # ---- Normalised quantities ----
        self.d_star_nrm         = self.d_star    / d_max_n
        self.h_star_nrm         = self.h_star    / scale_h_n
        self.r_nrm              = self.r         / r_scale_e
        self.prv_mask_nodes_nrm = self.prv_mask_nodes / scale_h_n

        self.pump_curve_coefs_nrm = self.pump_curve_coefs.clone()
        self.pump_curve_coefs_nrm[..., 0:1] = self.pump_curve_coefs[..., 0:1] / scale_h_e
        C_vals = self.pump_curve_coefs[..., 2:3]
        self.pump_curve_coefs_nrm[..., 1:2] = (
            self.pump_curve_coefs[..., 1:2] * torch.pow(d_max_e, C_vals)
        ) / scale_h_e

    def _denormalize_hydraulics(self):
        self.scale_h = self._scale_h_n                      # kept for external inspection
        self.d_star  = self.d_star_nrm  * self._d_max_n
        self.d_hat   = self.d_hat_nrm   * self._d_max_n
        self.d_tilde = self.d_tilde_nrm * self._d_max_n
        self.q_hat   = self.q_hat_nrm   * self._d_max_e
        self.q_tilde = self.q_tilde_nrm * self._d_max_e
        self.h_tilde = self.h_tilde_nrm * self._scale_h_n

# ---------------------------------------------------------------------------
# Internal graph data container
# ---------------------------------------------------------------------------

@dataclass
class _WDNGraph:
    """Simple container mirroring the original WDN_Graph."""
    X: Optional[torch.Tensor] = None  # [T, N, F_node]
    edge_index: Optional[torch.Tensor] = None  # [T, 2, E]  or [2, E]
    edge_attr: Optional[torch.Tensor] = None  # [T, E, F_edge]
    base_demands: Optional[torch.Tensor] = None
    time_interval: float = 1800.0
    node_names: Optional[np.ndarray] = None
    node_coords: Optional[Any] = None
    reservoirs: List[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal PyG dataset
# ---------------------------------------------------------------------------

class _WDNDatasetIM(InMemoryDataset):
    """In-memory dataset: one PyG ``Data`` object per timestep."""

    def __init__(self):
        super().__init__()
        self.data_list: Optional[np.ndarray] = None
        self._wdn_data: list = []

    def len(self):
        return len(self._wdn_data)

    def get(self, idx):
        return self._wdn_data[idx]

    def load_wds(self, wds: _WDNGraph, reservoirs: List[int], n_nodes: int, masked: bool = True):
        self._wdn_data = []
        assert wds.X is not None and wds.edge_attr is not None and wds.edge_index is not None
        Y = wds.X.clone()
        for idx in self.data_list:
            mask = torch.zeros((n_nodes, 1), dtype=torch.float32)
            mask[reservoirs] = 1.0
            d = Data(
                x=wds.X[idx, :, :].clone(),
                y=Y[idx, :, :].clone(),
                edge_attr=wds.edge_attr[idx],
                edge_index=wds.edge_index[idx],
            )
            if masked:
                assert d.x is not None
                d.x[mask[:, 0] == 0, 0] = 0
            self._wdn_data.append(d)
        return Y


def _load_dataset(wds: _WDNGraph, n_nodes: int, reservoirs: List[int], masked: bool = True):
    ds = _WDNDatasetIM()
    assert wds.X is not None
    ds.data_list = np.arange(wds.X.shape[0])
    Y = ds.load_wds(wds, reservoirs, n_nodes, masked)
    return ds, Y


# ---------------------------------------------------------------------------
# Unit conversion  (EPANET native -> SI)
# ---------------------------------------------------------------------------

def _get_si_conversions(flow_units: int, pressure_units: int) -> Dict[str, float]:
    """Multiplicative factors to convert .inp native units to SI (m, m³/s).

    EPANET flow-unit codes:
        0=CFS, 1=GPM, 2=MGD, 3=IMGD, 4=AFD  (US customary)
        5=LPS, 6=LPM, 7=MLD, 8=CMH, 9=CMD 10=CMS  (SI)
    """
    _flow_to_cms: Dict[int, float] = {
        EpanetConstants.EN_CFS: 0.028316846592,     # CFS  -> m^3/s
        EpanetConstants.EN_GPM: 6.30902e-5,         # GPM  -> m^3/s
        EpanetConstants.EN_MGD: 0.04381264,         # MGD  -> m^3/s
        EpanetConstants.EN_IMGD: 0.05261678,         # IMGD -> m^3/s
        EpanetConstants.EN_AFD: 0.01427641,         # AFD  -> m^3/s
        EpanetConstants.EN_LPS: 1e-3,               # LPS  -> m^3/s
        EpanetConstants.EN_LPM: 1e-3 / 60.0,       # LPM  -> m^3/s
        EpanetConstants.EN_MLD: 1e3 / 86400.0,     # MLD  -> m^3/s
        EpanetConstants.EN_CMH: 1.0 / 3600.0,      # CMH  -> m^3/s
        EpanetConstants.EN_CMD: 1.0 / 86400.0,     # CMD  -> m^3/s
        EpanetConstants.EN_CMS: 1.                # CMS  -> m^3/s
    }

    _pressure_to_head: Dict[int, float] = {
        EpanetConstants.EN_PSI: 0.70324961490205,
        EpanetConstants.EN_KPA: 0.10199773339984,
        EpanetConstants.EN_METERS: 1.,
        EpanetConstants.EN_BAR: 10.199773339984,
        EpanetConstants.EN_FEET: 0.3048
    }

    is_us = flow_units <= 4
    _ft2m = 0.3048
    return {
        "flow": _flow_to_cms.get(flow_units, 1.0),
        "length": _ft2m if is_us else 1.0,                       # ft -> m  | m -> m
        "diameter": 0.0254 if is_us else 1e-3,                   # in -> m  | mm -> m
        "elevation": _ft2m if is_us else 1.0,                    # ft -> m  | m -> m
        "pressure_to_head": _pressure_to_head.get(pressure_units, 1.0),
        "power": 745.69987 if is_us else 1.0,                    # HP -> W  | W -> W
        "volume": _ft2m ** 3 if is_us else 1.0,                  # ft^3 -> m^3| m^3 -> m^3
    }


# ---------------------------------------------------------------------------
# Graph builder from .inp file  (using EPyT-Flow topology)
# ---------------------------------------------------------------------------

def _convert_to_bi_edges(edge_index, edge_attr=None):
    swap = edge_index.clone()
    swap_copy = swap.clone()
    swap[0, :] = swap_copy[1, :]
    swap[1, :] = swap_copy[0, :]
    ei_bi = torch.cat([edge_index, swap], dim=-1)
    if edge_attr is not None:
        ea_bi = torch.cat([edge_attr, edge_attr], dim=1)
        return ei_bi, ea_bi
    return ei_bi


def _compute_pump_abc(curve_points: list) -> Tuple[float, float, float]:
    """Compute pump curve coefficients *A*, *B*, *C* where ``H = A - B·Q^C``.

    Replicates EPANET's ``powercurve()`` logic (src/input3.c, v2.2) and
    extends it to arbitrary N-point HEAD curves via log-linear regression.

    **1-point curve** ``(Q1, H1)``
        EPANET internally generates a synthetic 3-point curve
        ``(0, 4/3·H1), (Q1, H1), (2·Q1, 0)`` and applies the exact
        power-curve formula, yielding:
        ``A = 4/3·H1``,  ``C = 2``,  ``B = H1 / (3·Q1²)``.

    **N-point curve** ``(0, H0), (Q1, H1), …`` *(first point must be the
    shutoff point at Q = 0)*
        EPANET's ``powercurve()`` is an exact pass-through formula for the
        3-point case:
            ``A = H0``
            ``C = log((H0-H2)/(H0-H1)) / log(Q2/Q1)``
            ``B = (H0-H1) / Q1^C``
        For N = 2 total points (shutoff + one design point), there is no
        unique solution so *C = 2* is assumed (matching the single-point
        expansion convention) and *B* is derived from the design point.
        For N ≥ 3 total points, the model ``ln(A-H) = ln(B) + C·ln(Q)``
        is linearised and *B*, *C* are obtained by ordinary least-squares
        regression (exact for N = 3, best-fit for N > 3).  This is
        consistent with EPANET's underlying power-curve physics.

    Parameters
    ----------
    curve_points:
        List of ``(Q, H)`` pairs in SI units (m³/s, m).  For single-point
        curves supply one pair with Q > 0.  For multi-point curves the
        first pair **must** be the shutoff point ``(0, H_shutoff)``.

    Returns
    -------
    A, B, C : float
        Coefficients of ``H = A - B·Q^C``.

    Raises
    ------
    NotImplementedError
        If the curve has no shutoff point (Q = 0) and contains more than
        one design point (cannot uniquely determine A).
    ValueError
        If the curve data are physically inconsistent (e.g. H ≥ A for
        some flow point, or Q ≤ 0 for a non-shutoff point).
    """
    pts = np.array(curve_points, dtype=np.float64)

    # Sort by flow so the shutoff point is first
    pts = pts[np.argsort(pts[:, 0])]
    n = len(pts)

    # ------------------------------------------------------------------
    # 1-point curve: EPANET single-point expansion (powercurve() path)
    # ------------------------------------------------------------------
    if n == 1:
        Q1, H1 = pts[0]
        if Q1 <= 0.0:
            raise ValueError("Single-point pump curve must have Q > 0.")
        # Synthetic 3-point curve: (0, 4/3·H1), (Q1, H1), (2·Q1, 0)
        # Yields C = 2 exactly via powercurve()
        A = (4.0 / 3.0) * H1
        C = 2.0
        B = H1 / (3.0 * Q1 ** 2)
        return float(A), float(B), float(C)

    # ------------------------------------------------------------------
    # N-point curve (N ≥ 2)
    # ------------------------------------------------------------------
    if pts[0, 0] > 1e-12:
        # No shutoff point present; uniquely determining A is not possible
        # without nonlinear optimisation.
        raise NotImplementedError(
            f"Pump curve with {n} point(s) but no shutoff point (Q = 0) "
            "is not supported.  Include the shutoff head as the first "
            "curve point, e.g. (0, H_shutoff)."
        )

    A = float(pts[0, 1])               # shutoff head
    non_zero = pts[1:]                 # design / operating points (Q > 0)

    if len(non_zero) == 0:
        raise ValueError("Degenerate pump curve: only a shutoff point was provided.")

    if len(non_zero) == 1:
        # 2-point total: shutoff + one design point.
        # C cannot be determined uniquely; assume C = 2
        # (consistent with EPANET single-point convention).
        Q1, H1 = non_zero[0]
        if Q1 <= 0.0:
            raise ValueError("Design point flow must be > 0.")
        dH = A - H1
        if dH <= 0.0:
            raise ValueError(
                f"Invalid pump curve: shutoff head A={A:.4g} must exceed "
                f"the head at Q={Q1:.4g} (H={H1:.4g})."
            )
        C = 2.0
        B = dH / (Q1 ** C)
    else:
        # 3+ total points: log-linear regression
        # ln(A - H_i) = ln(B) + C · ln(Q_i)
        # For exactly 2 non-shutoff points (3-point curve) this is the
        # exact EPANET powercurve() formula; for more points it is OLS.
        Q_vals = non_zero[:, 0]
        H_vals = non_zero[:, 1]
        if np.any(Q_vals <= 0.0):
            raise ValueError("All non-shutoff pump curve points must have Q > 0.")
        dH_vals = A - H_vals
        if np.any(dH_vals <= 0.0):
            raise ValueError(
                "Invalid pump curve: shutoff head A must strictly exceed "
                "the head H at every operating point."
            )
        X = np.log(Q_vals)
        Y = np.log(dH_vals)
        # polyfit returns [C, ln(B)]
        coeffs = np.polyfit(X, Y, deg=1)
        C = float(coeffs[0])
        B = float(np.exp(coeffs[1]))
        if C <= 0.0:
            raise ValueError(
                f"Pump curve regression yielded a non-positive exponent C={C:.4g}. "
                "Check that the curve is monotonically decreasing."
            )

    return float(A), float(B), float(C)


# ---------------------------------------------------------------------------
# Core graph builder  (uses only epytflow topology)
# ---------------------------------------------------------------------------

def _build_graph_core(topo, time_interval: float,
                      heads: torch.Tensor, demands: torch.Tensor,
                      flow_units: int, pressure_units: int) -> _WDNGraph:
    """Build a ``_WDNGraph`` from an already-loaded
    :class:`~epyt_flow.topology.NetworkTopology` and time-series tensors.

    Inputs (``heads``, ``demands``) must already be in **SI**
    (meters, m^3/s).  Topology-derived quantities (pipe lengths, diameters,
    elevations, valve/pump/tank properties) are converted to SI internally
    according to ``flow_units``.

    The output feature layout matches that expected by :class:`_PI_GNN`:

    * ``X``  - ``[T, N, F_node]`` where *F_node* includes
      heads, demands, prv_mask, pump_mask, res_mask,
      base_demands, elevations

    * ``edge_attr`` - ``[T, 2*E, F_edge]`` where *F_edge* includes
      r, edge_direct_mask, prv_mask, pump_mask, A, B, C, W,
            l, d, c, pipe_mask

    * ``flows_gt`` - optional ``[T, 2*E, 1]`` flow targets kept outside
        ``edge_attr`` to avoid flow leakage into model inputs.
    """
    conv = _get_si_conversions(flow_units, pressure_units)

    # ------------------------------------------------------------------
    #  Nodes
    # ------------------------------------------------------------------
    all_node_ids = topo.get_all_nodes()  # list of str in EPANET order
    n_nodes = len(all_node_ids)
    node_to_idx: Dict[str, int] = {nid: i for i, nid in enumerate(all_node_ids)}
    node_names = np.array(all_node_ids)

    elevs_np = np.zeros(n_nodes, dtype=np.float32)
    node_coords_list: list = []
    reservoir_indices: List[int] = []
    tank_indices: List[int] = []

    for i, nid in enumerate(all_node_ids):
        ninfo = topo.get_node_info(nid)
        elevs_np[i] = float(ninfo.get("elevation", 0.0))
        node_coords_list.append(ninfo.get("coord", (0.0, 0.0)))
        ntype = int(ninfo.get("type", 0))
        if ntype == EpanetConstants.EN_RESERVOIR:     # Reservoir
            reservoir_indices.append(i)
        elif ntype == EpanetConstants.EN_TANK:   # Tank
            tank_indices.append(i)

    reservoirs_all = reservoir_indices + tank_indices
    elevs_np *= conv['elevation']                          # -> m
    elevs = torch.tensor(elevs_np, dtype=torch.float32)
    elevs = torch.nan_to_num(elevs, nan=0, posinf=0, neginf=0)

    # ------------------------------------------------------------------
    #  Links
    # ------------------------------------------------------------------
    all_links_data = topo.get_all_links()  # [(link_id, [start, end]), ...]
    n_edges = len(all_links_data)
    link_ids: List[str] = []

    all_edge_indices = np.zeros((2, n_edges), dtype=int)
    lengths = np.zeros(n_edges, dtype=np.float32)
    diameters = np.zeros(n_edges, dtype=np.float32)
    roughnesses = np.zeros(n_edges, dtype=np.float32)
    link_type_arr = np.zeros(n_edges, dtype=int)
    init_status_arr = np.ones(n_edges, dtype=np.float32)

    for j, (lid, _endpoints) in enumerate(all_links_data):
        link_ids.append(lid)
        linfo = topo.get_link_info(lid)
        nodes_pair = linfo["nodes"]
        all_edge_indices[0, j] = node_to_idx[nodes_pair[0]]
        all_edge_indices[1, j] = node_to_idx[nodes_pair[1]]
        lengths[j] = float(linfo.get("length", 0.0))
        diameters[j] = float(linfo.get("diameter", 0.0))
        roughnesses[j] = float(linfo.get("roughness_coeff", 0.0))
        link_type_arr[j] = int(linfo.get("type", 1))
        init_status_arr[j] = float(linfo.get("init_status", 1))
    lengths *= conv['length']                              # -> m
    diameters *= conv['diameter']                           # -> m

    edge_indices_orig = torch.tensor(all_edge_indices, dtype=torch.long)

    T = heads.shape[0]

    # ------------------------------------------------------------------
    #  Node features: heads + demands
    # ------------------------------------------------------------------
    X = torch.zeros(T, n_nodes, 2, dtype=torch.float32)
    X[:, :, 0] = heads[:, :n_nodes]
    X[:, :, 1] = demands[:, :n_nodes].relu()  # Demands should be non-negative; relu to remove reservoir demands if present
    X = X.nan_to_num(0)

    # ------------------------------------------------------------------
    #  Resistance  r = 10.667 · L · C^{-1.852} · D^{-4.871}
    # ------------------------------------------------------------------
    ldc = torch.tensor(
        np.stack([lengths, diameters, roughnesses], axis=1), dtype=torch.float32
    )
    ldc = torch.nan_to_num(ldc, nan=0, posinf=0, neginf=0)
    constnt = 10.667
    r = constnt * ldc[..., 0:1] * torch.pow(ldc[..., 2:3], -1.852) * torch.pow(ldc[..., 1:2], -4.871)
    r = torch.nan_to_num(r, nan=0, posinf=0, neginf=0)
    edge_attr_orig = torch.zeros(T, n_edges, 1, dtype=torch.float32)
    edge_attr_orig[:, :, 0:1] = r

    # ------------------------------------------------------------------
    #  PRV / pump feature tensors  (6 channels)
    # ------------------------------------------------------------------
    X_ppf = torch.zeros(T, n_nodes, 2, dtype=torch.float32)
    edge_attr_ppf = torch.zeros(T, n_edges * 2, 6, dtype=torch.float32)

    # ---- Valves (PRV / FCV) ----
    # EPANET link-type codes: 3=PRV, 4=PSV, 5=PBV, 6=FCV, 7=TCV, 8=GPV
    valve_ids = topo.get_all_valves()
    prv_idx = np.zeros(n_edges, dtype=bool)
    prv_settings = np.zeros(n_edges, dtype=np.float32)

    _fcv_ids_found: List[str] = []
    for vid in valve_ids:
        vinfo = topo.get_valve_info(vid)
        vtype = int(vinfo.get("type", 0))
        j = link_ids.index(vid)
        if vtype == EpanetConstants.EN_PRV:    # PRV
            prv_idx[j] = True
            prv_settings[j] = float(vinfo.get("initial_setting", 0.0))
        elif vtype == EpanetConstants.EN_FCV:  # FCV – not yet supported
            _fcv_ids_found.append(vid)
    if _fcv_ids_found:
        raise NotImplementedError(
            f"Network contains {len(_fcv_ids_found)} FCV(s) "
            f"({', '.join(_fcv_ids_found)}). "
            "Flow control valves are not yet supported."
        )
    prv_settings *= conv['pressure_to_head']

    # PRV node mask
    prv_nodes = edge_indices_orig[:, prv_idx]
    prv_vals = torch.tensor(prv_settings[prv_idx], dtype=torch.float32)
    prv_values_mask = torch.ones(n_nodes, dtype=torch.float32) * X[..., 0].max() * 10
    if len(prv_vals) > 0:
        prv_values_mask[prv_nodes[1, :]] = prv_vals + elevs[prv_nodes[1, :]]
    prv_idx_t = torch.cat([
        torch.tensor(prv_idx, dtype=torch.float32),
        torch.tensor(prv_idx, dtype=torch.float32) * -1,
    ], dim=-1)
    X_ppf[:, :, 0] = prv_values_mask
    edge_attr_ppf[:, :, 0] = prv_idx_t

    # ---- Pumps ----
    pump_ids = topo.get_all_pumps()
    pump_start_nodes: List[int] = []
    pump_nodes_list: List[int] = []
    pump_curves_indxs: List[int] = []
    pump_idx_arr = np.zeros((n_edges, 5), dtype=np.float32)

    _supported_pump_count = 0
    for pidx, pid in enumerate(pump_ids):
        pinfo = topo.get_pump_info(pid)
        j = link_ids.index(pid)

        # EPANET pump link-type: 2 = HEAD pump, 3 = POWER pump
        ptype = int(pinfo.get("type", 2))
        if ptype == 3:
            raise NotImplementedError(
                f"Pump '{pid}' is a POWER pump, which is not yet supported."
            )

        pump_edge = np.zeros(n_edges, dtype=np.float32)
        pump_edge[j] = 1.0
        pump_idx_arr[:, 0] += pump_edge
        pump_start_nodes.append(int(edge_indices_orig[0, j]))
        pump_nodes_list.append(int(edge_indices_orig[1, j]))

        A: float = 0.0
        B: float = 0.0
        C: float = 0.0
        W: float = float(pinfo.get("init_setting", 1.0))
        curve_id = str(pinfo.get("curve_id", ""))

        if curve_id and curve_id in topo.curves:
            # HEAD pump with explicit curve - convert to SI
            _, curve_pts = topo.curves[curve_id]
            curve_pts_si = [(q * conv['flow'], h * conv['elevation'])
                           for q, h in curve_pts]
            A, B, C = _compute_pump_abc(curve_pts_si)

        pump_idx_arr[pump_edge == 1, 1:5] = [A, B, C, W]
        pump_idx_arr[pump_idx_arr[:, 0] == 0, 1:5] = 1.0
        _supported_pump_count += 1
        pump_curves_indxs.append(_supported_pump_count)

    pump_nodes_mask = torch.ones(n_nodes, dtype=torch.float32)
    if pump_nodes_list:
        pump_nodes_mask[pump_nodes_list] = torch.tensor(pump_curves_indxs, dtype=torch.float32)
        pump_nodes_mask[pump_start_nodes] = torch.tensor(pump_curves_indxs, dtype=torch.float32) * -1
    pump_idx_bi = torch.cat([torch.tensor(pump_idx_arr), torch.tensor(pump_idx_arr)], dim=0)
    nz = pump_idx_bi[pump_idx_arr.shape[0]:, 0:1] != 0
    pump_idx_bi[pump_idx_arr.shape[0]:, 0:1][nz] *= -1

    X_ppf[:, :, 1] = pump_nodes_mask
    edge_attr_ppf[:, :, 1:6] = pump_idx_bi

    # ---- Check Valves ---- (not yet supported)
    _cv_count = int((link_type_arr == 0).sum())
    if _cv_count > 0:
        raise NotImplementedError(
            f"Network contains {_cv_count} check valve(s) (CVPIPE links). "
            "Check valves are not yet supported."
        )

    X = torch.cat((X, X_ppf), dim=-1)
    res_mask = torch.zeros(T, n_nodes, 1, dtype=torch.float32)
    res_mask[:, reservoirs_all, :] = 1.0
    X = torch.cat((X, res_mask), dim=-1)  # index 4 = res_mask

    # ------------------------------------------------------------------
    #  Tanks  (not yet supported)
    # ------------------------------------------------------------------
    if tank_indices:
        raise NotImplementedError(
            f"Network contains {len(tank_indices)} tank(s). "
            "Tanks are not yet supported."
        )
    X = torch.nan_to_num(X, nan=0, posinf=0, neginf=0)

    edge_indices_orig_bi, edge_attr_bi = _convert_to_bi_edges(edge_indices_orig, edge_attr_orig.clone())

    edge_direct_mask = torch.ones(T, n_edges, 1, dtype=torch.float32)
    edge_direct_mask_bi = torch.cat([edge_direct_mask, edge_direct_mask * 0], dim=1)

    edge_attr_full = torch.cat((edge_attr_bi, edge_direct_mask_bi, edge_attr_ppf), dim=-1)
    edge_index_full = edge_indices_orig_bi.repeat(T, 1, 1)

    ldc_rep = ldc.unsqueeze(0).repeat(T, 1, 1)
    closed_links = torch.tensor(init_status_arr == 0, dtype=torch.float32)[:, None]
    closed_links_rep = closed_links.unsqueeze(0).repeat(T, 1, 1)
    edge_attr_full = torch.cat(
        (edge_attr_full,
         torch.cat((ldc_rep, ldc_rep), dim=1),
         torch.cat((closed_links_rep, closed_links_rep), dim=1)),
        dim=-1,
    )

    # Zero out pipe attributes on PRV / pump edges
    rldc_idx = [0, 8, 9, 10]
    for _idx in rldc_idx:
        edge_attr_full[:, :, _idx : _idx + 1][edge_attr_full[:, :, 2:3] != 0] = 1.0
        edge_attr_full[:, :, _idx : _idx + 1][edge_attr_full[:, :, 3:4] != 0] = 1.0

    # Base demands + elevations as final node features
    base_demands = demands[:, :n_nodes].unsqueeze(2)
    elevs_full = elevs[None, :, None].repeat(T, 1, 1)
    X = torch.cat((X, base_demands, elevs_full), dim=-1)

    return _WDNGraph(
        X=X,
        edge_index=edge_index_full,
        edge_attr=edge_attr_full,
        base_demands=base_demands,
        time_interval=time_interval,
        node_names=node_names,
        node_coords=np.array(node_coords_list, dtype=object),
        reservoirs=reservoirs_all,
    )


def _build_graph(inp_file: str, heads: torch.Tensor, demands: torch.Tensor) -> _WDNGraph:
    """Build a ``_WDNGraph`` from an ``.inp`` file and time-series tensors."""
    with ScenarioSimulator(f_inp_in=inp_file) as sim:
        if sim.epanet_api.getoption(EpanetConstants.EN_HEADLOSSFORM) != EpanetConstants.EN_HW:
            raise ValueError("Unsupport head loss formula! Only Hazen-Williams is supported.")

        topo = sim.get_topology()
        flow_units = sim.get_flow_units()
        pressure_units = sim.get_pressure_units()
        try:
            config = sim.get_scenario_config()
            time_interval = float(config.general_params.get("hydraulic_time_step", 1800))
        except (ValueError, Exception):
            time_interval = float(sim.epanet_api.get_hydraulic_time_step())
    return _build_graph_core(topo, time_interval, heads, demands,
                             flow_units=flow_units, pressure_units=pressure_units)


# ---------------------------------------------------------------------------
# Graph builder from EPytFlow ScadaData
# ---------------------------------------------------------------------------

def _build_graph_from_epytflow(inp_file: str, scada_data) -> Tuple[_WDNGraph, torch.Tensor]:
    """Build a ``_WDNGraph`` from EPytFlow ``ScadaData`` + the corresponding
    ``.inp`` file.  Pressures, demands and flows are read from the SCADA
    object; all topology is parsed via *epytflow* from the ``.inp``.
    """
    with ScenarioSimulator(f_inp_in=inp_file) as sim:
        if sim.epanet_api.getoption(EpanetConstants.EN_HEADLOSSFORM) != EpanetConstants.EN_HW:
            raise ValueError("Unsupport head loss formula! Only Hazen-Williams is supported.")

        topo = sim.get_topology()
        flow_units = sim.get_flow_units()
        pressure_units = sim.get_pressure_units()
        try:
            config = sim.get_scenario_config()
            time_interval = float(config.general_params.get("hydraulic_time_step", 1800))
        except (ValueError, Exception):
            time_interval = float(sim.epanet_api.get_hydraulic_time_step())

    conv = _get_si_conversions(flow_units, pressure_units)
    n_nodes = len(topo.get_all_nodes())
    n_edges = len(topo.get_all_links())

    # ---- Extract time-series from ScadaData (convert to SI) ----
    gt_pressures = scada_data.get_data_pressures()            # [T, N]  native pressure units
    gt_demands = scada_data.get_data_demands() * conv['flow']            # -> m^3/s
    gt_flows = scada_data.get_data_flows() * conv['flow']                # -> m^3/s

    T = gt_pressures.shape[0]

    # Elevations from topology (in meters)
    elevs_np = np.array(
        [float(topo.get_node_info(nid).get("elevation", 0.0))
         for nid in topo.get_all_nodes()],
        dtype=np.float32,
    ) * conv['elevation']                                     # -> m
    elevs_tile = elevs_np[None, :].repeat(T, axis=0)

    # Heads(m) = pressure × conversion + elevation(m)
    gt_heads = (gt_pressures[:, :n_nodes] * conv['pressure_to_head']
                + elevs_tile[:, :n_nodes])

    heads_t = torch.tensor(gt_heads, dtype=torch.float32)
    demands_t = torch.tensor(gt_demands[:, :n_nodes], dtype=torch.float32)
    flows_t = torch.tensor(gt_flows[:, :n_edges], dtype=torch.float32)
    # Build bidirectional gt flows [T, 2*E, 1] for external comparison only
    flows_bi_gt = torch.cat(
        (flows_t.unsqueeze(2), flows_t.unsqueeze(2) * -1), dim=1
    )
    graph = _build_graph_core(topo, time_interval, heads_t, demands_t,
                              flow_units=flow_units, pressure_units=pressure_units)
    return graph, flows_bi_gt


# ---------------------------------------------------------------------------
# Demand augmentation helpers
# ---------------------------------------------------------------------------

def _add_noise_to_demands(demands, dem_dist_fac, dist="uniform"):
    shape = demands.shape
    if dist == "normal":
        noise = torch.randn(shape, dtype=torch.float32) * dem_dist_fac
    elif dist == "uniform":
        noise = torch.empty(shape, dtype=torch.float32).uniform_(-dem_dist_fac, dem_dist_fac)
    else:
        noise = torch.zeros(shape, dtype=torch.float32)
    return (demands * (1 + noise)).clip(min=0.0)

def _add_noise_to_diameters(diameters, dia_dist_fac, dist='uniform'):
    _min, _max = diameters.min(), diameters.max()
    shape = diameters.shape
    if dist == 'normal':
        noise = torch.randn(shape, dtype=torch.float32) * dia_dist_fac
    elif dist == 'uniform':
        noise = torch.empty(shape, dtype=torch.float32).uniform_(-dia_dist_fac, dia_dist_fac)
    else:
        noise = torch.zeros(shape, dtype=torch.float32)
    diameters = diameters * (1 + noise)
    return diameters.clip(min=_min, max=_max)


# ===========================================================================
#  Public API  –  PIGNNModel
# ===========================================================================

class PIGNNModel:
    """
    Self-contained wrapper that integrates the PI_GNN model with EPytFlow.

    Parameters
    ----------
    inp_file : str
        Path to the EPANET ``.inp`` network file.
    device : torch.device or str, optional
        Compute device (default: auto-detect CUDA).
    """

    def __init__(self, inp_file: str, device: Optional[Union[str, torch.device]] = None):
        self.inp_file = inp_file
        self.device = torch.device(device) if device else globals()["device"]

        # Will be populated by data-loading helpers
        self._wdn_graph: Optional[_WDNGraph] = None
        self._reservoirs: Optional[List[int]] = None
        self._tanks: Optional[List[int]] = None

        self._alldata_wds: Optional[_WDNGraph] = None

        # Splits
        self._train_wds: Optional[_WDNGraph] = None
        self._val_wds: Optional[_WDNGraph] = None
        self._test_wds: Optional[_WDNGraph] = None

        # Model
        self.model: Optional[_PI_GNN] = None
        self._model_path: Optional[str] = None
        # Ground-truth flows for external comparison (never fed into the model)
        self._gt_flows_raw: Optional[torch.Tensor] = None   # [T, 2*E, 1] full dataset
        self._gt_flows_test: Optional[torch.Tensor] = None  # [T_test, 2*E, 1] test split

        # Hyperparameters (sensible defaults matching original paper)
        self.batch_size: int = 48
        self.train_with_rand_demands: bool = True
        self.dem_dist: str = "uniform"
        self.dem_dist_fac: float = 0.5
        self.train_with_rand_dias: bool = True
        self.dia_dist: str = "uniform"
        self.dia_dist_fac: float = 0.1

    @classmethod
    def from_network(cls, network_name: str, load_pretrained_model: bool = False,
                     device: Optional[Union[str, torch.device]] = None) -> "PIGNNModel":
        """Create a :class:`PIGNNModel` from a built-in EPytFlow network.

        The ``.inp`` file is downloaded (and cached) automatically.

        Parameters
        ----------
        network_name : str
            One of: ``anytown``, ``hanoi``, ``ltown_a``.
        load_pretrained_model : bool, optional
            If True, a pre-trained model will be downloaded and loaded.

            The default is False.
        device : str or torch.device, optional
            Compute device.

        Returns
        -------
        PIGNNModel
        """
        name = network_name.lower()
        if name not in EPYTFLOW_NETWORKS_PIGNNHYDSURROGATE:
            raise ValueError(
                f"Unknown network {network_name!r}. "
                f"Available: {sorted(EPYTFLOW_NETWORKS_PIGNNHYDSURROGATE.keys())}"
            )
        config = EPYTFLOW_NETWORKS_PIGNNHYDSURROGATE[name]()
        m = cls(inp_file=config.f_inp_in, device=device)

        if load_pretrained_model is True:
            url = f"https://raw.githubusercontent.com/WaterFutures/EPyT-Control/refs/heads/main/pretrained-models/{name}_pignn.pt"
            f_in = os.path.join(get_temp_folder(), f"{name}_pignn.pt")
            download_if_necessary(f_in, url)

            m.load_model(f_in)

        return m 

    # ------------------------------------------------------------------
    #  Data loading
    # ------------------------------------------------------------------

    def load_from_arrays(
        self,
        heads: np.ndarray,
        demands: np.ndarray,
    ) -> None:
        """
        Build graph data from raw numpy arrays.

        Parameters
        ----------
        heads : ndarray, shape [T, N]
            Hydraulic heads at every node for T timesteps.
        demands : ndarray, shape [T, N]
            Demands at every node (in m³/s).

        """
        self._wdn_graph = _build_graph(
            self.inp_file,
            torch.tensor(heads, dtype=torch.float32),
            torch.tensor(demands, dtype=torch.float32),
        )
        self._reservoirs = self._wdn_graph.reservoirs

    def load_epytflow_scada(
        self,
        scada_data_or_path: Union[str, Any],  # str or ScadaData
    ) -> None:
        """
        Build graph data from EPytFlow ScadaData.

        Parameters
        ----------
        scada_data_or_path : str or ScadaData
            Either a path to a persisted ``.epytflow_scada_data`` file or an
            already-loaded ``ScadaData`` object.

        """
        if isinstance(scada_data_or_path, str):
            scada_data_or_path = ScadaData.load_from_file(scada_data_or_path)

        self._wdn_graph, self._gt_flows_raw = _build_graph_from_epytflow(self.inp_file, scada_data_or_path)
        self._reservoirs = self._wdn_graph.reservoirs

    def load_from_inp(
        self,
        simulation_duration_sec: int = 120 * 24 * 3600,
        hydraulic_timestep: int = 1800,
        randomize_demands: bool = True,
    ) -> None:
        """
        Run an EPytFlow simulation from the ``.inp`` file and load the
        resulting data.

        Parameters
        ----------
        simulation_duration_sec : int
            Total simulation time in seconds (default 120 days).
        hydraulic_timestep : int
            Hydraulic time step in seconds (default 1800 = 30 min).
        randomize_demands : bool
            Whether to randomize demands in the simulation.

        """
        with ScenarioSimulator(f_inp_in=self.inp_file) as sim:
            sim.set_general_parameters(
                simulation_duration=simulation_duration_sec,
                hydraulic_time_step=hydraulic_timestep,
                reporting_time_step=hydraulic_timestep
            )
            if randomize_demands:
                sim.randomize_demands()

            sensor_config = sim.sensor_config
            sim.set_pressure_sensors(sensor_config.nodes)
            sim.set_demand_sensors(sensor_config.nodes)
            sim.set_flow_sensors(sensor_config.links)
            scada_data = sim.run_simulation()

        self._wdn_graph, self._gt_flows_raw = _build_graph_from_epytflow(self.inp_file, scada_data)
        self._reservoirs = self._wdn_graph.reservoirs

    # ------------------------------------------------------------------
    #  Data preparation (splits)
    # ------------------------------------------------------------------

    def prepare_data(
        self,
        n_samples: Optional[int] = None,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
    ) -> None:
        """
        Split the loaded graph data into train / val / test sets.

        Parameters
        ----------
        n_samples : int, optional
            How many timesteps to use (default: all).
        train_ratio, val_ratio : float
            Fractions for training and validation; the remainder is test.

        """
        if self._wdn_graph is None:
            warnings.warn("No data loaded. Generating data with default settings")
            self.load_from_inp()

        g = self._wdn_graph
        assert g.X is not None and g.edge_index is not None and g.edge_attr is not None
        T = g.X.shape[0] if n_samples is None else min(n_samples, g.X.shape[0])

        X = g.X[:T].clone()
        ei = g.edge_index[:T].clone()
        ea = g.edge_attr[:T].clone()
        wnd = g.base_demands[:T].clone() if g.base_demands is not None else None

        train_end = int(train_ratio * T)
        val_end = int((train_ratio + val_ratio) * T)

        self._alldata_wds = _WDNGraph(
            X=X,
            edge_index=ei,
            edge_attr=ea,
        )
        self._train_wds = _WDNGraph(
            X=X[:train_end],
            edge_index=ei[:train_end],
            edge_attr=ea[:train_end],
        )
        self._val_wds = _WDNGraph(
            X=X[train_end:val_end],
            edge_index=ei[train_end:val_end],
            edge_attr=ea[train_end:val_end],
        )
        self._test_wds = _WDNGraph(
            X=X[val_end:],
            edge_index=ei[val_end:],
            edge_attr=ea[val_end:],
        )
        self._full_wds = _WDNGraph(X=X, edge_index=ei, edge_attr=ea, base_demands=wnd,
                                   time_interval=g.time_interval)
        # Slice gt flows for the test set (external comparison only)
        if self._gt_flows_raw is not None:
            self._gt_flows_test = self._gt_flows_raw[:T][val_end:]
        else:
            self._gt_flows_test = None

    # ------------------------------------------------------------------
    #  Model construction
    # ------------------------------------------------------------------

    def _compute_graph_diameter(self) -> int:
        g = self._wdn_graph
        assert g is not None and g.edge_index is not None
        G = nx.DiGraph()
        edge_list = [(u, v) for u, v in zip(*np.array(g.edge_index[0]))]
        G.add_edges_from(edge_list)
        try:
            return nx.diameter(G)
        except nx.NetworkXError:
            # Not strongly connected – use the undirected diameter
            return nx.diameter(G.to_undirected())

    def build_model(
        self,
        M_l: int = 128,
        I: int = 5,
        n_epochs: int = 1500,
        verbose: bool = True
    ) -> None:
        """
        Instantiate the PI_GNN model.

        Parameters
        ----------
        M_l : int
            Latent dimension (default 128).
        I   : int
            Number of GNN layers (default 5).
        n_epochs : int
            Number of training epochs (default 1500).
        verbose : bool
            Verbosity
        """
        dia = self._compute_graph_diameter()
        n_iter = (dia // I + 5) if dia // I > 1 else 5
        self.model = _PI_GNN(
            M_n=3, out_dim=1, M_e=2, M_l=M_l, I=I,
            aggr="max", dia=dia, n_iter=n_iter, bias=False, n_epochs=n_epochs,
        ).to(self.device)

        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        if verbose is True:
            print(f"PI_GNN  -  total params: {total:,}  trainable: {trainable:,}  diameter: {dia}")

    def get_model_params(self) -> dict:
        return {"I": self.model.I, "M_l": self.model.M_l,  "n_epochs": self.model.n_epochs}

    # ------------------------------------------------------------------
    #  Training
    # ------------------------------------------------------------------

    def train(
        self,
        n_epochs: int = 1500,
        lr: float = 1e-4,
        decay_step: int = 150,
        decay_rate: float = 0.75,
        rho: float = 0.1,
        delta: float = 0.1,
        grad_clip: float = 1e-6,
        save_dir: Optional[str] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Train the model.

        Parameters
        ----------
        n_epochs : int
            Number of training epochs.
        lr : float
            Learning rate.
        decay_step, decay_rate : LR scheduler params.
        rho, delta : Loss weighting coefficients.
        grad_clip : float
            Max gradient norm for clipping.
        save_dir : str, optional
            Where to save the model checkpoint.
            If None, the current working directory is used.

            The default is None.
        verbose : bool, optional
            Print progress.

            The default is True.

        Returns
        -------
        dict with keys ``"model_state"``, ``"model_path"``, ``"train_losses"``, ``"val_losses"``.
        """
        if self.model is None:
            raise RuntimeError("Model not built. Call build_model() first.")
        if self._train_wds is None:
            raise RuntimeError("Data not prepared. Call prepare_data() first.")

        model = self.model
        assert self._train_wds.X is not None
        assert self._val_wds is not None
        assert self._reservoirs is not None
        n_nodes = self._train_wds.X.shape[1]

        optimizer = Adam(model.parameters(), lr=lr, weight_decay=0.0, eps=1e-12)
        scheduler = lr_scheduler.MultiStepLR(
            optimizer, list(range(decay_step, decay_step * 1000, decay_step)), gamma=decay_rate
        )

        train_ds, _ = _load_dataset(self._train_wds, n_nodes, self._reservoirs, masked=True)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_ds, _ = _load_dataset(self._val_wds, n_nodes, self._reservoirs, masked=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=True)

        if save_dir is None:
            save_dir = os.path.join(os.getcwd(), str(datetime.date.today()))
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, "pi_gnn_model.pt")
        self._model_path = model_path

        all_train_losses: List[float] = []
        all_val_losses: List[float] = []
        log_every = max(1, n_epochs // 300)

        state = None
        for epoch in tqdm(range(n_epochs), disable=not verbose):
            torch.random.manual_seed(epoch + 42)
            epoch_losses = []

            for batch in train_loader:
                if self.train_with_rand_dias:
                    assert self.dia_dist in ("uniform", "normal"), "Invalid dia_dist; must be 'uniform' or 'normal'"
                    assert self.dia_dist_fac >= 0, "dia_dist_fac must be non-negative"
                    batch.edge_attr[:, 9:10] = _add_noise_to_diameters(batch.edge_attr[:, 9:10],
                                                                       self.dia_dist_fac,
                                                                       dist=self.dia_dist)
                    r = 10.667 * batch.edge_attr[..., 8:9] * \
                        torch.pow(batch.edge_attr[..., 10:11], -1.852) * \
                            torch.pow(batch.edge_attr[..., 9:10], -4.871)
                    r = torch.nan_to_num(r, nan=0, posinf=0, neginf=0)
                    batch.edge_attr[:, 0:1] = r
                if self.train_with_rand_demands:
                    assert self.dem_dist in ("uniform", "normal"), "Invalid dem_dist; must be 'uniform' or 'normal'"
                    assert self.dem_dist_fac >= 0, "dem_dist_fac must be non-negative"
                    batch.x[:, 1:2] = _add_noise_to_demands(batch.x[:, 1:2], self.dem_dist_fac,
                                                            dist=self.dem_dist)

                batch = batch.to(self.device)
                model.train()
                model.zero_grad()
                _ = model(batch, r_iter=5, epoch=epoch, zeta=1e-12)
                loss = model.loss(rho=rho, delta=delta)
                loss.backward()
                epoch_losses.append(loss.detach().cpu().item())

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()

            scheduler.step()
            all_train_losses.append(float(np.mean(epoch_losses)))

            # Validation
            if epoch % log_every == 0:
                model.eval()
                vloss_list = []
                for batch_val in val_loader:
                    if self.train_with_rand_dias:
                        batch_val.edge_attr[:, 9:10] = \
                            _add_noise_to_diameters(batch_val.edge_attr[:, 9:10], self.dia_dist_fac,
                                                    dist=self.dia_dist)
                        r = 10.667 * batch_val.edge_attr[..., 8:9] * \
                            torch.pow(batch_val.edge_attr[..., 10:11], -1.852) * \
                                torch.pow(batch_val.edge_attr[..., 9:10], -4.871)
                        r = torch.nan_to_num(r, nan=0, posinf=0, neginf=0)
                        batch_val.edge_attr[:, 0:1] = r
                    if self.train_with_rand_demands:
                        batch_val.x[:, 1:2] = _add_noise_to_demands(batch_val.x[:, 1:2],
                                                                    self.dem_dist_fac,
                                                                    dist=self.dem_dist)

                    batch_val = batch_val.to(self.device)
                    with torch.no_grad():
                        _ = model(batch_val, r_iter=5, epoch=epoch, zeta=1e-12)
                    vloss_list.append(model.loss(rho=rho, delta=delta).detach().cpu().item())

                mean_val = float(np.mean(vloss_list))
                all_val_losses.append(mean_val)
                if verbose:
                    tqdm.write(f"Epoch {epoch:5d}  train={all_train_losses[-1]:.8f}  val={mean_val:.8f}")

                # Checkpoint
                state = {"epoch": epoch, "model": model.state_dict(),
                         "optimizer": optimizer.state_dict()}
                torch.save(state, model_path)

        return {
            "model_state": state,
            "model_path": model_path,
            "train_losses": all_train_losses,
            "val_losses": all_val_losses,
        }

    # ------------------------------------------------------------------
    #  Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(
        self,
        model_path: Optional[str] = None,
        batch_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate the model on the test split.

        Parameters
        ----------
        model_path : str, optional
            Path to a saved checkpoint. If *None*, uses the last trained model.
        batch_size : int, optional
            Override batch size for evaluation.

        Returns
        -------
        dict with keys:
            ``"heads_pred"``   - predicted heads [T_test, N, 4]
            ``"heads_gt"``     - ground-truth heads [T_test, N]
            ``"demands_pred"`` - estimated demands [T_test, N, 4]
            ``"demands_gt"``   - ground-truth demands [T_test, N]
            ``"flows_pred"``   - estimated flows [T_test, E, 4]
            ``"flows_gt"``     - ground-truth flows [T_test, E] or ``None``
            ``"test_losses"``  - per-batch losses
        """
        if self.model is None:
            raise RuntimeError("Model not built.")
        if self._test_wds is None:
            raise RuntimeError("Data not prepared.")

        model = self.model
        assert self._test_wds.X is not None and self._test_wds.edge_attr is not None
        assert self._reservoirs is not None

        if model_path is not None:
            state = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state["model"])
        elif self._model_path is not None and os.path.exists(self._model_path):
            state = torch.load(self._model_path, map_location=self.device)
            model.load_state_dict(state["model"])

        model.eval()
        bs = batch_size or self.batch_size
        n_nodes = self._test_wds.X.shape[1]
        n_edges = self._test_wds.edge_attr[0].shape[0]

        test_ds, Y_test = _load_dataset(self._test_wds, n_nodes, self._reservoirs, masked=True)
        test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False)

        all_losses, Y_hat_all, D_star_all, D_hat_all, F_hat_all = [], [], [], [], []

        for batch in test_loader:
            batch = batch.to(self.device)

            y_hat = model(batch, r_iter=5, zeta=1e-12)
            loss = model.loss(rho=0.1, delta=0.1)
            all_losses.append(loss.detach().cpu().item())
            Y_hat_all.append(y_hat.detach().cpu())
            D_star_all.append(model.d_star.relu().detach().cpu())
            D_hat_all.append(model.d_tilde.relu().detach().cpu())
            F_hat_all.append(model.q_tilde.detach().cpu())

        Y_hat = torch.stack(torch.vstack(Y_hat_all).split(n_nodes))
        D_star = torch.stack(torch.vstack(D_star_all).split(n_nodes))
        D_hat = torch.stack(torch.vstack(D_hat_all).split(n_nodes))
        F_star = self._gt_flows_test  # [T_test, 2*E, 1] stored at prepare_data time
        F_hat = torch.stack(torch.vstack(F_hat_all).split(n_edges))[:, :n_edges//2, :]

        print(f"Test loss: {np.mean(all_losses):.8f}")

        return {
            "heads_pred": Y_hat,
            "heads_gt": Y_test[..., 0:1],
            "demands_pred": D_hat,
            "demands_gt": D_star,
            "flows_pred": F_hat,
            "flows_gt": F_star,
            "test_losses": all_losses,
        }

    # ------------------------------------------------------------------
    #  Prediction (no ground-truth required)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        reservoir_heads: np.ndarray,
        demands: np.ndarray,
        batch_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run forward inference on new data.

        Parameters
        ----------
        reservoir_heads : numpy.ndarray
            Heads at all reservoirs.
        demands : numpy.ndarray
            Demands (in m^3/s) at every node.

        Returns
        -------
        dict with ``"heads_pred"``, ``"demands_pred"``, ``"flows_pred"``.
        """
        if self.model is None:
            raise RuntimeError("Model not built.")

        #print(self._reservoirs)
        heads = np.zeros(demands.shape)
        for idx, res_idx in enumerate(self._reservoirs):
            heads[:, res_idx] = reservoir_heads[:, idx]
        self.load_from_arrays(heads, demands)

        g = self._wdn_graph
        model = self.model

        model.eval()
        bs = batch_size or self.batch_size
        assert g.X is not None and g.edge_attr is not None
        n_nodes = g.X.shape[1]
        n_edges = g.edge_attr[0].shape[0]

        ds, _ = _load_dataset(g, n_nodes, g.reservoirs, masked=True)
        loader = DataLoader(ds, batch_size=bs, shuffle=False)

        Y_hat_all, D_hat_all, F_hat_all = [], [], []

        for batch in loader:
            batch = batch.to(self.device)

            y_hat = model(batch, r_iter=5, zeta=1e-12)
            Y_hat_all.append(model.h_tilde.detach().cpu())
            D_hat_all.append(model.d_tilde.detach().cpu())
            F_hat_all.append(model.q_tilde.detach().cpu())

        return {
            "heads_pred": torch.stack(torch.vstack(Y_hat_all).split(n_nodes)),
            "demands_pred": torch.stack(torch.vstack(D_hat_all).split(n_nodes)),
            "flows_pred": torch.stack(torch.vstack(F_hat_all).split(n_edges))[:, :n_edges//2, :],
        }

    # ------------------------------------------------------------------
    #  Predicting and computing gradients (no ground-truth required)
    # ------------------------------------------------------------------

    def compute_gradients_from_forward_pass(
        self,
        reservoir_heads: np.ndarray,
        demands: np.ndarray,
        gradient_input: str = "demands",
        gradient_output: str = "heads",
        n_samples: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run forward inference on new data and compute gradients.

        Parameters
        ----------
        reservoir_heads : numpy.ndarray
            Heads at all reservoirs.
        demands : numpy.ndarray
            Demands (in m^3/s) at every node.
        gradient_input : `str`, optional
            Quantity of interest for the sensitivity analysis -- i.e.,
            target variable for backpropagation.
            Must be either "demands" or "diameters".

            The default is "demands".
        gradient_output : `str`, optional
            Source for the sensitivity analysis -- i.e., what is backpropagates.
            Must be either "flows" or "heads".

            The default is "heads".
        n_samples : int, optional
            Number of timesteps to compute gradients for. Defaults to all.
            Use a small value (e.g. 10-50) to speed up computation.

        Returns
        -------
        dict with ``"heads_pred"``, ``"demands_pred"``, ``"flows_pred"``, ``"grads"``.
        """
        if self.model is None:
            raise RuntimeError("Model not built.")

        # Build temporary graph
        heads = np.zeros(demands.shape)
        for idx, res_idx in enumerate(self._reservoirs):
            heads[:, res_idx] = reservoir_heads[:, idx]
        self.load_from_arrays(heads, demands)

        g = self._wdn_graph
        model = self.model

        model.eval()
        bs = 1
        assert g.X is not None and g.edge_attr is not None
        n_nodes = g.X.shape[1]
        n_edges = g.edge_attr[0].shape[0]

        T = g.X.shape[0] if n_samples is None else min(n_samples, g.X.shape[0])
        g_slice = _WDNGraph(
            X=g.X[:T],
            edge_index=g.edge_index[:T],
            edge_attr=g.edge_attr[:T],
            reservoirs=g.reservoirs,
        )
        ds, _ = _load_dataset(g_slice, n_nodes, g.reservoirs, masked=True)
        loader = DataLoader(ds, batch_size=bs, shuffle=False)

        Y_hat_all, D_hat_all, F_hat_all, Grads = [], [], [], []

        for batch in loader:
            batch = batch.to(self.device)

            # Create a proper leaf variable for the input we want to differentiate,
            # spliced into batch BEFORE the forward pass so it IS in the compute graph.
            if gradient_input == "demands":
                input = batch.x[:, 1:2].detach().clone().requires_grad_(True)
                batch.x = torch.cat((batch.x[:, :1], input, batch.x[:, 2:]), dim=1)
            elif gradient_input == "diameters":
                input = batch.edge_attr[:, 9:10].detach().clone().requires_grad_(True)
                r = 10.667 * batch.edge_attr[..., 8:9] * \
                    torch.pow(batch.edge_attr[..., 10:11], -1.852) * \
                        torch.pow(input, -4.871)
                r = torch.nan_to_num(r, nan=0, posinf=0, neginf=0)
                batch.edge_attr = torch.cat((r, batch.edge_attr[:, 1:9], input,
                                             batch.edge_attr[:, 10:]), dim=1)
            else:
                raise ValueError("Invalid gradient_input; must be 'demands' or 'diameters'")

            _ = model(batch, r_iter=5, zeta=1e-12)

            Y_hat_all.append(model.h_tilde.detach().cpu())
            D_hat_all.append(model.d_tilde.detach().cpu())
            F_hat_all.append(model.q_tilde.detach().cpu())

            if gradient_output == "heads":
                output = model.h_tilde  # heads
            elif gradient_output == "flows":
                output = model.q_tilde  # flows
            else:
                raise ValueError("Invalid gradient_output; must be 'heads' or 'flows'")

            grads = torch.zeros_like(output).unsqueeze(2).repeat(1, input.shape[0], 1)
            for i in range(output.shape[0]):
                g = torch.autograd.grad(inputs=input, outputs=output[i], allow_unused=True,
                                        create_graph=True, retain_graph=True)[0]
                if g is not None:
                    grads[i, :, 0:1] = g

            Grads.append(grads.detach().cpu())

            model.zero_grad()


        return {
            "heads_pred": torch.stack(torch.vstack(Y_hat_all).split(n_nodes)),
            "demands_pred": torch.stack(torch.vstack(D_hat_all).split(n_nodes)),
            "flows_pred": torch.stack(torch.vstack(F_hat_all).split(n_edges)),
            "grads": torch.stack(Grads),
        }

    # ------------------------------------------------------------------
    #  Computing gradient based on given target outputs
    # ------------------------------------------------------------------

    def compute_gradients(
        self,
        reservoir_heads: np.ndarray,
        demands: np.ndarray,
        gradient_input: str = "demands",  # "demands" or "diameters",
        gradient_output: str = "heads",  # "flows" or "heads"
        output_func: Callable = None,
        n_samples: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run forward inference on given data and compute gradients based on a
        custom function output function (e.g., loss function).

        Parameters
        ----------
        reservoir_heads : numpy.ndarray
            Heads at all reservoirs.
        demands : numpy.ndarray
            Demands (in m^3/s) at every node.
        gradient_input : `str`, optional
            Quantity of interest for the sensitivity analysis -- i.e.,
            target variable for backpropagation.
            Must be either "demands" or "diameters".

            The default is "demands".
        gradient_output : `str`, optional
            Source for the sensitivity analysis -- i.e., what is backpropagates.
            Must be either "flows" or "heads".

            The default is "heads".
        output_func : Callable,
            Function (e.g., custom loss function) applied to the output.
            Gradients will be computed through this function.

            If None, the identity function will be used -- note that in this case,
            this function behaves identical to 'compute_gradients_from_forward_pass'.

            The default is None.
        n_samples : int, optional
            Number of timesteps to compute gradients for. Defaults to all.
            Use a small value (e.g. 10-50) to speed up computation.

        Returns
        -------
        dict with ``"heads_pred"``, ``"demands_pred"``, ``"flows_pred"``, ``"grads"``.
        """
        if self.model is None:
            raise RuntimeError("Model not built.")

        heads = np.zeros(demands.shape)
        for idx, res_idx in enumerate(self._reservoirs):
            heads[:, res_idx] = reservoir_heads[:, idx]
        self.load_from_arrays(heads, demands)

        g = self._wdn_graph
        model = self.model

        model.eval()
        bs = 1
        assert g.X is not None and g.edge_attr is not None
        n_nodes = g.X.shape[1]
        n_edges = g.edge_attr[0].shape[0]

        T = g.X.shape[0] if n_samples is None else min(n_samples, g.X.shape[0])
        g_slice = _WDNGraph(
            X=g.X[:T],
            edge_index=g.edge_index[:T],
            edge_attr=g.edge_attr[:T],
            reservoirs=g.reservoirs,
        )
        ds, _ = _load_dataset(g_slice, n_nodes, g.reservoirs, masked=True)
        loader = DataLoader(ds, batch_size=bs, shuffle=False)

        Y_hat_all, D_hat_all, F_hat_all, Grads = [], [], [], []

        for batch in loader:
            batch = batch.to(self.device)

            # Create a proper leaf variable for the input we want to differentiate,
            # spliced into batch BEFORE the forward pass so it IS in the compute graph.
            if gradient_input == "demands":
                input = batch.x[:, 1:2].detach().clone().requires_grad_(True)
                batch.x = torch.cat((batch.x[:, :1], input, batch.x[:, 2:]), dim=1)
            elif gradient_input == "diameters":
                input = batch.edge_attr[:, 9:10].detach().clone().requires_grad_(True)
                r = 10.667 * batch.edge_attr[..., 8:9] * \
                    torch.pow(batch.edge_attr[..., 10:11], -1.852) * \
                        torch.pow(input, -4.871)
                r = torch.nan_to_num(r, nan=0, posinf=0, neginf=0)
                batch.edge_attr = torch.cat((r, batch.edge_attr[:, 1:9], input,
                                             batch.edge_attr[:, 10:]), dim=1)
            else:
                raise ValueError("Invalid gradient_input; must be 'demands' or 'diameters'")

            _ = model(batch, r_iter=5, zeta=1e-12)

            Y_hat_all.append(model.h_tilde.detach().cpu())
            D_hat_all.append(model.d_tilde.detach().cpu())
            F_hat_all.append(model.q_tilde.detach().cpu())

            if gradient_output == "heads":
                output = model.h_tilde  # heads
            elif gradient_output == "flows":
                output = model.q_tilde  # flows
            else:
                raise ValueError("Invalid gradient_output; must be 'heads' or 'flows'")

            if output_func is not None:
                output = output_func(output)

            grads = torch.zeros_like(output).unsqueeze(2).repeat(1, input.shape[0], 1)
            for i in range(output.shape[0]):
                g = torch.autograd.grad(inputs=input, outputs=output[i], allow_unused=True,
                                        create_graph=True, retain_graph=True)[0]
                if g is not None:
                    grads[i, :, 0:1] = g
                    
            Grads.append(grads.detach().cpu())

            model.zero_grad()

        return {
            "heads_pred": torch.stack(torch.vstack(Y_hat_all).split(n_nodes)),
            "demands_pred": torch.stack(torch.vstack(D_hat_all).split(n_nodes)),
            "flows_pred": torch.stack(torch.vstack(F_hat_all).split(n_edges)),
            "grads": torch.stack(Grads),
        }

    # ------------------------------------------------------------------
    #  Save / Load
    # ------------------------------------------------------------------

    def save_model(self, path: str) -> None:
        """Save model checkpoint."""
        if self.model is None:
            raise RuntimeError("No model to save.")
        torch.save({"model": self.model.state_dict(),
                    "model_params": self.get_model_params()}, path)

    def load_model(self, path: str) -> None:
        """Load model checkpoint."""
        state = torch.load(path, map_location=self.device)

        self.load_from_inp(simulation_duration_sec=3600)
        self.build_model(**state["model_params"], verbose=False)
        self.model.load_state_dict(state["model"])
        self._model_path = path

    # ------------------------------------------------------------------
    #  Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        parts = [f"PIGNNModel(inp_file={self.inp_file!r}"]
        if self._wdn_graph is not None:
            assert self._wdn_graph.X is not None
            T, N = self._wdn_graph.X.shape[:2]
            parts.append(f"  data: T={T}, N_nodes={N}")
        if self.model is not None:
            parts.append(f"  model: I={self.model.I}, M_l={self.model.M_l}, dia={self.model.dia}")
        parts.append(")")
        return "\n".join(parts)
