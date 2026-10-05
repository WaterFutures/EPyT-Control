"""
This module contains an implementation of the water quality surrogate model proposed in
"MeGA-MP: Metric Graph Advection Message Passing", Janine Strotherm, Luca Hermes, André Artelt, Barbara Hammer,
Transactions of Machine Learning Research (2026), https://openreview.net/forum?id=2aZPFrKYYb.
"""
from .quality_surrogate import QualitySurrogateModel, QualitySurrogateModelEx
from .utils import get_edge_attribute, make_edge_index, flow_to_velocity
