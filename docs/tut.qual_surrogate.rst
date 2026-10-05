.. _tut.qual_surrogate:

***********************
Water Quality Surrogate
***********************

EPyT-Control also provides an implementation of a *single-species* water quality surrogate model
as proposed in [#f1]_. For this, two classes are available:
:class:`~epyt_control.models.qualsurrogate_mega_mp.quality_surrogate.QualitySurrogateModel`
providing an easy-to-use high-level interface and
:class:`~epyt_control.models.qualsurrogate_mega_mp.quality_surrogate.QualitySurrogateModelEx`
providing a low-level interface, offering full control over all parameters and outputs.

.. note::

    This surrogate model does not support networks with tanks.


The surrogate predicts the specie's concentration at every node based on the given hydraulics
(i.e., flow rates at every link) and concentration at source nodes (i.e., "injection").

.. code-block:: python

    # Load water distribution system and extract it's topology
    network_topo = ...

    # Run hydraulic simulation to get flow rates at every link
    scada_data = ....

    # Create new instance of the water quality surrogate
    surrogate = QualitySurrogateModel(network_topo)   # Topology of the network is needed

    # Predict concentrations based on the source concentrations and the hydraulics (flow rates)
    source_pattern = ....
    conc_pred = surrogate.predict({"1": source_pattern}, scada_data)  # Specify source concentration at node "1"
    print(conc_pred.shape)  # 1. dimension: time; 2. dimension: nodes


The low-level interface
:class:`~epyt_control.models.qualsurrogate_mega_mp.quality_surrogate.QualitySurrogateModelEx`
provides more control over all prameters and is also less dependent on EPyT-Flow -- e.g.,
you can pass flow rates as NumPy arrays to the surrogate.

More details and working examples can be found on the Jupyter Notebooks.

.. note::

    The surrogate dependencies (PyTorch and related packages) are not installed by default
    (unless you install the `all` option) and must be separately installed by installing
    the `qualsurrogate` option:

    .. code-block:: bash

        pip install epyt-control[qualsurrogate]


.. [#f1] "MeGA-MP: Metric Graph Advection Message Passing", Janine Strotherm, Luca Hermes, André Artelt, Barbara Hammer, Transactions of Machine Learning Research (2026), https://openreview.net/forum?id=2aZPFrKYYb