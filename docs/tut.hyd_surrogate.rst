.. _tut.hyd_surrogate:

********************
Hydraulic Surrogates
********************

EPyT-Control also provides an implementation of an AI-based hydraulic surrogate model
as proposed in [#f1]_.
The surrogate model is implemented in the :class:`~epyt_control.models.pi_gnn_hydsurrogate.PIGNNModel`
class and can, once trained on a specific network, predict pressures and flow rates everywhere in the
network based on given demands at nodes. 
For making predictions, the function
:func:`~epyt_control.models.pi_gnn_hydsurrogate.PIGNNModel.predict` takes the reservoirs' heads and
demands at every node as an input and outputs flow rates and pressures everywhere.
For the users' convienice, there also exist a predict function (see :func:`~epyt_control.models.pi_gnn_hydsurrogate.PIGNNModel.predict_as_numpy`),
which returns everything as NumPy arrays,
and another one (see :func:`~epyt_control.models.pi_gnn_hydsurrogate.PIGNNModel.predict_as_scada_data`),
which returns the results as an
`epyt_flow.simulation.ScadaData <https://epyt-flow.readthedocs.io/en/stable/epyt_flow.simulation.scada.html#epyt_flow.simulation.scada.scada_data.ScadaData>`_
instance.
Furthermore, we can compute gradients w.r.t. to demands and the network topology
(e.g., pipe diameter) -- see function
:func:`~epyt_control.models.pi_gnn_hydsurrogate.PIGNNModel.compute_gradients`.

.. note::

    This surrogate model does not support networks with tanks.

Creating a new surrogate model for a specific water distribution network, requires passing
the .inp file the constructor of :class:`~epyt_control.models.pi_gnn_hydsurrogate.PIGNNModel`.

.. code-block:: python

    # Create a new hydraulic surrogate model for Anytown.inp
    surrogate = PIGNNModel("Anytown.inp") 

    ....

For the user's convienice, EPyT-Control already ships pre-trained surrogate models for
some popular networks -- those can be loaded by calling the static function
:func:`~epyt_control.models.pi_gnn_hydsurrogate.PIGNNModel.from_network`:

- `Anytown <https://waterfutures.github.io/WaterBenchmarkHub/benchmarks/network-Anytown.html>`_
- `Hanoi <https://waterfutures.github.io/WaterBenchmarkHub/benchmarks/network-Hanoi.html>`_
- `Blacksburg <https://waterfutures.github.io/WaterBenchmarkHub/benchmarks/network-blacksburg.html>`_
- `Fossolo <https://waterfutures.github.io/WaterBenchmarkHub/benchmarks/network-Fossolo.html>`_
- `Fourteen Pipe <https://waterfutures.github.io/WaterBenchmarkHub/benchmarks/network-Fourteen-Pipe.html>`_
- `Federally Owned Water Main system (FOWM) <https://waterfutures.github.io/WaterBenchmarkHub/benchmarks/network-FOWM.html>`_
- `Jilin <https://waterfutures.github.io/WaterBenchmarkHub/benchmarks/network-Jilin.html>`_
- `KL <https://waterfutures.github.io/WaterBenchmarkHub/benchmarks/network-KL.html>`_
- `Modena <https://waterfutures.github.io/WaterBenchmarkHub/benchmarks/network-Modena.html>`_
- `Nineteen Pipe <https://waterfutures.github.io/WaterBenchmarkHub/benchmarks/network-Nineteen-Pipe.html>`_
- `Modified Nineteen Pipe <https://waterfutures.github.io/WaterBenchmarkHub/benchmarks/network-Modified-Nineteen-Pipe.html>`_
- `New York City Tunnel <https://waterfutures.github.io/WaterBenchmarkHub/benchmarks/network-NYC-Tunnel.html>`_
- `Pescara Network <https://waterfutures.github.io/WaterBenchmarkHub/benchmarks/network-pescara.html>`_
- `Zhi Jiang <https://waterfutures.github.io/WaterBenchmarkHub/benchmarks/network-Zhi-Jiang.html>`_


.. code-block:: python

    # Create a new hydrauic surrogate model for Anytown
    model = PIGNNModel.from_network("anytown")  # Note: Only works for pre-trained surrogate that come with EPyT-Control

    ...
  
Pre-trained hydraulic surrogate models can be loaded by setting
`load_pretrained_model=True` when calling
:func:`~epyt_control.models.pi_gnn_hydsurrogate.PIGNNModel.load_model`.
Alternatively, you can load custom weights by using the
:func:`~epyt_control.models.pi_gnn_hydsurrogate.PIGNNModel.load_model` function.


.. code-block:: python

    # Create and load a pre-trained hydrauic surrogate model for Anytown
    model = PIGNNModel.from_network("anytown")#, load_pretrained_model=True)
    model.load_model("anytown_pignn.pt")  # Alternatively, you can load your own pre-trained weights

    # Create test data
    heads, reservoir_idx, demands, flows = ....

    # Use the hydraulic surrogate model to predict flow rates and heads
    hyd_pred = model.predict(reservoir_heads=heads[:, reservoir_idx],
                             demands=demands)

Otherwise, training a surrogate model is done by calling the
:func:`~epyt_control.models.pi_gnn_hydsurrogate.PIGNNModel.train` function after the model has been
build (see :func:`~epyt_control.models.pi_gnn_hydsurrogate.PIGNNModel.build_model` function).
For the training itself, training data can be automatically generate by calling
:func:`~epyt_control.models.pi_gnn_hydsurrogate.PIGNNModel.load_from_inp` followed by a call of
:func:`~epyt_control.models.pi_gnn_hydsurrogate.PIGNNModel.prepare_data`.

.. code-block:: python
    # Create new hydraulic surrogate model
    model = .....

    # Generate test data and training demands from the .inp file
    model.load_from_inp()
    model.prepare_data(train_ratio=0.6, val_ratio=0.2)

    # Build the neural network
    model.build_model()

    # Train the model
    model.train()


More details and working examples can be found in the Jupyter Notebooks.

.. note::

    The surrogate dependencies (PyTorch and related packages) are not installed by default
    (unless you install the `all` option) and must be separately installed by installing
    the `hydsurrogate` option:

    .. code-block:: bash

        pip install epyt-control[hydsurrogate]

.. rubric::

.. [#f1] I. Ashraf, A. Artelt, B. Hammer, "Scalable and Robust Physics-Informed Graph Neural Networks for Water Distribution Systems", IEEE International Joint Conference on Neural Networks, 2025 doi: 10.1109/IJCNN64981.2025.11229349
