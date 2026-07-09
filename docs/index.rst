Welcome to EPyT-Control's documentation!
========================================

EPANET Python Toolkit Control -- EPyT-Control
+++++++++++++++++++++++++++++++++++++++++++++

EPyT-Control is a Python package building on top of `EPyT-Flow <https://github.com/WaterFutures/EPyT-Flow>`_ 
for implementing and evaluating control algorithms & strategies in water distribution networks
(WDN).

Besides related control tasks such as state estimation and event diagnosis, a special focus of this
Python package is Reinforcement Learning for data-driven control in WDNs and therefore it provides
full compatibility with the
`Stable-Baselines3 <https://stable-baselines3.readthedocs.io/en/master/>`_ package.


Unique Features
---------------

Unique features of EPyT-Control are the following:

- Support of hydraulic and (advanced) water quality simulation (i.e. EPANET and EPANET-MSX are supported)
- Compatibility with `Gymnasium <https://github.com/Farama-Foundation/Gymnasium>`_ and integration of `Stable-Baselines3 <https://stable-baselines3.readthedocs.io/en/master/>`_
- Wide variety of pre-defined actions (e.g. pump state actions, pump speed actons, valve state actions, species injection actions, etc.)
- Implementation of classic control aglorithms such as PID and LQR controllers
- Signal processing methods such as state estimation (e.g. Kalman filters) and event diagnosis
- State-of-the-art (pre-trained) hydraulic surrogate models
- High- and low-level interface
- Object-orientated design that is easy to extend and customize



.. toctree::
    :maxdepth: 2
    :caption: User Guide

    installation
    tut.basic_usage
    tut.create_env
    tut.pid_controller
    tut.hyd_surrogate


.. _tut.examples:

Examples
========
.. toctree::
   :maxdepth: 2
   :caption: Jupyter notebooks

   examples/basic_usage
   examples/pid_control
   examples/pump_control
   examples/event_detection



API Reference
=============
.. toctree::
    :maxdepth: 2
    :caption: API Reference

    epyt_control


Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
