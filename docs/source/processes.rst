Statistical processes
======================

This module contains the implementation of different statistical processes that will be useful when pricing
derivatives by simulation.

Currently, the following statistical processes are implemented:

- :func:`GaussianWhiteNoise() <pricing.simulation.processes.GaussianWhiteNoise>`.
- :func:`GaussianWhiteNoise() <pricing.simulation.processes.Wiener>`.


API Documentation
-----------------

.. autoclass:: pricing.simulation.processes.GaussianWhiteNoise
    :show-inheritance:
    :members:
    :special-members: __init__
    :exclude-members: __str__, __setattr__
    :inherited-members:

.. autoclass:: pricing.simulation.processes.Wiener
    :show-inheritance:
    :members:
    :special-members: __init__
    :exclude-members: __str__, __setattr__
    :inherited-members:
