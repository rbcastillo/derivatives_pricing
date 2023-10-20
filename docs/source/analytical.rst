Analytical pricing
=====================

This module contains pricing tools based on the applicable analytical formulae.

Currently, the following methodologies are implemented:

- :func:`Forward() <pricing.analytical.forwards.Forward>`.
- :func:`EuropeanCall() <pricing.analytical.european_options.EuropeanCall>`.
- :func:`EuropeanPut() <pricing.analytical.european_options.EuropeanPut>`.


API Documentation
-----------------

.. autoclass:: pricing.analytical.forwards.Forward
    :show-inheritance:
    :members:
    :special-members: __init__
    :inherited-members:

.. autoclass:: pricing.analytical.european_options.EuropeanCall
    :show-inheritance:
    :members:
    :special-members: __init__
    :exclude-members: __str__, __setattr__
    :inherited-members:

.. autoclass:: pricing.analytical.european_options.EuropeanPut
    :show-inheritance:
    :members:
    :special-members: __init__
    :exclude-members: __str__, __setattr__
    :inherited-members:
