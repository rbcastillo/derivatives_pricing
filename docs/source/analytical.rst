Analytical pricing
=====================

This module contains pricing tools based on the applicable analytical formulae.

Currently, the following methodologies are implemented:

- :func:`ZeroCouponBond() <pricing.analytical.bonds.ZeroCouponBond>`.
- :func:`Forward() <pricing.analytical.forwards.Forward>`.
- :func:`EuropeanCall() <pricing.analytical.european_options.EuropeanCall>`.
- :func:`EuropeanPut() <pricing.analytical.european_options.EuropeanPut>`.

.. _zero_coupon_bond:

.. autoclass:: pricing.analytical.bonds.ZeroCouponBond
    :show-inheritance:
    :members:
    :special-members: __init__
    :inherited-members:

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
