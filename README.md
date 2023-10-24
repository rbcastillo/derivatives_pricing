# Derivatives pricing

This is a personal project aimed at facilitating my own learning and providing a resource for others interested in this 
field. The objective is to build a comprehensive collection of derivative pricing tools and utilities, with a commitment 
to expanding it as my knowledge in this domain grows.

It's important to note that this project is not geared toward commercial applications or achieving the highest level of 
speed and efficiency in derivative pricing, as there are numerous existing solutions that fulfill these purposes. 
Instead, the primary goal is to create a code project where the calculations and underlying logic are presented in the 
most comprehensible manner possible. This approach is intended to assist individuals in grasping the theoretical 
foundations of derivative pricing.


Libraries:

![NumPy](https://img.shields.io/badge/NumPy-v1.24.4-green)
![SciPy](https://img.shields.io/badge/SciPy-v1.10.1-green)

Dependencies:

![Python](https://img.shields.io/badge/Python-v3.8-blue)

## Installation

This project can be cloned locally using the following command.

```bash
$ git clone https://github.com/rbcastillo/derivatives_pricing
```

## Usage

The project is intended to cover the key pricing aspects for a wide variety of derivatives. Consequently, the specific 
commands required for each use case will vary, contingent on the specific derivative under consideration.

To maintain a cohesive structure and enhance user-friendliness, a shared framework has been employed. This framework 
adheres to a consistent interface whenever possible, achieved through the implementation of an object-oriented 
programming (OOP) structure. This approach streamlines the experience for users dealing with diverse derivatives 
within the project.

The complete documentation for the project can be found <a href="https://derivatives-pricing.readthedocs.io/en/latest/index.html" target="_blank">here</a>.
Additionally, the following examples showcase typical uses of the tools within the project

**Example 1**\
Calculate the forward price for a forward contract where the spot price of the underlying is $100, the underlying does 
not pay dividends, the risk-free rate is 5% and the maturity of the contract is 5 years.

```python
from pricing.analytical.forwards import Forward

fw = Forward(s=100, r=0.05, t=5)
f_0 = fw.get_forward_price()
print(round(f_0, 9))  # Prints 128.402541669
```

**Example 2**\
Calculate the value and the delta of a European put with $120 strike price and 5 years to maturity, considering an
underlying with $100 spot price, volatility of 20% and dividend yield of 2% under a risk-free rate of 5%.

```python
from pricing.analytical.european_options import EuropeanPut

put = EuropeanPut(s=100, k=120, r=0.05, t=5, sigma=0.2, q=0.02)
put_price = put.price()
print(round(put_price, 9))  # Prints 17.800805857
put_delta = put.get_delta()
print(round(put_delta, 9))  # Prints -0.397998423
```

## License

[MIT](https://choosealicense.com/licenses/mit/)