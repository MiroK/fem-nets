# Finite Element Neural Networks

Representation of some finite element function spaces (as defined in
FEniCS) in terms of neural networks. That is, we construct neural networks
whose weights are the coefficient vectors (as ordered in FEniCS)

## Dependencies
- `FEniCS` (2019.1.0 and higher) stack
- `pytorch`
- [`gmshnics`](https://github.com/MiroK/gmshnics) for some tests

## Usage
See [tests](https://github.com/MiroK/fem-nets/blob/master/test/test_lagrange1.py#L36) for example

## TODO
- [ ] Suport for 1, 2, 3 d
- [ ] Support for Discontinuous Lagrange
- [ ] Support for higher order
- [ ] Tensor values spaces (symmetric and skew because why not)
- [ ] convenience functions `to_nn(V)` where `V` is a function space
