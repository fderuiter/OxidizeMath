# Pure Math

Rigorous abstract algebra, number theory, differential geometry, and statistical algorithms.

## 1. Overview & Architecture

The `pure_math` crate provides rigorous algorithmic implementations of foundational mathematical structures. Unlike domain crates that focus on applied physical phenomena, `pure_math` develops the core mathematical machinery from abstract algebra, analysis, and topology.

Implementations prioritize mathematical correctness, transparent algorithmic structure, and educational clarity, serving as canonical reference implementations across the OxidizeMath ecosystem.

### Module Structure

- **`algebra`**: Abstract algebraic structures and type hierarchies:
  - Groups, rings, and finite fields (`Fp`).
  - Polynomial rings and polynomial arithmetic over arbitrary fields.
- **`number_theory`**: Prime number generation (`primes_up_to`), primality testing, integer partition theory, and Q-series.
- **`analysis`**: Differential equation solvers (`OdeSystem`, Runge-Kutta steppers), 1D and zero-allocation 2D partial differential equation solvers (`pde::wave_2d::Wave2DSolver` for hyperbolic wave tanks, Kirchhoff-Love biharmonic Chladni plates, Chladni sand drift, Marching-Squares nodal contours, and pre-planned 2D FFT spectra), numerical integration, and calculus algorithms.
- **`differential_geometry`**: Manifolds, metric tensors, and Riemannian curvature.
- **`elliptic_curves`**: Zero-allocation short-Weierstrass curve group arithmetic ($P + Q$, $\[k\]P$, cyclic orbits) over $\mathbb{R}$ and prime fields $\mathbb{F}_p$, Frobenius traces $a_p$, Hasse-Weil verification, Breuer modular polynomial valuation bounds, and bridges to binary quadratic forms (`BQF`), Hurwitz-Kronecker class numbers, and modular $q$-series (`QSeries<i64>`).
- **`graph_theory`**: Network graph structures, centrality metrics, and traversal algorithms.
- **`statistics`**: Stochastic processes (Markov chains, Ornstein-Uhlenbeck), ranking systems (Glicko-2), and Kelly criterion decision models.
- **`special_functions`**: Classical mathematical functions (Bessel functions, Legendre polynomials).
- **`tilings`**: Aperiodic 2D tilings (`PenroseP2`, `PenroseP3`, `EinsteinHat`, `EinsteinSpectre`) with shared interior edge deduplication for vector fabrication.

---

## 2. Developer Instructions

### Prime Number Sieve and Finite Field Arithmetic

Generate prime sequences and compute polynomial arithmetic over finite fields:

```rust
# use pure_math::__doctest_facade as math_explorer;
use math_explorer::pure_math::algebra::fields::Fp;
use math_explorer::pure_math::algebra::polynomial::Polynomial;
use math_explorer::pure_math::number_theory::primes::primes_up_to;

// Generate prime numbers up to a specified limit
let primes = primes_up_to(20);
assert_eq!(primes, vec![2, 3, 5, 7, 11, 13, 17, 19]);

// Perform arithmetic in the finite field F_7
let a = Fp::<7>::new(3);
let b = Fp::<7>::new(5);
assert_eq!(a + b, Fp::<7>::new(1)); // (3 + 5) % 7 = 1
assert_eq!(a * b, Fp::<7>::new(1)); // (3 * 5) % 7 = 15 % 7 = 1

// Construct and sum polynomials over F_7
let p1 = Polynomial::new(vec![b, Fp::<7>::new(0), a]);
let q1 = Polynomial::new(vec![Fp::<7>::new(1), Fp::<7>::new(2)]);
let sum = p1 + q1;

assert_eq!(sum.coeffs[0], Fp::<7>::new(6));
assert_eq!(sum.coeffs[1], Fp::<7>::new(2));
assert_eq!(sum.coeffs[2], Fp::<7>::new(3));
```

---

## 3. Ecosystem Integration

The abstract structures in `pure_math` supply algorithms to domain engines (e.g., differential equation integrators in `domain_physics` and statistics in `domain_applied`) and interactive educational visualizers in `math_explorer_gui`.
