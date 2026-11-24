Computes modular exponentiation $r \equiv b^e \pmod{m}$ via binary square-and-multiply.

Reference: Menezes, van Oorschot & Vanstone (1996), *Handbook of Applied Cryptography*, Algorithm 2.143,
DOI: 10.1201/9781439821916; [`docs/math_specs/pure_math.md`](../pure_math.md).

# Stability
Reduces intermediate products modulo $|m|$ after every squaring and multiplication step, bounding
arbitrary-precision bit growth by $2\lceil \log_2 |m| \rceil$ bits.

# Domain of Validity
Requires non-negative exponent $e \ge 0$ and non-zero modulus $m \neq 0$; returns `NumberTheoryError` otherwise.

# Examples

```rust
# use pure_math::__doctest_facade as math_explorer;
use math_explorer::math_types::Integer;

let base = Integer::from(7);
let exp = Integer::from(13);
let modulus = Integer::from(19);
assert_eq!(base.pow_mod(&exp, &modulus).unwrap(), Integer::from(7));
```
