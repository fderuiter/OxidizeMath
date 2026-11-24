Computes the multiplicative modular inverse $x \equiv a^{-1} \pmod{m}$ satisfying $a x \equiv 1 \pmod{m}$.

Reference: Menezes, van Oorschot & Vanstone (1996), *Handbook of Applied Cryptography*, Algorithm 2.142,
DOI: 10.1201/9781439821916; [`docs/math_specs/pure_math.md`](../pure_math.md).

# Stability
Executes the exact Extended Euclidean algorithm over arbitrary-precision integers with monotonically decreasing remainders.

# Domain of Validity
Exists and succeeds if and only if $m > 1$ and $\gcd(a, m) = 1$; returns `NumberTheoryError` when $a$ and $m$ are not coprime or $m \le 0$.
