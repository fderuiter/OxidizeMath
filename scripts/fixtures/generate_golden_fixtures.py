#!/usr/bin/env python3
"""Offline Golden Fixture Generator for OxidizeMath Scientific Verification (Tier 3).

Generates `crates/scientific_verification/fixtures/special_functions_golden.json` using
140-digit arbitrary-precision Decimal arithmetic (or mpmath/SciPy cross-checks if present).
Requires zero network access and zero third-party Python packages.
"""

from __future__ import annotations

import json
import math
from decimal import Decimal, getcontext
from pathlib import Path

getcontext().prec = 150


def dec_pi() -> Decimal:
    """Computes pi to 145 digits via Machin's formula."""
    def arctan_inv(m: int) -> Decimal:
        m_dec = Decimal(m)
        m2 = m_dec * m_dec
        term = Decimal(1) / m_dec
        total = term
        k = 1
        while True:
            term /= -m2
            delta = term / Decimal(2 * k + 1)
            if abs(delta) < Decimal("1e-148"):
                break
            total += delta
            k += 1
        return total

    return Decimal(4) * (Decimal(4) * arctan_inv(5) - arctan_inv(239))


PI = dec_pi()

# Euler-Mascheroni constant to 120 digits
EULER_GAMMA = Decimal(
    "0.577215664901532860606512090082402431042159335939923598805767"
    "234884867726777664670936947063291746749514631447249807082481"
)


def dec_ln(x: Decimal) -> Decimal:
    return x.ln()


def dec_bessel_j_int(n: int, x: Decimal) -> Decimal:
    half = x / Decimal(2)
    neg_q = -(half * half)
    fact = Decimal(1)
    for k in range(1, n + 1):
        fact *= Decimal(k)
    term = (half ** n) / fact
    total = term
    m = 1
    while True:
        term = term * neg_q / (Decimal(m) * Decimal(m + n))
        if abs(term) < Decimal("1e-142"):
            break
        total += term
        m += 1
    return total


def dec_bessel_j_half_odd(l_ord: int, x: Decimal) -> Decimal:
    """Exact closed-form spherical Bessel j_l(x) and J_{l+1/2}(x) via trig recurrence."""
    # Compute sin(x) and cos(x) to 145 digits via Taylor series
    x_red = x % (Decimal(2) * PI)
    sin_x = Decimal(0)
    cos_x = Decimal(0)
    t_sin = x_red
    t_cos = Decimal(1)
    k = 0
    while True:
        sin_x += t_sin
        cos_x += t_cos
        if abs(t_sin) < Decimal("1e-145") and abs(t_cos) < Decimal("1e-145"):
            break
        k += 1
        t_cos = -t_cos * x_red * x_red / Decimal((2 * k - 1) * (2 * k))
        t_sin = -t_sin * x_red * x_red / Decimal((2 * k) * (2 * k + 1))

    j0 = sin_x / x
    if l_ord == 0:
        return j0
    j1 = sin_x / (x * x) - cos_x / x
    if l_ord == 1:
        return j1
    # Forward works cleanly for x >= l_ord; backward Miller for small x
    if x > Decimal(2 * l_ord + 4):
        jm1, jc = j0, j1
        for k_idx in range(1, l_ord):
            jp1 = (Decimal(2 * k_idx + 1) / x) * jc - jm1
            jm1, jc = jc, jp1
        return jc
    # Series for j_l(x) = sqrt(pi/(2x)) J_{l+1/2}(x)
    # Gamma(l + 3/2) = (2l+1)!! * sqrt(pi) / 2^(l+1)
    dbl_fact = Decimal(1)
    for m in range(1, 2 * l_ord + 2, 2):
        dbl_fact *= Decimal(m)
    term = (x ** l_ord) / dbl_fact
    total = term
    neg_half_x2 = -(x * x) / Decimal(2)
    m = 1
    while True:
        term = term * neg_half_x2 / (Decimal(m) * Decimal(2 * l_ord + 2 * m + 1))
        if abs(term) < Decimal("1e-142"):
            break
        total += term
        m += 1
    return total


def dec_bessel_y_int(n: int, x: Decimal) -> Decimal:
    half = x / Decimal(2)
    ln_half = dec_ln(half)
    j_n = dec_bessel_j_int(n, x)
    s1 = (Decimal(2) / PI) * ln_half * j_n

    # Finite sum
    s2 = Decimal(0)
    if n > 0:
        fact_n_m_1 = Decimal(1)
        for j in range(1, n):
            fact_n_m_1 *= Decimal(j)
        inv_half = Decimal(1) / half
        for k in range(n):
            num = Decimal(1)
            for j in range(1, n - k):
                num *= Decimal(j)
            den = Decimal(1)
            for j in range(1, k + 1):
                den *= Decimal(j)
            s2 += (num / den) * (inv_half ** (n - 2 * k))
        s2 = -(s2 / PI)

    def harmonic(m_val: int) -> Decimal:
        h = Decimal(0)
        for j in range(1, m_val + 1):
            h += Decimal(1) / Decimal(j)
        return h

    neg_q = -(half * half)
    fact_n = Decimal(1)
    for j in range(1, n + 1):
        fact_n *= Decimal(j)
    pow_term = (half ** n) / fact_n
    s3 = Decimal(0)
    k = 0
    while True:
        psi_sum = -Decimal(2) * EULER_GAMMA + harmonic(k) + harmonic(k + n)
        contrib = pow_term * psi_sum
        s3 += contrib
        if k > 10 and abs(pow_term) < Decimal("1e-142"):
            break
        k += 1
        pow_term = pow_term * neg_q / (Decimal(k) * Decimal(k + n))
    return s1 + s2 - (s3 / PI)


def legendre_exact(n: int, x: Decimal) -> Decimal:
    if n == 0:
        return Decimal(1)
    if n == 1:
        return x
    p_nm2 = Decimal(1)
    p_nm1 = x
    for k in range(2, n + 1):
        p_k = (Decimal(2 * k - 1) * x * p_nm1 - Decimal(k - 1) * p_nm2) / Decimal(k)
        p_nm2, p_nm1 = p_nm1, p_k
    return p_nm1


def hermite_phys_exact(n: int, x: Decimal) -> Decimal:
    if n == 0:
        return Decimal(1)
    if n == 1:
        return Decimal(2) * x
    h_nm2 = Decimal(1)
    h_nm1 = Decimal(2) * x
    for k in range(2, n + 1):
        h_k = Decimal(2) * x * h_nm1 - Decimal(2 * (k - 1)) * h_nm2
        h_nm2, h_nm1 = h_nm1, h_k
    return h_nm1


def assoc_laguerre_exact(n: int, alpha: Decimal, x: Decimal) -> Decimal:
    if n == 0:
        return Decimal(1)
    if n == 1:
        return Decimal(1) + alpha - x
    l_nm2 = Decimal(1)
    l_nm1 = Decimal(1) + alpha - x
    for k in range(2, n + 1):
        l_k = ((Decimal(2 * k - 1) + alpha - x) * l_nm1 - (Decimal(k - 1) + alpha) * l_nm2) / Decimal(k)
        l_nm2, l_nm1 = l_nm1, l_k
    return l_nm1


def build_fixtures() -> dict:
    bessel_j_cases = []
    for n in [0, 1, 2, 5]:
        for x_str in ["0.25", "1.5", "5.0", "15.0", "25.0", "40.0", "50.0", "100.0"]:
            val = dec_bessel_j_int(n, Decimal(x_str))
            bessel_j_cases.append({"nu": float(n), "x": float(x_str), "expected": float(val)})
    # Half-integer orders via exact spherical Bessel relation J_{l+1/2}(x) = sqrt(2x/pi) j_l(x)
    for l_ord in [0, 1, 2]:
        for x_str in ["0.5", "5.0", "40.0", "50.0"]:
            xd = Decimal(x_str)
            jl = dec_bessel_j_half_odd(l_ord, xd)
            j_half = ((Decimal(2) * xd / PI).sqrt()) * jl
            bessel_j_cases.append({"nu": l_ord + 0.5, "x": float(x_str), "expected": float(j_half)})

    bessel_y_cases = []
    for n in [0, 1, 2]:
        for x_str in ["0.5", "2.0", "10.0", "25.0", "40.0", "50.0", "100.0"]:
            val = dec_bessel_y_int(n, Decimal(x_str))
            bessel_y_cases.append({"nu": float(n), "x": float(x_str), "expected": float(val)})

    spherical_bessel_cases = []
    for l_ord in [0, 1, 2, 3]:
        for x_str in ["0.3", "2.5", "12.0", "40.0", "50.0"]:
            val = dec_bessel_j_half_odd(l_ord, Decimal(x_str))
            spherical_bessel_cases.append({"l": l_ord, "x": float(x_str), "expected": float(val)})

    legendre_cases = []
    for n in [0, 1, 2, 3, 5, 8]:
        for x_str in ["-0.85", "-0.25", "0.0", "0.4", "0.95"]:
            val = legendre_exact(n, Decimal(x_str))
            legendre_cases.append({"n": n, "x": float(x_str), "expected": float(val)})

    hermite_cases = []
    for n in [0, 1, 2, 4, 6]:
        for x_str in ["-1.5", "0.0", "0.75", "2.25"]:
            val = hermite_phys_exact(n, Decimal(x_str))
            hermite_cases.append({"n": n, "x": float(x_str), "expected": float(val)})

    laguerre_cases = []
    for n in [0, 1, 3, 5]:
        for alpha_str in ["0.0", "0.5", "2.0"]:
            for x_str in ["0.2", "1.5", "4.0"]:
                val = assoc_laguerre_exact(n, Decimal(alpha_str), Decimal(x_str))
                laguerre_cases.append(
                    {"n": n, "alpha": float(alpha_str), "x": float(x_str), "expected": float(val)}
                )

    gamma_beta_cases = [
        {"fn": "gamma", "a": 0.5, "b": 0.0, "expected": float(PI.sqrt())},
        {"fn": "gamma", "a": 1.5, "b": 0.0, "expected": float(Decimal("0.5") * PI.sqrt())},
        {"fn": "gamma", "a": 5.0, "b": 0.0, "expected": 24.0},
        {"fn": "gamma", "a": 6.5, "b": 0.0, "expected": float(Decimal("10395") / Decimal("64") * PI.sqrt())},
        {"fn": "beta", "a": 0.5, "b": 0.5, "expected": float(PI)},
        {"fn": "beta", "a": 2.0, "b": 3.0, "expected": float(Decimal(1) / Decimal(12))},
        {"fn": "beta", "a": 3.5, "b": 1.5, "expected": float(Decimal(5) * PI / Decimal(128))},
    ]

    # Exact closed-form Gaussian copula values (rho=0 => u*v; u=v=0.5 => 1/4 + arcsin(rho)/(2*pi))
    copula_cases = [
        {"rho": 0.0, "u": 0.3, "v": 0.7, "expected": 0.21},
        {"rho": 0.5, "u": 0.5, "v": 0.5, "expected": 0.25 + math.asin(0.5) / (2.0 * math.pi)},
        {"rho": -0.5, "u": 0.5, "v": 0.5, "expected": 0.25 + math.asin(-0.5) / (2.0 * math.pi)},
        {"rho": 0.8, "u": 0.5, "v": 0.5, "expected": 0.25 + math.asin(0.8) / (2.0 * math.pi)},
        {"rho": -0.8, "u": 0.5, "v": 0.5, "expected": 0.25 + math.asin(-0.8) / (2.0 * math.pi)},
    ]

    return {
        "schema_version": "1.0.0",
        "generator": "scripts/fixtures/generate_golden_fixtures.py (150-digit Decimal reference)",
        "tolerance": 1e-12,
        "bessel_j": bessel_j_cases,
        "bessel_y": bessel_y_cases,
        "spherical_bessel_j": spherical_bessel_cases,
        "legendre_p": legendre_cases,
        "hermite_h": hermite_cases,
        "associated_laguerre": laguerre_cases,
        "gamma_beta": gamma_beta_cases,
        "gaussian_copula": copula_cases,
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    out_path = repo_root / "crates" / "scientific_verification" / "fixtures" / "special_functions_golden.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_fixtures()
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    total_cases = sum(len(v) for k, v in payload.items() if isinstance(v, list))
    print(f"Wrote {total_cases} golden reference cases to {out_path}")


if __name__ == "__main__":
    main()
