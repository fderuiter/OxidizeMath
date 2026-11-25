#!/usr/bin/env julia
# Offline Julia BigFloat Reference Generator for OxidizeMath Golden Fixtures (Tier 3)
# Uses Julia Base `BigFloat` (256-bit precision) with zero external packages required.

setprecision(BigFloat, 256)

function besselj_big(n::Int, x::BigFloat)::BigFloat
    half = x / BigFloat(2)
    neg_q = -(half * half)
    term = (half ^ n) / factorial(BigInt(n))
    total = term
    m = 1
    while true
        term = term * neg_q / (BigFloat(m) * BigFloat(m + n))
        abs(term) < BigFloat("1e-70") && break
        total += term
        m += 1
    end
    return total
end

for n in (0, 1, 2, 5), x_s in ("0.25", "1.5", "5.0", "15.0", "25.0", "40.0", "50.0", "100.0")
    val = Float64(besselj_big(n, BigFloat(x_s)))
    println("bessel_j(nu=$(n), x=$(x_s)) = $(repr(val))")
end
