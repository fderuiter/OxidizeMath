# IEEE-754 Numerical Primitives & Series Kernel Specification (`spec:math_commons_v1`)

- **Spec ID**: `spec:math_commons_v1`
- **Traceability Module Key**: `math_commons`

## Governing Equations

Zero-copy 2D sample series $\{(x_i, y_i)\}_{i=0}^{n-1}$ support indexed coordinate projections $\pi_x(p_i) = x_i, \pi_y(p_i) = y_i$, plateau-run detection on $[x_{\min}, x_{\max}]$, and weighted least-squares / critical-point primitives:

$$
\mathcal{S} = \bigl\{(x_i, y_i) \in \mathbb{R}^2 : 0 \le i < n\bigr\}, \qquad r(i_0) = \min\bigl\{k \ge i_0 : k = n \;\lor\; y_k \neq y_{\mathrm{flat}} \;\lor\; x_k \notin [x_{\min}, x_{\max}]\bigr\}
$$

## Numerical Stability & Error Bounds

- Coordinate projections perform zero arithmetic rounding ($0\,\mathrm{ULP}$ exact pass-through of IEEE-754 `f64` values) with zero heap allocations.
- Chunked 4-wide branchless plateau predicates `(y == flat_y) & (x >= x_min) & (x <= x_max)` produce bit-identical results to scalar sequential scans and safely evaluate `false` when encountering IEEE-754 `NaN` values.

## Domain of Validity

- Valid sample index $0 \le i < n$ (out-of-bounds indexing panics on slice-backed containers; paired-slice views clamp length to $\min(|X|, |Y|)$) and ordered bounds $x_{\min} \le x_{\max}$ for non-empty plateau intervals.

## References

- `@goldberg1991every` — Goldberg (1991), *What Every Computer Scientist Should Know About Floating-Point Arithmetic*, `DOI:10.1145/103162.103163`.
