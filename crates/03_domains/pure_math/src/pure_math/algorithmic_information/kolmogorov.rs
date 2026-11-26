//! Computable upper bounds and estimators for Kolmogorov complexity $K(x)$.
//!
//! Implements Elias-style prefix complexity bounds, precision-$r$ dyadic point complexity
//! $K(\lfloor p \rfloor_r)$, Lempel-Ziv (`LZ76`) exhaustive parsing complexity, 1D Block
//! Decomposition Method (`BDM`) complexity, and empirical 0th-order Shannon information.

use crate::math_types::Integer;
use crate::pure_math::algorithmic_information::geometry::{DyadicPoint2D, Point2D};
use dashu::base::{Abs, BitTest};
use dashu::integer::IBig;
use std::collections::BTreeMap;

/// Maximum binary block width used in 1D Block Decomposition Method (`BDM`) estimation.
const BDM_BLOCK_SIZE: usize = 12;

/// Maximum sliding lookback window for `LZ76` phrase matching on long bitstrings.
const LZ76_MAX_LOOKBACK: usize = 2048;

/// Computes $\log_2(|n|)$ for an arbitrary-precision positive integer without `f64` overflow.
#[verified_engine::verified]
fn log2_positive_ibig(mag: &IBig) -> f64 {
    if let Ok(small) = u64::try_from(mag) {
        return (small.max(1) as f64).log2();
    }
    let bit_len = mag.bit_len();
    let shift = bit_len.saturating_sub(53);
    let top_bits = mag >> shift;
    let mantissa = u64::try_from(&top_bits).unwrap_or(u64::MAX) as f64;
    mantissa.log2() + (shift as f64)
}

/// Computes the Elias prefix Kolmogorov complexity upper bound
/// $\log_2(n) + 2\log_2(\log_2(n)) + 1$ for positive integers $n \ge 2$ (returning `1.0` for $n \le 1$).
///
/// # Examples
///
/// ```rust
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::math_types::Integer;
/// use math_explorer::pure_math::algorithmic_information::kolmogorov::prefix_kolmogorov_approx;
///
/// let bound = prefix_kolmogorov_approx(&Integer::from(10));
/// assert!(bound > 7.7 && bound < 7.8);
/// ```
#[must_use]
#[verified_engine::verified]
pub fn prefix_kolmogorov_approx(n: &Integer) -> f64 {
    if n.0 <= IBig::from(1u8) {
        return 1.0;
    }
    let log_n = log2_positive_ibig(&n.0);
    if log_n <= 1.0 {
        return log_n + 1.0;
    }
    log_n + 2.0 * log_n.log2() + 1.0
}

/// Approximates the prefix Kolmogorov complexity $K(n)$ in bits by rounding [`prefix_kolmogorov_approx`].
///
/// # Examples
///
/// ```rust
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::math_types::Integer;
/// use math_explorer::pure_math::algorithmic_information::kolmogorov::kolmogorov_complexity;
///
/// let k = kolmogorov_complexity(&Integer::from(1_000_000));
/// assert_eq!(k, 30);
/// ```
#[must_use]
#[verified_engine::verified]
pub fn kolmogorov_complexity(n: &Integer) -> u64 {
    let mag = Integer(n.0.clone().abs());
    let approx = prefix_kolmogorov_approx(&mag);
    if !approx.is_finite() || approx <= 1.0 {
        1
    } else {
        approx.round() as u64
    }
}

/// Bijectively maps a signed 64-bit integer to a non-negative zigzag integer in $\mathbb{N}_0$.
#[verified_engine::verified]
fn zigzag_encode_i64(value: i64) -> IBig {
    if value >= 0 {
        IBig::from(u128::from(value as u64) * 2)
    } else {
        let abs_minus_one = value.unsigned_abs() - 1;
        IBig::from(u128::from(abs_minus_one) * 2 + 1)
    }
}

/// Computes Cantor's bijection $\pi(a, b) = \frac{(a + b)(a + b + 1)}{2} + b$ on $\mathbb{N}_0^2$.
#[verified_engine::verified]
fn cantor_pair(a: &IBig, b: &IBig) -> IBig {
    let sum = a + b;
    let sum_plus_one = &sum + IBig::from(1u8);
    ((&sum * &sum_plus_one) / IBig::from(2u8)) + b
}

/// Approximates the precision-$r$ Kolmogorov complexity $K(\lfloor p \rfloor_r)$ of a 2D point.
///
/// Quantizes `point` to its reduced dyadic floor $\lfloor p \rfloor_r$, encodes the reduced
/// coordinate pairs via Cantor pairing, and evaluates [`prefix_kolmogorov_approx`].
///
/// # Examples
///
/// ```rust
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::algorithmic_information::geometry::Point2D;
/// use math_explorer::pure_math::algorithmic_information::kolmogorov::k_floor_r;
///
/// let origin = Point2D::from_ints(0, 0);
/// let non_dyadic = Point2D::new(
///     math_explorer::math_types::Rational::from((1, 3)),
///     math_explorer::math_types::Rational::from((1, 5)),
/// );
/// assert!(k_floor_r(&origin, 16) < k_floor_r(&non_dyadic, 16));
/// ```
#[must_use]
#[verified_engine::verified]
pub fn k_floor_r(point: &Point2D, r: u32) -> f64 {
    let dyadic = DyadicPoint2D::from_point_floor(point, r);
    let rx = dyadic.x.reduced();
    let ry = dyadic.y.reduced();
    let code_x = cantor_pair(&zigzag_encode_i64(rx.m), &IBig::from(rx.r));
    let code_y = cantor_pair(&zigzag_encode_i64(ry.m), &IBig::from(ry.r));
    let combined = Integer(cantor_pair(&code_x, &code_y) + IBig::from(1u8));
    prefix_kolmogorov_approx(&combined)
}

/// Extracts the big-endian binary digits (`0` or `1`) of $|n|$.
#[must_use]
#[verified_engine::verified]
pub fn integer_to_bits(n: &Integer) -> Vec<u8> {
    let mag = n.0.clone().abs();
    let bit_len = mag.bit_len();
    if bit_len == 0 {
        return vec![0];
    }
    let mut bits = Vec::with_capacity(bit_len);
    for idx in (0..bit_len).rev() {
        let bit = u8::from(mag.bit(idx));
        bits.push(bit);
    }
    bits
}

/// Encodes active prime-exponent registers into a self-delimiting binary bitstring.
#[must_use]
#[verified_engine::verified]
pub fn exponents_to_bits(exponents: &[u32]) -> Vec<u8> {
    let active_len = exponents
        .iter()
        .rposition(|&e| e > 0)
        .map_or(0, |idx| idx + 1);
    if active_len == 0 {
        return vec![0];
    }
    let mut bits = Vec::with_capacity(active_len * 4);
    for &exp in &exponents[..active_len] {
        if exp == 0 {
            bits.push(0);
        } else {
            let width = u32::BITS - exp.leading_zeros();
            bits.push(1);
            for shift in (0..width).rev() {
                bits.push(((exp >> shift) & 1) as u8);
            }
        }
    }
    bits
}

/// Normalizes raw byte/bit slices (accepting both `{0, 1}` and ASCII `{b'0', b'1'}`) into `{0, 1}`.
#[verified_engine::verified]
fn normalize_bits(bits: &[u8]) -> Vec<u8> {
    bits.iter()
        .map(|&b| u8::from(b != 0 && b != b'0'))
        .collect()
}

/// Computes total empirical 0th-order Shannon information $N \cdot H_0(x)$ in bits.
#[must_use]
#[verified_engine::verified]
pub fn shannon_entropy_bits(bits: &[u8]) -> f64 {
    if bits.is_empty() {
        return 0.0;
    }
    let n = bits.len() as f64;
    let ones = bits.iter().filter(|&&b| b != 0 && b != b'0').count() as f64;
    if ones <= 0.0 || ones >= n {
        return 0.0;
    }
    let p1 = ones / n;
    let p0 = 1.0 - p1;
    let h = -(p0 * p0.log2() + p1 * p1.log2());
    n * h
}

/// Finds the longest prefix of `norm[pos..]` that already occurs starting before `pos + match_len`.
#[verified_engine::verified]
fn longest_lz76_match(norm: &[u8], pos: usize) -> usize {
    let n = norm.len();
    let search_start = pos.saturating_sub(LZ76_MAX_LOOKBACK);
    let mut best = 0usize;
    for candidate in search_start..pos {
        let mut len = 0usize;
        while pos + len < n && norm[candidate + len] == norm[pos + len] {
            len += 1;
        }
        if len > best {
            best = len;
        }
    }
    best
}

/// Counts exhaustive Lempel-Ziv (`LZ76`) production phrases $c(x)$ in a binary sequence.
#[must_use]
#[verified_engine::verified]
pub fn lz76_phrases(bits: &[u8]) -> usize {
    let norm = normalize_bits(bits);
    let n = norm.len();
    if n == 0 {
        return 0;
    }
    let mut phrases = 0usize;
    let mut pos = 0usize;
    while pos < n {
        let longest = longest_lz76_match(&norm, pos);
        let step = (longest + 1).min(n - pos);
        phrases += 1;
        pos += step;
    }
    phrases
}

/// Computes the Lempel-Ziv (`LZ76`) Kolmogorov complexity upper bound $c(x)\log_2 \max(|x|, 2)$ in bits.
///
/// # Examples
///
/// ```rust
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::algorithmic_information::kolmogorov::lz76_complexity;
///
/// let zeros = vec![0u8; 64];
/// let irregular = b"0110100110010110100101100110100110010110011010010110100110010110";
/// assert!(lz76_complexity(&zeros) < lz76_complexity(irregular));
/// ```
#[must_use]
#[verified_engine::verified]
pub fn lz76_complexity(bits: &[u8]) -> f64 {
    if bits.is_empty() {
        return 0.0;
    }
    if bits.len() == 1 {
        return 1.0;
    }
    let phrases = lz76_phrases(bits) as f64;
    phrases * (bits.len() as f64).log2()
}

/// Reverses the lowest `len` bits of `word`.
#[verified_engine::verified]
fn reverse_low_bits(word: u16, len: usize) -> u16 {
    let mut acc = 0u16;
    for i in 0..len {
        let bit = (word >> i) & 1;
        acc = (acc << 1) | bit;
    }
    acc
}

/// Canonicalizes a binary block of length `1..=12` under complement and reversal symmetries.
#[verified_engine::verified]
fn canonical_block_key(word: u16, len: usize) -> u16 {
    let mask = if len >= 16 {
        u16::MAX
    } else {
        (1u16 << len) - 1
    };
    let w = word & mask;
    let comp = (!w) & mask;
    let rev = reverse_low_bits(w, len);
    let rev_comp = reverse_low_bits(comp, len);
    w.min(comp).min(rev).min(rev_comp)
}

/// Estimates the Coding Theorem Method (`CTM`) micro-complexity of a short block (`1 <= len <= 12`).
#[verified_engine::verified]
fn ctm_block_complexity(word: u16, len: usize) -> f64 {
    let canon = canonical_block_key(word, len);
    let mut period = len;
    for p in 1..len {
        let mut periodic = true;
        for idx in p..len {
            if ((canon >> idx) & 1) != ((canon >> (idx % p)) & 1) {
                periodic = false;
                break;
            }
        }
        if periodic {
            period = p;
            break;
        }
    }
    let mut transitions = 0usize;
    for idx in 1..len {
        if ((canon >> idx) & 1) != ((canon >> (idx - 1)) & 1) {
            transitions += 1;
        }
    }
    let motif_cost = 1.5 + (period as f64) * 0.82;
    let repeat_cost = ((len as f64) / (period as f64)).log2();
    let transition_cost = (transitions as f64) * 0.22;
    motif_cost + repeat_cost + transition_cost
}

/// Computes the 1D Block Decomposition Method (`BDM`) Kolmogorov complexity estimate
/// $\sum_b (\text{CTM}(b) + \log_2 m_b)$ in bits over non-overlapping blocks of up to 12 bits.
///
/// # Examples
///
/// ```rust
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::algorithmic_information::kolmogorov::bdm_complexity;
///
/// let periodic = b"010101010101010101010101010101010101010101010101";
/// let complex = b"011010011100101100011010110100101110010101101100";
/// assert!(bdm_complexity(periodic) < bdm_complexity(complex));
/// ```
#[must_use]
#[verified_engine::verified]
pub fn bdm_complexity(bits: &[u8]) -> f64 {
    let norm = normalize_bits(bits);
    if norm.is_empty() {
        return 0.0;
    }
    let mut counts: BTreeMap<(u8, u16), usize> = BTreeMap::new();
    let mut pos = 0usize;
    while pos < norm.len() {
        let len = (norm.len() - pos).min(BDM_BLOCK_SIZE);
        let mut word = 0u16;
        for &bit in &norm[pos..pos + len] {
            word = (word << 1) | u16::from(bit);
        }
        let canon = canonical_block_key(word, len);
        *counts.entry((len as u8, canon)).or_insert(0) += 1;
        pos += len;
    }
    let mut total = 0.0;
    for (&(len_u8, canon), &multiplicity) in &counts {
        total += ctm_block_complexity(canon, usize::from(len_u8)) + (multiplicity as f64).log2();
    }
    total
}

// [cite:algorithmic_information_rust]
