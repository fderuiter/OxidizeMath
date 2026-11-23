//! Joe-Kuo (`new-joe-kuo-6.21201`, 2008/2010) primitive polynomials and initial direction numbers.

/// Maximum coordinate dimension supported by [`super::SobolSequence`].
pub const MAX_SOBOL_DIM: usize = 32;

/// Number of binary bits per 32-bit Sobol direction-number table row.
pub(crate) const SOBOL_BITS: usize = 32;

/// Primitive polynomial over GF(2) and odd initial direction numbers $m_1, \dots, m_s$ for one axis.
#[derive(Debug, Clone, Copy)]
pub(crate) struct JoeKuoPoly {
    /// Polynomial degree $s \in 1..=7$.
    pub degree: usize,
    /// Bitmask of interior coefficients $a_1 \dots a_{s-1}$ for $x^s + a_1 x^{s-1} + \dots + a_{s-1} x + 1$.
    pub coeffs: u32,
    /// Odd initial direction numbers $m_1, \dots, m_s$ satisfying $m_i < 2^i$.
    pub initial_m: &'static [u32],
}

/// Joe-Kuo `new-joe-kuo-6.21201` parameters for dimensions `2..=32` (31 entries; dimension 1 uses $m_k = 1$).
pub(crate) const JOE_KUO_POLYS_D2_TO_D32: [JoeKuoPoly; MAX_SOBOL_DIM - 1] = [
    JoeKuoPoly {
        degree: 1,
        coeffs: 0,
        initial_m: &[1],
    },
    JoeKuoPoly {
        degree: 2,
        coeffs: 1,
        initial_m: &[1, 3],
    },
    JoeKuoPoly {
        degree: 3,
        coeffs: 1,
        initial_m: &[1, 3, 1],
    },
    JoeKuoPoly {
        degree: 3,
        coeffs: 2,
        initial_m: &[1, 1, 1],
    },
    JoeKuoPoly {
        degree: 4,
        coeffs: 1,
        initial_m: &[1, 1, 3, 3],
    },
    JoeKuoPoly {
        degree: 4,
        coeffs: 4,
        initial_m: &[1, 3, 5, 13],
    },
    JoeKuoPoly {
        degree: 5,
        coeffs: 2,
        initial_m: &[1, 1, 5, 5, 17],
    },
    JoeKuoPoly {
        degree: 5,
        coeffs: 4,
        initial_m: &[1, 1, 5, 5, 5],
    },
    JoeKuoPoly {
        degree: 5,
        coeffs: 7,
        initial_m: &[1, 1, 7, 11, 19],
    },
    JoeKuoPoly {
        degree: 5,
        coeffs: 11,
        initial_m: &[1, 1, 5, 1, 1],
    },
    JoeKuoPoly {
        degree: 5,
        coeffs: 13,
        initial_m: &[1, 1, 1, 3, 11],
    },
    JoeKuoPoly {
        degree: 5,
        coeffs: 14,
        initial_m: &[1, 3, 5, 5, 31],
    },
    JoeKuoPoly {
        degree: 6,
        coeffs: 1,
        initial_m: &[1, 3, 3, 9, 7, 49],
    },
    JoeKuoPoly {
        degree: 6,
        coeffs: 13,
        initial_m: &[1, 1, 1, 15, 21, 21],
    },
    JoeKuoPoly {
        degree: 6,
        coeffs: 16,
        initial_m: &[1, 3, 1, 13, 27, 49],
    },
    JoeKuoPoly {
        degree: 6,
        coeffs: 19,
        initial_m: &[1, 1, 1, 15, 7, 5],
    },
    JoeKuoPoly {
        degree: 6,
        coeffs: 22,
        initial_m: &[1, 3, 1, 15, 13, 25],
    },
    JoeKuoPoly {
        degree: 6,
        coeffs: 25,
        initial_m: &[1, 1, 5, 5, 19, 61],
    },
    JoeKuoPoly {
        degree: 7,
        coeffs: 1,
        initial_m: &[1, 3, 7, 11, 23, 15, 103],
    },
    JoeKuoPoly {
        degree: 7,
        coeffs: 4,
        initial_m: &[1, 3, 7, 13, 13, 15, 69],
    },
    JoeKuoPoly {
        degree: 7,
        coeffs: 7,
        initial_m: &[1, 1, 3, 13, 7, 35, 63],
    },
    JoeKuoPoly {
        degree: 7,
        coeffs: 8,
        initial_m: &[1, 3, 5, 9, 1, 25, 53],
    },
    JoeKuoPoly {
        degree: 7,
        coeffs: 14,
        initial_m: &[1, 3, 1, 13, 9, 35, 107],
    },
    JoeKuoPoly {
        degree: 7,
        coeffs: 19,
        initial_m: &[1, 3, 1, 5, 27, 61, 31],
    },
    JoeKuoPoly {
        degree: 7,
        coeffs: 21,
        initial_m: &[1, 1, 5, 11, 19, 41, 61],
    },
    JoeKuoPoly {
        degree: 7,
        coeffs: 28,
        initial_m: &[1, 3, 5, 3, 3, 13, 69],
    },
    JoeKuoPoly {
        degree: 7,
        coeffs: 31,
        initial_m: &[1, 1, 7, 13, 1, 19, 1],
    },
    JoeKuoPoly {
        degree: 7,
        coeffs: 32,
        initial_m: &[1, 3, 7, 5, 13, 19, 59],
    },
    JoeKuoPoly {
        degree: 7,
        coeffs: 37,
        initial_m: &[1, 1, 3, 9, 25, 29, 41],
    },
    JoeKuoPoly {
        degree: 7,
        coeffs: 41,
        initial_m: &[1, 3, 5, 13, 23, 1, 55],
    },
    JoeKuoPoly {
        degree: 7,
        coeffs: 42,
        initial_m: &[1, 3, 7, 3, 13, 59, 17],
    },
];

/// Computes the 32 scaled direction numbers $V_1, \dots, V_{32}$ for coordinate axis `dim_idx` (`0`-based).
#[verified_engine::verified]
pub(crate) fn compute_direction_numbers(dim_idx: usize) -> [u32; SOBOL_BITS] {
    let mut v = [0_u32; SOBOL_BITS];
    if dim_idx == 0 {
        for (bit, slot) in v.iter_mut().enumerate() {
            *slot = 1_u32 << (31 - bit);
        }
        return v;
    }

    let poly = JOE_KUO_POLYS_D2_TO_D32[dim_idx - 1];
    let s = poly.degree;
    for (idx, &m_val) in poly.initial_m.iter().enumerate() {
        v[idx] = m_val << (31 - idx);
    }
    for k in s..SOBOL_BITS {
        let mut next_v = v[k - s] ^ (v[k - s] >> s);
        for lag in 1..s {
            if ((poly.coeffs >> (s - 1 - lag)) & 1) != 0 {
                next_v ^= v[k - lag];
            }
        }
        v[k] = next_v;
    }
    v
}

// [cite:clinical_trials_statistics]
