//! Deterministic `#![no_std]` Joe-Kuo base-2 Sobol sequence with Gray-code stepping.

/// Maximum supported dimension for [`SobolSequence`].
pub const MAX_SOBOL_DIM: usize = 32;

const INV_TWO_POW_32: f64 = 1.0 / 4_294_967_296.0;

/// Joe & Kuo (`new-joe-kuo-6.21201`) primitive polynomials over GF(2) and initial odd direction
/// numbers `(degree s, interior coeff mask a, [m_1..m_s])` for dimensions 2..=32 (indices 1..32).
struct PolySpec {
    s: usize,
    a: u32,
    m: [u32; 7],
}

const JOE_KUO_SPECS: [PolySpec; MAX_SOBOL_DIM - 1] = [
    PolySpec {
        s: 1,
        a: 0,
        m: [1, 0, 0, 0, 0, 0, 0],
    },
    PolySpec {
        s: 2,
        a: 1,
        m: [1, 3, 0, 0, 0, 0, 0],
    },
    PolySpec {
        s: 3,
        a: 1,
        m: [1, 3, 1, 0, 0, 0, 0],
    },
    PolySpec {
        s: 3,
        a: 2,
        m: [1, 1, 1, 0, 0, 0, 0],
    },
    PolySpec {
        s: 4,
        a: 1,
        m: [1, 1, 3, 3, 0, 0, 0],
    },
    PolySpec {
        s: 4,
        a: 4,
        m: [1, 3, 5, 13, 0, 0, 0],
    },
    PolySpec {
        s: 5,
        a: 2,
        m: [1, 1, 5, 5, 17, 0, 0],
    },
    PolySpec {
        s: 5,
        a: 4,
        m: [1, 1, 5, 5, 5, 0, 0],
    },
    PolySpec {
        s: 5,
        a: 7,
        m: [1, 1, 7, 11, 19, 0, 0],
    },
    PolySpec {
        s: 5,
        a: 11,
        m: [1, 1, 5, 1, 1, 0, 0],
    },
    PolySpec {
        s: 5,
        a: 13,
        m: [1, 1, 1, 3, 11, 0, 0],
    },
    PolySpec {
        s: 5,
        a: 14,
        m: [1, 3, 5, 5, 31, 0, 0],
    },
    PolySpec {
        s: 6,
        a: 1,
        m: [1, 3, 3, 9, 7, 49, 0],
    },
    PolySpec {
        s: 6,
        a: 13,
        m: [1, 1, 1, 15, 21, 21, 0],
    },
    PolySpec {
        s: 6,
        a: 16,
        m: [1, 3, 1, 13, 27, 49, 0],
    },
    PolySpec {
        s: 6,
        a: 19,
        m: [1, 1, 1, 15, 7, 5, 0],
    },
    PolySpec {
        s: 6,
        a: 22,
        m: [1, 3, 1, 15, 13, 25, 0],
    },
    PolySpec {
        s: 6,
        a: 25,
        m: [1, 1, 5, 5, 19, 61, 0],
    },
    PolySpec {
        s: 7,
        a: 1,
        m: [1, 3, 7, 11, 23, 15, 103],
    },
    PolySpec {
        s: 7,
        a: 4,
        m: [1, 3, 7, 13, 13, 15, 69],
    },
    PolySpec {
        s: 7,
        a: 7,
        m: [1, 1, 3, 13, 7, 35, 63],
    },
    PolySpec {
        s: 7,
        a: 8,
        m: [1, 3, 5, 9, 1, 25, 53],
    },
    PolySpec {
        s: 7,
        a: 14,
        m: [1, 3, 1, 13, 9, 35, 107],
    },
    PolySpec {
        s: 7,
        a: 19,
        m: [1, 3, 1, 5, 27, 61, 31],
    },
    PolySpec {
        s: 7,
        a: 21,
        m: [1, 1, 5, 11, 19, 41, 61],
    },
    PolySpec {
        s: 7,
        a: 28,
        m: [1, 3, 5, 3, 3, 13, 69],
    },
    PolySpec {
        s: 7,
        a: 31,
        m: [1, 1, 7, 13, 1, 15, 1],
    },
    PolySpec {
        s: 7,
        a: 32,
        m: [1, 3, 7, 5, 13, 19, 59],
    },
    PolySpec {
        s: 7,
        a: 37,
        m: [1, 1, 3, 9, 25, 29, 41],
    },
    PolySpec {
        s: 7,
        a: 41,
        m: [1, 3, 5, 13, 23, 1, 55],
    },
    PolySpec {
        s: 7,
        a: 42,
        m: [1, 3, 7, 3, 13, 59, 17],
    },
];

const fn build_direction_table() -> [[u32; 32]; MAX_SOBOL_DIM] {
    let mut table = [[0_u32; 32]; MAX_SOBOL_DIM];
    let mut k = 0;
    while k < 32 {
        table[0][k] = 1_u32 << (31 - k);
        k += 1;
    }

    let mut dim = 1;
    while dim < MAX_SOBOL_DIM {
        let spec = &JOE_KUO_SPECS[dim - 1];
        let s = spec.s;
        let mut idx = 0;
        while idx < s {
            table[dim][idx] = spec.m[idx] << (31 - idx);
            idx += 1;
        }
        while idx < 32 {
            let prev_s = table[dim][idx - s];
            let mut val = prev_s ^ (prev_s >> s);
            let mut i = 1;
            while i < s {
                if ((spec.a >> (s - 1 - i)) & 1) != 0 {
                    val ^= table[dim][idx - i];
                }
                i += 1;
            }
            table[dim][idx] = val;
            idx += 1;
        }
        dim += 1;
    }
    table
}

const DIRECTION_NUMBERS: [[u32; 32]; MAX_SOBOL_DIM] = build_direction_table();

#[inline]
fn fract_nonneg(x: f64) -> f64 {
    let f = x - libm::floor(x);
    if f >= 1.0 || f < 0.0 { 0.0 } else { f }
}

/// `#![no_std]` base-2 Sobol low-discrepancy generator using Antonov-Saleev Gray-code updates.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SobolSequence {
    dim: usize,
    index: u32,
    state: [u32; MAX_SOBOL_DIM],
}

impl SobolSequence {
    /// Creates a Sobol generator for `1..=MAX_SOBOL_DIM` dimensions starting prior to index `1`.
    #[must_use]
    pub fn new(dim: usize) -> Option<Self> {
        if dim == 0 || dim > MAX_SOBOL_DIM {
            return None;
        }
        Some(Self {
            dim,
            index: 0,
            state: [0_u32; MAX_SOBOL_DIM],
        })
    }

    /// Dimensionality of generated points.
    #[inline]
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.dim
    }

    /// Current 1-based point count already emitted.
    #[inline]
    #[must_use]
    pub fn index(&self) -> u32 {
        self.index
    }

    /// Resets the internal Gray-code state and index back to zero.
    #[inline]
    pub fn reset(&mut self) {
        self.index = 0;
        self.state = [0_u32; MAX_SOBOL_DIM];
    }

    /// Evaluates the 0-based Sobol net point `zero_based_index` via Gray code `n ^ (n >> 1)` in $(0, 1)^d$.
    pub fn point_at(&self, zero_based_index: u32, digital_shift: Option<&[u32]>, out: &mut [f64]) {
        assert!(out.len() >= self.dim, "output slice shorter than dimension");
        let gray = zero_based_index ^ (zero_based_index >> 1);
        for (j, slot) in out.iter_mut().take(self.dim).enumerate() {
            let mut acc = 0_u32;
            let mut g = gray;
            let dirs = &DIRECTION_NUMBERS[j];
            for &dir in dirs {
                if g == 0 {
                    break;
                }
                if (g & 1) != 0 {
                    acc ^= dir;
                }
                g >>= 1;
            }
            let mask = digital_shift.and_then(|s| s.get(j)).copied().unwrap_or(0);
            *slot = (f64::from(acc ^ mask) + 0.5) * INV_TWO_POW_32;
        }
    }

    /// Emits the current Sobol $(t, s)$-net point in $(0, 1)^d$ and advances by one Gray-code step.
    pub fn next_into(&mut self, out: &mut [f64]) {
        assert!(out.len() >= self.dim, "output slice shorter than dimension");
        let c = (self.index.trailing_ones() as usize).min(31);
        for (j, slot) in out.iter_mut().take(self.dim).enumerate() {
            *slot = (f64::from(self.state[j]) + 0.5) * INV_TWO_POW_32;
            self.state[j] ^= DIRECTION_NUMBERS[j][c];
        }
        self.index = self.index.wrapping_add(1);
    }

    /// Emits the current Sobol point with Cranley-Patterson modulo-1 `shift` and advances by one step.
    pub fn next_shifted_into(&mut self, shift: &[f64], out: &mut [f64]) {
        assert!(
            shift.len() >= self.dim,
            "shift slice shorter than dimension"
        );
        assert!(out.len() >= self.dim, "output slice shorter than dimension");
        let c = (self.index.trailing_ones() as usize).min(31);
        for (j, slot) in out.iter_mut().take(self.dim).enumerate() {
            let base = (f64::from(self.state[j]) + 0.5) * INV_TWO_POW_32;
            *slot = fract_nonneg(base + shift[j]);
            self.state[j] ^= DIRECTION_NUMBERS[j][c];
        }
        self.index = self.index.wrapping_add(1);
    }

    /// Emits the current Sobol point with base-2 digital XOR `digital_shift` in $(0, 1)^d$ and advances.
    pub fn next_digital_shifted_into(&mut self, digital_shift: &[u32], out: &mut [f64]) {
        assert!(
            digital_shift.len() >= self.dim,
            "digital_shift slice shorter than dimension"
        );
        assert!(out.len() >= self.dim, "output slice shorter than dimension");
        let c = (self.index.trailing_ones() as usize).min(31);
        for (j, slot) in out.iter_mut().take(self.dim).enumerate() {
            let scrambled = self.state[j] ^ digital_shift[j];
            *slot = (f64::from(scrambled) + 0.5) * INV_TWO_POW_32;
            self.state[j] ^= DIRECTION_NUMBERS[j][c];
        }
        self.index = self.index.wrapping_add(1);
    }
}
