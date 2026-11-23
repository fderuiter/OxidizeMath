//! Deterministic `#![no_std]` Richtmyer and Halton low-discrepancy generators.

/// Maximum supported dimension for [`RichtmyerSequence`] and [`HaltonSequence`].
pub const MAX_QMC_DIM: usize = 64;

/// First 64 prime numbers used as irrational square-root generators and Halton bases.
pub const PRIMES_64: [u32; MAX_QMC_DIM] = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
    101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193,
    197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307,
    311,
];

#[inline]
fn fract_nonneg(x: f64) -> f64 {
    let f = x - libm::floor(x);
    if !(0.0..1.0).contains(&f) { 0.0 } else { f }
}

const fn is_prime_u32(candidate: u32) -> bool {
    if candidate < 2 {
        return false;
    }
    let mut div = 2_u32;
    while div.saturating_mul(div) <= candidate {
        if candidate.is_multiple_of(div) {
            return false;
        }
        div += 1;
    }
    true
}

/// Returns the `index`-th prime number (0-indexed: `prime_at(0) == 2`, `prime_at(1) == 3`, ...).
#[must_use]
pub const fn prime_at(index: usize) -> u32 {
    if index < PRIMES_64.len() {
        return PRIMES_64[index];
    }
    let mut count = PRIMES_64.len();
    let mut candidate = PRIMES_64[PRIMES_64.len() - 1] + 2;
    while count <= index {
        if is_prime_u32(candidate) {
            if count == index {
                return candidate;
            }
            count += 1;
        }
        candidate += 2;
    }
    candidate
}

/// Returns the `index`-th Richtmyer irrational rotation generator $\{\sqrt{p_{\text{index}}}\} \in (0, 1)$.
#[inline]
#[must_use]
pub fn richtmyer_generator(index: usize) -> f64 {
    fract_nonneg(libm::sqrt(f64::from(prime_at(index))))
}

/// Applies Baker's periodizing tent transformation $t(u) = 1 - |2u - 1| \in [0, 1]$.
#[inline]
#[must_use]
pub fn baker_tent(u: f64) -> f64 {
    (1.0 - libm::fabs(2.0 * u - 1.0)).clamp(0.0, 1.0)
}

/// Computes the van der Corput radical inverse $\Phi_b(n) \in [0, 1)$ in integer `base >= 2`.
#[inline]
#[must_use]
pub fn radical_inverse(mut index: u64, base: u32) -> f64 {
    assert!(base >= 2, "Halton base must be >= 2");
    let inv_base = 1.0 / (base as f64);
    let mut factor = inv_base;
    let mut result = 0.0;
    let b64 = u64::from(base);
    for _ in 0..64 {
        if index == 0 {
            break;
        }
        let digit = index % b64;
        result += (digit as f64) * factor;
        index /= b64;
        factor *= inv_base;
    }
    result.clamp(0.0, 1.0 - f64::EPSILON)
}

/// `#![no_std]` Kronecker-Richtmyer irrational-rotation sequence $u_{n,j} = \{n\sqrt{p_j} + \Delta_j\}$.
#[derive(Clone, Debug, PartialEq)]
pub struct RichtmyerSequence {
    dim: usize,
    step: u64,
    use_baker_tent: bool,
    generators: [f64; MAX_QMC_DIM],
}

impl RichtmyerSequence {
    /// Creates a Richtmyer sequence of dimension `1..=MAX_QMC_DIM` without Baker's tent transform.
    #[must_use]
    pub fn new(dim: usize) -> Option<Self> {
        Self::with_baker_tent(dim, false)
    }

    /// Creates a Richtmyer sequence with optional Baker periodizing tent transformation.
    #[must_use]
    pub fn with_baker_tent(dim: usize, use_baker_tent: bool) -> Option<Self> {
        if dim == 0 || dim > MAX_QMC_DIM {
            return None;
        }
        let mut generators = [0.0_f64; MAX_QMC_DIM];
        for j in 0..dim {
            generators[j] = libm::sqrt(f64::from(PRIMES_64[j]));
        }
        Some(Self {
            dim,
            step: 0,
            use_baker_tent,
            generators,
        })
    }

    /// Dimensionality of generated points.
    #[inline]
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.dim
    }

    /// Current 1-based index count already emitted via [`Self::next_into`].
    #[inline]
    #[must_use]
    pub fn index(&self) -> u64 {
        self.step
    }

    /// Resets the sequence counter back to zero.
    #[inline]
    pub fn reset(&mut self) {
        self.step = 0;
    }

    /// Evaluates the 1-based point `one_based_index` with optional Cranley-Patterson `shift` into `out`.
    pub fn point_at(&self, one_based_index: u64, shift: Option<&[f64]>, out: &mut [f64]) {
        assert!(out.len() >= self.dim, "output slice shorter than dimension");
        let n = one_based_index as f64;
        for (j, slot) in out.iter_mut().take(self.dim).enumerate() {
            let delta = shift.and_then(|s| s.get(j)).copied().unwrap_or(0.0);
            let raw = fract_nonneg(n * self.generators[j] + delta);
            *slot = if self.use_baker_tent {
                baker_tent(raw)
            } else {
                raw
            };
        }
    }

    /// Advances the sequence by one step and writes the unshifted point into `out`.
    pub fn next_into(&mut self, out: &mut [f64]) {
        self.step = self.step.saturating_add(1);
        self.point_at(self.step, None, out);
    }

    /// Advances the sequence by one step and writes the Cranley-Patterson shifted point into `out`.
    pub fn next_shifted_into(&mut self, shift: &[f64], out: &mut [f64]) {
        assert!(
            shift.len() >= self.dim,
            "shift slice shorter than dimension"
        );
        self.step = self.step.saturating_add(1);
        self.point_at(self.step, Some(shift), out);
    }
}

/// `#![no_std]` Halton low-discrepancy sequence using radical inverses in prime bases.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HaltonSequence {
    dim: usize,
    step: u64,
}

impl HaltonSequence {
    /// Creates a Halton sequence of dimension `1..=MAX_QMC_DIM` starting prior to index `1`.
    #[must_use]
    pub fn new(dim: usize) -> Option<Self> {
        if dim == 0 || dim > MAX_QMC_DIM {
            return None;
        }
        Some(Self { dim, step: 0 })
    }

    /// Dimensionality of generated points.
    #[inline]
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.dim
    }

    /// Current 1-based index count already emitted.
    #[inline]
    #[must_use]
    pub fn index(&self) -> u64 {
        self.step
    }

    /// Resets the sequence counter back to zero.
    #[inline]
    pub fn reset(&mut self) {
        self.step = 0;
    }

    /// Evaluates the 1-based Halton point `one_based_index` with optional modulo-1 `shift`.
    pub fn point_at(&self, one_based_index: u64, shift: Option<&[f64]>, out: &mut [f64]) {
        assert!(out.len() >= self.dim, "output slice shorter than dimension");
        let idx = one_based_index.max(1);
        for (j, slot) in out.iter_mut().take(self.dim).enumerate() {
            let base_val = radical_inverse(idx, PRIMES_64[j]);
            let delta = shift.and_then(|s| s.get(j)).copied().unwrap_or(0.0);
            *slot = fract_nonneg(base_val + delta);
        }
    }

    /// Advances the sequence by one step and writes the next Halton point into `out`.
    pub fn next_into(&mut self, out: &mut [f64]) {
        self.step = self.step.saturating_add(1);
        self.point_at(self.step, None, out);
    }

    /// Advances the sequence by one step and writes the Cranley-Patterson shifted point into `out`.
    pub fn next_shifted_into(&mut self, shift: &[f64], out: &mut [f64]) {
        assert!(
            shift.len() >= self.dim,
            "shift slice shorter than dimension"
        );
        self.step = self.step.saturating_add(1);
        self.point_at(self.step, Some(shift), out);
    }
}
