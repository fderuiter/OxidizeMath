//! Canonical FRACTRAN programs and error types for [`super::fractran::FractranProgram`].

use crate::math_types::Rational;
use dashu::base::Sign;
use dashu::integer::UBig;
use std::fmt;

/// First 64 prime numbers ($p_0 = 2$ through $p_{63} = 311$) indexing zero-allocation registers.
pub const FIRST_64_PRIMES: [u64; 64] = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
    101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193,
    197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307,
    311,
];

/// Maximum number of prime registers supported by the stack-allocated FRACTRAN VM.
pub const MAX_FRACTRAN_PRIMES: usize = 64;

/// Largest prime factor ($p_{63} = 311$) supported by the zero-allocation register file.
pub const MAX_FRACTRAN_PRIME: u64 = 311;

/// Maximum exponent allowed in any single prime register to prevent arithmetic overflow.
pub const MAX_REGISTER_EXPONENT: u32 = 1_000_000;

/// Conway's 14-fraction `PRIMEGAME` $(n_0 = 2 \implies 2^2, 2^3, 2^5, 2^7, 2^{11}, \dots)$.
pub(super) const PRIMEGAME_PAIRS: [(i128, i128); 14] = [
    (17, 91),
    (78, 85),
    (19, 51),
    (23, 38),
    (29, 33),
    (77, 29),
    (95, 23),
    (77, 19),
    (1, 17),
    (11, 13),
    (13, 11),
    (15, 2),
    (1, 7),
    (55, 1),
];

/// Canonical FRACTRAN adder mapping $2^a 3^b \to 3^{a+b}$.
pub(super) const ADDER_PAIRS: [(i128, i128); 1] = [(3, 2)];

/// Canonical FRACTRAN multiplier mapping $2^a 3^b \to 5^{a \cdot b}$.
pub(super) const MULTIPLIER_PAIRS: [(i128, i128); 6] =
    [(455, 33), (11, 13), (1, 11), (3, 7), (11, 2), (1, 3)];

/// Canonical FRACTRAN Fibonacci machine mapping $2^1 5^n \to 2^{F_{n+1}} 3^{F_n}$.
pub(super) const FIBONACCI_PAIRS: [(i128, i128); 9] = [
    (1547, 22),
    (11, 17),
    (119, 33),
    (19, 11),
    (46, 133),
    (19, 23),
    (69, 247),
    (1, 19),
    (11, 5),
];

/// Descriptive validation and execution errors for the zero-allocation FRACTRAN VM.
///
/// Kept `Copy` (zero heap fields) so even error paths inside `VerifiedAllocator` never allocate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FractranError {
    /// Program contains zero fraction rules.
    EmptyProgram,
    /// Rule numerator is zero.
    ZeroNumerator {
        /// Zero-based index of the offending fraction rule.
        rule_index: usize,
    },
    /// Rule denominator is zero.
    ZeroDenominator {
        /// Zero-based index of the offending fraction rule.
        rule_index: usize,
    },
    /// Rule numerator or denominator is negative.
    NegativeFraction {
        /// Zero-based index of the offending fraction rule.
        rule_index: usize,
    },
    /// Rule or initial integer contains a prime factor exceeding [`MAX_FRACTRAN_PRIME`] (`311`).
    UnsupportedPrimeFactor {
        /// Zero-based index of the offending fraction rule (`usize::MAX` for initial integer).
        rule_index: usize,
        /// Smallest unsupported prime factor $> 311$.
        factor: u64,
    },
    /// Initial integer is zero or invalid for FRACTRAN encoding.
    InvalidInitialValue,
    /// Register exponent exceeded [`MAX_REGISTER_EXPONENT`] (`1_000_000`) or `u32::MAX`.
    ExponentOverflow {
        /// Prime base whose register overflowed.
        prime: u64,
        /// Attempted new exponent value.
        attempted: u64,
    },
    /// Custom fraction text could not be parsed at the given token index.
    InvalidFractionSyntax {
        /// Zero-based token index in the input string.
        token_index: usize,
    },
}

impl fmt::Display for FractranError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::EmptyProgram => write!(f, "FRACTRAN program must contain at least one fraction"),
            Self::ZeroNumerator { rule_index } => {
                write!(
                    f,
                    "Fraction rule #{} has a zero numerator; rules must be strictly positive",
                    rule_index + 1
                )
            }
            Self::ZeroDenominator { rule_index } => {
                write!(
                    f,
                    "Fraction rule #{} has a zero denominator; division by zero is undefined",
                    rule_index + 1
                )
            }
            Self::NegativeFraction { rule_index } => {
                write!(
                    f,
                    "Fraction rule #{} is negative; FRACTRAN rules must be strictly positive",
                    rule_index + 1
                )
            }
            Self::UnsupportedPrimeFactor { rule_index, factor } => {
                if rule_index == usize::MAX {
                    write!(
                        f,
                        "Initial state contains prime factor {factor} exceeding maximum supported prime {MAX_FRACTRAN_PRIME}"
                    )
                } else {
                    write!(
                        f,
                        "Fraction rule #{} contains prime factor {factor} exceeding maximum supported prime {MAX_FRACTRAN_PRIME}",
                        rule_index + 1
                    )
                }
            }
            Self::InvalidInitialValue => {
                write!(f, "Initial FRACTRAN state must be a positive integer >= 1")
            }
            Self::ExponentOverflow { prime, attempted } => {
                write!(
                    f,
                    "Register exponent for prime {prime} reached {attempted}, exceeding cap {MAX_REGISTER_EXPONENT}"
                )
            }
            Self::InvalidFractionSyntax { token_index } => {
                write!(
                    f,
                    "Malformed fraction token #{} (expected 'num/den' or positive integer)",
                    token_index + 1
                )
            }
        }
    }
}

impl std::error::Error for FractranError {}

/// Cycle parameters discovered by Floyd's tortoise-and-hare algorithm.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FractranCycleInfo {
    /// First trajectory step $\mu$ entering the periodic orbit.
    pub start_step: usize,
    /// Minimal positive period $\lambda \ge 1$ of the cycle.
    pub period: usize,
    /// Repeating exponent vector at step `start_step`.
    pub state: [u32; MAX_FRACTRAN_PRIMES],
}

/// Bounded trajectory classification returned by [`super::fractran::FractranProgram::detect_cycle`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FractranTrajectoryOutcome {
    /// Machine reached a fixed halting state where no fraction rule applies.
    Halted {
        /// Number of transitions executed prior to halting.
        steps: usize,
        /// Final prime-exponent register configuration.
        final_exponents: [u32; MAX_FRACTRAN_PRIMES],
    },
    /// Machine entered a deterministic repeating cycle detected by Floyd's algorithm.
    CycleDetected(FractranCycleInfo),
    /// Machine executed `steps` transitions without halting or repeating within budget.
    StepLimitReached {
        /// Number of transitions simulated.
        steps: usize,
        /// Exponent configuration at the step budget boundary.
        state: [u32; MAX_FRACTRAN_PRIMES],
    },
}

/// Pre-factored FRACTRAN rule $n / d = \prod_{i=0}^{63} p_i^{\Delta_i}$ stored inline on the stack.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FractranRule {
    /// Reduced numerator magnitude (clamped to `u128::MAX` for display).
    pub numer: u128,
    /// Reduced denominator magnitude (clamped to `u128::MAX` for display).
    pub denom: u128,
    /// Signed prime-exponent deltas $\Delta_i$ for $p_0 = 2 \dots p_{63} = 311$.
    pub deltas: [i32; MAX_FRACTRAN_PRIMES],
    /// Exclusive upper index of the highest prime touched by this rule.
    pub active_len: usize,
}

/// Finds the smallest prime divisor $> 311$ of `rem` for diagnostic reporting.
#[verified_engine::verified]
pub(super) fn smallest_factor_above_311(rem: &UBig) -> u64 {
    if let Ok(val) = u64::try_from(rem) {
        let mut d = 313u64;
        while d.saturating_mul(d) <= val && d < 200_000 {
            if val.is_multiple_of(d) {
                return d;
            }
            d += 2;
        }
        return val;
    }
    u64::MAX
}

/// Factors an unsigned `UBig` across [`FIRST_64_PRIMES`], accumulating `sign * count` into `deltas`.
#[verified_engine::verified]
fn factor_component_into_deltas(
    value: &UBig,
    sign: i32,
    rule_index: usize,
    deltas: &mut [i32; MAX_FRACTRAN_PRIMES],
    active_len: &mut usize,
) -> Result<(), FractranError> {
    let mut rem = value.clone();
    let one = UBig::from(1u8);
    let zero = UBig::from(0u8);
    for (idx, &prime) in FIRST_64_PRIMES.iter().enumerate() {
        if rem <= one {
            break;
        }
        let p_ubig = UBig::from(prime);
        while &rem % &p_ubig == zero {
            rem /= &p_ubig;
            deltas[idx] += sign;
            let mag = deltas[idx].unsigned_abs();
            if mag > MAX_REGISTER_EXPONENT {
                return Err(FractranError::ExponentOverflow {
                    prime,
                    attempted: u64::from(mag),
                });
            }
            if idx + 1 > *active_len {
                *active_len = idx + 1;
            }
        }
    }
    if rem > one {
        return Err(FractranError::UnsupportedPrimeFactor {
            rule_index,
            factor: smallest_factor_above_311(&rem),
        });
    }
    Ok(())
}

/// Compiles a single [`Rational`] fraction into a stack-inline [`FractranRule`].
#[verified_engine::verified]
pub(super) fn compile_rational_rule(
    rule_index: usize,
    frac: &Rational,
) -> Result<FractranRule, FractranError> {
    let (sign, num_mag) = frac.0.numerator().clone().into_parts();
    if num_mag == UBig::from(0u8) {
        return Err(FractranError::ZeroNumerator { rule_index });
    }
    if sign == Sign::Negative {
        return Err(FractranError::NegativeFraction { rule_index });
    }
    let den_mag = frac.0.denominator().clone();
    let mut deltas = [0i32; MAX_FRACTRAN_PRIMES];
    let mut active_len = 0usize;
    factor_component_into_deltas(&num_mag, 1, rule_index, &mut deltas, &mut active_len)?;
    factor_component_into_deltas(&den_mag, -1, rule_index, &mut deltas, &mut active_len)?;
    Ok(FractranRule {
        numer: u128::try_from(&num_mag).unwrap_or(u128::MAX),
        denom: u128::try_from(&den_mag).unwrap_or(u128::MAX),
        deltas,
        active_len,
    })
}

// [cite:algorithmic_information_rust]
