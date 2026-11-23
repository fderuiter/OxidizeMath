//! Prime-exponent FRACTRAN register virtual machine and Minsky counter-machine decoding.
//!
//! Compiles exact [`Rational`] fractions into contiguous `u32` prime-exponent tables so
//! runtime stepping executes with zero dynamic heap allocations.
//!
//! # Example
//!
//! ```rust
//! # use pure_math::__doctest_facade as math_explorer;
//! use math_explorer::pure_math::algorithmic_information::fractran::FractranPreset;
//!
//! let (program, _state, initial) = FractranPreset::Add.build();
//! let vm = program.compile().expect("valid FRACTRAN program");
//! let mut regs = vm.encode_value(initial).expect("valid initial value");
//! while vm.step_registers(&mut regs).expect("no overflow").is_some() {}
//! assert_eq!(vm.power_of_two_exponent(&regs), Some(5));
//! ```

#[path = "fractran_factor.rs"]
mod factor;

use crate::math_types::Rational;
use diagnostics::{Diagnostic, Severity};
use std::collections::HashMap;

/// Upper bound on prime factors accepted during synchronous fraction compilation.
pub const MAX_SMOOTH_PRIME: u64 = 10_000;

/// Typed validation and execution errors for [`FractranProgram`] and [`FractranCompiledVm`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum FractranError {
    /// A fraction rule at `rule_index` is zero or negative.
    NonPositiveFraction {
        /// Zero-based index of the invalid rule.
        rule_index: usize,
    },
    /// Prime factorization encountered a factor exceeding `max_prime` or overflowing `u32` exponents.
    UnsmoothFactor {
        /// Maximum supported smooth prime factor.
        max_prime: u64,
    },
    /// Textual fraction list contained malformed `numerator/denominator` syntax.
    InvalidFractionSyntax,
    /// Initial machine integer must be strictly positive (`>= 1`).
    InitialValueZero,
    /// Initial integer contains a prime factor absent from the compiled VM basis.
    UnrepresentedPrime {
        /// Prime factor missing from the VM basis.
        prime: u64,
    },
    /// Register slice length did not match the compiled VM prime basis length.
    BasisLengthMismatch {
        /// Expected register count (`vm.primes().len()`).
        expected: usize,
        /// Provided register slice length.
        actual: usize,
    },
    /// Stepping `rule_index` would overflow `u32::MAX` in the register for `prime`.
    RegisterOverflow {
        /// Zero-based index of the firing rule.
        rule_index: usize,
        /// Prime register that reached `u32::MAX`.
        prime: u64,
    },
}

impl std::fmt::Display for FractranError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonPositiveFraction { rule_index } => {
                write!(
                    f,
                    "FRACTRAN rule {} must be strictly positive (> 0)",
                    rule_index + 1
                )
            }
            Self::UnsmoothFactor { max_prime } => write!(
                f,
                "number contains a prime factor exceeding smooth prime limit ({max_prime})"
            ),
            Self::InvalidFractionSyntax => {
                write!(f, "invalid fraction syntax; expected positive N/D ratios")
            }
            Self::InitialValueZero => write!(f, "initial FRACTRAN integer must be at least 1"),
            Self::UnrepresentedPrime { prime } => {
                write!(f, "prime factor {prime} is not present in the VM basis")
            }
            Self::BasisLengthMismatch { expected, actual } => {
                write!(
                    f,
                    "register slice length mismatch: expected {expected}, got {actual}"
                )
            }
            Self::RegisterOverflow { rule_index, prime } => write!(
                f,
                "register exponent overflow on prime {prime} at rule {}",
                rule_index + 1
            ),
        }
    }
}

impl std::error::Error for FractranError {}

impl Diagnostic for FractranError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }

    #[verified_engine::verified]
    fn metadata(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert("error_type".to_string(), "FractranError".to_string());
        map.insert("description".to_string(), self.to_string());
        map
    }
}

/// Canonical FRACTRAN counter-machine programs from Conway (1987).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum FractranPreset {
    /// Conway's 14-fraction prime-generating universal register program (`PRIMEGAME`).
    PrimeGame,
    /// Conway's 6-fraction multiplication machine (`2^a * 3^b -> 5^(a*b)`).
    Mult,
    /// Single-fraction addition machine (`2^a * 3^b -> 2^(a+b)`).
    Add,
    /// Nine-fraction Minsky Fibonacci pair machine (`3 * 5^n -> 2^F_n * 3^F_{n+1}`).
    Fibonacci,
}

impl FractranPreset {
    /// Ordered catalog of all built-in FRACTRAN presets.
    pub const ALL: [Self; 4] = [Self::PrimeGame, Self::Mult, Self::Add, Self::Fibonacci];

    /// Short human-readable uppercase identifier for UI selectors.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::PrimeGame => "PRIMEGAME",
            Self::Mult => "MULT",
            Self::Add => "ADD",
            Self::Fibonacci => "FIBONACCI",
        }
    }

    /// Default positive starting integer $N_0$ for this preset.
    #[must_use]
    pub const fn default_initial_value(self) -> u64 {
        match self {
            Self::PrimeGame => 2,
            Self::Mult | Self::Add => 72, // 2^3 * 3^2
            Self::Fibonacci => 9375,      // 3^1 * 5^5 -> 2^5 * 3^8
        }
    }

    /// Constructs the exact [`FractranProgram`] fraction sequence for this preset.
    #[must_use]
    pub fn program(self) -> FractranProgram {
        let fractions = factor::preset_pairs(self)
            .iter()
            .copied()
            .map(Rational::from)
            .collect();
        FractranProgram::new(fractions)
    }

    /// Returns the Minsky control-state vs data-counter prime partition for this preset.
    #[must_use]
    pub fn minsky_state(self) -> FractranState {
        match self {
            Self::PrimeGame => FractranState::new(vec![11, 13, 17, 19, 23, 29], vec![2, 3, 5, 7]),
            Self::Mult => FractranState::new(vec![11, 13], vec![2, 3, 5, 7]),
            Self::Add => FractranState::new(Vec::new(), vec![2, 3]),
            Self::Fibonacci => FractranState::new(vec![13, 17, 19, 23], vec![2, 3, 5, 7, 11]),
        }
    }

    /// Returns `(program, minsky_state, default_initial_value)` for this preset.
    #[must_use]
    pub fn build(self) -> (FractranProgram, FractranState, u64) {
        (
            self.program(),
            self.minsky_state(),
            self.default_initial_value(),
        )
    }
}

/// Ordered list of rational multiplier rules defining a FRACTRAN program.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FractranProgram {
    /// Ordered rational rules $f_1, f_2, \dots, f_k$.
    pub fractions: Vec<Rational>,
}

impl FractranProgram {
    /// Wraps a sequence of [`Rational`] fractions into a [`FractranProgram`].
    #[verified_engine::verified]
    pub fn new(fractions: Vec<Rational>) -> Self {
        Self { fractions }
    }

    /// Parses comma- or whitespace-separated positive `N/D` (or integer `N`) tokens.
    pub fn parse_fractions(input: &str) -> Result<Self, FractranError> {
        factor::parse_fraction_tokens(input).map(|fractions| Self { fractions })
    }

    /// Formats the program's fractions as comma-separated `numerator/denominator` ratios.
    #[must_use]
    pub fn format_fractions(&self) -> String {
        self.fractions
            .iter()
            .map(|r| format!("{}/{}", r.0.numerator(), r.0.denominator()))
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// Compiles the program and encodes `initial_value` (`>= 1`), extending the prime basis if needed.
    pub fn compile_for_initial(
        &self,
        initial_value: u64,
    ) -> Result<(FractranCompiledVm, Vec<u32>), FractranError> {
        if initial_value == 0 {
            return Err(FractranError::InitialValueZero);
        }
        let mut init_factors = Vec::new();
        factor::factor_u64_smooth(initial_value, &mut init_factors)?;
        let extra: Vec<u64> = init_factors.iter().map(|&(p, _)| p).collect();
        let vm = self.compile_with_primes(&extra)?;
        let regs = vm.encode_value(initial_value)?;
        Ok((vm, regs))
    }

    /// Compiles all fraction numerators and denominators into a preallocated [`FractranCompiledVm`].
    pub fn compile(&self) -> Result<FractranCompiledVm, FractranError> {
        self.compile_with_primes(&[])
    }

    /// Compiles the program while ensuring every smooth prime in `extra_primes` is included in the basis.
    pub fn compile_with_primes(
        &self,
        extra_primes: &[u64],
    ) -> Result<FractranCompiledVm, FractranError> {
        let tables = factor::compile_exponent_tables(&self.fractions, extra_primes)?;
        Ok(FractranCompiledVm {
            primes: tables.primes,
            num_rules: tables.num_rules,
            numer_exponents: tables.numer_exponents,
            denom_exponents: tables.denom_exponents,
        })
    }
}

/// Preallocated prime-exponent virtual machine for zero-allocation FRACTRAN execution.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FractranCompiledVm {
    primes: Vec<u64>,
    num_rules: usize,
    numer_exponents: Vec<u32>,
    denom_exponents: Vec<u32>,
}

impl FractranCompiledVm {
    /// Sorted prime basis $p_0 < p_1 < \dots < p_{m-1}$ indexing register slices.
    #[must_use]
    pub fn primes(&self) -> &[u64] {
        &self.primes
    }

    /// Number of compiled fraction rules.
    #[must_use]
    pub const fn num_rules(&self) -> usize {
        self.num_rules
    }

    /// Returns `(numerator_exponents, denominator_exponents)` for `rule_index`.
    #[must_use]
    pub fn rule_exponents(&self, rule_index: usize) -> Option<(&[u32], &[u32])> {
        if rule_index >= self.num_rules {
            return None;
        }
        let width = self.primes.len();
        let start = rule_index * width;
        let end = start + width;
        Some((
            &self.numer_exponents[start..end],
            &self.denom_exponents[start..end],
        ))
    }

    /// Factors `value` (`>= 1`) into `out_registers` aligned with [`Self::primes`].
    pub fn encode_u64(&self, value: u64, out_registers: &mut [u32]) -> Result<(), FractranError> {
        if value == 0 {
            return Err(FractranError::InitialValueZero);
        }
        if out_registers.len() != self.primes.len() {
            return Err(FractranError::BasisLengthMismatch {
                expected: self.primes.len(),
                actual: out_registers.len(),
            });
        }
        out_registers.fill(0);
        let mut factors = Vec::new();
        factor::factor_u64_smooth(value, &mut factors)?;
        for (prime, exp) in factors {
            let idx = self
                .primes
                .binary_search(&prime)
                .map_err(|_| FractranError::UnrepresentedPrime { prime })?;
            out_registers[idx] = exp;
        }
        Ok(())
    }

    /// Allocates and populates a register vector encoding `value` (`>= 1`).
    pub fn encode_value(&self, value: u64) -> Result<Vec<u32>, FractranError> {
        let mut regs = vec![0_u32; self.primes.len()];
        self.encode_u64(value, &mut regs)?;
        Ok(regs)
    }

    /// Executes one deterministic FRACTRAN step in-place with zero heap allocations.
    ///
    /// Returns `Ok(Some(rule_index))` when rule `rule_index` fires, `Ok(None)` when halted,
    /// or `Err(FractranError::RegisterOverflow)` without mutating `registers` on `u32` overflow.
    #[verified_engine::verified]
    pub fn step_registers(&self, registers: &mut [u32]) -> Result<Option<usize>, FractranError> {
        let width = self.primes.len();
        if registers.len() != width {
            return Err(FractranError::BasisLengthMismatch {
                expected: width,
                actual: registers.len(),
            });
        }
        assert_eq!(self.denom_exponents.len(), self.num_rules * width);
        assert_eq!(self.numer_exponents.len(), self.num_rules * width);

        for rule_index in 0..self.num_rules {
            let start = rule_index * width;
            let end = start + width;
            let den_row = &self.denom_exponents[start..end];
            let num_row = &self.numer_exponents[start..end];
            if registers.iter().zip(den_row).all(|(&reg, &den)| reg >= den) {
                for ((&reg, &den), (&num, &prime)) in registers
                    .iter()
                    .zip(den_row)
                    .zip(num_row.iter().zip(&self.primes))
                {
                    if (reg - den).checked_add(num).is_none() {
                        return Err(FractranError::RegisterOverflow { rule_index, prime });
                    }
                }
                for ((reg, &den), &num) in registers.iter_mut().zip(den_row).zip(num_row) {
                    *reg = (*reg - den) + num;
                }
                return Ok(Some(rule_index));
            }
        }
        Ok(None)
    }

    /// Returns `Some(k)` iff `registers` encodes a pure prime power of two $N = 2^k$ ($k \ge 1$).
    #[must_use]
    #[verified_engine::verified]
    pub fn power_of_two_exponent(&self, registers: &[u32]) -> Option<u32> {
        let width = self.primes.len();
        if registers.len() != width || width == 0 || self.primes[0] != 2 {
            return None;
        }
        assert_eq!(registers.len(), width);
        let exp_two = registers[0];
        if exp_two == 0 || registers[1..width].iter().any(|&other| other != 0) {
            return None;
        }
        Some(exp_two)
    }
}

/// Decoded Minsky control-state flags and data-counter values at a FRACTRAN configuration.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MinskyDecodedState {
    /// Active `(prime, exponent)` pairs among [`FractranState::state_primes`] with `exponent > 0`.
    pub active_state_primes: Vec<(u64, u32)>,
    /// Data register `(prime, exponent)` counter values ordered by [`FractranState::register_primes`].
    pub counters: Vec<(u64, u32)>,
}

/// Directed Minsky counter-machine transition extracted from a compiled FRACTRAN rule.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MinskyTransition {
    /// Zero-based index of the corresponding FRACTRAN fraction rule.
    pub rule_index: usize,
    /// Control-state prime required in the denominator (`None` for resting state $q_0$).
    pub from_state: Option<u64>,
    /// Control-state prime produced in the numerator (`None` for resting state $q_0$).
    pub to_state: Option<u64>,
    /// Data-counter `(prime, decrement)` guards consumed by the denominator.
    pub guard_decrements: Vec<(u64, u32)>,
    /// Data-counter `(prime, increment)` updates added by the numerator.
    pub increments: Vec<(u64, u32)>,
}

/// Partition of FRACTRAN primes into Minsky control-state flags and integer data registers.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FractranState {
    /// Primes acting as finite-state control indicators.
    pub state_primes: Vec<u64>,
    /// Primes whose exponents encode unbounded non-negative integer counters.
    pub register_primes: Vec<u64>,
}

impl FractranState {
    /// Creates a new Minsky prime partition from explicit `state_primes` and `register_primes`.
    #[must_use]
    pub fn new(state_primes: Vec<u64>, register_primes: Vec<u64>) -> Self {
        Self {
            state_primes,
            register_primes,
        }
    }

    /// Infers a Minsky prime partition for custom programs from a compiled VM prime basis.
    #[must_use]
    pub fn from_vm_heuristic(vm: &FractranCompiledVm) -> Self {
        let (mut state_primes, mut register_primes) = (Vec::new(), Vec::new());
        for (col, &p) in vm.primes().iter().enumerate() {
            let is_unit_control = p >= 11
                && (0..vm.num_rules()).all(|r| {
                    let (num, den) = vm.rule_exponents(r).unwrap_or((&[], &[]));
                    num.get(col).copied().unwrap_or(0) <= 1
                        && den.get(col).copied().unwrap_or(0) <= 1
                });
            if is_unit_control {
                state_primes.push(p);
            } else {
                register_primes.push(p);
            }
        }
        Self {
            state_primes,
            register_primes,
        }
    }

    /// Decodes `registers` into active Minsky state flags and data register counters.
    #[must_use]
    pub fn decode(&self, vm: &FractranCompiledVm, registers: &[u32]) -> MinskyDecodedState {
        let lookup = |p: u64| {
            vm.primes()
                .binary_search(&p)
                .ok()
                .and_then(|i| registers.get(i).copied())
                .unwrap_or(0)
        };
        let active_state_primes = self
            .state_primes
            .iter()
            .filter(|&&p| lookup(p) > 0)
            .map(|&p| (p, lookup(p)))
            .collect();
        let counters = self
            .register_primes
            .iter()
            .map(|&p| (p, lookup(p)))
            .collect();
        MinskyDecodedState {
            active_state_primes,
            counters,
        }
    }

    /// Extracts directed Minsky state-machine transitions across all rules of `vm`.
    #[must_use]
    pub fn transitions(&self, vm: &FractranCompiledVm) -> Vec<MinskyTransition> {
        factor::extract_minsky_transitions(vm, &self.state_primes)
    }
}

// [cite:algorithmic_information_rust]
