//! Zero-allocation prime-exponent FRACTRAN virtual machine, Minsky counter-machine decoding,
//! and Floyd tortoise-and-hare cycle detection.
//!
//! Supports both stack-inline 64-prime delta stepping ([`FractranProgram::step_exponents`],
//! [`FractranProgram::detect_cycle`]) and dynamic smooth-prime matrix compilation
//! ([`FractranProgram::compile`] -> [`FractranCompiledVm::step_registers`]).
//!
//! # Example
//!
//! ```rust
//! # use pure_math::__doctest_facade as math_explorer;
//! use math_explorer::pure_math::algorithmic_information::fractran::{FractranPreset, FractranProgram};
//!
//! let (program, _state, initial) = FractranPreset::Add.build();
//! let vm = program.compile().expect("valid FRACTRAN program");
//! let mut regs = vm.encode_value(initial).expect("valid initial value");
//! while vm.step_registers(&mut regs).expect("no overflow").is_some() {}
//! assert_eq!(vm.power_of_two_exponent(&regs), Some(5));
//!
//! let adder = FractranProgram::adder();
//! let mut inline_regs = FractranProgram::encode_u64(72).unwrap();
//! while adder.step_exponents(&mut inline_regs).unwrap().is_some() {}
//! assert_eq!(inline_regs[1], 5);
//! ```

use super::fractran_factor;
use super::fractran_programs::{
    ADDER_PAIRS, FIBONACCI_PAIRS, MULTIPLIER_PAIRS, PRIMEGAME_PAIRS, compile_rational_rule,
    smallest_factor_above_311,
};
pub use super::fractran_programs::{
    FIRST_64_PRIMES, FractranCycleInfo, FractranError, FractranRule, FractranTrajectoryOutcome,
    MAX_FRACTRAN_PRIME, MAX_FRACTRAN_PRIMES, MAX_REGISTER_EXPONENT, MAX_SMOOTH_PRIME,
};
pub use super::fractran_vm::{
    FractranCompiledVm, FractranPreset, FractranState, MinskyDecodedState, MinskyTransition,
};
use crate::math_types::{Integer, Rational};
use dashu::integer::{IBig, UBig};

/// Ordered list of rational multiplier rules defining a FRACTRAN program.
#[derive(Clone, Debug)]
pub struct FractranProgram {
    /// Ordered rational rules $f_1, f_2, \dots, f_k$.
    pub fractions: Vec<Rational>,
    rules: Vec<FractranRule>,
    validation_error: Option<FractranError>,
}

impl PartialEq for FractranProgram {
    fn eq(&self, other: &Self) -> bool {
        self.fractions == other.fractions
    }
}

impl FractranProgram {
    /// Wraps a sequence of [`Rational`] fractions into a [`FractranProgram`], pre-factoring
    /// stack-inline rules over $p \le 311$ when applicable without panicking.
    #[must_use]
    #[verified_engine::verified]
    pub fn new(fractions: Vec<Rational>) -> Self {
        if fractions.is_empty() {
            return Self {
                fractions,
                rules: Vec::new(),
                validation_error: Some(FractranError::EmptyProgram),
            };
        }
        let mut rules = Vec::with_capacity(fractions.len());
        let mut validation_error = None;
        for (idx, frac) in fractions.iter().enumerate() {
            match compile_rational_rule(idx, frac) {
                Ok(rule) => rules.push(rule),
                Err(err) => {
                    validation_error = Some(err);
                    break;
                }
            }
        }
        Self {
            fractions,
            rules,
            validation_error,
        }
    }

    /// Parses comma- or whitespace-separated positive `N/D` (or integer `N`) tokens using arbitrary-precision rationals.
    pub fn parse_fractions(input: &str) -> Result<Self, FractranError> {
        fractran_factor::parse_fraction_tokens(input).map(Self::new)
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
        fractran_factor::factor_u64_smooth(initial_value, &mut init_factors)?;
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
        let tables = fractran_factor::compile_exponent_tables(&self.fractions, extra_primes)?;
        Ok(FractranCompiledVm {
            primes: tables.primes,
            num_rules: tables.num_rules,
            numer_exponents: tables.numer_exponents,
            denom_exponents: tables.denom_exponents,
        })
    }

    /// Validates and compiles `fractions` for the 64-prime stack VM, rejecting empty lists, non-positive fractions, or primes $> 311$.
    #[verified_engine::verified]
    pub fn try_new(fractions: Vec<Rational>) -> Result<Self, FractranError> {
        let program = Self::new(fractions);
        if let Some(err) = program.validation_error {
            return Err(err);
        }
        Ok(program)
    }

    /// Validates and compiles `(numerator, denominator)` pairs before constructing rationals.
    #[verified_engine::verified]
    pub fn try_from_pairs(pairs: &[(i128, i128)]) -> Result<Self, FractranError> {
        if pairs.is_empty() {
            return Err(FractranError::EmptyProgram);
        }
        let mut fractions = Vec::with_capacity(pairs.len());
        for (idx, &(num, den)) in pairs.iter().enumerate() {
            if den == 0 {
                return Err(FractranError::ZeroDenominator { rule_index: idx });
            }
            if num < 0 || den < 0 {
                return Err(FractranError::NegativeFraction { rule_index: idx });
            }
            if num == 0 {
                return Err(FractranError::ZeroNumerator { rule_index: idx });
            }
            fractions.push(Rational::from((num, den)));
        }
        Self::try_new(fractions)
    }

    /// Parses comma- or whitespace-separated fractions into a validated 64-prime stack program.
    #[verified_engine::verified]
    pub fn parse(input: &str) -> Result<Self, FractranError> {
        let mut pairs = Vec::new();
        for raw in input
            .split(|c: char| c == ',' || c == ';' || c.is_whitespace())
            .filter(|s| !s.is_empty())
        {
            if let Some((lhs, rhs)) = raw.split_once('/') {
                if rhs.contains('/') {
                    return Err(FractranError::InvalidFractionSyntax);
                }
                let num = lhs
                    .trim()
                    .parse::<i128>()
                    .map_err(|_| FractranError::InvalidFractionSyntax)?;
                let den = rhs
                    .trim()
                    .parse::<i128>()
                    .map_err(|_| FractranError::InvalidFractionSyntax)?;
                pairs.push((num, den));
            } else {
                let num = raw
                    .trim()
                    .parse::<i128>()
                    .map_err(|_| FractranError::InvalidFractionSyntax)?;
                pairs.push((num, 1));
            }
        }
        Self::try_from_pairs(&pairs)
    }

    /// Returns the pre-factored 64-prime stack rules for zero-allocation inspection.
    #[must_use]
    pub fn rules(&self) -> &[FractranRule] {
        &self.rules
    }

    /// Constructs John Horton Conway's 14-fraction `PRIMEGAME` prime generator.
    #[must_use]
    #[verified_engine::verified]
    pub fn primegame() -> Self {
        Self::try_from_pairs(&PRIMEGAME_PAIRS).expect("canonical PRIMEGAME fractions are valid")
    }

    /// Constructs the canonical 1-fraction adder ($2^a 3^b \to 3^{a+b}$).
    #[must_use]
    #[verified_engine::verified]
    pub fn adder() -> Self {
        Self::try_from_pairs(&ADDER_PAIRS).expect("canonical adder fraction is valid")
    }

    /// Constructs the canonical 6-fraction multiplier ($2^a 3^b \to 5^{a \cdot b}$).
    #[must_use]
    #[verified_engine::verified]
    pub fn multiplier() -> Self {
        Self::try_from_pairs(&MULTIPLIER_PAIRS).expect("canonical multiplier fractions are valid")
    }

    /// Constructs the canonical 9-fraction Fibonacci register machine ($2^1 5^n \to 2^{F_{n+1}} 3^{F_n}$).
    #[must_use]
    #[verified_engine::verified]
    pub fn fibonacci() -> Self {
        Self::try_from_pairs(&FIBONACCI_PAIRS).expect("canonical Fibonacci fractions are valid")
    }

    #[inline]
    #[verified_engine::verified]
    fn rule_matches(rule: &FractranRule, exponents: &[u32]) -> bool {
        for (idx, &delta) in rule.deltas.iter().enumerate().take(rule.active_len) {
            if delta < 0 && exponents.get(idx).copied().unwrap_or(0) < delta.unsigned_abs() {
                return false;
            }
        }
        true
    }

    #[inline]
    #[verified_engine::verified]
    fn apply_rule_in_place(
        rule: &FractranRule,
        exponents: &mut [u32],
    ) -> Result<(), FractranError> {
        for (idx, &delta) in rule.deltas.iter().enumerate().take(rule.active_len) {
            if delta > 0 {
                let add = delta as u32;
                let prime = FIRST_64_PRIMES[idx];
                let Some(slot) = exponents.get(idx).copied() else {
                    return Err(FractranError::ExponentOverflow {
                        prime,
                        attempted: u64::from(add),
                    });
                };
                let next = slot
                    .checked_add(add)
                    .ok_or(FractranError::ExponentOverflow {
                        prime,
                        attempted: u64::from(slot) + u64::from(add),
                    })?;
                if next > MAX_REGISTER_EXPONENT {
                    return Err(FractranError::ExponentOverflow {
                        prime,
                        attempted: u64::from(next),
                    });
                }
            }
        }
        for (slot, &delta) in exponents
            .iter_mut()
            .zip(rule.deltas.iter())
            .take(rule.active_len)
        {
            if delta > 0 {
                *slot += delta as u32;
            } else if delta < 0 {
                *slot -= delta.unsigned_abs();
            }
        }
        Ok(())
    }

    /// Returns the zero-based index of the first applicable fraction rule for `exponents`, or `None` if halted.
    #[must_use]
    #[verified_engine::verified]
    pub fn matching_rule(&self, exponents: &[u32]) -> Option<usize> {
        if self.validation_error.is_some() {
            return None;
        }
        self.rules
            .iter()
            .position(|rule| Self::rule_matches(rule, exponents))
    }

    /// Returns `true` when no fraction rule in the program divides the state encoded by `exponents`.
    #[must_use]
    #[verified_engine::verified]
    pub fn is_halted(&self, exponents: &[u32]) -> bool {
        self.matching_rule(exponents).is_none()
    }

    /// Advances `exponents` in place by one FRACTRAN transition with zero dynamic heap allocation.
    #[verified_engine::verified]
    pub fn step_exponents(&self, exponents: &mut [u32]) -> Result<Option<usize>, FractranError> {
        if let Some(err) = self.validation_error {
            return Err(err);
        }
        for (rule_idx, rule) in self.rules.iter().enumerate() {
            if Self::rule_matches(rule, exponents) {
                Self::apply_rule_in_place(rule, exponents)?;
                return Ok(Some(rule_idx));
            }
        }
        Ok(None)
    }

    #[verified_engine::verified]
    pub(super) fn locate_cycle_origin_and_period(
        &self,
        initial: &[u32; MAX_FRACTRAN_PRIMES],
        meeting: &[u32; MAX_FRACTRAN_PRIMES],
    ) -> Result<FractranCycleInfo, FractranError> {
        let (mut p1, mut p2, mut start_step) = (*initial, *meeting, 0usize);
        while p1 != p2 {
            let _ = self.step_exponents(&mut p1)?;
            let _ = self.step_exponents(&mut p2)?;
            start_step += 1;
        }
        let mut runner = p1;
        let _ = self.step_exponents(&mut runner)?;
        let mut period = 1usize;
        while runner != p1 {
            let _ = self.step_exponents(&mut runner)?;
            period += 1;
        }
        Ok(FractranCycleInfo {
            start_step,
            period,
            state: p1,
        })
    }

    /// Executes Floyd's tortoise-and-hare cycle detector for up to `max_steps` transitions on stack buffers.
    #[verified_engine::verified]
    pub fn detect_cycle(
        &self,
        initial: &[u32; MAX_FRACTRAN_PRIMES],
        max_steps: usize,
    ) -> Result<FractranTrajectoryOutcome, FractranError> {
        if let Some(err) = self.validation_error {
            return Err(err);
        }
        let (mut tortoise, mut hare, mut hare_steps) = (*initial, *initial, 0usize);
        for _ in 0..max_steps {
            if self.step_exponents(&mut tortoise)?.is_none() {
                return Ok(FractranTrajectoryOutcome::Halted {
                    steps: hare_steps,
                    final_exponents: tortoise,
                });
            }
            for _ in 0..2 {
                if self.step_exponents(&mut hare)?.is_none() {
                    return Ok(FractranTrajectoryOutcome::Halted {
                        steps: hare_steps,
                        final_exponents: hare,
                    });
                }
                hare_steps += 1;
            }
            if tortoise == hare {
                return Ok(FractranTrajectoryOutcome::CycleDetected(
                    self.locate_cycle_origin_and_period(initial, &tortoise)?,
                ));
            }
        }
        Ok(FractranTrajectoryOutcome::StepLimitReached {
            steps: max_steps,
            state: tortoise,
        })
    }

    /// Factors a positive 64-bit integer $n \ge 1$ into a 64-prime exponent array.
    #[verified_engine::verified]
    pub fn encode_u64(mut n: u64) -> Result<[u32; MAX_FRACTRAN_PRIMES], FractranError> {
        if n == 0 {
            return Err(FractranError::InvalidInitialValue);
        }
        let mut exponents = [0u32; MAX_FRACTRAN_PRIMES];
        for (idx, &prime) in FIRST_64_PRIMES.iter().enumerate() {
            while n.is_multiple_of(prime) {
                n /= prime;
                exponents[idx] += 1;
            }
            if n == 1 {
                return Ok(exponents);
            }
        }
        Err(FractranError::UnsupportedPrimeFactor {
            rule_index: usize::MAX,
            factor: smallest_factor_above_311(&UBig::from(n)),
        })
    }

    /// Reconstructs the exact arbitrary-precision state integer $\prod_i p_i^{e_i}$ from `exponents`.
    #[must_use]
    #[verified_engine::verified]
    pub fn decode_to_integer(exponents: &[u32]) -> Integer {
        let mut acc = IBig::from(1u8);
        for (idx, &exp) in exponents.iter().take(MAX_FRACTRAN_PRIMES).enumerate() {
            if exp > 0 {
                acc *= IBig::from(FIRST_64_PRIMES[idx]).pow(exp as usize);
            }
        }
        Integer(acc)
    }

    /// Returns `Some(k)` if `exponents` encodes a pure power of two $2^k$ with $k \ge 2$.
    #[must_use]
    #[verified_engine::verified]
    pub fn power_of_two_exponent(exponents: &[u32]) -> Option<u32> {
        let r2 = exponents.first().copied().unwrap_or(0);
        if r2 >= 2 && exponents.iter().skip(1).all(|&e| e == 0) {
            Some(r2)
        } else {
            None
        }
    }
}

// [cite:algorithmic_information_rust]
