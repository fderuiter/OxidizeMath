//! Zero-allocation prime-exponent FRACTRAN virtual machine and cycle detector.
//!
//! Pre-factors positive rational rules into stack-inline signed prime-exponent delta
//! vectors over the first 64 primes ($p \le 311$) so [`FractranProgram::step_exponents`]
//! and [`FractranProgram::detect_cycle`] execute without dynamic heap allocation.

use super::fractran_programs::{
    ADDER_PAIRS, FIBONACCI_PAIRS, MULTIPLIER_PAIRS, PRIMEGAME_PAIRS, compile_rational_rule,
    smallest_factor_above_311,
};
pub use super::fractran_programs::{
    FIRST_64_PRIMES, FractranCycleInfo, FractranError, FractranRule, FractranTrajectoryOutcome,
    MAX_FRACTRAN_PRIME, MAX_FRACTRAN_PRIMES, MAX_REGISTER_EXPONENT,
};
use crate::math_types::{Integer, Rational};
use dashu::integer::{IBig, UBig};

/// A FRACTRAN program consisting of ordered positive fractions pre-factored into prime-exponent deltas.
///
/// # Examples
///
/// ```rust
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::algorithmic_information::fractran::FractranProgram;
///
/// let adder = FractranProgram::adder();
/// let mut regs = FractranProgram::encode_u64(72).unwrap(); // 2^3 * 3^2
/// while adder.step_exponents(&mut regs).unwrap().is_some() {}
/// assert_eq!(regs[0], 0);
/// assert_eq!(regs[1], 5); // 3^(3 + 2)
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct FractranProgram {
    /// Original ordered fraction list preserved for inspection and display.
    pub fractions: Vec<Rational>,
    rules: Vec<FractranRule>,
    validation_error: Option<FractranError>,
}

impl FractranProgram {
    /// Constructs a [`FractranProgram`] from `fractions`, pre-factoring rules over $p \le 311$.
    ///
    /// Preserves the infallible constructor signature while recording any rule validation error
    /// so subsequent [`Self::step_exponents`] calls fail cleanly without panicking.
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

    /// Validates and compiles `fractions`, rejecting empty lists, non-positive fractions, or primes $> 311$.
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

    /// Parses comma- or whitespace-separated fractions (e.g. `"17/91, 78/85, 55/1"`) into a validated program.
    #[verified_engine::verified]
    pub fn parse(input: &str) -> Result<Self, FractranError> {
        let mut pairs = Vec::new();
        for (idx, raw) in input
            .split(|c: char| c == ',' || c == ';' || c.is_whitespace())
            .filter(|s| !s.is_empty())
            .enumerate()
        {
            if let Some((lhs, rhs)) = raw.split_once('/') {
                if rhs.contains('/') {
                    return Err(FractranError::InvalidFractionSyntax { token_index: idx });
                }
                let num = lhs
                    .trim()
                    .parse::<i128>()
                    .map_err(|_| FractranError::InvalidFractionSyntax { token_index: idx })?;
                let den = rhs
                    .trim()
                    .parse::<i128>()
                    .map_err(|_| FractranError::InvalidFractionSyntax { token_index: idx })?;
                pairs.push((num, den));
            } else {
                let num = raw
                    .trim()
                    .parse::<i128>()
                    .map_err(|_| FractranError::InvalidFractionSyntax { token_index: idx })?;
                pairs.push((num, 1));
            }
        }
        Self::try_from_pairs(&pairs)
    }

    /// Returns the pre-factored prime-exponent rules for zero-allocation inspection.
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

    /// Checks whether `rule` is applicable to `exponents` without mutating state.
    #[inline]
    #[verified_engine::verified]
    fn rule_matches(rule: &FractranRule, exponents: &[u32]) -> bool {
        for (idx, &delta) in rule.deltas.iter().enumerate().take(rule.active_len) {
            if delta < 0 {
                let needed = delta.unsigned_abs();
                let current = exponents.get(idx).copied().unwrap_or(0);
                if current < needed {
                    return false;
                }
            }
        }
        true
    }

    /// Validates cap bounds (`<= 1_000_000`) via `u32::checked_add` and updates `exponents` in place.
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

    /// Returns the index of the first matching fraction rule for `exponents`, or `None` if halted.
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
    ///
    /// Returns `Ok(Some(rule_idx))` when rule `rule_idx` fired, or `Ok(None)` when the machine halted.
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

    /// Recovers exact cycle parameters $(\mu, \lambda)$ once Floyd's tortoise and hare meet.
    #[verified_engine::verified]
    fn locate_cycle_origin_and_period(
        &self,
        initial: &[u32; MAX_FRACTRAN_PRIMES],
        meeting: &[u32; MAX_FRACTRAN_PRIMES],
    ) -> Result<FractranCycleInfo, FractranError> {
        let mut p1 = *initial;
        let mut p2 = *meeting;
        let mut start_step = 0usize;
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

    /// Executes Floyd's tortoise-and-hare cycle detector for up to `max_steps` transitions
    /// using stack-only register buffers (zero dynamic heap allocation).
    #[verified_engine::verified]
    pub fn detect_cycle(
        &self,
        initial: &[u32; MAX_FRACTRAN_PRIMES],
        max_steps: usize,
    ) -> Result<FractranTrajectoryOutcome, FractranError> {
        if let Some(err) = self.validation_error {
            return Err(err);
        }
        let mut tortoise = *initial;
        let mut hare = *initial;
        let mut hare_steps = 0usize;
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
                let info = self.locate_cycle_origin_and_period(initial, &tortoise)?;
                return Ok(FractranTrajectoryOutcome::CycleDetected(info));
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

    /// If `exponents` encodes a pure power of two $2^k$ with $k \ge 2$ and all odd prime registers zero,
    /// returns `Some(k)` (used to stream primes emitted by Conway's `PRIMEGAME`).
    #[must_use]
    #[verified_engine::verified]
    pub fn power_of_two_exponent(exponents: &[u32]) -> Option<u32> {
        let r2 = exponents.first().copied().unwrap_or(0);
        if r2 < 2 {
            return None;
        }
        if exponents.iter().skip(1).all(|&e| e == 0) {
            Some(r2)
        } else {
            None
        }
    }
}

/// Live prime-register state with incremental Floyd tortoise-and-hare cycle tracking.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FractranState {
    /// Control-state prime bases associated with Minsky state indicators.
    pub state_primes: Vec<u64>,
    /// Data-register prime bases holding Minsky counters.
    pub register_primes: Vec<u64>,
    /// Live 64-prime exponent array (serving simultaneously as Floyd's tortoise).
    pub exponents: [u32; MAX_FRACTRAN_PRIMES],
    /// Total transitions executed since initialization or reset.
    pub steps: usize,
    /// Whether the machine has reached a halting configuration.
    pub halted: bool,
    /// Cycle diagnostics once Floyd's hare meets the live exponent vector.
    pub cycle: Option<FractranCycleInfo>,
    initial_exponents: [u32; MAX_FRACTRAN_PRIMES],
    hare_exponents: [u32; MAX_FRACTRAN_PRIMES],
    hare_halted: bool,
}

impl FractranState {
    /// Creates a live [`FractranState`] from an initial 64-prime exponent configuration.
    #[must_use]
    #[verified_engine::verified]
    pub fn from_exponents(exponents: [u32; MAX_FRACTRAN_PRIMES]) -> Self {
        Self {
            state_primes: vec![11, 13, 17, 19, 23, 29],
            register_primes: vec![2, 3, 5, 7],
            exponents,
            steps: 0,
            halted: false,
            cycle: None,
            initial_exponents: exponents,
            hare_exponents: exponents,
            hare_halted: false,
        }
    }

    /// Creates a live [`FractranState`] by factoring a positive 64-bit integer $n \ge 1$.
    #[verified_engine::verified]
    pub fn from_u64(n: u64) -> Result<Self, FractranError> {
        let exponents = FractranProgram::encode_u64(n)?;
        Ok(Self::from_exponents(exponents))
    }

    /// Advances the machine by one step and updates incremental Floyd cycle detection.
    #[verified_engine::verified]
    pub fn step(&mut self, program: &FractranProgram) -> Result<Option<usize>, FractranError> {
        let Some(rule_idx) = program.step_exponents(&mut self.exponents)? else {
            self.halted = true;
            return Ok(None);
        };
        self.steps += 1;
        if !self.hare_halted && self.cycle.is_none() {
            for _ in 0..2 {
                if program.step_exponents(&mut self.hare_exponents)?.is_none() {
                    self.hare_halted = true;
                    break;
                }
            }
            if !self.hare_halted && self.exponents == self.hare_exponents {
                let info = program
                    .locate_cycle_origin_and_period(&self.initial_exponents, &self.exponents)?;
                self.cycle = Some(info);
            }
        }
        Ok(Some(rule_idx))
    }
}

// [cite:algorithmic_information_rust]
