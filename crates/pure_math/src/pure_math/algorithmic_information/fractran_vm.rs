//! Preallocated dynamic-basis [`FractranCompiledVm`], [`FractranPreset`], and Minsky counter-machine types.

use super::fractran::FractranProgram;
use super::fractran_factor;
use super::fractran_programs::{FractranCycleInfo, FractranError, MAX_FRACTRAN_PRIMES};
use crate::math_types::Rational;

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
        let fractions = fractran_factor::preset_pairs(self)
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

/// Preallocated prime-exponent virtual machine for zero-allocation FRACTRAN execution.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct FractranCompiledVm {
    pub(super) primes: Vec<u64>,
    pub(super) num_rules: usize,
    pub(super) numer_exponents: Vec<u32>,
    pub(super) denom_exponents: Vec<u32>,
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
        fractran_factor::factor_u64_smooth(value, &mut factors)?;
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

/// Minsky prime partition and live 64-prime register state with incremental Floyd cycle tracking.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FractranState {
    /// Primes acting as finite-state control indicators.
    pub state_primes: Vec<u64>,
    /// Primes whose exponents encode unbounded non-negative integer counters.
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
    /// Creates a new Minsky prime partition from explicit `state_primes` and `register_primes`.
    #[must_use]
    pub fn new(state_primes: Vec<u64>, register_primes: Vec<u64>) -> Self {
        Self {
            state_primes,
            register_primes,
            exponents: [0; MAX_FRACTRAN_PRIMES],
            steps: 0,
            halted: false,
            cycle: None,
            initial_exponents: [0; MAX_FRACTRAN_PRIMES],
            hare_exponents: [0; MAX_FRACTRAN_PRIMES],
            hare_halted: false,
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
        Self::new(state_primes, register_primes)
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
        fractran_factor::extract_minsky_transitions(vm, &self.state_primes)
    }

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
        Ok(Self::from_exponents(FractranProgram::encode_u64(n)?))
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
                self.cycle = Some(
                    program
                        .locate_cycle_origin_and_period(&self.initial_exponents, &self.exponents)?,
                );
            }
        }
        Ok(Some(rule_idx))
    }
}

#[cfg(feature = "serde")]
impl serde::Serialize for FractranProgram {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        #[derive(serde::Serialize)]
        struct RawProgram<'a> {
            fractions: &'a [Rational],
        }
        RawProgram {
            fractions: &self.fractions,
        }
        .serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for FractranProgram {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct RawProgram {
            fractions: Vec<Rational>,
        }
        let raw = RawProgram::deserialize(deserializer)?;
        Ok(Self::new(raw.fractions))
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for FractranCompiledVm {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct RawVm {
            primes: Vec<u64>,
            num_rules: usize,
            numer_exponents: Vec<u32>,
            denom_exponents: Vec<u32>,
        }
        let raw = RawVm::deserialize(deserializer)?;
        if raw.primes.first().is_some_and(|&p| p < 2)
            || !raw.primes.windows(2).all(|pair| pair[0] < pair[1])
        {
            return Err(serde::de::Error::custom(
                "FractranCompiledVm primes must be strictly ascending integers >= 2",
            ));
        }
        let expected_len = raw
            .num_rules
            .checked_mul(raw.primes.len())
            .ok_or_else(|| serde::de::Error::custom("FractranCompiledVm table size overflow"))?;
        if raw.numer_exponents.len() != expected_len || raw.denom_exponents.len() != expected_len {
            return Err(serde::de::Error::custom(
                "FractranCompiledVm exponent matrix length does not match num_rules * primes.len()",
            ));
        }
        Ok(Self {
            primes: raw.primes,
            num_rules: raw.num_rules,
            numer_exponents: raw.numer_exponents,
            denom_exponents: raw.denom_exponents,
        })
    }
}

// [cite:algorithmic_information_rust]
