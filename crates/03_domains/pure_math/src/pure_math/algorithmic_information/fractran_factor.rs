//! Smooth-prime factorization, fraction parsing, and Minsky transition helpers.

use super::fractran_programs::{
    FractranError, MAX_SMOOTH_PRIME, MULTIPLIER_PAIRS, PRIMEGAME_PAIRS,
};
use super::fractran_vm::{FractranCompiledVm, FractranPreset, MinskyTransition};
use crate::math_types::Rational;
use dashu::base::Sign;
use dashu::integer::{IBig, UBig};
use dashu::rational::RBig;
use std::collections::BTreeSet;

pub(super) type FactoredRule = (Vec<(u64, u32)>, Vec<(u64, u32)>);

pub(super) struct CompiledTables {
    pub(super) primes: Vec<u64>,
    pub(super) num_rules: usize,
    pub(super) numer_exponents: Vec<u32>,
    pub(super) denom_exponents: Vec<u32>,
}

const MINSKY_FIBONACCI_PAIRS: [(i128, i128); 9] = [
    (119, 26),
    (13, 17),
    (19, 13),
    (506, 57),
    (19, 23),
    (1, 19),
    (3, 7),
    (3, 11),
    (13, 5),
];

pub(super) fn preset_pairs(preset: FractranPreset) -> &'static [(i128, i128)] {
    match preset {
        FractranPreset::PrimeGame => &PRIMEGAME_PAIRS,
        FractranPreset::Mult => &MULTIPLIER_PAIRS,
        FractranPreset::Add => &[(2, 3)],
        FractranPreset::Fibonacci => &MINSKY_FIBONACCI_PAIRS,
    }
}

pub(super) fn factor_u64_smooth(
    mut rem: u64,
    out: &mut Vec<(u64, u32)>,
) -> Result<(), FractranError> {
    if rem <= 1 {
        return Ok(());
    }
    let tz = rem.trailing_zeros();
    if tz > 0 {
        out.push((2, tz));
        rem >>= tz;
    }
    let mut d = 3_u64;
    while d <= MAX_SMOOTH_PRIME && d.saturating_mul(d) <= rem {
        if rem.is_multiple_of(d) {
            let mut exp = 0_u32;
            while rem.is_multiple_of(d) {
                rem /= d;
                exp = exp.checked_add(1).ok_or(FractranError::UnsmoothFactor {
                    max_prime: MAX_SMOOTH_PRIME,
                })?;
            }
            out.push((d, exp));
        }
        d += 2;
    }
    if rem > 1 {
        if rem > MAX_SMOOTH_PRIME {
            return Err(FractranError::UnsmoothFactor {
                max_prime: MAX_SMOOTH_PRIME,
            });
        }
        out.push((rem, 1));
    }
    Ok(())
}

pub(super) fn factor_ubig_smooth(mut val: UBig) -> Result<Vec<(u64, u32)>, FractranError> {
    let mut factors = Vec::new();
    if let Ok(small) = u64::try_from(&val) {
        factor_u64_smooth(small, &mut factors)?;
        return Ok(factors);
    }
    let mut d = 2_u64;
    while d <= MAX_SMOOTH_PRIME && val > UBig::ONE {
        let divisor = UBig::from(d);
        if (&val % &divisor).is_zero() {
            let mut exp = 0_u32;
            while (&val % &divisor).is_zero() {
                val /= &divisor;
                exp = exp.checked_add(1).ok_or(FractranError::UnsmoothFactor {
                    max_prime: MAX_SMOOTH_PRIME,
                })?;
            }
            factors.push((d, exp));
            if let Ok(small) = u64::try_from(&val) {
                factor_u64_smooth(small, &mut factors)?;
                return Ok(factors);
            }
        }
        d = if d == 2 { 3 } else { d + 2 };
    }
    if val > UBig::ONE {
        return Err(FractranError::UnsmoothFactor {
            max_prime: MAX_SMOOTH_PRIME,
        });
    }
    Ok(factors)
}

pub(super) fn parse_fraction_tokens(input: &str) -> Result<Vec<Rational>, FractranError> {
    let mut fractions = Vec::new();
    for token in input.split(|c: char| c == ',' || c == ';' || c.is_whitespace()) {
        let trimmed = token.trim();
        if trimmed.is_empty() {
            continue;
        }
        let (num_str, den_str) = match trimmed.split_once('/') {
            Some((n, d)) => (n.trim(), d.trim()),
            None => (trimmed, "1"),
        };
        let numer =
            IBig::from_str_radix(num_str, 10).map_err(|_| FractranError::InvalidFractionSyntax)?;
        let denom =
            UBig::from_str_radix(den_str, 10).map_err(|_| FractranError::InvalidFractionSyntax)?;
        if denom.is_zero() {
            return Err(FractranError::InvalidFractionSyntax);
        }
        if numer <= IBig::ZERO {
            return Err(FractranError::NonPositiveFraction {
                rule_index: fractions.len(),
            });
        }
        fractions.push(Rational(RBig::from_parts(numer, denom)));
    }
    Ok(fractions)
}

pub(super) fn compile_exponent_tables(
    fractions: &[Rational],
    extra_primes: &[u64],
) -> Result<CompiledTables, FractranError> {
    let mut basis_set: BTreeSet<u64> = BTreeSet::from([2]);
    for &p in extra_primes {
        let mut extra_factors = Vec::new();
        factor_u64_smooth(p, &mut extra_factors)?;
        for (prime, _) in extra_factors {
            basis_set.insert(prime);
        }
    }
    let mut factored_rules: Vec<FactoredRule> = Vec::with_capacity(fractions.len());
    for (rule_index, frac) in fractions.iter().enumerate() {
        let (sign, num_mag) = frac.0.numerator().clone().into_parts();
        if sign == Sign::Negative || num_mag.is_zero() {
            return Err(FractranError::NonPositiveFraction { rule_index });
        }
        let num_factors = factor_ubig_smooth(num_mag)?;
        let den_factors = factor_ubig_smooth(frac.0.denominator().clone())?;
        for &(p, _) in num_factors.iter().chain(den_factors.iter()) {
            basis_set.insert(p);
        }
        factored_rules.push((num_factors, den_factors));
    }
    let primes: Vec<u64> = basis_set.into_iter().collect();
    let width = primes.len();
    let num_rules = factored_rules.len();
    let mut numer_exponents = vec![0_u32; num_rules * width];
    let mut denom_exponents = vec![0_u32; num_rules * width];
    for (r, (nf, df)) in factored_rules.into_iter().enumerate() {
        let row = r * width;
        for (p, exp) in nf {
            if let Ok(col) = primes.binary_search(&p) {
                numer_exponents[row + col] = exp;
            }
        }
        for (p, exp) in df {
            if let Ok(col) = primes.binary_search(&p) {
                denom_exponents[row + col] = exp;
            }
        }
    }
    Ok(CompiledTables {
        primes,
        num_rules,
        numer_exponents,
        denom_exponents,
    })
}

pub(super) fn extract_minsky_transitions(
    vm: &FractranCompiledVm,
    state_primes: &[u64],
) -> Vec<MinskyTransition> {
    let mut out = Vec::with_capacity(vm.num_rules());
    for rule_index in 0..vm.num_rules() {
        let Some((num, den)) = vm.rule_exponents(rule_index) else {
            continue;
        };
        let (mut from_state, mut to_state) = (None, None);
        let (mut guard_decrements, mut increments) = (Vec::new(), Vec::new());
        for (col, &p) in vm.primes().iter().enumerate() {
            let is_state = state_primes.contains(&p);
            if den[col] > 0 {
                if is_state && from_state.is_none() {
                    from_state = Some(p);
                } else {
                    guard_decrements.push((p, den[col]));
                }
            }
            if num[col] > 0 {
                if is_state && to_state.is_none() {
                    to_state = Some(p);
                } else {
                    increments.push((p, num[col]));
                }
            }
        }
        out.push(MinskyTransition {
            rule_index,
            from_state,
            to_state,
            guard_decrements,
            increments,
        });
    }
    out
}

// [cite:algorithmic_information_rust]
