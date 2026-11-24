//! # Cooperative (Coalitional) Game Theory & Exact Shapley Values
//!
//! Zero-allocation bitmask (`u16`) evaluation for transferable-utility coalitional games
//! $(N, v)$ with up to `MAX_COALITION_PLAYERS` (`16`) players, including exact `f64`
//! Shapley values, superadditivity, convexity (supermodularity), and Core excess diagnostics.
//!
//! ## Quick Start
//!
//! ```rust
//! # use domain_applied::__doctest_facade as math_explorer;
//! use math_explorer::applied::game_theory::cooperative::{is_in_core, shapley_values};
//!
//! // 3-player glove game: Player 0 holds a left glove; Players 1 and 2 hold right gloves.
//! // Coalitions with Player 0 and at least one of {1, 2} generate value 1.0.
//! let v = [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
//! let mut phi = [0.0; 3];
//! shapley_values(3, &v, &mut phi).unwrap();
//! assert!((phi[0] - 2.0 / 3.0).abs() < 1e-12);
//! assert!((phi[1] - 1.0 / 6.0).abs() < 1e-12);
//! assert!(!is_in_core(3, &v, &phi).unwrap()); // Unique core imputation is (1, 0, 0).
//! assert!(is_in_core(3, &v, &[1.0, 0.0, 0.0]).unwrap());
//! ```

use crate::error::GameTheoryError;
use math_commons::registry::TOLERANCE_FAST;

/// Maximum number of players supported by bitmask cooperative game solvers ($2^{16} = 65,536$ coalitions).
pub const MAX_COALITION_PLAYERS: usize = 16;

/// Validates `n_players` (`1..=16`) and a $2^n$-entry characteristic function $v$ with $v(\emptyset) = 0$.
#[verified_engine::verified]
pub fn validate_characteristic_function(
    n_players: usize,
    characteristic_values: &[f64],
) -> Result<(), GameTheoryError> {
    if n_players == 0 {
        return Err(GameTheoryError::EmptyStrategies);
    }
    if n_players > MAX_COALITION_PLAYERS {
        return Err(GameTheoryError::CoalitionCapExceeded {
            players: n_players,
            max_players: MAX_COALITION_PLAYERS,
        });
    }
    let expected = 1usize << n_players;
    if characteristic_values.len() != expected {
        return Err(GameTheoryError::DimensionMismatch {
            expected,
            actual: characteristic_values.len(),
        });
    }
    for (mask, &val) in characteristic_values.iter().enumerate() {
        if !val.is_finite() {
            return Err(GameTheoryError::InvalidPayoff {
                row: mask,
                col: 0,
                value: val,
            });
        }
    }
    if characteristic_values[0].abs() > TOLERANCE_FAST {
        return Err(GameTheoryError::InvalidPayoff {
            row: 0,
            col: 0,
            value: characteristic_values[0],
        });
    }
    Ok(())
}

/// Returns the `u16` bitmask of the grand coalition $N = \{0, \dots, n - 1\}$.
#[verified_engine::verified]
pub fn grand_coalition_mask(n_players: usize) -> Result<u16, GameTheoryError> {
    if n_players == 0 {
        return Err(GameTheoryError::EmptyStrategies);
    }
    if n_players > MAX_COALITION_PLAYERS {
        return Err(GameTheoryError::CoalitionCapExceeded {
            players: n_players,
            max_players: MAX_COALITION_PLAYERS,
        });
    }
    if n_players == MAX_COALITION_PLAYERS {
        Ok(u16::MAX)
    } else {
        Ok(((1u32 << n_players) - 1) as u16)
    }
}

/// Builds a `u16` coalition bitmask from player indices in `0..n_players`.
#[verified_engine::verified]
pub fn coalition_mask(players: &[usize], n_players: usize) -> Result<u16, GameTheoryError> {
    let _ = grand_coalition_mask(n_players)?;
    let mut mask = 0u16;
    for &p in players {
        if p >= n_players {
            return Err(GameTheoryError::DimensionMismatch {
                expected: n_players,
                actual: p,
            });
        }
        mask |= 1u16 << p;
    }
    Ok(mask)
}

/// Evaluates the characteristic worth $v(S)$ of a `u16` coalition bitmask.
#[verified_engine::verified]
pub fn evaluate_coalition(
    n_players: usize,
    characteristic_values: &[f64],
    mask: u16,
) -> Result<f64, GameTheoryError> {
    validate_characteristic_function(n_players, characteristic_values)?;
    let grand = grand_coalition_mask(n_players)?;
    if (mask & !grand) != 0 {
        return Err(GameTheoryError::DimensionMismatch {
            expected: grand as usize,
            actual: mask as usize,
        });
    }
    Ok(characteristic_values[mask as usize])
}

/// Computes exact `f64` Shapley values $\phi_i(v)$ for all `0..n_players` (`n_players <= 16`)
/// into `shapley_out` with zero dynamic heap allocations.
#[verified_engine::verified]
pub fn shapley_values(
    n_players: usize,
    characteristic_values: &[f64],
    shapley_out: &mut [f64],
) -> Result<(), GameTheoryError> {
    validate_characteristic_function(n_players, characteristic_values)?;
    if shapley_out.len() < n_players {
        return Err(GameTheoryError::WorkspaceTooSmall {
            required: n_players,
            actual: shapley_out.len(),
        });
    }

    let mut weights = [0.0; MAX_COALITION_PLAYERS];
    let mut binom = 1u64;
    let n_f64 = n_players as f64;
    for (s, w) in weights.iter_mut().enumerate().take(n_players) {
        *w = 1.0 / (n_f64 * (binom as f64));
        if s + 1 < n_players {
            binom = (binom * ((n_players - 1 - s) as u64)) / ((s + 1) as u64);
        }
    }

    let out = &mut shapley_out[..n_players];
    out.fill(0.0);

    let total_masks = 1usize << n_players;
    let full_mask = total_masks - 1;
    for mask in 0..full_mask {
        let v_s = characteristic_values[mask];
        let w = weights[mask.count_ones() as usize];
        let mut missing = (!mask) & full_mask;
        while missing != 0 {
            let player = missing.trailing_zeros() as usize;
            missing &= missing - 1;
            let marginal = characteristic_values[mask | (1usize << player)] - v_s;
            out[player] += w * marginal;
        }
    }
    Ok(())
}

/// Checks whether $(N, v)$ is superadditive: $v(S \cup T) \ge v(S) + v(T)$ for all disjoint $S, T$.
#[verified_engine::verified]
pub fn is_superadditive(
    n_players: usize,
    characteristic_values: &[f64],
) -> Result<bool, GameTheoryError> {
    validate_characteristic_function(n_players, characteristic_values)?;
    let total_masks = 1usize << n_players;
    let full_mask = total_masks - 1;

    for s in 1..full_mask {
        let v_s = characteristic_values[s];
        let comp = (!s) & full_mask;
        let mut t = comp;
        while t != 0 {
            if t > s
                && characteristic_values[s | t] + TOLERANCE_FAST < v_s + characteristic_values[t]
            {
                return Ok(false);
            }
            t = (t - 1) & comp;
        }
    }
    Ok(true)
}

/// Checks whether $(N, v)$ is convex (supermodular): marginal contributions are non-decreasing in $S$.
#[verified_engine::verified]
pub fn is_convex(n_players: usize, characteristic_values: &[f64]) -> Result<bool, GameTheoryError> {
    validate_characteristic_function(n_players, characteristic_values)?;
    let total_masks = 1usize << n_players;
    for mask in 0..total_masks {
        for i in 0..n_players {
            if (mask & (1usize << i)) != 0 {
                continue;
            }
            for j in (i + 1)..n_players {
                if (mask & (1usize << j)) != 0 {
                    continue;
                }
                let lhs = characteristic_values[mask | (1usize << i) | (1usize << j)]
                    - characteristic_values[mask | (1usize << i)];
                let rhs = characteristic_values[mask | (1usize << j)] - characteristic_values[mask];
                if lhs + TOLERANCE_FAST < rhs {
                    return Ok(false);
                }
            }
        }
    }
    Ok(true)
}

/// Computes the maximum blocking excess $\max_{\emptyset \subsetneq S \subsetneq N} (v(S) - x(S))$
/// and its coalition bitmask for an efficient payoff vector $x$.
#[verified_engine::verified]
pub fn max_core_excess(
    n_players: usize,
    characteristic_values: &[f64],
    allocation: &[f64],
) -> Result<(f64, u16), GameTheoryError> {
    validate_characteristic_function(n_players, characteristic_values)?;
    if allocation.len() < n_players {
        return Err(GameTheoryError::DimensionMismatch {
            expected: n_players,
            actual: allocation.len(),
        });
    }
    let total_masks = 1usize << n_players;
    let grand = total_masks - 1;
    let mut sum_alloc = 0.0;
    for (idx, &x) in allocation.iter().enumerate().take(n_players) {
        if !x.is_finite() {
            return Err(GameTheoryError::InvalidPayoff {
                row: idx,
                col: 0,
                value: x,
            });
        }
        sum_alloc += x;
    }
    let eff_err = (characteristic_values[grand] - sum_alloc).abs();
    if eff_err > TOLERANCE_FAST {
        return Ok((eff_err, grand as u16));
    }

    let mut worst_excess = 0.0_f64;
    let mut worst_mask = 0u16;
    for (mask, &v_s) in characteristic_values.iter().enumerate().take(grand).skip(1) {
        let mut x_s = 0.0;
        let mut bits = mask;
        while bits != 0 {
            let i = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            x_s += allocation[i];
        }
        let excess = v_s - x_s;
        if worst_mask == 0 || excess > worst_excess {
            worst_excess = excess;
            worst_mask = mask as u16;
        }
    }
    Ok((worst_excess, worst_mask))
}

/// Returns `true` iff `allocation` is efficient ($\sum_i x_i = v(N)$) and coalitionally rational ($x(S) \ge v(S)$).
#[verified_engine::verified]
pub fn is_in_core(
    n_players: usize,
    characteristic_values: &[f64],
    allocation: &[f64],
) -> Result<bool, GameTheoryError> {
    let (excess, _) = max_core_excess(n_players, characteristic_values, allocation)?;
    Ok(excess <= TOLERANCE_FAST)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use math_commons::registry::TOLERANCE_HIGH;

    #[test]
    #[verified_engine::verified]
    fn test_shapley_axioms_and_convex_core() {
        // Convex 3-player synergy game: v(S) = |S|^2 - |S| (0 for singletons, 2 for pairs, 6 for grand).
        let v = [0.0, 0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 6.0];
        assert!(is_superadditive(3, &v).unwrap());
        assert!(is_convex(3, &v).unwrap());

        let mut phi = [0.0; 3];
        shapley_values(3, &v, &mut phi).unwrap();
        for &val in &phi {
            assert_relative_eq!(val, 2.0, epsilon = TOLERANCE_HIGH);
        }
        assert!(is_in_core(3, &v, &phi).unwrap());

        // Dummy player axiom: Player 3 contributes 0 to every coalition.
        let mut v4 = [0.0; 16];
        for mask in 0..8 {
            v4[mask] = v[mask];
            v4[mask | 8] = v[mask];
        }
        let mut phi4 = [0.0; 4];
        shapley_values(4, &v4, &mut phi4).unwrap();
        assert_relative_eq!(phi4[3], 0.0, epsilon = TOLERANCE_HIGH);
        assert_relative_eq!(phi4[0] + phi4[1] + phi4[2], 6.0, epsilon = TOLERANCE_HIGH);
    }

    #[test]
    #[verified_engine::verified]
    fn test_coalition_cap_enforced_at_16() {
        let mut out = [0.0; 17];
        assert!(matches!(
            shapley_values(17, &[], &mut out),
            Err(GameTheoryError::CoalitionCapExceeded {
                players: 17,
                max_players: 16
            })
        ));
        assert_eq!(grand_coalition_mask(16).unwrap(), u16::MAX);
        assert_eq!(coalition_mask(&[0, 2], 3).unwrap(), 0b101);
    }
}

// [cite:graph_parameters_rust]
