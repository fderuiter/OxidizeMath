//! Bounded 2D stack-allocated score probability grid indexed via `flatten_2d_index`.

use crate::error::MatchModelError;
use math_commons::math_kernel::types::flatten_2d_index;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Default maximum goals per team (`0..=15`, `16 x 16 = 256` active cells).
pub const DEFAULT_MAX_GOALS: usize = 15;

/// Hard upper bound on `max_goals` (`0..=31`, `32 x 32 = 1024` active cells).
pub const MAX_GOALS_LIMIT: usize = 31;

/// Fixed internal row stride (`MAX_GOALS_LIMIT + 1 = 32`).
pub const GRID_STRIDE: usize = MAX_GOALS_LIMIT + 1;

/// Fixed stack buffer capacity (`32 * 32 = 1024` cells, zero heap allocation).
pub const GRID_CAPACITY: usize = GRID_STRIDE * GRID_STRIDE;

/// Bounded 2D joint probability grid over `(home_goals, away_goals)` in `0..=max_goals`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScoreGrid {
    max_goals: usize,
    cells: [f64; GRID_CAPACITY],
    unnormalized_mass: f64,
    tail_mass: f64,
    normalized: bool,
}

impl ScoreGrid {
    /// Builds a validated [`ScoreGrid`] on `0..=max_goals` from a cell generator `f(home, away)`
    /// without heap allocation.
    #[verified_engine::verified]
    pub fn from_fn<F>(max_goals: usize, normalize: bool, mut f: F) -> Result<Self, MatchModelError>
    where
        F: FnMut(usize, usize) -> f64,
    {
        if max_goals > MAX_GOALS_LIMIT {
            return Err(MatchModelError::InvalidMaxGoals {
                max_goals,
                limit: MAX_GOALS_LIMIT,
            });
        }
        let mut cells = [0.0_f64; GRID_CAPACITY];
        let mut sum = 0.0_f64;
        for home in 0..=max_goals {
            for away in 0..=max_goals {
                let p = f(home, away);
                if !p.is_finite() {
                    return Err(MatchModelError::InvalidProbabilityMass { mass: p });
                }
                if p < 0.0 {
                    return Err(MatchModelError::NegativeCellProbability {
                        home,
                        away,
                        value: p,
                    });
                }
                let idx = flatten_2d_index(away, home, GRID_STRIDE);
                cells[idx] = p;
                sum += p;
            }
        }
        Self::finalize_cells(max_goals, cells, sum, normalize)
    }

    /// Builds a validated [`ScoreGrid`] from a row-major slice of length `(max_goals + 1)^2`.
    #[verified_engine::verified]
    pub fn from_row_major_slice(
        max_goals: usize,
        values: &[f64],
        normalize: bool,
    ) -> Result<Self, MatchModelError> {
        if max_goals > MAX_GOALS_LIMIT {
            return Err(MatchModelError::InvalidMaxGoals {
                max_goals,
                limit: MAX_GOALS_LIMIT,
            });
        }
        let side = max_goals + 1;
        if values.len() != side * side {
            return Err(MatchModelError::InvalidProbabilityMass {
                mass: values.len() as f64,
            });
        }
        Self::from_fn(max_goals, normalize, |home, away| {
            values[flatten_2d_index(away, home, side)]
        })
    }

    #[verified_engine::verified]
    fn finalize_cells(
        max_goals: usize,
        mut cells: [f64; GRID_CAPACITY],
        sum: f64,
        normalize: bool,
    ) -> Result<Self, MatchModelError> {
        if !sum.is_finite() || sum <= 0.0 {
            return Err(MatchModelError::InvalidProbabilityMass { mass: sum });
        }
        let tail_mass = (1.0 - sum).max(0.0);
        if normalize {
            let inv = 1.0 / sum;
            for home in 0..=max_goals {
                for away in 0..=max_goals {
                    let idx = flatten_2d_index(away, home, GRID_STRIDE);
                    cells[idx] *= inv;
                }
            }
        }
        Ok(Self {
            max_goals,
            cells,
            unnormalized_mass: sum,
            tail_mass,
            normalized: normalize,
        })
    }

    /// Returns a mass-normalized copy of this grid while retaining original `unnormalized_mass` and `tail_mass`.
    #[verified_engine::verified]
    pub fn normalized(&self) -> Result<Self, MatchModelError> {
        if self.normalized {
            return Ok(*self);
        }
        let mut copy = *self;
        let inv = 1.0 / self.unnormalized_mass;
        for home in 0..=self.max_goals {
            for away in 0..=self.max_goals {
                let idx = flatten_2d_index(away, home, GRID_STRIDE);
                copy.cells[idx] *= inv;
            }
        }
        copy.normalized = true;
        Ok(copy)
    }

    /// Maximum goals tracked per team (`0..=max_goals`).
    #[verified_engine::verified]
    pub fn max_goals(&self) -> usize {
        self.max_goals
    }

    /// Unnormalized probability mass summed over `0..=max_goals` prior to optional normalization.
    #[verified_engine::verified]
    pub fn unnormalized_mass(&self) -> f64 {
        self.unnormalized_mass
    }

    /// Truncated upper-tail residual `(1.0 - unnormalized_mass).max(0.0)`.
    #[verified_engine::verified]
    pub fn tail_mass(&self) -> f64 {
        self.tail_mass
    }

    /// Whether cell probabilities have been renormalized to sum to `1.0`.
    #[verified_engine::verified]
    pub fn is_normalized(&self) -> bool {
        self.normalized
    }

    /// Active probability mass currently stored in the grid (`1.0` if normalized, else `unnormalized_mass`).
    #[verified_engine::verified]
    pub fn active_mass(&self) -> f64 {
        let mut sum = 0.0;
        for home in 0..=self.max_goals {
            for away in 0..=self.max_goals {
                sum += self.prob_unchecked(home, away);
            }
        }
        sum
    }

    ///Checked probability lookup $P(X = \text{home}, Y = \text{away})$.
    #[verified_engine::verified]
    pub fn try_prob(&self, home: usize, away: usize) -> Result<f64, MatchModelError> {
        if home > self.max_goals || away > self.max_goals {
            return Err(MatchModelError::ScoreOutOfBounds {
                home,
                away,
                max_goals: self.max_goals,
            });
        }
        Ok(self.prob_unchecked(home, away))
    }

    /// Returns $P(X = \text{home}, Y = \text{away})$, or `0.0` when outside `0..=max_goals`.
    #[verified_engine::verified]
    pub fn prob(&self, home: usize, away: usize) -> f64 {
        if home > self.max_goals || away > self.max_goals {
            0.0
        } else {
            self.prob_unchecked(home, away)
        }
    }

    #[inline(always)]
    #[verified_engine::verified]
    pub(crate) fn prob_unchecked(&self, home: usize, away: usize) -> f64 {
        self.cells[flatten_2d_index(away, home, GRID_STRIDE)]
    }

    /// Expected home goals $\sum_{i,j} i \cdot P(i, j)$ over the bounded grid.
    #[verified_engine::verified]
    pub fn expected_home_goals(&self) -> f64 {
        let mut mean = 0.0;
        for home in 1..=self.max_goals {
            let mut row_sum = 0.0;
            for away in 0..=self.max_goals {
                row_sum += self.prob_unchecked(home, away);
            }
            mean += (home as f64) * row_sum;
        }
        mean
    }

    /// Expected away goals $\sum_{i,j} j \cdot P(i, j)$ over the bounded grid.
    #[verified_engine::verified]
    pub fn expected_away_goals(&self) -> f64 {
        let mut mean = 0.0;
        for home in 0..=self.max_goals {
            for away in 1..=self.max_goals {
                mean += (away as f64) * self.prob_unchecked(home, away);
            }
        }
        mean
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ScoreGridWire {
    max_goals: usize,
    normalized: bool,
    cells: Vec<f64>,
}

impl Serialize for ScoreGrid {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let side = self.max_goals + 1;
        let mut cells = Vec::with_capacity(side * side);
        for home in 0..=self.max_goals {
            for away in 0..=self.max_goals {
                cells.push(self.prob_unchecked(home, away));
            }
        }
        let wire = ScoreGridWire {
            max_goals: self.max_goals,
            normalized: self.normalized,
            cells,
        };
        wire.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ScoreGrid {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let wire = ScoreGridWire::deserialize(deserializer)?;
        Self::from_row_major_slice(wire.max_goals, &wire.cells, wire.normalized)
            .map_err(serde::de::Error::custom)
    }
}
