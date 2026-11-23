//! Zero-allocation slice-backed discrete factor-graph representation (`FactorSpec`, `FactorGraphView`).

use crate::error::GraphEngineError;

/// Maximum supported factor arity ($k \le 4$) for zero-allocation stack stride decoding.
pub const MAX_FACTOR_ARITY: usize = 4;

/// Maximum variable state cardinality supported by stack-buffered factor message marginalization.
pub const MAX_CARDINALITY: usize = 64;

/// Describes a discrete factor $\psi_f(x_{v_0}, \dots, x_{v_{k-1}})$ referencing a contiguous
/// row-major slice `potentials[table_offset .. table_offset + table_len]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FactorSpec {
    /// Participating variable indices (`vars[0..arity]`).
    pub vars: [usize; MAX_FACTOR_ARITY],
    /// Number of variables in the factor (`1..=MAX_FACTOR_ARITY`).
    pub arity: usize,
    /// Starting index in the shared `potentials` slice.
    pub table_offset: usize,
    /// Product of participating variable cardinalities.
    pub table_len: usize,
}

impl FactorSpec {
    /// Constructs a unary factor $\psi_f(x_v)$ of length `cardinality` at `table_offset`.
    #[must_use]
    #[inline]
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    pub const fn unary(var: usize, table_offset: usize, cardinality: usize) -> Self {
        Self {
            vars: [var, 0, 0, 0],
            arity: 1,
            table_offset,
            table_len: cardinality,
        }
    }

    /// Constructs a pairwise factor $\psi_f(x_a, x_b)$ of row-major shape `(card_a, card_b)` at `table_offset`.
    #[must_use]
    #[inline]
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    pub const fn pairwise(
        var_a: usize,
        var_b: usize,
        table_offset: usize,
        card_a: usize,
        card_b: usize,
    ) -> Self {
        Self {
            vars: [var_a, var_b, 0, 0],
            arity: 2,
            table_offset,
            table_len: card_a * card_b,
        }
    }

    /// Constructs a factor of arity `1..=MAX_FACTOR_ARITY` over `vars`.
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    pub fn from_vars(
        vars: &[usize],
        cardinalities: &[usize],
        table_offset: usize,
    ) -> Result<Self, GraphEngineError> {
        if vars.is_empty() || vars.len() > MAX_FACTOR_ARITY {
            return Err(GraphEngineError::InvalidFactor {
                reason: "factor arity must be in 1..=MAX_FACTOR_ARITY",
            });
        }
        let mut arr = [0usize; MAX_FACTOR_ARITY];
        let mut table_len = 1usize;
        for (idx, &v) in vars.iter().enumerate() {
            if v >= cardinalities.len() {
                return Err(GraphEngineError::InvalidVariable {
                    var: v,
                    value: cardinalities.len(),
                });
            }
            for &prev in &vars[..idx] {
                if prev == v {
                    return Err(GraphEngineError::InvalidFactor {
                        reason: "duplicate variable in factor scope",
                    });
                }
            }
            arr[idx] = v;
            table_len = table_len.saturating_mul(cardinalities[v]);
        }
        debug_assert!(table_len > 0);
        Ok(Self {
            vars: arr,
            arity: vars.len(),
            table_offset,
            table_len,
        })
    }

    /// Active variable slice `vars[0..arity]`.
    #[must_use]
    #[inline]
    pub fn active_vars(&self) -> &[usize] {
        &self.vars[..self.arity]
    }
}

/// Validated zero-allocation borrowed view of a discrete factor graph.
#[derive(Debug, Clone, Copy)]
pub struct FactorGraphView<'a> {
    cardinalities: &'a [usize],
    factors: &'a [FactorSpec],
    potentials: &'a [f64],
    total_belief_len: usize,
    total_edge_slots: usize,
}

fn validate_factor_spec(
    f: &FactorSpec,
    cardinalities: &[usize],
    potentials: &[f64],
) -> Result<usize, GraphEngineError> {
    if f.arity == 0 || f.arity > MAX_FACTOR_ARITY {
        return Err(GraphEngineError::InvalidFactor {
            reason: "factor arity outside 1..=MAX_FACTOR_ARITY",
        });
    }
    let mut expected_len = 1usize;
    let mut edge_slots = 0usize;
    for pos in 0..f.arity {
        let v = f.vars[pos];
        if v >= cardinalities.len() {
            return Err(GraphEngineError::InvalidVariable {
                var: v,
                value: cardinalities.len(),
            });
        }
        for prev in 0..pos {
            if f.vars[prev] == v {
                return Err(GraphEngineError::InvalidFactor {
                    reason: "duplicate variable inside factor",
                });
            }
        }
        expected_len = expected_len.saturating_mul(cardinalities[v]);
        edge_slots += cardinalities[v];
    }
    if f.table_len != expected_len || f.table_offset.saturating_add(f.table_len) > potentials.len()
    {
        return Err(GraphEngineError::InvalidFactor {
            reason: "factor potential table slice out of bounds or mismatched length",
        });
    }
    let table = &potentials[f.table_offset..f.table_offset + f.table_len];
    let mut total_mass = 0.0;
    for &val in table {
        if !val.is_finite() || val < 0.0 {
            return Err(GraphEngineError::InvalidPotential {
                reason: "factor potentials must be finite and non-negative",
            });
        }
        total_mass += val;
    }
    if total_mass <= 0.0 {
        return Err(GraphEngineError::InvalidPotential {
            reason: "factor potential table cannot be identically zero",
        });
    }
    Ok(edge_slots)
}

impl<'a> FactorGraphView<'a> {
    /// Validates `cardinalities`, `factors`, and `potentials` without heap allocations.
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    pub fn new(
        cardinalities: &'a [usize],
        factors: &'a [FactorSpec],
        potentials: &'a [f64],
    ) -> Result<Self, GraphEngineError> {
        if cardinalities.is_empty() {
            return Err(GraphEngineError::EmptyGraph {
                context: "cardinalities",
            });
        }
        if factors.is_empty() {
            return Err(GraphEngineError::EmptyGraph { context: "factors" });
        }
        let mut total_belief_len = 0usize;
        for (v, &k) in cardinalities.iter().enumerate() {
            if k == 0 || k > MAX_CARDINALITY {
                return Err(GraphEngineError::InvalidVariable { var: v, value: k });
            }
            total_belief_len += k;
        }
        let mut total_edge_slots = 0usize;
        for f in factors {
            total_edge_slots += validate_factor_spec(f, cardinalities, potentials)?;
        }
        debug_assert!(total_belief_len > 0 && total_edge_slots > 0);
        Ok(Self {
            cardinalities,
            factors,
            potentials,
            total_belief_len,
            total_edge_slots,
        })
    }

    /// Number of discrete random variables in the factor graph.
    #[must_use]
    #[inline]
    pub const fn num_vars(&self) -> usize {
        self.cardinalities.len()
    }

    /// Number of factors in the factor graph.
    #[must_use]
    #[inline]
    pub const fn num_factors(&self) -> usize {
        self.factors.len()
    }

    /// Variable state cardinalities slice.
    #[must_use]
    #[inline]
    pub const fn cardinalities(&self) -> &'a [usize] {
        self.cardinalities
    }

    /// Factor specifications slice.
    #[must_use]
    #[inline]
    pub const fn factors(&self) -> &'a [FactorSpec] {
        self.factors
    }

    /// Flat potential values slice.
    #[must_use]
    #[inline]
    pub const fn potentials(&self) -> &'a [f64] {
        self.potentials
    }

    /// Required length of the flat variable-beliefs output buffer ($\sum_v K_v$).
    #[must_use]
    #[inline]
    pub const fn beliefs_len(&self) -> usize {
        self.total_belief_len
    }

    /// Total number of scalar message entries across all factor-variable edges ($\sum_{(f, v)} K_v$).
    #[must_use]
    #[inline]
    pub const fn edge_slot_len(&self) -> usize {
        self.total_edge_slots
    }

    /// Required `f64` scratch buffer length (`3 * edge_slot_len()`) for sum-product and max-product BP.
    #[must_use]
    #[inline]
    pub const fn bp_scratch_len(&self) -> usize {
        3 * self.total_edge_slots
    }

    /// Starting index of variable `var` inside a flat `beliefs` buffer of length `self.beliefs_len()`.
    #[must_use]
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    pub fn var_belief_offset(&self, var: usize) -> usize {
        debug_assert!(var < self.cardinalities.len());
        let mut off = 0usize;
        let mut i = 0usize;
        while i < var {
            off += self.cardinalities[i];
            i += 1;
        }
        off
    }

    /// Returns the marginal belief slice for variable `var` from `beliefs`.
    #[must_use]
    pub fn var_belief<'b>(&self, beliefs: &'b [f64], var: usize) -> &'b [f64] {
        let off = self.var_belief_offset(var);
        let k = self.cardinalities[var];
        &beliefs[off..off + k]
    }
}
