//! `#![no_std]` zero-allocation log-domain max-product (max-sum) belief propagation.

use crate::bp_sum_product::{BpConfig, compute_factor_strides};
use crate::error::GraphEngineError;
use crate::factor_graph::{FactorGraphView, FactorSpec, MAX_CARDINALITY, MAX_FACTOR_ARITY};

/// Summary report returned by [`max_product_bp_into`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MaxProductReport {
    /// Number of max-sum sweeps executed.
    pub iterations: usize,
    /// Whether `max_residual <= config.tolerance`.
    pub converged: bool,
    /// Maximum $\ell_\infty$ change across shifted log-messages in the final sweep.
    pub max_residual: f64,
    /// Unnormalized log-potential score $\sum_f \ln \psi_f(\hat{x}_f)$ of the decoded MAP assignment.
    pub log_map_score: f64,
}

#[inline]
fn shift_log_message_zero_max(slice: &mut [f64]) {
    let mut max_v = f64::NEG_INFINITY;
    for &v in slice.iter() {
        if v > max_v {
            max_v = v;
        }
    }
    if max_v.is_finite() {
        for v in slice.iter_mut() {
            *v -= max_v;
        }
    } else {
        slice.fill(0.0);
    }
}

fn update_var_to_factor_log(
    graph: &FactorGraphView<'_>,
    log_f_to_v: &[f64],
    log_v_to_f: &mut [f64],
) {
    let mut target_cursor = 0usize;
    for (f_idx, f) in graph.factors().iter().enumerate() {
        for (pos, &v) in f.active_vars().iter().enumerate() {
            let k = graph.cardinalities()[v];
            let out = &mut log_v_to_f[target_cursor..target_cursor + k];
            out.fill(0.0);
            let mut scan_cursor = 0usize;
            for (other_f_idx, other_f) in graph.factors().iter().enumerate() {
                for (other_pos, &other_v) in other_f.active_vars().iter().enumerate() {
                    let other_k = graph.cardinalities()[other_v];
                    if other_v == v && (other_f_idx != f_idx || other_pos != pos) {
                        for s in 0..k {
                            out[s] += log_f_to_v[scan_cursor + s];
                        }
                    }
                    scan_cursor += other_k;
                }
            }
            shift_log_message_zero_max(out);
            target_cursor += k;
        }
    }
}

fn compute_single_factor_max_log(
    graph: &FactorGraphView<'_>,
    f: &FactorSpec,
    factor_edge_base: usize,
    log_v_to_f: &[f64],
    next_log_f_to_v: &mut [f64],
) {
    let cards = graph.cardinalities();
    let strides = compute_factor_strides(f, cards);
    let mut slot_offsets = [0usize; MAX_FACTOR_ARITY];
    let mut cur = factor_edge_base;
    for pos in 0..f.arity {
        slot_offsets[pos] = cur;
        cur += cards[f.vars[pos]];
    }
    let table = &graph.potentials()[f.table_offset..f.table_offset + f.table_len];
    for target_pos in 0..f.arity {
        let target_v = f.vars[target_pos];
        let target_k = cards[target_v];
        let mut best = [f64::NEG_INFINITY; MAX_CARDINALITY];
        for (flat_idx, &pot) in table.iter().enumerate() {
            if pot <= 0.0 {
                continue;
            }
            let target_state = (flat_idx / strides[target_pos]) % target_k;
            let mut score = libm::log(pot);
            for other_pos in 0..f.arity {
                if other_pos != target_pos {
                    let ov = f.vars[other_pos];
                    let os = (flat_idx / strides[other_pos]) % cards[ov];
                    score += log_v_to_f[slot_offsets[other_pos] + os];
                }
            }
            if score > best[target_state] {
                best[target_state] = score;
            }
        }
        let dst =
            &mut next_log_f_to_v[slot_offsets[target_pos]..slot_offsets[target_pos] + target_k];
        dst.copy_from_slice(&best[..target_k]);
        shift_log_message_zero_max(dst);
    }
}

fn decode_map_and_score(
    graph: &FactorGraphView<'_>,
    log_f_to_v: &[f64],
    max_marginals_out: &mut [f64],
    map_states_out: &mut [usize],
) -> f64 {
    let cards = graph.cardinalities();
    max_marginals_out[..graph.beliefs_len()].fill(0.0);
    let mut cursor = 0usize;
    for f in graph.factors() {
        for &v in f.active_vars() {
            let k = cards[v];
            let v_off = graph.var_belief_offset(v);
            for s in 0..k {
                max_marginals_out[v_off + s] += log_f_to_v[cursor + s];
            }
            cursor += k;
        }
    }
    for v in 0..graph.num_vars() {
        let v_off = graph.var_belief_offset(v);
        let k = cards[v];
        let slice = &mut max_marginals_out[v_off..v_off + k];
        let mut best_idx = 0usize;
        let mut best_val = f64::NEG_INFINITY;
        for (s, &val) in slice.iter().enumerate() {
            if val > best_val {
                best_val = val;
                best_idx = s;
            }
        }
        map_states_out[v] = best_idx;
        shift_log_message_zero_max(slice);
        let mut norm = 0.0;
        for slot in slice.iter_mut() {
            *slot = libm::exp(*slot);
            norm += *slot;
        }
        if norm > 0.0 {
            for slot in slice.iter_mut() {
                *slot /= norm;
            }
        }
    }
    evaluate_assignment_log_score(graph, map_states_out)
}

/// Evaluates $\sum_f \ln \psi_f(x_f)$ for a discrete state assignment `states`.
#[cfg_attr(feature = "verified", verified_engine::verified)]
pub fn evaluate_assignment_log_score(graph: &FactorGraphView<'_>, states: &[usize]) -> f64 {
    debug_assert!(states.len() >= graph.num_vars());
    let cards = graph.cardinalities();
    let mut total_log = 0.0;
    for f in graph.factors() {
        let strides = compute_factor_strides(f, cards);
        let mut idx = 0usize;
        for (&v, &stride) in f.active_vars().iter().zip(strides.iter()) {
            idx += states[v].min(cards[v] - 1) * stride;
        }
        let pot = graph.potentials()[f.table_offset + idx];
        if pot <= 0.0 {
            return f64::NEG_INFINITY;
        }
        total_log += libm::log(pot);
    }
    total_log
}

/// Runs log-domain max-product (max-sum) belief propagation on `graph` using caller-supplied
/// `max_marginals_out`, `map_states_out`, and `scratch` with zero heap allocations.
#[cfg_attr(feature = "verified", verified_engine::verified)]
pub fn max_product_bp_into(
    graph: &FactorGraphView<'_>,
    config: &BpConfig,
    max_marginals_out: &mut [f64],
    map_states_out: &mut [usize],
    scratch: &mut [f64],
) -> Result<MaxProductReport, GraphEngineError> {
    if max_marginals_out.len() < graph.beliefs_len() {
        return Err(GraphEngineError::BufferTooSmall {
            required: graph.beliefs_len(),
            actual: max_marginals_out.len(),
        });
    }
    if map_states_out.len() < graph.num_vars() {
        return Err(GraphEngineError::BufferTooSmall {
            required: graph.num_vars(),
            actual: map_states_out.len(),
        });
    }
    let req = graph.bp_scratch_len();
    if scratch.len() < req {
        return Err(GraphEngineError::BufferTooSmall {
            required: req,
            actual: scratch.len(),
        });
    }
    let damping = config.damping.clamp(0.0, 0.99);
    let edge_len = graph.edge_slot_len();
    let (log_f_to_v, rest) = scratch[..req].split_at_mut(edge_len);
    let (log_v_to_f, next_log_f_to_v) = rest.split_at_mut(edge_len);
    log_f_to_v.fill(0.0);

    let max_iters = config.max_iters.max(1);
    let mut iterations = 0usize;
    let mut max_residual = f64::INFINITY;
    let mut converged = false;

    for iter in 1..=max_iters {
        update_var_to_factor_log(graph, log_f_to_v, log_v_to_f);
        let mut factor_edge_base = 0usize;
        for f in graph.factors() {
            compute_single_factor_max_log(graph, f, factor_edge_base, log_v_to_f, next_log_f_to_v);
            for &v in f.active_vars() {
                factor_edge_base += graph.cardinalities()[v];
            }
        }
        max_residual = 0.0;
        for i in 0..edge_len {
            let updated = if next_log_f_to_v[i].is_finite() && log_f_to_v[i].is_finite() {
                (1.0 - damping) * next_log_f_to_v[i] + damping * log_f_to_v[i]
            } else {
                next_log_f_to_v[i]
            };
            let diff = if updated.is_finite() && log_f_to_v[i].is_finite() {
                (updated - log_f_to_v[i]).abs()
            } else if updated == log_f_to_v[i] {
                0.0
            } else {
                f64::INFINITY
            };
            if diff > max_residual {
                max_residual = diff;
            }
            log_f_to_v[i] = updated;
        }
        iterations = iter;
        if max_residual <= config.tolerance {
            converged = true;
            break;
        }
    }
    let log_map_score = decode_map_and_score(graph, log_f_to_v, max_marginals_out, map_states_out);
    Ok(MaxProductReport {
        iterations,
        converged,
        max_residual,
        log_map_score,
    })
}

/// Convenience `alloc`-enabled wrapper for [`max_product_bp_into`].
#[cfg(feature = "alloc")]
#[cfg_attr(feature = "verified", verified_engine::verified)]
pub fn max_product_bp(
    graph: &FactorGraphView<'_>,
    config: &BpConfig,
) -> Result<
    (
        MaxProductReport,
        alloc::vec::Vec<usize>,
        alloc::vec::Vec<f64>,
    ),
    GraphEngineError,
> {
    let mut marginals = alloc::vec![0.0_f64; graph.beliefs_len()];
    let mut states = alloc::vec![0usize; graph.num_vars()];
    let mut scratch = alloc::vec![0.0_f64; graph.bp_scratch_len()];
    let report = max_product_bp_into(graph, config, &mut marginals, &mut states, &mut scratch)?;
    Ok((report, states, marginals))
}
