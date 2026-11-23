//! `#![no_std]` zero-allocation sum-product belief propagation and Bethe log-partition estimation.

use crate::error::GraphEngineError;
use crate::factor_graph::{FactorGraphView, FactorSpec, MAX_CARDINALITY, MAX_FACTOR_ARITY};

/// Configuration for iterative sum-product and max-product message passing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BpConfig {
    /// Maximum synchronous message-passing sweeps.
    pub max_iters: usize,
    /// $\ell_\infty$ message residual threshold for convergence.
    pub tolerance: f64,
    /// Convex message damping factor in `[0.0, 1.0)` (`0.0` = undamped).
    pub damping: f64,
}

impl Default for BpConfig {
    fn default() -> Self {
        Self {
            max_iters: 100,
            tolerance: 1.0e-12,
            damping: 0.0,
        }
    }
}

/// Summary report returned by [`sum_product_bp_into`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BpReport {
    /// Number of sweeps executed.
    pub iterations: usize,
    /// Whether `max_residual <= config.tolerance`.
    pub converged: bool,
    /// Maximum $\ell_\infty$ change across all factor-to-variable messages in the final sweep.
    pub max_residual: f64,
    /// Exact $\ln Z$ on trees/chains (Bethe free-energy approximation $-\mathcal{F}_{\text{Bethe}}$ on loopy graphs).
    pub log_partition: f64,
}

pub(crate) fn compute_factor_strides(
    f: &FactorSpec,
    cardinalities: &[usize],
) -> [usize; MAX_FACTOR_ARITY] {
    let mut strides = [1usize; MAX_FACTOR_ARITY];
    let mut running = 1usize;
    for pos in (0..f.arity).rev() {
        strides[pos] = running;
        running *= cardinalities[f.vars[pos]];
    }
    strides
}

fn init_uniform_messages(graph: &FactorGraphView<'_>, msg_buf: &mut [f64]) {
    let mut cursor = 0usize;
    for f in graph.factors() {
        for &v in f.active_vars() {
            let k = graph.cardinalities()[v];
            let unif = 1.0 / (k as f64);
            msg_buf[cursor..cursor + k].fill(unif);
            cursor += k;
        }
    }
}

fn update_var_to_factor_messages(
    graph: &FactorGraphView<'_>,
    msg_f_to_v: &[f64],
    msg_v_to_f: &mut [f64],
) {
    let mut target_cursor = 0usize;
    for (f_idx, f) in graph.factors().iter().enumerate() {
        for (pos, &v) in f.active_vars().iter().enumerate() {
            let k = graph.cardinalities()[v];
            let out = &mut msg_v_to_f[target_cursor..target_cursor + k];
            out.fill(1.0);
            let mut scan_cursor = 0usize;
            for (other_f_idx, other_f) in graph.factors().iter().enumerate() {
                for (other_pos, &other_v) in other_f.active_vars().iter().enumerate() {
                    let other_k = graph.cardinalities()[other_v];
                    if other_v == v && (other_f_idx != f_idx || other_pos != pos) {
                        for s in 0..k {
                            out[s] *= msg_f_to_v[scan_cursor + s];
                        }
                    }
                    scan_cursor += other_k;
                }
            }
            normalize_message_slice(out);
            target_cursor += k;
        }
    }
}

#[inline]
fn normalize_message_slice(slice: &mut [f64]) {
    let sum: f64 = slice.iter().copied().sum();
    if sum > 0.0 {
        for v in slice.iter_mut() {
            *v /= sum;
        }
    } else {
        let unif = 1.0 / (slice.len() as f64);
        slice.fill(unif);
    }
}

fn compute_single_factor_messages(
    graph: &FactorGraphView<'_>,
    f: &FactorSpec,
    factor_edge_base: usize,
    msg_v_to_f: &[f64],
    next_f_to_v: &mut [f64],
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
        let mut accum = [0.0_f64; MAX_CARDINALITY];
        for (flat_idx, &pot) in table.iter().enumerate() {
            if pot == 0.0 {
                continue;
            }
            let target_state = (flat_idx / strides[target_pos]) % target_k;
            let mut prod = pot;
            for other_pos in 0..f.arity {
                if other_pos != target_pos {
                    let ov = f.vars[other_pos];
                    let os = (flat_idx / strides[other_pos]) % cards[ov];
                    prod *= msg_v_to_f[slot_offsets[other_pos] + os];
                }
            }
            accum[target_state] += prod;
        }
        let dst = &mut next_f_to_v[slot_offsets[target_pos]..slot_offsets[target_pos] + target_k];
        dst.copy_from_slice(&accum[..target_k]);
        normalize_message_slice(dst);
    }
}

fn compute_beliefs_and_bethe(
    graph: &FactorGraphView<'_>,
    msg_f_to_v: &[f64],
    msg_v_to_f: &[f64],
    beliefs_out: &mut [f64],
) -> f64 {
    let cards = graph.cardinalities();
    beliefs_out[..graph.beliefs_len()].fill(1.0);
    let mut cursor = 0usize;
    for f in graph.factors() {
        for &v in f.active_vars() {
            let k = cards[v];
            let v_off = graph.var_belief_offset(v);
            for s in 0..k {
                beliefs_out[v_off + s] *= msg_f_to_v[cursor + s];
            }
            cursor += k;
        }
    }
    let mut bethe_var_term = 0.0;
    for (v, &k) in cards.iter().enumerate() {
        let v_off = graph.var_belief_offset(v);
        normalize_message_slice(&mut beliefs_out[v_off..v_off + k]);
        let degree = graph
            .factors()
            .iter()
            .filter(|f| f.active_vars().contains(&v))
            .count();
        let coeff = (degree as f64) - 1.0;
        if coeff != 0.0 {
            for &bv in &beliefs_out[v_off..v_off + k] {
                if bv > 0.0 {
                    bethe_var_term += coeff * bv * libm::log(bv);
                }
            }
        }
    }
    let bethe_factor_term = factor_bethe_energy(graph, msg_v_to_f);
    bethe_factor_term + bethe_var_term
}

fn factor_bethe_energy(graph: &FactorGraphView<'_>, msg_v_to_f: &[f64]) -> f64 {
    let cards = graph.cardinalities();
    let mut factor_energy = 0.0;
    let mut edge_base = 0usize;
    for f in graph.factors() {
        let strides = compute_factor_strides(f, cards);
        let mut slot_offsets = [0usize; MAX_FACTOR_ARITY];
        for pos in 0..f.arity {
            slot_offsets[pos] = edge_base;
            edge_base += cards[f.vars[pos]];
        }
        let table = &graph.potentials()[f.table_offset..f.table_offset + f.table_len];
        let mut z_f = 0.0;
        for (flat_idx, &pot) in table.iter().enumerate() {
            if pot == 0.0 {
                continue;
            }
            let mut w = pot;
            for pos in 0..f.arity {
                let v = f.vars[pos];
                let st = (flat_idx / strides[pos]) % cards[v];
                w *= msg_v_to_f[slot_offsets[pos] + st];
            }
            z_f += w;
        }
        if z_f <= 0.0 {
            continue;
        }
        for (flat_idx, &pot) in table.iter().enumerate() {
            if pot == 0.0 {
                continue;
            }
            let mut w = pot;
            for pos in 0..f.arity {
                let v = f.vars[pos];
                let st = (flat_idx / strides[pos]) % cards[v];
                w *= msg_v_to_f[slot_offsets[pos] + st];
            }
            let bf = w / z_f;
            if bf > 0.0 {
                factor_energy += bf * (libm::log(pot) - libm::log(bf));
            }
        }
    }
    factor_energy
}

/// Runs sum-product belief propagation on `graph` using caller-supplied `beliefs_out` and `scratch`
/// slices with zero heap allocations.
#[cfg_attr(feature = "verified", verified_engine::verified)]
pub fn sum_product_bp_into(
    graph: &FactorGraphView<'_>,
    config: &BpConfig,
    beliefs_out: &mut [f64],
    scratch: &mut [f64],
) -> Result<BpReport, GraphEngineError> {
    if beliefs_out.len() < graph.beliefs_len() {
        return Err(GraphEngineError::BufferTooSmall {
            required: graph.beliefs_len(),
            actual: beliefs_out.len(),
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
    let (msg_f_to_v, rest) = scratch[..req].split_at_mut(edge_len);
    let (msg_v_to_f, next_f_to_v) = rest.split_at_mut(edge_len);
    init_uniform_messages(graph, msg_f_to_v);

    let max_iters = config.max_iters.max(1);
    let mut iterations = 0usize;
    let mut max_residual = f64::INFINITY;
    let mut converged = false;

    for iter in 1..=max_iters {
        update_var_to_factor_messages(graph, msg_f_to_v, msg_v_to_f);
        let mut factor_edge_base = 0usize;
        for f in graph.factors() {
            compute_single_factor_messages(graph, f, factor_edge_base, msg_v_to_f, next_f_to_v);
            for &v in f.active_vars() {
                factor_edge_base += graph.cardinalities()[v];
            }
        }
        max_residual = 0.0;
        for i in 0..edge_len {
            let updated = (1.0 - damping) * next_f_to_v[i] + damping * msg_f_to_v[i];
            let diff = (updated - msg_f_to_v[i]).abs();
            if diff > max_residual {
                max_residual = diff;
            }
            msg_f_to_v[i] = updated;
        }
        iterations = iter;
        if max_residual <= config.tolerance {
            converged = true;
            break;
        }
    }
    update_var_to_factor_messages(graph, msg_f_to_v, msg_v_to_f);
    let log_partition = compute_beliefs_and_bethe(graph, msg_f_to_v, msg_v_to_f, beliefs_out);
    debug_assert!(log_partition.is_finite());
    Ok(BpReport {
        iterations,
        converged,
        max_residual,
        log_partition,
    })
}

/// Convenience `alloc`-enabled wrapper for [`sum_product_bp_into`].
#[cfg(feature = "alloc")]
#[cfg_attr(feature = "verified", verified_engine::verified)]
pub fn sum_product_bp(
    graph: &FactorGraphView<'_>,
    config: &BpConfig,
) -> Result<(BpReport, alloc::vec::Vec<f64>), GraphEngineError> {
    let mut beliefs = alloc::vec![0.0_f64; graph.beliefs_len()];
    let mut scratch = alloc::vec![0.0_f64; graph.bp_scratch_len()];
    let report = sum_product_bp_into(graph, config, &mut beliefs, &mut scratch)?;
    Ok((report, beliefs))
}
