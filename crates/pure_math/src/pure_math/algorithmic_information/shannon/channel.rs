//! Closed-form channel capacities (`BSC`, `BEC`, `AWGN`) and Blahut-Arimoto DMC capacity solver.

use super::ShannonError;
use math_commons::math_kernel::types::flatten_2d_index;
use math_commons::registry::{TOLERANCE_FAST, TOLERANCE_HIGH};

/// Result summary returned by [`blahut_arimoto_capacity`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChannelCapacityResult {
    /// Estimated channel capacity in bits per channel use (`0.5 * (lower + upper)` at termination).
    pub capacity_bits: f64,
    /// Achievable lower bound $I(r^{(t)}; W)$ in bits per channel use.
    pub lower_bound_bits: f64,
    /// Minimax dual upper bound $\max_x D_{\mathrm{KL}}(W(\cdot \mid x) \parallel q^{(t)})$ in bits.
    pub upper_bound_bits: f64,
    /// Number of Blahut-Arimoto iterations executed.
    pub iterations: usize,
    /// Whether `upper_bound_bits - lower_bound_bits <= tolerance`.
    pub converged: bool,
}

/// Preallocated scratch buffers for zero-allocation [`blahut_arimoto_capacity`] execution.
#[derive(Clone, Debug)]
pub struct BlahutArimotoWorkspace {
    output_dist: Vec<f64>,
    input_kl: Vec<f64>,
}

impl BlahutArimotoWorkspace {
    /// Allocates scratch space supporting up to `max_inputs` input symbols and `max_outputs` output symbols.
    #[must_use]
    pub fn new(max_inputs: usize, max_outputs: usize) -> Self {
        Self {
            output_dist: vec![0.0; max_outputs],
            input_kl: vec![0.0; max_inputs],
        }
    }

    /// Resizes internal scratch vectors if needed outside verified zero-allocation regions.
    pub fn ensure_capacity(&mut self, n_inputs: usize, n_outputs: usize) {
        if self.output_dist.len() < n_outputs {
            self.output_dist.resize(n_outputs, 0.0);
        }
        if self.input_kl.len() < n_inputs {
            self.input_kl.resize(n_inputs, 0.0);
        }
    }
}

/// Binary Shannon entropy $h_2(p) = -p \log_2 p - (1-p)\log_2(1-p)$ in bits for $p \in [0, 1]$.
#[verified_engine::verified]
pub fn binary_entropy(p: f64) -> Result<f64, ShannonError> {
    if !p.is_finite() || !(0.0..=1.0).contains(&p) {
        return Err(ShannonError::InvalidChannelParameter { value: p });
    }
    if p <= 0.0 || p >= 1.0 {
        return Ok(0.0);
    }
    let q = 1.0 - p;
    Ok((-p * p.log2() - q * q.log2()).clamp(0.0, 1.0))
}

/// Closed-form Binary Symmetric Channel capacity $C_{\mathrm{BSC}}(p) = 1 - h_2(p)$ in bits/use.
#[verified_engine::verified]
pub fn bsc_capacity(crossover_prob: f64) -> Result<f64, ShannonError> {
    let h = binary_entropy(crossover_prob)?;
    Ok((1.0 - h).clamp(0.0, 1.0))
}

/// Closed-form Binary Erasure Channel capacity $C_{\mathrm{BEC}}(\epsilon) = 1 - \epsilon$ in bits/use.
#[verified_engine::verified]
pub fn bec_capacity(erasure_prob: f64) -> Result<f64, ShannonError> {
    if !erasure_prob.is_finite() || !(0.0..=1.0).contains(&erasure_prob) {
        return Err(ShannonError::InvalidChannelParameter {
            value: erasure_prob,
        });
    }
    Ok((1.0 - erasure_prob).clamp(0.0, 1.0))
}

/// Discrete-time real AWGN channel capacity $C = \frac{1}{2}\log_2(1 + \mathrm{SNR})$ in bits per real channel use.
#[verified_engine::verified]
pub fn awgn_capacity(snr_linear: f64) -> Result<f64, ShannonError> {
    if !snr_linear.is_finite() || snr_linear < 0.0 {
        return Err(ShannonError::InvalidChannelParameter { value: snr_linear });
    }
    Ok(0.5 * (1.0 + snr_linear).log2())
}

/// Continuous-time bandlimited AWGN capacity $C = B \log_2(1 + \mathrm{SNR})$ in bits per second.
#[verified_engine::verified]
pub fn awgn_bandwidth_capacity(bandwidth_hz: f64, snr_linear: f64) -> Result<f64, ShannonError> {
    if !bandwidth_hz.is_finite() || bandwidth_hz <= 0.0 {
        return Err(ShannonError::InvalidChannelParameter {
            value: bandwidth_hz,
        });
    }
    if !snr_linear.is_finite() || snr_linear < 0.0 {
        return Err(ShannonError::InvalidChannelParameter { value: snr_linear });
    }
    Ok(bandwidth_hz * (1.0 + snr_linear).log2())
}

#[verified_engine::verified]
fn validate_dmc_matrix(
    transition: &[f64],
    n_inputs: usize,
    n_outputs: usize,
) -> Result<(), ShannonError> {
    if n_inputs == 0 || n_outputs == 0 {
        return Err(ShannonError::EmptyDistribution);
    }
    let expected = n_inputs * n_outputs;
    if transition.len() != expected {
        return Err(ShannonError::DimensionMismatch {
            expected,
            actual: transition.len(),
        });
    }
    for x in 0..n_inputs {
        let mut row_sum = 0.0;
        for y in 0..n_outputs {
            let w = transition[flatten_2d_index(y, x, n_outputs)];
            if !w.is_finite() || w < -TOLERANCE_HIGH || w > 1.0 + TOLERANCE_FAST {
                return Err(ShannonError::InvalidChannelMatrix {
                    row: x,
                    sum: f64::NAN,
                });
            }
            row_sum += w.max(0.0);
        }
        if (row_sum - 1.0).abs() > TOLERANCE_FAST {
            return Err(ShannonError::InvalidChannelMatrix {
                row: x,
                sum: row_sum,
            });
        }
    }
    Ok(())
}

#[verified_engine::verified]
fn blahut_arimoto_step(
    transition: &[f64],
    n_inputs: usize,
    n_outputs: usize,
    r: &mut [f64],
    q: &mut [f64],
    d: &mut [f64],
) -> (f64, f64) {
    q.fill(0.0);
    for (x, &rx) in r.iter().enumerate() {
        if rx <= 0.0 {
            continue;
        }
        for (y, qy) in q.iter_mut().enumerate() {
            let w = transition[flatten_2d_index(y, x, n_outputs)].max(0.0);
            *qy += rx * w;
        }
    }

    let mut lower = 0.0;
    let mut upper = 0.0_f64;
    for x in 0..n_inputs {
        let mut dx = 0.0;
        for (y, &qy) in q.iter().enumerate() {
            let w = transition[flatten_2d_index(y, x, n_outputs)].max(0.0);
            if w > 0.0 {
                dx += w * (w / qy.max(f64::MIN_POSITIVE)).log2();
            }
        }
        let dx_clamped = dx.max(0.0);
        d[x] = dx_clamped;
        lower += r[x] * dx_clamped;
        if x == 0 || dx_clamped > upper {
            upper = dx_clamped;
        }
    }

    let mut norm = 0.0;
    for x in 0..n_inputs {
        let weight = r[x] * (d[x] - upper).exp2();
        r[x] = weight;
        norm += weight;
    }
    if norm > 0.0 {
        let inv_norm = 1.0 / norm;
        for rx in r.iter_mut() {
            *rx *= inv_norm;
        }
    }
    (lower.max(0.0), upper.max(lower.max(0.0)))
}

/// Computes the capacity and capacity-achieving input distribution of a discrete memoryless channel
/// via the Blahut-Arimoto alternating minimization algorithm with zero heap allocations.
#[verified_engine::verified]
pub fn blahut_arimoto_capacity(
    transition: &[f64],
    n_inputs: usize,
    n_outputs: usize,
    tolerance: f64,
    max_iterations: usize,
    workspace: &mut BlahutArimotoWorkspace,
    optimal_input_out: &mut [f64],
) -> Result<ChannelCapacityResult, ShannonError> {
    validate_dmc_matrix(transition, n_inputs, n_outputs)?;
    if !tolerance.is_finite() || tolerance <= 0.0 || max_iterations == 0 {
        return Err(ShannonError::InvalidChannelParameter { value: tolerance });
    }
    if optimal_input_out.len() < n_inputs {
        return Err(ShannonError::WorkspaceTooSmall {
            required: n_inputs,
            actual: optimal_input_out.len(),
        });
    }
    if workspace.output_dist.len() < n_outputs {
        return Err(ShannonError::WorkspaceTooSmall {
            required: n_outputs,
            actual: workspace.output_dist.len(),
        });
    }
    if workspace.input_kl.len() < n_inputs {
        return Err(ShannonError::WorkspaceTooSmall {
            required: n_inputs,
            actual: workspace.input_kl.len(),
        });
    }

    let r = &mut optimal_input_out[..n_inputs];
    let q = &mut workspace.output_dist[..n_outputs];
    let d = &mut workspace.input_kl[..n_inputs];
    r.fill(1.0 / (n_inputs as f64));

    let mut lower = 0.0;
    let mut upper = 0.0;
    for iter in 1..=max_iterations {
        let (step_lower, step_upper) =
            blahut_arimoto_step(transition, n_inputs, n_outputs, r, q, d);
        lower = step_lower;
        upper = step_upper;
        if upper - lower <= tolerance {
            return Ok(ChannelCapacityResult {
                capacity_bits: 0.5 * (lower + upper),
                lower_bound_bits: lower,
                upper_bound_bits: upper,
                iterations: iter,
                converged: true,
            });
        }
    }

    Err(ShannonError::ConvergenceFailed {
        iterations: max_iterations,
        gap: upper - lower,
    })
}

// [cite:algorithmic_information_rust]
