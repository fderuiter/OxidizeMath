//! Zero-allocation `32x32` stack-buffered radix-2 2D FFT soliton viability probe.

use super::fft_radix2::{HERMITIAN_CELLS, convolve_real_hermitian_32, fft_2d_32_in_place};
use super::kernel::{
    CompiledGrowthRule, CompiledKernelRule, GrowthParams, KernelShells,
    write_normalized_torus_kernel,
};
use super::presets::{LeniaSpecies, analytic_species_density};
use math_commons::math_kernel::types::flatten_2d_index;
use num_complex::Complex;

/// Side length of the downsampled 2D periodic FFT probe torus (`32`).
pub const PROBE_SIDE: usize = 32;
/// Total cell count of the `32x32` FFT probe (`1024`).
pub const PROBE_CELLS: usize = PROBE_SIDE * PROBE_SIDE;
/// Number of discrete integration steps evaluated per parameter-atlas probe cell.
pub const PROBE_STEPS: usize = 24;
/// Spatial downsampling factor mapping full-grid Lenia radii onto the `32x32` probe torus.
pub const PROBE_DOWNSAMPLE: f64 = 0.5;

/// Qualitative dynamical outcome of a 24-step `32x32` Lenia soliton probe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LeniaProbeOutcome {
    /// Initial soliton decays toward the homogeneous zero vacuum ($A \to 0$).
    Extinction,
    /// Bounded, self-sustaining localized soliton or dividing multi-lobe lifeform.
    LocalizedSoliton,
    /// Soliton expands uncontrollably into a domain-filling active blob while $G(0) \le 0$.
    OvergrownBlob,
    /// Unstable zero vacuum ($G(0) > 0$) causing spontaneous global activation explosion.
    VacuumExplosion,
}

/// Quantitative summary emitted by [`evaluate_lenia_probe_32`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LeniaProbeSummary {
    /// Classified qualitative viability regime.
    pub outcome: LeniaProbeOutcome,
    /// Final integrated activation mass after `PROBE_STEPS`.
    pub final_mass: f64,
    /// Initial seeded activation mass at $t = 0$.
    pub initial_mass: f64,
    /// Fraction of `32x32` cells exceeding `0.01` activation at completion.
    pub active_fraction: f64,
    /// Peak neighborhood potential $\max(U)$ observed on the final probe step.
    pub peak_potential: f64,
}

/// Stack-friendly specification for a single `32x32` parameter-atlas probe evaluation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LeniaProbeSpec<'a> {
    /// Seeded canonical species template.
    pub species: LeniaSpecies,
    /// Reference organism radius in full-grid coordinates (controls downsampled seed size).
    pub seed_radius: f64,
    /// Swept interaction kernel radius in full-grid coordinates.
    pub kernel_radius: f64,
    /// Concentric ring weights used when `custom_kernel` is `None`.
    pub ring_weights: &'a [f64],
    /// Gaussian growth parameters used when `custom_growth` is `None`.
    pub growth: GrowthParams,
    /// Integration step size $\Delta t$.
    pub dt: f64,
    /// Optional custom compiled growth formula and active parameter slice.
    pub custom_growth: Option<(&'a CompiledGrowthRule, &'a [f64])>,
    /// Optional custom compiled radial kernel formula and active parameter slice.
    pub custom_kernel: Option<(&'a CompiledKernelRule, &'a [f64])>,
}

/// Executes an unscaled in-place 2D radix-2 forward FFT on a `32x32` stack buffer.
#[verified_engine::verified]
pub fn fft_2d_forward_32(grid: &mut [Complex<f64>; PROBE_CELLS]) {
    fft_2d_32_in_place(grid, false);
}

/// Executes an unscaled in-place 2D radix-2 inverse FFT on a `32x32` stack buffer.
#[verified_engine::verified]
pub fn fft_2d_inverse_32(grid: &mut [Complex<f64>; PROBE_CELLS]) {
    fft_2d_32_in_place(grid, true);
}

/// Seeds a downsampled [`LeniaSpecies`] soliton centered on a `32x32` stack array, returning initial mass.
#[verified_engine::verified]
pub fn seed_probe_species_32(
    species: LeniaSpecies,
    full_seed_radius: f64,
    out: &mut [f64; PROBE_CELLS],
) -> f64 {
    *out = [0.0; PROBE_CELLS];
    let r_probe = (full_seed_radius * PROBE_DOWNSAMPLE).clamp(2.5, 10.5);
    let cx = (PROBE_SIDE / 2) as isize;
    let cy = (PROBE_SIDE / 2) as isize;
    let span = ((1.45 * r_probe).ceil() as isize).clamp(4, 15);
    let mut mass = 0.0;
    for dy in -span..=span {
        let y = (cy + dy).rem_euclid(PROBE_SIDE as isize) as usize;
        for dx in -span..=span {
            let x = (cx + dx).rem_euclid(PROBE_SIDE as isize) as usize;
            let val =
                analytic_species_density(species, (dx as f64) / r_probe, (dy as f64) / r_probe);
            if val > 1e-4 {
                let idx = flatten_2d_index(x, y, PROBE_SIDE);
                let clamped = val.clamp(0.0, 1.0);
                out[idx] = clamped;
                mass += clamped;
            }
        }
    }
    mass
}

/// Precomputes the normalized spectrum $\hat{K} / 1024$ on a `32x32` stack buffer.
#[verified_engine::verified]
pub fn precompute_probe_kernel_hat_32(
    spec: &LeniaProbeSpec<'_>,
    kernel_hat: &mut [Complex<f64>; PROBE_CELLS],
) {
    let r_probe = (spec.kernel_radius * PROBE_DOWNSAMPLE).clamp(2.0, 15.0);
    if let Some((rule, params)) = spec.custom_kernel {
        rule.write_wrapped_spatial_kernel_with_params(
            PROBE_SIDE, PROBE_SIDE, r_probe, params, kernel_hat,
        );
    } else {
        let shells = KernelShells::new(spec.kernel_radius, spec.ring_weights);
        write_normalized_torus_kernel(PROBE_SIDE, PROBE_SIDE, r_probe, kernel_hat, |r| {
            shells.shell_unnormalized(r)
        });
    }
    fft_2d_32_in_place(kernel_hat, false);
    let inv_area = 1.0 / (PROBE_CELLS as f64);
    for bin in kernel_hat.iter_mut() {
        *bin *= inv_area;
    }
}

#[inline]
fn eval_probe_growth(spec: &LeniaProbeSpec<'_>, u: f64) -> f64 {
    if let Some((rule, params)) = spec.custom_growth {
        rule.evaluate_with_params(u, params)
    } else {
        spec.growth.evaluate(u)
    }
}

/// Evaluates a 24-step `32x32` 2D FFT soliton viability probe from a pre-seeded state and optional
/// precomputed $\hat{K}$ spectrum using exclusively stack-allocated `1024`-element buffers.
#[verified_engine::verified]
pub fn evaluate_lenia_probe_32_preseeded(
    spec: &LeniaProbeSpec<'_>,
    initial_state: &[f64; PROBE_CELLS],
    initial_mass: f64,
    precomputed_kernel_hat: Option<&[Complex<f64>; PROBE_CELLS]>,
) -> LeniaProbeSummary {
    if eval_probe_growth(spec, 0.0) > 0.0 {
        return LeniaProbeSummary {
            outcome: LeniaProbeOutcome::VacuumExplosion,
            final_mass: PROBE_CELLS as f64,
            initial_mass: 0.0,
            active_fraction: 1.0,
            peak_potential: 1.0,
        };
    }

    let mut state = *initial_state;
    let mut scratch_k_hat = [Complex::new(0.0, 0.0); PROBE_CELLS];
    let kernel_hat = if let Some(cached) = precomputed_kernel_hat {
        cached
    } else {
        precompute_probe_kernel_hat_32(spec, &mut scratch_k_hat);
        &scratch_k_hat
    };

    let dt = if spec.dt.is_finite() {
        spec.dt.clamp(0.01, 0.5)
    } else {
        0.1
    };
    let (final_mass, active_count, peak_potential) =
        step_probe_loop(spec, dt, &mut state, kernel_hat);
    let active_fraction = (active_count as f64) / (PROBE_CELLS as f64);
    let outcome = classify_probe_outcome(initial_mass, final_mass, active_fraction);

    LeniaProbeSummary {
        outcome,
        final_mass,
        initial_mass,
        active_fraction,
        peak_potential,
    }
}

/// Evaluates a 24-step `32x32` 2D FFT soliton viability probe using exclusively stack-allocated
/// `1024`-element buffers (zero heap allocations).
#[verified_engine::verified]
pub fn evaluate_lenia_probe_32(spec: &LeniaProbeSpec<'_>) -> LeniaProbeSummary {
    if eval_probe_growth(spec, 0.0) > 0.0 {
        return LeniaProbeSummary {
            outcome: LeniaProbeOutcome::VacuumExplosion,
            final_mass: PROBE_CELLS as f64,
            initial_mass: 0.0,
            active_fraction: 1.0,
            peak_potential: 1.0,
        };
    }
    let mut state = [0.0_f64; PROBE_CELLS];
    let initial_mass = seed_probe_species_32(spec.species, spec.seed_radius, &mut state);
    evaluate_lenia_probe_32_preseeded(spec, &state, initial_mass, None)
}

#[inline]
fn advance_probe_cells(
    spec: &LeniaProbeSpec<'_>,
    dt: f64,
    mu: f64,
    inv_neg_two_sigma_sq: f64,
    state: &mut [f64; PROBE_CELLS],
    u_buf: &[f64; PROBE_CELLS],
) -> (f64, usize, f64, f64) {
    let mut mass = 0.0;
    let mut active = 0usize;
    let mut peak_u = 0.0_f64;
    let mut max_a = 0.0_f64;
    if let Some((rule, params)) = spec.custom_growth {
        let mut idx = 0usize;
        while idx < PROBE_CELLS {
            let u = u_buf[idx];
            if u > peak_u {
                peak_u = u;
            }
            let next_a = (state[idx] + dt * rule.evaluate_with_params(u, params)).clamp(0.0, 1.0);
            state[idx] = next_a;
            mass += next_a;
            if next_a > max_a {
                max_a = next_a;
            }
            if next_a > 0.01 {
                active += 1;
            }
            idx += 1;
        }
    } else {
        let mut idx = 0usize;
        while idx < PROBE_CELLS {
            let u = u_buf[idx];
            if u > peak_u {
                peak_u = u;
            }
            let d = u - mu;
            let g = 2.0 * ((d * d) * inv_neg_two_sigma_sq).exp() - 1.0;
            let next_a = (state[idx] + dt * g).clamp(0.0, 1.0);
            state[idx] = next_a;
            mass += next_a;
            if next_a > max_a {
                max_a = next_a;
            }
            if next_a > 0.01 {
                active += 1;
            }
            idx += 1;
        }
    }
    (mass, active, peak_u, max_a)
}

#[inline]
fn step_probe_loop(
    spec: &LeniaProbeSpec<'_>,
    dt: f64,
    state: &mut [f64; PROBE_CELLS],
    kernel_hat: &[Complex<f64>; PROBE_CELLS],
) -> (f64, usize, f64) {
    let mut tilde_cols = [Complex { re: 0.0, im: 0.0 }; HERMITIAN_CELLS];
    let mut u_buf = [0.0_f64; PROBE_CELLS];
    let mut stats = (0.0, 0usize, 0.0_f64);
    let mu = spec.growth.mu;
    let inv_neg_two_sigma_sq = -0.5 / (spec.growth.sigma * spec.growth.sigma);
    let monotone_eligible = spec.custom_growth.is_none()
        && spec.custom_kernel.is_none()
        && spec.ring_weights.iter().all(|&w| w >= 0.0);

    let mut step_idx = 0usize;
    while step_idx < PROBE_STEPS {
        convolve_real_hermitian_32(state, kernel_hat, &mut tilde_cols, &mut u_buf);
        let (mass, active, peak_u, max_a) =
            advance_probe_cells(spec, dt, mu, inv_neg_two_sigma_sq, state, &u_buf);
        stats = (mass, active, peak_u);
        if mass <= 0.02 {
            break;
        }
        if monotone_eligible && peak_u < mu {
            let d = peak_u - mu;
            let g_bound = 2.0 * ((d * d) * inv_neg_two_sigma_sq).exp() - 1.0;
            let rem = (PROBE_STEPS - 1 - step_idx) as f64;
            if g_bound < 0.0 && max_a + rem * dt * g_bound <= 0.0 {
                stats = (0.0, 0, 0.0);
                break;
            }
        }
        step_idx += 1;
    }
    stats
}

#[inline]
fn classify_probe_outcome(
    initial_mass: f64,
    final_mass: f64,
    active_fraction: f64,
) -> LeniaProbeOutcome {
    if final_mass < (0.20 * initial_mass).max(0.5) {
        LeniaProbeOutcome::Extinction
    } else if active_fraction > 0.42 || final_mass > (3.2 * initial_mass).max(24.0) {
        LeniaProbeOutcome::OvergrownBlob
    } else {
        LeniaProbeOutcome::LocalizedSoliton
    }
}
