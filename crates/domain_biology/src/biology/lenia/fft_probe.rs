//! Zero-allocation `32x32` stack-buffered radix-2 2D FFT soliton viability probe.

use super::kernel::{
    CompiledGrowthRule, CompiledKernelRule, GrowthParams, KernelShells,
    write_normalized_torus_kernel,
};
use super::presets::{LeniaSpecies, analytic_species_density};
use math_commons::math_kernel::types::flatten_2d_index;
use num_complex::Complex;
use std::f64::consts::FRAC_1_SQRT_2;

/// Side length of the downsampled 2D periodic FFT probe torus (`32`).
pub const PROBE_SIDE: usize = 32;
/// Total cell count of the `32x32` FFT probe (`1024`).
pub const PROBE_CELLS: usize = PROBE_SIDE * PROBE_SIDE;
/// Number of discrete integration steps evaluated per parameter-atlas probe cell.
pub const PROBE_STEPS: usize = 24;
/// Spatial downsampling factor mapping full-grid Lenia radii onto the `32x32` probe torus.
pub const PROBE_DOWNSAMPLE: f64 = 0.5;

const C1: f64 = 0.980_785_280_403_230_4;
const S1: f64 = 0.195_090_322_016_128_25;
const C2: f64 = 0.923_879_532_511_286_7;
const S2: f64 = 0.382_683_432_365_089_8;
const C3: f64 = 0.831_469_612_302_545_2;
const S3: f64 = 0.555_570_233_019_602_2;

/// Precomputed forward radix-2 twiddle factors $\cos(2\pi k/32), -\sin(2\pi k/32)$ for $k \in 0..16$.
const TWIDDLE_32_FWD: [(f64, f64); 16] = [
    (1.0, 0.0),
    (C1, -S1),
    (C2, -S2),
    (C3, -S3),
    (FRAC_1_SQRT_2, -FRAC_1_SQRT_2),
    (S3, -C3),
    (S2, -C2),
    (S1, -C1),
    (0.0, -1.0),
    (-S1, -C1),
    (-S2, -C2),
    (-S3, -C3),
    (-FRAC_1_SQRT_2, -FRAC_1_SQRT_2),
    (-C3, -S3),
    (-C2, -S2),
    (-C1, -S1),
];

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

#[verified_engine::verified]
fn fft_1d_32_in_place(row: &mut [Complex<f64>], inverse: bool) {
    for idx in 0..PROBE_SIDE {
        let rev = ((idx as u8).reverse_bits() >> 3) as usize;
        if idx < rev {
            row.swap(idx, rev);
        }
    }
    let sign = if inverse { -1.0 } else { 1.0 };
    let mut len = 2usize;
    while len <= PROBE_SIDE {
        let half = len / 2;
        let stride = PROBE_SIDE / len;
        let mut base = 0usize;
        while base < PROBE_SIDE {
            for j in 0..half {
                let (tw_re, tw_im) = TWIDDLE_32_FWD[j * stride];
                let w = Complex::new(tw_re, sign * tw_im);
                let u = row[base + j];
                let v = row[base + j + half] * w;
                row[base + j] = u + v;
                row[base + j + half] = u - v;
            }
            base += len;
        }
        len *= 2;
    }
}

#[verified_engine::verified]
fn transpose_square_32(grid: &mut [Complex<f64>; PROBE_CELLS]) {
    for y in 0..PROBE_SIDE {
        for x in (y + 1)..PROBE_SIDE {
            grid.swap(y * PROBE_SIDE + x, x * PROBE_SIDE + y);
        }
    }
}

#[verified_engine::verified]
fn fft_2d_32_in_place(grid: &mut [Complex<f64>; PROBE_CELLS], inverse: bool) {
    for row in grid.as_chunks_mut::<PROBE_SIDE>().0 {
        fft_1d_32_in_place(row, inverse);
    }
    transpose_square_32(grid);
    for col in grid.as_chunks_mut::<PROBE_SIDE>().0 {
        fft_1d_32_in_place(col, inverse);
    }
    transpose_square_32(grid);
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
    fft_2d_forward_32(kernel_hat);
    let inv_area = 1.0 / (PROBE_CELLS as f64);
    for bin in kernel_hat.iter_mut() {
        *bin *= inv_area;
    }
}

#[verified_engine::verified]
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
    let mut freq_buf = [Complex::new(0.0, 0.0); PROBE_CELLS];

    let dt = if spec.dt.is_finite() {
        spec.dt.clamp(0.01, 0.5)
    } else {
        0.1
    };
    let (final_mass, active_count, peak_potential) =
        step_probe_loop(spec, dt, &mut state, kernel_hat, &mut freq_buf);
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

#[verified_engine::verified]
fn step_probe_loop(
    spec: &LeniaProbeSpec<'_>,
    dt: f64,
    state: &mut [f64; PROBE_CELLS],
    kernel_hat: &[Complex<f64>; PROBE_CELLS],
    freq_buf: &mut [Complex<f64>; PROBE_CELLS],
) -> (f64, usize, f64) {
    let mut mass = 0.0;
    let mut active = 0usize;
    let mut peak_u = 0.0_f64;

    for _ in 0..PROBE_STEPS {
        for (dst, &val) in freq_buf.iter_mut().zip(state.iter()) {
            *dst = Complex::new(val, 0.0);
        }
        fft_2d_forward_32(freq_buf);
        for (freq, &k_hat) in freq_buf.iter_mut().zip(kernel_hat.iter()) {
            *freq *= k_hat;
        }
        fft_2d_inverse_32(freq_buf);

        mass = 0.0;
        active = 0;
        peak_u = 0.0;
        for (cell, freq) in state.iter_mut().zip(freq_buf.iter()) {
            let u = freq.re;
            if u > peak_u {
                peak_u = u;
            }
            let next_a = (*cell + dt * eval_probe_growth(spec, u)).clamp(0.0, 1.0);
            *cell = next_a;
            mass += next_a;
            if next_a > 0.01 {
                active += 1;
            }
        }
        if mass <= 1e-6 {
            break;
        }
    }
    (mass, active, peak_u)
}

#[verified_engine::verified]
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
