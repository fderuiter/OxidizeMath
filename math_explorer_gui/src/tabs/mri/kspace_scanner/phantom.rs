//! Cross-domain `64x64` MRI phantom synthesis weighted by [`BlochSimulator`] spin-echo dynamics.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::state::{MriKSpaceScannerState, PhantomSource};
use super::{KSPACE_DIM, KSPACE_LEN};
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::physics::mri::bloch::BlochSimulator;
use math_explorer::pure_math::analysis::ode::TimeStepper;
use nalgebra::Vector3;

#[derive(Clone, Copy)]
struct EllipseSpec {
    cx: f64,
    cy: f64,
    a: f64,
    b: f64,
    theta_deg: f64,
    compartment: TissueCompartment,
}

#[derive(Clone, Copy)]
enum TissueCompartment {
    SkullScalp,
    BrainBackground,
    VentricleCsf,
    GrayMatterLesion,
}

const SHEPP_LOGAN_ELLIPSES: [EllipseSpec; 10] = [
    EllipseSpec {
        cx: 0.0,
        cy: 0.0,
        a: 0.69,
        b: 0.92,
        theta_deg: 0.0,
        compartment: TissueCompartment::SkullScalp,
    },
    EllipseSpec {
        cx: 0.0,
        cy: -0.0184,
        a: 0.6624,
        b: 0.874,
        theta_deg: 0.0,
        compartment: TissueCompartment::BrainBackground,
    },
    EllipseSpec {
        cx: 0.22,
        cy: 0.0,
        a: 0.11,
        b: 0.31,
        theta_deg: -18.0,
        compartment: TissueCompartment::VentricleCsf,
    },
    EllipseSpec {
        cx: -0.22,
        cy: 0.0,
        a: 0.16,
        b: 0.41,
        theta_deg: 18.0,
        compartment: TissueCompartment::VentricleCsf,
    },
    EllipseSpec {
        cx: 0.0,
        cy: 0.35,
        a: 0.21,
        b: 0.25,
        theta_deg: 0.0,
        compartment: TissueCompartment::GrayMatterLesion,
    },
    EllipseSpec {
        cx: 0.0,
        cy: 0.1,
        a: 0.046,
        b: 0.046,
        theta_deg: 0.0,
        compartment: TissueCompartment::GrayMatterLesion,
    },
    EllipseSpec {
        cx: 0.0,
        cy: -0.1,
        a: 0.046,
        b: 0.046,
        theta_deg: 0.0,
        compartment: TissueCompartment::GrayMatterLesion,
    },
    EllipseSpec {
        cx: -0.08,
        cy: -0.605,
        a: 0.046,
        b: 0.023,
        theta_deg: 0.0,
        compartment: TissueCompartment::GrayMatterLesion,
    },
    EllipseSpec {
        cx: 0.0,
        cy: -0.606,
        a: 0.023,
        b: 0.023,
        theta_deg: 0.0,
        compartment: TissueCompartment::GrayMatterLesion,
    },
    EllipseSpec {
        cx: 0.06,
        cy: -0.605,
        a: 0.023,
        b: 0.046,
        theta_deg: 0.0,
        compartment: TissueCompartment::GrayMatterLesion,
    },
];

/// Computes transverse spin-echo signal amplitude for a compartment using [`BlochSimulator`].
///
/// Integrates longitudinal $T_1$ recovery from saturation ($M_z = 0 \to M_0 = \rho$) over `tr_s`,
/// tips recovered longitudinal magnetization by $90^\circ$ into the transverse plane, and
/// integrates transverse $T_2$ decay over `te_s`.
#[must_use]
pub fn spin_echo_weight_from_bloch(rho: f64, t1: f64, t2: f64, tr_s: f64, te_s: f64) -> f64 {
    let safe_t1 = t1.max(0.01);
    let safe_t2 = t2.max(0.002);
    let substeps = 16_usize;

    let mut longitudinal = BlochSimulator::new(Vector3::zeros(), rho);
    longitudinal.set_b_field(Vector3::zeros());
    longitudinal.set_relaxation(safe_t1, f64::INFINITY);
    let dt_tr = tr_s.max(1e-4) / (substeps as f64);
    for _ in 0..substeps {
        <BlochSimulator as TimeStepper<Vector3<f64>>>::step(&mut longitudinal, dt_tr);
    }
    let mz_recovered = longitudinal.magnetization.z.clamp(0.0, rho);

    let mut transverse = BlochSimulator::new(Vector3::new(0.0, mz_recovered, 0.0), rho);
    transverse.set_b_field(Vector3::zeros());
    transverse.set_relaxation(f64::INFINITY, safe_t2);
    let dt_te = te_s.max(1e-4) / (substeps as f64);
    for _ in 0..substeps {
        <BlochSimulator as TimeStepper<Vector3<f64>>>::step(&mut transverse, dt_te);
    }
    transverse.magnetization.y.clamp(0.0, rho)
}

#[derive(Clone, Copy)]
struct CompartmentWeights {
    skull: f64,
    white_matter: f64,
    csf: f64,
    gray_matter: f64,
}

fn compute_compartment_weights(state: &MriKSpaceScannerState) -> CompartmentWeights {
    let s1 = state.t1_scale;
    let s2 = state.t2_scale;
    CompartmentWeights {
        skull: spin_echo_weight_from_bloch(0.95, 0.35 * s1, 0.045 * s2, state.tr_s, state.te_s),
        white_matter: spin_echo_weight_from_bloch(
            0.78,
            0.65 * s1,
            0.08 * s2,
            state.tr_s,
            state.te_s,
        ),
        csf: spin_echo_weight_from_bloch(1.0, 2.4 * s1, 0.32 * s2, state.tr_s, state.te_s),
        gray_matter: spin_echo_weight_from_bloch(
            0.88,
            0.95 * s1,
            0.11 * s2,
            state.tr_s,
            state.te_s,
        ),
    }
}

fn point_in_ellipse(x: f64, y: f64, e: &EllipseSpec) -> bool {
    let rad = e.theta_deg.to_radians();
    let (sin_t, cos_t) = rad.sin_cos();
    let dx = x - e.cx;
    let dy = y - e.cy;
    let xr = dx * cos_t + dy * sin_t;
    let yr = -dx * sin_t + dy * cos_t;
    (xr * xr) / (e.a * e.a) + (yr * yr) / (e.b * e.b) <= 1.0
}

/// Synthesizes a `64x64` spin-echo weighted Shepp-Logan phantom normalized to `[0.0, 1.0]`.
#[must_use]
pub fn synthesize_shepp_logan(state: &MriKSpaceScannerState) -> Vec<f64> {
    let weights = compute_compartment_weights(state);
    let mut grid = vec![0.0_f64; KSPACE_LEN];
    let mut peak = 0.0_f64;

    for row in 0..KSPACE_DIM {
        let y = 1.0 - 2.0 * ((row as f64 + 0.5) / KSPACE_DIM as f64);
        for col in 0..KSPACE_DIM {
            let x = 2.0 * ((col as f64 + 0.5) / KSPACE_DIM as f64) - 1.0;
            let mut val = 0.0_f64;
            for ellipse in &SHEPP_LOGAN_ELLIPSES {
                if point_in_ellipse(x, y, ellipse) {
                    val = match ellipse.compartment {
                        TissueCompartment::SkullScalp => weights.skull,
                        TissueCompartment::BrainBackground => weights.white_matter,
                        TissueCompartment::VentricleCsf => weights.csf,
                        TissueCompartment::GrayMatterLesion => weights.gray_matter,
                    };
                }
            }
            peak = peak.max(val);
            grid[flatten_2d_index(col, row, KSPACE_DIM)] = val;
        }
    }

    if peak > 1e-9 {
        for v in &mut grid {
            *v = (*v / peak).clamp(0.0, 1.0);
        }
    }
    grid
}

/// Returns whether the cross-domain `biology` Turing reaction-diffusion kernel is compiled in.
#[must_use]
pub const fn is_biology_phantom_available() -> bool {
    cfg!(feature = "biology")
}

#[cfg(feature = "biology")]
fn compute_turing_activator(seed: u64, steps: usize) -> Vec<f64> {
    use math_explorer::biology::diffusion::FiniteDifference2D;
    use math_explorer::biology::morphogenesis::{DiffusionCoeff, SchnakenbergKinetics, TuringSystem};
    use math_explorer::math_kernel::types::{Dimension, StepSize};
    use oxidize_core::rng::OxidizeRng;
    use rand::Rng;

    let a = 0.1_f64;
    let b = 0.9_f64;
    let diffusion = FiniteDifference2D::new(
        Dimension(KSPACE_DIM),
        Dimension(KSPACE_DIM),
        StepSize(1.0),
        StepSize(1.0),
    );
    let mut system = TuringSystem::new_with_kinetics(
        Dimension(KSPACE_LEN),
        DiffusionCoeff(1.0),
        DiffusionCoeff(40.0),
        SchnakenbergKinetics { a, b },
        diffusion,
    );
    let u_eq = a + b;
    let v_eq = b / (u_eq * u_eq);
    let mut rng = OxidizeRng::new(seed);
    for idx in 0..KSPACE_LEN {
        system.u_mut()[idx] = u_eq + rng.gen_range(-0.12_f64..0.12_f64);
        system.v_mut()[idx] = v_eq + rng.gen_range(-0.12_f64..0.12_f64);
    }
    for _ in 0..steps {
        system.step(0.04);
    }
    system.u().to_vec()
}

#[cfg(feature = "biology")]
fn map_activator_to_spin_echo(u_field: &[f64], state: &MriKSpaceScannerState) -> Vec<f64> {
    let weights = compute_compartment_weights(state);
    let (mut min_u, mut max_u) = (f64::INFINITY, f64::NEG_INFINITY);
    for &u in u_field {
        if u.is_finite() {
            min_u = min_u.min(u);
            max_u = max_u.max(u);
        }
    }
    let span = (max_u - min_u).max(1e-6);
    let mut out = vec![0.0_f64; KSPACE_LEN];
    let mut peak = 0.0_f64;

    for row in 0..KSPACE_DIM {
        let y = 2.0 * ((row as f64 + 0.5) / KSPACE_DIM as f64) - 1.0;
        for col in 0..KSPACE_DIM {
            let x = 2.0 * ((col as f64 + 0.5) / KSPACE_DIM as f64) - 1.0;
            let idx = flatten_2d_index(col, row, KSPACE_DIM);
            if x * x + y * y <= 0.88 * 0.88 {
                let u_val = u_field[idx];
                let norm_u = if u_val.is_finite() {
                    ((u_val - min_u) / span).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let signal = (1.0 - norm_u) * weights.white_matter + norm_u * weights.csf;
                peak = peak.max(signal);
                out[idx] = signal;
            }
        }
    }
    if peak > 1e-9 {
        for v in &mut out {
            *v = (*v / peak).clamp(0.0, 1.0);
        }
    }
    out
}

/// Synthesizes the active `64x64` phantom (`SheppLogan` or `TuringMorphogenesis` with fallback).
#[must_use]
pub fn synthesize_phantom(state: &MriKSpaceScannerState) -> Vec<f64> {
    match state.phantom {
        PhantomSource::SheppLogan => synthesize_shepp_logan(state),
        PhantomSource::TuringMorphogenesis => {
            #[cfg(feature = "biology")]
            {
                let u_field = compute_turing_activator(state.rng_seed, state.turing_steps);
                map_activator_to_spin_echo(&u_field, state)
            }
            #[cfg(not(feature = "biology"))]
            {
                synthesize_shepp_logan(state)
            }
        }
    }
}
