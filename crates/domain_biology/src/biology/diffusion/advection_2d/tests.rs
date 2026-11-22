use super::*;
use crate::biology::morphogenesis::{DiffusionCoeff, ReactionKinetics, TuringSystem};

struct ZeroKinetics;

impl ReactionKinetics<2> for ZeroKinetics {
    #[verified_engine::verified]
    fn reaction(&self, _concentrations: [f64; 2]) -> [f64; 2] {
        [0.0, 0.0]
    }
}

fn center_of_mass(field: &[f64], width: usize, height: usize) -> (f64, f64, f64) {
    let mut sum_m = 0.0;
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    for y in 0..height {
        for x in 0..width {
            let m = field[flatten_2d_index(x, y, width)];
            sum_m += m;
            sum_x += m * (x as f64);
            sum_y += m * (y as f64);
        }
    }
    (sum_x / sum_m, sum_y / sum_m, sum_m)
}

#[test]
fn test_gaussian_pulse_drift_and_mass_conservation() {
    let width = 32;
    let height = 32;
    let mut adv = AdvectionDiffusion2D::new(
        Dimension(width),
        Dimension(height),
        StepSize(1.0),
        StepSize(1.0),
    );
    adv.set_uniform_velocity(0.6, -0.3);

    let mut sys = TuringSystem::new_with_kinetics(
        Dimension(width * height),
        DiffusionCoeff(0.05),
        DiffusionCoeff(0.05),
        ZeroKinetics,
        adv,
    );

    for y in 0..height {
        for x in 0..width {
            let dx = (x as f64) - 12.0;
            let dy = (y as f64) - 18.0;
            let val = (-(dx * dx + dy * dy) / 8.0).exp();
            let idx = flatten_2d_index(x, y, width);
            sys.u_mut()[idx] = val;
            sys.v_mut()[idx] = 0.5 * val;
        }
    }

    let (cx0, cy0, mass0) = center_of_mass(sys.u(), width, height);

    verified_engine::engine::VerifiedEngine::run_verified(|| {
        for _ in 0..40 {
            sys.step(0.1);
        }
    });

    let (cx1, cy1, mass1) = center_of_mass(sys.u(), width, height);

    assert!(
        cx1 - cx0 > 1.8,
        "Gaussian pulse should drift in +x direction: cx0={cx0:.3}, cx1={cx1:.3}"
    );
    assert!(
        cy1 - cy0 < -0.9,
        "Gaussian pulse should drift in -y direction: cy0={cy0:.3}, cy1={cy1:.3}"
    );
    assert!(
        (mass1 - mass0).abs() < 1e-11,
        "Conservative zero-flux upwind must conserve fluid mass: m0={mass0}, m1={mass1}"
    );
}

#[test]
fn test_zero_flux_across_internal_obstacle_barrier() {
    let width = 20;
    let height = 12;
    let mut adv = AdvectionDiffusion2D::new(
        Dimension(width),
        Dimension(height),
        StepSize(1.0),
        StepSize(1.0),
    );
    // Solid vertical wall at x = 10 dividing left and right chambers
    for y in 0..height {
        adv.set_obstacle(10, y, true);
    }
    // Strong drift pushing toward the obstacle wall (+x)
    adv.set_uniform_velocity(0.8, 0.1);

    let mut sys = TuringSystem::new_with_kinetics(
        Dimension(width * height),
        DiffusionCoeff(0.25),
        DiffusionCoeff(0.5),
        ZeroKinetics,
        adv,
    );

    for y in 0..height {
        for x in 0..10 {
            let idx = flatten_2d_index(x, y, width);
            sys.u_mut()[idx] = 1.0;
            sys.v_mut()[idx] = 2.0;
        }
    }

    let initial_left_mass: f64 = sys.u().iter().sum();

    for _ in 0..60 {
        sys.step(0.05);
    }

    // Zero reagent may leak into the obstacle wall (x = 10) or right chamber (x > 10)
    for y in 0..height {
        for x in 10..width {
            let idx = flatten_2d_index(x, y, width);
            assert_eq!(
                sys.u()[idx],
                0.0,
                "Non-zero reagent leaked across obstacle barrier at ({x}, {y}): {}",
                sys.u()[idx]
            );
            assert_eq!(sys.v()[idx], 0.0);
        }
    }

    let final_left_mass: f64 = sys.u().iter().sum();
    assert!(
        (final_left_mass - initial_left_mass).abs() < 1e-11,
        "Mass in left chamber must be conserved against solid obstacle: {initial_left_mass} vs {final_left_mass}"
    );
}

#[test]
fn test_vorticity_and_cfl_substeps() {
    let width = 16;
    let height = 16;
    let mut adv = AdvectionDiffusion2D::new(
        Dimension(width),
        Dimension(height),
        StepSize(1.0),
        StepSize(1.0),
    );
    // Solid-body rotation u_x = -0.5 * y, u_y = 0.5 * x -> omega_z = duy/dx - dux/dy = 0.5 - (-0.5) = 1.0
    for y in 0..height {
        for x in 0..width {
            let idx = flatten_2d_index(x, y, width);
            adv.velocity_x_mut()[idx] = -0.5 * (y as f64);
            adv.velocity_y_mut()[idx] = 0.5 * (x as f64);
        }
    }
    let mut vort = vec![0.0; width * height];
    verified_engine::engine::VerifiedEngine::run_verified(|| {
        adv.compute_vorticity(&mut vort);
    });
    let center = flatten_2d_index(8, 8, width);
    assert!((vort[center] - 1.0).abs() < 1e-12);
    assert!(adv.cfl_substeps(0.5, 2.0, 0.45, 64) > 1);
}
