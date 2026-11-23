//! Unit and zero-allocation tests for 2D upwind advective-diffusive morphogenesis and zero-flux masks.
#![cfg_attr(any(), verified(opt_out = "tests"))]

use domain_biology::biology::diffusion::{AdvectiveFiniteDifference2D, FiniteDifference2D};
use domain_biology::biology::morphogenesis::{
    DiffusionCoeff, GrayScottKinetics, ReactionKinetics, SchnakenbergKinetics, TuringSystem,
};
use math_commons::math_kernel::types::{Dimension, StepSize, flatten_2d_index};
use verified_engine::allocator::{VerifiedAllocator, lock_allocations, unlock_allocations};
use verified_engine::metrics::{disable_verification, enable_verification};

#[global_allocator]
static TEST_ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

static ALLOC_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[derive(Clone, Copy, Default)]
struct ZeroReaction;

impl ReactionKinetics<2> for ZeroReaction {
    fn reaction(&self, _concentrations: [f64; 2]) -> [f64; 2] {
        [0.0, 0.0]
    }
}

fn centroid_of_fluid(slice: &[f64], obs: &[bool], width: usize, height: usize) -> (f64, f64) {
    let mut sum_m = 0.0;
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    for y in 0..height {
        for x in 0..width {
            let idx = flatten_2d_index(x, y, width);
            if !obs[idx] {
                let m = slice[idx];
                sum_m += m;
                sum_x += m * x as f64;
                sum_y += m * y as f64;
            }
        }
    }
    (sum_x / sum_m.max(1e-12), sum_y / sum_m.max(1e-12))
}

#[test]
fn test_turing_system_2d_upwind_advection_translates_pulse() {
    let w = 32_usize;
    let h = 24_usize;
    let mut sys = TuringSystem::new_advective_2d(
        (Dimension(w), Dimension(h)),
        DiffusionCoeff(0.02),
        DiffusionCoeff(0.02),
        (StepSize(1.0), StepSize(1.0)),
        2.5,
        ZeroReaction,
    );

    {
        let (ux, uy, obs) = sys.velocity_and_obstacles_mut();
        ux.fill(0.12);
        uy.fill(-0.06);
        obs.fill(false);
    }

    for y in 0..h {
        for x in 0..w {
            let idx = flatten_2d_index(x, y, w);
            let dx = x as f64 - 8.0;
            let dy = y as f64 - 14.0;
            sys.u_mut()[idx] = (-0.25 * (dx * dx + dy * dy)).exp();
            sys.v_mut()[idx] = 0.5 * sys.u()[idx];
        }
    }

    let (x0, y0) = centroid_of_fluid(sys.u(), sys.obstacles(), w, h);
    let dt = sys.max_cfl_dt(0.45).min(0.2);
    for _ in 0..50 {
        sys.step(dt);
    }
    let (x1, y1) = centroid_of_fluid(sys.u(), sys.obstacles(), w, h);

    assert!(x1 > x0 + 2.0, "Expected +x upwind drift, got {x0} -> {x1}");
    assert!(y1 < y0 - 1.0, "Expected -y upwind drift, got {y0} -> {y1}");
}

fn fluid_mass_sum(vals: &[f64], obs: &[bool]) -> f64 {
    vals.iter()
        .zip(obs.iter())
        .filter(|&(_, &o)| !o)
        .map(|(&c, _)| c)
        .sum()
}

fn seed_obstacle_flow(
    sys: &mut TuringSystem<2, ZeroReaction, AdvectiveFiniteDifference2D>,
    w: usize,
    h: usize,
) {
    let (ux, uy, obs) = sys.velocity_and_obstacles_mut();
    for y in 0..h {
        for x in 0..w {
            let idx = flatten_2d_index(x, y, w);
            let is_solid = (x as i32 - 12).pow(2) + (y as i32 - 10).pow(2) <= 9;
            obs[idx] = is_solid;
            if !is_solid {
                ux[idx] = 0.10 * (1.0 - (y as f64 / h as f64 - 0.5).powi(2));
                uy[idx] = 0.04 * ((x as f64 * 0.3).sin());
            }
        }
    }
    for y in 0..h {
        for x in 0..w {
            let idx = flatten_2d_index(x, y, w);
            if sys.obstacles()[idx] {
                sys.u_mut()[idx] = 9.0;
                sys.v_mut()[idx] = 3.0;
            } else {
                let (dx, dy) = (x as f64 - 6.0, y as f64 - 10.0);
                sys.u_mut()[idx] = (-0.2 * (dx * dx + dy * dy)).exp();
                sys.v_mut()[idx] = 0.25 + 0.5 * sys.u()[idx];
            }
        }
    }
}

#[test]
fn test_zero_flux_conservation_around_internal_obstacle() {
    let (w, h) = (28_usize, 20_usize);
    let mut sys = TuringSystem::new_advective_2d(
        (Dimension(w), Dimension(h)),
        DiffusionCoeff(0.12),
        DiffusionCoeff(0.24),
        (StepSize(1.0), StepSize(1.0)),
        3.0,
        ZeroReaction,
    );
    seed_obstacle_flow(&mut sys, w, h);

    let u_init = fluid_mass_sum(sys.u(), sys.obstacles());
    let v_init = fluid_mass_sum(sys.v(), sys.obstacles());

    let dt = sys.max_cfl_dt(0.45).min(0.15);
    for _ in 0..120 {
        sys.step(dt);
    }

    let u_final = fluid_mass_sum(sys.u(), sys.obstacles());
    let v_final = fluid_mass_sum(sys.v(), sys.obstacles());
    assert!(
        (u_final - u_init).abs() < 1e-10,
        "u mass not conserved at obstacle: {u_init} -> {u_final}"
    );
    assert!(
        (v_final - v_init).abs() < 1e-10,
        "v mass not conserved at obstacle: {v_init} -> {v_final}"
    );

    for (i, &is_obs) in sys.obstacles().iter().enumerate() {
        if is_obs {
            assert_eq!(sys.u()[i], 9.0);
            assert_eq!(sys.v()[i], 3.0);
        }
    }
}

#[test]
fn test_pure_diffusion_equivalence_at_zero_velocity() {
    let w = 18_usize;
    let h = 14_usize;
    let fd_pure = FiniteDifference2D::new(Dimension(w), Dimension(h), StepSize(1.0), StepSize(1.0));
    let mut pure_schnak = TuringSystem::new_with_kinetics(
        Dimension(w * h),
        DiffusionCoeff(0.2),
        DiffusionCoeff(4.0),
        SchnakenbergKinetics::new(0.1, 0.9),
        fd_pure,
    );
    let mut adv_schnak = TuringSystem::new_advective_2d(
        (Dimension(w), Dimension(h)),
        DiffusionCoeff(0.2),
        DiffusionCoeff(4.0),
        (StepSize(1.0), StepSize(1.0)),
        5.0,
        SchnakenbergKinetics::new(0.1, 0.9),
    );

    let mut pure_gs = TuringSystem::new_with_kinetics(
        Dimension(w * h),
        DiffusionCoeff(0.16),
        DiffusionCoeff(0.08),
        GrayScottKinetics::mitosis(),
        fd_pure,
    );
    let mut adv_gs = TuringSystem::new_advective_2d(
        (Dimension(w), Dimension(h)),
        DiffusionCoeff(0.16),
        DiffusionCoeff(0.08),
        (StepSize(1.0), StepSize(1.0)),
        5.0,
        GrayScottKinetics::mitosis(),
    );

    for i in 0..(w * h) {
        let u0 = 1.0 + 0.05 * ((i as f64 * 0.37).sin());
        let v0 = 0.9 + 0.04 * ((i as f64 * 0.23).cos());
        pure_schnak.u_mut()[i] = u0;
        pure_schnak.v_mut()[i] = v0;
        adv_schnak.u_mut()[i] = u0;
        adv_schnak.v_mut()[i] = v0;

        let gu = 1.0 - 0.2 * ((i as f64 * 0.19).sin().abs());
        let gv = 0.25 * ((i as f64 * 0.19).sin().abs());
        pure_gs.u_mut()[i] = gu;
        pure_gs.v_mut()[i] = gv;
        adv_gs.u_mut()[i] = gu;
        adv_gs.v_mut()[i] = gv;
    }

    for _ in 0..25 {
        pure_schnak.step(0.02);
        adv_schnak.step(0.02);
        pure_gs.step(0.5);
        adv_gs.step(0.5);
    }

    for i in 0..(w * h) {
        assert!((pure_schnak.u()[i] - adv_schnak.u()[i]).abs() < 1e-13);
        assert!((pure_schnak.v()[i] - adv_schnak.v()[i]).abs() < 1e-13);
        assert!((pure_gs.u()[i] - adv_gs.u()[i]).abs() < 1e-13);
        assert!((pure_gs.v()[i] - adv_gs.v()[i]).abs() < 1e-13);
    }
}

#[test]
fn test_verified_allocator_zero_heap_advective_turing_steps() {
    let _guard = ALLOC_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let w = 24_usize;
    let h = 16_usize;
    let mut schnak =
        TuringSystem::<2, SchnakenbergKinetics, AdvectiveFiniteDifference2D>::new_advective_2d(
            (Dimension(w), Dimension(h)),
            DiffusionCoeff(0.2),
            DiffusionCoeff(4.0),
            (StepSize(1.0), StepSize(1.0)),
            2.0,
            SchnakenbergKinetics::new(0.1, 0.9),
        );
    let mut gs =
        TuringSystem::<2, GrayScottKinetics, AdvectiveFiniteDifference2D>::new_advective_2d(
            (Dimension(w), Dimension(h)),
            DiffusionCoeff(0.16),
            DiffusionCoeff(0.08),
            (StepSize(1.0), StepSize(1.0)),
            2.0,
            GrayScottKinetics::coral(),
        );

    schnak.u_mut().fill(1.0);
    schnak.v_mut().fill(0.9);
    gs.u_mut().fill(1.0);
    gs.v_mut().fill(0.0);

    enable_verification();
    lock_allocations();
    for _ in 0..5 {
        let dt_s = schnak.max_cfl_dt(0.45).min(0.02);
        schnak.step(dt_s);
        let dt_g = gs.max_cfl_dt(0.45).min(0.5);
        gs.step(dt_g);
    }
    unlock_allocations();
    disable_verification();
}
