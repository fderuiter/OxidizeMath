//! Co-stepped LatticeBoltzmannD2Q9 and TuringSystem zero-heap-allocation tests under VerifiedAllocator.
#![cfg(all(feature = "biology", feature = "physics"))]

use math_explorer::biology::diffusion::AdvectiveFiniteDifference2D;
use math_explorer::biology::morphogenesis::{
    DiffusionCoeff, GrayScottKinetics, SchnakenbergKinetics, TuringSystem,
};
use math_explorer::math_kernel::types::{Dimension, StepSize, flatten_2d_index};
use math_explorer::physics::fluid_dynamics::lattice_boltzmann::{
    BgkCollision, LatticeBoltzmannD2Q9, advance_wind_tunnel_step, compute_vorticity_into,
};
use math_explorer::verified_engine::allocator::{
    VerifiedAllocator, lock_allocations, unlock_allocations,
};
use math_explorer::verified_engine::metrics::{disable_verification, enable_verification};

#[global_allocator]
static TEST_ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

static ALLOC_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn build_cylinder_lbm(width: usize, height: usize) -> LatticeBoltzmannD2Q9<BgkCollision> {
    let mut lbm = LatticeBoltzmannD2Q9::<BgkCollision>::new(width, height, 0.56);
    for y in 20..30 {
        for x in 26..34 {
            let dx = x as i32 - 30;
            let dy = y as i32 - 25;
            if dx * dx + dy * dy <= 25 {
                lbm.set_obstacle(x, y, true);
            }
        }
    }
    lbm.set_inlet(0, 0, width, height, 0.1, 0.0);
    lbm
}

#[test]
fn test_co_stepped_lbm_d2q9_and_turing_zero_heap_allocations() {
    let _guard = ALLOC_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let (width, height) = (100_usize, 50_usize);
    let mut lbm = build_cylinder_lbm(width, height);
    let dims = (Dimension(width), Dimension(height));
    let spacing = (StepSize(1.0), StepSize(1.0));

    let mut schnak =
        TuringSystem::<2, SchnakenbergKinetics, AdvectiveFiniteDifference2D>::new_advective_2d(
            dims,
            DiffusionCoeff(0.2),
            DiffusionCoeff(4.0),
            spacing,
            8.0,
            SchnakenbergKinetics::new(0.1, 0.9),
        );
    let mut gray_scott =
        TuringSystem::<2, GrayScottKinetics, AdvectiveFiniteDifference2D>::new_advective_2d(
            dims,
            DiffusionCoeff(0.16),
            DiffusionCoeff(0.08),
            spacing,
            4.0,
            GrayScottKinetics::coral(),
        );

    let mut vorticity = vec![0.0; width * height];
    for y in 0..height {
        for x in 0..width {
            let idx = flatten_2d_index(x, y, width);
            schnak.u_mut()[idx] = 1.0 + 0.02 * ((idx as f64 * 0.17).sin());
            schnak.v_mut()[idx] = 0.9;
            gray_scott.u_mut()[idx] = 1.0;
            gray_scott.v_mut()[idx] =
                usize::from((36..44).contains(&x) && (22..28).contains(&y)) as f64 * 0.25;
        }
    }

    enable_verification();
    lock_allocations();
    for _ in 0..6 {
        advance_wind_tunnel_step(&mut lbm, 0.1);
        compute_vorticity_into(&lbm, &mut vorticity);
        let (_fd, _fl) = lbm.hydrodynamic_forces();
        let (ux, uy, obs) = schnak.velocity_and_obstacles_mut();
        lbm.copy_velocity_and_obstacles_row_major(ux, uy, obs);
        schnak.pin_solid_species([1.0, 0.9]);
        schnak.step(schnak.max_cfl_dt(0.45).min(0.02));

        let (ux_g, uy_g, obs_g) = gray_scott.velocity_and_obstacles_mut();
        lbm.copy_velocity_and_obstacles_row_major(ux_g, uy_g, obs_g);
        gray_scott.pin_solid_species([1.0, 0.0]);
        gray_scott.step(gray_scott.max_cfl_dt(0.45).min(0.8));
    }
    unlock_allocations();
    disable_verification();

    assert!(schnak.u().iter().all(|v| v.is_finite()));
    assert!(gray_scott.v().iter().all(|v| v.is_finite()));
}
