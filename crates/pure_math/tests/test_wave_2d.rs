//! Integration tests verifying zero dynamic heap allocations across 100 consecutive
//! 2D Verlet wave/plate, Chladni sand particle, Marching-Squares, and 2D FFT steps
//! under `VerifiedAllocator::lock_allocations()`.

// [cite:advanced_linear_algebra]

use pure_math::pure_math::analysis::pde::wave_2d::particles_and_contours::DEFAULT_SAND_PARTICLES;
use pure_math::pure_math::analysis::pde::wave_2d::{
    BoundaryMode2D, ChladniSandSystem, DriverConfig2D, MarchingSquares2D, SpectralAnalyzer2D,
    Wave2DMedium, Wave2DParams, Wave2DSolver, apply_point_clamp,
};
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

fn build_verified_fixture(
    grid_n: usize,
) -> (
    Wave2DSolver,
    Wave2DParams,
    DriverConfig2D,
    ChladniSandSystem,
) {
    let mut solver = Wave2DSolver::new_unit_square(grid_n);
    apply_point_clamp(&mut solver.wall_mask, grid_n, grid_n, 0.5, 0.5, 2.0);
    solver.inject_gaussian_pulse(0.32, 0.68, 0.06, 0.5);

    let params = Wave2DParams {
        medium: Wave2DMedium::BiharmonicPlate,
        boundary: BoundaryMode2D::Neumann,
        wave_speed: 0.05,
        plate_stiffness: 1.0e-4,
        damping: 0.1,
    };
    let driver = DriverConfig2D {
        enabled: true,
        line_source: false,
        x_norm: 0.25,
        y_norm: 0.35,
        frequency: 14.0,
        amplitude: 30.0,
        radius_cells: 2.0,
    };
    let sand = ChladniSandSystem::new(
        DEFAULT_SAND_PARTICLES,
        0xC41A_D210,
        grid_n,
        grid_n,
        &solver.wall_mask,
    );
    (solver, params, driver, sand)
}

#[test]
fn test_100_wave_sand_and_fft_steps_zero_allocations() {
    let grid_n = 128_usize;
    let (mut solver, params, driver, mut sand) = build_verified_fixture(grid_n);
    let mut fft = SpectralAnalyzer2D::new(grid_n, grid_n);
    let mut contours = MarchingSquares2D::new(8_192);
    let mut detector_line = vec![0.0; grid_n];
    let mut radial_bins = vec![0.0; grid_n / 2];
    let mut radial_counts = vec![0_usize; grid_n / 2];

    let (peak_mag, _metrics) = VerifiedEngine::run_verified(|| {
        let mut latest_mag = 0.0_f64;
        for _ in 0..100 {
            solver.step_driven(&params, &driver, 2.0e-4, Some(grid_n / 2));
            sand.step(
                solver.energy_field(),
                &solver.wall_mask,
                grid_n,
                grid_n,
                10.0,
                0.25,
                0.08,
            );
            let _ = fft.compute_spectrum(solver.displacement());
            latest_mag = fft.dominant_peak().magnitude;
            fft.compute_radial_profile(&mut radial_bins, &mut radial_counts);
            let _ = contours.extract_zero_contours(
                solver.modal_field(),
                &solver.wall_mask,
                grid_n,
                grid_n,
                1.0e-5,
            );
            solver.read_detector_profile(grid_n / 2, &mut detector_line);
        }
        latest_mag
    });

    assert!(peak_mag.is_finite() && peak_mag > 0.0);
    assert!(solver.displacement().iter().all(|v| v.is_finite()));
}
