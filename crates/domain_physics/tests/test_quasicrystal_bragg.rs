//! Integration tests for zero-allocation 2D quasicrystals and reciprocal Bragg diffraction.

// [cite:solid_state]

use domain_physics::physics::solid_state::quasicrystal::{
    GOLDEN_RATIO, QuasicrystalKind, QuasicrystalParams, QuasicrystalWorkspace, SILVER_RATIO,
    STAMPFLI_RATIO,
};
use domain_physics::physics::solid_state::reciprocal::{
    BraggDiffractionWorkspace, BravaisLattice2D, Lattice2DParams, kinematic_structure_factor,
};
use std::f64::consts::PI;
use std::sync::Mutex;
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

static ALLOC_TEST_LOCK: Mutex<()> = Mutex::new(());

fn lock_alloc() -> std::sync::MutexGuard<'static, ()> {
    ALLOC_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[test]
fn quasicrystal_projections_match_golden_silver_and_stampfli_ratios_within_one_percent() {
    let _guard = lock_alloc();
    let mut ws = QuasicrystalWorkspace::new();

    let penrose = ws
        .project_into(&QuasicrystalParams::new(
            QuasicrystalKind::Penrose5D,
            1.0,
            28.0,
        ))
        .expect("Penrose projection");
    let penrose_rel_err = (penrose.empirical_ratio - GOLDEN_RATIO).abs() / GOLDEN_RATIO;
    assert!(
        penrose_rel_err < 0.01,
        "Penrose ratio {} deviated from golden ratio {} by {:.4}%",
        penrose.empirical_ratio,
        GOLDEN_RATIO,
        penrose_rel_err * 100.0
    );

    let ammann = ws
        .project_into(&QuasicrystalParams::new(
            QuasicrystalKind::AmmannBeenker4D,
            1.0,
            28.0,
        ))
        .expect("Ammann-Beenker projection");
    let ammann_rel_err = (ammann.empirical_ratio - SILVER_RATIO).abs() / SILVER_RATIO;
    assert!(
        ammann_rel_err < 0.01,
        "Ammann-Beenker ratio {} deviated from silver ratio {} by {:.4}%",
        ammann.empirical_ratio,
        SILVER_RATIO,
        ammann_rel_err * 100.0
    );

    let stampfli = ws
        .project_into(&QuasicrystalParams::new(
            QuasicrystalKind::Stampfli6D,
            1.0,
            32.0,
        ))
        .expect("Stampfli projection");
    let stampfli_rel_err = (stampfli.empirical_ratio - STAMPFLI_RATIO).abs() / STAMPFLI_RATIO;
    assert!(
        stampfli_rel_err < 0.01,
        "Stampfli ratio {} deviated from 2 + sqrt(3) = {} by {:.4}%",
        stampfli.empirical_ratio,
        STAMPFLI_RATIO,
        stampfli_rel_err * 100.0
    );
    assert!(
        (stampfli.secondary_geometric_ratio - SILVER_RATIO).abs() / SILVER_RATIO < 0.01,
        "Stampfli square-plus-edge geometric span must match silver ratio"
    );
}

#[test]
fn quasicrystal_256x256_bragg_maps_exhibit_forbidden_10_8_and_12_fold_symmetries() {
    let _guard = ALLOC_TEST_LOCK.lock().expect("lock alloc mutex");
    let mut qc_ws = QuasicrystalWorkspace::new();
    let mut fft_ws = BraggDiffractionWorkspace::new(256).expect("256x256 FFT workspace");

    let cases = [
        (QuasicrystalKind::Penrose5D, 10_usize, 0.99_f64),
        (QuasicrystalKind::AmmannBeenker4D, 8_usize, 0.99_f64),
        (QuasicrystalKind::Stampfli6D, 12_usize, 0.98_f64),
    ];

    for (kind, order, threshold) in cases {
        let params = QuasicrystalParams::new(kind, 1.0, 28.0);
        qc_ws.project_into(&params).expect("project quasicrystal");
        let report = fft_ws
            .compute_diffraction_into(
                qc_ws.positions_xy(),
                params.edge_length,
                params.aperture_radius,
            )
            .expect("compute Bragg FFT");
        let cn = report.correlation(order);
        assert!(
            cn >= threshold,
            "{:?} C_{} = {:.5} fell below threshold {:.2} (all C4..C12 = {:?})",
            kind,
            order,
            cn,
            threshold,
            report.c4_to_c12
        );
        assert!(
            !fft_ws.peaks().is_empty() && fft_ws.peaks()[0].structure_factor_amp > 0.05,
            "{:?} must yield non-trivial Bragg structure-factor peaks",
            kind
        );
    }
}

#[test]
fn all_five_periodic_2d_bravais_nets_satisfy_orthogonality_bz_area_and_diffraction_symmetry() {
    let _guard = ALLOC_TEST_LOCK.lock().expect("lock alloc mutex");
    let mut sites = Vec::with_capacity(8_192);
    let mut fft_ws = BraggDiffractionWorkspace::new(256).expect("256x256 FFT workspace");

    let lattices = [
        Lattice2DParams::new(BravaisLattice2D::Oblique, 1.0, 1.35, 1.25).expect("oblique"),
        Lattice2DParams::new(BravaisLattice2D::Rectangular, 1.0, 1.45, PI * 0.5).expect("rect"),
        Lattice2DParams::new(BravaisLattice2D::CenteredRectangular, 1.2, 1.9, PI * 0.5)
            .expect("centered rect"),
        Lattice2DParams::square(1.0).expect("square"),
        Lattice2DParams::hexagonal(1.0).expect("hexagonal"),
    ];

    for lat in lattices {
        let [a1, a2] = lat.direct_basis();
        let [b1, b2] = lat.reciprocal_basis();
        assert!((a1.dot(&b1) - 2.0 * PI).abs() < 1e-12);
        assert!((a2.dot(&b2) - 2.0 * PI).abs() < 1e-12);
        assert!(a1.dot(&b2).abs() < 1e-12);
        assert!(a2.dot(&b1).abs() < 1e-12);

        let bz = lat.first_brillouin_zone();
        let expected_area = 4.0 * PI * PI / lat.unit_cell_area();
        assert!(
            (bz.area() - expected_area).abs() / expected_area < 1e-9,
            "{:?} BZ area {} != expected {}",
            lat.lattice,
            bz.area(),
            expected_area
        );

        lat.generate_sites_into(26.0, &mut sites);
        let sf_g = kinematic_structure_factor(&sites, b1 + b2);
        assert!(
            (sf_g.norm() - 1.0).abs() < 1e-10,
            "Exact reciprocal vector G = b1 + b2 must yield |S(G)| = 1 for {:?}",
            lat.lattice
        );

        let report = fft_ws
            .compute_diffraction_into(&sites, lat.a, 26.0)
            .expect("Bravais FFT");
        let order = lat.lattice.symmetry_order();
        assert!(
            report.correlation(order) >= 0.98,
            "{:?} expected C_{} >= 0.98, got {:.5}",
            lat.lattice,
            order,
            report.correlation(order)
        );
        assert!(
            report.correlation(5) < 0.75 && report.correlation(10) < 0.75,
            "Periodic {:?} must not exhibit crystallographic-forbidden 5/10-fold symmetry",
            lat.lattice
        );
    }
}

#[test]
fn phason_shifts_and_256x256_rustfft_execute_with_zero_heap_allocations() {
    let _guard = ALLOC_TEST_LOCK.lock().expect("lock alloc mutex");
    let mut qc_ws = QuasicrystalWorkspace::new();
    let mut fft_ws = BraggDiffractionWorkspace::new(256).expect("256x256 FFT workspace");

    let ((stats_small, report_small, stats_large, report_large), metrics) =
        VerifiedEngine::run_verified(|| {
            let mut p1 = QuasicrystalParams::new(QuasicrystalKind::Penrose5D, 1.0, 28.0);
            p1.phason = [0.025, -0.018];
            let s1 = qc_ws.project_into(&p1).expect("zero-alloc projection 1");
            let r1 = fft_ws
                .compute_diffraction_into(qc_ws.positions_xy(), p1.edge_length, p1.aperture_radius)
                .expect("zero-alloc 2D Bragg FFT 1");

            let mut p2 = p1;
            p2.phason = [0.22, -0.15];
            let s2 = qc_ws.project_into(&p2).expect("zero-alloc projection 2");
            let r2 = fft_ws
                .compute_diffraction_into(qc_ws.positions_xy(), p2.edge_length, p2.aperture_radius)
                .expect("zero-alloc 2D Bragg FFT 2");
            (s1, r1, s2, r2)
        });

    assert!(
        metrics.function_calls > 0,
        "Verified metrics must record calls inside run_verified"
    );
    assert!(
        stats_small.flipped_tile_count > 0 && report_small.correlation(10) >= 0.99,
        "Moderate phason shift must flip tiles ({}) and preserve C_10 >= 0.99 (got {:.5})",
        stats_small.flipped_tile_count,
        report_small.correlation(10)
    );
    assert!(
        stats_large.flipped_tile_count > 100 && report_large.correlation(10) >= 0.97,
        "Large phason shift must flip >100 tiles ({}) and preserve C_10 >= 0.97 (got {:.5})",
        stats_large.flipped_tile_count,
        report_large.correlation(10)
    );
}

#[test]
fn bragg_diffraction_workspace_rejects_non_power_of_two_and_oversized_grids() {
    let _guard = lock_alloc();
    for bad_size in [0_usize, 16, 48, 8_192, 65_536, usize::MAX] {
        assert!(
            BraggDiffractionWorkspace::new(bad_size).is_err(),
            "BraggDiffractionWorkspace::new({bad_size}) must fail cleanly without overflow"
        );
    }
}
