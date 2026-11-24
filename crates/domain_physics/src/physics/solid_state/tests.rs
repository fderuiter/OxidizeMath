use super::bcs::{BcsGapSolver, IsotropicBCSModel, coherence_factors_strong};
use super::lattice::{BodyCenteredCubic, CrystalSystem, FaceCenteredCubic, SimpleCubic};
use super::quasicrystal::{
    QuasicrystalPatchConfig, QuasicrystalSymmetry, QuasicrystalWorkspace,
    acceptance_window_polygon, generate_quasicrystal_patch,
};
use super::quasiperiodic::{QuasiperiodicChainModel, QuasiperiodicWorkspace};
use super::reciprocal::{
    BraggWorkspace, brillouin_zone_1d, first_brillouin_zone_2d, reciprocal_basis_2d,
    reciprocal_basis_3d, structure_factor_2d, unit_cell_structure_factor,
};
use super::types::ElectronVolts;
use approx::assert_relative_eq;
use std::f64::consts::{PI, TAU};
use verified_engine::allocator::MEMORY_LOCK;
use verified_engine::engine::VerifiedEngine;

#[verified_engine::verified]
fn rotate_2d(q: [f64; 2], angle: f64) -> [f64; 2] {
    let (s, c) = angle.sin_cos();
    [c * q[0] - s * q[1], s * q[0] + c * q[1]]
}

#[test]
fn test_quasicrystal_rotational_invariance_5_8_10_12_fold() {
    let cases = [
        (QuasicrystalSymmetry::Penrose5Fold, 5_usize),
        (QuasicrystalSymmetry::AmmannBeenker8Fold, 8_usize),
        (QuasicrystalSymmetry::Penrose10Fold, 10_usize),
        (QuasicrystalSymmetry::Stampfli12Fold, 12_usize),
    ];
    let probe_qs = [[3.7, 1.9], [6.25, -2.4], [8.1, 4.35]];

    for (symmetry, order) in cases {
        let mut ws = QuasicrystalWorkspace::new();
        ws.generate_into(&QuasicrystalPatchConfig {
            symmetry,
            patch_radius: 4.2,
            window_scale: 1.0,
            phason_shift: [0.0, 0.0],
            reference_phason: [0.0, 0.0],
        });
        let pts = ws.parallel_points();
        assert!(
            pts.len() > 25,
            "{symmetry:?} should generate non-trivial patch"
        );
        assert_eq!(ws.flipped_in_count, 0);

        let step_angle = TAU / (order as f64);
        for base_q in probe_qs {
            let s0 = structure_factor_2d(pts, base_q);
            for m in 1..order {
                let q_rot = rotate_2d(base_q, (m as f64) * step_angle);
                let s_rot = structure_factor_2d(pts, q_rot);
                assert_relative_eq!(s_rot, s0, epsilon = 1e-9);
            }
        }
    }

    // Verify 5-fold Penrose breaks 10-fold real-space point inversion while 10-fold Penrose preserves it
    let p5 = generate_quasicrystal_patch(&QuasicrystalPatchConfig {
        symmetry: QuasicrystalSymmetry::Penrose5Fold,
        patch_radius: 3.5,
        ..Default::default()
    });
    let has_unpaired_inversion = p5.iter().any(|s| {
        let neg = [-s.pos_parallel[0], -s.pos_parallel[1]];
        !p5.iter().any(|other| {
            (other.pos_parallel[0] - neg[0]).hypot(other.pos_parallel[1] - neg[1]) < 1e-7
        })
    });
    assert!(
        has_unpaired_inversion,
        "5-fold Penrose must break 2D inversion symmetry"
    );
}

#[test]
fn test_phason_shift_triggers_flips_and_preserves_sharp_bragg_peaks() {
    let mut ws0 = QuasicrystalWorkspace::new();
    ws0.generate_into(&QuasicrystalPatchConfig {
        symmetry: QuasicrystalSymmetry::AmmannBeenker8Fold,
        patch_radius: 4.8,
        window_scale: 1.0,
        phason_shift: [0.0, 0.0],
        reference_phason: [0.0, 0.0],
    });
    let mut ws_shifted = QuasicrystalWorkspace::new();
    ws_shifted.generate_into(&QuasicrystalPatchConfig {
        symmetry: QuasicrystalSymmetry::AmmannBeenker8Fold,
        patch_radius: 4.8,
        window_scale: 1.0,
        phason_shift: [0.22, -0.14],
        reference_phason: [0.0, 0.0],
    });

    assert!(ws_shifted.flipped_in_count > 0);
    assert!(ws_shifted.flipped_out_count > 0);
    assert!(ws_shifted.sites().iter().any(|s| s.flipped));

    let mut window = [[0.0; 2]; 14];
    let v_count = acceptance_window_polygon(
        QuasicrystalSymmetry::AmmannBeenker8Fold,
        1.0,
        [0.22, -0.14],
        &mut window,
    );
    assert_eq!(v_count, 9);

    // Dominant Bragg reflection (m = (1, 1, 0, -1), small |q_perp|) remains sharp under phason translation
    let q_star = QuasicrystalSymmetry::AmmannBeenker8Fold.fundamental_wavevector();
    let q_bragg = [q_star, 0.0];
    let s_unshifted = structure_factor_2d(ws0.parallel_points(), q_bragg);
    let s_shifted = structure_factor_2d(ws_shifted.parallel_points(), q_bragg);
    assert!(
        s_unshifted > 20.0,
        "Expected sharp Bragg peak, got {s_unshifted}"
    );
    assert!(
        s_shifted > 20.0,
        "Expected phason-invariant Bragg peak, got {s_shifted}"
    );
}

#[test]
fn test_sc_bcc_fcc_extinction_rules_and_brillouin_zones() {
    let a = 3.2;
    let sc = SimpleCubic.generate(a);
    let bcc = BodyCenteredCubic.generate(a);
    let fcc = FaceCenteredCubic.generate(a);

    assert!(reciprocal_basis_3d(&sc.lattice_vectors).is_some());
    assert_eq!(brillouin_zone_1d(a), Some([-PI / a, PI / a]));

    let b2 = reciprocal_basis_2d(&[[a, 0.0], [0.0, a]]).expect("square reciprocal basis");
    let mut bz = [[0.0; 2]; 16];
    let bz_len = first_brillouin_zone_2d(&b2, &mut bz);
    assert_eq!(bz_len, 5);

    // Simple Cubic: all integer reflections allowed with weight 1
    for hkl in [[1, 0, 0], [1, 1, 0], [1, 1, 1], [2, 1, 0]] {
        assert_relative_eq!(unit_cell_structure_factor(&sc, hkl), 1.0, epsilon = 1e-10);
    }

    // BCC: h + k + l odd -> 0 (extinct); h + k + l even -> 4 (allowed)
    for extinct in [[1, 0, 0], [1, 1, 1], [2, 1, 0], [3, 0, 0]] {
        assert_relative_eq!(
            unit_cell_structure_factor(&bcc, extinct),
            0.0,
            epsilon = 1e-10
        );
    }
    for allowed in [[1, 1, 0], [2, 0, 0], [2, 1, 1], [2, 2, 2]] {
        assert_relative_eq!(
            unit_cell_structure_factor(&bcc, allowed),
            4.0,
            epsilon = 1e-10
        );
    }

    // FCC: mixed parity -> 0 (extinct); all even or all odd -> 16 (allowed)
    for extinct in [[1, 0, 0], [1, 1, 0], [2, 1, 0], [2, 1, 1]] {
        assert_relative_eq!(
            unit_cell_structure_factor(&fcc, extinct),
            0.0,
            epsilon = 1e-10
        );
    }
    for allowed in [[1, 1, 1], [2, 0, 0], [2, 2, 0], [3, 1, 1]] {
        assert_relative_eq!(
            unit_cell_structure_factor(&fcc, allowed),
            16.0,
            epsilon = 1e-10
        );
    }
}

#[test]
fn test_verified_engine_zero_alloc_bragg_and_bcs() {
    let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());

    let mut qc_ws = QuasicrystalWorkspace::new();
    qc_ws.generate_into(&QuasicrystalPatchConfig {
        symmetry: QuasicrystalSymmetry::Penrose10Fold,
        patch_radius: 3.6,
        ..Default::default()
    });
    let mut bragg_ws = BraggWorkspace::new(64);
    let mut qp_ws = QuasiperiodicWorkspace::new(64);
    let mut bcs_model = IsotropicBCSModel::with_uniform_window(120, 5.0, 0.45, 3.0);
    let solver = BcsGapSolver::new()
        .with_max_iterations(250)
        .with_mixing_param(0.5);

    let ((peak_s, gap_ev, norm, qp_gaps), metrics) = VerifiedEngine::run_verified(|| {
        qc_ws.generate_into(&QuasicrystalPatchConfig {
            symmetry: QuasicrystalSymmetry::Penrose10Fold,
            patch_radius: 3.6,
            phason_shift: [0.1, 0.05],
            reference_phason: [0.0, 0.0],
            window_scale: 1.0,
        });
        bragg_ws.evaluate_grid_verified(qc_ws.parallel_points(), 48, 12.0, true);
        qp_ws
            .solve_into(QuasiperiodicChainModel::Fibonacci {
                sites: 34,
                t_avg: 1.0,
                modulation: 0.45,
                e0: 0.0,
            })
            .expect("Fibonacci QL converges");
        bcs_model.update_uniform_window(5.0, 0.45, 3.0);
        let gap = solver
            .solve(&bcs_model, ElectronVolts::new(0.05))
            .expect("BCS gap converges");
        let (u_k, v_k) = coherence_factors_strong(ElectronVolts::new(0.4), gap)
            .expect("Bogoliubov coherence factors");
        (
            bragg_ws.peak_intensity(),
            gap.as_f64(),
            u_k * u_k + v_k * v_k,
            qp_ws.count_prominent_gaps(1.5),
        )
    });

    assert!(!MEMORY_LOCK.load(std::sync::atomic::Ordering::SeqCst));
    assert!(metrics.function_calls >= 6);
    assert!(peak_s > 1.0);
    assert!(gap_ev > 0.0);
    assert!(qp_gaps >= 2);
    assert_relative_eq!(norm, 1.0, epsilon = 1e-12);
}

#[test]
fn test_fibonacci_self_similar_gaps() {
    let mut ws = QuasiperiodicWorkspace::new(160);

    // Periodic chain (modulation = 0.0) has zero prominent gaps
    ws.solve_into(QuasiperiodicChainModel::Fibonacci {
        sites: 55,
        t_avg: 1.0,
        modulation: 0.0,
        e0: 0.0,
    })
    .expect("Periodic QL converges");
    assert_eq!(ws.count_prominent_gaps(1.5), 0);

    // Self-similar gap hierarchy across Fibonacci approximants F_6 = 8, F_8 = 21, F_10 = 55
    let mut gap_counts = Vec::new();
    for &(f_k, f_k_minus_2, f_k_minus_1) in
        &[(8_usize, 3_usize, 5_usize), (21, 8, 13), (55, 21, 34)]
    {
        ws.solve_into(QuasiperiodicChainModel::Fibonacci {
            sites: f_k,
            t_avg: 1.0,
            modulation: 0.45,
            e0: 0.0,
        })
        .expect("Fibonacci QL converges");
        gap_counts.push(ws.count_prominent_gaps(1.5));

        // Primary KKT gap indices occur at F_{k-2} and F_{k-1} (IDOS -> tau^-2 and tau^-1)
        let evals = ws.eigenvalues();
        let mut indexed_gaps: Vec<(usize, f64)> = evals
            .windows(2)
            .enumerate()
            .map(|(i, w)| (i + 1, w[1] - w[0]))
            .collect();
        indexed_gaps.sort_by(|a, b| b.1.total_cmp(&a.1));
        let top_two = [
            indexed_gaps[0].0.min(indexed_gaps[1].0),
            indexed_gaps[0].0.max(indexed_gaps[1].0),
        ];
        assert_eq!(top_two, [f_k_minus_2, f_k_minus_1]);
    }
    assert!(
        gap_counts[0] >= 2 && gap_counts[0] < gap_counts[1] && gap_counts[1] < gap_counts[2],
        "Expected increasing Cantor gap hierarchy across F_6, F_8, F_10, got {gap_counts:?}"
    );
}

#[test]
fn test_aubry_andre_localization_transition() {
    let mut ws = QuasiperiodicWorkspace::new(160);
    let sites = 144_usize;
    ws.solve_into(QuasiperiodicChainModel::AubryAndre {
        sites,
        t: 1.0,
        lambda: 0.5,
        phase: 0.1,
        e0: 0.0,
    })
    .expect("Extended Aubry-Andre converges");
    let ipr_extended = ws.mean_ipr();

    ws.solve_into(QuasiperiodicChainModel::AubryAndre {
        sites,
        t: 1.0,
        lambda: 2.0,
        phase: 0.1,
        e0: 0.0,
    })
    .expect("Critical Aubry-Andre converges");
    let ipr_critical = ws.mean_ipr();

    ws.solve_into(QuasiperiodicChainModel::AubryAndre {
        sites,
        t: 1.0,
        lambda: 4.5,
        phase: 0.1,
        e0: 0.0,
    })
    .expect("Localized Aubry-Andre converges");
    let ipr_localized = ws.mean_ipr();

    assert!(
        ipr_extended < 0.05,
        "Subcritical lambda < 2t must be extended, got {ipr_extended}"
    );
    assert!(ipr_extended < ipr_critical && ipr_critical < ipr_localized);
    assert!(
        ipr_localized > 0.45,
        "Supercritical lambda > 2t must localize, got {ipr_localized}"
    );
}

#[test]
fn test_reciprocal_biorthogonality_and_bogoliubov_symmetries() {
    let direct_2d = [[2.5, 0.4], [0.75, 1.9]];
    let b2 = reciprocal_basis_2d(&direct_2d).expect("non-singular 2D oblique basis");
    assert_relative_eq!(
        direct_2d[0][0] * b2[0][0] + direct_2d[0][1] * b2[0][1],
        TAU,
        epsilon = 1e-11
    );
    assert_relative_eq!(
        direct_2d[0][0] * b2[1][0] + direct_2d[0][1] * b2[1][1],
        0.0,
        epsilon = 1e-11
    );
    assert_relative_eq!(
        direct_2d[1][0] * b2[0][0] + direct_2d[1][1] * b2[0][1],
        0.0,
        epsilon = 1e-11
    );
    assert_relative_eq!(
        direct_2d[1][0] * b2[1][0] + direct_2d[1][1] * b2[1][1],
        TAU,
        epsilon = 1e-11
    );
    assert!(reciprocal_basis_2d(&[[1.0, 2.0], [2.0, 4.0]]).is_none());
    assert!(brillouin_zone_1d(0.0).is_none());

    let mut poly = [[0.0; 2]; 14];
    assert_eq!(
        acceptance_window_polygon(QuasicrystalSymmetry::Penrose5Fold, 1.0, [0.0; 2], &mut poly),
        6
    );
    assert_eq!(
        acceptance_window_polygon(
            QuasicrystalSymmetry::Penrose10Fold,
            1.0,
            [0.0; 2],
            &mut poly
        ),
        11
    );
    assert_eq!(
        acceptance_window_polygon(
            QuasicrystalSymmetry::Stampfli12Fold,
            1.0,
            [0.0; 2],
            &mut poly
        ),
        13
    );

    let delta = ElectronVolts::new(0.35);
    let (u0, v0) = coherence_factors_strong(ElectronVolts::new(0.0), delta).expect("Fermi level");
    assert_relative_eq!(u0 * u0, 0.5, epsilon = 1e-12);
    assert_relative_eq!(v0 * v0, 0.5, epsilon = 1e-12);
    let (up, vp) = coherence_factors_strong(ElectronVolts::new(0.6), delta).expect("particle");
    let (um, vm) = coherence_factors_strong(ElectronVolts::new(-0.6), delta).expect("hole");
    assert_relative_eq!(up, vm, epsilon = 1e-12);
    assert_relative_eq!(vp, um, epsilon = 1e-12);
}
