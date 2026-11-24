#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::atlas::{IsingAtlasRegime, IsingCartographer, onsager_critical_temp};
use super::challenge::{
    CHALLENGE_FERMI_PD_NUCLEATION, CHALLENGE_ISING_DROPLET_NUCLEATION, IsingChallengeWorkspace,
};
use super::*;
use crate::async_sim::SimulationRunner;
use crate::async_sim::atlas::{
    AtlasPassMeta, AtlasRenderRequest, AtlasRunner, AtlasTeleportEvent, RegimeCartographer,
};
use scientific_metadata::theory::TheoryDescribable;
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, RwLock};

#[test]
fn test_boltzmann_j_scaling_and_antiferromagnetic_staggered_snapshot() {
    // Verify Onsager critical temperature scales linearly with |J| (J = 0.5 => Tc ≈ 1.1346)
    let tc_half = onsager_critical_temp(0.5);
    assert!((tc_half - 1.134_592_6).abs() < 1e-4);

    // At J = 0.5 and T = 1.6 > Tc(0.5) = 1.135 (which under the old * |J| bug acted like T_eff = 0.8 < Tc!),
    // the lattice must thermalize into the disordered paramagnetic phase (|M| < 0.45).
    let params_para = Arc::new(RwLock::new(IsingSharedParams {
        temperature: 1.6,
        j_coupling: 0.5,
        h_field: 0.0,
        steps_per_frame: 250_000,
        ..Default::default()
    }));
    let mut runner_para = IsingRunner {
        lattice: SpinLattice::new(24, 24, Some(77)),
        params: params_para,
    };
    runner_para.lattice.reset_uniform_with_seed(1, 77);
    runner_para.step();
    let m_para = (runner_para.lattice.magnetization() as f64 / (24.0 * 24.0)).abs();
    assert!(
        m_para < 0.45,
        "Expected paramagnetic disorder at J=0.5, T=1.6 > Tc(0.5)=1.135, got |M| = {m_para:.3}"
    );

    // Antiferromagnetic J = -1.0 reports checkerboard staggered magnetization in snapshot custom_data[0]
    let params_afm = Arc::new(RwLock::new(IsingSharedParams {
        temperature: 0.6,
        j_coupling: -1.0,
        h_field: 0.0,
        mode: IsingInteractionMode::Ferromagnetic,
        steps_per_frame: 1000,
        ..Default::default()
    }));
    let mut runner_afm = IsingRunner {
        lattice: SpinLattice::new(16, 16, Some(99)),
        params: params_afm,
    };
    for y in 0..16 {
        for x in 0..16 {
            runner_afm
                .lattice
                .set(x, y, if (x + y) % 2 == 0 { 1 } else { -1 });
        }
    }
    let snap = runner_afm.get_snapshot();
    assert_eq!(snap.custom_data.len(), 2);
    assert!(
        (snap.custom_data[0] - 1.0).abs() < 1e-6,
        "Expected staggered magnetization +1.0 for checkerboard J < 0, got {}",
        snap.custom_data[0]
    );
}

#[test]
fn test_absorbed_fermi_lattice_auto_reseeds_on_preset_and_atlas_teleport() {
    let ctx = egui::Context::default();
    let mut tool = IsingModelTool {
        mode: IsingInteractionMode::FermiGame,
        uniform_magnetization: -1.0,
        magnetization: -1.0,
        ode_initial_cooperator: 0.0,
        ..Default::default()
    };

    // Switching preset from an absorbed all-defector lattice auto-reseeds 50/50 so 0D & 2D escape 0.0
    tool.apply_game_preset(FermiGamePreset::HawkDove, &ctx);
    assert!(
        (tool.cooperator_fraction() - 0.5).abs() < 1e-6,
        "Absorbed Fermi lattice must auto-reseed on preset change"
    );
    let ode_end = tool.replicator_curve.last().expect("HawkDove ODE")[1];
    assert!(
        ode_end > 0.25 && ode_end < 0.75,
        "Hawk-Dove 0D ODE must converge to interior polymorphism instead of sticking at 0.0, got {ode_end}"
    );

    // Simulate re-absorption and verify (T, S) atlas teleport also auto-reseeds
    tool.uniform_magnetization = -1.0;
    tool.magnetization = -1.0;
    tool.apply_atlas_teleport(
        AtlasTeleportEvent {
            x_key: "T",
            x_val: 1.25,
            y_key: "S",
            y_val: -0.1,
            regime_label: "PD Spatial Reciprocity Wedge (2R + 2S ≥ T + 3P)",
        },
        Some(&ctx),
    );
    assert!((tool.cooperator_fraction() - 0.5).abs() < 1e-6);
}

#[test]
fn test_ising_cartographer_pd_freeze_fraction_and_contours() {
    let tool = IsingModelTool::default();
    let theory = tool.theory_parameters();
    let game_cart =
        IsingCartographer::new(&theory, ("T", "S"), 0.08, 1.0, 0.0, [1.0, -0.1, 1.25, 0.0]);

    assert_eq!(game_cart.overlay_contour().unwrap().0, "T = R");
    let extra = game_cart.extra_overlay_contours();
    assert_eq!(extra.len(), 2);
    assert_eq!(extra[0].0, "S = P");
    assert_eq!(extra[1].0, "2R + 2S = T");

    let mut pd_total = 0_usize;
    let mut pd_freeze = 0_usize;
    for it in 0..100 {
        let t = 1.005 + (it as f64) * 0.01;
        for is_idx in 0..100 {
            let s = -0.995 + (is_idx as f64) * 0.01;
            let cell = game_cart.evaluate_cell(t, s);
            pd_total += 1;
            if cell.regime == IsingAtlasRegime::DisorderedOrDefectorFreeze as u8 {
                pd_freeze += 1;
            }
        }
    }
    let freeze_pct = 100.0 * pd_freeze as f64 / pd_total as f64;
    assert!(
        (freeze_pct - 75.0).abs() < 1.5,
        "Expected ~75.0% Prisoner's Dilemma defector freeze in (T, S), got {freeze_pct:.2}%"
    );
}

#[test]
fn test_ising_cartographer_zero_alloc_and_128x128_convergence() {
    let tool = IsingModelTool::default();
    let theory = tool.theory_parameters();
    let game_cart =
        IsingCartographer::new(&theory, ("T", "S"), 0.08, 1.0, 0.0, [1.0, -0.1, 1.25, 0.0]);
    let ferro_cart = IsingCartographer::new(
        &theory,
        ("temperature", "j_coupling"),
        2.2,
        1.0,
        0.0,
        [1.0, -0.1, 1.25, 0.0],
    );
    let mut ws = IsingChallengeWorkspace::default();
    let super_seed = [DropletSeed {
        cx: 50,
        cy: 50,
        r: 8,
        spin: 1,
    }];
    {
        let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let (acc, _) = verified_engine::engine::VerifiedEngine::run_verified(|| {
            let c1 = game_cart.evaluate_cell(1.25, -0.1);
            let c2 = ferro_cart.evaluate_cell(1.5, -1.0);
            let s1 = ws.evaluate_ising_nucleation(&super_seed);
            let s2 = ws.evaluate_fermi_nucleation(&super_seed);
            c1.primary + c2.primary + s1.final_plus_fraction + s2.final_plus_fraction
        });
        assert!(acc.is_finite());
    }

    let latest_gen = Arc::new(AtomicU64::new(1));
    let req = Arc::new(RwLock::new(AtlasRenderRequest {
        generation: 1,
        width: 128,
        height: 128,
        high_contrast: true,
        cartographer: Arc::new(game_cart),
    }));
    let mut runner = AtlasRunner::new(req, latest_gen);
    for _ in 0..64 {
        runner.step();
    }
    let snap = runner.get_snapshot();
    assert_eq!((snap.width, snap.height), (128, 128));
    let meta = snap
        .structured_data
        .as_ref()
        .and_then(|d| d.downcast_ref::<AtlasPassMeta>())
        .expect("AtlasPassMeta in snapshot");
    assert!(meta.stride == 1 && meta.is_final);
}

#[test]
fn test_ising_th_atlas_128x128_high_contrast_and_parameter_clamping() {
    let tool = IsingModelTool::default();
    let theory = tool.theory_parameters();
    let th_cart = IsingCartographer::new(
        &theory,
        ("temperature", "h_field"),
        2.2,
        1.0,
        0.0,
        [1.0, -0.1, 1.25, 0.0],
    );
    for (p1, p2) in [(1.0, 0.2), (2.269, 0.0), (4.5, 0.5)] {
        let hc = th_cart.color_for_cell(th_cart.evaluate_cell(p1, p2), true);
        let lin = |u: u8| {
            let x = f64::from(u) / 255.0;
            if x <= 0.04045 {
                x / 12.92
            } else {
                ((x + 0.055) / 1.055).powf(2.4)
            }
        };
        let lum = 0.2126 * lin(hc.r()) + 0.7152 * lin(hc.g()) + 0.0722 * lin(hc.b());
        assert!((lum + 0.05) / 0.05 >= 3.0);
    }
    let latest_th = Arc::new(AtomicU64::new(1));
    let req_th = Arc::new(RwLock::new(AtlasRenderRequest {
        generation: 1,
        width: 128,
        height: 128,
        high_contrast: true,
        cartographer: Arc::new(th_cart),
    }));
    let mut runner_th = AtlasRunner::new(req_th, latest_th);
    for _ in 0..64 {
        runner_th.step();
    }
    let snap_th = runner_th.get_snapshot();
    assert_eq!((snap_th.width, snap_th.height), (128, 128));
    let meta_th = snap_th
        .structured_data
        .as_ref()
        .and_then(|d| d.downcast_ref::<AtlasPassMeta>())
        .expect("AtlasPassMeta in (T, H) snapshot");
    assert!(meta_th.stride == 1 && meta_th.is_final);

    let mut dirty = IsingModelTool {
        temperature: f64::NAN,
        ..Default::default()
    };
    assert!(!dirty.has_finite_parameters());
    assert!(dirty.clamp_parameters());
    assert!(dirty.has_finite_parameters());
}

#[test]
fn test_droplet_seed_persistence_bounds_and_nucleation_challenges() {
    let mut tool = IsingModelTool {
        defector_sea: true,
        mutation_rate: 0.02,
        ..Default::default()
    };
    for i in 0..MAX_DROPLET_SEEDS {
        tool.droplet_seeds.push(DropletSeed {
            cx: (i * 3) as i32,
            cy: (i * 2) as i32,
            r: 4,
            spin: 1,
        });
    }
    let json = tool.save_state().expect("serializes 32 droplet seeds");
    assert!(
        json.len() <= 4096,
        "32 droplet seeds + IsingState must stay <= 4096 bytes, got {}",
        json.len()
    );
    let mut restored = IsingModelTool::default();
    restored.try_load_state(&json).expect("restores 32 seeds");
    assert_eq!(restored.droplet_seeds.len(), MAX_DROPLET_SEEDS);
    assert!(restored.defector_sea);
    assert!((restored.mutation_rate - 0.02).abs() < 1e-9);

    //Both solid_state nucleation challenges fail at subcritical baseline and pass with r=8 supercritical droplet
    for cid in [
        CHALLENGE_ISING_DROPLET_NUCLEATION,
        CHALLENGE_FERMI_PD_NUCLEATION,
    ] {
        let meta = crate::starter_gallery::find_challenge_metadata(cid).expect("registered");
        let base_json = meta.generate_baseline_state();
        let mut ctool = IsingModelTool::default();
        ctool.apply_challenge(Some(cid)).unwrap();
        ctool.try_load_state(&base_json).unwrap();
        ctool.apply_challenge(Some(cid)).unwrap();
        assert!(
            !ctool.evaluate_challenge().unwrap().passed,
            "Challenge '{cid}' baseline must fail"
        );

        ctool.droplet_seeds = vec![DropletSeed {
            cx: 50,
            cy: 50,
            r: 8,
            spin: 1,
        }];
        ctool.refresh_challenge_workspace();
        let solved = ctool.evaluate_challenge().unwrap();
        assert!(
            solved.passed,
            "Supercritical r=8 droplet must solve '{cid}': {:?}",
            solved
        );
    }
}

#[test]
fn test_shared_bus_graph_mode_coexists_with_nucleation_challenges_and_state_roundtrip() {
    use crate::app::{SharedSimulationBus, TopologySource};

    let ctx = egui::Context::default();
    let bus = SharedSimulationBus::default();
    bus.install_in_context(&ctx, true);
    bus.publish_graph_if_changed(
        1,
        false,
        &[
            (0, [0.1, 0.1]),
            (1, [0.9, 0.1]),
            (2, [0.9, 0.9]),
            (3, [0.1, 0.9]),
        ],
        &[(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 0, 1.0)],
    );

    let mut tool = IsingModelTool::default();
    tool.sync_shared_topology(&ctx);
    assert!(tool.is_graph_mode());
    assert_eq!(tool.active_graph_node_count(), Some(4));
    tool.step_graph_trials(64);
    let expected_coop = ((tool.uniform_magnetization + 1.0) * 0.5).clamp(0.0, 1.0);
    assert!((tool.cooperator_fraction() - expected_coop).abs() < 1e-12);

    // Activating a nucleation challenge detaches CSR graph mode and locks onto 100x100 lattice
    tool.apply_challenge(Some(CHALLENGE_ISING_DROPLET_NUCLEATION))
        .unwrap();
    assert!(!tool.is_graph_mode());
    tool.sync_shared_topology(&ctx);
    assert!(!tool.is_graph_mode());

    // Exiting challenge re-enables SharedBus ingestion; LocalPreset round-trips cleanly
    tool.apply_challenge(None).unwrap();
    tool.sync_shared_topology(&ctx);
    assert!(tool.is_graph_mode());
    tool.topology_source = TopologySource::LocalPreset;
    let saved = tool.save_state().unwrap();
    let mut restored = IsingModelTool::default();
    restored.try_load_state(&saved).unwrap();
    assert_eq!(restored.topology_source, TopologySource::LocalPreset);
}
