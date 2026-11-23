use super::*;
use crate::async_sim::unified::{UnifiedModel, UnifiedSimRunner};
use crate::tabs::chaos::attractors::{
    default_flow_bank, AttractorMode, AttractorRunner, AttractorSharedParams, CompiledFlowJob,
};
use crate::tabs::morphogenesis::{
    MorphogenesisRunner, MorphogenesisSharedParams, MorphogenesisTab,
};
use crate::tabs::mri::{MriRunner, MriSharedParams};
use crate::tabs::neuroscience::hodgkin_huxley::HodgkinHuxleyUnified;
use nalgebra::Vector3;
use std::collections::HashMap;
use std::sync::RwLock;

fn make_morpho_runner(
    width: usize,
    height: usize,
    dt: f64,
    d_v: f64,
    noise_amplitude: f64,
) -> (MorphogenesisRunner, Arc<RwLock<MorphogenesisSharedParams>>) {
    let params = Arc::new(RwLock::new(MorphogenesisSharedParams {
        a: 0.14,
        b: 0.86,
        d_u: 1.0,
        d_v,
        dt,
        high_contrast: false,
    }));
    let mut runner = MorphogenesisTab::create_runner(&params, width, height);
    runner.noise_amplitude = noise_amplitude;
    (runner, params)
}

fn step_with_ring<R: SimulationRunner>(runner: &mut R, ring: &mut CheckpointRing, steps: usize) {
    ring.prepare_forward_advance();
    for _ in 0..steps {
        ring.record_active_params(runner);
        runner.step_once();
        ring.on_substep_advanced(runner);
    }
}

fn assert_slice_bits_eq(actual: &[f64], expected: &[f64], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (idx, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            e.to_bits(),
            "{label}: bit mismatch at index {idx}: {a} vs {e}"
        );
    }
}

#[test]
fn test_attractor_k1_bit_exact_backward_and_forward_scrub() {
    let params = Arc::new(RwLock::new(AttractorSharedParams {
        sigma: 10.0,
        rho: 28.0,
        beta: 8.0 / 3.0,
        dt: 0.01,
        simulation_speed: 5,
        max_points: 60,
        mode: AttractorMode::Lorenz,
        custom_flow: None,
    }));
    let mut runner = AttractorRunner::from_params(Arc::clone(&params));
    let mut ring = CheckpointRing::new(1);
    ring.reset_and_seed(&runner);
    step_with_ring(&mut runner, &mut ring, 100);

    let mut direct_42 = AttractorRunner::from_params(Arc::clone(&params));
    for _ in 0..42 {
        direct_42.step_once();
    }
    ring.seek_to_step(&mut runner, 42);
    assert_eq!(runner.step_count, 42);
    assert_eq!(ring.metadata().max_step, 100);
    assert_eq!(
        runner.system.state.vec.x.to_bits(),
        direct_42.system.state.vec.x.to_bits()
    );
    assert_eq!(runner.history.len(), direct_42.history.len());
    for (a, e) in runner.history.iter().zip(direct_42.history.iter()) {
        assert_eq!(a.x.to_bits(), e.x.to_bits());
        assert_eq!(a.y.to_bits(), e.y.to_bits());
        assert_eq!(a.z.to_bits(), e.z.to_bits());
    }

    let mut direct_85 = AttractorRunner::from_params(Arc::clone(&params));
    for _ in 0..85 {
        direct_85.step_once();
    }
    ring.seek_to_step(&mut runner, 85);
    assert_eq!(runner.history.len(), 60);
    assert_eq!(
        runner.system.state.vec.z.to_bits(),
        direct_85.system.state.vec.z.to_bits()
    );
    for (a, e) in runner.history.iter().zip(direct_85.history.iter()) {
        assert_eq!(a.x.to_bits(), e.x.to_bits());
    }
}

#[test]
fn test_attractor_custom_flow_checkpoint_and_substep_replay() {
    let bank = default_flow_bank();
    let job = Arc::new(CompiledFlowJob::from_bank(&bank));
    let params = Arc::new(RwLock::new(AttractorSharedParams {
        sigma: 10.0,
        rho: 28.0,
        beta: 8.0 / 3.0,
        dt: 0.015,
        simulation_speed: 5,
        max_points: 50,
        mode: AttractorMode::Custom,
        custom_flow: Some(job),
    }));
    let mut runner = AttractorRunner::from_params(Arc::clone(&params));
    let mut ring = CheckpointRing::new(4);
    ring.reset_and_seed(&runner);
    step_with_ring(&mut runner, &mut ring, 20);

    let mut direct_11 = AttractorRunner::from_params(Arc::clone(&params));
    for _ in 0..11 {
        direct_11.step_once();
    }
    ring.seek_to_step(&mut runner, 11);
    assert_eq!(runner.step_count, 11);
    assert_eq!(
        runner.system.state.vec.x.to_bits(),
        direct_11.system.state.vec.x.to_bits()
    );
    assert_eq!(
        runner.system.state.vec.y.to_bits(),
        direct_11.system.state.vec.y.to_bits()
    );
    assert_eq!(
        runner.system.state.vec.z.to_bits(),
        direct_11.system.state.vec.z.to_bits()
    );
}

#[test]
fn test_mri_k1_bit_exact_replay_with_param_edits() {
    let mri_params = Arc::new(RwLock::new(MriSharedParams {
        t1: 1.0,
        t2: 0.2,
        b_field: Vector3::new(0.0, 0.0, 1.0e-7),
        time_scale: 1.0,
    }));
    let mut mri = MriRunner::new(Arc::clone(&mri_params), Vector3::new(1.0, 0.0, 0.0), 1.0);
    let mut mri_ring = CheckpointRing::new(1);
    mri_ring.reset_and_seed(&mri);
    step_with_ring(&mut mri, &mut mri_ring, 30);
    mri_params.write().unwrap().b_field.x = 0.25e-7;
    step_with_ring(&mut mri, &mut mri_ring, 40);

    let mri_direct_params = Arc::new(RwLock::new(MriSharedParams {
        t1: 1.0,
        t2: 0.2,
        b_field: Vector3::new(0.0, 0.0, 1.0e-7),
        time_scale: 1.0,
    }));
    let mut mri_direct = MriRunner::new(
        Arc::clone(&mri_direct_params),
        Vector3::new(1.0, 0.0, 0.0),
        1.0,
    );
    for _ in 0..30 {
        mri_direct.step_once();
    }
    mri_direct_params.write().unwrap().b_field.x = 0.25e-7;
    for _ in 0..15 {
        mri_direct.step_once();
    }
    mri_ring.seek_to_step(&mut mri, 45);
    assert_eq!(
        mri.current_time.to_bits(),
        mri_direct.current_time.to_bits()
    );
    assert_eq!(
        mri.simulator.magnetization.x.to_bits(),
        mri_direct.simulator.magnetization.x.to_bits()
    );
    assert_eq!(mri.history.len(), mri_direct.history.len());
}

#[test]
fn test_hodgkin_huxley_k1_bit_exact_replay_with_param_edits() {
    let hh_map = HashMap::from([
        ("g_na".to_string(), 120.0),
        ("g_k".to_string(), 36.0),
        ("g_l".to_string(), 0.3),
        ("i_ext".to_string(), 10.0),
    ]);
    let hh_params = Arc::new(RwLock::new(hh_map.clone()));
    let mut hh = UnifiedSimRunner::<HodgkinHuxleyUnified>::new(Arc::clone(&hh_params));
    let mut hh_ring = CheckpointRing::new(1);
    hh_ring.reset_and_seed(&hh);
    step_with_ring(&mut hh, &mut hh_ring, 25);
    hh_params.write().unwrap().insert("i_ext".to_string(), 22.0);
    step_with_ring(&mut hh, &mut hh_ring, 35);

    let mut hh_direct = HodgkinHuxleyUnified::new(&hh_map);
    for _ in 0..25 {
        hh_direct.step(&hh_map);
    }
    let mut hh_edited = hh_map;
    hh_edited.insert("i_ext".to_string(), 22.0);
    for _ in 0..15 {
        hh_direct.step(&hh_edited);
    }
    hh_ring.seek_to_step(&mut hh, 40);
    assert_slice_bits_eq(
        &hh.get_snapshot().custom_data,
        &hh_direct.get_snapshot().custom_data,
        "HodgkinHuxleyUnified scrub(40)",
    );
}

#[test]
fn test_morphogenesis_k16_mid_stride_param_edit_and_stochastic_rng_replay() {
    let (mut runner, params) = make_morpho_runner(24, 24, 0.002, 50.0, 5e-4);
    let mut ring = CheckpointRing::new(16);
    ring.reset_and_seed(&runner);

    step_with_ring(&mut runner, &mut ring, 23);
    {
        let mut p = params.write().unwrap();
        p.a = 0.22;
        p.b = 0.78;
        p.d_v = 60.0;
    }
    step_with_ring(&mut runner, &mut ring, 25);

    let (mut direct_28, direct_params) = make_morpho_runner(24, 24, 0.002, 50.0, 5e-4);
    for _ in 0..23 {
        direct_28.step_once();
    }
    {
        let mut p = direct_params.write().unwrap();
        p.a = 0.22;
        p.b = 0.78;
        p.d_v = 60.0;
    }
    for _ in 0..5 {
        direct_28.step_once();
    }

    ring.seek_to_step(&mut runner, 28);
    assert_eq!(runner.step_count, 28);
    assert_slice_bits_eq(runner.system.u(), direct_28.system.u(), "Morpho u(28)");
    assert_slice_bits_eq(runner.system.v(), direct_28.system.v(), "Morpho v(28)");
    assert_eq!(
        rand::RngCore::next_u64(&mut runner.rng),
        rand::RngCore::next_u64(&mut direct_28.rng),
        "Cloned OxidizeRng stream must align bit-for-bit after K=16 sub-step replay"
    );
}

#[test]
fn test_future_keyframes_preserved_on_scrub_and_truncated_only_on_forward_resume() {
    let (mut runner, _) = make_morpho_runner(16, 16, 0.002, 50.0, 0.0);
    let mut ring = CheckpointRing::new(16);
    ring.reset_and_seed(&runner);
    step_with_ring(&mut runner, &mut ring, 64);
    assert_eq!(ring.metadata().max_step, 64);
    assert_eq!(ring.keyframe_count(), 5);

    ring.seek_to_step(&mut runner, 20);
    assert_eq!(ring.metadata().current_step, 20);
    assert_eq!(ring.metadata().max_step, 64);
    assert_eq!(ring.keyframe_count(), 5);

    ring.seek_to_step(&mut runner, 52);
    assert_eq!(ring.metadata().current_step, 52);
    assert_eq!(ring.metadata().max_step, 64);

    ring.seek_to_step(&mut runner, 20);
    step_with_ring(&mut runner, &mut ring, 1);
    assert_eq!(ring.metadata().current_step, 21);
    assert_eq!(ring.metadata().max_step, 21);
    assert_eq!(ring.keyframe_count(), 2);
}

#[test]
fn test_ring_enforces_2mib_cap_over_10000_steps_and_clears_on_reset() {
    let (mut runner, _) = make_morpho_runner(100, 100, 0.002, 50.0, 0.0);
    let mut ring = CheckpointRing::new(16);
    ring.reset_and_seed(&runner);

    for _ in 0..10_000 {
        ring.record_active_params(&runner);
        runner.step_once();
        ring.on_substep_advanced(&runner);
        assert!(
            ring.total_bytes() <= MAX_CHECKPOINT_RING_BYTES,
            "Ring exceeded 2 MiB: {} > {}",
            ring.total_bytes(),
            MAX_CHECKPOINT_RING_BYTES
        );
    }

    let meta = ring.metadata();
    assert_eq!(meta.current_step, 10_000);
    assert_eq!(meta.max_step, 10_000);
    assert!(
        meta.min_step > 9_000,
        "Oldest 100x100 keyframes must evict under 2 MiB cap, got min_step={}",
        meta.min_step
    );

    let target = meta.min_step + 7;
    ring.seek_to_step(&mut runner, target);
    assert_eq!(runner.current_step(), target);
    assert!(runner.is_finite_state());

    runner.process_command(SimCommand::Reset);
    ring.reset_and_seed(&runner);
    assert_eq!(ring.keyframe_count(), 1);
    assert_eq!(
        ring.metadata(),
        TimelineMetadata {
            current_step: 0,
            min_step: 0,
            max_step: 0,
            stride: 16,
        }
    );
    assert!(ring.total_bytes() < 200_000);
}

#[test]
fn test_zero_allocation_substep_replay_under_verified_engine() {
    let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());

    let (mut morpho, _) = make_morpho_runner(40, 40, 0.002, 50.0, 1e-4);
    let mut morpho_ring = CheckpointRing::new(16);
    morpho_ring.reset_and_seed(&morpho);
    step_with_ring(&mut morpho, &mut morpho_ring, 32);

    let hh_params = Arc::new(RwLock::new(HashMap::from([
        ("g_na".to_string(), 120.0),
        ("g_k".to_string(), 36.0),
        ("g_l".to_string(), 0.3),
        ("i_ext".to_string(), 10.0),
    ])));
    let mut hh = UnifiedSimRunner::<HodgkinHuxleyUnified>::new(hh_params);
    let mut hh_ring = CheckpointRing::new(1);
    hh_ring.reset_and_seed(&hh);
    step_with_ring(&mut hh, &mut hh_ring, 10);

    let (sought_step, _) = verified_engine::engine::VerifiedEngine::run_verified(|| {
        hh_ring.seek_to_step(&mut hh, 5);
        morpho_ring.seek_to_step(&mut morpho, 31)
    });
    assert_eq!(sought_step, 31);
    assert_eq!(hh.current_step(), 5);
}

#[test]
fn test_worst_case_k16_substep_replay_completes_within_slice_budget() {
    let (mut morpho, _) = make_morpho_runner(100, 100, 0.002, 50.0, 0.0);
    let mut ring = CheckpointRing::new(16);
    ring.reset_and_seed(&morpho);
    step_with_ring(&mut morpho, &mut ring, 32);

    let mut best_elapsed = std::time::Duration::MAX;
    for _ in 0..5 {
        let start = crate::web_io::PlatformInstant::now();
        ring.seek_to_step(&mut morpho, 31);
        best_elapsed = best_elapsed.min(start.elapsed());
    }
    #[cfg(not(debug_assertions))]
    let budget = std::time::Duration::from_millis(8);
    #[cfg(debug_assertions)]
    let budget = std::time::Duration::from_millis(250);
    assert!(
        best_elapsed <= budget,
        "Expected 15-substep 100x100 replay <= {budget:?}, got {best_elapsed:?}"
    );
}
