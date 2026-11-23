//! Unit and integration tests for demand-driven snapshots, sub-step slice budgets,
//! zero-allocation stepping, and native worker visibility suspension.

use super::unified::{UnifiedModel, UnifiedSimRunner};
use super::worker::{advance_running_frame, drain_worker_commands, FrameTickOutcome};
use super::*;
use crate::tabs::chaos::attractors::{AttractorMode, AttractorRunner, AttractorSharedParams};
use crate::tabs::chaos::fractals::{FractalMode, FractalRenderRequest, FractalRunner};
use crate::tabs::fluid_dynamics::lattice_boltzmann::LbmUnified;
use crate::tabs::mri::{MriRunner, MriSharedParams};
use crate::tabs::solid_state::ising::{IsingRunner, IsingSharedParams};
use math_explorer::physics::stat_mech::ising::SpinLattice;
use nalgebra::Vector3;
use num_complex::Complex;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Duration;

struct CountingRunner {
    steps: Arc<AtomicUsize>,
    snapshots: Arc<AtomicUsize>,
    complete: Arc<AtomicBool>,
    finite: Arc<AtomicBool>,
}

impl CountingRunner {
    fn new(
        steps: Arc<AtomicUsize>,
        snapshots: Arc<AtomicUsize>,
        complete: Arc<AtomicBool>,
        finite: Arc<AtomicBool>,
    ) -> Self {
        Self {
            steps,
            snapshots,
            complete,
            finite,
        }
    }
}

impl SimulationRunner for CountingRunner {
    fn process_command(&mut self, cmd: SimCommand) {
        if matches!(cmd, SimCommand::Reset) {
            self.steps.store(0, Ordering::SeqCst);
        }
    }

    fn step(&mut self) {
        self.steps.fetch_add(1, Ordering::SeqCst);
    }

    fn get_snapshot(&self) -> StateSnapshot {
        self.snapshots.fetch_add(1, Ordering::SeqCst);
        let s = self.steps.load(Ordering::SeqCst) as f64;
        StateSnapshot {
            width: 1,
            height: 1,
            pixels: Arc::new(RwLock::new(vec![eframe::egui::Color32::BLACK])),
            custom_data: vec![s],
            structured_data: None,
            sonification: crate::audio::SonificationFrame::silent(),
        }
    }

    fn get_steps_per_frame(&self) -> usize {
        1
    }

    fn is_complete(&self) -> bool {
        self.complete.load(Ordering::SeqCst)
    }

    fn is_finite_state(&self) -> bool {
        self.finite.load(Ordering::SeqCst)
    }
}

#[test]
fn running_ticks_skip_get_snapshot_while_unread_snapshot_waits() {
    let steps = Arc::new(AtomicUsize::new(0));
    let snapshots = Arc::new(AtomicUsize::new(0));
    let mut runner = CountingRunner::new(
        Arc::clone(&steps),
        Arc::clone(&snapshots),
        Arc::new(AtomicBool::new(false)),
        Arc::new(AtomicBool::new(true)),
    );
    let (state_tx, state_rx) = sim_state_channel();
    let mut ring = None;
    let mut state = SimWorkerState::Running;

    for _ in 0..10 {
        let outcome = advance_running_frame(&mut runner, &mut ring, 1, None, &mut state, &state_tx);
        assert!(matches!(outcome, FrameTickOutcome::ContinueRunning));
    }
    assert_eq!(steps.load(Ordering::SeqCst), 10);
    assert_eq!(snapshots.load(Ordering::SeqCst), 1);

    let first_batch = state_rx.drain();
    assert_eq!(first_batch.latest_snapshot.unwrap().custom_data, vec![1.0]);
    assert!(!state_tx.has_unread_snapshot());

    let outcome = advance_running_frame(&mut runner, &mut ring, 1, None, &mut state, &state_tx);
    assert!(matches!(outcome, FrameTickOutcome::ContinueRunning));
    assert_eq!(snapshots.load(Ordering::SeqCst), 2);
    assert_eq!(
        state_rx.drain().latest_snapshot.unwrap().custom_data,
        vec![11.0]
    );
}

#[test]
fn completion_and_fault_emit_terminal_snapshots_over_unread_slot() {
    let steps = Arc::new(AtomicUsize::new(0));
    let snapshots = Arc::new(AtomicUsize::new(0));
    let complete = Arc::new(AtomicBool::new(false));
    let finite = Arc::new(AtomicBool::new(true));
    let mut runner = CountingRunner::new(
        Arc::clone(&steps),
        snapshots,
        Arc::clone(&complete),
        Arc::clone(&finite),
    );
    let (state_tx, state_rx) = sim_state_channel();
    let mut ring = None;

    let mut state = SimWorkerState::Running;
    let _ = advance_running_frame(&mut runner, &mut ring, 1, None, &mut state, &state_tx);
    assert!(state_tx.has_unread_snapshot());
    complete.store(true, Ordering::SeqCst);
    let _ = advance_running_frame(&mut runner, &mut ring, 1, None, &mut state, &state_tx);
    assert_eq!(state, SimWorkerState::Ready);
    assert_eq!(
        state_rx.drain().latest_snapshot.unwrap().custom_data,
        vec![2.0]
    );

    complete.store(false, Ordering::SeqCst);
    state = SimWorkerState::Running;
    let _ = advance_running_frame(&mut runner, &mut ring, 1, None, &mut state, &state_tx);
    finite.store(false, Ordering::SeqCst);
    let _ = advance_running_frame(&mut runner, &mut ring, 1, None, &mut state, &state_tx);
    assert_eq!(state, SimWorkerState::Faulted);
    assert_eq!(
        state_rx.drain().latest_snapshot.unwrap().custom_data,
        vec![4.0]
    );
}

#[test]
fn pause_step_and_reset_commands_emit_terminal_snapshots_over_unread_slot() {
    let mut runner = CountingRunner::new(
        Arc::new(AtomicUsize::new(0)),
        Arc::new(AtomicUsize::new(0)),
        Arc::new(AtomicBool::new(false)),
        Arc::new(AtomicBool::new(true)),
    );
    let (_cmd_tx, cmd_rx) = std::sync::mpsc::channel::<SimCommand>();
    let (state_tx, state_rx) = sim_state_channel();
    let mut ring = None;
    let mut spf = 1;
    let mut state = SimWorkerState::Running;

    let _ = advance_running_frame(&mut runner, &mut ring, 1, None, &mut state, &state_tx);
    runner.step();
    assert!(drain_worker_commands(
        SimCommand::Pause,
        &cmd_rx,
        &mut runner,
        &mut ring,
        &mut state,
        &mut spf,
        &state_tx
    ));
    assert_eq!(
        state_rx.drain().latest_snapshot.unwrap().custom_data,
        vec![2.0]
    );

    let _ = state_tx.send_snapshot(runner.get_snapshot());
    assert!(drain_worker_commands(
        SimCommand::Step(2),
        &cmd_rx,
        &mut runner,
        &mut ring,
        &mut state,
        &mut spf,
        &state_tx
    ));
    assert_eq!(
        state_rx.drain().latest_snapshot.unwrap().custom_data,
        vec![4.0]
    );

    let _ = state_tx.send_snapshot(runner.get_snapshot());
    assert!(drain_worker_commands(
        SimCommand::Reset,
        &cmd_rx,
        &mut runner,
        &mut ring,
        &mut state,
        &mut spf,
        &state_tx
    ));
    assert_eq!(
        state_rx.drain().latest_snapshot.unwrap().custom_data,
        vec![0.0]
    );
}

#[test]
fn fractal_and_ising_enforce_single_step_slice_budgets() {
    let gen = Arc::new(AtomicU64::new(1));
    let shared_req = Arc::new(RwLock::new(FractalRenderRequest {
        generation: 0,
        width: 80,
        height: 60,
        center: Complex::new(-0.1, 0.0),
        zoom: 4.0,
        max_iter: 2500,
        mode: FractalMode::Mandelbrot,
        julia_c: Complex::new(0.0, 0.0),
        high_contrast: false,
    }));
    let mut fractal = FractalRunner::new(Arc::clone(&shared_req), gen);
    {
        let mut guard = shared_req.write().unwrap();
        guard.generation = 1;
        guard.width = 2048;
        guard.height = 2048;
    }
    let (state_tx, _state_rx) = sim_state_channel();
    let mut ring = None;
    let mut state = SimWorkerState::Running;
    let spf = fractal.get_steps_per_frame();
    let _ = advance_running_frame(
        &mut fractal,
        &mut ring,
        spf,
        Some(Duration::from_micros(500)),
        &mut state,
        &state_tx,
    );
    assert!(!fractal.is_complete());

    let ising_params = Arc::new(RwLock::new(IsingSharedParams {
        steps_per_frame: 50_000,
        ..IsingSharedParams::default()
    }));
    let mut ising = IsingRunner {
        lattice: SpinLattice::new(64, 64, Some(42)),
        params: Arc::clone(&ising_params),
    };
    ising.step_sliced(Some(Duration::ZERO));
    let trials = ising_params
        .read()
        .unwrap()
        .total_trials
        .load(Ordering::Relaxed);
    assert_eq!(trials, 512);
}

#[test]
fn mri_and_attractor_enforce_single_step_slice_budgets() {
    let mri_params = Arc::new(RwLock::new(MriSharedParams {
        t1: 1.0,
        t2: 0.2,
        b_field: Vector3::new(0.0, 0.0, 1.0e-7),
        time_scale: 80.0,
    }));
    let mut mri = MriRunner::new(mri_params, Vector3::new(1.0, 0.0, 0.0), 1.0);
    mri.step_sliced(Some(Duration::ZERO));
    assert_eq!(mri.current_step(), 1);

    let attr_params = Arc::new(RwLock::new(AttractorSharedParams {
        sigma: 10.0,
        rho: 28.0,
        beta: 8.0 / 3.0,
        dt: 0.01,
        simulation_speed: 60,
        max_points: 200,
        mode: AttractorMode::Lorenz,
        custom_flow: None,
    }));
    let mut attr = AttractorRunner::from_params(attr_params);
    attr.step_sliced(Some(Duration::ZERO));
    assert_eq!(attr.current_step(), 1);
}

#[test]
fn ising_and_unified_runner_step_with_zero_heap_allocations() {
    let ising_params = Arc::new(RwLock::new(IsingSharedParams {
        steps_per_frame: 2048,
        ..IsingSharedParams::default()
    }));
    let mut ising = IsingRunner {
        lattice: SpinLattice::new(32, 32, Some(7)),
        params: ising_params,
    };
    ising.step_sliced(Some(Duration::from_millis(2)));

    let lbm_defaults: HashMap<String, f64> = LbmUnified::parameters()
        .into_iter()
        .map(|(k, v)| (k, (v.min + v.max) * 0.5))
        .collect();
    let mut unified = UnifiedSimRunner::<LbmUnified>::new(Arc::new(RwLock::new(lbm_defaults)));
    let mut ring = CheckpointRing::new(16);
    ring.reset_and_seed(&unified);
    unified.step();

    let _alloc_guard = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    verified_engine::engine::VerifiedEngine::run_verified(|| {
        ising.step_sliced(Some(Duration::from_millis(2)));
        ring.record_active_params(&unified);
        unified.step();
    });
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn controller_suspension_halts_background_steps_while_preserving_running_state() {
    let steps = Arc::new(AtomicUsize::new(0));
    let runner = CountingRunner::new(
        Arc::clone(&steps),
        Arc::new(AtomicUsize::new(0)),
        Arc::new(AtomicBool::new(false)),
        Arc::new(AtomicBool::new(true)),
    );
    let mut controller = SimulationController::new(runner);
    controller.send_command(SimCommand::Start).unwrap();

    for _ in 0..40 {
        if steps.load(Ordering::SeqCst) >= 2 {
            break;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    assert!(steps.load(Ordering::SeqCst) >= 2);

    controller.set_suspended(true);
    assert!(controller.is_suspended());
    assert_eq!(controller.state(), SimWorkerState::Running);
    std::thread::sleep(Duration::from_millis(25));
    let frozen = steps.load(Ordering::SeqCst);
    std::thread::sleep(Duration::from_millis(55));
    assert_eq!(steps.load(Ordering::SeqCst), frozen);
    assert_eq!(controller.state(), SimWorkerState::Running);

    controller.set_suspended(false);
    assert!(!controller.is_suspended());
    for _ in 0..40 {
        if steps.load(Ordering::SeqCst) > frozen {
            break;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    assert!(steps.load(Ordering::SeqCst) > frozen);
}
