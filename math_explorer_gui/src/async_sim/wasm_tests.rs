use super::*;
#[cfg(feature = "physics")]
use crate::tabs::chaos::attractors::{AttractorMode, AttractorRunner, AttractorSharedParams};
#[cfg(feature = "biology")]
use crate::tabs::morphogenesis::{MorphogenesisSharedParams, MorphogenesisTool};
use std::sync::RwLock;
use wasm_bindgen::JsCast;
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

async fn tick_microtasks() {
    let promise = js_sys::Promise::new(&mut |resolve, _| {
        let global = js_sys::global();
        if let Ok(win) = global.dyn_into::<web_sys::Window>() {
            let _ = win.set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, 0);
        } else {
            let _ = resolve.call0(&wasm_bindgen::JsValue::UNDEFINED);
        }
    });
    let _ = wasm_bindgen_futures::JsFuture::from(promise).await;
}

#[cfg(any(feature = "biology", feature = "physics"))]
async fn wait_wasm_step(controller: &mut SimulationController, expected: usize) {
    for _ in 0..100 {
        let _ = controller.update();
        if controller.state() == SimWorkerState::Ready
            && controller
                .timeline_metadata()
                .is_some_and(|m| m.current_step == expected)
        {
            return;
        }
        tick_microtasks().await;
    }
    panic!(
        "WASM worker did not settle at step {expected}; state={} meta={:?}",
        controller.state(),
        controller.timeline_metadata()
    );
}

#[cfg(feature = "biology")]
#[wasm_bindgen_test]
async fn wasm_worker_checkpoint_step_stepback_seek_and_reset() {
    let params = Arc::new(RwLock::new(MorphogenesisSharedParams {
        a: 0.14,
        b: 0.86,
        d_u: 1.0,
        d_v: 50.0,
        dt: 0.002,
        high_contrast: false,
        periodic_boundary: false,
        theme_mode: crate::app::ThemeMode::Dark,
        heatmap_mode: crate::tabs::morphogenesis::MorphogenesisHeatmapMode::ActiveUb,
        mask_rle: None,
    }));
    let runner = MorphogenesisTool::create_runner(&params, 20, 20);
    let mut controller = SimulationController::new(runner);
    assert_eq!(controller.timeline_metadata().unwrap().stride, 16);

    controller.send_command(SimCommand::Step(20)).unwrap();
    wait_wasm_step(&mut controller, 20).await;

    controller.send_command(SimCommand::StepBack(1)).unwrap();
    wait_wasm_step(&mut controller, 19).await;

    controller.send_command(SimCommand::SeekStep(10)).unwrap();
    wait_wasm_step(&mut controller, 10).await;
    assert_eq!(controller.timeline_metadata().unwrap().max_step, 20);

    controller.send_command(SimCommand::Reset).unwrap();
    wait_wasm_step(&mut controller, 0).await;
    assert_eq!(controller.timeline_metadata().unwrap().max_step, 0);
}

#[cfg(feature = "physics")]
#[wasm_bindgen_test]
async fn wasm_attractor_running_slice_and_seek_pauses_cleanly() {
    let params = Arc::new(RwLock::new(AttractorSharedParams {
        sigma: 10.0,
        rho: 28.0,
        beta: 8.0 / 3.0,
        dt: 0.01,
        simulation_speed: 5,
        max_points: 200,
        mode: AttractorMode::Lorenz,
        custom_flow: None,
        sim_bus: None,
    }));
    let mut controller = SimulationController::new(AttractorRunner::from_params(params));
    controller.send_command(SimCommand::Start).unwrap();
    for _ in 0..8 {
        tick_microtasks().await;
        let _ = controller.update();
    }
    let reached = controller.timeline_metadata().unwrap().max_step;
    assert!(
        reached >= 5,
        "Expected running frames to advance >= 5 steps"
    );

    controller.send_command(SimCommand::SeekStep(3)).unwrap();
    wait_wasm_step(&mut controller, 3).await;
    assert_eq!(controller.state(), SimWorkerState::Ready);
}

async fn await_animation_frame() {
    let promise = js_sys::Promise::new(&mut |resolve, _| {
        let global = js_sys::global();
        if let Ok(win) = global.dyn_into::<web_sys::Window>() {
            let hidden = win.document().is_some_and(|d| d.hidden());
            if !hidden && win.request_animation_frame(&resolve).is_ok() {
                return;
            }
            let _ = win.set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, 18);
        } else {
            let _ = resolve.call0(&wasm_bindgen::JsValue::UNDEFINED);
        }
    });
    let _ = wasm_bindgen_futures::JsFuture::from(promise).await;
}

struct WasmCountingRunner {
    steps: Arc<std::sync::atomic::AtomicUsize>,
    snapshots: Arc<std::sync::atomic::AtomicUsize>,
}

impl SimulationRunner for WasmCountingRunner {
    fn process_command(&mut self, _cmd: SimCommand) {}

    fn step(&mut self) {
        self.steps.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }

    fn get_snapshot(&self) -> StateSnapshot {
        self.snapshots
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        StateSnapshot {
            width: 1,
            height: 1,
            pixels: Arc::new(RwLock::new(vec![eframe::egui::Color32::WHITE])),
            custom_data: vec![self.steps.load(std::sync::atomic::Ordering::SeqCst) as f64],
            structured_data: None,
            sonification: crate::audio::SonificationFrame::silent(),
        }
    }

    fn get_steps_per_frame(&self) -> usize {
        1
    }
}

#[wasm_bindgen_test]
async fn wasm_worker_ticks_at_most_once_per_animation_frame_and_skips_unread_snapshots() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let steps = Arc::new(AtomicUsize::new(0));
    let snapshots = Arc::new(AtomicUsize::new(0));
    let mut controller = SimulationController::new(WasmCountingRunner {
        steps: Arc::clone(&steps),
        snapshots: Arc::clone(&snapshots),
    });
    let initial_snaps = snapshots.load(Ordering::SeqCst);
    controller.send_command(SimCommand::Start).unwrap();

    // Extra Custom commands in the same frame epoch must NOT double-tick within the epoch.
    controller
        .send_command(SimCommand::Custom("noop".to_string()))
        .unwrap();
    controller
        .send_command(SimCommand::Custom("noop".to_string()))
        .unwrap();
    wasm_bindgen_futures::JsFuture::from(js_sys::Promise::resolve(&wasm_bindgen::JsValue::NULL))
        .await
        .unwrap();
    assert!(
        steps.load(Ordering::SeqCst) <= 1,
        "Worker must tick at most once in the current frame epoch despite mid-frame commands"
    );

    // Advance 3 animation frames WITHOUT calling controller.update() to verify unread snapshots are skipped.
    for _ in 0..3 {
        await_animation_frame().await;
    }
    let advanced = steps.load(Ordering::SeqCst);
    assert!(
        (2..=5).contains(&advanced),
        "Expected 1 tick per frame, got {advanced}"
    );
    assert_eq!(
        snapshots.load(Ordering::SeqCst),
        initial_snaps + 1,
        "Unread snapshot slot must skip intermediate get_snapshot() clones across unpolled frames"
    );
    controller.send_command(SimCommand::Pause).unwrap();
    drop(controller);
    tick_microtasks().await;
}

#[cfg(feature = "physics")]
fn make_quad_pane_controllers() -> [SimulationController; 4] {
    use crate::async_sim::unified::{UnifiedModel, UnifiedSimRunner};
    use crate::tabs::chaos::fractals::{FractalMode, FractalRenderRequest, FractalRunner};
    use crate::tabs::fluid_dynamics::lattice_boltzmann::LbmUnified;
    use crate::tabs::solid_state::ising::{IsingRunner, IsingSharedParams};
    use math_explorer::physics::stat_mech::ising::SpinLattice;
    use num_complex::Complex;
    use std::collections::HashMap;
    use std::sync::atomic::AtomicU64;

    let gen = Arc::new(AtomicU64::new(1));
    let req = Arc::new(RwLock::new(FractalRenderRequest {
        generation: 0,
        width: 64,
        height: 48,
        center: Complex::new(-0.5, 0.0),
        zoom: 2.0,
        max_iter: 1200,
        mode: FractalMode::Mandelbrot,
        julia_c: Complex::new(0.0, 0.0),
        high_contrast: false,
    }));
    let fractal = FractalRunner::with_formula(
        Arc::clone(&req),
        Arc::clone(&gen),
        Arc::new(RwLock::new(None)),
    );
    if let Ok(mut g) = req.write() {
        g.generation = 1;
        g.width = 2048;
        g.height = 2048;
    }
    let ising = IsingRunner {
        lattice: SpinLattice::new(64, 64, Some(11)),
        params: Arc::new(RwLock::new(IsingSharedParams {
            steps_per_frame: 50_000,
            ..IsingSharedParams::default()
        })),
    };
    let lbm_params: HashMap<String, f64> = LbmUnified::parameters()
        .into_iter()
        .map(|(k, v)| (k, (v.min + v.max) * 0.5))
        .collect();
    let lbm_a = UnifiedSimRunner::<LbmUnified>::new(Arc::new(RwLock::new(lbm_params.clone())));
    let lbm_b = UnifiedSimRunner::<LbmUnified>::new(Arc::new(RwLock::new(lbm_params)));
    [
        SimulationController::new(fractal),
        SimulationController::new(ising),
        SimulationController::new(lbm_a),
        SimulationController::new(lbm_b),
    ]
}

#[cfg(feature = "physics")]
#[wasm_bindgen_test]
async fn wasm_quad_pane_workers_share_8ms_frame_slice_budget() {
    let mut controllers = make_quad_pane_controllers();
    for (idx, c) in controllers.iter_mut().enumerate() {
        c.set_suspended(idx > 0);
        c.send_command(SimCommand::Start).unwrap();
    }
    assert_eq!(wasm_scheduler::test_active_running_count(), 1);
    await_animation_frame().await;
    let single_elapsed = wasm_scheduler::test_current_epoch_elapsed();
    assert!(single_elapsed >= std::time::Duration::ZERO);
    #[cfg(not(debug_assertions))]
    assert!(
        single_elapsed <= std::time::Duration::from_millis(8),
        "Single-pane 2048x2048 FractalRunner exceeded 8ms slice: {single_elapsed:?}",
    );

    controllers[1].set_suspended(false);
    assert_eq!(wasm_scheduler::test_active_running_count(), 2);
    await_animation_frame().await;
    let split_elapsed = wasm_scheduler::test_current_epoch_elapsed();
    assert!(split_elapsed >= std::time::Duration::ZERO);
    #[cfg(not(debug_assertions))]
    assert!(
        split_elapsed <= std::time::Duration::from_millis(8),
        "Split-pane FractalRunner + IsingRunner exceeded 8ms slice: {split_elapsed:?}",
    );

    controllers[2].set_suspended(false);
    controllers[3].set_suspended(false);
    assert_eq!(wasm_scheduler::test_active_running_count(), 4);
    for c in &controllers {
        assert!(
            wasm_scheduler::test_worker_slice_budget(c.handle)
                <= std::time::Duration::from_millis(2)
        );
    }
    await_animation_frame().await;
    for c in &mut controllers {
        let _ = c.update();
    }
    await_animation_frame().await;
    let quad_elapsed = wasm_scheduler::test_current_epoch_elapsed();
    assert!(quad_elapsed >= std::time::Duration::ZERO);
    #[cfg(not(debug_assertions))]
    assert!(
        quad_elapsed <= std::time::Duration::from_millis(8),
        "QuadGrid 4-pane workers exceeded shared 8ms slice: {quad_elapsed:?}",
    );

    for c in &mut controllers {
        let _ = c.update();
        let _ = c.send_command(SimCommand::Pause);
    }
    drop(controllers);
    tick_microtasks().await;
    assert_eq!(wasm_scheduler::test_active_running_count(), 0);
}

#[wasm_bindgen_test]
async fn wasm_suspended_worker_runs_zero_ticks_and_resumes_running() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let steps = Arc::new(AtomicUsize::new(0));
    let snapshots = Arc::new(AtomicUsize::new(0));
    let mut controller = SimulationController::new(WasmCountingRunner {
        steps: Arc::clone(&steps),
        snapshots,
    });
    controller.send_command(SimCommand::Start).unwrap();
    await_animation_frame().await;
    assert!(steps.load(Ordering::SeqCst) >= 1);

    controller.set_suspended(true);
    assert_eq!(controller.state(), SimWorkerState::Running);
    await_animation_frame().await;
    let frozen = steps.load(Ordering::SeqCst);
    await_animation_frame().await;
    await_animation_frame().await;
    assert_eq!(steps.load(Ordering::SeqCst), frozen);

    controller.set_suspended(false);
    await_animation_frame().await;
    await_animation_frame().await;
    let resumed = steps.load(Ordering::SeqCst);
    assert!(resumed > frozen);

    // Pause, wait through an idle frame gap (>= 16 ms), then restart: worker ticks immediately with 8 ms budget.
    controller.send_command(SimCommand::Pause).unwrap();
    for _ in 0..3 {
        await_animation_frame().await;
    }
    let before_restart = steps.load(Ordering::SeqCst);
    assert_eq!(
        wasm_scheduler::test_worker_slice_budget(controller.handle),
        std::time::Duration::from_millis(8)
    );
    controller.send_command(SimCommand::Start).unwrap();
    wasm_bindgen_futures::JsFuture::from(js_sys::Promise::resolve(&wasm_bindgen::JsValue::NULL))
        .await
        .unwrap();
    assert_eq!(steps.load(Ordering::SeqCst), before_restart + 1);
    controller.send_command(SimCommand::Pause).unwrap();
    drop(controller);
    tick_microtasks().await;
}

#[cfg(all(feature = "ai", feature = "biology", feature = "physics"))]
#[wasm_bindgen_test]
async fn wasm_all_ten_simulation_controller_tools_pace_once_per_raf_and_suspend() {
    use crate::framework::InteractiveTool;
    use crate::tabs::ai::reward_plots::RewardPlotsTool;
    use crate::tabs::chaos::attractors::AttractorPlotter;
    use crate::tabs::chaos::bifurcation::BifurcationDiagram;
    use crate::tabs::chaos::fractals::FractalViewer;
    use crate::tabs::morphogenesis::MorphogenesisTool;
    use crate::tabs::mri::{MriKSpaceTool, MriTool};
    use crate::tabs::optics_waves::OpticsWaveTool;
    use crate::tabs::quantum::wave_sim::WaveSimulatorTool;
    use crate::tabs::solid_state::ising::IsingModelTool;

    let mut fractal = FractalViewer::default();
    let mut bifurc = BifurcationDiagram::default();
    let mut attract = AttractorPlotter::default();
    let mut ising = IsingModelTool::default();
    let mut mri = MriTool::default();
    let mut kspace = MriKSpaceTool::default();
    let mut morpho = MorphogenesisTool::default();
    let mut wave = WaveSimulatorTool::default();
    let mut optics = OpticsWaveTool::default();
    let mut rewards = RewardPlotsTool::default();

    let _ = fractal.worker.send_command(SimCommand::Start);
    let _ = bifurc.worker.send_command(SimCommand::Start);
    let _ = attract.worker.send_command(SimCommand::Start);
    let _ = ising.worker.send_command(SimCommand::Start);
    let _ = mri.worker.send_command(SimCommand::Start);
    let _ = kspace.worker.send_command(SimCommand::Start);
    let _ = morpho.worker.send_command(SimCommand::Start);
    let _ = wave.worker.send_command(SimCommand::Start);
    let _ = optics.worker.send_command(SimCommand::Start);
    let _ = rewards
        .controller
        .send_command(SimCommand::TrainEpisodes(50));

    assert_eq!(wasm_scheduler::test_active_running_count(), 10);
    await_animation_frame().await;

    fractal.set_suspended(true);
    bifurc.set_suspended(true);
    attract.set_suspended(true);
    ising.set_suspended(true);
    mri.set_suspended(true);
    kspace.set_suspended(true);
    morpho.set_suspended(true);
    wave.set_suspended(true);
    optics.set_suspended(true);
    rewards.set_suspended(true);
    assert_eq!(wasm_scheduler::test_active_running_count(), 0);
}
