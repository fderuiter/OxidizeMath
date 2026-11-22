#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::async_sim::unified::UnifiedSimTool;
use crate::framework::recovery::DiagnosticReport;
use crate::framework::InteractiveTool;
use crate::tabs::ai::q_table_inspector::QTableInspectorTool;
use crate::tabs::ai::reward_plots::RewardPlotsTool;
use crate::tabs::chaos::attractors::AttractorPlotter;
use crate::tabs::fluid_dynamics::lattice_boltzmann::LbmUnified;
use crate::tabs::mri::MriTool;
use crate::tabs::neuroscience::hodgkin_huxley::HodgkinHuxleyUnified;
use crate::tabs::quantum::wave_sim::WaveUnified;
use crate::tabs::solid_state::ising::IsingModelTool;
use eframe::egui;
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(not(target_arch = "wasm32"))]
use std::time::Duration;

struct DivergeOnStepRunner {
    diverge_next: Arc<AtomicBool>,
    diverged: bool,
    panic_next: Arc<AtomicBool>,
    value: f64,
}

impl DivergeOnStepRunner {
    fn new(diverge_next: Arc<AtomicBool>, panic_next: Arc<AtomicBool>) -> Self {
        Self {
            diverge_next,
            diverged: false,
            panic_next,
            value: 1.0,
        }
    }
}

impl SimulationRunner for DivergeOnStepRunner {
    fn process_command(&mut self, cmd: SimCommand) {
        match cmd {
            SimCommand::Reset => {
                self.diverged = false;
                self.value = 1.0;
            }
            SimCommand::Custom(payload) => {
                if let Ok(v) = payload.parse::<f64>() {
                    self.value = v;
                }
            }
            _ => {}
        }
    }

    fn step(&mut self) {
        if self.panic_next.load(Ordering::SeqCst) {
            panic!("diverged worker subsequently panicked");
        }
        if self.diverge_next.load(Ordering::SeqCst) {
            self.diverged = true;
            self.value = f64::NAN;
        } else {
            self.value += 1.0;
        }
    }

    fn get_snapshot(&self) -> StateSnapshot {
        StateSnapshot {
            width: 1,
            height: 1,
            pixels: Arc::new(std::sync::RwLock::new(vec![egui::Color32::WHITE])),
            custom_data: vec![if self.value.is_finite() {
                self.value
            } else {
                0.0
            }],
            structured_data: None,
        }
    }

    fn get_steps_per_frame(&self) -> usize {
        1
    }

    fn is_finite_state(&self) -> bool {
        !self.diverged
    }

    fn diagnostic_report(&self) -> Option<DiagnosticReport> {
        self.diverged
            .then(|| DiagnosticReport::new(0.1, 1, "value", "Runner diverged"))
    }
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_live_diverged_worker_resets_in_place_and_clears_diagnostic_report() {
    let diverge_flag = Arc::new(AtomicBool::new(true));
    let panic_flag = Arc::new(AtomicBool::new(false));
    let mut controller = SimulationController::new(DivergeOnStepRunner::new(
        Arc::clone(&diverge_flag),
        Arc::clone(&panic_flag),
    ));
    let _ = controller.update();

    assert!(controller.send_command(SimCommand::Start).is_ok());
    for _ in 0..400 {
        std::thread::sleep(Duration::from_millis(10));
        let _ = controller.update();
        if controller.state() == SimWorkerState::Faulted {
            break;
        }
    }

    assert_eq!(controller.state(), SimWorkerState::Faulted);
    assert!(controller.diagnostic_report().is_some());
    assert!(controller.send_command(SimCommand::Start).is_err());
    assert!(controller.send_command(SimCommand::Pause).is_err());
    assert!(controller
        .send_command(SimCommand::TrainEpisodes(5))
        .is_err());

    diverge_flag.store(false, Ordering::SeqCst);
    let respawned = controller.reset_or_respawn(|| -> DivergeOnStepRunner {
        panic!("live worker should reset in place")
    });
    assert!(!respawned);
    assert_eq!(controller.state(), SimWorkerState::Ready);
    assert!(controller.diagnostic_report().is_none());

    for _ in 0..400 {
        std::thread::sleep(Duration::from_millis(10));
        if controller.update().is_some() {
            break;
        }
    }
    assert_eq!(controller.latest_snapshot().unwrap().custom_data[0], 1.0);
}

#[test]
fn test_disconnected_faulted_worker_preserves_diagnostic_until_respawn() {
    let diverge_flag = Arc::new(AtomicBool::new(true));
    let panic_flag = Arc::new(AtomicBool::new(false));
    let runner = DivergeOnStepRunner::new(Arc::clone(&diverge_flag), Arc::clone(&panic_flag));
    let (cmd_tx, cmd_rx) = mpsc::channel::<SimCommand>();
    let (state_tx, state_rx) = sim_state_channel();
    let seed = runner.get_snapshot();
    drop(cmd_rx);
    drop(state_tx);

    let mut controller = SimulationController {
        cmd_tx,
        state_rx,
        latest_snapshot: Some(seed),
        timeline_metadata: None,
        pending_snapshot: false,
        awaiting_pause_ack: false,
        state: SimWorkerState::Faulted,
        diagnostic_report: Some(DiagnosticReport::new(0.2, 2, "x", "Diverged before drop")),
        #[cfg(target_arch = "wasm32")]
        handle: 0,
    };

    assert!(matches!(
        controller.send_command(SimCommand::Reset),
        Err(SimCommandError::ChannelDisconnected)
    ));
    assert!(controller.diagnostic_report().is_some());

    diverge_flag.store(false, Ordering::SeqCst);
    let respawned = controller.reset_or_respawn(|| {
        DivergeOnStepRunner::new(Arc::clone(&diverge_flag), Arc::clone(&panic_flag))
    });
    assert!(respawned);
    assert_eq!(controller.state(), SimWorkerState::Ready);
    assert!(controller.diagnostic_report().is_none());
}

#[test]
fn test_unified_tools_render_non_empty_panels_on_frame_zero() {
    let ctx = egui::Context::default();

    let mut wave = UnifiedSimTool::<WaveUnified>::new();
    if let Some(snap) = wave.controller.latest_snapshot() {
        if let Some(wdata) = snap
            .structured_data
            .as_ref()
            .and_then(|d| d.downcast_ref::<crate::tabs::quantum::wave_sim::WaveData>())
        {
            let psi_pts: Vec<[f64; 2]> = wdata
                .prob_density
                .iter()
                .enumerate()
                .map(|(i, &p)| [i as f64, p])
                .collect();
            let pot_pts: Vec<[f64; 2]> = wdata
                .potential
                .iter()
                .enumerate()
                .map(|(i, &v)| [i as f64, v * 0.2])
                .collect();
            let datasets = vec![
                ("Seeded |psi(x, 0)|^2".to_string(), psi_pts),
                ("Scaled V(x)".to_string(), pot_pts),
            ];
            let opts = egui_plot::export::ExportOptions::default();
            if let Ok(png) = egui_plot::export::export_custom_png(&datasets, &opts) {
                let _ = std::fs::write("/tmp/seeded_wavepacket_frame_zero.png", png);
            }
        }
    }
    let wave_out = ctx.run(egui::RawInput::default(), |ctx| {
        wave.show(ctx);
    });
    assert!(!wave_out.shapes.is_empty());

    let mut hh = UnifiedSimTool::<HodgkinHuxleyUnified>::new();
    let hh_out = ctx.run(egui::RawInput::default(), |ctx| {
        hh.show(ctx);
    });
    assert!(!hh_out.shapes.is_empty());

    let mut lbm = UnifiedSimTool::<LbmUnified>::new();
    let lbm_out = ctx.run(egui::RawInput::default(), |ctx| {
        lbm.show(ctx);
    });
    assert!(!lbm_out.shapes.is_empty());
    assert!(!lbm_out.textures_delta.set.is_empty());
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_lbm_paused_apply_brush_emits_fresh_snapshot() {
    let mut lbm = UnifiedSimTool::<LbmUnified>::new();
    let _ = lbm.controller.update();
    assert_eq!(lbm.state(), SimWorkerState::Ready);

    assert!(lbm
        .controller
        .send_command(SimCommand::ApplyBrush {
            cx: 80,
            cy: 10,
            r: 3,
            is_obstacle: true,
        })
        .is_ok());

    let mut updated = false;
    for _ in 0..400 {
        std::thread::sleep(Duration::from_millis(10));
        if lbm.controller.update().is_some() {
            updated = true;
            break;
        }
    }
    assert!(updated, "Paused ApplyBrush must emit a fresh snapshot");

    let snap = lbm.controller.latest_snapshot().expect("snapshot present");
    let pixels = snap.pixels.read().expect("pixels lock");
    let flipped_y = snap.height - 1 - 10;
    let obstacle_idx = flipped_y * snap.width + 80;
    assert_eq!(pixels[obstacle_idx], egui::Color32::from_gray(50));
    let rgb_flat: Vec<[u8; 3]> = pixels.iter().map(|c| [c.r(), c.g(), c.b()]).collect();
    if let Ok(json) = serde_json::to_string(&(snap.width, snap.height, rgb_flat)) {
        let _ = std::fs::write("/tmp/seeded_lbm_pixels.json", json);
    }
}

#[test]
fn test_mri_and_ising_and_qtable_seed_without_duplicate_models() {
    let ctx = egui::Context::default();

    let mut mri = MriTool::default();
    let mri_out = ctx.run(egui::RawInput::default(), |ctx| {
        mri.show_ui(ctx);
    });
    assert_eq!(mri.worker.state(), SimWorkerState::Ready);
    assert!(!mri.mx_points.is_empty());
    assert!(!mri_out.shapes.is_empty());

    let mut ising = IsingModelTool::default();
    let ising_out = ctx.run(egui::RawInput::default(), |ctx| {
        ising.show(ctx);
    });
    assert!(!ising_out.textures_delta.set.is_empty());

    let mut qtable = QTableInspectorTool::default();
    let qtable_out = ctx.run(egui::RawInput::default(), |ctx| {
        qtable.show(ctx);
    });
    assert!(!qtable_out.shapes.is_empty());
}

#[test]
fn test_fault_banners_and_reset_across_rl_and_attractor_tools() {
    let ctx = egui::Context::default();

    let mut reward = RewardPlotsTool::default();
    let _ = reward.controller.update();
    reward
        .controller
        .set_state_for_test(SimWorkerState::Faulted);
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        reward.show(ctx);
    });
    assert_eq!(reward.controller.state(), SimWorkerState::Faulted);
    reward.reset();
    assert_eq!(reward.controller.state(), SimWorkerState::Ready);

    let mut qtable = QTableInspectorTool::default();
    let _ = qtable.worker.update();
    qtable.worker.set_state_for_test(SimWorkerState::Faulted);
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        qtable.show(ctx);
    });
    assert_eq!(qtable.worker.state(), SimWorkerState::Faulted);
    qtable.reset();
    assert_eq!(qtable.worker.state(), SimWorkerState::Ready);

    let mut attractor = AttractorPlotter::default();
    let _ = attractor.worker.update();
    attractor.worker.set_state_for_test(SimWorkerState::Faulted);
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        attractor.show(ctx);
    });
    assert_eq!(attractor.worker.state(), SimWorkerState::Faulted);
    attractor.reset();
    assert_eq!(attractor.worker.state(), SimWorkerState::Ready);
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_unpolled_running_worker_diverges_without_blocking_sender() {
    let diverge_flag = Arc::new(AtomicBool::new(false));
    let panic_flag = Arc::new(AtomicBool::new(false));
    let mut controller = SimulationController::new(DivergeOnStepRunner::new(
        Arc::clone(&diverge_flag),
        Arc::clone(&panic_flag),
    ));

    // Start running and intentionally do not call update() while > 8 snapshots accumulate.
    assert!(controller.send_command(SimCommand::Start).is_ok());
    std::thread::sleep(Duration::from_millis(220));
    diverge_flag.store(true, Ordering::SeqCst);

    for _ in 0..100 {
        std::thread::sleep(Duration::from_millis(10));
        let _ = controller.update();
        if controller.state() == SimWorkerState::Faulted {
            break;
        }
    }

    assert_eq!(controller.state(), SimWorkerState::Faulted);
    assert!(controller.diagnostic_report().is_some());
}
