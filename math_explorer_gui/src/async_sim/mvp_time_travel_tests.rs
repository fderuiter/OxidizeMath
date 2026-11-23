use super::*;
use crate::async_sim::unified::UnifiedSimTool;
use crate::framework::{InteractiveTool, MAX_TOOL_STATE_BYTES};
use crate::tabs::chaos::attractors::AttractorPlotter;
use crate::tabs::morphogenesis::MorphogenesisTool;
use crate::tabs::mri::MriTool;
use crate::tabs::neuroscience::hodgkin_huxley::HodgkinHuxleyUnified;
use crate::tabs::quantum::wave_sim::WaveUnified;
use eframe::egui;
use std::time::Duration;

fn wait_for_step(controller: &mut SimulationController, expected_step: usize) {
    for _ in 0..400 {
        let _ = controller.update();
        if controller.state() == SimWorkerState::Ready
            && controller
                .timeline_metadata()
                .is_some_and(|m| m.current_step == expected_step)
        {
            return;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    panic!(
        "Timed out waiting for worker Ready at step {expected_step}; got state={} meta={:?}",
        controller.state(),
        controller.timeline_metadata()
    );
}

fn wait_for_tool_step<T: InteractiveTool>(
    ctx: &egui::Context,
    tool: &mut T,
    check: impl Fn(&T) -> bool,
) {
    for _ in 0..400 {
        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| tool.show_inside(ui));
        });
        if check(tool) {
            return;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    panic!("Timed out waiting for tool UI to settle at target step");
}

#[test]
fn test_attractor_and_mri_tools_end_to_end_step_back_and_scrub() {
    let ctx = egui::Context::default();
    let mut attractor = AttractorPlotter::default();
    assert_eq!(
        attractor.worker.timeline_metadata(),
        Some(TimelineMetadata {
            current_step: 0,
            min_step: 0,
            max_step: 0,
            stride: 1,
        })
    );
    attractor.worker.send_command(SimCommand::Step(12)).unwrap();
    wait_for_tool_step(&ctx, &mut attractor, |t| {
        t.worker.state() == SimWorkerState::Ready && t.step_count == 12
    });

    attractor
        .worker
        .send_command(SimCommand::StepBack(1))
        .unwrap();
    wait_for_tool_step(&ctx, &mut attractor, |t| {
        t.worker.state() == SimWorkerState::Ready && t.step_count == 11
    });

    attractor
        .worker
        .send_command(SimCommand::SeekStep(4))
        .unwrap();
    wait_for_tool_step(&ctx, &mut attractor, |t| {
        t.worker.state() == SimWorkerState::Ready && t.step_count == 4
    });
    assert_eq!(attractor.step_count, 4);
    assert_eq!(attractor.history.len(), 4);

    let mut mri = MriTool::default();
    mri.params.write().unwrap().b_field = nalgebra::Vector3::new(0.0, 0.0, 1.0e-7);
    mri.worker.send_command(SimCommand::Reset).unwrap();
    wait_for_tool_step(&ctx, &mut mri, |t| {
        t.worker.state() == SimWorkerState::Ready
    });
    mri.worker.send_command(SimCommand::Step(10)).unwrap();
    wait_for_tool_step(&ctx, &mut mri, |t| {
        t.worker.state() == SimWorkerState::Ready
            && t.worker
                .timeline_metadata()
                .is_some_and(|m| m.current_step == 10)
    });
    mri.worker.send_command(SimCommand::SeekStep(3)).unwrap();
    wait_for_tool_step(&ctx, &mut mri, |t| {
        t.worker.state() == SimWorkerState::Ready && (t.current_time - 0.003).abs() < 1e-12
    });
    assert!((mri.current_time - 0.003).abs() < 1e-12);
    assert_eq!(mri.history.len(), 4);
}

#[test]
fn test_hodgkin_huxley_vs_non_opted_in_unified_and_morphogenesis_k16() {
    let mut hh = UnifiedSimTool::<HodgkinHuxleyUnified>::new();
    assert_eq!(hh.controller.timeline_metadata().unwrap().stride, 1);
    hh.controller.send_command(SimCommand::Step(15)).unwrap();
    wait_for_step(hh.controller.controller_mut(), 15);
    hh.controller.send_command(SimCommand::StepBack(1)).unwrap();
    wait_for_step(hh.controller.controller_mut(), 14);
    hh.controller.send_command(SimCommand::SeekStep(6)).unwrap();
    wait_for_step(hh.controller.controller_mut(), 6);

    let wave = UnifiedSimTool::<WaveUnified>::new();
    assert!(
        wave.controller.timeline_metadata().is_none(),
        "Non-opted-in UnifiedModel must have timeline_metadata() == None"
    );

    let mut morpho = MorphogenesisTool {
        pending_initial_start: false,
        ..Default::default()
    };
    {
        let mut p = morpho.params.write().unwrap();
        p.d_v = 50.0;
        p.dt = 0.002;
    }
    assert_eq!(morpho.worker.timeline_metadata().unwrap().stride, 16);
    morpho.worker.send_command(SimCommand::Step(35)).unwrap();
    wait_for_step(morpho.worker.controller_mut(), 35);

    morpho
        .worker
        .send_command(SimCommand::SeekStep(20))
        .unwrap();
    wait_for_step(morpho.worker.controller_mut(), 20);
    assert_eq!(morpho.worker.timeline_metadata().unwrap().max_step, 35);

    morpho.reset();
    wait_for_step(morpho.worker.controller_mut(), 0);
    assert_eq!(morpho.worker.timeline_metadata().unwrap().max_step, 0);
}

#[test]
fn test_preset_save_state_and_ctrl_z_remain_parameter_only_after_scrubbing() {
    let mut morpho = MorphogenesisTool {
        pending_initial_start: false,
        ..Default::default()
    };
    {
        let mut p = morpho.params.write().unwrap();
        p.d_v = 50.0;
        p.dt = 0.002;
    }
    morpho.worker.send_command(SimCommand::Step(32)).unwrap();
    wait_for_step(morpho.worker.controller_mut(), 32);
    morpho
        .worker
        .send_command(SimCommand::SeekStep(18))
        .unwrap();
    wait_for_step(morpho.worker.controller_mut(), 18);

    let saved = morpho.save_state().expect("Morphogenesis save_state");
    assert!(
        saved.len() < 512 && saved.len() <= MAX_TOOL_STATE_BYTES,
        "Saved state must only serialize scalar parameters, got {} bytes",
        saved.len()
    );
    assert!(!saved.contains("keyframes") && !saved.contains("checkpoint"));

    let app = crate::MathExplorerApp::default();
    assert!(!app.session_manager.can_undo());
    let ctrl_z = egui::RawInput {
        events: vec![egui::Event::Key {
            key: egui::Key::Z,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers {
                ctrl: true,
                command: true,
                ..Default::default()
            },
        }],
        ..Default::default()
    };
    let ctx = egui::Context::default();
    let _ = ctx.run(ctrl_z, |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| morpho.show_inside(ui));
    });
    assert_eq!(
        morpho.worker.timeline_metadata().unwrap().current_step,
        18,
        "Ctrl+Z must never trigger playback step-back or scrub"
    );
}
