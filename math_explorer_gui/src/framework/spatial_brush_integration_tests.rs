//! Multi-domain GUI integration tests verifying `InteractiveTool::on_brush`,
//! pixel-to-solver Y-coordinate alignment, physics/biology invariants, and
//! compact URL share-link round-trips (`< 4 KiB` JSON) across all four tools.

#![cfg_attr(any(), verified(opt_out = "tests"))]

use crate::async_sim::unified::{UnifiedModel, UnifiedSimTool};
use crate::async_sim::SimCommand;
use crate::framework::recovery::FiniteStateGuard;
use crate::framework::{
    BrushMode, BrushStroke, InteractionContext, InteractiveTool, SimulationFramework,
};
use crate::tabs::fluid_dynamics::lattice_boltzmann::{
    LbmSnapshotMetrics, LbmUnified, LBM_HEIGHT, LBM_WIDTH,
};
use crate::tabs::fluid_dynamics::lattice_boltzmann_sculpt::{
    export_solver_rle, stamp_airfoil_on_solver,
};
use crate::tabs::fluid_dynamics::potential_flow::PotentialFlowTool;
use crate::tabs::morphogenesis::MorphogenesisTool;
use crate::tabs::quantum::wave_sim::{WaveData, WaveUnified};
use crate::url_state::{encode_share_fragment, parse_share_location, MAX_DECODED_URL_STATE_BYTES};
use crate::web_io::PlatformInstant;
use eframe::egui;
use math_explorer::physics::fluid_dynamics::lattice_boltzmann::{
    BgkCollision, LatticeBoltzmannD2Q9,
};
use std::time::Duration;

const WORKER_POLL_TIMEOUT: Duration = Duration::from_secs(20);

fn poll_until(timeout: Duration, mut predicate: impl FnMut() -> bool) {
    let start = PlatformInstant::now();
    loop {
        if predicate() || start.elapsed() >= timeout {
            return;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
}

fn make_stroke(start: [f64; 2], end: [f64; 2], radius: f64, mode: BrushMode) -> BrushStroke {
    BrushStroke {
        start,
        end,
        radius,
        mode,
        drag_vector: [end[0] - start[0], end[1] - start[1]],
    }
}

fn send_brush_stroke(tool: &mut dyn InteractiveTool, stroke: BrushStroke) {
    let ctx = egui::Context::default();
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            let (_, resp) = ui.allocate_exact_size(egui::vec2(200.0, 100.0), egui::Sense::click());
            let ic = InteractionContext::from_brush_stroke(&resp, stroke);
            tool.on_brush(&ic);
        });
    });
}

fn wait_for_unified_snapshot<M: UnifiedModel>(tool: &mut UnifiedSimTool<M>) {
    poll_until(WORKER_POLL_TIMEOUT, || {
        let _ = tool.controller.update();
        tool.controller.latest_snapshot().is_some()
    });
}

fn wait_for_morpho_snapshot(tool: &mut MorphogenesisTool) {
    poll_until(WORKER_POLL_TIMEOUT, || {
        let _ = tool.worker.update();
        tool.worker.latest_snapshot().is_some()
    });
}

#[test]
fn test_lbm_on_brush_y_alignment() {
    let mut tool = UnifiedSimTool::<LbmUnified>::new();
    wait_for_unified_snapshot(&mut tool);

    let _ = tool.controller.send_command(SimCommand::ClearObstacles);
    poll_until(WORKER_POLL_TIMEOUT, || {
        let _ = tool.controller.update();
        tool.export_raster_image()
            .is_some_and(|img| img.pixels[25 * LBM_WIDTH + 30] != egui::Color32::from_gray(50))
    });

    send_brush_stroke(
        &mut tool,
        make_stroke([20.5, 8.5], [32.5, 14.5], 1.0, BrushMode::Solid),
    );
    poll_until(WORKER_POLL_TIMEOUT, || {
        let _ = tool.controller.update();
        tool.export_raster_image()
            .is_some_and(|img| img.pixels[41 * LBM_WIDTH + 20] == egui::Color32::from_gray(50))
    });

    let img = tool.export_raster_image().expect("LBM raster image");
    assert_eq!(img.size, [LBM_WIDTH, LBM_HEIGHT]);
    let bottom_px = img.pixels[41 * LBM_WIDTH + 20];
    let mirrored_top_px = img.pixels[8 * LBM_WIDTH + 20];
    assert_eq!(bottom_px, egui::Color32::from_gray(50));
    assert_ne!(mirrored_top_px, egui::Color32::from_gray(50));
}

#[test]
fn test_lbm_naca_vorticity_and_url_roundtrip() {
    let mut tool = UnifiedSimTool::<LbmUnified>::new();
    wait_for_unified_snapshot(&mut tool);
    {
        let mut p = tool.params.write().unwrap_or_else(|e| e.into_inner());
        p.insert("naca_profile".to_string(), 2412.0);
        p.insert("angle_of_attack".to_string(), 14.0);
        p.insert("field_view".to_string(), 1.0);
    }
    {
        let mut temp = LatticeBoltzmannD2Q9::<BgkCollision>::new(LBM_WIDTH, LBM_HEIGHT, 0.56);
        stamp_airfoil_on_solver(&mut temp, 2412, 14.0);
        tool.sculpt_state
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .rle_mask = export_solver_rle(&temp);
    }
    let _ = tool
        .controller
        .send_command(SimCommand::Custom("stamp_naca:2412:14.00".to_string()));
    let _ = tool.controller.send_command(SimCommand::Step(10));
    poll_until(WORKER_POLL_TIMEOUT, || {
        let _ = tool.controller.update();
        tool.controller
            .latest_snapshot()
            .and_then(|s| {
                s.structured_data
                    .as_ref()?
                    .downcast_ref::<LbmSnapshotMetrics>()
            })
            .is_some_and(|m| m.max_vorticity > 0.0)
    });

    let snap = tool.controller.latest_snapshot().expect("LBM snapshot");
    let metrics = snap
        .structured_data
        .as_ref()
        .and_then(|d| d.downcast_ref::<LbmSnapshotMetrics>())
        .expect("LbmSnapshotMetrics");
    assert!(metrics.drag_coeff.is_finite() && metrics.lift_coeff.is_finite());
    assert!(metrics.max_vorticity > 0.0);

    let saved = tool.save_state().expect("LBM save_state");
    assert!(saved.len() < MAX_DECODED_URL_STATE_BYTES);
    let fragment =
        encode_share_fragment("fluid_dynamics", Some("LatticeBoltzmann2D"), Some(&saved))
            .expect("encode LBM share URL");
    let decoded = parse_share_location(&fragment, "")
        .expect("parse LBM share URL")
        .expect("ShareUrlParams");
    let mut restored = UnifiedSimTool::<LbmUnified>::new();
    restored
        .try_load_state(decoded.state_json.as_deref().unwrap())
        .expect("restore LBM state");
    assert_eq!(restored.save_state().unwrap(), saved);
}

#[test]
fn test_morphogenesis_on_brush_y_alignment_maze_and_url_roundtrip() {
    let mut tool = MorphogenesisTool::default();
    wait_for_morpho_snapshot(&mut tool);

    // Brush a diagonal ZeroFluxWall near plot_y = 12.5 on a 100x100 grid -> texture row = 100 - 1 - 12 = 87.
    send_brush_stroke(
        &mut tool,
        make_stroke([15.5, 12.5], [35.5, 28.5], 1.0, BrushMode::ZeroFluxWall),
    );
    let _ = tool.worker.send_command(SimCommand::Step(2));
    poll_until(WORKER_POLL_TIMEOUT, || {
        let _ = tool.worker.update();
        tool.export_raster_image()
            .is_some_and(|img| img.pixels[87 * 100 + 15] == egui::Color32::from_gray(55))
    });

    let img = tool.export_raster_image().expect("Morpho raster image");
    assert_eq!(img.size, [100, 100]);
    assert_eq!(
        img.pixels[87 * 100 + 15],
        egui::Color32::from_gray(55),
        "Morphogenesis wall at plot_y = 12.5 must align with visual row 87"
    );
    assert!(FiniteStateGuard::is_finite_state(&tool));

    // Stamp microfluidic serpentine maze and verify URL share round-trip within 4 KiB
    *tool.mask_rle.write().unwrap_or_else(|e| e.into_inner()) =
        crate::tabs::morphogenesis::sculpt::build_maze_rle(tool.width, tool.height);
    let _ = tool
        .worker
        .send_command(SimCommand::Custom("stamp_maze".to_string()));
    let saved = tool.save_state().expect("Morphogenesis save_state");
    assert!(
        saved.len() < MAX_DECODED_URL_STATE_BYTES,
        "Morphogenesis maze JSON ({} bytes) must fit inside 4 KiB URL cap",
        saved.len()
    );
    let fragment = encode_share_fragment("morphogenesis", Some("MorphogenesisTool"), Some(&saved))
        .expect("encode Morphogenesis share URL");
    let decoded = parse_share_location(&fragment, "")
        .expect("parse Morphogenesis share URL")
        .expect("ShareUrlParams");
    let mut restored = MorphogenesisTool::default();
    restored
        .try_load_state(decoded.state_json.as_deref().unwrap())
        .expect("restore Morphogenesis state");
    assert_eq!(
        *restored.mask_rle.read().unwrap(),
        *tool.mask_rle.read().unwrap()
    );
}

#[test]
fn test_wave_on_brush_unitarity_and_url_roundtrip() {
    let mut wave = UnifiedSimTool::<WaveUnified>::new();
    wait_for_unified_snapshot(&mut wave);

    send_brush_stroke(
        &mut wave,
        make_stroke([-0.5, 2.0], [-0.2, 2.0], 0.2, BrushMode::PotentialBarrier),
    );
    send_brush_stroke(
        &mut wave,
        make_stroke([0.2, 2.0], [0.5, 2.0], 0.2, BrushMode::PotentialBarrier),
    );
    let _ = wave.controller.send_command(SimCommand::Step(25));
    poll_until(WORKER_POLL_TIMEOUT, || {
        let _ = wave.controller.update();
        wave.controller
            .latest_snapshot()
            .and_then(|s| s.structured_data.as_ref()?.downcast_ref::<WaveData>())
            .is_some_and(|w| w.time > 0.0 && w.potential.iter().any(|&v| v > 4.0))
    });

    let snap = wave.controller.latest_snapshot().expect("Wave snapshot");
    let wdata = snap
        .structured_data
        .as_ref()
        .and_then(|d| d.downcast_ref::<WaveData>())
        .expect("WaveData");
    assert!((wdata.norm_sq - 1.0).abs() < 1e-6);
    assert!(wdata.potential.iter().any(|&v| v > 4.0));

    let wave_saved = wave.save_state().expect("Wave save_state");
    assert!(wave_saved.len() < MAX_DECODED_URL_STATE_BYTES);
    let wave_frag = encode_share_fragment("quantum", Some("WaveSimulator"), Some(&wave_saved))
        .expect("encode Wave share URL");
    let wave_dec = parse_share_location(&wave_frag, "")
        .expect("parse Wave share URL")
        .expect("ShareUrlParams");
    let mut wave_restored = UnifiedSimTool::<WaveUnified>::new();
    wave_restored
        .try_load_state(wave_dec.state_json.as_deref().unwrap())
        .expect("restore Wave state");
    assert_eq!(wave_restored.save_state().unwrap(), wave_saved);
}

#[test]
fn test_potential_flow_on_brush_and_url_roundtrip() {
    let mut pflow = PotentialFlowTool::default();
    let initial_len = pflow.elements.len();
    send_brush_stroke(
        &mut pflow,
        make_stroke([-1.5, 1.2], [-1.5, 1.2], 0.35, BrushMode::FlowVortex),
    );
    assert_eq!(pflow.elements.len(), initial_len + 1);

    send_brush_stroke(
        &mut pflow,
        make_stroke([-1.45, 1.22], [-1.0, 1.4], 0.35, BrushMode::FlowVortex),
    );
    assert_eq!(pflow.elements.len(), initial_len + 1);

    let pflow_saved = pflow.save_state().expect("PotentialFlow save_state");
    let pflow_frag =
        encode_share_fragment("fluid_dynamics", Some("PotentialFlow"), Some(&pflow_saved))
            .expect("encode PotentialFlow share URL");
    let pflow_dec = parse_share_location(&pflow_frag, "")
        .expect("parse PotentialFlow share URL")
        .expect("ShareUrlParams");
    let mut pflow_restored = PotentialFlowTool::default();
    pflow_restored
        .try_load_state(pflow_dec.state_json.as_deref().unwrap())
        .expect("restore PotentialFlow state");
    assert_eq!(pflow_restored.elements.len(), initial_len + 1);

    send_brush_stroke(
        &mut pflow,
        make_stroke([-1.0, 1.4], [-1.0, 1.4], 0.4, BrushMode::Erase),
    );
    assert_eq!(pflow.elements.len(), initial_len);
}

#[test]
fn test_lbm_live_aoa_rotation_and_discrete_apply_brush() {
    let mut params = std::collections::HashMap::new();
    params.insert("viscosity".to_string(), 0.02);
    params.insert("naca_profile".to_string(), 2412.0);
    params.insert("angle_of_attack".to_string(), 0.0);
    let mut lbm = LbmUnified::new(&params);
    lbm.solver.set_obstacle(80, 10, true);
    for _ in 0..8 {
        lbm.step(&params);
    }
    let lift_zero = lbm.latest_aero.lift_coeff;

    params.insert("angle_of_attack".to_string(), 18.0);
    for _ in 0..16 {
        lbm.step(&params);
    }
    assert_eq!(lbm.stamped_airfoil, Some((2412, 18.0)));
    assert!(
        lbm.solver.is_obstacle(80, 10),
        "Live AoA rotation must preserve hand-painted walls outside the airfoil footprint"
    );
    assert!((lbm.latest_aero.lift_coeff - lift_zero).abs() > 1e-3);

    let mut wave_params = std::collections::HashMap::new();
    wave_params.insert("potential_type".to_string(), 0.0);
    wave_params.insert("dt".to_string(), 0.02);
    let mut wave = WaveUnified::new(&wave_params);
    wave.process_command(
        SimCommand::ApplyBrush {
            cx: 100,
            cy: 0,
            r: 3,
            is_obstacle: true,
        },
        &wave_params,
    );
    assert!((wave.potential[100] - 16.0).abs() < 1e-9);
    wave.process_command(
        SimCommand::ApplyBrush {
            cx: 100,
            cy: 0,
            r: 3,
            is_obstacle: false,
        },
        &wave_params,
    );
    assert!(wave.potential[100].abs() < 1e-9);
}

#[test]
fn test_simulation_framework_encode_and_import_share_url_all_domains() {
    let cases = [
        (
            "fluid_dynamics",
            "LatticeBoltzmannTool",
            make_stroke([22.0, 15.0], [40.0, 22.0], 2.0, BrushMode::Solid),
        ),
        (
            "biology",
            "MorphogenesisTool",
            make_stroke([20.0, 20.0], [65.0, 50.0], 2.0, BrushMode::ZeroFluxWall),
        ),
        (
            "quantum",
            "WaveSimulator",
            make_stroke([-0.4, 2.4], [0.4, 2.4], 0.2, BrushMode::PotentialBarrier),
        ),
    ];

    for (domain, tool_id, stroke) in cases {
        let mut fw = SimulationFramework::new(domain);
        fw.import_url_preset_json(Some(tool_id), None)
            .expect("select target tool");
        send_brush_stroke(fw.active_tool.as_mut().unwrap().as_mut(), stroke);
        let expected_state = fw.active_tool.as_ref().unwrap().save_state().unwrap();

        let full_url = fw.encode_share_url().expect("encode_share_url");
        let hash_idx = full_url.find('#').expect("URL hash fragment");
        let parsed = parse_share_location(&full_url[hash_idx..], "")
            .expect("parse_share_location")
            .expect("ShareUrlParams");
        let preset_json = parsed.state_json.as_deref().expect("state_json");
        assert!(
            preset_json.len() <= MAX_DECODED_URL_STATE_BYTES,
            "{domain}/{tool_id} PresetWrapper JSON {} bytes exceeded 4 KiB cap",
            preset_json.len()
        );

        let mut target_fw = SimulationFramework::new(domain);
        target_fw
            .import_url_preset_json(parsed.tool.as_deref(), Some(preset_json))
            .expect("import_url_preset_json");
        assert_eq!(
            target_fw
                .active_tool
                .as_ref()
                .unwrap()
                .save_state()
                .unwrap(),
            expected_state
        );
    }
}
