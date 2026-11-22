//! Spatial brush controls, NACA 4-digit airfoil & Venturi stamps, RLE mask sync,
//! and aerodynamic readout widgets for [`super::lattice_boltzmann::LbmUnified`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::lattice_boltzmann::{LbmSnapshotMetrics, LbmUnified, LBM_HEIGHT, LBM_WIDTH};
use crate::async_sim::{SculptStatePayload, SimCommand, StateSnapshot};
use crate::framework::{
    decode_rle_mask, encode_rle_mask, rasterize_disk_stroke, BrushInteractionMode, BrushMode,
    BrushStroke, SpatialBrushKit,
};
use eframe::egui;
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::physics::fluid_dynamics::lattice_boltzmann::{
    inject_velocity_jet, stamp_naca4_airfoil, stamp_venturi_nozzle, BgkCollision,
    LatticeBoltzmannD2Q9, Naca4Airfoil,
};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

pub(crate) fn build_default_obstacle_bytes(params: &HashMap<String, f64>) -> Vec<u8> {
    let mut temp = LatticeBoltzmannD2Q9::<BgkCollision>::new(LBM_WIDTH, LBM_HEIGHT, 0.56);
    LbmUnified::seed_initial_geometry(&mut temp, params);
    let mut out = vec![0_u8; LBM_WIDTH * LBM_HEIGHT];
    for y in 0..LBM_HEIGHT {
        for x in 0..LBM_WIDTH {
            if temp.is_obstacle(x, y) {
                out[flatten_2d_index(x, y, LBM_WIDTH)] = 1;
            }
        }
    }
    out
}

pub(crate) fn apply_stroke_to_lbm(solver: &mut LatticeBoltzmannD2Q9<BgkCollision>, stroke: &BrushStroke) {
    let w = solver.width();
    let h = solver.height();
    let (x0, y0) = SpatialBrushKit::plot_to_grid_y_up(stroke.start, w, h);
    let (x1, y1) = SpatialBrushKit::plot_to_grid_y_up(stroke.end, w, h);
    let r = (stroke.radius.round() as i32).clamp(0, 16);

    match stroke.mode {
        BrushMode::Solid => {
            rasterize_disk_stroke((x0, y0), (x1, y1), r, (w, h), |gx, gy| {
                solver.set_obstacle(gx, gy, true);
            });
        }
        BrushMode::Erase => {
            rasterize_disk_stroke((x0, y0), (x1, y1), r, (w, h), |gx, gy| {
                solver.set_obstacle(gx, gy, false);
            });
        }
        BrushMode::VelocityJet => {
            let jx = (stroke.drag_vector[0] * 0.05).clamp(-0.16, 0.16);
            let jy = (stroke.drag_vector[1] * 0.05).clamp(-0.16, 0.16);
            let (ux, uy) = if jx.hypot(jy) < 1e-6 {
                (0.14, 0.0)
            } else {
                (jx, jy)
            };
            for (cx, cy) in crate::framework::bresenham_line(x0, y0, x1, y1) {
                inject_velocity_jet(solver, cx, cy, r.max(1), ux, uy);
            }
        }
        _ => {}
    }
}

pub(crate) fn record_stroke_in_rle(
    stroke: &BrushStroke,
    params: &HashMap<String, f64>,
    sculpt_state: &mut SculptStatePayload,
) {
    if !matches!(stroke.mode, BrushMode::Solid | BrushMode::Erase) {
        return;
    }
    let total = LBM_WIDTH * LBM_HEIGHT;
    let mut mask = if sculpt_state.rle_mask.is_empty() {
        build_default_obstacle_bytes(params)
    } else {
        decode_rle_mask(&sculpt_state.rle_mask, total)
            .unwrap_or_else(|_| build_default_obstacle_bytes(params))
    };

    let (x0, y0) = SpatialBrushKit::plot_to_grid_y_up(stroke.start, LBM_WIDTH, LBM_HEIGHT);
    let (x1, y1) = SpatialBrushKit::plot_to_grid_y_up(stroke.end, LBM_WIDTH, LBM_HEIGHT);
    let r = (stroke.radius.round() as i32).clamp(0, 16);
    let val = u8::from(stroke.mode == BrushMode::Solid);

    rasterize_disk_stroke(
        (x0, y0),
        (x1, y1),
        r,
        (LBM_WIDTH, LBM_HEIGHT),
        |gx, gy| {
            mask[flatten_2d_index(gx, gy, LBM_WIDTH)] = val;
        },
    );
    sculpt_state.rle_mask = encode_rle_mask(&mask);
}

pub(crate) fn export_solver_rle(solver: &LatticeBoltzmannD2Q9<BgkCollision>) -> Vec<[u32; 2]> {
    let w = solver.width();
    let h = solver.height();
    let mut bytes = vec![0_u8; w * h];
    for y in 0..h {
        for x in 0..w {
            if solver.is_obstacle(x, y) {
                bytes[flatten_2d_index(x, y, w)] = 1;
            }
        }
    }
    encode_rle_mask(&bytes)
}

pub(crate) fn stamp_airfoil_on_solver(
    solver: &mut LatticeBoltzmannD2Q9<BgkCollision>,
    code: u16,
    aoa_deg: f64,
) {
    solver.clear_obstacles();
    let airfoil = Naca4Airfoil::from_code(
        code,
        26.0,
        0.32 * solver.width() as f64,
        0.5 * solver.height() as f64,
        aoa_deg,
    );
    stamp_naca4_airfoil(solver, &airfoil);
    solver.set_inlet(
        0,
        0,
        solver.width(),
        solver.height(),
        super::lattice_boltzmann::LBM_INLET_U,
        0.0,
    );
}

pub(crate) fn rotate_airfoil_in_place(
    solver: &mut LatticeBoltzmannD2Q9<BgkCollision>,
    code: u16,
    old_aoa_deg: f64,
    new_aoa_deg: f64,
) {
    let w = solver.width();
    let h = solver.height();
    let cx = 0.32 * w as f64;
    let cy = 0.5 * h as f64;
    let old_af = Naca4Airfoil::from_code(code, 26.0, cx, cy, old_aoa_deg);
    let new_af = Naca4Airfoil::from_code(code, 26.0, cx, cy, new_aoa_deg);
    let u_in = super::lattice_boltzmann::LBM_INLET_U;

    for y in 0..h {
        for x in 0..w {
            let px = x as f64 + 0.5;
            let py = y as f64 + 0.5;
            if new_af.contains_point(px, py) {
                solver.set_obstacle(x, y, true);
            } else if old_af.contains_point(px, py) {
                solver.set_obstacle(x, y, false);
                solver.set_inlet(x, y, 1, 1, u_in, 0.0);
            }
        }
    }
    solver.set_inlet(0, 0, 2, h, u_in, 0.0);
}

fn sync_live_aoa_slider(
    ui: &egui::Ui,
    raw_code: f64,
    code: u16,
    aoa: f64,
    cmds: &mut Vec<SimCommand>,
) {
    let aoa_id = ui.id().with("lbm_last_aoa");
    if raw_code < 1.0 || !aoa.is_finite() {
        ui.ctx().data_mut(|d| {
            d.remove_temp::<f64>(aoa_id);
        });
        return;
    }
    let prev = ui.ctx().data_mut(|d| {
        let old = d.get_temp::<f64>(aoa_id);
        d.insert_temp(aoa_id, aoa);
        old
    });
    if let Some(prev_aoa) = prev {
        if (aoa - prev_aoa).abs() > 1e-6 {
            cmds.push(SimCommand::Custom(format!("rotate_naca:{code}:{aoa:.2}")));
        }
    }
}

pub(crate) fn draw_lbm_brush_panel(
    ui: &mut egui::Ui,
    brush_kit: &mut SpatialBrushKit,
    params: &Arc<RwLock<HashMap<String, f64>>>,
    sculpt_state: &Arc<RwLock<SculptStatePayload>>,
    snapshot: Option<&StateSnapshot>,
) -> Vec<SimCommand> {
    let mut cmds = Vec::new();
    ui.label("Wind-Tunnel Sculpting");

    ui.horizontal(|ui| {
        ui.selectable_value(
            &mut brush_kit.interaction_mode,
            BrushInteractionMode::Sculpt,
            "🖌 Sculpt",
        );
        ui.selectable_value(
            &mut brush_kit.interaction_mode,
            BrushInteractionMode::Pan,
            "✋ Pan",
        );
    });

    ui.horizontal(|ui| {
        ui.selectable_value(&mut brush_kit.mode, BrushMode::Solid, "Solid");
        ui.selectable_value(&mut brush_kit.mode, BrushMode::Erase, "Erase");
        ui.selectable_value(&mut brush_kit.mode, BrushMode::VelocityJet, "Jet");
    });

    ui.add(egui::Slider::new(&mut brush_kit.radius, 1.0..=8.0).text("Brush Radius"));

    let (raw_code, code, aoa) = {
        let p = params.read().unwrap_or_else(|e| e.into_inner());
        let rc = *p.get("naca_profile").unwrap_or(&2412.0);
        let code_u16 = if rc < 1.0 { 2412 } else { rc.round() as u16 };
        (rc, code_u16, *p.get("angle_of_attack").unwrap_or(&0.0))
    };
    sync_live_aoa_slider(ui, raw_code, code, aoa, &mut cmds);
    draw_lbm_stamp_buttons(ui, params, sculpt_state, raw_code, code, aoa, &mut cmds);
    draw_aerodynamic_metrics(ui, snapshot);
    cmds
}

fn draw_lbm_stamp_buttons(
    ui: &mut egui::Ui,
    params: &Arc<RwLock<HashMap<String, f64>>>,
    sculpt_state: &Arc<RwLock<SculptStatePayload>>,
    raw_code: f64,
    code: u16,
    aoa: f64,
    cmds: &mut Vec<SimCommand>,
) {
    ui.horizontal_wrapped(|ui| {
        if ui.button("✈ Stamp NACA Airfoil").clicked() {
            if raw_code < 1.0 {
                params
                    .write()
                    .unwrap_or_else(|e| e.into_inner())
                    .insert("naca_profile".to_string(), f64::from(code));
            }
            ui.ctx()
                .data_mut(|d| d.insert_temp(ui.id().with("lbm_last_aoa"), aoa));
            let mut temp = LatticeBoltzmannD2Q9::<BgkCollision>::new(LBM_WIDTH, LBM_HEIGHT, 0.56);
            stamp_airfoil_on_solver(&mut temp, code, aoa);
            sculpt_state
                .write()
                .unwrap_or_else(|e| e.into_inner())
                .rle_mask = export_solver_rle(&temp);
            cmds.push(SimCommand::Custom(format!("stamp_naca:{code}:{aoa:.2}")));
        }
        if ui.button("⏳ Stamp Venturi").clicked() {
            params
                .write()
                .unwrap_or_else(|e| e.into_inner())
                .insert("naca_profile".to_string(), 0.0);
            let mut temp = LatticeBoltzmannD2Q9::<BgkCollision>::new(LBM_WIDTH, LBM_HEIGHT, 0.56);
            stamp_venturi_nozzle(&mut temp, 0.44);
            sculpt_state
                .write()
                .unwrap_or_else(|e| e.into_inner())
                .rle_mask = export_solver_rle(&temp);
            cmds.push(SimCommand::Custom("stamp_venturi".to_string()));
        }
    });
}

fn draw_aerodynamic_metrics(ui: &mut egui::Ui, snapshot: Option<&StateSnapshot>) {
    let Some(metrics) = snapshot
        .and_then(|s| s.structured_data.as_ref())
        .and_then(|d| d.downcast_ref::<LbmSnapshotMetrics>())
    else {
        return;
    };
    let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
    ui.separator();
    ui.label("Aerodynamic Diagnostics");
    let ld_ratio = if metrics.drag_coeff.abs() > 1e-6 {
        metrics.lift_coeff / metrics.drag_coeff
    } else {
        0.0
    };
    ui.colored_label(
        palette.info,
        format!(
            "Lift C_L: {:+.3}   Drag C_D: {:+.3}",
            metrics.lift_coeff, metrics.drag_coeff
        ),
    );
    ui.colored_label(
        palette.muted_text,
        format!(
            "L/D Ratio: {:+.2}   Max |ω_z|: {:.4}",
            ld_ratio, metrics.max_vorticity
        ),
    );
}

fn parse_naca_action(action: &str, params: &HashMap<String, f64>) -> (u16, f64) {
    let mut parts = action.split(':');
    let _ = parts.next();
    let code = parts
        .next()
        .and_then(|s| s.parse::<u16>().ok())
        .unwrap_or(2412);
    let aoa = parts
        .next()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or_else(|| *params.get("angle_of_attack").unwrap_or(&0.0));
    (code, aoa)
}

pub(crate) fn apply_lbm_custom_action(
    solver: &mut LatticeBoltzmannD2Q9<BgkCollision>,
    stamped_airfoil: &mut Option<(u16, f64)>,
    action: &str,
    params: &HashMap<String, f64>,
) -> bool {
    if action == "Clear Obstacles" {
        solver.clear_obstacles();
        *stamped_airfoil = None;
        return true;
    }
    if action == "stamp_venturi" {
        solver.clear_obstacles();
        stamp_venturi_nozzle(solver, 0.44);
        solver.set_inlet(
            0,
            0,
            solver.width(),
            solver.height(),
            super::lattice_boltzmann::LBM_INLET_U,
            0.0,
        );
        *stamped_airfoil = None;
        return true;
    }
    if action.starts_with("stamp_naca:") {
        let (code, aoa) = parse_naca_action(action, params);
        stamp_airfoil_on_solver(solver, code, aoa);
        *stamped_airfoil = Some((code, aoa));
        return true;
    }
    if action.starts_with("rotate_naca:") {
        let (code, aoa) = parse_naca_action(action, params);
        match *stamped_airfoil {
            Some((prev_code, prev_aoa)) if prev_code == code => {
                rotate_airfoil_in_place(solver, code, prev_aoa, aoa);
            }
            _ => stamp_airfoil_on_solver(solver, code, aoa),
        }
        *stamped_airfoil = Some((code, aoa));
        return true;
    }
    false
}
