//! Keyboard-accessible sculpting UI, fairing stamps, and `UnifiedUiExtension` for [`super::lattice_boltzmann::LbmUnified`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::lbm_fairing::{
    apply_stroke_to_fairing_mask, decode_mask_hex, encode_fairing_mask_rle, encode_mask_hex,
    restore_inlet_and_core_mask, FairingStamp, LbmFairingEvaluator, LbmSerializedState,
    CHALLENGE_LBM_FAIRING, CHALLENGE_LBM_VISCOSITY, LBM_CELL_COUNT, LBM_CORE_RADIUS, LBM_CORE_X,
    LBM_CORE_Y, LBM_HEIGHT, LBM_INLET_COLS, LBM_MIN_SOLID_CELLS, LBM_WIDTH,
};
use crate::accessibility::{
    announce_live_status, lock_canvas_arrow_keys, paint_canvas_focus_ring,
    register_canvas_accessibility, AccessibleHoverText,
};
use crate::async_sim::unified::UnifiedUiExtension;
use crate::async_sim::{SculptStatePayload, SimCommand, StateSnapshot};
use crate::framework::{BrushStroke, CalloutManager, ChallengeEvaluation};
use eframe::egui;
use egui_plot::{Points, Polygon};
use math_commons::math_kernel::types::flatten_2d_index;
use std::collections::HashMap;

/// UI-side sculpting controller and challenge adapter for `UnifiedSimTool<LbmUnified>`.
pub struct LbmUiExtension {
    /// Current `100 x 50` obstacle bitmask in lattice coordinates `(x, y)`.
    pub mask: [bool; LBM_CELL_COUNT],
    /// Canonical bluff-cylinder mask for fast default-state comparison.
    default_mask: [bool; LBM_CELL_COUNT],
    /// Active challenge ID when launched in inverse challenge mode.
    pub active_challenge: Option<&'static str>,
    /// Preallocated zero-allocation drag evaluator.
    pub evaluator: LbmFairingEvaluator,
    /// Keyboard sculpting cursor X coordinate (`2..100`).
    pub cursor_x: i32,
    /// Keyboard sculpting cursor Y coordinate (`0..50`).
    pub cursor_y: i32,
    /// Monotonic edit counter mixed into `state_revision()`.
    revision_counter: u64,
}

impl Default for LbmUiExtension {
    fn default() -> Self {
        let mut mask = [false; LBM_CELL_COUNT];
        FairingStamp::BluffCylinder.write_mask(&mut mask);
        Self {
            mask,
            default_mask: mask,
            active_challenge: None,
            evaluator: LbmFairingEvaluator::default(),
            cursor_x: LBM_CORE_X + LBM_CORE_RADIUS + 2,
            cursor_y: LBM_CORE_Y,
            revision_counter: 1,
        }
    }
}

impl LbmUiExtension {
    /// Applies a built-in [`FairingStamp`], re-evaluates drag, and returns worker sync commands.
    pub fn apply_stamp(&mut self, stamp: FairingStamp, viscosity: f64) -> Vec<SimCommand> {
        stamp.write_mask(&mut self.mask);
        self.evaluator.evaluate_mask_in_place(&self.mask, viscosity);
        self.revision_counter = self.revision_counter.wrapping_add(1);
        vec![SimCommand::Custom(stamp.command_name().to_string())]
    }

    /// Nudges the keyboard sculpting cursor `(dx, dy)` within valid non-inlet grid bounds.
    pub fn nudge_cursor(&mut self, dx: i32, dy: i32) -> (i32, i32) {
        self.cursor_x = (self.cursor_x + dx).clamp(LBM_INLET_COLS as i32, LBM_WIDTH as i32 - 1);
        self.cursor_y = (self.cursor_y + dy).clamp(0, LBM_HEIGHT as i32 - 1);
        (self.cursor_x, self.cursor_y)
    }

    /// Toggles the obstacle cell under `(cursor_x, cursor_y)` while protecting the `R_core = 4` core
    /// when `lbm_low_drag_fairing` is active. Returns `Ok(new_is_solid)` or `Err(reason)`.
    pub fn toggle_cursor_cell(
        &mut self,
        viscosity: f64,
    ) -> Result<(bool, Vec<SimCommand>), &'static str> {
        let x = self.cursor_x as usize;
        let y = self.cursor_y as usize;
        if x < LBM_INLET_COLS {
            return Err("Inlet columns x < 2 are locked open for uniform inflow");
        }
        let dx = self.cursor_x - LBM_CORE_X;
        let dy = self.cursor_y - LBM_CORE_Y;
        let in_core = dx * dx + dy * dy <= LBM_CORE_RADIUS * LBM_CORE_RADIUS;
        let idx = flatten_2d_index(x, y, LBM_WIDTH);
        if in_core && self.mask[idx] {
            return Err("Protected core cell (R_core = 4) cannot be erased");
        }
        let next_solid = !self.mask[idx];
        self.mask[idx] = next_solid;
        self.evaluator.evaluate_mask_in_place(&self.mask, viscosity);
        self.revision_counter = self.revision_counter.wrapping_add(1);
        let cmd = SimCommand::ApplyBrush {
            cx: self.cursor_x,
            cy: self.cursor_y,
            r: 0,
            is_obstacle: next_solid,
        };
        Ok((next_solid, vec![cmd, SimCommand::Reset]))
    }

    fn sync_mask_command(&self) -> SimCommand {
        SimCommand::Custom(format!("MaskHex:{}", encode_mask_hex(&self.mask)))
    }

    fn poll_sculpt_key_input(ctx: &egui::Context) -> ((i32, i32), bool) {
        ctx.input(|i| {
            let dx = i32::from(i.key_pressed(egui::Key::ArrowRight))
                - i32::from(i.key_pressed(egui::Key::ArrowLeft));
            let dy = i32::from(i.key_pressed(egui::Key::ArrowUp))
                - i32::from(i.key_pressed(egui::Key::ArrowDown));
            let toggle = i.key_pressed(egui::Key::Enter) || i.key_pressed(egui::Key::Space);
            ((dx, dy), toggle)
        })
    }

    fn handle_sculpt_keys_ctx(
        &mut self,
        ctx: &egui::Context,
        resp: &egui::Response,
    ) -> Vec<SimCommand> {
        if !resp.has_focus() {
            return Vec::new();
        }
        if resp.gained_focus() {
            ctx.request_repaint();
        }
        let filter = egui::EventFilter {
            tab: false,
            horizontal_arrows: true,
            vertical_arrows: true,
            escape: false,
        };
        ctx.memory_mut(|m| m.set_focus_lock_filter(resp.id, filter));
        let (delta, toggle_pressed) = Self::poll_sculpt_key_input(ctx);
        let mut cmds = Vec::new();
        if delta != (0, 0) {
            let (cx, cy) = self.nudge_cursor(delta.0, delta.1);
            let word = if self.mask[flatten_2d_index(cx as usize, cy as usize, LBM_WIDTH)] {
                "solid"
            } else {
                "fluid"
            };
            announce_live_status(ctx, format!("Sculpt cursor moved to ({cx}, {cy}), {word}"));
        }
        if toggle_pressed {
            match self.toggle_cursor_cell(CHALLENGE_LBM_VISCOSITY) {
                Ok((is_solid, toggle_cmds)) => {
                    let word = if is_solid { "solid" } else { "fluid" };
                    let m = &self.evaluator.latest;
                    let msg = format!(
                        "Cell ({}, {}) set to {word}; C_D = {:.3}, N_solid = {}",
                        self.cursor_x, self.cursor_y, m.c_d, m.n_solid
                    );
                    announce_live_status(ctx, msg);
                    cmds.extend(toggle_cmds);
                }
                Err(reason) => announce_live_status(ctx, reason),
            }
        }
        cmds
    }

    fn handle_sculpt_keys(&mut self, ui: &egui::Ui, resp: &egui::Response) -> Vec<SimCommand> {
        lock_canvas_arrow_keys(ui, resp);
        paint_canvas_focus_ring(ui, resp);
        self.handle_sculpt_keys_ctx(ui.ctx(), resp)
    }

    fn draw_stamp_buttons(&mut self, ui: &mut egui::Ui) -> Vec<SimCommand> {
        let mut cmds = Vec::new();
        ui.label("Fairing Stamps (Keyboard Accessible):");
        ui.horizontal_wrapped(|ui| {
            let stamps = [
                FairingStamp::BluffCylinder,
                FairingStamp::TeardropFairing,
                FairingStamp::SymmetricWedge,
            ];
            for stamp in stamps {
                let tip = format!("Apply {} obstacle profile and reset LBM", stamp.label());
                if ui
                    .button(stamp.label())
                    .accessible_hover_text(tip)
                    .clicked()
                {
                    cmds = self.apply_stamp(stamp, CHALLENGE_LBM_VISCOSITY);
                    let m = &self.evaluator.latest;
                    let pct = 100.0 * m.c_d / m.c_d_cylinder.max(1e-9);
                    let msg = format!(
                        "Applied {}: C_D = {:.3} ({pct:.1}% of cylinder), N_solid = {}",
                        stamp.label(),
                        m.c_d,
                        m.n_solid
                    );
                    announce_live_status(ui.ctx(), msg);
                }
            }
        });
        cmds
    }

    fn draw_keyboard_pad_and_forces(
        &mut self,
        ui: &mut egui::Ui,
        latest_snapshot: Option<&StateSnapshot>,
    ) -> Vec<SimCommand> {
        let (cx, cy) = (self.cursor_x, self.cursor_y);
        let pad_label = format!("⌨ Sculpt Cursor ({cx}, {cy}) — Focus & use Arrow keys + Enter");
        let pad_resp = ui.add(egui::Button::new(pad_label).sense(egui::Sense::click()));
        let coord = format!("({cx}, {cy})");
        register_canvas_accessibility(
            ui,
            &pad_resp,
            egui::accesskit::Role::Application,
            "LBM Fairing Keyboard Sculpt Cursor",
            Some(&coord),
        );
        if pad_resp.clicked() {
            pad_resp.request_focus();
        }
        let cmds = self.handle_sculpt_keys(ui, &pad_resp);
        let m = &self.evaluator.latest;
        ui.small(format!(
            "Fairing Readout: F_D = {:.4}, F_L = {:.4}, C_D = {:.3} (Cyl = {:.3})",
            m.f_drag, m.f_lift, m.c_d, m.c_d_cylinder
        ));
        if let Some(s) =
            latest_snapshot.filter(|s| s.custom_data.len() >= 6 && s.custom_data[5] >= 20.0)
        {
            ui.small(format!(
                "Live Stream: F_D = {:.4}, F_L = {:.4}, C_D = {:.3}",
                s.custom_data[0], s.custom_data[1], s.custom_data[2]
            ));
        }
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        let geom_ok = m.n_solid >= LBM_MIN_SOLID_CELLS && m.core_intact;
        let geom_color = if geom_ok {
            palette.success
        } else {
            palette.warning
        };
        let core_txt = if m.core_intact { "Intact" } else { "Breached" };
        let msg = format!(
            "Solid Cells: {} (min {LBM_MIN_SOLID_CELLS}) | Core R=4: {core_txt}",
            m.n_solid
        );
        ui.colored_label(geom_color, msg);
        cmds
    }
}

impl UnifiedUiExtension for LbmUiExtension {
    fn save_extra_state(&self, params: &HashMap<String, f64>) -> Option<String> {
        self.active_challenge?;
        let viscosity = *params.get("viscosity").unwrap_or(&CHALLENGE_LBM_VISCOSITY);
        serde_json::to_string(&LbmSerializedState {
            viscosity,
            mask_hex: Some(encode_mask_hex(&self.mask)),
        })
        .ok()
    }

    fn try_load_extra_state(
        &mut self,
        state: &str,
        params: &mut HashMap<String, f64>,
    ) -> Option<Result<Vec<SimCommand>, String>> {
        let has_mask_hex = serde_json::from_str::<serde_json::Value>(state)
            .ok()
            .is_some_and(|v| v.get("mask_hex").is_some());
        if self.active_challenge.is_none() && !has_mask_hex {
            return None;
        }
        let parsed: LbmSerializedState = match serde_json::from_str(state) {
            Ok(s) => s,
            Err(e) => return Some(Err(format!("Invalid LBM state JSON: {e}"))),
        };
        if !parsed.viscosity.is_finite() || !(0.00333..=0.5).contains(&parsed.viscosity) {
            return Some(Err(format!(
                "LBM viscosity out of [0.00333, 0.5]: {}",
                parsed.viscosity
            )));
        }
        let mut next_mask = self.default_mask;
        if let Some(hex) = &parsed.mask_hex {
            if let Err(err) = decode_mask_hex(hex, &mut next_mask) {
                return Some(Err(err));
            }
        }
        for x in 0..LBM_INLET_COLS {
            for y in 0..LBM_HEIGHT {
                next_mask[flatten_2d_index(x, y, LBM_WIDTH)] = false;
            }
        }
        let visc = if self.active_challenge == Some(CHALLENGE_LBM_FAIRING) {
            CHALLENGE_LBM_VISCOSITY
        } else {
            parsed.viscosity
        };
        params.insert("viscosity".to_string(), visc);
        let mask_changed = self.mask != next_mask;
        self.mask = next_mask;
        if mask_changed || self.active_challenge.is_some() {
            self.evaluator.evaluate_mask_in_place(&self.mask, visc);
        }
        self.revision_counter = self.revision_counter.wrapping_add(1);
        Some(Ok(vec![self.sync_mask_command()]))
    }

    fn apply_challenge(
        &mut self,
        challenge_id: Option<&str>,
        params: &mut HashMap<String, f64>,
    ) -> Result<Vec<SimCommand>, String> {
        match challenge_id {
            None => {
                self.active_challenge = None;
                Ok(vec![SimCommand::Custom("ChallengeLockCore:0".to_string())])
            }
            Some(CHALLENGE_LBM_FAIRING) => {
                self.active_challenge = Some(CHALLENGE_LBM_FAIRING);
                params.insert("viscosity".to_string(), CHALLENGE_LBM_VISCOSITY);
                params.insert("naca_profile".to_string(), 0.0);
                params.insert("angle_of_attack".to_string(), 0.0);
                restore_inlet_and_core_mask(&mut self.mask);
                self.evaluator
                    .evaluate_mask_in_place(&self.mask, CHALLENGE_LBM_VISCOSITY);
                self.revision_counter = self.revision_counter.wrapping_add(1);
                Ok(vec![
                    SimCommand::Custom("ChallengeLockCore:1".to_string()),
                    self.sync_mask_command(),
                ])
            }
            Some(other) => Err(format!(
                "Unsupported challenge '{other}' for LatticeBoltzmannTool"
            )),
        }
    }

    fn active_challenge_id(&self) -> Option<&'static str> {
        self.active_challenge
    }

    fn evaluate_challenge(
        &self,
        _latest_snapshot: Option<&StateSnapshot>,
    ) -> Option<ChallengeEvaluation> {
        self.active_challenge
            .map(|_| self.evaluator.challenge_evaluation())
    }

    fn is_parameter_locked(&self, param_name: &str) -> bool {
        self.active_challenge == Some(CHALLENGE_LBM_FAIRING)
            && matches!(param_name, "viscosity" | "naca_profile" | "angle_of_attack")
    }

    fn on_reset(&mut self, params: &HashMap<String, f64>) {
        let visc = *params.get("viscosity").unwrap_or(&CHALLENGE_LBM_VISCOSITY);
        let prev = self.mask;
        restore_inlet_and_core_mask(&mut self.mask);
        if self.mask != prev || self.active_challenge.is_some() {
            self.evaluator.evaluate_mask_in_place(&self.mask, visc);
        }
        self.revision_counter = self.revision_counter.wrapping_add(1);
    }

    fn on_brush(&mut self, cx: i32, cy: i32, r: i32, is_obstacle: bool) -> bool {
        let mut any_changed = false;
        for y in (cy - r)..=(cy + r) {
            for x in (cx - r)..=(cx + r) {
                if x >= LBM_INLET_COLS as i32
                    && x < LBM_WIDTH as i32
                    && y >= 0
                    && y < LBM_HEIGHT as i32
                    && (x - cx) * (x - cx) + (y - cy) * (y - cy) <= r * r
                {
                    let dx = x - LBM_CORE_X;
                    let dy = y - LBM_CORE_Y;
                    if !is_obstacle
                        && self.active_challenge.is_some()
                        && dx * dx + dy * dy <= LBM_CORE_RADIUS * LBM_CORE_RADIUS
                    {
                        continue;
                    }
                    let idx = flatten_2d_index(x as usize, y as usize, LBM_WIDTH);
                    if self.mask[idx] != is_obstacle {
                        self.mask[idx] = is_obstacle;
                        any_changed = true;
                    }
                }
            }
        }
        if any_changed {
            self.evaluator
                .evaluate_mask_in_place(&self.mask, CHALLENGE_LBM_VISCOSITY);
            self.revision_counter = self.revision_counter.wrapping_add(1);
        }
        any_changed
    }

    fn on_brush_stroke(&mut self, stroke: &BrushStroke) -> bool {
        if apply_stroke_to_fairing_mask(&mut self.mask, stroke, self.active_challenge.is_some()) {
            self.evaluator
                .evaluate_mask_in_place(&self.mask, CHALLENGE_LBM_VISCOSITY);
            self.revision_counter = self.revision_counter.wrapping_add(1);
        }
        true
    }

    fn sync_sculpt_payload(&self, sculpt: &mut SculptStatePayload) {
        if self.active_challenge.is_none() {
            sculpt.rle_mask = if self.mask == self.default_mask {
                Vec::new()
            } else {
                encode_fairing_mask_rle(&self.mask)
            };
        }
    }

    fn on_custom_action(&mut self, action: &str) {
        if action == "Clear Obstacles" {
            self.mask.fill(false);
            if self.active_challenge.is_some() {
                restore_inlet_and_core_mask(&mut self.mask);
            }
            self.evaluator
                .evaluate_mask_in_place(&self.mask, CHALLENGE_LBM_VISCOSITY);
            self.revision_counter = self.revision_counter.wrapping_add(1);
        }
    }

    fn draw_side_panel_controls(
        &mut self,
        ui: &mut egui::Ui,
        latest_snapshot: Option<&StateSnapshot>,
    ) -> Vec<SimCommand> {
        self.evaluator.ensure_baseline();
        let mut cmds = Vec::new();
        ui.group(|ui| {
            cmds.extend(self.draw_stamp_buttons(ui));
            ui.separator();
            cmds.extend(self.draw_keyboard_pad_and_forces(ui, latest_snapshot));
        });
        ui.separator();
        cmds
    }

    fn handle_canvas_overlay(
        &mut self,
        ctx: &egui::Context,
        visuals: &egui::Visuals,
        plot_ui: &mut egui_plot::PlotUi,
    ) -> Vec<SimCommand> {
        let palette = crate::app::SemanticPalette::from_visuals(visuals);
        let ring_pts: Vec<[f64; 2]> = (0..24)
            .map(|k| {
                let theta = std::f64::consts::TAU * (k as f64) / 24.0;
                [
                    LBM_CORE_X as f64 + 0.5 + (LBM_CORE_RADIUS as f64) * theta.cos(),
                    LBM_CORE_Y as f64 + 0.5 + (LBM_CORE_RADIUS as f64) * theta.sin(),
                ]
            })
            .collect();
        plot_ui.polygon(
            Polygon::new("Protected Core (R=4)", ring_pts)
                .fill_color(egui::Color32::TRANSPARENT)
                .stroke(egui::Stroke::new(1.5_f32, palette.warning)),
        );
        plot_ui.points(
            Points::new(
                "Sculpt Cursor",
                vec![[self.cursor_x as f64 + 0.5, self.cursor_y as f64 + 0.5]],
            )
            .radius(5.0_f32)
            .color(palette.accent),
        );
        let resp = plot_ui.response().clone();
        self.handle_sculpt_keys_ctx(ctx, &resp)
    }

    fn sync_callouts(&self, _snap: Option<&StateSnapshot>, callouts: &mut CalloutManager) {
        let m = &self.evaluator.latest;
        let core_pos = [LBM_CORE_X as f64 + 0.5, LBM_CORE_Y as f64 + 0.5];
        let core_lbl = format!("Core R=4 | N={}", m.n_solid);
        callouts.set_auto_callout("lbm_core", core_pos, core_lbl, [-36.0, -26.0]);
        let drag_pos = [LBM_CORE_X as f64 + 14.0, LBM_CORE_Y as f64 + 10.0];
        let drag_lbl = format!("C_D = {:.3} (≤ {:.3})", m.c_d, 0.82 * m.c_d_cylinder);
        callouts.set_auto_callout("lbm_drag", drag_pos, drag_lbl, [28.0, -16.0]);
    }

    fn state_revision_salt(&self) -> u64 {
        self.revision_counter
    }
}

// [cite:dwarf_galaxy_empirical_dependencies]
