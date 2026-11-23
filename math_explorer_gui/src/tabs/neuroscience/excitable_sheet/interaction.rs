#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::state::{BrushMode, LeadCoord, PropagationMode, ScenarioPreset};
use super::ExcitableSheetTool;
use crate::framework::InteractionContext;
use crate::web_io::PlatformInstant;
use eframe::egui;
use math_explorer::biology::neuroscience::{SheetPresetKind, DEFAULT_SHEET_DIM};

/// Cooperative frame time budget (`6.5 ms`) safely below the `8 ms` `WASM_FRAME_SLICE_BUDGET`.
const FRAME_SLICE_BUDGET_SECS: f64 = 0.0065;

/// Split canvas geometry separating the interactive tissue sheet from the 3-lead waveform strip.
#[derive(Clone, Copy, Debug)]
pub struct CanvasSplitLayout {
    /// Square or wide interactive tissue viewport.
    pub sheet_rect: egui::Rect,
    /// Lower 3-lead pseudo-ECG / LFP waveform strip.
    pub ecg_rect: egui::Rect,
}

/// Computes deterministic tissue and 3-lead ECG sub-rectangles inside `canvas_rect`.
pub fn compute_split_layout(canvas_rect: egui::Rect) -> CanvasSplitLayout {
    let pad = 8.0_f32;
    let inner = canvas_rect.shrink(pad);
    let top_h = (inner.height() * 0.63).max(80.0).min(inner.height() - 48.0);
    let side = top_h.min(inner.width()).max(64.0);
    let sheet_min_x = inner.center().x - side * 0.5;
    let sheet_rect = egui::Rect::from_min_size(
        egui::pos2(sheet_min_x, inner.top()),
        egui::vec2(side, top_h),
    );
    let ecg_top = (sheet_rect.bottom() + 8.0).min(inner.bottom() - 24.0);
    let ecg_rect = egui::Rect::from_min_max(
        egui::pos2(inner.left(), ecg_top),
        egui::pos2(inner.right(), inner.bottom()),
    );
    CanvasSplitLayout {
        sheet_rect,
        ecg_rect,
    }
}

impl ExcitableSheetTool {
    /// Maps a screen position inside `sheet_rect` to `(x, y)` grid coordinates on `[0, 96]`.
    pub fn screen_to_grid(pos: egui::Pos2, sheet_rect: egui::Rect) -> Option<(f64, f64)> {
        if !sheet_rect.contains(pos) || sheet_rect.width() <= 1.0 || sheet_rect.height() <= 1.0 {
            return None;
        }
        let nx = ((pos.x - sheet_rect.left()) / sheet_rect.width()).clamp(0.0, 1.0) as f64;
        let ny = ((pos.y - sheet_rect.top()) / sheet_rect.height()).clamp(0.0, 1.0) as f64;
        let max_c = (DEFAULT_SHEET_DIM - 1) as f64;
        Some((nx * max_c, ny * max_c))
    }

    /// Maps `(x, y)` grid coordinates on `[0, 96]` to screen coordinates inside `sheet_rect`.
    pub fn grid_to_screen(x: f64, y: f64, sheet_rect: egui::Rect) -> egui::Pos2 {
        let max_c = (DEFAULT_SHEET_DIM - 1) as f64;
        let ux = (x / max_c).clamp(0.0, 1.0) as f32;
        let uy = (y / max_c).clamp(0.0, 1.0) as f32;
        egui::pos2(
            sheet_rect.left() + ux * sheet_rect.width(),
            sheet_rect.top() + uy * sheet_rect.height(),
        )
    }

    /// Handles continuous `InteractiveTool::on_brush` strokes across tissue or virtual leads.
    pub fn handle_brush(&mut self, ctx: &InteractionContext) {
        let (prev_pos, pos) = if let Some(pos) = ctx.pointer_pos {
            (pos - ctx.delta, pos)
        } else if let Some(stroke) = ctx.brush_stroke {
            (
                egui::pos2(stroke.start[0] as f32, stroke.start[1] as f32),
                egui::pos2(stroke.end[0] as f32, stroke.end[1] as f32),
            )
        } else {
            return;
        };
        let layout = compute_split_layout(ctx.response.rect);

        if self.dragging_lead.is_none() {
            if let Some(hit) = self.find_lead_near(prev_pos, layout.sheet_rect, 15.0) {
                self.dragging_lead = Some(hit);
                self.cfg.active_lead = hit;
            } else if self.cfg.brush_mode == BrushMode::MoveLead {
                self.dragging_lead = Some(self.cfg.active_lead);
            }
        }

        if let Some(lead_idx) = self.dragging_lead {
            if let Some((gx, gy)) = Self::screen_to_grid(pos, layout.sheet_rect) {
                self.move_lead_to(lead_idx, gx, gy);
            }
            return;
        }

        let Some(end_g) = Self::screen_to_grid(pos, layout.sheet_rect) else {
            return;
        };
        let start_g = Self::screen_to_grid(prev_pos, layout.sheet_rect).unwrap_or(end_g);
        self.cursor_grid = end_g;
        self.apply_brush_segment(start_g, end_g);
        self.record_lead_sample();
        self.bump_revision();
    }

    /// Handles `InteractiveTool::on_click` focal stimulation, clamping, scar dabbing, or lead placement.
    pub fn handle_click(&mut self, ctx: &InteractionContext) {
        self.dragging_lead = None;
        let Some(pos) = ctx.pointer_pos else {
            return;
        };
        let layout = compute_split_layout(ctx.response.rect);
        if let Some(hit) = self.find_lead_near(pos, layout.sheet_rect, 15.0) {
            self.cfg.active_lead = hit;
            self.bump_revision();
            return;
        }
        let Some((gx, gy)) = Self::screen_to_grid(pos, layout.sheet_rect) else {
            return;
        };
        self.cursor_grid = (gx, gy);
        if self.cfg.brush_mode == BrushMode::MoveLead {
            self.move_lead_to(self.cfg.active_lead, gx, gy);
        } else {
            self.apply_focal_click(gx, gy);
            self.record_lead_sample();
        }
        self.bump_revision();
    }

    /// Updates hover cursor coordinates and clears completed lead drags.
    pub fn handle_hover(&mut self, ctx: &InteractionContext) {
        if !ctx.is_dragging {
            self.dragging_lead = None;
        }
        if let Some(pos) = ctx.pointer_pos {
            let layout = compute_split_layout(ctx.response.rect);
            if let Some(g) = Self::screen_to_grid(pos, layout.sheet_rect) {
                self.cursor_grid = g;
            }
        }
    }

    /// Handles keyboard navigation, electrode pulsing, scar painting, and defibrillation on focused canvas.
    pub fn handle_keyboard(&mut self, ctx: &InteractionContext) {
        let ui_ctx = ctx.response.ctx.clone();
        let step = if ctx.modifiers.shift { 6.0 } else { 2.0 };
        let mut moved = false;
        ui_ctx.input(|i| {
            if i.key_pressed(egui::Key::ArrowLeft) {
                self.nudge_cursor_or_lead(-step, 0.0);
                moved = true;
            }
            if i.key_pressed(egui::Key::ArrowRight) {
                self.nudge_cursor_or_lead(step, 0.0);
                moved = true;
            }
            if i.key_pressed(egui::Key::ArrowUp) {
                self.nudge_cursor_or_lead(0.0, -step);
                moved = true;
            }
            if i.key_pressed(egui::Key::ArrowDown) {
                self.nudge_cursor_or_lead(0.0, step);
                moved = true;
            }
            if i.key_pressed(egui::Key::Num1) {
                self.cfg.active_lead = 0;
            }
            if i.key_pressed(egui::Key::Num2) {
                self.cfg.active_lead = 1;
            }
            if i.key_pressed(egui::Key::Num3) {
                self.cfg.active_lead = 2;
            }
            if i.key_pressed(egui::Key::Space) {
                let (cx, cy) = self.cursor_grid;
                self.pulse_at(cx, cy);
                moved = true;
            }
            if i.key_pressed(egui::Key::S) {
                let (cx, cy) = self.cursor_grid;
                self.sheet
                    .paint_scar_disc(cx, cy, self.cfg.brush_radius, true);
                self.cable.paint_scar(cx, self.cfg.brush_radius, true);
                moved = true;
            }
            if i.key_pressed(egui::Key::D) {
                self.trigger_defibrillation();
                moved = true;
            }
            if i.key_pressed(egui::Key::M) {
                self.cfg.propagation_mode = match self.cfg.propagation_mode {
                    PropagationMode::FitzHughNagumo2D => PropagationMode::HodgkinHuxleyCable1D,
                    PropagationMode::HodgkinHuxleyCable1D => PropagationMode::FitzHughNagumo2D,
                };
                moved = true;
            }
        });
        if moved {
            self.record_lead_sample();
            self.bump_revision();
        }
    }

    /// Advances the active PDE engine within the `6.5 ms` cooperative slice budget (`< 8 ms`).
    pub fn advance_frame_within_budget(&mut self) {
        let start = PlatformInstant::now();
        let count = self.cfg.steps_per_frame.clamp(1, 16);
        for _ in 0..count {
            self.step_once_guarded();
            if start.elapsed().as_secs_f64() >= FRAME_SLICE_BUDGET_SECS {
                break;
            }
        }
        self.sheet.clear_probes();
        self.cable.clear_probes();
    }

    /// Applies one of the shared starter scenarios to both 2D sheet, 1D cable, and 3-lead array.
    pub fn apply_scenario(&mut self, scenario: ScenarioPreset) {
        self.cfg.scenario = scenario;
        self.cfg.ablated_to_edge = false;
        let (leads, bipolar) = super::dipole_bridge::LeadIntegratorBridge::shared_preset_leads(
            scenario,
            DEFAULT_SHEET_DIM,
            DEFAULT_SHEET_DIM,
        );
        self.cfg.leads = leads;
        self.cfg.readout_bipolar = bipolar;
        self.lead_bridge.sync_leads(leads);

        let kind = match scenario {
            ScenarioPreset::DualEdgeCollision => SheetPresetKind::DualEdgeCollision,
            ScenarioPreset::S1S2SpiralInduction => SheetPresetKind::S1S2SpiralInduction,
            ScenarioPreset::ScarPinnedRotor => SheetPresetKind::ScarPinnedRotor,
            ScenarioPreset::WholeSheetDefibrillation => SheetPresetKind::WholeSheetDefibrillation,
        };
        self.sheet.apply_preset(kind);
        self.cable.clear_scars();
        self.cable.reset_resting();
        if scenario == ScenarioPreset::DualEdgeCollision {
            self.cable.trigger_dual_end_collision();
        } else if scenario != ScenarioPreset::WholeSheetDefibrillation {
            self.cable.pulse_compartment_range(0, 6, 15.0);
        }
        self.clear_trace_history();
        self.record_lead_sample();
        self.bump_revision();
    }

    fn apply_brush_segment(&mut self, start: (f64, f64), end: (f64, f64)) {
        let dist = (end.0 - start.0).hypot(end.1 - start.1).ceil() as usize;
        let steps = dist.max(1);
        let r = self.cfg.brush_radius;
        for i in 0..=steps {
            let t = i as f64 / steps as f64;
            let x = start.0 + (end.0 - start.0) * t;
            let y = start.1 + (end.1 - start.1) * t;
            match self.cfg.brush_mode {
                BrushMode::InjectCurrent => {
                    self.sheet
                        .paint_current_disc(x, y, r, self.cfg.brush_current);
                    self.sheet.pulse_voltage_disc(x, y, r * 0.7, 1.85);
                    self.cable.paint_brush(x, r, self.cfg.brush_current * 8.0);
                    self.cable.pulse_compartment_range(
                        (x - r).max(0.0) as usize,
                        (x + r + 1.0) as usize,
                        12.0,
                    );
                }
                BrushMode::ClampVoltage => {
                    let target_2d = self.cfg.clamp_voltage.clamp(-2.0, 2.2);
                    self.sheet
                        .paint_voltage_clamp_disc(x, y, r, target_2d, true);
                    self.cable.paint_clamp(x, r, self.cfg.clamp_voltage, true);
                }
                BrushMode::PaintScar => {
                    self.sheet.paint_scar_disc(x, y, r, true);
                    self.cable.paint_scar(x, r, true);
                }
                BrushMode::EraseScar => {
                    self.sheet.paint_scar_disc(x, y, r, false);
                    self.cable.paint_scar(x, r, false);
                }
                BrushMode::MoveLead => self.move_lead_to(self.cfg.active_lead, x, y),
            }
        }
    }

    fn apply_focal_click(&mut self, gx: f64, gy: f64) {
        if self.cfg.brush_mode == BrushMode::InjectCurrent {
            self.pulse_at(gx, gy);
        } else {
            self.apply_brush_segment((gx, gy), (gx, gy));
        }
    }

    fn pulse_at(&mut self, gx: f64, gy: f64) {
        let r = self.cfg.brush_radius;
        self.sheet.pulse_voltage_disc(gx, gy, r, 1.9);
        let lo = (gx - r).max(0.0) as usize;
        let hi = (gx + r + 1.0) as usize;
        self.cable.pulse_compartment_range(lo, hi, 15.0);
    }

    fn nudge_cursor_or_lead(&mut self, dx: f64, dy: f64) {
        let max_c = (DEFAULT_SHEET_DIM - 1) as f64;
        if self.cfg.brush_mode == BrushMode::MoveLead {
            let lead = self.cfg.leads[self.cfg.active_lead];
            self.move_lead_to(
                self.cfg.active_lead,
                (lead.x + dx).clamp(0.0, max_c),
                (lead.y + dy).clamp(0.0, max_c),
            );
        } else {
            self.cursor_grid.0 = (self.cursor_grid.0 + dx).clamp(0.0, max_c);
            self.cursor_grid.1 = (self.cursor_grid.1 + dy).clamp(0.0, max_c);
        }
    }

    fn find_lead_near(
        &self,
        pos: egui::Pos2,
        sheet_rect: egui::Rect,
        radius_px: f32,
    ) -> Option<usize> {
        let r2 = radius_px * radius_px;
        self.cfg.leads.iter().enumerate().find_map(|(i, l)| {
            let p = Self::grid_to_screen(l.x, l.y, sheet_rect);
            (p.distance_sq(pos) <= r2).then_some(i)
        })
    }

    pub(super) fn move_lead_to(&mut self, idx: usize, gx: f64, gy: f64) {
        if idx >= 3 {
            return;
        }
        let max_c = DEFAULT_SHEET_DIM as f64;
        let z = self.cfg.leads[idx].z;
        let updated = LeadCoord::new(gx.clamp(0.0, max_c), gy.clamp(0.0, max_c), z);
        self.cfg.leads[idx] = updated;
        self.lead_bridge.set_lead(idx, updated);
        self.record_lead_sample();
    }
}
