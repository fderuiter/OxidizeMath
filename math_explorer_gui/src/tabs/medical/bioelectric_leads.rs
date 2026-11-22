#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::bioelectric_presets::{BioelectricLeadState, MAP_DIM};
use crate::accessibility::{
    lock_canvas_arrow_keys, paint_canvas_focus_ring, register_canvas_accessibility,
};
use crate::app::palette::{SemanticPalette, DARK_PALETTE};
use crate::app::theme::ThemeMode;
use crate::framework::{
    AccessibleHoverText, ExportCapabilities, InteractionContext, InteractiveTool, ToolMetadata,
    ToolStateBounds,
};
use eframe::egui::{self, Color32, ColorImage, Pos2, Rect, Stroke, Vec2};
use math_explorer::math_kernel::colormap::{heatmap_color, heatmap_color_high_contrast};
use math_explorer::math_kernel::types::flatten_2d_index;
use math_explorer::physics::medical::bioelectric::{
    synthesize_bioelectric_preset_sheet, BioelectricPresetKind, DipoleLeadArray, LeadPosition,
};
use std::collections::VecDeque;

const TRACE_LIMIT: usize = 240;

/// Interactive 3-lead extracellular dipole mapper sharing electrophysiology presets with `ExcitableSheetTool`.
pub struct BioelectricLeadTool {
    cfg: BioelectricLeadState,
    dipole: DipoleLeadArray,
    voltage: Vec<f64>,
    scars: Vec<u8>,
    phase: f64,
    active_lead: usize,
    history: VecDeque<(f64, [f64; 3])>,
    texture: Option<egui::TextureHandle>,
    revision: u64,
}

impl Default for BioelectricLeadTool {
    fn default() -> Self {
        let cfg = BioelectricLeadState::default();
        let mut dipole = DipoleLeadArray::new_default(MAP_DIM, MAP_DIM);
        dipole.set_leads(cfg.leads);
        let mut tool = Self {
            cfg,
            dipole,
            voltage: vec![-1.2; MAP_DIM * MAP_DIM],
            scars: vec![0; MAP_DIM * MAP_DIM],
            phase: 0.4,
            active_lead: 0,
            history: VecDeque::with_capacity(TRACE_LIMIT),
            texture: None,
            revision: 1,
        };
        tool.rebuild_field_and_sample();
        for _ in 0..12 {
            tool.advance_phase();
        }
        tool.revision = 1;
        tool
    }
}

impl BioelectricLeadTool {
    fn apply_shared_preset(&mut self, kind: BioelectricPresetKind) {
        let spec = kind.spec(MAP_DIM, MAP_DIM);
        self.cfg.preset = kind;
        self.cfg.leads = spec.leads;
        self.cfg.readout_mode = spec.readout_mode;
        self.cfg.ablated_to_edge = spec.ablated_to_edge;
        self.dipole.set_leads(spec.leads);
        self.history.clear();
        self.phase = 0.2;
        self.rebuild_scars_from_cfg();
        self.rebuild_field_and_sample();
        self.revision = self.revision.wrapping_add(1);
    }

    fn rebuild_scars_from_cfg(&mut self) {
        self.scars.fill(0);
        let spec = self.cfg.preset.spec(MAP_DIM, MAP_DIM);
        if spec.central_scar_radius > 0.0 {
            let r2 = spec.central_scar_radius * spec.central_scar_radius;
            for y in 0..MAP_DIM {
                for x in 0..MAP_DIM {
                    let dx = x as f64 - 48.0;
                    let dy = y as f64 - 48.0;
                    if dx * dx + dy * dy <= r2 {
                        let idx = flatten_2d_index(x, y, MAP_DIM);
                        self.scars[idx] = 1;
                    }
                }
            }
        }
        if self.cfg.ablated_to_edge {
            for y in 46..=50 {
                for x in 48..MAP_DIM {
                    let idx = flatten_2d_index(x, y, MAP_DIM);
                    self.scars[idx] = 1;
                }
            }
        }
    }

    fn advance_phase(&mut self) {
        self.phase += self.cfg.phase_rate;
        self.rebuild_field_and_sample();
    }

    fn rebuild_field_and_sample(&mut self) {
        if self.cfg.ablated_to_edge && self.cfg.preset == BioelectricPresetKind::ScarPinningAblation {
            self.voltage.fill(-1.2);
        } else {
            synthesize_bioelectric_preset_sheet(
                &mut self.voltage,
                &self.scars,
                MAP_DIM,
                MAP_DIM,
                self.cfg.preset,
                self.phase,
            );
        }
        let sample = self
            .dipole
            .evaluate_sheet(&self.voltage, &self.scars, self.cfg.readout_mode);
        if self.history.len() >= TRACE_LIMIT {
            self.history.pop_front();
        }
        self.history.push_back((self.phase, sample));
    }

    fn place_active_lead_from_screen(&mut self, pos: Pos2, map_rect: Rect) {
        if !map_rect.contains(pos) || map_rect.width() <= 2.0 || map_rect.height() <= 2.0 {
            return;
        }
        let gx = ((pos.x - map_rect.left()) / map_rect.width()).clamp(0.0, 1.0) as f64 * 96.0;
        let gy = ((pos.y - map_rect.top()) / map_rect.height()).clamp(0.0, 1.0) as f64 * 96.0;
        let next = LeadPosition::new(gx, gy, self.cfg.lead_height_z);
        self.cfg.leads[self.active_lead] = next;
        self.dipole.set_lead(self.active_lead, next);
        self.rebuild_field_and_sample();
        self.revision = self.revision.wrapping_add(1);
    }

    fn map_subrect(rect: Rect) -> Rect {
        let inner = rect.shrink(10.0);
        let side = (inner.height() * 0.62).min(inner.width()).max(64.0);
        Rect::from_min_size(
            egui::pos2(inner.center().x - side * 0.5, inner.top()),
            Vec2::splat(side),
        )
    }

    fn build_color_image(&self, palette: &SemanticPalette) -> ColorImage {
        let mut pixels = vec![Color32::BLACK; MAP_DIM * MAP_DIM];
        for y in 0..MAP_DIM {
            for x in 0..MAP_DIM {
                let idx = flatten_2d_index(x, y, MAP_DIM);
                if self.scars[idx] != 0 {
                    pixels[idx] = palette.surface_fill;
                } else {
                    let norm = ((self.voltage[idx] + 1.5) / 3.5).clamp(0.0, 1.0);
                    let [r, g, b] = if self.cfg.high_contrast {
                        heatmap_color_high_contrast(norm)
                    } else {
                        heatmap_color(norm)
                    };
                    pixels[idx] = Color32::from_rgb(r, g, b);
                }
            }
        }
        ColorImage {
            size: [MAP_DIM, MAP_DIM],
            source_size: Vec2::splat(MAP_DIM as f32),
            pixels,
        }
    }

    fn paint_trace_strip(
        &self,
        painter: &egui::Painter,
        strip: Rect,
        palette: &SemanticPalette,
        bg: Color32,
    ) {
        if strip.height() <= 20.0 || self.history.len() < 2 {
            return;
        }
        painter.rect_filled(strip, 4.0_f32, bg);
        let peak = self
            .history
            .iter()
            .flat_map(|(_, v)| v.iter().map(|x| x.abs()))
            .fold(0.25_f64, f64::max);
        for (k, &col) in palette.series.iter().take(3).enumerate() {
            let pts: Vec<Pos2> = self
                .history
                .iter()
                .enumerate()
                .map(|(idx, &(_, v))| {
                    let ux = idx as f32 / (self.history.len() - 1) as f32;
                    let uy = 0.5 - 0.42 * (v[k] / peak).clamp(-1.0, 1.0) as f32;
                    egui::pos2(strip.left() + ux * strip.width(), strip.top() + uy * strip.height())
                })
                .collect();
            painter.add(egui::Shape::line(pts, Stroke::new(1.8_f32, col)));
        }
    }
}

impl InteractiveTool for BioelectricLeadTool {
    fn name(&self) -> &'static str {
        "Bioelectric 3-Lead Dipole Mapper"
    }

    fn domain(&self) -> &'static str {
        "medical"
    }

    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }

    fn show_ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            let lbl = if self.cfg.running { "⏸ Pause" } else { "▶ Animate" };
            if ui
                .button(lbl)
                .accessible_hover_text("Toggle dipole wave animation")
                .clicked()
            {
                self.cfg.running = !self.cfg.running;
                self.revision = self.revision.wrapping_add(1);
            }
            if ui.button("Ablate Isthmus").clicked() {
                self.cfg.ablated_to_edge = !self.cfg.ablated_to_edge;
                self.rebuild_scars_from_cfg();
                self.rebuild_field_and_sample();
                self.revision = self.revision.wrapping_add(1);
            }
        });
        ui.separator();
        ui.label("Shared Clinical Presets:");
        for (kind, tag) in [
            (BioelectricPresetKind::DualEdgeCollision, "Dual-Edge Collision"),
            (BioelectricPresetKind::S1S2SpiralInduction, "S1-S2 Spiral"),
            (BioelectricPresetKind::ScarPinningAblation, "Scar Rotor"),
            (BioelectricPresetKind::WholeSheetDefibrillation, "Defibrillation"),
        ] {
            if ui.selectable_label(self.cfg.preset == kind, tag).clicked() {
                self.apply_shared_preset(kind);
                self.cfg.running = true;
            }
        }
        ui.separator();
        ui.horizontal(|ui| {
            for (i, tag) in [(0, "Lead 1"), (1, "Lead 2"), (2, "Lead 3")] {
                if ui.selectable_label(self.active_lead == i, tag).clicked() {
                    self.active_lead = i;
                }
            }
        });
        if ui
            .add(egui::Slider::new(&mut self.cfg.lead_height_z, 1.0..=30.0).text("Stand-off Z"))
            .changed()
        {
            self.cfg.leads[self.active_lead].z = self.cfg.lead_height_z;
            self.dipole
                .set_lead(self.active_lead, self.cfg.leads[self.active_lead]);
            self.rebuild_field_and_sample();
            self.revision = self.revision.wrapping_add(1);
        }
        if ui
            .checkbox(&mut self.cfg.high_contrast, "High-Contrast Colormap")
            .changed()
        {
            self.revision = self.revision.wrapping_add(1);
        }
    }

    fn draw(&mut self, ui: &mut egui::Ui, response: &egui::Response, painter: &egui::Painter) {
        if self.cfg.running {
            self.advance_phase();
            ui.ctx().request_repaint();
        }
        register_canvas_accessibility(
            ui,
            response,
            egui::accesskit::Role::GraphicsDocument,
            "Bioelectric 3-Lead Dipole Mapper Visualization",
            Some("Interactive 3-lead volume-conductor dipole field"),
        );
        lock_canvas_arrow_keys(ui, response);

        let palette = ThemeMode::from_visuals(ui.visuals()).palette();
        let bg = ui.visuals().extreme_bg_color;
        let map_rect = Self::map_subrect(response.rect);
        let img = self.build_color_image(&palette);
        match &mut self.texture {
            Some(tex) => tex.set(img, egui::TextureOptions::NEAREST),
            None => {
                self.texture = Some(ui.ctx().load_texture(
                    "medical_bioelectric_map",
                    img,
                    egui::TextureOptions::NEAREST,
                ));
            }
        }
        if let Some(tex) = &self.texture {
            let uv = Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0));
            painter.image(tex.id(), map_rect, uv, Color32::WHITE);
        }
        for (i, lead) in self.cfg.leads.iter().enumerate() {
            let p = egui::pos2(
                map_rect.left() + (lead.x as f32 / 96.0) * map_rect.width(),
                map_rect.top() + (lead.y as f32 / 96.0) * map_rect.height(),
            );
            painter.circle_filled(p, 6.5_f32, bg);
            painter.circle_stroke(p, 6.5_f32, Stroke::new(2.0_f32, palette.series[i]));
        }
        let strip = Rect::from_min_max(
            egui::pos2(response.rect.left() + 10.0, map_rect.bottom() + 8.0),
            egui::pos2(response.rect.right() - 10.0, response.rect.bottom() - 10.0),
        );
        self.paint_trace_strip(painter, strip, &palette, bg);
        paint_canvas_focus_ring(ui, response);
    }

    fn on_brush(&mut self, ctx: &InteractionContext) {
        if let Some(pos) = ctx.pointer_pos {
            self.place_active_lead_from_screen(pos, Self::map_subrect(ctx.response.rect));
        }
    }

    fn on_click(&mut self, ctx: &InteractionContext) {
        if let Some(pos) = ctx.pointer_pos {
            self.place_active_lead_from_screen(pos, Self::map_subrect(ctx.response.rect));
        }
    }

    fn on_keyboard(&mut self, ctx: &InteractionContext) {
        let mut delta = (0.0, 0.0);
        ctx.response.ctx.input(|i| {
            if i.key_pressed(egui::Key::ArrowLeft) {
                delta.0 -= 2.0;
            }
            if i.key_pressed(egui::Key::ArrowRight) {
                delta.0 += 2.0;
            }
            if i.key_pressed(egui::Key::ArrowUp) {
                delta.1 -= 2.0;
            }
            if i.key_pressed(egui::Key::ArrowDown) {
                delta.1 += 2.0;
            }
        });
        if delta != (0.0, 0.0) {
            let cur = self.cfg.leads[self.active_lead];
            let next = LeadPosition::new(
                (cur.x + delta.0).clamp(0.0, 96.0),
                (cur.y + delta.1).clamp(0.0, 96.0),
                cur.z,
            );
            self.cfg.leads[self.active_lead] = next;
            self.dipole.set_lead(self.active_lead, next);
            self.rebuild_field_and_sample();
            self.revision = self.revision.wrapping_add(1);
        }
    }

    fn state_revision(&self) -> u64 {
        self.revision
    }

    fn has_finite_parameters(&self) -> bool {
        self.cfg.validate().is_ok()
    }

    fn state_bounds(&self) -> ToolStateBounds {
        ToolStateBounds::DEFAULT
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&self.cfg).ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        let parsed: BioelectricLeadState =
            serde_json::from_str(state).map_err(|e| format!("Invalid BioelectricLead JSON: {e}"))?;
        parsed.validate()?;
        self.cfg = parsed;
        self.dipole.set_leads(self.cfg.leads);
        self.rebuild_scars_from_cfg();
        self.rebuild_field_and_sample();
        self.revision = self.revision.wrapping_add(1);
        Ok(())
    }

    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        ["Lead I", "Lead II", "Lead III"]
            .into_iter()
            .enumerate()
            .map(|(k, name)| {
                let pts = self.history.iter().map(|&(t, v)| [t, v[k]]).collect();
                (name.to_string(), pts)
            })
            .collect()
    }

    fn export_raster_image(&self) -> Option<ColorImage> {
        Some(self.build_color_image(&DARK_PALETTE))
    }

    fn step_simulation(&mut self) {
        self.advance_phase();
    }
}

inventory::submit! {
    ToolMetadata {
        id: "BioelectricLeadTool",
        display_name: "Bioelectric 3-Lead Dipole Mapper",
        domain: "medical",
        tags: &["ecg", "bioelectric", "dipole", "ablation"],
        build: || Box::new(BioelectricLeadTool::default()),
    }
}
