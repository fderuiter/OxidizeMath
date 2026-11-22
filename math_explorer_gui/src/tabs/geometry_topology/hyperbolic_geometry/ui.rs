#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{
    BilliardPreset, ChartLayout, HyperbolicGeometryTool, LabMode, TilingPreset,
};
use crate::web_io::PlatformInstant;
use eframe::egui;
use math_explorer::pure_math::hyperbolic::{
    MAX_POLYGON_VERTICES, MAX_TILING_DEPTH, Psu11, validate_hyperbolic_schlafli,
};

impl HyperbolicGeometryTool {
    pub(super) fn tick_realtime(&mut self, ctx: &egui::Context) {
        let now = PlatformInstant::now();
        let dt = self
            .last_frame_instant
            .map_or(1.0 / 60.0, |prev| prev.elapsed().as_secs_f64().clamp(0.0, 0.05));
        self.last_frame_instant = Some(now);

        if self.state.mode == LabMode::ChaoticBilliards && self.state.billiards_running {
            let ds = dt * self.state.billiard_speed.clamp(0.1, 10.0);
            if ds > 0.0 {
                self.billiards.step(ds);
                self.bump_revision();
            }
            ctx.request_repaint();
        }
    }

    pub(super) fn render_controls_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("Hyperbolic Laboratory");
        ui.label("Constant Gaussian curvature K = -1");
        ui.separator();

        self.render_mode_and_layout_selectors(ui);
        ui.separator();

        match self.state.mode {
            LabMode::TilingExplorer => self.render_tiling_controls(ui),
            LabMode::ChaoticBilliards => self.render_billiard_controls(ui),
        }

        ui.separator();
        self.render_isometry_readout(ui);
    }

    fn render_mode_and_layout_selectors(&mut self, ui: &mut egui::Ui) {
        ui.label(egui::RichText::new("Laboratory Mode").strong());
        ui.horizontal(|ui| {
            if ui
                .selectable_value(
                    &mut self.state.mode,
                    LabMode::TilingExplorer,
                    "Schläfli Tilings",
                )
                .changed()
            {
                self.bump_revision();
            }
            if ui
                .selectable_value(
                    &mut self.state.mode,
                    LabMode::ChaoticBilliards,
                    "Geodesic Billiards",
                )
                .changed()
            {
                self.bump_revision();
            }
        });

        ui.add_space(4.0);
        self.render_chart_layout_selector(ui);
    }

    fn render_chart_layout_selector(&mut self, ui: &mut egui::Ui) {
        ui.label(egui::RichText::new("Conformal Chart View").strong());
        egui::ComboBox::from_id_salt("hyperbolic_chart_layout")
            .selected_text(match self.state.chart_layout {
                ChartLayout::DualSynchronized => "Dual Synchronized (D + H)",
                ChartLayout::PoincareDiskOnly => "Poincaré Disk D Only",
                ChartLayout::UpperHalfPlaneOnly => "Upper Half-Plane H Only",
            })
            .show_ui(ui, |ui| {
                let mut changed = false;
                changed |= ui
                    .selectable_value(
                        &mut self.state.chart_layout,
                        ChartLayout::DualSynchronized,
                        "Dual Synchronized (D + H)",
                    )
                    .changed();
                changed |= ui
                    .selectable_value(
                        &mut self.state.chart_layout,
                        ChartLayout::PoincareDiskOnly,
                        "Poincaré Disk D Only",
                    )
                    .changed();
                changed |= ui
                    .selectable_value(
                        &mut self.state.chart_layout,
                        ChartLayout::UpperHalfPlaneOnly,
                        "Upper Half-Plane H Only",
                    )
                    .changed();
                if changed {
                    self.bump_revision();
                }
            });

        if self.state.chart_layout != ChartLayout::PoincareDiskOnly
            && ui
                .add(
                    egui::Slider::new(&mut self.state.half_plane_span, 0.75..=12.0)
                        .text("H Half-Span"),
                )
                .changed()
        {
            self.bump_revision();
        }
    }

    fn render_tiling_controls(&mut self, ui: &mut egui::Ui) {
        ui.label(egui::RichText::new("Coxeter Tiling {p, q}").strong());
        let mut rebuild = false;
        egui::ComboBox::from_id_salt("hyperbolic_tiling_preset")
            .selected_text(tiling_preset_label(self.state.tiling_preset))
            .show_ui(ui, |ui| {
                for preset in [
                    TilingPreset::Heptagonal73,
                    TilingPreset::Pentagonal54,
                    TilingPreset::Octagonal83,
                    TilingPreset::Hexagonal64,
                    TilingPreset::IdealModularInf3,
                    TilingPreset::IdealHeckeInf4,
                    TilingPreset::CustomSchlafli,
                ] {
                    rebuild |= ui
                        .selectable_value(
                            &mut self.state.tiling_preset,
                            preset,
                            tiling_preset_label(preset),
                        )
                        .changed();
                }
            });

        if self.state.tiling_preset == TilingPreset::CustomSchlafli {
            rebuild |= self.render_custom_schlafli_sliders(ui);
        }

        rebuild |= ui
            .add(
                egui::Slider::new(&mut self.state.tiling_depth, 1..=MAX_TILING_DEPTH)
                    .text("Reflection Depth"),
            )
            .changed();

        if rebuild {
            self.rebuild_tiling();
            self.bump_revision();
        }

        if ui
            .checkbox(&mut self.state.show_horocycles, "Show Tangent Horocycle")
            .changed()
        {
            self.bump_revision();
        }
        if self.state.show_horocycles
            && ui
                .add(
                    egui::Slider::new(&mut self.state.horocycle_radius, 0.08..=0.85)
                        .text("Horocycle r"),
                )
                .changed()
        {
            self.bump_revision();
        }

        ui.small(format!("Generated tiles: {}", self.transformed_tile_count));
        if let Some(area) = self.central_tile_area() {
            ui.small(format!("Gauss-Bonnet Tile Area: {area:.6}"));
        } else {
            ui.small("Ideal {∞, q} apeirogon (cusps on ∂D)");
        }
    }

    fn render_custom_schlafli_sliders(&mut self, ui: &mut egui::Ui) -> bool {
        let mut cand_p = self.state.schlafli_p;
        let mut cand_q = self.state.schlafli_q;
        let p_changed = ui
            .add(egui::Slider::new(&mut cand_p, 3..=MAX_POLYGON_VERTICES as u32).text("Sides p"))
            .changed();
        let q_changed = ui
            .add(egui::Slider::new(&mut cand_q, 3..=MAX_POLYGON_VERTICES as u32).text("Valence q"))
            .changed();
        if validate_hyperbolic_schlafli(cand_p, cand_q).is_ok() {
            if p_changed || q_changed {
                self.state.schlafli_p = cand_p;
                self.state.schlafli_q = cand_q;
                return true;
            }
        } else {
            ui.colored_label(
                ui.visuals().warn_fg_color,
                format!("{{{cand_p}, {cand_q}}} requires (p-2)(q-2) > 4"),
            );
        }
        false
    }

    fn render_billiard_controls(&mut self, ui: &mut egui::Ui) {
        ui.label(egui::RichText::new("Geodesic Twin-Ray Billiards").strong());
        let mut reconfigure = false;
        egui::ComboBox::from_id_salt("hyperbolic_billiard_preset")
            .selected_text(billiard_preset_label(self.state.billiard_preset))
            .show_ui(ui, |ui| {
                for preset in [
                    BilliardPreset::CompactOctagon83,
                    BilliardPreset::IdealTriangle,
                    BilliardPreset::CompactHeptagon73,
                    BilliardPreset::IdealQuad,
                ] {
                    reconfigure |= ui
                        .selectable_value(
                            &mut self.state.billiard_preset,
                            preset,
                            billiard_preset_label(preset),
                        )
                        .changed();
                }
            });

        reconfigure |= ui
            .add(
                egui::Slider::new(&mut self.state.perturbation_log10, -10.0..=-2.0)
                    .text("log10(δ₀)"),
            )
            .changed();

        ui.horizontal(|ui| {
            ui.label("RNG Seed:");
            reconfigure |= ui
                .add(egui::DragValue::new(&mut self.state.billiard_seed).range(0..=999_999))
                .changed();
        });

        if reconfigure {
            self.rebuild_billiards();
            self.bump_revision();
        }

        if ui
            .add(egui::Slider::new(&mut self.state.billiard_speed, 0.2..=6.0).text("Flow Speed"))
            .changed()
        {
            self.bump_revision();
        }

        self.render_billiard_action_buttons(ui);
    }

    fn render_billiard_action_buttons(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            let play_label = if self.state.billiards_running {
                "Pause Flow"
            } else {
                "Play Flow"
            };
            if ui.button(play_label).clicked() {
                self.state.billiards_running = !self.state.billiards_running;
                self.bump_revision();
            }
            if ui.button("+10 Bounces").clicked() {
                self.billiards.advance_reflections(10);
                self.bump_revision();
            }
            if ui.button("Reseed").clicked() {
                self.state.billiard_seed = self.state.billiard_seed.wrapping_add(1);
                self.rebuild_billiards();
                self.bump_revision();
            }
        });

        let n_codes = self.billiards.copy_edge_hits_into(&mut self.edge_hit_scratch);
        let tail_start = n_codes.saturating_sub(12);
        let code_str: String = self.edge_hit_scratch[tail_start..n_codes]
            .iter()
            .map(|c| c.to_string())
            .collect::<Vec<_>>()
            .join("·");
        ui.small(format!(
            "Bounces: {} | s = {:.2} | ln(δ_t/δ_0) = {:.3}",
            self.billiards.total_bounces(),
            self.billiards.elapsed_length(),
            self.billiards.current_log_separation(),
        ));
        ui.small(format!("Edge code tail: [{code_str}]"));
    }

    fn render_isometry_readout(&mut self, ui: &mut egui::Ui) {
        ui.label(egui::RichText::new("Möbius Isometry Group").strong());
        let a = self.state.disk_isometry.a();
        let b = self.state.disk_isometry.b();
        let psl = self.state.disk_isometry.to_psl2r().entries();
        ui.small(format!(
            "PSU(1,1): a={:+.3}{:+.3}i, b={:+.3}{:+.3}i (det={:.6})",
            a.re,
            a.im,
            b.re,
            b.im,
            self.state.disk_isometry.determinant()
        ));
        ui.small(format!(
            "PSL(2,R): [{:+.2}, {:+.2}; {:+.2}, {:+.2}]",
            psl[0], psl[1], psl[2], psl[3]
        ));
        if ui.button("Reset Möbius Isometry").clicked() {
            self.state.disk_isometry = Psu11::identity();
            self.refresh_transformed_tiles();
            self.bump_revision();
        }
        ui.small("Drag inside D or H (or focus canvas + Arrow keys) to apply live Möbius isometries.");
    }
}

fn tiling_preset_label(preset: TilingPreset) -> &'static str {
    match preset {
        TilingPreset::Heptagonal73 => "{7, 3} Heptagonal (Klein)",
        TilingPreset::Pentagonal54 => "{5, 4} Right Pentagonal",
        TilingPreset::Octagonal83 => "{8, 3} Octagonal (Bolza)",
        TilingPreset::Hexagonal64 => "{6, 4} Right Hexagonal",
        TilingPreset::IdealModularInf3 => "{∞, 3} Modular Apeirogons",
        TilingPreset::IdealHeckeInf4 => "{∞, 4} Hecke Apeirogons",
        TilingPreset::CustomSchlafli => "Custom {p, q}",
    }
}

fn billiard_preset_label(preset: BilliardPreset) -> &'static str {
    match preset {
        BilliardPreset::CompactOctagon83 => "Compact Octagon {8, 3}",
        BilliardPreset::IdealTriangle => "Ideal Triangle {3}",
        BilliardPreset::CompactHeptagon73 => "Compact Heptagon {7, 3}",
        BilliardPreset::IdealQuad => "Ideal Quadrilateral {4}",
    }
}
