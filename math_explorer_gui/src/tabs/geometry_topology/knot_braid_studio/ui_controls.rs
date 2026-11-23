#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Side-panel controls and topological invariant readouts for `KnotBraidStudioTool`.

use super::model::{DiagramEditOp, StudioFamily};
use super::KnotBraidStudioTool;
use crate::app::SemanticPalette;
use eframe::egui;
use math_explorer::pure_math::algebra::BraidWord;

impl KnotBraidStudioTool {
    pub(super) fn draw_side_panel(&mut self, parent_ui: &mut egui::Ui) {
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "knot_braid_studio_controls");
        crate::framework::PaneSidePanel::left(panel_id)
            .default_width(335.0)
            .show_inside(parent_ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui.heading("Knot, Braid & Skein Studio");
                    ui.separator();
                    self.draw_family_selector(ui);
                    ui.separator();
                    self.draw_family_parameters(ui);
                    ui.separator();
                    self.draw_reidemeister_panel(ui);
                    ui.separator();
                    self.draw_invariant_readouts(ui);
                    ui.separator();
                    self.draw_view_and_tube_controls(ui);
                });
            });
    }

    fn draw_family_selector(&mut self, ui: &mut egui::Ui) {
        ui.label("Select Knot / Link Family:");
        let prev = self.family;
        egui::ComboBox::from_id_salt("knot_studio_family_combo")
            .selected_text(self.family.label())
            .show_ui(ui, |ui| {
                for fam in StudioFamily::ALL {
                    ui.selectable_value(&mut self.family, fam, fam.label());
                }
            });
        if self.family != prev {
            self.reidemeister_ops.clear();
            self.status_feedback = None;
        }
    }

    fn draw_family_parameters(&mut self, ui: &mut egui::Ui) {
        match self.family {
            StudioFamily::ArtinBraid => self.draw_braid_composer(ui),
            StudioFamily::TorusKnot => {
                let mut changed = false;
                changed |= ui
                    .add(egui::Slider::new(&mut self.torus_p, 2..=8).text("Longitudinal p"))
                    .changed();
                let mut q = self.torus_q;
                if ui
                    .add(egui::Slider::new(&mut q, -11..=11).text("Meridional q"))
                    .changed()
                {
                    self.torus_q = if q == 0 { 1 } else { q };
                    changed = true;
                }
                if ui.button("Flip Handedness (q -> -q)").clicked() {
                    self.torus_q = -self.torus_q;
                    changed = true;
                }
                if changed {
                    self.reidemeister_ops.clear();
                }
            }
            StudioFamily::LissajousKnot => self.draw_lissajous_controls(ui),
            StudioFamily::FigureEight | StudioFamily::HopfLink | StudioFamily::BorromeanRings => {
                ui.small("Canonical alternating link preset. Use Reidemeister moves below or switch to Artin Braid Closure to edit generators.");
            }
        }
    }

    fn draw_braid_composer(&mut self, ui: &mut egui::Ui) {
        ui.strong("Artin Braid Composer");
        let prev_strands = self.braid_strands;
        if ui
            .add(egui::Slider::new(&mut self.braid_strands, 2..=8).text("Strands (n)"))
            .changed()
            && self.braid_strands != prev_strands
        {
            let max_g = (self.braid_strands - 1) as i32;
            self.braid_generators
                .retain(|&g| g.unsigned_abs() as i32 <= max_g);
            self.reidemeister_ops.clear();
        }
        ui.horizontal_wrapped(|ui| {
            for i in 1..self.braid_strands {
                let g = i as i32;
                if ui
                    .button(format!("+s{i}"))
                    .on_hover_text(format!("Append positive Artin generator sigma_{i}"))
                    .clicked()
                    && self.braid_generators.len() < 64
                {
                    self.braid_generators.push(g);
                    self.reidemeister_ops.clear();
                }
                if ui
                    .button(format!("-s{i}"))
                    .on_hover_text(format!("Append inverse Artin generator sigma_{i}^-1"))
                    .clicked()
                    && self.braid_generators.len() < 64
                {
                    self.braid_generators.push(-g);
                    self.reidemeister_ops.clear();
                }
            }
        });
        self.draw_braid_word_actions(ui);
    }

    fn draw_braid_word_actions(&mut self, ui: &mut egui::Ui) {
        let word_str = if self.braid_generators.is_empty() {
            "e (identity)".to_string()
        } else {
            self.braid_generators
                .iter()
                .map(|&g| {
                    if g > 0 {
                        format!("s{g}")
                    } else {
                        format!("s{0}^-1", -g)
                    }
                })
                .collect::<Vec<_>>()
                .join(" ")
        };
        ui.monospace(format!("Word ({}): {word_str}", self.braid_generators.len()));
        ui.horizontal_wrapped(|ui| {
            if ui.button("Pop").clicked() {
                self.braid_generators.pop();
                self.reidemeister_ops.clear();
            }
            if ui.button("Free Reduce").clicked() {
                if let Ok(mut b) = BraidWord::new(self.braid_strands, self.braid_generators.clone())
                {
                    let cancelled = b.free_reduce_in_place();
                    self.braid_generators = b.generators().to_vec();
                    self.reidemeister_ops.clear();
                    self.status_feedback = Some(format!("Cancelled {cancelled} inverse pair(s)."));
                }
            }
            if ui.button("Mirror").clicked() {
                for g in &mut self.braid_generators {
                    *g = -*g;
                }
                self.reidemeister_ops.clear();
            }
            if ui.button("Inverse").clicked() {
                self.braid_generators.reverse();
                for g in &mut self.braid_generators {
                    *g = -*g;
                }
                self.reidemeister_ops.clear();
            }
        });
        self.draw_braid_relation_buttons(ui);
    }

    fn draw_braid_relation_buttons(&mut self, ui: &mut egui::Ui) {
        let Ok(mut braid) = BraidWord::new(self.braid_strands, self.braid_generators.clone()) else {
            return;
        };
        if braid.len() >= 2 {
            let max_idx = braid.len() - 1;
            ui.add(egui::Slider::new(&mut self.selected_braid_idx, 0..=max_idx).text("Word Index"));
        }
        let idx = self.selected_braid_idx;
        ui.horizontal_wrapped(|ui| {
            let can_comm = braid.can_apply_far_commutativity(idx);
            if ui
                .add_enabled(can_comm, egui::Button::new("Swap Commuting"))
                .clicked()
                && braid.apply_far_commutativity(idx).is_ok()
            {
                self.braid_generators = braid.generators().to_vec();
                self.reidemeister_ops.clear();
            }
            let can_artin = braid.can_apply_artin_relation(idx);
            if ui
                .add_enabled(can_artin, egui::Button::new("Artin s_i s_j s_i"))
                .clicked()
                && braid.apply_artin_relation(idx).is_ok()
            {
                self.braid_generators = braid.generators().to_vec();
                self.reidemeister_ops.clear();
            }
        });
    }

    fn draw_lissajous_controls(&mut self, ui: &mut egui::Ui) {
        let mut changed = false;
        for (axis, lbl) in ["x", "y", "z"].into_iter().enumerate() {
            ui.horizontal(|ui| {
                changed |= ui
                    .add(
                        egui::Slider::new(&mut self.lissajous_freqs[axis], 1..=9)
                            .text(format!("n_{lbl}")),
                    )
                    .changed();
                changed |= ui
                    .add(
                        egui::Slider::new(
                            &mut self.lissajous_phases[axis],
                            -std::f64::consts::PI..=std::f64::consts::PI,
                        )
                        .text(format!("phi_{lbl}")),
                    )
                    .changed();
            });
        }
        if changed {
            self.reidemeister_ops.clear();
        }
    }

    fn draw_reidemeister_panel(&mut self, ui: &mut egui::Ui) {
        ui.strong("Reidemeister Ambient Isotopy");
        let c_len = self.model.diagram.crossings.len();
        let s_len = self.model.diagram.segments.len();
        ui.small(format!(
            "Selected: crossing c{}, pair c{}, segment s{} (Keys: Arrows, S, 1, U, 2, 3)",
            self.selected_crossing, self.second_crossing, self.selected_segment
        ));
        if s_len >= 2 {
            ui.horizontal(|ui| {
                ui.add(
                    egui::Slider::new(&mut self.selected_segment, 0..=s_len - 1).text("Over Seg"),
                );
                ui.add(
                    egui::Slider::new(&mut self.second_segment, 0..=s_len - 1).text("Under Seg"),
                );
            });
        }
        if c_len >= 2 {
            ui.horizontal(|ui| {
                ui.add(egui::Slider::new(&mut self.selected_crossing, 0..=c_len - 1).text("c1"));
                ui.add(egui::Slider::new(&mut self.second_crossing, 0..=c_len - 1).text("c2"));
            });
        }
        self.draw_reidemeister_action_buttons(ui);
    }

    fn draw_reidemeister_action_buttons(&mut self, ui: &mut egui::Ui) {
        self.draw_reidemeister_ri_buttons(ui);
        self.draw_reidemeister_rii_riii_buttons(ui);
        if let Some(msg) = &self.status_feedback {
            ui.small(msg);
        }
    }

    fn draw_reidemeister_ri_buttons(&mut self, ui: &mut egui::Ui) {
        let can_ri_rem = self
            .model
            .diagram
            .can_apply_reidemeister_i_remove(self.selected_crossing);
        ui.horizontal_wrapped(|ui| {
            if ui.button("R-I Twist (+)").clicked() {
                let res = self.push_reidemeister_op(DiagramEditOp::RiAdd {
                    segment: self.selected_segment,
                    positive: true,
                });
                self.record_move_result(res);
            }
            if ui.button("R-I Twist (-)").clicked() {
                let res = self.push_reidemeister_op(DiagramEditOp::RiAdd {
                    segment: self.selected_segment,
                    positive: false,
                });
                self.record_move_result(res);
            }
            if ui
                .add_enabled(can_ri_rem, egui::Button::new("R-I Untwist"))
                .clicked()
            {
                let res = self.push_reidemeister_op(DiagramEditOp::RiRemove {
                    crossing: self.selected_crossing,
                });
                self.record_move_result(res);
            }
        });
    }

    fn draw_reidemeister_rii_riii_buttons(&mut self, ui: &mut egui::Ui) {
        let can_rii_rem = self
            .model
            .diagram
            .can_apply_reidemeister_ii_remove(self.selected_crossing, self.second_crossing);
        let riii_opt = self.preferred_riii_triangle();
        ui.horizontal_wrapped(|ui| {
            let can_rii_add = self.model.diagram.crossings.is_empty()
                || self.selected_segment != self.second_segment;
            if ui
                .add_enabled(can_rii_add, egui::Button::new("R-II Bigon Poke"))
                .clicked()
            {
                let res = self.push_reidemeister_op(DiagramEditOp::RiiAdd {
                    over_seg: self.selected_segment,
                    under_seg: self.second_segment,
                });
                self.record_move_result(res);
            }
            if ui
                .add_enabled(can_rii_rem, egui::Button::new("R-II Cancel Bigon"))
                .clicked()
            {
                let res = self.push_reidemeister_op(DiagramEditOp::RiiRemove {
                    c1: self.selected_crossing,
                    c2: self.second_crossing,
                });
                self.record_move_result(res);
            }
            if ui
                .add_enabled(riii_opt.is_some(), egui::Button::new("R-III Slide"))
                .clicked()
            {
                if let Some(tri) = riii_opt {
                    let res = self.push_reidemeister_op(DiagramEditOp::RiiiSlide {
                        c_tb: tri.c_top_bot,
                        c_tm: tri.c_top_mid,
                        c_mb: tri.c_mid_bot,
                    });
                    self.record_move_result(res);
                }
            }
            if !self.reidemeister_ops.is_empty() && ui.button("Reset Moves").clicked() {
                self.reidemeister_ops.clear();
                self.cached_key = 0;
                self.sync_model();
            }
        });
    }

    pub(crate) fn preferred_riii_triangle(
        &self,
    ) -> Option<math_explorer::pure_math::knot_theory::ReidemeisterTriangle> {
        let sel = self.selected_crossing;
        self.model
            .riii_triangles
            .iter()
            .copied()
            .find(|t| t.c_top_bot == sel || t.c_top_mid == sel || t.c_mid_bot == sel)
            .or_else(|| self.model.riii_triangles.first().copied())
    }

    pub(crate) fn record_move_result(&mut self, res: Result<(), String>) {
        match res {
            Ok(()) => self.status_feedback = Some("Applied ambient isotopy move.".to_string()),
            Err(err) => self.status_feedback = Some(err),
        }
    }

    fn draw_invariant_readouts(&self, ui: &mut egui::Ui) {
        let palette = SemanticPalette::from_visuals(ui.visuals());
        ui.strong("Topological & Geometric Readouts");
        let m = &self.model;
        let g_writhe: f64 = m.writhe_gauss_3d.iter().sum();
        ui.monospace(format!(
            "Components: {} | Crossings: {}",
            m.diagram.component_count(),
            m.diagram.crossing_count()
        ));
        ui.monospace(format!(
            "Diagram Writhe w(D): {:+} | 3D Gauss Writhe: {:+.3}",
            m.writhe_diagram, g_writhe
        ));
        if m.linking_matrix.len() >= 2 {
            ui.monospace(format!("Gauss Linking Lk: {:?}", m.linking_matrix));
        }
        let fox_str = m
            .alexander_fox
            .as_ref()
            .map_or_else(|| "0".to_string(), |p| p.format_in_var("t"));
        ui.monospace(format!("Alexander (Fox):   {fox_str}"));
        if let Some(burau) = &m.alexander_burau {
            ui.monospace(format!("Alexander (Burau): {}", burau.format_in_var("t")));
        }
        if let Some(notice) = &m.crossing_cap_notice {
            egui::Frame::group(ui.style())
                .fill(palette.error_banner_bg)
                .stroke(egui::Stroke::new(1.5_f32, palette.warning))
                .show(ui, |ui| {
                    ui.colored_label(palette.warning, notice);
                });
        } else {
            if let Some(kb) = &m.kauffman_bracket {
                ui.monospace(format!("Kauffman <D>(A):   {}", kb.format_in_var("A")));
            }
            let span = m
                .jones_half_power
                .as_ref()
                .map_or(0, |j| j.span() / 2);
            ui.monospace(format!("Jones V(t) [span {span}]: {}", m.jones_display));
            if let Some(chiral) = m.is_chiral {
                let tag = if chiral {
                    "Chiral (V(L) != V(mirror))"
                } else {
                    "Amphichiral candidate (V(L) == V(mirror))"
                };
                ui.colored_label(palette.info, tag);
            }
        }
    }

    fn draw_view_and_tube_controls(&mut self, ui: &mut egui::Ui) {
        egui::CollapsingHeader::new("3D Tube & Export Mesh")
            .default_open(true)
            .show(ui, |ui| {
                ui.add(egui::Slider::new(&mut self.tube_radius, 0.04..=0.45).text("Tube Radius"));
                ui.add(
                    egui::Slider::new(&mut self.major_radius, 1.2..=3.6).text("Major Scale (R)"),
                );
                if self.family == StudioFamily::TorusKnot {
                    let max_r = (self.major_radius * 0.48).max(0.2);
                    ui.add(
                        egui::Slider::new(&mut self.minor_radius, 0.18..=max_r)
                            .text("Minor Radius (r)"),
                    );
                }
                ui.add(
                    egui::Slider::new(&mut self.curve_samples, 24..=160).text("Curve Samples"),
                );
                ui.add(
                    egui::Slider::new(&mut self.radial_segments, 6..=24).text("Tube Radial Sides"),
                );
                ui.checkbox(&mut self.show_tube_rings, "Show Tubular Cross-Rings");
                ui.checkbox(&mut self.show_bishop_frames, "Show Bishop Parallel Frames");
                self.camera.ui(ui);
            });
    }
}
