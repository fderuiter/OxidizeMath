#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Side-panel fabrication controls, `web_io` SVG file dialog dispatch, URL share-link
//! copy action, and 1:1 millimeter `egui_plot` bed & pen-travel preview.

use super::state::{
    FabricationSourceKind, FabricationTilingSelector, StudioBedMode, StudioFlowElement,
    StudioFlowElementKind, MAX_STUDIO_FLOW_ELEMENTS, MAX_STUDIO_TURING_GRID,
};
use super::FabricationStudioTool;
use crate::accessibility::{announce_status_with_priority, AccessibleHoverText};
use crate::framework::{bounded_drag_value, scoped_panel_id, PaneSidePanel};
use crate::web_io::{spawn_save_as_file_dialog, SaveAsDialogConfig};
use eframe::egui;
use egui_plot::{
    Line, LineStyle, Plot, PlotPoints, MAX_FABRICATION_BED_MM, MIN_FABRICATION_BED_MM,
};
use math_commons::registry::MAX_GRID_SIZE;
use math_explorer::pure_math::tilings::MAX_TILING_DEPTH;
use std::f64::consts::PI;
use std::sync::mpsc::channel;

const FABRICATION_SVG_DIALOG: SaveAsDialogConfig = SaveAsDialogConfig {
    filter_name: "Fabrication Vector SVG (*.svg)",
    extensions: &["svg"],
    fallback_filename: "fabrication_toolpath.svg",
    write_error_prefix: "Failed to save fabrication SVG",
};

impl FabricationStudioTool {
    pub(super) fn poll_async_io(&mut self, ctx: &egui::Context) {
        if let Some(rx) = &self.save_rx {
            if let Ok(outcome) = rx.try_recv() {
                match outcome {
                    Ok(saved) => {
                        let msg = format!("Saved millimeter fabrication SVG to {saved}");
                        announce_status_with_priority(&msg, "polite");
                        self.status_banner = Some((msg, false));
                    }
                    Err(err) => {
                        announce_status_with_priority(&err, "assertive");
                        self.status_banner = Some((err, true));
                    }
                }
                self.save_rx = None;
                ctx.request_repaint();
            }
        }
    }

    pub(super) fn trigger_save_svg_dialog(&mut self) {
        match self.serialize_fabrication_svg() {
            Ok(svg_xml) => {
                let default_name = self.default_svg_filename();
                let (tx, rx) = channel();
                spawn_save_as_file_dialog(
                    tx,
                    FABRICATION_SVG_DIALOG,
                    &default_name,
                    svg_xml,
                    |path, fallback| {
                        path.file_name()
                            .map(|f| f.to_string_lossy().into_owned())
                            .unwrap_or_else(|| fallback.to_string())
                    },
                );
                self.save_rx = Some(rx);
            }
            Err(err) => {
                announce_status_with_priority(&err, "assertive");
                self.status_banner = Some((err, true));
            }
        }
    }

    fn copy_studio_share_url(&mut self, ctx: &egui::Context) {
        match self.encode_studio_share_url() {
            Ok(url) => {
                crate::url_state::copy_share_url_to_clipboard(ctx, url, |_| {});
                let msg = "Copied fabrication preset share URL to clipboard!".to_string();
                announce_status_with_priority(&msg, "polite");
                self.status_banner = Some((msg, false));
            }
            Err(err) => {
                announce_status_with_priority(&err, "assertive");
                self.status_banner = Some((err, true));
            }
        }
    }

    pub(super) fn draw_side_panel(&mut self, parent_ui: &mut egui::Ui) {
        let panel_id = scoped_panel_id(parent_ui, "fabrication_studio_controls");
        PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("Fabrication Studio");
                ui.label("Millimeter laser-cut & pen-plotter vector generator.");
                ui.separator();

                let before = self.state.clone();
                self.draw_source_selector(ui);
                ui.separator();
                self.draw_active_source_controls(ui);
                ui.separator();
                self.draw_bed_and_travel_controls(ui);
                if self.state != before {
                    self.state.clamp_in_place();
                    self.mark_dirty();
                }

                ui.separator();
                self.draw_telemetry_and_export_buttons(ui);
            });
        });
    }

    fn draw_source_selector(&mut self, ui: &mut egui::Ui) {
        ui.label(egui::RichText::new("1. Geometry Source").strong());
        for kind in [
            FabricationSourceKind::AperiodicTiling,
            FabricationSourceKind::TuringField,
            FabricationSourceKind::PotentialFlow,
        ] {
            ui.radio_value(&mut self.state.source, kind, kind.label());
        }
    }

    fn draw_active_source_controls(&mut self, ui: &mut egui::Ui) {
        ui.label(egui::RichText::new("2. Generator Parameters").strong());
        match self.state.source {
            FabricationSourceKind::AperiodicTiling => self.draw_tiling_controls(ui),
            FabricationSourceKind::TuringField => self.draw_turing_controls(ui),
            FabricationSourceKind::PotentialFlow => self.draw_flow_controls(ui),
        }
    }

    fn draw_tiling_controls(&mut self, ui: &mut egui::Ui) {
        for sel in [
            FabricationTilingSelector::EinsteinHat,
            FabricationTilingSelector::EinsteinSpectre,
            FabricationTilingSelector::PenroseP3,
            FabricationTilingSelector::PenroseP2,
        ] {
            ui.radio_value(&mut self.state.tiling_kind, sel, sel.label());
        }
        ui.add(
            egui::Slider::new(&mut self.state.tiling_depth, 0..=MAX_TILING_DEPTH)
                .text("Substitution Depth"),
        );
    }

    fn draw_turing_controls(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            if ui
                .button("Spots Coaster")
                .accessible_hover_text("Apply Schnakenberg Turing spots coaster preset")
                .clicked()
            {
                self.state.turing_a = 0.10;
                self.state.turing_b = 0.90;
                self.state.turing_d_v = 45.0;
            }
            if ui
                .button("Labyrinth Stripes")
                .accessible_hover_text("Apply Schnakenberg labyrinth stripes coaster preset")
                .clicked()
            {
                self.state.turing_a = 0.14;
                self.state.turing_b = 0.86;
                self.state.turing_d_v = 28.0;
            }
        });
        let max_g = MAX_STUDIO_TURING_GRID.min(MAX_GRID_SIZE);
        ui.add(egui::Slider::new(&mut self.state.turing_a, 0.02..=0.5).text("Rate a"));
        ui.add(egui::Slider::new(&mut self.state.turing_b, 0.2..=1.8).text("Rate b"));
        ui.add(egui::Slider::new(&mut self.state.turing_d_v, 10.0..=150.0).text("Inhibitor D_v"));
        ui.add(egui::Slider::new(&mut self.state.turing_grid_size, 16..=max_g).text("Grid Size"));
        ui.add(egui::Slider::new(&mut self.state.turing_steps, 20..=300).text("RD Steps"));
        ui.add(egui::Slider::new(&mut self.state.turing_score_levels, 1..=8).text("Score Isolines"));
        ui.checkbox(
            &mut self.state.turing_coaster_border,
            "Circular Coaster Cut Perimeter",
        );
    }

    fn draw_flow_controls(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            if ui
                .button("Cylinder Flow")
                .accessible_hover_text("Uniform freestream past a circular cylinder doublet")
                .clicked()
            {
                self.apply_cylinder_flow_elements();
            }
            if ui
                .button("Rankine Oval")
                .accessible_hover_text("Uniform freestream past a symmetric source-sink pair")
                .clicked()
            {
                self.apply_rankine_oval_elements();
            }
        });
        ui.add(egui::Slider::new(&mut self.state.flow_view_range, 4.0..=30.0).text("Domain Range"));
        ui.add(
            egui::Slider::new(&mut self.state.flow_streamline_count, 4..=48)
                .text("Streamline Count"),
        );
        if self.state.flow_elements.len() < MAX_STUDIO_FLOW_ELEMENTS
            && ui
                .button("➕ Add Vortex")
                .accessible_hover_text("Append a circulation vortex element to the flow field")
                .clicked()
        {
            self.state.flow_elements.push(StudioFlowElement {
                kind: StudioFlowElementKind::Vortex,
                strength: 18.0,
                x: 0.0,
                y: 2.5,
                angle_deg: 0.0,
            });
        }
    }

    fn apply_cylinder_flow_elements(&mut self) {
        let kappa = 2.0 * PI * 10.0 * 4.0;
        self.state.flow_elements = vec![
            StudioFlowElement {
                kind: StudioFlowElementKind::Uniform,
                strength: 10.0,
                x: 0.0,
                y: 0.0,
                angle_deg: 0.0,
            },
            StudioFlowElement {
                kind: StudioFlowElementKind::Doublet,
                strength: kappa,
                x: 0.0,
                y: 0.0,
                angle_deg: 0.0,
            },
        ];
    }

    fn apply_rankine_oval_elements(&mut self) {
        self.state.flow_elements = vec![
            StudioFlowElement {
                kind: StudioFlowElementKind::Uniform,
                strength: 8.0,
                x: 0.0,
                y: 0.0,
                angle_deg: 0.0,
            },
            StudioFlowElement {
                kind: StudioFlowElementKind::Source,
                strength: 45.0,
                x: -2.5,
                y: 0.0,
                angle_deg: 0.0,
            },
            StudioFlowElement {
                kind: StudioFlowElementKind::Sink,
                strength: 45.0,
                x: 2.5,
                y: 0.0,
                angle_deg: 0.0,
            },
        ];
    }

    fn draw_bed_and_travel_controls(&mut self, ui: &mut egui::Ui) {
        ui.label(egui::RichText::new("3. Physical Bed & Toolpaths").strong());
        ui.horizontal_wrapped(|ui| {
            ui.radio_value(&mut self.state.bed_mode, StudioBedMode::A4, "A4 (210×297 mm)");
            ui.radio_value(
                &mut self.state.bed_mode,
                StudioBedMode::UsLetter,
                "US Letter",
            );
            ui.radio_value(&mut self.state.bed_mode, StudioBedMode::Custom, "Custom");
        });
        if self.state.bed_mode == StudioBedMode::Custom {
            ui.horizontal(|ui| {
                ui.label("Width (mm):");
                ui.add(bounded_drag_value(
                    &mut self.state.custom_width_mm,
                    MIN_FABRICATION_BED_MM..=MAX_FABRICATION_BED_MM,
                    1.0,
                ));
            });
            ui.horizontal(|ui| {
                ui.label("Height (mm):");
                ui.add(bounded_drag_value(
                    &mut self.state.custom_height_mm,
                    MIN_FABRICATION_BED_MM..=MAX_FABRICATION_BED_MM,
                    1.0,
                ));
            });
        }
        let (bw, bh) = self.state.bed_preset().clamped_dimensions_mm();
        let max_m = (0.4 * bw.min(bh)).min(100.0);
        ui.add(egui::Slider::new(&mut self.state.margin_mm, 0.0..=max_m).text("Margin (mm)"));
        ui.checkbox(
            &mut self.state.optimize_travel,
            "Nearest-Neighbor Travel Ordering",
        );
        ui.checkbox(&mut self.state.show_pen_travel, "Preview Rapid Pen Lifts");
    }

    fn draw_telemetry_and_export_buttons(&mut self, ui: &mut egui::Ui) {
        self.ensure_artifact();
        if let Ok(art) = self.artifact_or_build() {
            let saved_pct = if art.raw_travel_mm > 1e-6 {
                ((art.raw_travel_mm - art.active_travel_mm) / art.raw_travel_mm * 100.0).max(0.0)
            } else {
                0.0
            };
            ui.label(format!(
                "Cut (#layer-cut): {} paths ({:.1} mm)",
                art.cut_paths_mm.len(),
                art.cut_length_mm
            ));
            ui.label(format!(
                "Score (#layer-score): {} paths ({:.1} mm)",
                art.score_paths_mm.len(),
                art.score_length_mm
            ));
            ui.label(format!(
                "Pen-lift travel: {:.1} mm ({saved_pct:.0}% saved vs {:.1} mm)",
                art.active_travel_mm, art.raw_travel_mm
            ));
        }

        if ui
            .button("💾 Save Fabrication SVG (mm)")
            .accessible_hover_text(
                "Export millimeter SVG with #layer-cut and #layer-score stroke paths",
            )
            .clicked()
        {
            self.trigger_save_svg_dialog();
        }
        if ui
            .button("🔗 Copy Share Link")
            .accessible_hover_text("Copy bookmarkable URL encoding all fabrication parameters")
            .clicked()
        {
            let ctx = ui.ctx().clone();
            self.copy_studio_share_url(&ctx);
        }
        if let Some((ref msg, is_err)) = self.status_banner {
            let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
            let color = if is_err { palette.error } else { palette.success };
            ui.colored_label(color, msg);
        }
    }

    pub(super) fn draw_bed_preview(&mut self, ui: &mut egui::Ui) {
        self.ensure_artifact();
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        let cut_color = palette.error;
        let score_color = palette.series_color(0);
        let travel_color = palette.warning.linear_multiply(0.65);
        let sheet_color = palette.series_color(2).linear_multiply(0.55);

        let Ok(art) = self.artifact_or_build() else {
            if let Some(ref err) = self.last_error {
                ui.colored_label(palette.error, err);
            }
            return;
        };
        let show_lifts = self.state.show_pen_travel;

        egui::CentralPanel::default().show_inside(ui, |ui| {
            Plot::new("fabrication_studio_bed_plot")
                .data_aspect(1.0)
                .show_axes([true, true])
                .x_axis_label("Bed X (mm)")
                .y_axis_label("Bed Y (mm)")
                .show(ui, |plot_ui| {
                    let h = art.bed_height_mm;
                    draw_bed_sheet_guides(plot_ui, &art, sheet_color, travel_color, show_lifts);
                    for (idx, poly) in art.score_paths_mm.iter().enumerate() {
                        plot_ui.line(
                            Line::new(
                                format!("Score Path {}", idx + 1),
                                PlotPoints::from(flip_y_polyline(poly, h)),
                            )
                            .color(score_color)
                            .width(1.5_f32),
                        );
                    }
                    for (idx, poly) in art.cut_paths_mm.iter().enumerate() {
                        plot_ui.line(
                            Line::new(
                                format!("Cut Path {}", idx + 1),
                                PlotPoints::from(flip_y_polyline(poly, h)),
                            )
                            .color(cut_color)
                            .width(2.0_f32),
                        );
                    }
                });
        });
    }
}

fn draw_bed_sheet_guides(
    plot_ui: &mut egui_plot::PlotUi<'_>,
    art: &super::FabricationArtifact,
    sheet_color: egui::Color32,
    travel_color: egui::Color32,
    show_lifts: bool,
) {
    let (w, h, m) = (art.bed_width_mm, art.bed_height_mm, art.margin_mm);
    let sheet = vec![[0.0, 0.0], [w, 0.0], [w, h], [0.0, h], [0.0, 0.0]];
    plot_ui.line(Line::new("Bed Sheet (mm)", PlotPoints::from(sheet)).color(sheet_color));
    let margin_box = vec![[m, m], [w - m, m], [w - m, h - m], [m, h - m], [m, m]];
    plot_ui.line(
        Line::new("Safe Margin", PlotPoints::from(margin_box))
            .color(sheet_color)
            .style(LineStyle::dashed_dense()),
    );
    if show_lifts {
        for (idx, seg) in art.pen_lifts_mm.iter().enumerate() {
            let y_flipped = vec![[seg[0][0], h - seg[0][1]], [seg[1][0], h - seg[1][1]]];
            plot_ui.line(
                Line::new(format!("Pen Lift {}", idx + 1), PlotPoints::from(y_flipped))
                    .color(travel_color)
                    .width(1.0_f32)
                    .style(LineStyle::dashed_loose()),
            );
        }
    }
}

fn flip_y_polyline(poly: &egui_plot::FabricationPolyline, bed_h: f64) -> Vec<[f64; 2]> {
    let mut pts: Vec<[f64; 2]> = poly.points.iter().map(|&[x, y]| [x, bed_h - y]).collect();
    if poly.closed && pts.len() >= 3 {
        pts.push(pts[0]);
    }
    pts
}
