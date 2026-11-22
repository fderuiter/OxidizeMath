#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::braess::BraessNetworkParadoxTool;
use super::render_tool_header;
use crate::accessibility::{
    announce_live_status, paint_canvas_focus_ring, register_canvas_accessibility,
    AccessibleHoverText,
};
use crate::app::SemanticPalette;
use crate::framework::{StarterPresetMetadata, ToolMetadata};
use eframe::egui;
use egui_plot::{Legend, Line, LineStyle, Plot, PlotPoints, VLine};

impl BraessNetworkParadoxTool {
    pub(super) fn render_braess_inside(&mut self, parent_ui: &mut egui::Ui) {
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "braess_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            render_tool_header(ui, "Braess's Paradox");
            self.render_controls(ui);
        });

        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            self.render_metrics_banner(ui);
            ui.add_space(4.0);
            self.render_wheatstone_diagram(ui);
            ui.add_space(4.0);
            self.render_active_plot(ui);
        });
    }

    fn render_controls(&mut self, ui: &mut egui::Ui) {
        let mut changed = false;
        changed |= ui
            .checkbox(&mut self.state.shortcut_open, "Open Central Shortcut (A → B)")
            .accessible_hover_text("Toggle directed shortcut A to B on the 4-node Wheatstone network")
            .changed();
        changed |= ui
            .checkbox(
                &mut self.state.continuum_mfg_mode,
                "Continuum Crowd Mode (HJB + Fokker-Planck)",
            )
            .accessible_hover_text("Switch plot between Wardrop latency curves and Mean-Field Game PDEs")
            .changed();

        ui.separator();
        changed |= ui
            .add(egui::Slider::new(&mut self.state.demand, 0.2..=2.0).text("Demand D"))
            .accessible_hover_text("Total commuter traffic demand from source S to destination T")
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.state.congestion_slope, 0.2..=3.0).text("Congestion α"))
            .accessible_hover_text("Linear latency slope alpha on bottleneck edges S->A and B->T")
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.state.bypass_latency, 0.2..=3.0).text("Bypass β"))
            .accessible_hover_text("Constant travel latency beta on highway edges A->T and S->B")
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.state.shortcut_cost, 0.0..=1.5).text("Shortcut Cost"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.state.shortcut_toll, 0.0..=1.5).text("Shortcut Toll τ"))
            .accessible_hover_text("Pigouvian congestion toll charged on shortcut A->B")
            .changed();

        if ui
            .button(format!("Apply Optimal Pigou Toll (τ* = {:.2})", self.metrics.optimal_toll))
            .accessible_hover_text("Set shortcut toll to marginal externality restoring Pigou social optimum")
            .clicked()
        {
            self.state.shortcut_toll = self.metrics.optimal_toll;
            changed = true;
        }

        ui.separator();
        ui.label("Continuum MFG Parameters:");
        changed |= ui
            .add(egui::Slider::new(&mut self.state.mfg_viscosity, 0.02..=0.25).text("Viscosity ν"))
            .accessible_hover_text("Stochastic crowd diffusion nu (clamped so nu*dt/dx^2 < 0.25)")
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.state.mfg_grid_points, 17..=65).text("Grid Nx"))
            .changed();
        if changed {
            self.recompute_all();
        }
    }

    fn render_metrics_banner(&self, ui: &mut egui::Ui) {
        let palette = SemanticPalette::from_visuals(ui.visuals());
        let m = self.metrics();
        let poa_color = if m.price_of_anarchy > 1.001 { palette.warning } else { palette.success };
        ui.horizontal_wrapped(|ui| {
            ui.colored_label(
                palette.series[0],
                format!("Wardrop: {:.3} (closed {:.3})", m.wardrop_latency, m.closed_latency),
            );
            ui.separator();
            ui.colored_label(palette.success, format!("Pigou Opt: {:.3}", m.pigou_latency));
            ui.separator();
            ui.colored_label(poa_color, format!("PoA: {:.4}", m.price_of_anarchy));
            ui.separator();
            ui.colored_label(
                if m.nash_fixed_point { palette.success } else { palette.error },
                format!("Nash Fixed Point: {}", if m.nash_fixed_point { "Verified" } else { "False" }),
            );
            ui.separator();
            ui.colored_label(palette.muted_text, format!("CFL: {:.3}", m.mfg_cfl_ratio));
        });
    }

    fn render_wheatstone_diagram(&mut self, ui: &mut egui::Ui) {
        let palette = SemanticPalette::from_visuals(ui.visuals());
        let width = ui.available_width().max(240.0);
        let (response, painter) =
            ui.allocate_painter(egui::Vec2::new(width, 132.0), egui::Sense::click());
        self.canvas_id = Some(response.id);

        let toggled_key = response.has_focus()
            && ui.input(|i| i.key_pressed(egui::Key::Space) || i.key_pressed(egui::Key::Enter));
        if response.clicked() || toggled_key {
            self.state.shortcut_open = !self.state.shortcut_open;
            self.recompute_all();
            announce_live_status(
                ui.ctx(),
                format!(
                    "Shortcut A to B {}, Wardrop latency {:.2}, Price of Anarchy {:.3}",
                    if self.state.shortcut_open { "opened" } else { "closed" },
                    self.metrics.wardrop_latency,
                    self.metrics.price_of_anarchy
                ),
            );
        }

        let summary = format!(
            "Shortcut {}, flows Upper {:.2}, Lower {:.2}, Zig-zag {:.2}, PoA {:.3}",
            if self.state.shortcut_open { "open" } else { "closed" },
            self.metrics.wardrop_flow[0],
            self.metrics.wardrop_flow[1],
            self.metrics.wardrop_flow[2],
            self.metrics.price_of_anarchy
        );
        register_canvas_accessibility(
            ui,
            &response,
            egui::accesskit::Role::GraphicsDocument,
            "Wheatstone 4-node traffic network (Click or press Space to toggle A to B shortcut)",
            Some(&summary),
        );
        painter.rect_filled(response.rect, 6.0, palette.surface_fill);
        self.paint_network_edges_and_nodes(&painter, response.rect, &palette);
        paint_canvas_focus_ring(ui, &response);
    }

    fn paint_network_edges_and_nodes(
        &self,
        painter: &egui::Painter,
        rect: egui::Rect,
        palette: &SemanticPalette,
    ) {
        let (cx, cy) = (rect.center().x, rect.center().y);
        let (dx, dy) = ((rect.width() * 0.34).min(180.0), 42.0);
        let (s_pos, t_pos) = (egui::pos2(cx - dx, cy), egui::pos2(cx + dx, cy));
        let (a_pos, b_pos) = (egui::pos2(cx, cy - dy), egui::pos2(cx, cy + dy));

        painter.line_segment([s_pos, a_pos], egui::Stroke::new(2.5_f32, palette.series[0]));
        painter.line_segment([a_pos, t_pos], egui::Stroke::new(2.0_f32, palette.graph_edge));
        painter.line_segment([s_pos, b_pos], egui::Stroke::new(2.0_f32, palette.graph_edge));
        painter.line_segment([b_pos, t_pos], egui::Stroke::new(2.5_f32, palette.series[0]));

        let ab_color = if self.state.shortcut_open {
            palette.error
        } else {
            palette.muted_text
        };
        painter.line_segment(
            [a_pos, b_pos],
            egui::Stroke::new(if self.state.shortcut_open { 3.2_f32 } else { 1.2_f32 }, ab_color),
        );

        let font = egui::FontId::proportional(12.0_f32);
        for (pos, label) in [(s_pos, "S"), (a_pos, "A"), (b_pos, "B"), (t_pos, "T")] {
            painter.circle_filled(pos, 12.0_f32, palette.surface_fill);
            painter.circle_stroke(pos, 12.0_f32, egui::Stroke::new(2.0_f32, palette.node_stroke));
            painter.text(pos, egui::Align2::CENTER_CENTER, label, font.clone(), palette.node_stroke);
        }
        let f = self.metrics.wardrop_flow;
        let p_ab = self.metrics.pigou_flow[2];
        painter.text(
            egui::pos2(rect.left() + 10.0, rect.top() + 14.0),
            egui::Align2::LEFT_CENTER,
            format!(
                "Flows: S→A→T {:.2} | S→B→T {:.2} | S→A→B→T {:.2} (Pigou A→B {:.2})",
                f[0], f[1], f[2], p_ab
            ),
            font,
            palette.info,
        );
    }

    fn render_active_plot(&self, ui: &mut egui::Ui) {
        let palette = SemanticPalette::from_visuals(ui.visuals());
        if self.state.continuum_mfg_mode {
            Plot::new(ui.id().with("braess_mfg_plot"))
                .legend(Legend::default())
                .x_axis_label("Corridor Position x")
                .y_axis_label("HJB Value u(x,t) / Fokker-Planck Crowd Density m(x,t)")
                .show(ui, |plot_ui| {
                    plot_ui.line(
                        Line::new("HJB Value u(x, 0)", PlotPoints::from(self.mfg_u0.clone()))
                            .color(palette.accent)
                            .style(LineStyle::Dashed { length: 7.0_f32 })
                            .width(2.0_f32),
                    );
                    for (name, pts, color, w) in [
                        ("Initial Density m(x, 0)", &self.mfg_m0, palette.series[0], 1.8_f32),
                        ("Midpoint Density m(x, T/2)", &self.mfg_m_mid, palette.series[1], 2.2_f32),
                        ("Terminal Density m(x, T)", &self.mfg_m_final, palette.success, 2.4_f32),
                    ] {
                        plot_ui.line(Line::new(name, PlotPoints::from(pts.clone())).color(color).width(w));
                    }
                });
        } else {
            Plot::new(ui.id().with("braess_latency_plot"))
                .legend(Legend::default())
                .x_axis_label("Commuter Demand D")
                .y_axis_label("Average Commuter Travel Time L(D)")
                .show(ui, |plot_ui| {
                    plot_ui.vline(
                        VLine::new("Active Demand D", self.state.demand)
                            .color(palette.accent)
                            .style(LineStyle::Dashed { length: 6.0_f32 }),
                    );
                    plot_ui.line(
                        Line::new("Wardrop Selfish Latency", PlotPoints::from(self.curve_wardrop.clone()))
                            .color(palette.error)
                            .width(2.5_f32),
                    );
                    plot_ui.line(
                        Line::new("Closed-Shortcut Latency", PlotPoints::from(self.curve_closed.clone()))
                            .color(palette.series[0])
                            .style(LineStyle::Dashed { length: 8.0_f32 })
                            .width(2.0_f32),
                    );
                    plot_ui.line(
                        Line::new("Pigou Social Optimum", PlotPoints::from(self.curve_pigou.clone()))
                            .color(palette.success)
                            .width(2.2_f32),
                    );
                });
        }
    }
}

impl scientific_metadata::theory::TheoryDescribable for BraessNetworkParadoxTool {
    fn theory_description(&self) -> String {
        "Braess's network paradox & Mean-Field congestion: adding a zero-cost link $A \\to B$ to a 4-node Wheatstone network shifts Wardrop Nash equilibrium ($x^* \\in B(x^*)$) from latency $1.50$ to $2.00$ ($\\text{PoA} = 4/3$), whereas coupled backward HJB and forward Fokker-Planck PDEs model continuum crowd concentration.".into()
    }
    fn phonetic_description(&self) -> String {
        "Braess network routing paradox and mean field crowd congestion".into()
    }
    fn theory_citation(&self) -> String {
        "Braess (1968), Uber ein Paradoxon aus der Verkehrsplanung; Lasry & Lions (2007), Mean field games, Jpn. J. Math. 2, 229-260".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
}

pub const PRESET_BRAESS_WHEATSTONE: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Braess's Paradox",
    "domain": "paradox_lab",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "demand": 1.1,
    "congestion_slope": 1.0,
    "bypass_latency": 1.0,
    "shortcut_cost": 0.0,
    "shortcut_toll": 0.0,
    "shortcut_open": true,
    "continuum_mfg_mode": false,
    "mfg_viscosity": 0.08,
    "mfg_grid_points": 33
  }
}"#;

inventory::submit! {
    ToolMetadata {
        id: "BraessNetworkParadoxTool",
        display_name: "Braess's Paradox",
        domain: "paradox_lab",
        tags: &["game_theory", "nash", "braess", "mean_field"],
        build: || Box::new(BraessNetworkParadoxTool::default()),
    }
}

inventory::submit! {
    StarterPresetMetadata {
        id: "braess_wheatstone_starter",
        title: "Wheatstone Shortcut & Price of Anarchy",
        description: "Opening zero-cost shortcut A->B raises selfish commuter travel time above Pigou's optimum.",
        category: "Paradox Lab",
        icon: "🛣",
        domain: "paradox_lab",
        tool_name: Some("BraessNetworkParadoxTool"),
        preset_generator: || PRESET_BRAESS_WHEATSTONE,
    }
}

// [cite:paradox_lab]
