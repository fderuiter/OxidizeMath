#![cfg_attr(any(), verified(opt_out = "gui_tool"))]
// [cite:quantum_mechanics]

//! Interactive wavepacket emitter `(x0, k0, sigma)` canvas handles, WCAG 2.2 AA
//! `GraphicsDocument` reticle metadata, and gallery starter presets for [`super::WaveSimulatorTool`].

use eframe::egui;
use egui_plot::{Arrows, Line, LineStyle, MarkerShape, PlotPoint, PlotPoints, Points};

use super::presets::{GRID_X_MAX, GRID_X_MIN, POTENTIAL_DISPLAY_SCALE};
use super::runner::WaveData;
use super::sculpt::{stamp_double_barrier_into, x_to_grid_index};
use super::WaveSimulatorTool;
use crate::accessibility::{announce_live_status, register_canvas_accessibility};
use crate::async_sim::SimCommand;
use crate::framework::{encode_sparse_potential, BrushInteractionMode, BrushMode};

/// Plot y-coordinate of the incident wavepacket position/momentum handle.
pub(crate) const EMITTER_HANDLE_Y: f64 = 0.20;
/// Plot y-coordinate of the Gaussian packet width envelope handle.
pub(crate) const EMITTER_SIGMA_Y: f64 = 0.16;
/// Visual scaling factor mapping wave number $k_0$ to plot arrow length.
pub(crate) const K0_ARROW_SCALE: f64 = 0.12;
const HANDLE_HIT_RADIUS_PX: f32 = 14.0;

/// Active pointer drag target for the wavepacket emitter overlay handles.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum ActiveEmitterDrag {
    /// No emitter handle is being dragged.
    #[default]
    None,
    /// Dragging the wavepacket center $x_0$ diamond handle.
    EmitterX0,
    /// Dragging the wave number $k_0$ arrow-tip circle handle.
    EmitterK0,
    /// Dragging the spatial spread $\sigma$ envelope square handle.
    EmitterSigma,
}

fn marker_point(
    name: &str,
    xy: [f64; 2],
    r: f32,
    shape: MarkerShape,
    c: egui::Color32,
) -> Points<'_> {
    Points::new(name, vec![xy]).radius(r).shape(shape).color(c)
}

/// Renders draggable wavepacket `(x0, k0, sigma)` handles on the quantum `Plot`.
pub(super) fn draw_emitter_overlays(
    plot_ui: &mut egui_plot::PlotUi<'_>,
    emitter: (f64, f64, f64),
    palette: &crate::app::SemanticPalette,
) {
    let (x0, k0, sigma) = emitter;
    let k0_tip_x = x0 + k0 * K0_ARROW_SCALE;
    let span = vec![[x0 - sigma, EMITTER_SIGMA_Y], [x0 + sigma, EMITTER_SIGMA_Y]];
    plot_ui.line(
        Line::new("Packet Width Envelope", PlotPoints::new(span))
            .style(LineStyle::Dashed { length: 4.0_f32 })
            .color(palette.accent),
    );
    plot_ui.arrows(
        Arrows::new(
            "Momentum Vector k₀",
            vec![[x0, EMITTER_HANDLE_Y]],
            vec![[k0_tip_x, EMITTER_HANDLE_Y]],
        )
        .color(palette.warning),
    );
    plot_ui.points(marker_point(
        "Emitter x₀",
        [x0, EMITTER_HANDLE_Y],
        6.0,
        MarkerShape::Diamond,
        palette.accent,
    ));
    plot_ui.points(marker_point(
        "Momentum Tip k₀",
        [k0_tip_x, EMITTER_HANDLE_Y],
        5.0,
        MarkerShape::Circle,
        palette.warning,
    ));
    plot_ui.points(marker_point(
        "Width Handle σ",
        [x0 + sigma, EMITTER_SIGMA_Y],
        5.0,
        MarkerShape::Square,
        palette.success,
    ));
}

fn hit_test_emitter_handle(
    plot_ui: &egui_plot::PlotUi<'_>,
    pointer_screen: egui::Pos2,
    emitter: (f64, f64, f64),
) -> ActiveEmitterDrag {
    let (x0, k0, sigma) = emitter;
    let candidates = [
        (
            ActiveEmitterDrag::EmitterK0,
            PlotPoint::new(x0 + k0 * K0_ARROW_SCALE, EMITTER_HANDLE_Y),
        ),
        (
            ActiveEmitterDrag::EmitterSigma,
            PlotPoint::new(x0 + sigma, EMITTER_SIGMA_Y),
        ),
        (
            ActiveEmitterDrag::EmitterX0,
            PlotPoint::new(x0, EMITTER_HANDLE_Y),
        ),
    ];
    candidates
        .into_iter()
        .map(|(t, pt)| (t, plot_ui.screen_from_plot(pt).distance(pointer_screen)))
        .filter(|(_, dist)| *dist <= HANDLE_HIT_RADIUS_PX)
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(ActiveEmitterDrag::None, |(t, _)| t)
}

/// Samples pointer presses and drags on the `(x0, k0, sigma)` emitter handles.
pub(super) fn sample_emitter_drag(
    plot_ui: &egui_plot::PlotUi<'_>,
    emitter: (f64, f64, f64),
    active_drag: &mut ActiveEmitterDrag,
) -> Option<(ActiveEmitterDrag, f64)> {
    let ctx = plot_ui.ctx();
    let (pressed, primary_down, pos_opt) = ctx.input(|i| {
        (
            i.pointer.primary_pressed(),
            i.pointer.primary_down(),
            i.pointer.interact_pos(),
        )
    });
    if !primary_down {
        *active_drag = ActiveEmitterDrag::None;
        return None;
    }
    let screen_pos = pos_opt?;
    if *active_drag == ActiveEmitterDrag::None
        && pressed
        && plot_ui.response().rect.contains(screen_pos)
    {
        let hit = hit_test_emitter_handle(plot_ui, screen_pos, emitter);
        if hit != ActiveEmitterDrag::None {
            plot_ui.response().request_focus();
            *active_drag = hit;
        }
    }
    if *active_drag != ActiveEmitterDrag::None {
        let plot_pt = plot_ui.plot_from_screen(screen_pos);
        return Some((*active_drag, plot_pt.x));
    }
    None
}

/// Applies an active emitter handle drag at `plot_x` to [`WaveSimulatorTool`].
pub(super) fn apply_emitter_drag(
    tool: &mut WaveSimulatorTool,
    drag: ActiveEmitterDrag,
    plot_x: f64,
    ctx: Option<&egui::Context>,
) -> bool {
    let mut x0 = tool.x0;
    let mut k0 = tool.k0;
    let mut sigma = tool.sigma;
    match drag {
        ActiveEmitterDrag::EmitterX0 => x0 = plot_x,
        ActiveEmitterDrag::EmitterK0 => k0 = (plot_x - x0) / K0_ARROW_SCALE,
        ActiveEmitterDrag::EmitterSigma => sigma = (plot_x - x0).abs(),
        ActiveEmitterDrag::None => return false,
    }
    tool.update_wavepacket_live(x0, k0, sigma, ctx);
    true
}

/// Formats and announces the WCAG 2.2 AA reticle status when keyboard cursor coordinates move.
pub(super) fn attach_quantum_canvas_accessibility(
    ui: &egui::Ui,
    response: &egui::Response,
    wave_data: &WaveData,
    cursor_pos: [f64; 2],
    cursor_moved: bool,
) {
    ui.ctx().data_mut(|d| {
        d.insert_temp(egui::Id::new("quantum_plot_response_id"), response.id);
    });
    let n = wave_data.x_axis.len().min(wave_data.potential.len());
    let idx = x_to_grid_index(cursor_pos[0], GRID_X_MIN, GRID_X_MAX, n);
    let rx = wave_data.x_axis.get(idx).copied().unwrap_or(cursor_pos[0]);
    let rv = if cursor_moved {
        cursor_pos[1] / POTENTIAL_DISPLAY_SCALE
    } else {
        wave_data.potential.get(idx).copied().unwrap_or(0.0)
    };
    let summary = format!(
        "Reticle x = {rx:.2}, V(x) = {rv:.1}, P_T = {:.1}%, P_R = {:.1}%",
        wave_data.p_transmitted * 100.0,
        wave_data.p_reflected * 100.0
    );
    register_canvas_accessibility(
        ui,
        response,
        egui::accesskit::Role::GraphicsDocument,
        "Quantum Potential Sculpting Canvas",
        Some(&summary),
    );
    if cursor_moved && response.has_focus() {
        announce_live_status(ui.ctx(), summary);
    }
}

impl WaveSimulatorTool {
    fn draw_sculpt_action_buttons(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.horizontal_wrapped(|ui| {
            if ui.button("⛩ Stamp Double Barrier").clicked() {
                self.stamp_double_barrier_edits(ctx);
            }
            if ui.button("Clear Potential Edits").clicked() {
                self.clear_sculpt_edits(ctx);
            }
            if ui.button("🚀 Re-launch ψ(x,0)").clicked() {
                let _ = self
                    .worker
                    .send_command(SimCommand::Custom("relaunch_wavepacket".to_string()));
                ctx.request_repaint();
            }
        });
    }

    fn stamp_double_barrier_edits(&mut self, ctx: &egui::Context) {
        let (x_axis, base) = self.base_potential_vec();
        let mut target = base.clone();
        stamp_double_barrier_into(&mut target, &x_axis);
        self.sparse_potential = encode_sparse_potential(&base, &target, 1e-6);
        if let Ok(mut cfg) = self.shared_config.write() {
            cfg.sparse_potential = self.sparse_potential.clone();
        }
        self.bump_revision();
        self.export_runner = None;
        self.export_data = None;
        let _ = self
            .worker
            .send_command(SimCommand::Custom("stamp_double_barrier".to_string()));
        ctx.request_repaint();
    }

    fn clear_sculpt_edits(&mut self, ctx: &egui::Context) {
        self.sparse_potential.clear();
        if let Ok(mut cfg) = self.shared_config.write() {
            cfg.sparse_potential.clear();
        }
        self.bump_revision();
        self.export_runner = None;
        self.export_data = None;
        let _ = self
            .worker
            .send_command(SimCommand::Custom("Clear Potential Edits".to_string()));
        ctx.request_repaint();
    }

    pub(crate) fn draw_sculpt_controls(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.label("Potential V(x) Sculpting");
        ui.horizontal(|ui| {
            ui.selectable_value(
                &mut self.brush_kit.interaction_mode,
                BrushInteractionMode::Sculpt,
                "🖌 Sculpt",
            );
            ui.selectable_value(
                &mut self.brush_kit.interaction_mode,
                BrushInteractionMode::Pan,
                "✋ Pan",
            );
        });
        ui.horizontal_wrapped(|ui| {
            ui.selectable_value(
                &mut self.brush_kit.mode,
                BrushMode::PotentialBarrier,
                "Barrier",
            );
            ui.selectable_value(&mut self.brush_kit.mode, BrushMode::PotentialWell, "Well");
            ui.selectable_value(&mut self.brush_kit.mode, BrushMode::PotentialStep, "Step");
            ui.selectable_value(&mut self.brush_kit.mode, BrushMode::Erase, "Erase");
        });
        ui.add(egui::Slider::new(&mut self.brush_kit.radius, 0.10..=1.20).text("Brush Width"));
        self.draw_sculpt_action_buttons(ui, ctx);
    }
}

const TUNNELING_BARRIER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Quantum Tunneling Barrier",
    "domain": "quantum",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "preset": "RectangularBarrier",
    "x0": -2.0,
    "k0": 5.0,
    "sigma": 0.5,
    "v0": 14.0,
    "width": 0.5,
    "dt": 0.02,
    "steps_per_frame": 5
  }
}"#;

const RESONANT_DOUBLE_WELL_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Resonant Double-Barrier Well",
    "domain": "quantum",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "preset": "DoubleBarrier",
    "x0": -2.5,
    "k0": 4.6385,
    "sigma": 0.55,
    "v0": 20.0,
    "width": 1.0,
    "dt": 0.02,
    "steps_per_frame": 5
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "wave_sim_tunneling_barrier",
        title: "Quantum Tunneling Barrier",
        description: "Sub-barrier Gaussian wavepacket scattering across a thin rectangular potential wall with live Born transmission and reflection integrals.",
        category: "Wave Mechanics",
        icon: "🧱",
        domain: "quantum",
        tool_name: Some("WaveSimulator"),
        preset_generator: || TUNNELING_BARRIER_PRESET_JSON,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "wave_sim_resonant_double_well",
        title: "Resonant Double-Barrier Well",
        description: "Fabry-Pérot quasi-bound quantum resonance where constructive matter-wave interference drives near-unity transmission across twin barriers.",
        category: "Wave Mechanics",
        icon: "⛩",
        domain: "quantum",
        tool_name: Some("WaveSimulator"),
        preset_generator: || RESONANT_DOUBLE_WELL_PRESET_JSON,
    }
}
