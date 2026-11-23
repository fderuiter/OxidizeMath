//! Side-panel `AnimatedSlider` controls, `FiniteStateGuard` recovery banner, Steiner/Dijkstra
//! scorecard, and paired `egui_plot` telemetry graphs for [`super::PhysarumTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{PhysarumBrushMode, PhysarumPreset, PhysarumTool};
use crate::async_sim::SimWorkerState;
use crate::framework::recovery::FiniteStateGuard;
use crate::framework::{PlaybackAction, PlaybackTransport};
use crate::reflective_ui::AnimatedSlider;
use eframe::egui;
use egui_plot::{Legend, Line, Plot, PlotPoints};
use math_commons::math_kernel::colormap::ensure_dark_canvas_contrast;
use math_explorer::biology::physarum::{MAX_PHYSARUM_AGENTS, MIN_PHYSARUM_AGENTS};

impl PhysarumTool {
    pub(crate) fn render_lab(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        let hc = crate::app::ThemeMode::from_visuals(ui.visuals())
            == crate::app::ThemeMode::HighContrast;
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        let obs_rgb = [
            palette.surface_fill.r(),
            palette.surface_fill.g(),
            palette.surface_fill.b(),
        ];
        let bcn_rgb = [palette.accent.r(), palette.accent.g(), palette.accent.b()];
        if self.high_contrast != hc || self.obstacle_rgb != obs_rgb || self.beacon_rgb != bcn_rgb {
            self.high_contrast = hc;
            self.obstacle_rgb = obs_rgb;
            self.beacon_rgb = bcn_rgb;
            self.refresh_pixels_in_place();
        }

        if self.playing && !FiniteStateGuard::diverged(self) {
            self.step_frame_cooperative();
            if self.playing {
                ctx.request_repaint();
            }
        }

        self.draw_side_controls(ui);
        self.draw_central_workspace(ui);
    }

    fn draw_side_controls(&mut self, ui: &mut egui::Ui) {
        let panel_id = crate::framework::scoped_panel_id(ui, "physarum_lab_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("Physarum Network Lab");
                ui.separator();
                self.draw_recovery_banner(ui);
                self.draw_playback_and_presets(ui);
                ui.separator();
                self.draw_chemotaxis_sliders(ui);
                ui.separator();
                self.draw_brush_and_scale_controls(ui);
            });
        });
    }

    fn draw_recovery_banner(&mut self, ui: &mut egui::Ui) {
        if !FiniteStateGuard::diverged(self) {
            return;
        }
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        egui::Frame::group(ui.style())
            .stroke(egui::Stroke::new(1.5_f32, palette.error))
            .show(ui, |ui| {
                ui.colored_label(palette.error, "⚠ Instability Guard Triggered");
                if let Some(diag) = &self.diagnostic {
                    ui.small(&diag.message);
                }
                ui.horizontal_wrapped(|ui| {
                    if ui.button("Auto-Scale Speed").clicked() {
                        FiniteStateGuard::auto_scale_step_size(self);
                    }
                    if ui.button("Clamp Params").clicked() {
                        FiniteStateGuard::clamp_parameters(self);
                    }
                    if ui.button("Reset Steiner").clicked() {
                        FiniteStateGuard::reset_parameters(self);
                    }
                });
            });
        ui.separator();
    }

    fn draw_playback_and_presets(&mut self, ui: &mut egui::Ui) {
        let state = if FiniteStateGuard::diverged(self) {
            SimWorkerState::Faulted
        } else if self.playing {
            SimWorkerState::Running
        } else {
            SimWorkerState::Ready
        };
        if let Some(action) = PlaybackTransport::from_worker_state(
            "physarum_transport",
            "Physarum Network Lab",
            state,
        )
        .can_reset(true)
        .show(ui)
        {
            match action {
                PlaybackAction::Play => self.playing = true,
                PlaybackAction::Pause => self.playing = false,
                PlaybackAction::Step => {
                    self.playing = false;
                    self.step_once();
                }
                PlaybackAction::Reset => self.rebuild_from_state(),
            }
        }
        ui.small(format!(
            "Step {}: {:.2} ms | Cities: {} | Flux: {:.0}%",
            self.engine.telemetry().step_count,
            self.last_step_ms,
            self.engine.terminal_count(),
            self.engine.telemetry().mean_flux * 100.0,
        ));
        ui.add_space(4.0);
        ui.label("Network Topologies:");
        ui.horizontal_wrapped(|ui| {
            for p in PhysarumPreset::ALL {
                if ui.selectable_label(self.preset == p, p.label()).clicked() {
                    self.apply_preset(p);
                }
            }
        });
    }

    fn draw_chemotaxis_sliders(&mut self, ui: &mut egui::Ui) {
        ui.label("Chemotaxis & Conductance Pruning:");
        let prev = self.current_params();
        ui.add(
            AnimatedSlider::new(&mut self.sensor_angle, 0.10..=1.50)
                .text("Sensor θs")
                .id_salt("phys_sa"),
        );
        ui.add(
            AnimatedSlider::new(&mut self.turn_angle, 0.05..=1.50)
                .text("Turn θr")
                .id_salt("phys_ta"),
        );
        ui.add(
            AnimatedSlider::new(&mut self.sensor_distance, 2.0..=32.0)
                .text("Lookahead ds")
                .id_salt("phys_sd"),
        );
        ui.add(
            AnimatedSlider::new(&mut self.step_size, 0.20..=3.50)
                .text("Speed v")
                .id_salt("phys_v"),
        );
        ui.add(
            AnimatedSlider::new(&mut self.deposition, 0.05..=5.00)
                .text("Deposit w")
                .id_salt("phys_dep"),
        );
        ui.add(
            AnimatedSlider::new(&mut self.diffusion_rate, 0.0..=1.0)
                .text("Diffuse α")
                .id_salt("phys_diff"),
        );
        ui.add(
            AnimatedSlider::new(&mut self.decay_rate, 0.005..=0.50)
                .text("Decay γ")
                .id_salt("phys_decay"),
        );
        ui.add(
            AnimatedSlider::new(&mut self.prune_rate, 0.0..=0.35)
                .text("Prune γp")
                .id_salt("phys_prune"),
        );
        ui.add(
            AnimatedSlider::new(&mut self.beacon_strength, 0.0..=5.0)
                .text("Beacon B")
                .id_salt("phys_beacon"),
        );
        if self.current_params() != prev {
            self.sync_engine_parameters();
        }
    }

    fn draw_brush_and_scale_controls(&mut self, ui: &mut egui::Ui) {
        ui.label("Interactive Brushes:");
        ui.horizontal_wrapped(|ui| {
            ui.selectable_value(
                &mut self.brush_mode,
                PhysarumBrushMode::PlaceCity,
                "📍 City",
            );
            ui.selectable_value(
                &mut self.brush_mode,
                PhysarumBrushMode::PaintWall,
                "🧱 Wall",
            );
            ui.selectable_value(
                &mut self.brush_mode,
                PhysarumBrushMode::EraseWall,
                "🧽 Erase",
            );
            ui.selectable_value(
                &mut self.brush_mode,
                PhysarumBrushMode::SeverTube,
                "✂ Sever",
            );
        });
        ui.add(
            AnimatedSlider::new(&mut self.brush_radius, 1.0..=24.0)
                .text("Brush r")
                .id_salt("phys_brush_r"),
        );
        ui.checkbox(&mut self.show_skeleton_overlay, "Show Skeleton & MST Graph");
        self.draw_population_and_grid_selectors(ui);
    }

    fn draw_population_and_grid_selectors(&mut self, ui: &mut egui::Ui) {
        let prev_agents = self.agent_count;
        let mut agents_f = self.agent_count as f64;
        ui.add(
            AnimatedSlider::new(
                &mut agents_f,
                (MIN_PHYSARUM_AGENTS as f64)..=(MAX_PHYSARUM_AGENTS as f64),
            )
            .text("Agents N")
            .id_salt("phys_agents"),
        );
        self.agent_count = (agents_f.round() as usize).clamp(MIN_PHYSARUM_AGENTS, MAX_PHYSARUM_AGENTS);
        if self.agent_count != prev_agents {
            self.engine.set_agent_count(self.agent_count);
        }
        ui.horizontal_wrapped(|ui| {
            ui.label("Grid:");
            for dim in [64usize, 128, 256] {
                if ui
                    .selectable_label(self.grid_size == dim, format!("{dim}²"))
                    .clicked()
                    && self.grid_size != dim
                {
                    self.grid_size = dim;
                    self.rebuild_from_state();
                }
            }
            if ui.button("🎲 Reseed").clicked() {
                self.rng_seed = self.rng_seed.wrapping_add(1);
                self.rebuild_from_state();
            }
        });
    }

    fn draw_central_workspace(&mut self, ui: &mut egui::Ui) {
        egui::CentralPanel::default().show_inside(ui, |ui| {
            let avail = ui.available_size();
            if avail.x >= 560.0 {
                ui.columns(2, |cols| {
                    let side = cols[0].available_width().min(avail.y - 22.0);
                    cols[0].vertical_centered(|ui| {
                        self.draw_interactive_canvas(ui, side);
                        ui.small(
                            "Click/drag brush (Shift/Right-drag severs); Arrows + Space/C/W/E/Del on canvas.",
                        );
                    });
                    self.draw_benchmark_panel(&mut cols[1], avail.y);
                });
            } else {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    self.draw_interactive_canvas(ui, avail.x.min(320.0));
                    self.draw_benchmark_panel(ui, 380.0);
                });
            }
        });
    }

    fn draw_benchmark_scorecard(&self, ui: &mut egui::Ui) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        let b = &self.latest_benchmark;
        egui::Frame::group(ui.style()).show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                let delta_col = if b.length_reduction_pct >= 0.0 {
                    palette.success
                } else {
                    palette.info
                };
                ui.colored_label(
                    delta_col,
                    format!(
                        "Wire/MST: {:.3} ({:+.1}% vs MST)",
                        b.cost_ratio, b.length_reduction_pct
                    ),
                );
                ui.separator();
                ui.label(format!(
                    "Dijkstra Stretch: {:.3} (MST {:.3})",
                    b.mean_route_stretch, b.mst_mean_stretch
                ));
            });
            ui.horizontal_wrapped(|ui| {
                ui.label(format!(
                    "Connected: {:.0}% | Fault Survival: {:.0}% | Steiner Nodes: {}",
                    b.connected_pair_fraction * 100.0,
                    b.pairwise_fault_survival * 100.0,
                    b.steiner_junctions.len()
                ));
            });
        });
    }

    fn draw_benchmark_panel(&self, ui: &mut egui::Ui, total_height: f32) {
        self.draw_benchmark_scorecard(ui);
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());

        let dark = ui.visuals().dark_mode;
        let col_cost = Self::contrast_line_color(palette.success, dark);
        let col_stretch = Self::contrast_line_color(palette.accent, dark);
        let col_fault = Self::contrast_line_color(palette.info, dark);
        let col_focus = Self::contrast_line_color(palette.warning, dark);
        let plot_h = ((total_height - 108.0) * 0.46).clamp(105.0, 290.0);
        let (cost_pts, stretch_pts, fault_pts, focus_pts) = self.build_plot_series();

        ui.label("Wire Cost Ratio & Dijkstra Route Stretch");
        Plot::new(crate::framework::scoped_panel_id(ui, "physarum_cost_plot"))
            .height(plot_h)
            .legend(Legend::default())
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new("Wire / MST Cost", PlotPoints::from(cost_pts))
                        .color(col_cost)
                        .width(2.0_f32),
                );
                plot_ui.line(
                    Line::new("Mean Dijkstra Stretch", PlotPoints::from(stretch_pts))
                        .color(col_stretch)
                        .width(2.0_f32),
                );
            });

        ui.label("Fault Survival & Shortest-Corridor Routing");
        Plot::new(crate::framework::scoped_panel_id(ui, "physarum_fault_plot"))
            .height(plot_h)
            .legend(Legend::default())
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new("Pairwise Fault Survival", PlotPoints::from(fault_pts))
                        .color(col_fault)
                        .width(2.0_f32),
                );
                plot_ui.line(
                    Line::new("Corridor / Pair Connectivity", PlotPoints::from(focus_pts))
                        .color(col_focus)
                        .width(1.8_f32),
                );
            });
    }
}

type BenchmarkPlotQuad = (Vec<[f64; 2]>, Vec<[f64; 2]>, Vec<[f64; 2]>, Vec<[f64; 2]>);

impl PhysarumTool {
    fn contrast_line_color(base: egui::Color32, dark_mode: bool) -> egui::Color32 {
        if dark_mode {
            let rgb = ensure_dark_canvas_contrast([base.r(), base.g(), base.b()]);
            egui::Color32::from_rgb(rgb[0], rgb[1], rgb[2])
        } else {
            base
        }
    }

    fn build_plot_series(&self) -> BenchmarkPlotQuad {
        let cap = self.history.len().max(1);
        let mut cost = Vec::with_capacity(cap);
        let mut stretch = Vec::with_capacity(cap);
        let mut fault = Vec::with_capacity(cap);
        let mut focus = Vec::with_capacity(cap);
        for s in &self.history {
            cost.push([s.time, s.cost_ratio]);
            stretch.push([s.time, s.mean_route_stretch]);
            fault.push([s.time, s.pairwise_fault_survival]);
            focus.push([s.time, s.corridor_focus]);
        }
        (cost, stretch, fault, focus)
    }

    pub(crate) fn collect_benchmark_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let (cost, stretch, fault, focus) = self.build_plot_series();
        let mass: Vec<[f64; 2]> = self
            .history
            .iter()
            .map(|s| [s.time, s.total_trail_mass])
            .collect();
        vec![
            ("Wire / MST Cost Ratio".to_string(), cost),
            ("Mean Dijkstra Stretch".to_string(), stretch),
            ("Pairwise Fault Survival".to_string(), fault),
            ("Corridor / Pair Connectivity".to_string(), focus),
            ("Total Trail Mass".to_string(), mass),
        ]
    }
}
