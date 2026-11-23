#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Side-panel controls, Abelian group proof actions, seismic sonification bar, and copyable
//! scientific readouts for `AbelianSandpileStudio`.

use eframe::egui;
use egui_plot::commands::{CommandRegistryData, CommandTrigger};
use math_explorer::physics::stat_mech::sandpile::{
    kirchhoff_determinant_bareiss, AvalancheObservable, SandpileElement, SandpileGrid,
    MAX_SANDPILE_DIM,
};
use math_explorer::pure_math::algebra::{generate_left_coset, is_subgroup, Semigroup, Zn};

use super::{
    AbelianSandpileStudio, SandpileBrush, SandpileColormap, SandpilePreset, SandpileViewMode,
    MIN_SANDPILE_DIM,
};
use crate::reflective_ui::render_copyable_metric;

impl AbelianSandpileStudio {
    pub(super) fn draw_controls_panel(&mut self, parent_ui: &mut egui::Ui) {
        self.register_shortcuts(parent_ui.ctx());
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "abelian_sandpile_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("Abelian Sandpile Studio");
                ui.separator();
                self.draw_preset_and_transport_controls(ui);
                ui.separator();
                self.draw_lattice_and_brush_controls(ui);
                ui.separator();
                self.draw_abelian_group_lab_controls(ui);
                ui.separator();
                crate::audio::render_sonification_panel(
                    ui,
                    "abelian_sandpile",
                    &self.latest_sonification,
                    self.paused,
                );
                ui.separator();
                self.draw_telemetry_readouts(ui);
            });
        });
    }

    fn register_shortcuts(&self, ctx: &egui::Context) {
        CommandRegistryData::register(
            ctx,
            "Abelian Sandpile: Drop Grain Brush",
            "Drop grains at the active lattice cursor using the selected brush geometry",
            CommandTrigger::Shortcut(egui::Modifiers::NONE, egui::Key::Enter),
            true,
            "Solid State Physics",
        );
    }

    fn draw_preset_and_transport_controls(&mut self, ui: &mut egui::Ui) {
        ui.label("Starter Preset:");
        let mut selected = self.state.preset;
        egui::ComboBox::from_id_salt("sandpile_preset_selector")
            .selected_text(selected.label())
            .show_ui(ui, |ui| {
                for p in [
                    SandpilePreset::CriticalAvalanches1F,
                    SandpilePreset::CreutzIdentityCrystal63,
                    SandpilePreset::AbelianGroupIdentityProof,
                ] {
                    ui.selectable_value(&mut selected, p, p.label());
                }
            });
        if selected != self.state.preset {
            self.apply_preset(selected);
        }

        ui.add_space(6.0);
        ui.horizontal_wrapped(|ui| {
            let play_label = if self.paused { "▶ Play" } else { "⏸ Pause" };
            if ui.button(play_label).clicked() {
                self.paused = !self.paused;
            }
            if ui.button("⏭ Step").clicked() {
                self.advance_simulation_step();
            }
            if ui.button("⏳ Drop Brush").clicked() {
                self.apply_brush_at(self.state.cursor_x, self.state.cursor_y);
            }
            if ui.button("↺ Reset").clicked() {
                self.rebuild_for_current_state();
            }
        });
    }

    fn draw_lattice_and_brush_controls(&mut self, ui: &mut egui::Ui) {
        let mut rebuild = false;
        let mut changed = false;

        ui.horizontal(|ui| {
            ui.label("Grid N×N:");
            let prev_n = self.state.grid_size;
            let slider = egui::Slider::new(
                &mut self.state.grid_size,
                MIN_SANDPILE_DIM..=MAX_SANDPILE_DIM,
            )
            .clamping(egui::SliderClamping::Always);
            if ui.add(slider).changed() && self.state.grid_size != prev_n {
                self.state.cursor_x = self.state.grid_size / 2;
                self.state.cursor_y = self.state.grid_size / 2;
                rebuild = true;
            }
            if ui.small_button("63×63").clicked() && self.state.grid_size != 63 {
                self.state.grid_size = 63;
                self.state.cursor_x = 31;
                self.state.cursor_y = 31;
                rebuild = true;
            }
        });

        ui.horizontal(|ui| {
            ui.label("Drops/Frame:");
            changed |= ui
                .add(
                    egui::Slider::new(&mut self.state.drops_per_frame, 1..=64)
                        .clamping(egui::SliderClamping::Always),
                )
                .changed();
            changed |= ui
                .checkbox(&mut self.state.animate_wavefronts, "Wavefront")
                .changed();
        });

        changed |= self.draw_brush_selector(ui);
        changed |= self.draw_view_and_observable_selectors(ui);

        if rebuild {
            self.state_rev = self.state_rev.wrapping_add(1);
            self.rebuild_for_current_state();
        } else if changed {
            self.state_rev = self.state_rev.wrapping_add(1);
            self.refresh_analysis_and_audio();
        }
    }

    fn draw_brush_selector(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        let prev_brush = self.state.brush;
        egui::ComboBox::from_id_salt("sandpile_brush")
            .selected_text(self.state.brush.label())
            .show_ui(ui, |ui| {
                for b in [
                    SandpileBrush::SingleGrain,
                    SandpileBrush::CriticalTower,
                    SandpileBrush::SymmetricCross,
                ] {
                    ui.selectable_value(&mut self.state.brush, b, b.label());
                }
            });
        changed |= self.state.brush != prev_brush;

        ui.horizontal(|ui| {
            ui.label("Brush Grains:");
            changed |= ui
                .add(
                    egui::Slider::new(&mut self.state.brush_grains, 1..=16)
                        .clamping(egui::SliderClamping::Always),
                )
                .changed();
        });
        changed
    }

    fn draw_view_and_observable_selectors(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        let prev_view = self.state.view_mode;
        egui::ComboBox::from_id_salt("sandpile_view_mode")
            .selected_text(self.state.view_mode.label())
            .show_ui(ui, |ui| {
                for m in [
                    SandpileViewMode::GrainHeights,
                    SandpileViewMode::DharBurningWaves,
                ] {
                    ui.selectable_value(&mut self.state.view_mode, m, m.label());
                }
            });
        changed |= self.state.view_mode != prev_view;

        let prev_cmap = self.state.colormap;
        egui::ComboBox::from_id_salt("sandpile_colormap")
            .selected_text(self.state.colormap.label())
            .show_ui(ui, |ui| {
                for c in [
                    SandpileColormap::CriticalThermal,
                    SandpileColormap::ApollonianHighContrast,
                    SandpileColormap::BurningWavePhase,
                ] {
                    ui.selectable_value(&mut self.state.colormap, c, c.label());
                }
            });
        changed |= self.state.colormap != prev_cmap;

        let prev_obs = self.state.observable;
        egui::ComboBox::from_id_salt("sandpile_observable")
            .selected_text(format!("Fit Observable: {:?}", self.state.observable))
            .show_ui(ui, |ui| {
                for o in [
                    AvalancheObservable::Size,
                    AvalancheObservable::Area,
                    AvalancheObservable::Duration,
                ] {
                    ui.selectable_value(&mut self.state.observable, o, format!("{o:?}"));
                }
            });
        changed |= self.state.observable != prev_obs;
        changed |= ui
            .checkbox(
                &mut self.state.show_burning_overlay,
                "Highlight Deepest Burn Wave",
            )
            .changed();
        changed
    }

    fn draw_abelian_group_lab_controls(&mut self, ui: &mut egui::Ui) {
        ui.label(egui::RichText::new("Abelian Sandpile Group Lab").strong());
        ui.horizontal_wrapped(|ui| {
            if ui.button("🔷 Creutz Identity e").clicked() {
                self.compute_creutz_identity_action();
            }
            if ui.button("📥 Store r").clicked() {
                self.store_register_r_action();
            }
            if ui.button("✔ Verify (r + e)° = r").clicked() {
                self.verify_identity_law_action();
            }
            if ui.button("🔄 Invert r⁻¹").clicked() {
                self.invert_current_state_action();
            }
            if ui.button("🔥 Dhar Burn Test").clicked() {
                self.burning_result = self.grid.dhar_burning_test();
                self.state.view_mode = SandpileViewMode::DharBurningWaves;
            }
        });
        if !self.group_status.is_empty() {
            ui.small(&self.group_status);
        }
        if !self.subgroup_summary.is_empty() {
            ui.small(&self.subgroup_summary);
        }
    }

    fn draw_telemetry_readouts(&self, ui: &mut egui::Ui) {
        let last = self.grid.telemetry().last_stats();
        render_copyable_metric(
            ui,
            "Avalanches Recorded",
            &self.grid.telemetry().total_avalanches().to_string(),
        );
        render_copyable_metric(
            ui,
            "Last Topples / Area / Waves",
            &format!("{} / {} / {}", last.topples, last.area, last.duration),
        );
        if let Some(fit) = &self.latest_fit {
            let b = fit.coefficients()[1];
            render_copyable_metric(ui, "Power-Law Exponent b", &format!("{b:.4}"));
            render_copyable_metric(ui, "Power-Law Fit R²", &format!("{:.4}", fit.r_squared));
        }
        render_copyable_metric(
            ui,
            "Dhar Recurrence",
            &format!(
                "{} ({}/{} burned, depth {})",
                if self.burning_result.is_recurrent {
                    "Recurrent"
                } else {
                    "Transient"
                },
                self.burning_result.burned_cells,
                self.burning_result.total_cells,
                self.burning_result.max_wave_depth
            ),
        );
    }

    pub(crate) fn compute_creutz_identity_action(&mut self) {
        let topples = self.grid.load_creutz_identity();
        self.identity_cache.copy_from_slice(self.grid.heights());
        self.paused = true;
        self.group_status = format!(
            "Loaded Creutz identity e ({}×{}, {topples} topples)",
            self.state.grid_size, self.state.grid_size
        );
        self.refresh_analysis_and_audio();
    }

    pub(crate) fn store_register_r_action(&mut self) {
        let _ = self.grid.stabilize_wavefront();
        self.register_r.copy_from_slice(self.grid.heights());
        self.group_status = "Stored current stabilized configuration in register r".into();
        self.refresh_analysis_and_audio();
    }

    pub(crate) fn verify_identity_law_action(&mut self) {
        let _ = self.grid.stabilize_wavefront();
        let r_before = self.grid.heights().to_vec();
        self.register_r.copy_from_slice(&r_before);
        if self.identity_cache.len() != r_before.len()
            || self.identity_cache.iter().all(|&v| v == 0)
        {
            self.capture_identity_into_cache();
        }
        let topples = self
            .grid
            .compose_with_slice(&self.identity_cache)
            .unwrap_or(0);
        let matches = self.grid.heights() == r_before.as_slice();
        let burn = self.grid.dhar_burning_test();
        self.group_status = if matches && burn.is_recurrent {
            format!("Verified (r + e)° == r ({topples} topples, Dhar recurrent)")
        } else {
            format!("Projected (z + e)° onto recurrent coset ({topples} topples)")
        };
        self.refresh_analysis_and_audio();
    }

    pub(crate) fn invert_current_state_action(&mut self) {
        let r_orig = self.grid.heights().to_vec();
        let inv_topples = self.grid.invert_in_place();
        let r_inv = self.grid.heights().to_vec();
        let mut check_grid =
            SandpileGrid::new(self.state.grid_size, self.state.grid_size, Some(1)).unwrap();
        let _ = check_grid.set_heights(&r_orig);
        let _ = check_grid.compose_with_slice(&r_inv);
        if self.identity_cache.len() != r_orig.len() || self.identity_cache.iter().all(|&v| v == 0)
        {
            let mut id_grid =
                SandpileGrid::new(self.state.grid_size, self.state.grid_size, Some(1)).unwrap();
            id_grid.load_creutz_identity();
            self.identity_cache.copy_from_slice(id_grid.heights());
        }
        let exact_e = check_grid.heights() == self.identity_cache.as_slice();
        self.paused = true;
        self.group_status =
            format!("Computed r⁻¹ ({inv_topples} topples) • (r + r⁻¹)° == e: {exact_e}");
        self.refresh_analysis_and_audio();
    }

    pub(super) fn refresh_zn_subgroup_summary(&mut self) {
        let gen = SandpileElement::<2, 2>::new([[3, 1], [1, 3]]);
        let subgroup = gen.cyclic_subgroup(64);
        let valid_sub = is_subgroup(&subgroup);
        let coset = generate_left_coset(&subgroup, &SandpileElement::<2, 2>::new([[3, 2], [2, 1]]));
        let z_sum = Zn::<24>::operate(&Zn::<24>::new(3), &Zn::<24>::new(5));
        let hom_ok = gen.zn_homomorphism(z_sum)
            == SandpileElement::operate(
                &gen.zn_homomorphism(Zn::<24>::new(3)),
                &gen.zn_homomorphism(Zn::<24>::new(5)),
            );
        let det_curr = kirchhoff_determinant_bareiss(self.state.grid_size, self.state.grid_size)
            .map(|d| format!(" • |G|={d}"))
            .unwrap_or_else(|_| " • |G_2x2|=192, |G_3x3|=100352".into());
        self.subgroup_summary = format!(
            "|<g>|={} (subgroup={valid_sub}, coset={}, Zn hom={hom_ok}){det_curr}",
            subgroup.len(),
            coset.len()
        );
    }
}
