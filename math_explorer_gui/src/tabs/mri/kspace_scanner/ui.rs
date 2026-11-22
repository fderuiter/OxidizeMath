//! Side-panel parameter controls and quantitative readouts for [`super::tool::MriKSpaceScannerTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::phantom::is_biology_phantom_available;
use super::state::{KSpaceBrushMode, KSpaceMaskFilter, KSpaceTrajectory, PhantomSource};
use super::tool::MriKSpaceScannerTool;
use super::trajectory::evaluate_bin_scanner_physics;
use super::{KSPACE_DIM, KSPACE_LEN};
use eframe::egui;
use math_commons::math_kernel::types::flatten_2d_index;

impl MriKSpaceScannerTool {
    fn draw_phantom_controls(&mut self, ui: &mut egui::Ui) {
        egui::CollapsingHeader::new("Phantom & Spin-Echo Contrast")
            .default_open(true)
            .show(ui, |ui| {
                egui::ComboBox::from_label("Phantom")
                    .selected_text(self.state.phantom.label())
                    .show_ui(ui, |ui| {
                        for src in [
                            PhantomSource::SheppLogan,
                            PhantomSource::TuringMorphogenesis,
                        ] {
                            if ui
                                .selectable_value(&mut self.state.phantom, src, src.label())
                                .changed()
                            {
                                self.mark_phantom_dirty();
                            }
                        }
                    });
                if self.state.phantom == PhantomSource::TuringMorphogenesis {
                    if is_biology_phantom_available() {
                        if ui
                            .add(
                                egui::Slider::new(&mut self.state.turing_steps, 0..=2000)
                                    .text("Turing Steps"),
                            )
                            .changed()
                        {
                            self.mark_phantom_dirty();
                        }
                        if ui.button("⏭ Evolve Morphogenesis (+40)").clicked() {
                            self.state.turing_steps = (self.state.turing_steps + 40).min(2000);
                            self.mark_phantom_dirty();
                        }
                    } else {
                        ui.small("Biology feature inactive — using Shepp-Logan fallback.");
                    }
                }
                let mut se_changed = false;
                se_changed |= ui
                    .add(egui::Slider::new(&mut self.state.tr_s, 0.1..=5.0).text("TR [s]"))
                    .changed();
                se_changed |= ui
                    .add(egui::Slider::new(&mut self.state.te_s, 0.005..=0.5).text("TE [s]"))
                    .changed();
                se_changed |= ui
                    .add(egui::Slider::new(&mut self.state.t1_scale, 0.25..=3.0).text("T1 Scale"))
                    .changed();
                se_changed |= ui
                    .add(egui::Slider::new(&mut self.state.t2_scale, 0.25..=3.0).text("T2 Scale"))
                    .changed();
                if se_changed {
                    self.mark_phantom_dirty();
                }
            });
    }

    fn draw_trajectory_controls(&mut self, ui: &mut egui::Ui) {
        egui::ComboBox::from_label("Trajectory")
            .selected_text(self.state.trajectory.label())
            .show_ui(ui, |ui| {
                for traj in [
                    KSpaceTrajectory::FullCartesian,
                    KSpaceTrajectory::CartesianLines,
                    KSpaceTrajectory::RadialSpokes,
                    KSpaceTrajectory::ArchimedeanSpiral,
                    KSpaceTrajectory::LorenzChaotic,
                ] {
                    if ui
                        .selectable_value(&mut self.state.trajectory, traj, traj.label())
                        .changed()
                    {
                        self.mark_parametric_mask_dirty();
                    }
                }
            });
        if self.draw_trajectory_sliders(ui) {
            self.mark_parametric_mask_dirty();
        }
    }

    fn draw_trajectory_sliders(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        match self.state.trajectory {
            KSpaceTrajectory::FullCartesian => {
                ui.small("Full 64×64 Nyquist grid (100% baseline).");
            }
            KSpaceTrajectory::CartesianLines => {
                changed |= ui
                    .add(
                        egui::Slider::new(&mut self.state.cartesian_stride, 1..=8)
                            .text("Line Stride (R)"),
                    )
                    .changed();
            }
            KSpaceTrajectory::RadialSpokes => {
                changed |= ui
                    .add(egui::Slider::new(&mut self.state.radial_spokes, 4..=128).text("Spokes"))
                    .changed();
            }
            KSpaceTrajectory::ArchimedeanSpiral => {
                changed |= ui
                    .add(
                        egui::Slider::new(&mut self.state.spiral_turns, 2.0..=32.0)
                            .text("Spiral Turns"),
                    )
                    .changed();
            }
            KSpaceTrajectory::LorenzChaotic => {
                changed |= ui
                    .add(
                        egui::Slider::new(&mut self.state.lorenz_steps, 64..=8192)
                            .text("Lorenz Steps"),
                    )
                    .changed();
                changed |= ui
                    .add(
                        egui::Slider::new(&mut self.state.lorenz_rho, 10.0..=60.0)
                            .text("Lorenz ρ"),
                    )
                    .changed();
            }
        }
        changed
    }

    fn draw_filter_controls(&mut self, ui: &mut egui::Ui) {
        egui::ComboBox::from_label("Frequency Mask")
            .selected_text(self.state.mask_filter.label())
            .show_ui(ui, |ui| {
                for filt in [
                    KSpaceMaskFilter::TrajectoryOnly,
                    KSpaceMaskFilter::Keyhole,
                    KSpaceMaskFilter::HighPass,
                    KSpaceMaskFilter::SeededRandom,
                ] {
                    if ui
                        .selectable_value(&mut self.state.mask_filter, filt, filt.label())
                        .changed()
                    {
                        self.mark_parametric_mask_dirty();
                    }
                }
            });
        let mut changed = false;
        match self.state.mask_filter {
            KSpaceMaskFilter::TrajectoryOnly => {}
            KSpaceMaskFilter::Keyhole | KSpaceMaskFilter::HighPass => {
                changed |= ui
                    .add(
                        egui::Slider::new(&mut self.state.cutoff_radius, 1..=45)
                            .text("Cutoff [bins]"),
                    )
                    .changed();
            }
            KSpaceMaskFilter::SeededRandom => {
                changed |= ui
                    .add(
                        egui::Slider::new(&mut self.state.random_keep_ratio, 0.05..=1.0)
                            .text("Keep Ratio"),
                    )
                    .changed();
                ui.horizontal(|ui| {
                    ui.label("RNG Seed");
                    changed |= ui
                        .add(egui::DragValue::new(&mut self.state.rng_seed).speed(1.0))
                        .changed();
                });
            }
        }
        if changed {
            self.mark_parametric_mask_dirty();
        }
    }

    fn draw_brush_controls(&mut self, ui: &mut egui::Ui) {
        egui::CollapsingHeader::new("Tactile k-Space Brush")
            .default_open(true)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.selectable_value(
                        &mut self.state.brush_mode,
                        KSpaceBrushMode::Erase,
                        "🧹 Erase",
                    );
                    ui.selectable_value(
                        &mut self.state.brush_mode,
                        KSpaceBrushMode::Paint,
                        "🖌 Paint",
                    );
                });
                ui.add(
                    egui::Slider::new(&mut self.state.brush_radius, 1..=16)
                        .text("Brush Radius [bins]"),
                );
                ui.horizontal_wrapped(|ui| {
                    if ui.button("Full 64×64").clicked() {
                        self.acquire_full_kspace();
                    }
                    if ui.button("Clear All").clicked() {
                        self.clear_all_kspace();
                    }
                    if self.state.custom_mask_hex.is_some()
                        && ui.button("Reset Brush").clicked()
                    {
                        self.reset_brush_edits();
                    }
                    if ui.button("↺ Reset Scanner").clicked() {
                        self.reset_to_defaults();
                    }
                });
            });
    }

    fn draw_readouts(&self, ui: &mut egui::Ui) {
        let phys = evaluate_bin_scanner_physics(self.reticle_row, self.reticle_col);
        let idx = flatten_2d_index(self.reticle_col, self.reticle_row, KSPACE_DIM);
        let sampled_label = if self.active_mask[idx] {
            "Sampled"
        } else {
            "Erased"
        };
        ui.label(format!("PSNR: {:.2} dB", self.recon.psnr_db));
        ui.label(format!(
            "Norm L2 Error: {:.2e}",
            self.recon.normalized_l2_error
        ));
        ui.label(format!(
            "k-Space Coverage: {:.1}% ({}/{KSPACE_LEN})",
            self.recon.sampling_pct, self.recon.sampled_bins
        ));
        ui.label(format!("2D FFT Pass: {:.2} ms", self.last_recon_ms));
        ui.separator();
        ui.small(format!(
            "Reticle ({}, {}): {sampled_label}",
            self.reticle_row, self.reticle_col
        ));
        ui.small(format!(
            "|k| = {:.1} m⁻¹ (kx={:.1}, ky={:.1})",
            phys.k_mag_per_m, phys.k_vec_per_m.x, phys.k_vec_per_m.y
        ));
        ui.small(format!(
            "Spin Phase φ(r_ref): {:.2} rad",
            phys.reference_phase_rad
        ));
    }

    fn draw_side_panel(&mut self, ui: &mut egui::Ui) {
        let panel_id = crate::framework::scoped_panel_id(ui, "mri_kspace_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("MRI 2D k-Space Control");
                ui.separator();
                self.draw_phantom_controls(ui);
                egui::CollapsingHeader::new("Trajectory & Frequency Mask")
                    .default_open(true)
                    .show(ui, |ui| {
                        self.draw_trajectory_controls(ui);
                        ui.separator();
                        self.draw_filter_controls(ui);
                    });
                self.draw_brush_controls(ui);
                ui.separator();
                self.draw_readouts(ui);
            });
        });
    }

    pub(crate) fn show_scanner_inside(&mut self, ui: &mut egui::Ui) {
        let high_contrast = crate::app::ThemeMode::from_visuals(ui.visuals())
            == crate::app::ThemeMode::HighContrast;
        let _ = self.recompute_if_dirty(high_contrast);
        self.ensure_textures(ui.ctx());
        self.draw_side_panel(ui);
        let _ = self.recompute_if_dirty(high_contrast);
        self.draw_central_quad(ui);
    }
}
