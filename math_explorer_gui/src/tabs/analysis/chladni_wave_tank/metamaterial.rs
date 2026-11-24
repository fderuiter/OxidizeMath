//! Condensed-matter Bravais phononic crystal scatterers, AST potential refraction,
//! and Debye/Einstein phonon dispersion overlays (`#[cfg(feature = "physics")]`).

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

// [cite:graph_parameters_rust]
// [cite:mmwave_radiotherapy_setup]

use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints, VLine};
use math_commons::ast::{parse_equation, CompiledExpr};
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::physics::solid_state::lattice::{
    BodyCenteredCubic, CrystalSystem, FaceCenteredCubic, SimpleCubic, UnitCell,
};
use math_explorer::physics::solid_state::phonons::{
    acoustic_dispersion, debye_heat_capacity_low_temp, optical_dispersion,
};

use super::sculpt::{encode_bounded_sculpt_rle, AcousticCellRole};
use super::{ChladniWaveTankTool, SecondaryPlotMode, GRID_SIZE};
use crate::app::SemanticPalette;
use crate::framework::rasterize_disk_stroke;
use crate::reflective_ui::render_copyable_metric;

/// 2D-projected Bravais crystal symmetry stamped as phononic scatterers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum BravaisLatticeKind {
    /// Simple Cubic square lattice (`1` atom per primitive cell).
    Simple,
    /// Body-Centered Cubic centered lattice (`2` projected sites per cell).
    BodyCentered,
    /// Face-Centered Cubic dense lattice (`4` projected sites per cell).
    #[default]
    FaceCentered,
}

impl BravaisLatticeKind {
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Simple => "Simple Cubic (SC)",
            Self::BodyCentered => "Body-Centered (BCC)",
            Self::FaceCentered => "Face-Centered (FCC)",
        }
    }

    pub(crate) fn generate_unit_cell(self, a: f64) -> UnitCell {
        match self {
            Self::Simple => SimpleCubic.generate(a),
            Self::BodyCentered => BodyCenteredCubic.generate(a),
            Self::FaceCentered => FaceCenteredCubic.generate(a),
        }
    }
}

const DEFAULT_LUNEBURG_FORMULA: &str =
    "sqrt(max(0.55, 2.0 - ((x - 0.44)^2 + (y - 0.5)^2) / 0.045))";

/// Runtime state for Bravais crystal stamping, AST refractive potentials, and phonon dispersion.
#[derive(Debug, Clone)]
pub(crate) struct MetamaterialWorkbench {
    pub lattice_kind: BravaisLatticeKind,
    pub lattice_constant: f64,
    pub scatterer_radius: f64,
    pub rigid_scatterers: bool,
    pub ast_enabled: bool,
    pub ast_formula: String,
    pub ast_compiled: Option<CompiledExpr>,
    pub ast_error: Option<String>,
    pub phonon_sound_speed: f64,
    pub optical_omega_e: f64,
    pub debye_temp_ratio: f64,
}

impl Default for MetamaterialWorkbench {
    fn default() -> Self {
        let mut wb = Self {
            lattice_kind: BravaisLatticeKind::FaceCentered,
            lattice_constant: 20.0,
            scatterer_radius: 2.0,
            rigid_scatterers: false,
            ast_enabled: false,
            ast_formula: DEFAULT_LUNEBURG_FORMULA.to_string(),
            ast_compiled: None,
            ast_error: None,
            phonon_sound_speed: 1.25,
            optical_omega_e: 24.0,
            debye_temp_ratio: 0.25,
        };
        wb.recompile_ast();
        wb
    }
}

impl MetamaterialWorkbench {
    pub(crate) fn set_ast_formula(&mut self, expr_text: &str) {
        self.ast_formula = expr_text.to_string();
        self.recompile_ast();
    }

    pub(crate) fn recompile_ast(&mut self) {
        match parse_equation(&self.ast_formula) {
            Ok((_lhs, expr)) => match CompiledExpr::compile_with_vars(&expr, &["x", "y"]) {
                Ok(compiled) => {
                    self.ast_compiled = Some(compiled);
                    self.ast_error = None;
                }
                Err(err) => {
                    self.ast_compiled = None;
                    self.ast_error = Some(format!("Compile error: {err:?}"));
                }
            },
            Err(err) => {
                self.ast_compiled = None;
                self.ast_error = Some(err.user_message());
            }
        }
    }

    /// Evaluates the precompiled 2D AST refractive index $n(x, y)$ onto `wave_speed_mask`
    /// ($s = 1 / n$) with zero heap allocations.
    pub(crate) fn apply_ast_to_wave_speed_mask(
        &self,
        wave_speed_mask: &mut [f64],
        width: usize,
        height: usize,
    ) {
        if !self.ast_enabled {
            return;
        }
        let Some(compiled) = &self.ast_compiled else {
            return;
        };
        let denom_x = (width.saturating_sub(1)).max(1) as f64;
        let denom_y = (height.saturating_sub(1)).max(1) as f64;
        for y in 0..height {
            let yn = (y as f64) / denom_y;
            for x in 0..width {
                let xn = (x as f64) / denom_x;
                let raw = compiled.eval_2d(xn, yn, &[]);
                if raw.is_finite() {
                    let n_eff = raw.clamp(0.45, 3.50);
                    wave_speed_mask[flatten_2d_index(x, y, width)] = 1.0 / n_eff;
                }
            }
        }
    }
}

impl ChladniWaveTankTool {
    /// Projects Bravais `UnitCell` atomic sites onto a 2D phononic crystal slab and stamps RLE roles.
    pub(crate) fn stamp_bravais_crystal(&mut self) {
        let a = self.metamaterial.lattice_constant.clamp(10.0, 36.0);
        let cell = self.metamaterial.lattice_kind.generate_unit_cell(a);
        let a1 = [cell.lattice_vectors[0].x, cell.lattice_vectors[0].y];
        let a2 = [cell.lattice_vectors[1].x, cell.lattice_vectors[1].y];
        let r = (self.metamaterial.scatterer_radius.round() as i32).clamp(1, 5);
        let role = if self.metamaterial.rigid_scatterers {
            AcousticCellRole::RigidClamp as u8
        } else {
            AcousticCellRole::PhononicScatterer as u8
        };

        let cols = ((56.0 / a).floor() as usize).clamp(2, 6);
        let rows = ((84.0 / a).floor() as usize).clamp(2, 6);
        let x_origin = 30.0_f64;
        let y_origin = 0.5 * ((GRID_SIZE as f64) - (rows as f64) * a);

        for iy in 0..rows {
            for ix in 0..cols {
                let base_x = x_origin + (ix as f64) * a1[0] + (iy as f64) * a2[0];
                let base_y = y_origin + (ix as f64) * a1[1] + (iy as f64) * a2[1];
                for atom in &cell.atomic_positions {
                    let sx = (base_x + atom.x).round() as i32;
                    let sy = (base_y + atom.y).round() as i32;
                    if (14..=(GRID_SIZE as i32 - 14)).contains(&sx)
                        && (12..=(GRID_SIZE as i32 - 12)).contains(&sy)
                    {
                        rasterize_disk_stroke(
                            (sx, sy),
                            (sx, sy),
                            r,
                            (GRID_SIZE, GRID_SIZE),
                            |gx, gy| {
                                self.sculpt_mask[flatten_2d_index(gx, gy, GRID_SIZE)] = role;
                            },
                        );
                    }
                }
            }
        }
        self.state.sculpt_rle = encode_bounded_sculpt_rle(&self.sculpt_mask, GRID_SIZE, GRID_SIZE);
        self.rebuild_masks();
        self.solver.reset_detector();
        self.analysis_dirty = true;
        self.state_rev = self.state_rev.wrapping_add(1);
    }

    /// Quantizes the active 2D AST refractive formula into `sculpt_rle` for URL sharing.
    pub(crate) fn stamp_ast_potential_to_sculpt(&mut self) {
        let Some(compiled) = &self.metamaterial.ast_compiled else {
            return;
        };
        let denom = (GRID_SIZE - 1) as f64;
        for y in 0..GRID_SIZE {
            let yn = (y as f64) / denom;
            for x in 0..GRID_SIZE {
                let xn = (x as f64) / denom;
                let n_val = compiled.eval_2d(xn, yn, &[]);
                let role = AcousticCellRole::from_refractive_index(n_val);
                self.sculpt_mask[flatten_2d_index(x, y, GRID_SIZE)] = role as u8;
            }
        }
        self.state.sculpt_rle = encode_bounded_sculpt_rle(&self.sculpt_mask, GRID_SIZE, GRID_SIZE);
        self.rebuild_masks();
        self.solver.reset_detector();
        self.analysis_dirty = true;
        self.state_rev = self.state_rev.wrapping_add(1);
    }

    pub(super) fn draw_physics_metamaterial_controls(&mut self, ui: &mut egui::Ui) {
        egui::CollapsingHeader::new("Phononic Crystal & AST Metamaterial")
            .default_open(false)
            .show(ui, |ui| {
                self.draw_bravais_controls(ui);
                ui.separator();
                self.draw_ast_potential_controls(ui);
                ui.separator();
                self.draw_phonon_model_controls(ui);
            });
    }

    fn draw_bravais_controls(&mut self, ui: &mut egui::Ui) {
        egui::ComboBox::from_id_salt("chladni_bravais_kind")
            .selected_text(self.metamaterial.lattice_kind.label())
            .show_ui(ui, |ui| {
                for kind in [
                    BravaisLatticeKind::Simple,
                    BravaisLatticeKind::BodyCentered,
                    BravaisLatticeKind::FaceCentered,
                ] {
                    ui.selectable_value(&mut self.metamaterial.lattice_kind, kind, kind.label());
                }
            });
        ui.add(
            egui::Slider::new(&mut self.metamaterial.lattice_constant, 12.0..=30.0)
                .text("Lattice a"),
        );
        ui.add(
            egui::Slider::new(&mut self.metamaterial.scatterer_radius, 1.0..=4.0).text("Atom R"),
        );
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.metamaterial.rigid_scatterers, "Rigid Bragg Pins");
            if ui.button("Stamp Crystal").clicked() {
                self.stamp_bravais_crystal();
            }
        });
    }

    fn draw_ast_potential_controls(&mut self, ui: &mut egui::Ui) {
        let mut mask_dirty = ui
            .checkbox(
                &mut self.metamaterial.ast_enabled,
                "Live AST Refractive n(x, y)",
            )
            .changed();
        ui.horizontal_wrapped(|ui| {
            for (label, expr) in [
                ("Luneburg", DEFAULT_LUNEBURG_FORMULA),
                ("Waveguide", "1.85 - 1.4 * (y - 0.5)^2"),
                (
                    "2-Barrier",
                    "1.0 + 1.5 * exp(-180.0 * (x - 0.36)^2) + 1.5 * exp(-180.0 * (x - 0.52)^2)",
                ),
            ] {
                if ui.small_button(label).clicked() {
                    self.metamaterial.ast_enabled = true;
                    self.metamaterial.set_ast_formula(expr);
                    mask_dirty = true;
                }
            }
        });
        if ui
            .text_edit_singleline(&mut self.metamaterial.ast_formula)
            .changed()
        {
            self.metamaterial.recompile_ast();
            mask_dirty = self.metamaterial.ast_enabled;
        }
        if let Some(err) = &self.metamaterial.ast_error {
            ui.colored_label(SemanticPalette::from_visuals(ui.visuals()).error, err);
        } else if ui.small_button("Bake AST to URL RLE").clicked() {
            self.stamp_ast_potential_to_sculpt();
        }
        if mask_dirty {
            self.rebuild_masks();
            self.solver.reset_detector();
            self.analysis_dirty = true;
            self.state_rev = self.state_rev.wrapping_add(1);
        }
    }

    fn draw_phonon_model_controls(&mut self, ui: &mut egui::Ui) {
        ui.add(
            egui::Slider::new(&mut self.metamaterial.phonon_sound_speed, 0.2..=4.0)
                .text("Debye v_s"),
        );
        ui.add(
            egui::Slider::new(&mut self.metamaterial.optical_omega_e, 2.0..=50.0)
                .text("Einstein ω_E"),
        );
        if ui.small_button("Show Phonon ω(|k|)").clicked() {
            self.state.secondary_plot = SecondaryPlotMode::PhononDispersion;
        }
        let cv = debye_heat_capacity_low_temp(self.metamaterial.debye_temp_ratio, 1.0, 1.0, 1.0);
        render_copyable_metric(ui, "Debye C_v / (N k_B)", &format!("{cv:.3}"));
    }

    fn build_phonon_dispersion_series(
        &self,
    ) -> (
        PlotPoints<'static>,
        PlotPoints<'static>,
        PlotPoints<'static>,
    ) {
        let max_k = self.radial_spectrum.len().max(2);
        let peak_s = self
            .radial_spectrum
            .iter()
            .skip(1)
            .copied()
            .filter(|v| v.is_finite())
            .fold(1.0e-9_f64, f64::max);
        let top_omega = acoustic_dispersion(max_k as f64, self.metamaterial.phonon_sound_speed)
            .max(self.metamaterial.optical_omega_e)
            .max(1.0);
        let ac_pts: PlotPoints = (0..max_k)
            .map(|k| {
                let kf = k as f64;
                [
                    kf,
                    acoustic_dispersion(kf, self.metamaterial.phonon_sound_speed),
                ]
            })
            .collect();
        let opt_pts: PlotPoints = (0..max_k)
            .map(|k| {
                let kf = k as f64;
                [
                    kf,
                    optical_dispersion(kf, self.metamaterial.optical_omega_e),
                ]
            })
            .collect();
        let fft_pts: PlotPoints = self
            .radial_spectrum
            .iter()
            .enumerate()
            .skip(1)
            .map(|(k, &val)| {
                let s = if val.is_finite() {
                    (val / peak_s) * 0.85 * top_omega
                } else {
                    0.0
                };
                [k as f64, s]
            })
            .collect();
        (ac_pts, opt_pts, fft_pts)
    }

    pub(super) fn draw_phonon_dispersion_plot(&mut self, ui: &mut egui::Ui) {
        let palette = SemanticPalette::from_visuals(ui.visuals());
        ui.label("Phonon Dispersion ω(|k|) vs. 2D FFT Wavenumber Spectrum");
        let (ac_pts, opt_pts, fft_pts) = self.build_phonon_dispersion_series();
        Plot::new("chladni_phonon_dispersion_plot")
            .allow_boxed_zoom(false)
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new("Acoustic ω_ac = v_s |k|", ac_pts)
                        .color(palette.series_color(0))
                        .width(2.1_f32),
                );
                plot_ui.line(
                    Line::new("Optical ω_opt = ω_E", opt_pts)
                        .color(palette.series_color(1))
                        .width(1.9_f32),
                );
                plot_ui.line(
                    Line::new("Scaled 2D FFT S(|k|)", fft_pts)
                        .color(palette.series_color(2))
                        .width(1.6_f32),
                );
                if self.dominant_peak.k_radial > 0.0 {
                    plot_ui.vline(
                        VLine::new("Dominant |k|", self.dominant_peak.k_radial)
                            .color(palette.warning),
                    );
                }
                self.callout_manager.render_on_plot(plot_ui);
            });
    }
}
