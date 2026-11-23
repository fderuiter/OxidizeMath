#![cfg_attr(any(), verified(opt_out = "gui_tool"))]
//! Central `SchwarzschildBlackHole` Paczyński-Wiita ISCO capture, Christoffel validation,
//! Doppler-boosted tracer colors, and interactive `NfwCusp` vs. `BurkertCore` caustic overlays.
// [cite:dwarf_galaxy_empirical_dependencies]
// [cite:general_relativity_geodesics]

use eframe::egui;
use egui_plot::{Line, LineStyle, MarkerShape, PlotPoints, Points};
use math_explorer::constants::C;
use math_explorer::physics::astrophysics::{
    evaluate_relativistic_beaming, tangential_critical_radius_kpc,
    validate_christoffel_radial_connection, CentralBlackHoleConfig, HaloProfile, OrbitParticleKind,
};

use crate::reflective_ui::render_copyable_metric;
use crate::volumetric_scene::lensing::build_circle_ring;

use super::clamp_finite;
use super::galactic_orbits_tool::GalacticOrbitSimulator;

/// Interactive `NfwCusp` vs. `BurkertCore` gravitational caustic overlay state on the orbit plane.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct OrbitLensOverlayConfig {
    /// Displays `NfwCusp` vs. `BurkertCore` critical/caustic rings on the midplane canvas.
    pub(crate) enabled: bool,
    /// Repositions `lens_center_kpc` when dragging on the orbit canvas.
    pub(crate) drag_on_canvas: bool,
    /// 2D midplane position `[x_kpc, y_kpc]` of the foreground halo/black-hole lens.
    pub(crate) lens_center_kpc: [f64; 2],
    /// Logarithmic critical surface density $\log_{10}(\Sigma_{\text{crit}} / (M_\odot\,\text{kpc}^{-2}))$.
    pub(crate) log10_sigma_crit: f64,
}

impl Default for OrbitLensOverlayConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            drag_on_canvas: true,
            lens_center_kpc: [0.0, 0.0],
            log10_sigma_crit: 7.40,
        }
    }
}

impl OrbitLensOverlayConfig {
    pub(crate) fn is_default(&self) -> bool {
        *self == Self::default()
    }

    pub(crate) fn is_valid(&self) -> bool {
        self.lens_center_kpc[0].is_finite()
            && (-25.0..=25.0).contains(&self.lens_center_kpc[0])
            && self.lens_center_kpc[1].is_finite()
            && (-25.0..=25.0).contains(&self.lens_center_kpc[1])
            && self.log10_sigma_crit.is_finite()
            && (6.0..=10.5).contains(&self.log10_sigma_crit)
    }

    pub(crate) fn clamp_in_place(&mut self) {
        self.lens_center_kpc[0] = clamp_finite(self.lens_center_kpc[0], -25.0, 25.0, 0.0);
        self.lens_center_kpc[1] = clamp_finite(self.lens_center_kpc[1], -25.0, 25.0, 0.0);
        self.log10_sigma_crit = clamp_finite(self.log10_sigma_crit, 6.0, 10.5, 7.40);
    }
}

pub(crate) fn is_valid_bh_config(bh: &CentralBlackHoleConfig) -> bool {
    bh.mass_solar.is_finite()
        && (1.0e5..=1.0e11).contains(&bh.mass_solar)
        && bh.effective_rs_kpc.is_finite()
        && (0.01..=2.0).contains(&bh.effective_rs_kpc)
        && bh.relativistic_beta_scale.is_finite()
        && (0.0..=0.05).contains(&bh.relativistic_beta_scale)
        && bh.magnetic_energy_density_j_m3.is_finite()
        && (0.0..=50.0).contains(&bh.magnetic_energy_density_j_m3)
}

pub(crate) fn clamp_bh_config(bh: &mut CentralBlackHoleConfig) {
    bh.mass_solar = clamp_finite(bh.mass_solar, 1.0e5, 1.0e11, 1.0e8);
    bh.effective_rs_kpc = clamp_finite(bh.effective_rs_kpc, 0.01, 2.0, 0.12);
    bh.relativistic_beta_scale = clamp_finite(bh.relativistic_beta_scale, 0.0, 0.05, 0.0025);
    bh.magnetic_energy_density_j_m3 = clamp_finite(bh.magnetic_energy_density_j_m3, 0.0, 50.0, 1.0);
}

impl GalacticOrbitSimulator {
    /// Repositions the interactive midplane caustic lens center in kpc.
    pub fn drag_orbit_lens_to(&mut self, xy_kpc: [f64; 2]) {
        let nx = clamp_finite(
            xy_kpc[0],
            -25.0,
            25.0,
            self.cfg.lens_overlay.lens_center_kpc[0],
        );
        let ny = clamp_finite(
            xy_kpc[1],
            -25.0,
            25.0,
            self.cfg.lens_overlay.lens_center_kpc[1],
        );
        if (self.cfg.lens_overlay.lens_center_kpc[0] - nx).abs() > 1.0e-9
            || (self.cfg.lens_overlay.lens_center_kpc[1] - ny).abs() > 1.0e-9
        {
            self.cfg.lens_overlay.lens_center_kpc = [nx, ny];
            self.revision = self.revision.wrapping_add(1).max(1);
        }
    }

    pub(crate) fn compute_caustic_radii_kpc(&self) -> (Option<f64>, Option<f64>) {
        let m200 = 10.0_f64.powf(self.cfg.halo_log10_mass.clamp(8.0, 12.0));
        let scale = self.cfg.halo_scale_kpc.clamp(0.3, 10.0);
        let conc = self.cfg.halo_concentration.clamp(4.0, 25.0);
        let sigma_crit = 10.0_f64.powf(self.cfg.lens_overlay.log10_sigma_crit.clamp(6.0, 10.5));
        let nfw = HaloProfile::NfwCusp {
            virial_mass_solar: m200,
            scale_radius_kpc: scale,
            concentration: conc,
        };
        let burk = HaloProfile::BurkertCore {
            virial_mass_solar: m200,
            core_radius_kpc: scale,
            concentration: conc,
        };
        (
            tangential_critical_radius_kpc(Some(&self.system.baryons), &nfw, sigma_crit, 20.0),
            tangential_critical_radius_kpc(Some(&self.system.baryons), &burk, sigma_crit, 20.0),
        )
    }

    pub(super) fn render_bh_and_caustic_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut rebuild = false;
        ui.separator();
        ui.heading("Schwarzschild Black Hole (Paczyński-Wiita)");
        rebuild |= ui
            .checkbox(
                &mut self.cfg.central_bh.enabled,
                "Enable Central BH & ISCO (3 R_s) Capture",
            )
            .changed();
        if self.cfg.central_bh.enabled {
            let mut log_mbh = self.cfg.central_bh.mass_solar.max(1.0e5).log10();
            if ui
                .add(egui::Slider::new(&mut log_mbh, 6.0..=10.2).text("log10(M_BH / M_☉)"))
                .changed()
            {
                self.cfg.central_bh.mass_solar = 10.0_f64.powf(log_mbh);
                rebuild = true;
            }
            rebuild |= ui
                .add(
                    egui::Slider::new(&mut self.cfg.central_bh.effective_rs_kpc, 0.03..=0.80)
                        .text("Horizon Scale R_s (kpc)"),
                )
                .changed();
            if ui
                .add(
                    egui::Slider::new(
                        &mut self.cfg.central_bh.relativistic_beta_scale,
                        0.0005..=0.02,
                    )
                    .text("Doppler Beaming Scale"),
                )
                .changed()
            {
                self.revision = self.revision.wrapping_add(1).max(1);
            }
            if ui.small_button("Rosetta & ISCO Plunge Preset").clicked() {
                self.cfg.central_bh.mass_solar = 2.0e9;
                self.cfg.central_bh.effective_rs_kpc = 0.22;
                self.cfg.tidal_tangential_ratio = 0.22;
                rebuild = true;
            }
            self.render_bh_telemetry(ui);
        }
        self.render_caustic_overlay_controls(ui);
        rebuild
    }

    fn render_caustic_overlay_controls(&mut self, ui: &mut egui::Ui) {
        ui.separator();
        ui.heading("Halo Lensing Caustic Overlay");
        let prev_lens = self.cfg.lens_overlay;
        ui.checkbox(
            &mut self.cfg.lens_overlay.enabled,
            "Overlay NFW Cusp vs. Burkert Core Caustics",
        );
        if self.cfg.lens_overlay.enabled {
            ui.checkbox(
                &mut self.cfg.lens_overlay.drag_on_canvas,
                "Drag Caustic Lens on Canvas",
            );
            ui.add(
                egui::Slider::new(&mut self.cfg.lens_overlay.log10_sigma_crit, 6.5..=9.5)
                    .text("log10(Σ_crit)"),
            );
            ui.add(
                egui::Slider::new(&mut self.cfg.lens_overlay.lens_center_kpc[0], -10.0..=10.0)
                    .text("Lens X (kpc)"),
            );
            ui.add(
                egui::Slider::new(&mut self.cfg.lens_overlay.lens_center_kpc[1], -10.0..=10.0)
                    .text("Lens Y (kpc)"),
            );
        }
        if self.cfg.lens_overlay != prev_lens {
            self.revision = self.revision.wrapping_add(1).max(1);
        }
    }

    fn render_bh_telemetry(&self, ui: &mut egui::Ui) {
        let isco = self.cfg.central_bh.isco_radius_kpc();
        let captured = self.system.captured_particle_count();
        render_copyable_metric(ui, "ISCO Radius (3 R_s)", &format!("{isco:.3} kpc"));
        render_copyable_metric(
            ui,
            "Captured at ISCO",
            &format!("{captured} / {}", self.system.particles().len()),
        );
        if let Ok(bh) = self.cfg.central_bh.to_schwarzschild_bh() {
            if let Ok((gamma_r00, _pw)) = validate_christoffel_radial_connection(&bh, 6.0) {
                render_copyable_metric(
                    ui,
                    "Christoffel Γ^r_00 (6 R_s)",
                    &format!("{gamma_r00:.3e} m⁻¹"),
                );
            }
        }
    }

    pub(super) fn draw_relativistic_orbit_overlays(
        &self,
        plot_ui: &mut egui_plot::PlotUi<'_>,
        palette: &crate::app::SemanticPalette,
    ) {
        if self.cfg.central_bh.enabled {
            let r_isco = self.cfg.central_bh.isco_radius_kpc();
            let r_rs = self.cfg.central_bh.horizon_radius_kpc();
            plot_ui.line(
                Line::new(
                    "ISCO Capture Horizon (3 R_s)",
                    PlotPoints::new(build_circle_ring([0.0, 0.0], r_isco).to_vec()),
                )
                .color(palette.error)
                .width(2.0_f32),
            );
            plot_ui.line(
                Line::new(
                    "Event Horizon (R_s)",
                    PlotPoints::new(build_circle_ring([0.0, 0.0], r_rs).to_vec()),
                )
                .color(palette.warning)
                .style(LineStyle::Dotted { spacing: 4.0 }),
            );
            self.draw_doppler_boosted_tracers(plot_ui, palette);
        }
        if self.cfg.lens_overlay.enabled {
            let c = self.cfg.lens_overlay.lens_center_kpc;
            let (rn, rb) = self.compute_caustic_radii_kpc();
            if let Some(r_nfw) = rn {
                plot_ui.line(
                    Line::new(
                        "NFW Cusp Critical/Caustic",
                        PlotPoints::new(build_circle_ring(c, r_nfw).to_vec()),
                    )
                    .color(palette.accent)
                    .width(1.8_f32),
                );
            }
            if let Some(r_burk) = rb {
                plot_ui.line(
                    Line::new(
                        "Burkert Core Critical/Caustic",
                        PlotPoints::new(build_circle_ring(c, r_burk).to_vec()),
                    )
                    .color(palette.success)
                    .style(LineStyle::Dashed { length: 6.0 })
                    .width(1.8_f32),
                );
            }
            plot_ui.points(
                Points::new("Caustic Lens Center", vec![c])
                    .color(palette.warning)
                    .shape(MarkerShape::Cross)
                    .radius(6.0_f32),
            );
        }
    }

    fn draw_doppler_boosted_tracers(
        &self,
        plot_ui: &mut egui_plot::PlotUi<'_>,
        palette: &crate::app::SemanticPalette,
    ) {
        let Ok(bh) = self.cfg.central_bh.to_schwarzschild_bh() else {
            return;
        };
        let rs_kpc = self.cfg.central_bh.horizon_radius_kpc();
        let beta_scale = self.cfg.central_bh.relativistic_beta_scale;
        let u_b = self.cfg.central_bh.magnetic_energy_density_j_m3;
        let (mut app_pts, mut rec_pts, mut cap_pts) = (Vec::new(), Vec::new(), Vec::new());

        for p in self.system.particles() {
            if p.captured {
                cap_pts.push([p.x_kpc, p.y_kpc]);
                continue;
            }
            let r = (p.x_kpc * p.x_kpc + p.y_kpc * p.y_kpc)
                .sqrt()
                .max(1.001 * rs_kpc);
            let speed_kms = (p.vx_kms * p.vx_kms + p.vy_kms * p.vy_kms).sqrt();
            let los_cos = if speed_kms > 1.0e-6 {
                (p.vy_kms / speed_kms).clamp(-1.0, 1.0)
            } else {
                0.0
            };
            let v_eff_m_s = (speed_kms * beta_scale * C).clamp(0.0, 0.998 * C);
            let boost = evaluate_relativistic_beaming(&bh, r / rs_kpc, v_eff_m_s, los_cos, u_b)
                .map_or(1.0, |b| b.radiance_weight);
            if boost >= 1.0 {
                app_pts.push([p.x_kpc, p.y_kpc]);
            } else {
                rec_pts.push([p.x_kpc, p.y_kpc]);
            }
        }
        Self::plot_beamed_tracer_groups(plot_ui, palette, app_pts, rec_pts, cap_pts);
    }

    fn plot_beamed_tracer_groups(
        plot_ui: &mut egui_plot::PlotUi<'_>,
        palette: &crate::app::SemanticPalette,
        app_pts: Vec<[f64; 2]>,
        rec_pts: Vec<[f64; 2]>,
        cap_pts: Vec<[f64; 2]>,
    ) {
        if !app_pts.is_empty() {
            plot_ui.points(
                Points::new("Doppler-Boosted (Approaching)", PlotPoints::new(app_pts))
                    .color(palette.warning)
                    .radius(5.2_f32)
                    .shape(MarkerShape::Circle),
            );
        }
        if !rec_pts.is_empty() {
            plot_ui.points(
                Points::new("Doppler-Dimmed (Receding)", PlotPoints::new(rec_pts))
                    .color(palette.info)
                    .radius(3.8_f32)
                    .shape(MarkerShape::Diamond),
            );
        }
        if !cap_pts.is_empty() {
            plot_ui.points(
                Points::new("Captured at ISCO (3 R_s)", PlotPoints::new(cap_pts))
                    .color(palette.error)
                    .radius(5.5_f32)
                    .shape(MarkerShape::Cross),
            );
        }
    }

    pub(super) fn append_relativistic_orbit_datasets(
        &self,
        out: &mut Vec<(String, Vec<[f64; 2]>)>,
    ) {
        if self.cfg.central_bh.enabled {
            let r_isco = self.cfg.central_bh.isco_radius_kpc();
            out.push((
                "ISCO Capture Horizon (3 R_s)".into(),
                build_circle_ring([0.0, 0.0], r_isco).to_vec(),
            ));
            let captured: Vec<[f64; 2]> = self
                .system
                .particles()
                .iter()
                .filter(|p| p.captured || p.kind == OrbitParticleKind::TidalStream)
                .map(|p| [p.x_kpc, p.y_kpc])
                .collect();
            if !captured.is_empty() {
                out.push(("Relativistic Stream & ISCO Tracers".into(), captured));
            }
        }
        if self.cfg.lens_overlay.enabled {
            let c = self.cfg.lens_overlay.lens_center_kpc;
            let (rn, rb) = self.compute_caustic_radii_kpc();
            if let Some(r_nfw) = rn {
                out.push((
                    "NFW Cusp Critical Curve".into(),
                    build_circle_ring(c, r_nfw).to_vec(),
                ));
            }
            if let Some(r_burk) = rb {
                out.push((
                    "Burkert Core Critical Curve".into(),
                    build_circle_ring(c, r_burk).to_vec(),
                ));
            }
        }
    }
}
