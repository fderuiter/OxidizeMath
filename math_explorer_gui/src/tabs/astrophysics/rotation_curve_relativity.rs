#![cfg_attr(any(), verified(opt_out = "gui_tool"))]
//! Simultaneous NFW vs. Burkert lensing shear, critical caustic radii, and Schwarzschild
//! gravitational time dilation curves for [`super::rotation_curve_tool::RotationCurveExplorer`].
// [cite:dwarf_galaxy_empirical_dependencies]
// [cite:general_relativity_geodesics]

use eframe::egui;
use egui_plot::{Line, LineStyle, Plot, PlotPoints};
use math_explorer::constants::SOLAR_MASS;
use math_explorer::physics::astrophysics::{
    sample_halo_lensing_into, tangential_critical_radius_kpc, HaloProfile, ThinLensSample,
};
use math_explorer::physics::high_energy::general_relativity::SchwarzschildBlackHole;

use crate::reflective_ui::render_copyable_metric;

use super::clamp_finite;
use super::rotation_curve_tool::RotationCurveExplorer;

/// Optional lensing & gravitational time-dilation overlay state for `RotationCurveExplorer`.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct RotationRelativityConfig {
    /// Shows the simultaneous `NfwCusp` vs. `BurkertCore` lensing shear & Schwarzschild clock plot.
    pub(crate) enabled: bool,
    /// Effective central Schwarzschild radius $R_s$ (kpc) for radial time dilation $d\tau/dt(r)$.
    pub(crate) central_bh_rs_kpc: f64,
    /// Logarithmic critical surface density $\log_{10}(\Sigma_{\text{crit}} / (M_\odot\,\text{kpc}^{-2}))$.
    pub(crate) log10_sigma_crit: f64,
}

impl Default for RotationRelativityConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            central_bh_rs_kpc: 0.15,
            log10_sigma_crit: 7.45,
        }
    }
}

impl RotationRelativityConfig {
    pub(crate) fn is_default(&self) -> bool {
        *self == Self::default()
    }

    pub(crate) fn is_valid(&self) -> bool {
        self.central_bh_rs_kpc.is_finite()
            && (0.01..=2.0).contains(&self.central_bh_rs_kpc)
            && self.log10_sigma_crit.is_finite()
            && (6.0..=10.5).contains(&self.log10_sigma_crit)
    }

    pub(crate) fn clamp_in_place(&mut self) {
        self.central_bh_rs_kpc = clamp_finite(self.central_bh_rs_kpc, 0.01, 2.0, 0.15);
        self.log10_sigma_crit = clamp_finite(self.log10_sigma_crit, 6.0, 10.5, 7.45);
    }
}

impl RotationCurveExplorer {
    pub(super) fn recalculate_relativity_profiles(&mut self, r_max: f64) {
        let baryons = self.current_baryons();
        let m200 = 10.0_f64.powf(self.cfg.halo_log10_mass.clamp(8.0, 12.0));
        let scale = self.cfg.halo_scale_kpc.clamp(0.3, 10.0);
        let conc = self.cfg.halo_concentration.clamp(4.0, 25.0);
        let sigma_crit = 10.0_f64.powf(self.cfg.relativity.log10_sigma_crit.clamp(6.0, 10.5));

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
        sample_halo_lensing_into(
            0.05,
            r_max,
            Some(&baryons),
            &nfw,
            sigma_crit,
            &mut self.nfw_lens_samples,
        );
        sample_halo_lensing_into(
            0.05,
            r_max,
            Some(&baryons),
            &burk,
            sigma_crit,
            &mut self.burkert_lens_samples,
        );
        self.nfw_crit_kpc = tangential_critical_radius_kpc(Some(&baryons), &nfw, sigma_crit, r_max);
        self.burkert_crit_kpc =
            tangential_critical_radius_kpc(Some(&baryons), &burk, sigma_crit, r_max);

        let bh = SchwarzschildBlackHole::new(1.0e8 * SOLAR_MASS).expect("valid bh");
        let bh_rs_phys = bh.schwarzschild_radius().unwrap_or(1.0);
        let rs_kpc = self.cfg.relativity.central_bh_rs_kpc.clamp(0.01, 2.0);
        let n = self.dilation_samples.len();
        for (idx, slot) in self.dilation_samples.iter_mut().enumerate() {
            let r_kpc = self
                .nfw_lens_samples
                .get(idx)
                .map_or(0.05 + (idx as f64 / n.max(1) as f64) * r_max, |s| s.radius);
            let r_over_rs = (r_kpc / rs_kpc).max(1.001);
            let g_tau = bh
                .gravitational_time_dilation(r_over_rs * bh_rs_phys)
                .unwrap_or(1.0);
            *slot = [r_kpc, g_tau];
        }
    }

    pub(super) fn render_relativity_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut ch = false;
        ui.separator();
        ui.heading("Lensing Caustics & GR Dilation");
        ch |= ui
            .checkbox(
                &mut self.cfg.relativity.enabled,
                "Compare NFW vs. Burkert Shear & Time Dilation",
            )
            .changed();
        if self.cfg.relativity.enabled {
            ch |= ui
                .add(
                    egui::Slider::new(&mut self.cfg.relativity.log10_sigma_crit, 6.5..=9.5)
                        .text("log10(Σ_crit / M_☉ kpc⁻²)"),
                )
                .changed();
            ch |= ui
                .add(
                    egui::Slider::new(&mut self.cfg.relativity.central_bh_rs_kpc, 0.02..=1.20)
                        .text("Central BH R_s (kpc)"),
                )
                .changed();
            let nfw_str = self
                .nfw_crit_kpc
                .map_or_else(|| "Subcritical".to_string(), |r| format!("{r:.3} kpc"));
            let burk_str = self
                .burkert_crit_kpc
                .map_or_else(|| "Subcritical".to_string(), |r| format!("{r:.3} kpc"));
            render_copyable_metric(ui, "NFW Cusp Critical R_E", &nfw_str);
            render_copyable_metric(ui, "Burkert Core Critical R_E", &burk_str);
        }
        ch
    }

    pub(super) fn render_relativity_plot(&self, ui: &mut egui::Ui) {
        if !self.cfg.relativity.enabled {
            return;
        }
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        let [nfw_shear, burk_shear, nfw_kappa, burk_kappa] = self.build_lensing_plot_series();
        ui.separator();
        ui.label("NFW Cusp vs. Burkert Core Tangential Shear γ(r), Mean Convergence κ̄(r), and Schwarzschild Time Dilation dτ/dt(r):");
        Plot::new("astrophysics_lensing_and_dilation_plot")
            .view_aspect(2.35)
            .x_axis_label("Galactocentric Radius r (kpc)")
            .y_axis_label("Dimensionless Optical / Clock Profile")
            .include_y(0.0)
            .include_y(1.05)
            .legend(egui_plot::Legend::default())
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new("NFW Cusp Shear γ(r)", PlotPoints::new(nfw_shear))
                        .color(palette.accent)
                        .width(2.0_f32),
                );
                plot_ui.line(
                    Line::new("Burkert Core Shear γ(r)", PlotPoints::new(burk_shear))
                        .color(palette.success)
                        .style(LineStyle::Dashed { length: 7.0 })
                        .width(2.0_f32),
                );
                plot_ui.line(
                    Line::new("NFW Mean Convergence κ̄(r)", PlotPoints::new(nfw_kappa))
                        .color(palette.axis_x)
                        .style(LineStyle::Dotted { spacing: 5.0 }),
                );
                plot_ui.line(
                    Line::new("Burkert Mean Convergence κ̄(r)", PlotPoints::new(burk_kappa))
                        .color(palette.info)
                        .style(LineStyle::Dotted { spacing: 5.0 }),
                );
                plot_ui.line(
                    Line::new(
                        "Schwarzschild Time Dilation dτ/dt(r)",
                        PlotPoints::new(self.dilation_samples.clone()),
                    )
                    .color(palette.warning)
                    .width(2.2_f32),
                );
            });
    }

    fn build_lensing_plot_series(&self) -> [Vec<[f64; 2]>; 4] {
        [
            self.nfw_lens_samples
                .iter()
                .map(|s: &ThinLensSample| [s.radius, s.shear])
                .collect(),
            self.burkert_lens_samples
                .iter()
                .map(|s: &ThinLensSample| [s.radius, s.shear])
                .collect(),
            self.nfw_lens_samples
                .iter()
                .map(|s: &ThinLensSample| [s.radius, s.mean_convergence])
                .collect(),
            self.burkert_lens_samples
                .iter()
                .map(|s: &ThinLensSample| [s.radius, s.mean_convergence])
                .collect(),
        ]
    }

    pub(super) fn append_relativity_export_datasets(&self, out: &mut Vec<(String, Vec<[f64; 2]>)>) {
        if !self.cfg.relativity.enabled {
            return;
        }
        out.push((
            "NFW Cusp Shear gamma".into(),
            self.nfw_lens_samples
                .iter()
                .map(|s| [s.radius, s.shear])
                .collect(),
        ));
        out.push((
            "Burkert Core Shear gamma".into(),
            self.burkert_lens_samples
                .iter()
                .map(|s| [s.radius, s.shear])
                .collect(),
        ));
        out.push((
            "Schwarzschild Time Dilation dtau_dt".into(),
            self.dilation_samples.clone(),
        ));
    }
}
