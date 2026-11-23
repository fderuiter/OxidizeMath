// @explorer_feature = "physics"
use crate::tabs::ExplorerTab;
use eframe::egui;
use math_explorer::physics::astrophysics::HaloProfile;

pub mod galactic_orbits_tool;
mod galactic_orbits_ui;
pub mod rotation_curve_tool;
mod rotation_curve_ui;

#[cfg(test)]
mod tests;

pub(crate) fn clamp_finite(val: f64, min: f64, max: f64, fallback: f64) -> f64 {
    if val.is_finite() {
        val.clamp(min, max)
    } else {
        fallback
    }
}

/// Selectable dark-matter halo family in [`rotation_curve_tool::RotationCurveExplorer`] and [`galactic_orbits_tool::GalacticOrbitSimulator`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum HaloKindChoice {
    /// Pure baryonic disk + gas without dark matter (Keplerian outer falloff).
    BaryonicOnly,
    /// Navarro-Frenk-White (1997) $\rho \propto r^{-1}$ cuspy halo.
    NfwCusp,
    /// Burkert (1995) constant-density cored dwarf halo.
    #[default]
    BurkertCore,
}

impl HaloKindChoice {
    /// Human-readable label for combo boxes and legends.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::BaryonicOnly => "Baryonic Only (Keplerian)",
            Self::NfwCusp => "NFW Cusp (CDM)",
            Self::BurkertCore => "Burkert Core (Dwarf)",
        }
    }

    /// Builds a [`HaloProfile`] from log10 virial mass, scale/core radius, and concentration.
    #[must_use]
    pub fn build_profile(
        self,
        log10_m200: f64,
        radius_kpc: f64,
        concentration: f64,
    ) -> HaloProfile {
        let m200 = 10.0_f64.powf(log10_m200.clamp(6.0, 14.0));
        let r_scale = radius_kpc.clamp(0.1, 25.0);
        let c = concentration.clamp(1.0, 40.0);
        match self {
            Self::BaryonicOnly => HaloProfile::BaryonicOnly,
            Self::NfwCusp => HaloProfile::NfwCusp {
                virial_mass_solar: m200,
                scale_radius_kpc: r_scale,
                concentration: c,
            },
            Self::BurkertCore => HaloProfile::BurkertCore {
                virial_mass_solar: m200,
                core_radius_kpc: r_scale,
                concentration: c,
            },
        }
    }
}

/// Interactive Astrophysics & Dwarf-Galaxy Dynamics tab backed by [`crate::framework::SimulationFramework`].
pub struct AstrophysicsTab {
    framework: crate::framework::SimulationFramework,
}

impl Default for AstrophysicsTab {
    fn default() -> Self {
        Self {
            framework: crate::framework::SimulationFramework::new("astrophysics"),
        }
    }
}

impl ExplorerTab for AstrophysicsTab {
    fn name(&self) -> &'static str {
        "Astrophysics"
    }

    fn academic_domain(&self) -> &'static str {
        "Physics"
    }

    fn url_slug(&self) -> &'static str {
        "astrophysics"
    }

    fn show_inside(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.framework.show_inside(ui, "astrophysics");
    }

    fn state_revision(&self) -> u64 {
        self.framework.state_revision()
    }

    fn save_state(&self) -> Option<String> {
        self.framework.save_state()
    }

    fn load_state(&mut self, state: &str) {
        self.framework.load_state(state);
    }

    fn focused_canvas_id(&self) -> Option<egui::Id> {
        self.framework.focused_canvas_id()
    }

    fn select_tool_by_name(&mut self, name: &str) -> bool {
        self.framework.select_tool_by_name(name)
    }

    fn framework(&self) -> Option<&crate::framework::SimulationFramework> {
        Some(&self.framework)
    }

    fn framework_mut(&mut self) -> Option<&mut crate::framework::SimulationFramework> {
        Some(&mut self.framework)
    }
}

// [cite:dwarf_galaxy_empirical_dependencies]
