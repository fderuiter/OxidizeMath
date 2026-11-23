// @explorer_feature = "physics"
// @explorer_domain = "Physics"
// @explorer_slug = "astrophysics"

#[cfg(test)]
use crate::tabs::ExplorerTab;
#[cfg(test)]
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

crate::declare_simulation_tab!(
    /// Interactive Astrophysics & Dwarf-Galaxy Dynamics tab.
    pub struct AstrophysicsTab,
    "Astrophysics",
);

// [cite:dwarf_galaxy_empirical_dependencies]
