#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Cross-tab deep-link handoff triggers opening [`super::RiemannianPdeStudio`] into curated
//! 3D manifold presets from `MorphogenesisTool`, `HodgkinHuxleyTool`, or tests.

use eframe::egui;

const OPEN_RIEMANNIAN_STUDIO_ID: &str = "OPEN_RIEMANNIAN_STUDIO_PRESET";

/// Target preset launched when navigating across tabs into [`super::RiemannianPdeStudio`].
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RiemannianPresetTarget {
    /// Spherical Gray-Scott mitosis spots over CSR `CotangentMeshDiffusion`.
    #[default]
    SphericalTuringMitosis,
    /// Gyral-sulcal `CorticalSheetRunner` Hodgkin-Huxley wave propagation.
    CorticalSpreadingDepression,
    /// Folded cortical FreeSurfer thickness & 12-subject group GLM `t_statistic` contrast.
    FreeSurferGlmContrast,
    /// Intrinsic `MeanCurvatureFlow` smoothing of a folded cortical shell.
    MeanCurvatureFlow,
}

impl RiemannianPresetTarget {
    /// All curated 3D manifold presets selectable from the studio or cross-tab deep links.
    pub const ALL: [Self; 4] = [
        Self::SphericalTuringMitosis,
        Self::CorticalSpreadingDepression,
        Self::FreeSurferGlmContrast,
        Self::MeanCurvatureFlow,
    ];

    /// Human-readable label for preset launcher menus.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::SphericalTuringMitosis => "🌐 Spherical Gray-Scott Mitosis",
            Self::CorticalSpreadingDepression => "🧠 Cortical Hodgkin-Huxley Wave",
            Self::FreeSurferGlmContrast => "🧬 FreeSurfer Cortical GLM Contrast",
            Self::MeanCurvatureFlow => "✨ Cortical Mean Curvature Flow",
        }
    }

    /// Serialized `< 4 KiB` JSON preset payload for this target.
    #[must_use]
    pub const fn preset_json(self) -> &'static str {
        match self {
            Self::SphericalTuringMitosis => super::presets::PRESET_SPHERICAL_TURING_MITOSIS_JSON,
            Self::CorticalSpreadingDepression => {
                super::presets::PRESET_CORTICAL_SPREADING_DEPRESSION_JSON
            }
            Self::FreeSurferGlmContrast => super::presets::PRESET_FREESURFER_GLM_CONTRAST_JSON,
            Self::MeanCurvatureFlow => super::presets::PRESET_MEAN_CURVATURE_FLOW_JSON,
        }
    }
}

/// Queues a cross-tab navigation request to open `Geometry & Topology -> Riemannian PDE Studio`
/// initialized with `target`.
#[cfg(any(feature = "biology", test))]
pub fn request_open_riemannian_preset(ctx: &egui::Context, target: RiemannianPresetTarget) {
    ctx.data_mut(|d| {
        d.insert_temp(egui::Id::new(OPEN_RIEMANNIAN_STUDIO_ID), target);
    });
    ctx.request_repaint();
}

/// Drains any pending cross-tab navigation request for [`super::RiemannianPdeStudio`].
#[must_use]
pub fn take_open_riemannian_preset(ctx: &egui::Context) -> Option<RiemannianPresetTarget> {
    ctx.data_mut(|d| {
        d.remove_temp::<RiemannianPresetTarget>(egui::Id::new(OPEN_RIEMANNIAN_STUDIO_ID))
    })
}
