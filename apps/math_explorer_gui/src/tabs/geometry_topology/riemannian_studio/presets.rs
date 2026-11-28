//! Curated `< 1 KiB` Starter Gallery presets for [`super::RiemannianPdeStudio`].

use super::handoff::RiemannianPresetTarget;

/// Starter preset: Mean Curvature Flow geometric smoothing on a folded cortical shell.
pub const PRESET_MEAN_CURVATURE_FLOW_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Riemannian Mean Curvature Flow",
    "domain": "geometry_topology",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "manifold": "FoldedCorticalShell",
    "regime": "MeanCurvatureSmoothing",
    "u_res": 26,
    "v_res": 18,
    "dt": 0.004,
    "sulcal_depth": 0.26,
    "camera": {
      "pitch": 0.45,
      "yaw": 0.55,
      "zoom": 1.05
    }
  }
}"#;

/// Starter preset: Gray-Scott mitosis Turing morphogenesis on a welded genus-0 sphere.
pub const PRESET_SPHERICAL_TURING_MITOSIS_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Spherical Turing Mitosis",
    "domain": "geometry_topology",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "manifold": "Sphere",
    "regime": "GrayScottTuring",
    "u_res": 28,
    "v_res": 20,
    "dt": 0.015,
    "gs_f": 0.0367,
    "gs_k": 0.0649,
    "d_u": 0.006,
    "d_v": 0.003,
    "camera": {
      "pitch": 0.40,
      "yaw": 0.65,
      "zoom": 1.10
    }
  }
}"#;

/// Starter preset: Hodgkin-Huxley cortical sheet wave modulated by FreeSurfer cortical thickness.
pub const PRESET_CORTICAL_SPREADING_DEPRESSION_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Cortical Hodgkin-Huxley Wave",
    "domain": "geometry_topology",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "manifold": "FoldedCorticalShell",
    "regime": "CorticalHodgkinHuxley",
    "u_res": 26,
    "v_res": 18,
    "dt": 0.006,
    "hh_conductance": 0.035,
    "hh_stimulus": 35.0,
    "sulcal_depth": 0.24,
    "camera": {
      "pitch": 0.50,
      "yaw": -0.45,
      "zoom": 1.05
    }
  }
}"#;

/// Starter preset: FreeSurfer cortical thickness and 12-subject group GLM `t_statistic` contrast.
pub const PRESET_FREESURFER_GLM_CONTRAST_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "FreeSurfer Cortical GLM Contrast",
    "domain": "geometry_topology",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "manifold": "FoldedCorticalShell",
    "regime": "FreeSurferMorphometryGlm",
    "overlay": "GlmTStatistic",
    "u_res": 26,
    "v_res": 18,
    "dt": 0.005,
    "sulcal_depth": 0.25,
    "fs_grad_strength": 0.40,
    "camera": {
      "pitch": 0.42,
      "yaw": 0.35,
      "zoom": 1.05
    }
  }
}"#;

/// Maps a [`RiemannianPresetTarget`] deep-link target to its `< 1 KiB` JSON preset envelope.
#[must_use]
pub const fn preset_json_for_target(target: RiemannianPresetTarget) -> &'static str {
    match target {
        RiemannianPresetTarget::SphericalTuringMitosis => PRESET_SPHERICAL_TURING_MITOSIS_JSON,
        RiemannianPresetTarget::CorticalSpreadingDepression => {
            PRESET_CORTICAL_SPREADING_DEPRESSION_JSON
        }
        RiemannianPresetTarget::FreeSurferGlmContrast => PRESET_FREESURFER_GLM_CONTRAST_JSON,
        RiemannianPresetTarget::MeanCurvatureFlow => PRESET_MEAN_CURVATURE_FLOW_JSON,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "riemannian_mean_curvature_flow",
        title: "Riemannian Mean Curvature Flow",
        description: "∂r/∂t = H·n: Intrinsic cotangent Laplace-Beltrami smoothing of a gyral-sulcal folded manifold.",
        category: "Differential Geometry",
        icon: "🫧",
        domain: "geometry_topology",
        tool_name: Some("RiemannianPdeStudio"),
        preset_generator: || PRESET_MEAN_CURVATURE_FLOW_JSON,
    }
}

#[cfg(feature = "biology")]
inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "riemannian_spherical_turing_mitosis",
        title: "Spherical Turing Mitosis",
        description: "∂u/∂t = D_u·Δ_S u + f(u,v): Pole-free Gray-Scott spot division on a welded cotangent sphere.",
        category: "Differential Geometry",
        icon: "🦠",
        domain: "geometry_topology",
        tool_name: Some("RiemannianPdeStudio"),
        preset_generator: || PRESET_SPHERICAL_TURING_MITOSIS_JSON,
    }
}

#[cfg(feature = "biology")]
inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "riemannian_cortical_spreading_depression",
        title: "Cortical Hodgkin-Huxley Wave",
        description: "C_m·∂V/∂t = g_s·Δ_S(h·V) - I_ion: Excitable cortical action-potential waves weighted by cortical thickness.",
        category: "Computational Neuroscience",
        icon: "🧠",
        domain: "geometry_topology",
        tool_name: Some("RiemannianPdeStudio"),
        preset_generator: || PRESET_CORTICAL_SPREADING_DEPRESSION_JSON,
    }
}

#[cfg(feature = "applied")]
inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "riemannian_freesurfer_glm_contrast",
        title: "FreeSurfer Cortical GLM Contrast",
        description: "t = cᵀβ̂ / SE(cᵀβ̂): Cortical white-pial thickness, deformable pial energy, and group GLM t-statistics.",
        category: "Biomedical Morphometry",
        icon: "🔬",
        domain: "geometry_topology",
        tool_name: Some("RiemannianPdeStudio"),
        preset_generator: || PRESET_FREESURFER_GLM_CONTRAST_JSON,
    }
}

// [cite:graph_parameters_rust]
