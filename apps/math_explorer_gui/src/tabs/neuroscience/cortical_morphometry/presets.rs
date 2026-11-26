#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Curated starter gallery presets for [`super::CorticalMorphometryTool`].

pub(super) const PRESET_SULCAL_THICKNESS_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Folded Sulcal Cortical Thickness",
    "domain": "neuroscience",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "overlay_mode": "Thickness",
    "surface_mode": "DualShell",
    "inflation": 0.15,
    "u_resolution": 30,
    "v_resolution": 22,
    "sulcal_depth_mm": 2.9,
    "base_thickness_mm": 2.6,
    "alpha": 0.4,
    "beta": 0.3,
    "sigma": 1.2,
    "learning_rate": 0.025,
    "evolution_steps": 5,
    "potts_beta": 1.3,
    "mri_noise_level": 7.5,
    "atrophy_effect_mm": 0.55,
    "collinear_glm_design": false,
    "camera": {
      "pitch": 0.48,
      "yaw": -0.58,
      "zoom": 1.05
    }
  }
}"#;

pub(super) const PRESET_GLM_ATROPHY_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Inflated GLM Atrophy Contrast",
    "domain": "neuroscience",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "overlay_mode": "GlmTStatistic",
    "surface_mode": "Pial",
    "inflation": 0.85,
    "u_resolution": 32,
    "v_resolution": 22,
    "sulcal_depth_mm": 2.6,
    "base_thickness_mm": 2.5,
    "alpha": 0.35,
    "beta": 0.25,
    "sigma": 1.2,
    "learning_rate": 0.02,
    "evolution_steps": 4,
    "potts_beta": 1.2,
    "mri_noise_level": 6.0,
    "atrophy_effect_mm": 0.95,
    "collinear_glm_design": false,
    "camera": {
      "pitch": 0.38,
      "yaw": -0.82,
      "zoom": 1.0
    }
  }
}"#;

pub(super) const PRESET_MRF_DEFORMABLE_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Noisy MRI Potts MRF Reconstruction",
    "domain": "neuroscience",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "overlay_mode": "MrfTissueClass",
    "surface_mode": "White",
    "inflation": 0.35,
    "u_resolution": 28,
    "v_resolution": 20,
    "sulcal_depth_mm": 2.3,
    "base_thickness_mm": 2.4,
    "alpha": 0.45,
    "beta": 0.35,
    "sigma": 1.5,
    "learning_rate": 0.03,
    "evolution_steps": 8,
    "potts_beta": 1.8,
    "mri_noise_level": 18.0,
    "atrophy_effect_mm": 0.5,
    "collinear_glm_design": false,
    "camera": {
      "pitch": 0.52,
      "yaw": -0.45,
      "zoom": 1.0
    }
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "cortical_sulcal_thickness",
        title: "Folded Sulcal Cortical Thickness",
        description: "Dual-shell pial and white-matter manifolds displaying symmetric cortical ribbon thickness across gyri and sulci.",
        category: "Computational Neuroanatomy",
        icon: "🧠",
        domain: "neuroscience",
        tool_name: Some("CorticalMorphometryTool"),
        preset_generator: || PRESET_SULCAL_THICKNESS_JSON,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "cortical_glm_atrophy_contrast",
        title: "Inflated GLM Atrophy Contrast",
        description: "Unfolded 85%-inflated cortical hemisphere revealing focal cohort cortical thinning t-statistics buried inside sulcal folds.",
        category: "Computational Neuroanatomy",
        icon: "📊",
        domain: "neuroscience",
        tool_name: Some("CorticalMorphometryTool"),
        preset_generator: || PRESET_GLM_ATROPHY_JSON,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "cortical_mrf_deformable_recon",
        title: "Noisy MRI Potts MRF Reconstruction",
        description: "6-connected Potts Markov Random Field denoising and deformable Laplacian surface evolution under heavy T1 noise.",
        category: "Computational Neuroanatomy",
        icon: "🧲",
        domain: "neuroscience",
        tool_name: Some("CorticalMorphometryTool"),
        preset_generator: || PRESET_MRF_DEFORMABLE_JSON,
    }
}
