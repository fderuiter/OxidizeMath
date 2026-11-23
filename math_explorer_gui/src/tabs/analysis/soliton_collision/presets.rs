#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

// [cite:advanced_linear_algebra]
// [cite:graph_parameters_rust]

use math_explorer::pure_math::analysis::pde::kdv::KdvEquationMode;

use super::{
    SolitonCollisionState, SolitonCollisionTool, SolitonColormap, SolitonPreset, COSINE_AMP_MAX,
    COSINE_AMP_MIN, DOMAIN_LEN_MAX, DOMAIN_LEN_MIN, DT_MAX, DT_MIN, NLS_COUPLING_MAX,
    NLS_COUPLING_MIN, POS_X_MAX, POS_X_MIN, SPEED_C_MAX, SPEED_C_MIN,
};

fn sanitize_bounded(val: f64, min: f64, max: f64, fallback: f64) -> f64 {
    if val.is_finite() {
        val.clamp(min, max)
    } else {
        fallback
    }
}

pub(super) fn is_valid_state(s: &SolitonCollisionState) -> bool {
    let half_l = 0.5 * s.domain_length;
    s.domain_length.is_finite()
        && (DOMAIN_LEN_MIN..=DOMAIN_LEN_MAX).contains(&s.domain_length)
        && s.fast_speed_c1.is_finite()
        && (SPEED_C_MIN..=SPEED_C_MAX).contains(&s.fast_speed_c1)
        && s.slow_speed_c2.is_finite()
        && (SPEED_C_MIN..=SPEED_C_MAX).contains(&s.slow_speed_c2)
        && s.fast_x0.is_finite()
        && (POS_X_MIN..=POS_X_MAX).contains(&s.fast_x0)
        && s.fast_x0.abs() <= half_l
        && s.slow_x0.is_finite()
        && (POS_X_MIN..=POS_X_MAX).contains(&s.slow_x0)
        && s.slow_x0.abs() <= half_l
        && s.pulse_x0.is_finite()
        && (POS_X_MIN..=POS_X_MAX).contains(&s.pulse_x0)
        && s.pulse_x0.abs() <= half_l
        && s.pulse_speed_c.is_finite()
        && (SPEED_C_MIN..=SPEED_C_MAX).contains(&s.pulse_speed_c)
        && s.cosine_amplitude.is_finite()
        && (COSINE_AMP_MIN..=COSINE_AMP_MAX).contains(&s.cosine_amplitude)
        && (1..=8).contains(&s.cosine_harmonic)
        && s.nls_coupling.is_finite()
        && (NLS_COUPLING_MIN..=NLS_COUPLING_MAX).contains(&s.nls_coupling)
        && s.time_step.is_finite()
        && (DT_MIN..=DT_MAX).contains(&s.time_step)
}

pub(super) fn clamp_state_fields(s: &mut SolitonCollisionState) {
    s.domain_length = sanitize_bounded(s.domain_length, DOMAIN_LEN_MIN, DOMAIN_LEN_MAX, 40.0);
    let half_l = (0.5 * s.domain_length - 0.5).max(4.0);
    s.fast_speed_c1 = sanitize_bounded(s.fast_speed_c1, SPEED_C_MIN, SPEED_C_MAX, 16.0);
    s.slow_speed_c2 = sanitize_bounded(s.slow_speed_c2, SPEED_C_MIN, SPEED_C_MAX, 4.0);
    s.fast_x0 = sanitize_bounded(s.fast_x0, -half_l, half_l, -10.0);
    s.slow_x0 = sanitize_bounded(s.slow_x0, -half_l, half_l, -1.0);
    s.pulse_x0 = sanitize_bounded(s.pulse_x0, -half_l, half_l, 0.0);
    s.pulse_speed_c = sanitize_bounded(s.pulse_speed_c, SPEED_C_MIN, SPEED_C_MAX, 12.0);
    s.cosine_amplitude = sanitize_bounded(s.cosine_amplitude, COSINE_AMP_MIN, COSINE_AMP_MAX, 2.4);
    s.cosine_harmonic = s.cosine_harmonic.clamp(1, 8);
    s.nls_coupling = sanitize_bounded(s.nls_coupling, NLS_COUPLING_MIN, NLS_COUPLING_MAX, 1.0);
    s.time_step = sanitize_bounded(s.time_step, DT_MIN, DT_MAX, 0.01);
}

fn base_soliton_state(preset: SolitonPreset) -> SolitonCollisionState {
    SolitonCollisionState {
        preset,
        mode: KdvEquationMode::KdvSoliton,
        colormap: SolitonColormap::PerceptualThermal,
        nonlinear_enabled: true,
        domain_length: 40.0,
        fast_speed_c1: 16.0,
        slow_speed_c2: 4.0,
        fast_x0: -10.0,
        slow_x0: -1.0,
        pulse_x0: 0.0,
        pulse_speed_c: 12.0,
        cosine_amplitude: 2.5,
        cosine_harmonic: 1,
        nls_coupling: 1.0,
        time_step: 0.01,
        show_initial_overlay: true,
        show_ballistic_markers: true,
    }
}

pub(super) fn default_state_for_preset(preset: SolitonPreset) -> SolitonCollisionState {
    match preset {
        SolitonPreset::TwoSolitonOvertaking => base_soliton_state(preset),
        SolitonPreset::LinearAiryCollapse => SolitonCollisionState {
            nonlinear_enabled: false,
            colormap: SolitonColormap::BipolarWave,
            ..base_soliton_state(preset)
        },
        SolitonPreset::ZabuskyKruskalFission => SolitonCollisionState {
            domain_length: 32.0,
            cosine_amplitude: 2.8,
            cosine_harmonic: 1,
            time_step: 0.008,
            show_ballistic_markers: false,
            ..base_soliton_state(preset)
        },
        SolitonPreset::CubicNlsEnvelopeCollision => SolitonCollisionState {
            mode: KdvEquationMode::CubicNls1D,
            colormap: SolitonColormap::WcagHighContrast,
            domain_length: 32.0,
            fast_speed_c1: 9.0,
            slow_speed_c2: 4.0,
            fast_x0: -6.0,
            slow_x0: 6.0,
            nls_coupling: 1.0,
            time_step: 0.008,
            show_ballistic_markers: false,
            ..base_soliton_state(preset)
        },
    }
}

impl scientific_metadata::theory::TheoryDescribable for SolitonCollisionTool {
    fn theory_description(&self) -> String {
        "1D periodic pseudospectral Lawson Integrating-Factor RK4 solver for the Korteweg-de Vries equation u_t + 6 alpha u u_x + u_xxx = 0 and Strang split-step 1D cubic Nonlinear Schrodinger envelopes i psi_t + psi_xx + 2 sigma |psi|^2 psi = 0, contrasting elastic inverse-scattering soliton recovery and analytical phase shifts against linear Airy dispersive collapse.".into()
    }

    fn phonetic_description(&self) -> String {
        "One dimensional Korteweg de Vries soliton collision, Airy dispersion, and cubic nonlinear Schrodinger studio"
            .into()
    }

    fn theory_citation(&self) -> String {
        "Korteweg & de Vries, On the Change of Form of Long Waves Advancing in a Rectangular Canal, Phil. Mag. 39 (1895) 10.1080/14786449508620739; Zabusky & Kruskal, Interaction of Solitons in a Collisionless Plasma, Phys. Rev. Lett. 15 (1965) 10.1103/PhysRevLett.15.240".into()
    }

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "TwoSolitonOvertaking".into(),
            "Elastic overtaking collision of fast c1 = 16 and slow c2 = 4 KdV sech^2 solitons with exact phase shifts +0.5493 and -1.0986".into(),
        );
        map.insert(
            "LinearAiryCollapse".into(),
            "Pure third-order Airy dispersion u_t + u_xxx = 0 showing oscillating dispersive wave-train collapse when 6 u u_x steepening is disabled".into(),
        );
        map.insert(
            "ZabuskyKruskalFission".into(),
            "Smooth cosine initial wave steepening and breaking into an ordered train of KdV solitons".into(),
        );
        map.insert(
            "CubicNlsEnvelopeCollision".into(),
            "Focusing 1D cubic Nonlinear Schrodinger bright envelope solitons colliding elastically under Strang splitting".into(),
        );
        map
    }
}

const TWO_SOLITON_OVERTAKING_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Two-Soliton Overtaking Collision",
    "domain": "analysis",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "preset": "TwoSolitonOvertaking",
    "mode": "KdvSoliton",
    "colormap": "PerceptualThermal",
    "nonlinear_enabled": true,
    "domain_length": 40.0,
    "fast_speed_c1": 16.0,
    "slow_speed_c2": 4.0,
    "fast_x0": -10.0,
    "slow_x0": -1.0,
    "pulse_x0": 0.0,
    "pulse_speed_c": 12.0,
    "cosine_amplitude": 2.5,
    "cosine_harmonic": 1,
    "nls_coupling": 1.0,
    "time_step": 0.01,
    "show_initial_overlay": true,
    "show_ballistic_markers": true
  }
}"#;

const LINEAR_AIRY_COLLAPSE_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Linear Airy Dispersion Collapse",
    "domain": "analysis",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "preset": "LinearAiryCollapse",
    "mode": "KdvSoliton",
    "colormap": "BipolarWave",
    "nonlinear_enabled": false,
    "domain_length": 40.0,
    "fast_speed_c1": 16.0,
    "slow_speed_c2": 4.0,
    "fast_x0": -10.0,
    "slow_x0": -1.0,
    "pulse_x0": 0.0,
    "pulse_speed_c": 12.0,
    "cosine_amplitude": 2.5,
    "cosine_harmonic": 1,
    "nls_coupling": 1.0,
    "time_step": 0.01,
    "show_initial_overlay": true,
    "show_ballistic_markers": true
  }
}"#;

const ZABUSKY_KRUSKAL_FISSION_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Zabusky-Kruskal Soliton Fission",
    "domain": "analysis",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "preset": "ZabuskyKruskalFission",
    "mode": "KdvSoliton",
    "colormap": "PerceptualThermal",
    "nonlinear_enabled": true,
    "domain_length": 32.0,
    "fast_speed_c1": 16.0,
    "slow_speed_c2": 4.0,
    "fast_x0": -10.0,
    "slow_x0": -1.0,
    "pulse_x0": 0.0,
    "pulse_speed_c": 12.0,
    "cosine_amplitude": 2.8,
    "cosine_harmonic": 1,
    "nls_coupling": 1.0,
    "time_step": 0.008,
    "show_initial_overlay": true,
    "show_ballistic_markers": false
  }
}"#;

const CUBIC_NLS_COLLISION_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Cubic NLS Bright Envelope Collision",
    "domain": "analysis",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "preset": "CubicNlsEnvelopeCollision",
    "mode": "CubicNls1D",
    "colormap": "WcagHighContrast",
    "nonlinear_enabled": true,
    "domain_length": 32.0,
    "fast_speed_c1": 9.0,
    "slow_speed_c2": 4.0,
    "fast_x0": -6.0,
    "slow_x0": 6.0,
    "pulse_x0": 0.0,
    "pulse_speed_c": 12.0,
    "cosine_amplitude": 2.5,
    "cosine_harmonic": 1,
    "nls_coupling": 1.0,
    "time_step": 0.008,
    "show_initial_overlay": true,
    "show_ballistic_markers": false
  }
}"#;

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "SolitonCollisionTool",
        display_name: "KdV Soliton & NLS Studio",
        domain: "analysis",
        tags: &["pde", "kdv", "soliton", "nls", "fft", "dispersion"],
        build: || Box::new(SolitonCollisionTool::default()),
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "kdv_two_soliton_overtaking_starter",
        title: "Two-Soliton Overtaking Collision",
        description: "Elastic overtaking collision of fast c1 = 16 and slow c2 = 4 KdV solitons verifying inverse-scattering phase shifts +0.5493 and -1.0986.",
        category: "Partial Differential Equations",
        icon: "🌊",
        domain: "analysis",
        tool_name: Some("SolitonCollisionTool"),
        preset_generator: || TWO_SOLITON_OVERTAKING_JSON,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "kdv_linear_airy_dispersion_starter",
        title: "Linear Airy Dispersion Collapse",
        description: "Disables 6 u u_x nonlinear steepening on the identical two-pulse profile to contrast oscillating Airy dispersion (> 60% peak decay).",
        category: "Partial Differential Equations",
        icon: "📉",
        domain: "analysis",
        tool_name: Some("SolitonCollisionTool"),
        preset_generator: || LINEAR_AIRY_COLLAPSE_JSON,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "kdv_zabusky_kruskal_fission_starter",
        title: "Zabusky-Kruskal Soliton Fission",
        description: "Classic 1965 Zabusky-Kruskal experiment breaking a smooth cosine wave into a rank-ordered soliton train with Fermi-Pasta-Ulam recurrence.",
        category: "Partial Differential Equations",
        icon: "⚡",
        domain: "analysis",
        tool_name: Some("SolitonCollisionTool"),
        preset_generator: || ZABUSKY_KRUSKAL_FISSION_JSON,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "nls_bright_envelope_collision_starter",
        title: "Cubic NLS Bright Envelope Collision",
        description: "Counter-propagating focusing 1D cubic Nonlinear Schrodinger bright envelope solitons evolved via unitary Strang split-step FFT.",
        category: "Partial Differential Equations",
        icon: "✨",
        domain: "analysis",
        tool_name: Some("SolitonCollisionTool"),
        preset_generator: || CUBIC_NLS_COLLISION_JSON,
    }
}
