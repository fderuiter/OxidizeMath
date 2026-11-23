#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

// [cite:quantum_mechanics]

use math_explorer::physics::quantum::gross_pitaevskii::{GpeTrapKind, GrossPitaevskiiSolver2D};

use super::{
    CondensateViewMode, SuperfluidPreset, SuperfluidVortexState, SuperfluidVortexTool,
    COUPLING_G_MAX, COUPLING_G_MIN, CURSOR_BOUND, GPE_DT_MAX, GPE_DT_MIN, HEALING_MAX, HEALING_MIN,
    OBSTACLE_H_MAX, OBSTACLE_H_MIN, OBSTACLE_W_MAX, OBSTACLE_W_MIN, OMEGA_MAX, OMEGA_MIN,
};

fn sanitize_bounded(val: f64, min: f64, max: f64, fallback: f64) -> f64 {
    if val.is_finite() {
        val.clamp(min, max)
    } else {
        fallback
    }
}

pub(super) fn is_valid_state(s: &SuperfluidVortexState) -> bool {
    s.interaction_g.is_finite()
        && (COUPLING_G_MIN..=COUPLING_G_MAX).contains(&s.interaction_g)
        && s.omega_x.is_finite()
        && (OMEGA_MIN..=OMEGA_MAX).contains(&s.omega_x)
        && s.omega_y.is_finite()
        && (OMEGA_MIN..=OMEGA_MAX).contains(&s.omega_y)
        && s.obstacle_height.is_finite()
        && (OBSTACLE_H_MIN..=OBSTACLE_H_MAX).contains(&s.obstacle_height)
        && s.obstacle_width.is_finite()
        && (OBSTACLE_W_MIN..=OBSTACLE_W_MAX).contains(&s.obstacle_width)
        && s.dipole_separation.is_finite()
        && (0.8..=6.0).contains(&s.dipole_separation)
        && s.healing_length.is_finite()
        && (HEALING_MIN..=HEALING_MAX).contains(&s.healing_length)
        && s.time_step.is_finite()
        && (GPE_DT_MIN..=GPE_DT_MAX).contains(&s.time_step)
        && s.cursor_x.is_finite()
        && (-CURSOR_BOUND..=CURSOR_BOUND).contains(&s.cursor_x)
        && s.cursor_y.is_finite()
        && (-CURSOR_BOUND..=CURSOR_BOUND).contains(&s.cursor_y)
        && (s.stamp_charge == 1 || s.stamp_charge == -1)
}

pub(super) fn clamp_state_fields(s: &mut SuperfluidVortexState) {
    s.interaction_g = sanitize_bounded(s.interaction_g, COUPLING_G_MIN, COUPLING_G_MAX, 220.0);
    s.omega_x = sanitize_bounded(s.omega_x, OMEGA_MIN, OMEGA_MAX, 1.0);
    s.omega_y = sanitize_bounded(s.omega_y, OMEGA_MIN, OMEGA_MAX, 1.0);
    s.obstacle_height = sanitize_bounded(s.obstacle_height, OBSTACLE_H_MIN, OBSTACLE_H_MAX, 12.0);
    s.obstacle_width = sanitize_bounded(s.obstacle_width, OBSTACLE_W_MIN, OBSTACLE_W_MAX, 1.0);
    s.dipole_separation = sanitize_bounded(s.dipole_separation, 0.8, 6.0, 2.8);
    s.healing_length = sanitize_bounded(s.healing_length, HEALING_MIN, HEALING_MAX, 0.34);
    s.time_step = sanitize_bounded(s.time_step, GPE_DT_MIN, GPE_DT_MAX, 0.003);
    s.cursor_x = sanitize_bounded(s.cursor_x, -CURSOR_BOUND, CURSOR_BOUND, 0.0);
    s.cursor_y = sanitize_bounded(s.cursor_y, -CURSOR_BOUND, CURSOR_BOUND, 0.0);
    if s.stamp_charge >= 0 {
        s.stamp_charge = 1;
    } else {
        s.stamp_charge = -1;
    }
}

fn base_superfluid_state(preset: SuperfluidPreset) -> SuperfluidVortexState {
    SuperfluidVortexState {
        preset,
        trap_kind: GpeTrapKind::HarmonicIsotropic,
        view_mode: CondensateViewMode::SuperfluidDensity,
        interaction_g: 220.0,
        imaginary_time_cooling: false,
        omega_x: 1.0,
        omega_y: 1.0,
        obstacle_height: 12.0,
        obstacle_width: 1.0,
        dipole_separation: 2.8,
        healing_length: 0.34,
        time_step: 0.003,
        cursor_x: 0.0,
        cursor_y: 0.0,
        stamp_charge: 1,
        show_vortex_markers: true,
    }
}

pub(super) fn default_state_for_preset(preset: SuperfluidPreset) -> SuperfluidVortexState {
    match preset {
        SuperfluidPreset::VortexDipole => base_superfluid_state(preset),
        SuperfluidPreset::VortexQuadrupole => SuperfluidVortexState {
            view_mode: CondensateViewMode::CyclicPhase,
            interaction_g: 240.0,
            dipole_separation: 3.0,
            ..base_superfluid_state(preset)
        },
        SuperfluidPreset::CoRotatingPair => SuperfluidVortexState {
            trap_kind: GpeTrapKind::HarmonicAnisotropic,
            omega_x: 1.0,
            omega_y: 1.15,
            dipole_separation: 3.0,
            ..base_superfluid_state(preset)
        },
        SuperfluidPreset::ObstaclePinningCooling => SuperfluidVortexState {
            trap_kind: GpeTrapKind::CentralObstacle,
            view_mode: CondensateViewMode::WcagHighContrastDensity,
            interaction_g: 260.0,
            imaginary_time_cooling: true,
            obstacle_height: 16.0,
            obstacle_width: 1.05,
            dipole_separation: 3.4,
            ..base_superfluid_state(preset)
        },
    }
}

pub(super) fn imprint_preset_vortices(
    solver: &mut GrossPitaevskiiSolver2D,
    state: &SuperfluidVortexState,
) {
    let half = 0.5 * state.dipole_separation;
    let xi = state.healing_length;
    match state.preset {
        SuperfluidPreset::VortexDipole => {
            solver.imprint_vortex_dipole(-half, 0.0, half, 0.0, xi);
        }
        SuperfluidPreset::VortexQuadrupole => {
            solver.imprint_vortex(-half, -half, 1, xi);
            solver.imprint_vortex(half, -half, -1, xi);
            solver.imprint_vortex(half, half, 1, xi);
            solver.imprint_vortex(-half, half, -1, xi);
        }
        SuperfluidPreset::CoRotatingPair => {
            solver.imprint_vortex(-half, 0.0, 1, xi);
            solver.imprint_vortex(half, 0.0, 1, xi);
        }
        SuperfluidPreset::ObstaclePinningCooling => {
            solver.imprint_vortex_dipole(-half, 0.5, half, -0.5, xi);
        }
    }
}

impl scientific_metadata::theory::TheoryDescribable for SuperfluidVortexTool {
    fn theory_description(&self) -> String {
        "2D Strang split-step Fourier solver for Gross-Pitaevskii Bose-Einstein condensates i d_t psi = (-1/2 nabla^2 + V(x,y) + g |psi|^2) psi with real-time unitary dynamics, imaginary-time ground-state cooling, and 2x2 lattice plaquette wrapped-phase-difference detection of quantized +2pi and -2pi topological vortex cores.".into()
    }

    fn phonetic_description(&self) -> String {
        "Two dimensional Gross Pitaevskii superfluid Bose Einstein condensate and quantized topological vortex studio"
            .into()
    }

    fn theory_citation(&self) -> String {
        "Gross, Structure of a quantized vortex in boson systems, Nuovo Cimento 20 (1961) 10.1007/BF02731494; Pitaevskii, Vortex lines in an imperfect Bose gas, Sov. Phys. JETP 13 (1961)".into()
    }

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "VortexDipole".into(),
            "Counter-rotating +2pi and -2pi vortex-antivortex dipole translating across a trapped superfluid condensate with zero net winding".into(),
        );
        map.insert(
            "VortexQuadrupole".into(),
            "Alternating four-vortex topological quadrupole (+1, -1, +1, -1) exhibiting branch-cut reconnection Resistance".into(),
        );
        map.insert(
            "CoRotatingPair".into(),
            "Same-sign +1/+1 quantized vortex pair precessing around the condensate center with total circulation +4pi".into(),
        );
        map.insert(
            "ObstaclePinningCooling".into(),
            "Wick-rotated imaginary-time relaxation around a central repulsive Gaussian laser barrier".into(),
        );
        map
    }
}

const VORTEX_DIPOLE_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Superfluid Vortex Dipole (+/-2pi)",
    "domain": "quantum",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "preset": "VortexDipole",
    "trap_kind": "HarmonicIsotropic",
    "view_mode": "SuperfluidDensity",
    "interaction_g": 220.0,
    "imaginary_time_cooling": false,
    "omega_x": 1.0,
    "omega_y": 1.0,
    "obstacle_height": 12.0,
    "obstacle_width": 1.0,
    "dipole_separation": 2.8,
    "healing_length": 0.34,
    "time_step": 0.003,
    "cursor_x": 0.0,
    "cursor_y": 0.0,
    "stamp_charge": 1,
    "show_vortex_markers": true
  }
}"#;

const VORTEX_QUADRUPOLE_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Quantized Vortex Quadrupole",
    "domain": "quantum",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "preset": "VortexQuadrupole",
    "trap_kind": "HarmonicIsotropic",
    "view_mode": "CyclicPhase",
    "interaction_g": 240.0,
    "imaginary_time_cooling": false,
    "omega_x": 1.0,
    "omega_y": 1.0,
    "obstacle_height": 12.0,
    "obstacle_width": 1.0,
    "dipole_separation": 3.0,
    "healing_length": 0.34,
    "time_step": 0.003,
    "cursor_x": 0.0,
    "cursor_y": 0.0,
    "stamp_charge": 1,
    "show_vortex_markers": true
  }
}"#;

const COROTATING_PAIR_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Co-Rotating Vortex Pair (+2pi/+2pi)",
    "domain": "quantum",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "preset": "CoRotatingPair",
    "trap_kind": "HarmonicAnisotropic",
    "view_mode": "SuperfluidDensity",
    "interaction_g": 220.0,
    "imaginary_time_cooling": false,
    "omega_x": 1.0,
    "omega_y": 1.15,
    "obstacle_height": 12.0,
    "obstacle_width": 1.0,
    "dipole_separation": 3.0,
    "healing_length": 0.34,
    "time_step": 0.003,
    "cursor_x": 0.0,
    "cursor_y": 0.0,
    "stamp_charge": 1,
    "show_vortex_markers": true
  }
}"#;

const OBSTACLE_COOLING_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Obstacle Pinning & Cooling Ground State",
    "domain": "quantum",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "preset": "ObstaclePinningCooling",
    "trap_kind": "CentralObstacle",
    "view_mode": "WcagHighContrastDensity",
    "interaction_g": 260.0,
    "imaginary_time_cooling": true,
    "omega_x": 1.0,
    "omega_y": 1.0,
    "obstacle_height": 16.0,
    "obstacle_width": 1.05,
    "dipole_separation": 3.4,
    "healing_length": 0.34,
    "time_step": 0.003,
    "cursor_x": 0.0,
    "cursor_y": 0.0,
    "stamp_charge": 1,
    "show_vortex_markers": true
  }
}"#;

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "SuperfluidVortexTool",
        display_name: "Superfluid Vortex & GPE Studio",
        domain: "quantum",
        tags: &["quantum", "superfluid", "vortex", "gpe", "bec", "fft"],
        build: || Box::new(SuperfluidVortexTool::default()),
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "gpe_vortex_dipole_starter",
        title: "Superfluid Vortex Dipole (+/-2pi)",
        description: "Counter-rotating +1/-1 quantized vortex dipole in an isotropic Bose-Einstein condensate with unitary norm conservation.",
        category: "Quantum Many-Body & Superfluids",
        icon: "🌀",
        domain: "quantum",
        tool_name: Some("SuperfluidVortexTool"),
        preset_generator: || VORTEX_DIPOLE_PRESET_JSON,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "gpe_vortex_quadrupole_starter",
        title: "Quantized Vortex Quadrupole",
        description: "Four-core alternating (+1, -1, +1, -1) superfluid vortex quadrupole visualized in WCAG cyclic [-pi, pi] phase space.",
        category: "Quantum Many-Body & Superfluids",
        icon: "🧭",
        domain: "quantum",
        tool_name: Some("SuperfluidVortexTool"),
        preset_generator: || VORTEX_QUADRUPOLE_PRESET_JSON,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "gpe_corotating_pair_starter",
        title: "Co-Rotating Vortex Pair (+2pi/+2pi)",
        description: "Two same-sign +1 quantized vortices precessing around an elliptic harmonic trap with quantized net circulation +4pi.",
        category: "Quantum Many-Body & Superfluids",
        icon: "⚛",
        domain: "quantum",
        tool_name: Some("SuperfluidVortexTool"),
        preset_generator: || COROTATING_PAIR_PRESET_JSON,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "gpe_obstacle_cooling_starter",
        title: "Obstacle Pinning & Cooling Ground State",
        description: "Imaginary-time Wick-rotated Gross-Pitaevskii relaxation around a central repulsive Gaussian laser barrier.",
        category: "Quantum Many-Body & Superfluids",
        icon: "❄",
        domain: "quantum",
        tool_name: Some("SuperfluidVortexTool"),
        preset_generator: || OBSTACLE_COOLING_PRESET_JSON,
    }
}
