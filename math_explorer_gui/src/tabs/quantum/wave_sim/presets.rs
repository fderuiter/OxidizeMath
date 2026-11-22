#![cfg_attr(any(), verified(opt_out = "gui_tool"))]
// [cite:quantum_mechanics]

use super::{DEFAULT_DT, DEFAULT_K0, DEFAULT_SIGMA, DEFAULT_X0, WaveUnified};
use scientific_metadata::theory::{ParameterConstraint, TheoryDescribable};
use std::collections::HashMap;

pub(super) const POTENTIAL_CHOICES: &[(&str, f64)] = &[
    ("Infinite Potential Well", 0.0),
    ("Harmonic Oscillator", 1.0),
    ("Quantum Tunneling Barrier", 2.0),
    ("Resonant Double Well", 3.0),
    ("Stark Linear Ramp", 4.0),
];

pub(super) fn wave_parameters() -> HashMap<String, ParameterConstraint> {
    let mut map = HashMap::new();
    map.insert(
        "potential_type".to_string(),
        ParameterConstraint {
            min: 0.0,
            max: 4.0,
            step: 1.0,
        },
    );
    map.insert(
        "dt".to_string(),
        ParameterConstraint {
            min: 0.001,
            max: 0.1,
            step: 0.005,
        },
    );
    map.insert(
        "x0".to_string(),
        ParameterConstraint {
            min: -4.5,
            max: 4.5,
            step: 0.1,
        },
    );
    map.insert(
        "k0".to_string(),
        ParameterConstraint {
            min: -15.0,
            max: 15.0,
            step: 0.25,
        },
    );
    map.insert(
        "sigma".to_string(),
        ParameterConstraint {
            min: 0.2,
            max: 2.0,
            step: 0.05,
        },
    );
    map
}

pub(super) fn wave_default_parameter(name: &str, constraint: &ParameterConstraint) -> f64 {
    match name {
        "potential_type" => 0.0,
        "dt" => DEFAULT_DT,
        "x0" => DEFAULT_X0,
        "k0" => DEFAULT_K0,
        "sigma" => DEFAULT_SIGMA,
        _ => (constraint.min + constraint.max) / 2.0,
    }
}

pub(super) fn wave_parameter_label(name: &str) -> String {
    match name {
        "potential_type" => "Potential Profile".to_string(),
        "dt" => "Time Step (Δt)".to_string(),
        "x0" => "Packet Center (x₀)".to_string(),
        "k0" => "Initial Momentum (k₀)".to_string(),
        "sigma" => "Packet Width (σ)".to_string(),
        other => other.to_string(),
    }
}

pub(super) fn wave_parameter_choices(name: &str) -> &'static [(&'static str, f64)] {
    match name {
        "potential_type" => POTENTIAL_CHOICES,
        _ => &[],
    }
}

pub(super) fn wave_sidebar_presets() -> Vec<(&'static str, HashMap<String, f64>)> {
    vec![
        (
            "Tunneling",
            HashMap::from([
                ("potential_type".to_string(), 2.0),
                ("dt".to_string(), 0.02),
                ("x0".to_string(), -2.4),
                ("k0".to_string(), 5.0),
                ("sigma".to_string(), 0.45),
            ]),
        ),
        (
            "Double Well",
            HashMap::from([
                ("potential_type".to_string(), 3.0),
                ("dt".to_string(), 0.02),
                ("x0".to_string(), -2.8),
                ("k0".to_string(), 5.2),
                ("sigma".to_string(), 0.4),
            ]),
        ),
        (
            "Stark Ramp",
            HashMap::from([
                ("potential_type".to_string(), 4.0),
                ("dt".to_string(), 0.02),
                ("x0".to_string(), -2.2),
                ("k0".to_string(), 5.0),
                ("sigma".to_string(), 0.5),
            ]),
        ),
        (
            "Harmonic",
            HashMap::from([
                ("potential_type".to_string(), 1.0),
                ("dt".to_string(), 0.02),
                ("x0".to_string(), -2.0),
                ("k0".to_string(), 5.0),
                ("sigma".to_string(), 0.5),
            ]),
        ),
    ]
}

impl TheoryDescribable for WaveUnified {
    fn theory_description(&self) -> String {
        "Time-dependent 1D Schrödinger wave packet evolution $i\\hbar \\partial_t \\psi(x,t) = [-\\frac{\\hbar^2}{2m}\\partial_x^2 + V(x)]\\psi(x,t)$ with interactive freehand potential sculpting, quantum tunneling barriers, resonant double-well cavities, Stark ramps, and live $P_T / P_R$ probability readouts.".to_string()
    }

    fn phonetic_description(&self) -> String {
        "Time-dependent one-dimensional Schrodinger wave packet and quantum potential sculpting simulation".to_string()
    }

    fn theory_citation(&self) -> String {
        "Schrödinger, E. (1926). An Undulatory Theory of the Mechanics of Atoms and Molecules. Physical Review, 28(6), 1049-1070; Griffiths, D. J., & Schroeter, D. F. (2018). Introduction to Quantum Mechanics (3rd ed.). Cambridge University Press.".to_string()
    }

    fn available_descriptions(&self) -> HashMap<String, String> {
        HashMap::from([
            (
                "potential_type".to_string(),
                "Selects among Infinite Potential Well, Harmonic Oscillator, Quantum Tunneling Barrier, Resonant Double Well, and Stark Linear Ramp baseline profiles.".to_string(),
            ),
            (
                "dt".to_string(),
                "Integration time step Δt controlling unitary wave-packet propagation per tick.".to_string(),
            ),
            (
                "x0".to_string(),
                "Initial center coordinate x₀ of the emitted Gaussian wavepacket.".to_string(),
            ),
            (
                "k0".to_string(),
                "Initial wave number k₀ controlling wavepacket momentum and incident kinetic energy k₀²/2.".to_string(),
            ),
            (
                "sigma".to_string(),
                "Spatial Gaussian envelope width σ ∈ [0.2, 2.0] controlling position-momentum spread.".to_string(),
            ),
        ])
    }
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Schr\u00f6dinger Wave Packet",
    "domain": "quantum",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "potential_type": 1.0,
    "dt": 0.02,
    "x0": -2.0,
    "k0": 5.0,
    "sigma": 0.5
  }
}"#;

const TUNNELING_BARRIER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Quantum Tunneling Barrier",
    "domain": "quantum",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "potential_type": 2.0,
    "dt": 0.02,
    "x0": -2.4,
    "k0": 5.0,
    "sigma": 0.45
  }
}"#;

const RESONANT_DOUBLE_WELL_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Resonant Double-Well Cavity",
    "domain": "quantum",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "potential_type": 3.0,
    "dt": 0.02,
    "x0": -2.8,
    "k0": 5.2,
    "sigma": 0.4
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "wave_sim_starter",
        title: "Schrödinger Wave Packet",
        description: "iℏ·∂ψ/∂t = -ℏ²/(2m)∇²ψ + V·ψ: 1D Gaussian wave packet evolution in harmonic and sculpted potentials.",
        category: "Wave Mechanics",
        icon: "🌊",
        domain: "quantum",
        tool_name: Some("WaveSimulator"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "wave_sim_tunneling_barrier",
        title: "Quantum Tunneling Barrier",
        description: "Sub-barrier wavepacket scattering across a rectangular potential barrier with reflected fringes and evanescent tunneling.",
        category: "Wave Mechanics",
        icon: "⛰",
        domain: "quantum",
        tool_name: Some("WaveSimulator"),
        preset_generator: || TUNNELING_BARRIER_PRESET_JSON,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "wave_sim_resonant_double_well",
        title: "Resonant Double-Well Cavity",
        description: "Fabry-Pérot quantum resonance and quasi-bound wavepacket trapping across symmetric dual potential barriers.",
        category: "Wave Mechanics",
        icon: "⛩",
        domain: "quantum",
        tool_name: Some("WaveSimulator"),
        preset_generator: || RESONANT_DOUBLE_WELL_PRESET_JSON,
    }
}
