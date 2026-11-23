//! Starter counterfactual shadow twin preset payloads for the four pilot simulation tools.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

/// JSON preset envelope for Lorenz Butterfly Perturbation (`AttractorPlotter`).
pub const PRESET_LORENZ_BUTTERFLY_PERTURBATION: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Lorenz Attractor - Butterfly Perturbation (δx = 1e-5)",
    "domain": "chaos",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "sigma": 10.0,
    "rho": 28.0,
    "beta": 2.6666666666666665,
    "dt": 0.01,
    "simulation_speed": 5,
    "paused": true,
    "show_annotations": true,
    "fork": {
      "fork_step": 150,
      "post_fork_steps": 2600,
      "baseline_sigma": 10.0,
      "baseline_rho": 28.0,
      "baseline_beta": 2.6666666666666665,
      "delta_x": 0.00001
    }
  }
}"#;

/// JSON preset envelope for SIR Lockdown Timing Intervention (`SirTool`).
pub const PRESET_SIR_LOCKDOWN_TIMING: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "SIR Epidemic - Day-10 Lockdown Intervention",
    "domain": "epidemiology",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "n": 1000.0,
    "i0": 1.0,
    "beta": 0.2,
    "gamma": 0.1,
    "duration": 100.0,
    "fork": {
      "fork_step": 100,
      "baseline_n": 1000.0,
      "baseline_i0": 1.0,
      "baseline_beta": 0.5,
      "baseline_gamma": 0.1,
      "delta_x": 0.0
    }
  }
}"#;

/// JSON preset envelope for MRI Spin-Echo Refocusing Intervention (`MriTool`).
pub const PRESET_MRI_SPIN_ECHO_REFOCUSING: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "MRI Bloch - 180° Spin-Echo Refocusing",
    "domain": "mri",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "t1": 1.0,
    "t2": 0.8,
    "bx": 0.0,
    "by": 0.0,
    "bz": 0.0000001,
    "time_scale": 5.0,
    "state": "Ready",
    "fork": {
      "fork_step": 120,
      "post_fork_steps": 120,
      "baseline_t1": 1.0,
      "baseline_t2": 0.8,
      "baseline_bx": 0.0,
      "baseline_by": 0.0,
      "baseline_bz": 0.0000001,
      "delta_x": 0.0,
      "refocus_180": true
    }
  }
}"#;

/// JSON preset envelope for Schnakenberg Inhibitor Diffusion Fork (`MorphogenesisTool`).
pub const PRESET_MORPHOGENESIS_INHIBITOR_FORK: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Turing Morphogenesis - Inhibitor Diffusion Fork",
    "domain": "biology",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "a": 0.1,
    "b": 0.9,
    "d_u": 1.0,
    "d_v": 18.0,
    "dt": 0.005,
    "width": 64,
    "height": 64,
    "state": "Ready",
    "fork": {
      "fork_step": 60,
      "post_fork_steps": 80,
      "baseline_a": 0.1,
      "baseline_b": 0.9,
      "baseline_du": 1.0,
      "baseline_dv": 40.0,
      "delta_u": 0.0,
      "heatmap_mode": "DifferenceAbs"
    }
  }
}"#;
