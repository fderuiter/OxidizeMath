#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

/// JSON preset envelope for Fluid Dynamics Cylinder Vector Field.
pub const PRESET_VECTOR_FIELD: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Potential Flow - Cylinder Vector Field",
    "domain": "fluid_dynamics",
    "timestamp": "2026-11-14T00:00:00Z"
  },
  "config": {
    "elements": [
      {
        "element_type": "Uniform",
        "strength": 10.0,
        "x": 0.0,
        "y": 0.0,
        "angle": 0.0
      },
      {
        "element_type": "Doublet",
        "strength": 251.32741228718345,
        "x": 0.0,
        "y": 0.0,
        "angle": 0.0
      }
    ],
    "grid_size": 20,
    "view_range": 10.0
  }
}"#;

/// JSON preset envelope for Fluid Dynamics Source-Sink Pair.
pub const PRESET_SOURCE_SINK: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Potential Flow - Source-Sink Singularity Pair",
    "domain": "fluid_dynamics",
    "timestamp": "2026-11-14T00:00:00Z"
  },
  "config": {
    "elements": [
      {
        "element_type": "Source",
        "strength": 10.0,
        "x": -3.0,
        "y": 0.0,
        "angle": 0.0
      },
      {
        "element_type": "Sink",
        "strength": 10.0,
        "x": 3.0,
        "y": 0.0,
        "angle": 0.0
      }
    ],
    "grid_size": 25,
    "view_range": 12.0
  }
}"#;

/// JSON preset envelope for Kutta-Joukowski Lifting Cylinder potential flow.
pub const PRESET_LIFTING_CYLINDER: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Potential Flow - Kutta-Joukowski Lifting Cylinder",
    "domain": "fluid_dynamics",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "elements": [
      { "element_type": "Uniform", "strength": 10.0, "x": 0.0, "y": 0.0, "angle": 0.0 },
      { "element_type": "Doublet", "strength": 251.32741228718345, "x": 0.0, "y": 0.0, "angle": 0.0 },
      { "element_type": "Vortex", "strength": -125.66370614359172, "x": 0.0, "y": 0.0, "angle": 0.0 }
    ],
    "grid_size": 20,
    "view_range": 10.0,
    "dye_rake_x": -8.5,
    "dye_rake_y": 0.0,
    "dye_rake_span": 7.0,
    "dye_count": 9
  }
}"#;

/// JSON preset envelope for Rankine Oval closed streamline body.
pub const PRESET_RANKINE_OVAL: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Potential Flow - Rankine Oval",
    "domain": "fluid_dynamics",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "elements": [
      { "element_type": "Uniform", "strength": 10.0, "x": 0.0, "y": 0.0, "angle": 0.0 },
      { "element_type": "Source", "strength": 60.0, "x": -3.0, "y": 0.0, "angle": 0.0 },
      { "element_type": "Sink", "strength": 60.0, "x": 3.0, "y": 0.0, "angle": 0.0 }
    ],
    "grid_size": 22,
    "view_range": 10.0,
    "dye_rake_x": -8.5,
    "dye_rake_y": 0.0,
    "dye_rake_span": 6.5,
    "dye_count": 9
  }
}"#;

/// JSON preset envelope for Kelvin-Stuart staggered vortex street.
pub const PRESET_VORTEX_STREET: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Potential Flow - Kelvin-Stuart Vortex Street",
    "domain": "fluid_dynamics",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "elements": [
      { "element_type": "Uniform", "strength": 6.0, "x": 0.0, "y": 0.0, "angle": 0.0 },
      { "element_type": "Vortex", "strength": -28.0, "x": -6.0, "y": 1.12, "angle": 0.0 },
      { "element_type": "Vortex", "strength": -28.0, "x": -2.0, "y": 1.12, "angle": 0.0 },
      { "element_type": "Vortex", "strength": -28.0, "x": 2.0, "y": 1.12, "angle": 0.0 },
      { "element_type": "Vortex", "strength": -28.0, "x": 6.0, "y": 1.12, "angle": 0.0 },
      { "element_type": "Vortex", "strength": 28.0, "x": -4.0, "y": -1.12, "angle": 0.0 },
      { "element_type": "Vortex", "strength": 28.0, "x": 0.0, "y": -1.12, "angle": 0.0 },
      { "element_type": "Vortex", "strength": 28.0, "x": 4.0, "y": -1.12, "angle": 0.0 },
      { "element_type": "Vortex", "strength": 28.0, "x": 8.0, "y": -1.12, "angle": 0.0 }
    ],
    "grid_size": 24,
    "view_range": 12.0,
    "dye_rake_x": -10.0,
    "dye_rake_y": 0.0,
    "dye_rake_span": 5.0,
    "dye_count": 9
  }
}"#;
