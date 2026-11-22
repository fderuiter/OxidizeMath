#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

/// Legacy starter preset JSON envelope for the Fractal Viewer.
pub const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Mandelbrot Explorer",
    "domain": "chaos",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "center_re": -0.75,
    "center_im": 0.0,
    "zoom": 1.0,
    "max_iter": 150,
    "mode": "Mandelbrot",
    "julia_re": -0.7,
    "julia_im": 0.27015
  }
}"#;

/// Starter preset JSON envelope just inside the Mandelbrot main cardioid cusp (`c = 0.24`).
pub const STARTER_CUSP_CONNECTED_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Mandelbrot Cusp: Connected Julia (c = 0.24)",
    "domain": "chaos",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "center_re": 0.25,
    "center_im": 0.0,
    "zoom": 4.0,
    "max_iter": 200,
    "mode": "Mandelbrot",
    "julia_re": 0.24,
    "julia_im": 0.0,
    "c_pinned": true
  }
}"#;

/// Starter preset JSON envelope just outside the Mandelbrot cusp (`c = 0.27`, Cantor dust `J_c`).
pub const STARTER_CUSP_DUST_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Mandelbrot Cusp: Cantor Dust (c = 0.27)",
    "domain": "chaos",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "center_re": 0.25,
    "center_im": 0.0,
    "zoom": 4.0,
    "max_iter": 200,
    "mode": "Mandelbrot",
    "julia_re": 0.27,
    "julia_im": 0.0,
    "c_pinned": true
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "fractal_viewer_starter",
        title: "Mandelbrot Explorer",
        description: "z_{n+1} = z_n^2 + c: Complex quadratic polynomial iteration and boundary fractal escape times.",
        category: "Fractals",
        icon: "❄️",
        domain: "chaos",
        tool_name: Some("FractalViewer"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "fractal_mandelbrot_cusp_connected",
        title: "Mandelbrot Cusp: Connected Julia (c = 0.24)",
        description: "Pinned c = 0.24 inside the main cardioid cusp with bounded critical orbit and connected J_c.",
        category: "Fractals",
        icon: "🔗",
        domain: "chaos",
        tool_name: Some("FractalViewer"),
        preset_generator: || STARTER_CUSP_CONNECTED_JSON,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "fractal_mandelbrot_cusp_dust",
        title: "Mandelbrot Cusp: Cantor Dust (c = 0.27)",
        description: "Pinned c = 0.27 past the c = 0.25 cusp bottleneck where escaping critical orbits shatter J_c into dust.",
        category: "Fractals",
        icon: "✨",
        domain: "chaos",
        tool_name: Some("FractalViewer"),
        preset_generator: || STARTER_CUSP_DUST_JSON,
    }
}
