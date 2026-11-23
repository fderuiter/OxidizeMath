#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

// [cite:graph_parameters_rust]

use math_explorer::pure_math::analysis::pde::wave_2d::{BoundaryMode2D, Wave2DMedium};

use super::{
    ChladniWaveTankState, ChladniWaveTankTool, ClampLayout, SecondaryPlotMode, TankPreset,
    WaveColormap, AMP_MAX, AMP_MIN, DAMPING_MAX, DAMPING_MIN, DT_MAX, DT_MIN, FREQ_MAX, FREQ_MIN,
    GRID_SIZE, REFRACTIVE_MAX, REFRACTIVE_MIN, SLIT_SEP_MAX, SLIT_SEP_MIN, SPEED_MAX, SPEED_MIN,
    STIFFNESS_MAX, STIFFNESS_MIN,
};

fn sanitize_bounded(val: f64, min: f64, max: f64, fallback: f64) -> f64 {
    if val.is_finite() {
        val.clamp(min, max)
    } else {
        fallback
    }
}

pub(super) fn is_valid_state(s: &ChladniWaveTankState) -> bool {
    s.wave_speed.is_finite()
        && (SPEED_MIN..=SPEED_MAX).contains(&s.wave_speed)
        && s.plate_stiffness.is_finite()
        && (STIFFNESS_MIN..=STIFFNESS_MAX).contains(&s.plate_stiffness)
        && s.damping.is_finite()
        && (DAMPING_MIN..=DAMPING_MAX).contains(&s.damping)
        && s.time_step.is_finite()
        && (DT_MIN..=DT_MAX).contains(&s.time_step)
        && s.driver_x.is_finite()
        && (0.0..=1.0).contains(&s.driver_x)
        && s.driver_y.is_finite()
        && (0.0..=1.0).contains(&s.driver_y)
        && s.driver_frequency.is_finite()
        && (FREQ_MIN..=FREQ_MAX).contains(&s.driver_frequency)
        && s.driver_amplitude.is_finite()
        && (AMP_MIN..=AMP_MAX).contains(&s.driver_amplitude)
        && (1..=7).contains(&s.slit_count)
        && (1..=12).contains(&s.slit_width)
        && s.slit_separation.is_finite()
        && (SLIT_SEP_MIN..=SLIT_SEP_MAX).contains(&s.slit_separation)
        && (4..=(GRID_SIZE - 8)).contains(&s.barrier_x)
        && (8..=(GRID_SIZE - 4)).contains(&s.detector_x)
        && s.refractive_index.is_finite()
        && (REFRACTIVE_MIN..=REFRACTIVE_MAX).contains(&s.refractive_index)
        && s.lens_aperture.is_finite()
        && (8.0..=56.0).contains(&s.lens_aperture)
        && super::sonification::are_custom_clamps_valid(&s.custom_clamps)
}

pub(super) fn clamp_state_fields(s: &mut ChladniWaveTankState) {
    s.wave_speed = sanitize_bounded(s.wave_speed, SPEED_MIN, SPEED_MAX, 1.0);
    s.plate_stiffness = sanitize_bounded(s.plate_stiffness, STIFFNESS_MIN, STIFFNESS_MAX, 1.8e-4);
    s.damping = sanitize_bounded(s.damping, DAMPING_MIN, DAMPING_MAX, 0.05);
    s.time_step = sanitize_bounded(s.time_step, DT_MIN, DT_MAX, 0.004);
    s.driver_x = sanitize_bounded(s.driver_x, 0.0, 1.0, 0.5);
    s.driver_y = sanitize_bounded(s.driver_y, 0.0, 1.0, 0.5);
    s.driver_frequency = sanitize_bounded(s.driver_frequency, FREQ_MIN, FREQ_MAX, 10.0);
    s.driver_amplitude = sanitize_bounded(s.driver_amplitude, AMP_MIN, AMP_MAX, 25.0);
    s.slit_count = s.slit_count.clamp(1, 7);
    s.slit_width = s.slit_width.clamp(1, 12);
    s.slit_separation = sanitize_bounded(s.slit_separation, SLIT_SEP_MIN, SLIT_SEP_MAX, 32.0);
    s.barrier_x = s.barrier_x.clamp(4, GRID_SIZE - 8);
    s.detector_x = s.detector_x.clamp(8, GRID_SIZE - 4);
    s.refractive_index = sanitize_bounded(s.refractive_index, REFRACTIVE_MIN, REFRACTIVE_MAX, 1.6);
    s.lens_aperture = sanitize_bounded(s.lens_aperture, 8.0, 56.0, 34.0);
    super::sonification::sanitize_custom_clamps(&mut s.custom_clamps);
}

fn base_tank_state(preset: TankPreset) -> ChladniWaveTankState {
    ChladniWaveTankState {
        preset,
        medium: Wave2DMedium::AcousticWave,
        boundary: BoundaryMode2D::Sponge,
        clamp_layout: ClampLayout::None,
        colormap: WaveColormap::PerceptualThermal,
        secondary_plot: SecondaryPlotMode::DetectorCrossSection,
        wave_speed: 1.0,
        plate_stiffness: 0.0,
        damping: 0.002,
        time_step: 0.35,
        driver_enabled: true,
        line_source: false,
        driver_x: 0.18,
        driver_y: 0.34,
        driver_frequency: 11.5,
        driver_amplitude: 95.0,
        barrier_enabled: false,
        slit_count: 2,
        slit_width: 3,
        slit_separation: 32.0,
        barrier_x: 28,
        detector_x: 96,
        lens_enabled: false,
        refractive_index: 1.6,
        lens_aperture: 34.0,
        show_nodal_contours: true,
        show_sand_particles: false,
        rng_seed: 0xC41A_D21E,
        custom_clamps: Vec::new(),
    }
}

fn plate_preset_state(preset: TankPreset) -> ChladniWaveTankState {
    match preset {
        TankPreset::SquareChladniPlate => ChladniWaveTankState {
            medium: Wave2DMedium::BiharmonicPlate,
            boundary: BoundaryMode2D::Neumann,
            clamp_layout: ClampLayout::CenterPin,
            secondary_plot: SecondaryPlotMode::RadialFftSpectrum,
            wave_speed: 0.0,
            plate_stiffness: 0.0004,
            damping: 0.08,
            time_step: 0.002,
            driver_x: 0.22,
            driver_y: 0.50,
            driver_frequency: 0.56,
            driver_amplitude: 65.0,
            show_sand_particles: true,
            ..base_tank_state(preset)
        },
        _ => ChladniWaveTankState {
            boundary: BoundaryMode2D::Dirichlet,
            colormap: WaveColormap::BipolarPhase,
            secondary_plot: SecondaryPlotMode::RadialFftSpectrum,
            damping: 0.03,
            time_step: 0.004,
            driver_x: 0.34,
            driver_y: 0.50,
            driver_frequency: 2.98,
            driver_amplitude: 42.0,
            show_sand_particles: true,
            rng_seed: 0xD80A_BEEF,
            ..base_tank_state(preset)
        },
    }
}

pub(super) fn default_state_for_preset(preset: TankPreset) -> ChladniWaveTankState {
    match preset {
        TankPreset::SquareChladniPlate | TankPreset::CircularDrumhead => plate_preset_state(preset),
        TankPreset::DoubleSlitInterference => ChladniWaveTankState {
            colormap: WaveColormap::BipolarPhase,
            line_source: true,
            driver_x: 18.0 / 127.0,
            driver_y: 0.5,
            driver_frequency: 0.10,
            driver_amplitude: 3.2,
            barrier_enabled: true,
            slit_width: 2,
            slit_separation: 40.0,
            barrier_x: 24,
            show_nodal_contours: false,
            rng_seed: 0x5117_2026,
            ..base_tank_state(preset)
        },
        TankPreset::AcousticLens => ChladniWaveTankState {
            time_step: 0.40,
            line_source: true,
            driver_x: 16.0 / 127.0,
            driver_y: 0.5,
            driver_frequency: 0.14,
            driver_amplitude: 2.8,
            detector_x: 92,
            lens_enabled: true,
            refractive_index: 1.65,
            lens_aperture: 38.0,
            rng_seed: 0x1E25_F0C0,
            ..base_tank_state(preset)
        },
    }
}

fn resonant_seed_sample(state: &ChladniWaveTankState, xn: f64, yn: f64) -> f64 {
    let pi = std::f64::consts::PI;
    if state.preset == TankPreset::CircularDrumhead {
        let (dx, dy) = (xn - 0.5, yn - 0.5);
        let r = (dx * dx + dy * dy).sqrt() / 0.45;
        if r < 1.0 && r > 1.0e-6 {
            let radial = (pi * r).sin() * (1.85 * pi * r).cos();
            0.55 * radial * (dx * dx - dy * dy) / (dx * dx + dy * dy)
        } else {
            0.0
        }
    } else if state.medium == Wave2DMedium::BiharmonicPlate {
        0.45 * ((3.0 * pi * xn).cos() * (pi * yn).cos() - (pi * xn).cos() * (3.0 * pi * yn).cos())
    } else {
        0.0
    }
}

pub(super) fn seed_initial_resonant_field(
    solver: &mut math_explorer::pure_math::analysis::pde::wave_2d::Wave2DSolver,
    state: &ChladniWaveTankState,
) {
    if state.line_source {
        return;
    }
    let (w, h) = (solver.width(), solver.height());
    let (denom_x, denom_y) = ((w - 1).max(1) as f64, (h - 1).max(1) as f64);
    let mut init = vec![0.0; w * h];
    for y in 0..h {
        let yn = (y as f64) / denom_y;
        for x in 0..w {
            let idx = y * w + x;
            if solver.wall_mask[idx] > 0.5 {
                init[idx] = resonant_seed_sample(state, (x as f64) / denom_x, yn);
            }
        }
    }
    if init.iter().any(|&v| v.abs() > 1.0e-9) {
        solver.set_stationary_displacement(&init);
    } else {
        solver.inject_gaussian_pulse(state.driver_x, state.driver_y, 0.07, 0.45);
    }
}

impl scientific_metadata::theory::TheoryDescribable for ChladniWaveTankTool {
    fn theory_description(&self) -> String {
        "2D Störmer-Verlet solver integrating hyperbolic wave propagation u_tt + gamma u_t = c(x,y)^2 nabla^2 u + F and fourth-order Kirchhoff-Love biharmonic thin-plate bending u_tt + gamma u_t = -D nabla^4 u + F with Chladni sand nodal drift and pre-planned 2D FFT spectral decomposition.".into()
    }

    fn phonetic_description(&self) -> String {
        "Two dimensional Chladni plate biharmonic vibration and wave tank interference simulation"
            .into()
    }

    fn theory_citation(&self) -> String {
        "Chladni, Entdeckungen über die Theorie des Klanges (1787); Leissa, Vibration of Plates, NASA SP-160 (1969)".into()
    }

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "SquareChladniPlate".into(),
            "Kirchhoff-Love biharmonic square plate standing modes with nodal sand drift".into(),
        );
        map.insert(
            "CircularDrumhead".into(),
            "2D circular Dirichlet membrane Bessel-mode standing waves".into(),
        );
        map.insert(
            "DoubleSlitInterference".into(),
            "Young double-slit diffraction in an absorbing wave tank".into(),
        );
        map.insert(
            "AcousticLens".into(),
            "Biconvex refractive wave-speed lens focusing plane wavefronts".into(),
        );
        map
    }
}

const SQUARE_CHLADNI_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Square Chladni Plate",
    "domain": "analysis",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "preset": "SquareChladniPlate",
    "medium": "BiharmonicPlate",
    "boundary": "Neumann",
    "clamp_layout": "CenterPin",
    "colormap": "PerceptualThermal",
    "secondary_plot": "RadialFftSpectrum",
    "wave_speed": 0.0,
    "plate_stiffness": 0.0004,
    "damping": 0.08,
    "time_step": 0.002,
    "driver_enabled": true,
    "line_source": false,
    "driver_x": 0.22,
    "driver_y": 0.5,
    "driver_frequency": 0.56,
    "driver_amplitude": 65.0,
    "barrier_enabled": false,
    "slit_count": 2,
    "slit_width": 3,
    "slit_separation": 32.0,
    "barrier_x": 28,
    "detector_x": 96,
    "lens_enabled": false,
    "refractive_index": 1.6,
    "lens_aperture": 34.0,
    "show_nodal_contours": true,
    "show_sand_particles": true,
    "rng_seed": 3290092062
  }
}"#;

const CIRCULAR_DRUMHEAD_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Circular Drumhead",
    "domain": "analysis",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "preset": "CircularDrumhead",
    "medium": "AcousticWave",
    "boundary": "Dirichlet",
    "clamp_layout": "None",
    "colormap": "BipolarPhase",
    "secondary_plot": "RadialFftSpectrum",
    "wave_speed": 1.0,
    "plate_stiffness": 0.0,
    "damping": 0.03,
    "time_step": 0.004,
    "driver_enabled": true,
    "line_source": false,
    "driver_x": 0.34,
    "driver_y": 0.5,
    "driver_frequency": 2.98,
    "driver_amplitude": 42.0,
    "barrier_enabled": false,
    "slit_count": 2,
    "slit_width": 3,
    "slit_separation": 32.0,
    "barrier_x": 28,
    "detector_x": 96,
    "lens_enabled": false,
    "refractive_index": 1.6,
    "lens_aperture": 34.0,
    "show_nodal_contours": true,
    "show_sand_particles": true,
    "rng_seed": 3624582895
  }
}"#;

const DOUBLE_SLIT_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Double-Slit Interference",
    "domain": "analysis",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "preset": "DoubleSlitInterference",
    "medium": "AcousticWave",
    "boundary": "Sponge",
    "clamp_layout": "None",
    "colormap": "BipolarPhase",
    "secondary_plot": "DetectorCrossSection",
    "wave_speed": 1.0,
    "plate_stiffness": 0.0,
    "damping": 0.002,
    "time_step": 0.35,
    "driver_enabled": true,
    "line_source": true,
    "driver_x": 0.14173228346456693,
    "driver_y": 0.5,
    "driver_frequency": 0.1,
    "driver_amplitude": 3.2,
    "barrier_enabled": true,
    "slit_count": 2,
    "slit_width": 2,
    "slit_separation": 40.0,
    "barrier_x": 24,
    "detector_x": 96,
    "lens_enabled": false,
    "refractive_index": 1.6,
    "lens_aperture": 34.0,
    "show_nodal_contours": false,
    "show_sand_particles": false,
    "rng_seed": 1360461862
  }
}"#;

const ACOUSTIC_LENS_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Acoustic Lens",
    "domain": "analysis",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "preset": "AcousticLens",
    "medium": "AcousticWave",
    "boundary": "Sponge",
    "clamp_layout": "None",
    "colormap": "PerceptualThermal",
    "secondary_plot": "DetectorCrossSection",
    "wave_speed": 1.0,
    "plate_stiffness": 0.0,
    "damping": 0.002,
    "time_step": 0.4,
    "driver_enabled": true,
    "line_source": true,
    "driver_x": 0.12598425196850394,
    "driver_y": 0.5,
    "driver_frequency": 0.14,
    "driver_amplitude": 2.8,
    "barrier_enabled": false,
    "slit_count": 2,
    "slit_width": 3,
    "slit_separation": 32.0,
    "barrier_x": 28,
    "detector_x": 92,
    "lens_enabled": true,
    "refractive_index": 1.65,
    "lens_aperture": 38.0,
    "show_nodal_contours": true,
    "show_sand_particles": false,
    "rng_seed": 505802944
  }
}"#;

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "ChladniWaveTankTool",
        display_name: "Wave Tank & 2D Chladni Plate",
        domain: "analysis",
        tags: &["pde", "chladni", "waves", "diffraction", "fft"],
        build: || Box::new(ChladniWaveTankTool::default()),
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "chladni_square_plate_starter",
        title: "Square Chladni Plate",
        description: "Kirchhoff-Love biharmonic plate vibration with center nodal pin and 4,000 drifting sand grains.",
        category: "Partial Differential Equations",
        icon: "🔔",
        domain: "analysis",
        tool_name: Some("ChladniWaveTankTool"),
        preset_generator: || SQUARE_CHLADNI_PRESET_JSON,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "chladni_circular_drumhead_starter",
        title: "Circular Drumhead",
        description: "Bessel-mode circular membrane eigenmodes with Marching-Squares nodal contours and Chladni sand.",
        category: "Partial Differential Equations",
        icon: "🥁",
        domain: "analysis",
        tool_name: Some("ChladniWaveTankTool"),
        preset_generator: || CIRCULAR_DRUMHEAD_PRESET_JSON,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "wave_tank_double_slit_starter",
        title: "Double-Slit Interference",
        description: "2D Young double-slit wave tank with absorbing sponge perimeter and live detector fringe cross-section.",
        category: "Wave Optics & Acoustics",
        icon: "🌊",
        domain: "analysis",
        tool_name: Some("ChladniWaveTankTool"),
        preset_generator: || DOUBLE_SLIT_PRESET_JSON,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "wave_tank_acoustic_lens_starter",
        title: "Acoustic Lens",
        description: "Refractive biconvex acoustic wave lens focusing incident plane waves onto a downstream focal spot.",
        category: "Wave Optics & Acoustics",
        icon: "🔍",
        domain: "analysis",
        tool_name: Some("ChladniWaveTankTool"),
        preset_generator: || ACOUSTIC_LENS_PRESET_JSON,
    }
}
