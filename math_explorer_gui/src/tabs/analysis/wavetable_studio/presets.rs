#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

// [cite:graph_parameters_rust]

use std::collections::BTreeMap;

use math_explorer::pure_math::analysis::pde::wave_2d::{BoundaryMode2D, Wave2DMedium};

use super::{
    WavetableSourceMode, WavetableStudioState, WavetableStudioTool, AUDITION_FREQ_MAX_HZ,
    AUDITION_FREQ_MIN_HZ, DEFAULT_WAVETABLE_FORMULA, PARAM_ABS_LIMIT, PDE_ANGLE_MAX_DEG,
    PDE_ANGLE_MIN_DEG, PDE_SPEED_MAX, PDE_SPEED_MIN, PDE_STIFFNESS_MAX, PDE_STIFFNESS_MIN,
};
use crate::audio::wav::{WavSampleFormat, WavSampleRate};
use crate::audio::wavetable::MAX_WAVETABLE_FRAMES;

pub(super) fn default_studio_state() -> WavetableStudioState {
    WavetableStudioState {
        source_mode: WavetableSourceMode::FormulaExpr,
        formula: DEFAULT_WAVETABLE_FORMULA.to_string(),
        formula_params: BTreeMap::new(),
        frame_count: 64,
        morph_pos: 0.35,
        audition_freq_hz: 110.0,
        bandlimit_fundamental_hz: 110.0,
        sample_rate: WavSampleRate::Hz48000,
        sample_format: WavSampleFormat::Float32,
        pde_medium: Wave2DMedium::BiharmonicPlate,
        pde_boundary: BoundaryMode2D::Clamped,
        pde_stiffness: 0.0006,
        pde_wave_speed: 1.2,
        pde_slice_angle_deg: 22.5,
    }
}

pub(super) fn is_valid_studio_state(s: &WavetableStudioState) -> bool {
    let max_f = MAX_WAVETABLE_FRAMES as u16;
    !s.formula.trim().is_empty()
        && s.formula.len() <= 512
        && s.formula_params.len() <= 8
        && s.formula_params
            .values()
            .all(|v| v.is_finite() && v.abs() <= PARAM_ABS_LIMIT)
        && (1..=max_f).contains(&s.frame_count)
        && s.morph_pos.is_finite()
        && (0.0..=1.0).contains(&s.morph_pos)
        && s.audition_freq_hz.is_finite()
        && (AUDITION_FREQ_MIN_HZ..=AUDITION_FREQ_MAX_HZ).contains(&s.audition_freq_hz)
        && s.bandlimit_fundamental_hz.is_finite()
        && (AUDITION_FREQ_MIN_HZ..=AUDITION_FREQ_MAX_HZ).contains(&s.bandlimit_fundamental_hz)
        && s.pde_stiffness.is_finite()
        && (PDE_STIFFNESS_MIN..=PDE_STIFFNESS_MAX).contains(&s.pde_stiffness)
        && s.pde_wave_speed.is_finite()
        && (PDE_SPEED_MIN..=PDE_SPEED_MAX).contains(&s.pde_wave_speed)
        && s.pde_slice_angle_deg.is_finite()
        && (PDE_ANGLE_MIN_DEG..=PDE_ANGLE_MAX_DEG).contains(&s.pde_slice_angle_deg)
}

pub(super) fn clamp_studio_state(s: &mut WavetableStudioState) {
    if s.formula.trim().is_empty() || s.formula.len() > 512 {
        s.formula = DEFAULT_WAVETABLE_FORMULA.to_string();
    }
    for val in s.formula_params.values_mut() {
        *val = if val.is_finite() {
            val.clamp(-PARAM_ABS_LIMIT, PARAM_ABS_LIMIT)
        } else {
            1.0
        };
    }
    s.frame_count = s.frame_count.clamp(1, MAX_WAVETABLE_FRAMES as u16);
    s.morph_pos = clamp_f64(s.morph_pos, 0.0, 1.0, 0.35);
    s.audition_freq_hz = clamp_f64(
        s.audition_freq_hz,
        AUDITION_FREQ_MIN_HZ,
        AUDITION_FREQ_MAX_HZ,
        110.0,
    );
    s.bandlimit_fundamental_hz = clamp_f64(
        s.bandlimit_fundamental_hz,
        AUDITION_FREQ_MIN_HZ,
        AUDITION_FREQ_MAX_HZ,
        110.0,
    );
    s.pde_stiffness = clamp_f64(s.pde_stiffness, PDE_STIFFNESS_MIN, PDE_STIFFNESS_MAX, 6e-4);
    s.pde_wave_speed = clamp_f64(s.pde_wave_speed, PDE_SPEED_MIN, PDE_SPEED_MAX, 1.2);
    s.pde_slice_angle_deg = clamp_f64(
        s.pde_slice_angle_deg,
        PDE_ANGLE_MIN_DEG,
        PDE_ANGLE_MAX_DEG,
        22.5,
    );
}

fn clamp_f64(v: f64, lo: f64, hi: f64, fallback: f64) -> f64 {
    if v.is_finite() {
        v.clamp(lo, hi)
    } else {
        fallback
    }
}

impl scientific_metadata::theory::TheoryDescribable for WavetableStudioTool {
    fn theory_description(&self) -> String {
        "Synthesizes DC-free, Nyquist band-limited 2048-sample multi-frame wavetable banks via pre-planned FFT spectral conditioning of symbolic CompiledExpr formulas f(x, m) and 2D Kirchhoff-Love / hyperbolic wave PDE spatial cross-sections.".into()
    }

    fn phonetic_description(&self) -> String {
        "Band limited two thousand forty eight sample Fourier and partial differential equation wavetable synthesizer studio".into()
    }

    fn theory_citation(&self) -> String {
        "Bristow-Johnson, Wavetable Synthesis 101 (AES, 1996); Chowning, Synthesis of Complex Audio Spectra by Means of Frequency Modulation (JAES, 1973)".into()
    }

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "FormulaExpr".into(),
            "Symbolic CompiledExpr f(x, m) phase-modulation and Chebyshev formant morphing".into(),
        );
        map.insert(
            "PdeWaveSlice".into(),
            "2D Wave2DSolver bilinear cross-section chord swept across modal eigenfields".into(),
        );
        map
    }
}

const FM_FOLD_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "FM Harmonic Fold Wavetable",
    "domain": "analysis",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "source_mode": "FormulaExpr",
    "formula": "sin(pi * x + (1.0 + 5.0 * m) * sin(2.0 * pi * x)) + 0.35 * m * sin(5.0 * pi * x)",
    "frame_count": 64,
    "morph_pos": 0.35,
    "audition_freq_hz": 110.0,
    "bandlimit_fundamental_hz": 110.0,
    "sample_rate": "Hz48000",
    "sample_format": "Float32",
    "pde_medium": "BiharmonicPlate",
    "pde_boundary": "Clamped",
    "pde_stiffness": 0.0006,
    "pde_wave_speed": 1.2,
    "pde_slice_angle_deg": 22.5
  }
}"#;

const CHEBYSHEV_FORMANT_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Chebyshev Formant Morph Bank",
    "domain": "analysis",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "source_mode": "FormulaExpr",
    "formula": "sin(pi * x) * cos((1.0 + 7.0 * m) * pi * x) * exp(-a * x * x)",
    "formula_params": { "a": 1.4 },
    "frame_count": 128,
    "morph_pos": 0.5,
    "audition_freq_hz": 110.0,
    "bandlimit_fundamental_hz": 110.0,
    "sample_rate": "Hz48000",
    "sample_format": "Float32",
    "pde_medium": "BiharmonicPlate",
    "pde_boundary": "Clamped",
    "pde_stiffness": 0.0006,
    "pde_wave_speed": 1.2,
    "pde_slice_angle_deg": 22.5
  }
}"#;

const PDE_BIHARMONIC_SLICE_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Chladni Biharmonic Plate Slice Bank",
    "domain": "analysis",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "source_mode": "PdeWaveSlice",
    "formula": "sin(pi * x + (1.0 + 5.0 * m) * sin(2.0 * pi * x))",
    "frame_count": 64,
    "morph_pos": 0.42,
    "audition_freq_hz": 110.0,
    "bandlimit_fundamental_hz": 110.0,
    "sample_rate": "Hz48000",
    "sample_format": "Float32",
    "pde_medium": "BiharmonicPlate",
    "pde_boundary": "Clamped",
    "pde_stiffness": 0.0008,
    "pde_wave_speed": 1.2,
    "pde_slice_angle_deg": 30.0
  }
}"#;

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "WavetableStudioTool",
        display_name: "PDE & Formula Wavetable Studio",
        domain: "analysis",
        tags: &["wavetable", "fourier", "pde", "synthesizer", "wav"],
        build: || Box::new(WavetableStudioTool::default()),
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "wavetable_fm_harmonic_fold_starter",
        title: "FM Harmonic Fold Wavetable",
        description: "Chowning phase-modulation morph bank sweeping rich odd/even overtones across 64 band-limited 2048-sample frames.",
        category: "Fourier & Spectral Audio",
        icon: "🎛️",
        domain: "analysis",
        tool_name: Some("WavetableStudioTool"),
        preset_generator: || FM_FOLD_PRESET_JSON,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "wavetable_chebyshev_formant_starter",
        title: "Chebyshev Formant Morph Bank",
        description: "Gaussian-windowed vocal formant wavetable bank with interactive envelope parameter a and 128 morph frames.",
        category: "Fourier & Spectral Audio",
        icon: "🎼",
        domain: "analysis",
        tool_name: Some("WavetableStudioTool"),
        preset_generator: || CHEBYSHEV_FORMANT_PRESET_JSON,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "wavetable_pde_biharmonic_slice_starter",
        title: "Chladni Biharmonic Plate Slice Bank",
        description: "Captures 2048-sample bilinear cross-sections swept across a clamped 2D Kirchhoff-Love biharmonic plate.",
        category: "Partial Differential Equations",
        icon: "🪩",
        domain: "analysis",
        tool_name: Some("WavetableStudioTool"),
        preset_generator: || PDE_BIHARMONIC_SLICE_PRESET_JSON,
    }
}
