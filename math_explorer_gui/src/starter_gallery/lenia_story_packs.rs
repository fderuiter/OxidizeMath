//! Built-in guided story tour and AST challenge for Lenia soliton viability and Geminium fission.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::pin;
use crate::starter_gallery::{GoalChallengeMetadata, PresetStoryMetadata, StoryStep};

const LENIA_STEP_ORBIUM: &str = r#"{"metadata":{"schema_version":"1.0","title":"Orbium Soliton Niche","domain":"biology","timestamp":"2026-11-23T00:00:00Z"},"config":{"radius":13.0,"mu":0.15,"sigma":0.015,"dt":0.1,"ring_weights":[1.0],"grid_size":128,"species":"Orbium","rng_seed":42,"perturbation":0.0,"brush_radius":4.0,"brush_mode":"Deposit","playing":true}}"#;
const LENIA_STEP_STARVE: &str = r#"{"metadata":{"schema_version":"1.0","title":"Orbium Niche Starvation","domain":"biology","timestamp":"2026-11-23T00:00:00Z"},"config":{"radius":13.0,"mu":0.28,"sigma":0.015,"dt":0.1,"ring_weights":[1.0],"grid_size":128,"species":"Orbium","rng_seed":42,"perturbation":0.0,"brush_radius":4.0,"brush_mode":"Deposit","playing":true}}"#;
const LENIA_STEP_GEMINIUM: &str = r#"{"metadata":{"schema_version":"1.0","title":"Multi-Ring Geminium Fission","domain":"biology","timestamp":"2026-11-23T00:00:00Z"},"config":{"radius":16.0,"mu":0.24,"sigma":0.036,"dt":0.1,"ring_weights":[1.0,0.55,0.75],"grid_size":128,"species":"Geminium","rng_seed":42,"perturbation":0.01,"brush_radius":4.0,"brush_mode":"Deposit","playing":true}}"#;
const LENIA_GEMINIUM_INIT: &str = r#"{"metadata":{"schema_version":"1.0","title":"Subcritical Geminium Twin","domain":"biology","timestamp":"2026-11-23T00:00:00Z"},"config":{"radius":16.0,"mu":0.34,"sigma":0.020,"dt":0.1,"ring_weights":[1.0,0.55,0.75],"grid_size":128,"species":"Geminium","rng_seed":42,"perturbation":0.01,"brush_radius":4.0,"brush_mode":"Deposit","playing":true}}"#;

const LENIA_VIABILITY_STEPS: &[StoryStep] = &[
    StoryStep {
        title: "Orbium Potential Niche Alignment",
        narrative: "Orbium's ring convolution U = A * K peaks near max(U) ≈ 0.225, straddling the positive Gaussian growth peak at μ = 0.15 (σ = 0.015) while empty space decays at G(0) = -1.",
        preset_generator: || LENIA_STEP_ORBIUM,
        callouts: &[pin(
            "lenia_niche",
            [0.15, 1.0],
            [18.0, -24.0],
            "Peak G(μ = 0.15) = +1 inside max(U) ≈ 0.225",
        )],
        focus_observables: &["mu", "sigma", "max_potential", "total_mass"],
    },
    StoryStep {
        title: "Starvation Above Peak Potential",
        narrative: "Shifting μ = 0.28 above max(U) ≈ 0.225 leaves every active cell in the negative decay zone G(U) < 0, collapsing the soliton to Extinction.",
        preset_generator: || LENIA_STEP_STARVE,
        callouts: &[pin(
            "lenia_niche",
            [0.28, 1.0],
            [-28.0, -24.0],
            "μ = 0.28 > max(U): soliton starves",
        )],
        focus_observables: &["mu", "max_potential", "total_mass"],
    },
    StoryStep {
        title: "Three-Ring Geminium Lobe Fission",
        narrative: "Expanding to three concentric rings [1.0, 0.55, 0.75] (R = 16) with wider σ = 0.036 at μ = 0.24 sustains twin-lobed Geminium budding and self-division.",
        preset_generator: || LENIA_STEP_GEMINIUM,
        callouts: &[pin(
            "lenia_niche",
            [0.24, 1.0],
            [14.0, -24.0],
            "3-ring shell drives twin-lobe fission",
        )],
        focus_observables: &["radius", "mu", "sigma", "total_mass"],
    },
];

inventory::submit! {
    PresetStoryMetadata {
        id: "lenia_soliton_viability_tour",
        title: "Lenia Soliton Viability & Fission Tour",
        description: "Explore how neighborhood potential U = A * K aligns with Growth G(u) to sustain Orbium gliding, starvation, and Geminium fission.",
        category: "Guided Story",
        icon: "🛸",
        domain: "biology",
        tool_name: "LeniaTool",
        steps: LENIA_VIABILITY_STEPS,
    }
}

inventory::submit! {
    GoalChallengeMetadata {
        id: "lenia_geminium_fission",
        title: "Geminium Twin-Lobe Fission",
        description: "Align the 3-ring Geminium growth niche (μ, σ) with U = A * K so the twin-lobed organism expands and divides without exploding vacuum.",
        category: "Challenge",
        icon: "🧬",
        domain: "biology",
        tool_name: "LeniaTool",
        initial_preset_generator: || LENIA_GEMINIUM_INIT,
        goal_summary: "Lower Growth μ near 0.24 and widen Width σ near 0.036 so Geminium undergoes lobe expansion/fission (220 < total_mass < 700, active_area > 340, centroid_speed > 0.04).",
        predicate_expr: "total_mass > 220.0 && total_mass < 700.0 && active_area > 340.0 && centroid_speed > 0.04",
        hint: "Drag Growth μ down from 0.34 to ~0.24 and Width σ up from 0.020 to ~0.036 (or open the 2D Parameter Atlas and click Stable Soliton Niche).",
        win_message: "Challenge solved: Geminium's 3-ring neighborhood potential entered the viable growth window and triggered twin-lobe fission!",
        callouts: &[pin("geminium_fission_niche", [0.24, 1.0], [16.0, -24.0], "Target fission niche (μ ≈ 0.24, σ ≈ 0.036)")],
        focus_observables: &["mu", "sigma", "max_potential", "total_mass", "active_area"],
    }
}
