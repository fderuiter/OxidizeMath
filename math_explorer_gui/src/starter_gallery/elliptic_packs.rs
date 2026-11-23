//! Guided story tours (`PresetStoryMetadata`), AST target challenges (`GoalChallengeMetadata`),
//! and inverse locked-prime challenges (`ChallengeMetadata`) across Elliptic Curves & Number Theory.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{ChallengeMetadata, GoalChallengeMetadata, PresetStoryMetadata, StoryStep};
use crate::framework::{CalloutSpec, InteractiveTool};
use crate::tabs::elliptic_curves::chord_tangent_tool::{
    CurveFieldMode, WeierstrassPlaygroundState, WeierstrassPlaygroundTool,
};
use crate::tabs::elliptic_curves::CHALLENGE_HASSE_WEIL_BOUNDARY_P23;

const fn pin(
    tag: &'static str,
    target_point: [f64; 2],
    label_offset: [f32; 2],
    text: &'static str,
) -> CalloutSpec {
    CalloutSpec {
        tag,
        target_point,
        label_offset,
        text,
    }
}

const WEIERSTRASS_SMOOTH_TWO_COMP: &str = r#"{"metadata":{"schema_version":"1.0","title":"Two-Component Smooth Cubic","domain":"elliptic_curves","timestamp":"2026-11-23T00:00:00Z"},"config":{"real_a":-3.0,"real_b":1.0,"real_px":-1.5,"real_qx":2.0,"real_p_upper":true,"real_q_upper":true,"field_mode":"Real","prime_p":23,"mod_a":1,"mod_b":1,"selected_p_idx":0,"selected_q_idx":2,"scalar_k":4,"show_secant":true,"show_orbit":true,"atlas_visible":true,"atlas_axes":["real_a","real_b"]}}"#;

const WEIERSTRASS_CRUNODE: &str = r#"{"metadata":{"schema_version":"1.0","title":"Self-Intersecting Crunode (-3, 2)","domain":"elliptic_curves","timestamp":"2026-11-23T00:00:00Z"},"config":{"real_a":-3.0,"real_b":2.0,"real_px":-1.5,"real_qx":2.0,"real_p_upper":true,"real_q_upper":true,"field_mode":"Real","prime_p":23,"mod_a":-3,"mod_b":2,"selected_p_idx":0,"selected_q_idx":1,"scalar_k":3,"show_secant":true,"show_orbit":true,"atlas_visible":true,"atlas_axes":["real_a","real_b"]}}"#;

const WEIERSTRASS_ORIGIN_CUSP: &str = r#"{"metadata":{"schema_version":"1.0","title":"Triple-Root Origin Cusp (0, 0)","domain":"elliptic_curves","timestamp":"2026-11-23T00:00:00Z"},"config":{"real_a":0.0,"real_b":0.0,"real_px":1.0,"real_qx":2.0,"real_p_upper":true,"real_q_upper":true,"field_mode":"Real","prime_p":23,"mod_a":0,"mod_b":0,"selected_p_idx":0,"selected_q_idx":1,"scalar_k":3,"show_secant":true,"show_orbit":true,"atlas_visible":true,"atlas_axes":["real_a","real_b"]}}"#;

const WEIERSTRASS_ACNODE: &str = r#"{"metadata":{"schema_version":"1.0","title":"Isolated Acnode (-3, -2)","domain":"elliptic_curves","timestamp":"2026-11-23T00:00:00Z"},"config":{"real_a":-3.0,"real_b":-2.0,"real_px":2.4,"real_qx":3.2,"real_p_upper":true,"real_q_upper":true,"field_mode":"Real","prime_p":23,"mod_a":-3,"mod_b":-2,"selected_p_idx":0,"selected_q_idx":1,"scalar_k":3,"show_secant":true,"show_orbit":true,"atlas_visible":true,"atlas_axes":["real_a","real_b"]}}"#;

const WEIERSTRASS_SINGLE_BRANCH: &str = r#"{"metadata":{"schema_version":"1.0","title":"Single-Branch Baseline","domain":"elliptic_curves","timestamp":"2026-11-23T00:00:00Z"},"config":{"real_a":1.0,"real_b":1.0,"real_px":0.2,"real_qx":1.2,"real_p_upper":true,"real_q_upper":true,"field_mode":"Real","prime_p":23,"mod_a":1,"mod_b":1,"selected_p_idx":0,"selected_q_idx":2,"scalar_k":4,"show_secant":true,"show_orbit":true,"atlas_visible":true,"atlas_axes":["real_a","real_b"]}}"#;

const MODULAR_CM_J0: &str = r#"{"metadata":{"schema_version":"1.0","title":"Equianharmonic CM Ray J = 0","domain":"elliptic_curves","timestamp":"2026-11-23T00:00:00Z"},"config":{"coeff_a":0,"coeff_b":1,"max_prime":199,"inspect_prime":23,"level_n":7,"deg_i":2,"deg_j":1,"q_precision":12,"q_series_kind":"RamanujanDelta","view_mode":"HasseWeilTraces","atlas_visible":true,"atlas_axes":["mod_a","mod_b"]}}"#;

const MODULAR_CM_J1728: &str = r#"{"metadata":{"schema_version":"1.0","title":"Harmonic CM Ray J = 1728 & Breuer Bounds","domain":"elliptic_curves","timestamp":"2026-11-23T00:00:00Z"},"config":{"coeff_a":-1,"coeff_b":0,"max_prime":199,"inspect_prime":23,"level_n":7,"deg_i":2,"deg_j":1,"q_precision":12,"q_series_kind":"RamanujanDelta","view_mode":"BreuerValuationBounds","atlas_visible":true,"atlas_axes":["mod_a","mod_b"]}}"#;

const MODULAR_HURWITZ: &str = r#"{"metadata":{"schema_version":"1.0","title":"Hurwitz-Kronecker Class Number Sums","domain":"elliptic_curves","timestamp":"2026-11-23T00:00:00Z"},"config":{"coeff_a":1,"coeff_b":1,"max_prime":199,"inspect_prime":23,"level_n":7,"deg_i":2,"deg_j":1,"q_precision":12,"q_series_kind":"RamanujanDelta","view_mode":"HurwitzKroneckerSums","atlas_visible":false,"atlas_axes":["mod_a","mod_b"]}}"#;

const MODULAR_ORDINARY_BASELINE: &str = r#"{"metadata":{"schema_version":"1.0","title":"Ordinary Trace Baseline","domain":"elliptic_curves","timestamp":"2026-11-23T00:00:00Z"},"config":{"coeff_a":1,"coeff_b":1,"max_prime":199,"inspect_prime":23,"level_n":7,"deg_i":2,"deg_j":1,"q_precision":12,"q_series_kind":"RamanujanDelta","view_mode":"HasseWeilTraces","atlas_visible":true,"atlas_axes":["mod_a","mod_b"]}}"#;

const NT_TORUS_ORDINARY: &str = r#"{"metadata":{"schema_version":"1.0","title":"Ordinary Torus Subgroup over F_23","domain":"number_theory","timestamp":"2026-11-23T00:00:00Z"},"config":{"p":23,"a":1,"b":1,"point_p":{"Finite":{"x":0,"y":1}},"point_q":{"Finite":{"x":6,"y":4}},"scalar_k":5,"hasse_p_max":251,"atlas_visible":true,"atlas_axes":["mod_a","mod_b"]}}"#;

const NT_TORUS_SUPERSINGULAR: &str = r#"{"metadata":{"schema_version":"1.0","title":"Supersingular J = 0 Torus over F_23","domain":"number_theory","timestamp":"2026-11-23T00:00:00Z"},"config":{"p":23,"a":0,"b":1,"point_p":{"Finite":{"x":0,"y":1}},"point_q":{"Finite":{"x":2,"y":3}},"scalar_k":6,"hasse_p_max":251,"atlas_visible":true,"atlas_axes":["mod_a","mod_b"]}}"#;

const NT_TORUS_J1728: &str = r#"{"metadata":{"schema_version":"1.0","title":"Harmonic J = 1728 Two-Torsion over F_23","domain":"number_theory","timestamp":"2026-11-23T00:00:00Z"},"config":{"p":23,"a":1,"b":0,"point_p":{"Finite":{"x":1,"y":5}},"point_q":{"Finite":{"x":0,"y":0}},"scalar_k":4,"hasse_p_max":251,"atlas_visible":true,"atlas_axes":["mod_a","mod_b"]}}"#;

const CUSP_TOUR_STEPS: &[StoryStep] = &[
    StoryStep {
        title: "Two-Component Smooth Cubic (Δ > 0)",
        narrative: "Inside the left discriminant wedge 27b² < -4a³, x³ - 3x + 1 has three distinct real roots, splitting E(R) into a compact egg oval plus an unbounded branch.",
        preset_generator: || WEIERSTRASS_SMOOTH_TWO_COMP,
        callouts: &[pin("oval_branch", [-1.5, 1.46], [-18.0, -24.0], "Δ = +1296 > 0: compact oval + smooth branch")],
        focus_observables: &["discriminant", "real_components", "j_invariant", "is_singular"],
    },
    StoryStep {
        title: "Self-Intersecting Crunode at (1, 0)",
        narrative: "Sliding b up to +2 hits 4(-3)³ + 27(2)² = 0: the oval pinches the unbounded branch at double root (1, 0) with crossing tangents y = ±√3(x - 1) (split G_m).",
        preset_generator: || WEIERSTRASS_CRUNODE,
        callouts: &[pin("crunode_vertex", [1.0, 0.0], [18.0, -26.0], "Crunode (1, 0): slopes ±√3")],
        focus_observables: &["discriminant", "real_components", "is_singular"],
    },
    StoryStep {
        title: "Triple-Root Semicubical Cusp at (0, 0)",
        narrative: "At the apex (a, b) = (0, 0) where both CM rays J = 0 and J = 1728 collide, all three roots coalesce at (0, 0) on y² = x³ (additive smooth group G_a).",
        preset_generator: || WEIERSTRASS_ORIGIN_CUSP,
        callouts: &[pin("cusp_origin", [0.0, 0.0], [22.0, -26.0], "Origin Cusp (0, 0): y² = x³")],
        focus_observables: &["discriminant", "real_components", "is_singular"],
    },
    StoryStep {
        title: "Isolated Acnode (-1, 0) + Smooth Branch x ≥ 2",
        narrative: "On the lower cusp arm (a, b) = (-3, -2), y² = (x + 1)²(x - 2) isolates (-1, 0) with conjugate imaginary tangents ±i√3 while the continuous real branch begins at x = 2.",
        preset_generator: || WEIERSTRASS_ACNODE,
        callouts: &[
            pin("acnode_pt", [-1.0, 0.0], [-18.0, -26.0], "Isolated Acnode (-1, 0)"),
            pin("acnode_branch", [2.0, 0.0], [20.0, -24.0], "Smooth branch x ≥ 2"),
        ],
        focus_observables: &["discriminant", "real_components", "is_singular"],
    },
];

const MODULAR_TOUR_STEPS: &[StoryStep] = &[
    StoryStep {
        title: "Equianharmonic CM Ray (J = 0, a = 0)",
        narrative: "Curves y² = x³ + b carry complex multiplication by Z[ω]. For every prime p ≡ 2 (mod 3) such as p = 23, Frobenius trace vanishes (a_23 = 0, #E(F_23) = 24).",
        preset_generator: || MODULAR_CM_J0,
        callouts: &[pin("ss_p23", [23.0, 0.0], [18.0, -24.0], "Supersingular at p = 23: a_23 = 0")],
        focus_observables: &["j_invariant", "group_order", "frobenius_trace", "hasse_gap"],
    },
    StoryStep {
        title: "Harmonic CM Ray (J = 1728, b = 0) & Breuer Valuations",
        narrative: "Curves y² = x³ + ax have CM by Z[i] (J = 1728) and control 2-, 3-, 5-, and 7-adic divisibility lower bounds of classical modular polynomial coefficients Φ_N(X, Y).",
        preset_generator: || MODULAR_CM_J1728,
        callouts: &[pin("breuer_v", [2.0, 12.0], [16.0, -22.0], "Breuer p-adic valuation bounds")],
        focus_observables: &["j_invariant", "group_order", "frobenius_trace", "is_singular"],
    },
    StoryStep {
        title: "Hurwitz-Kronecker Class Number Trace Sum",
        narrative: "Summing imaginary quadratic Hurwitz class numbers H(t² - p) across every Frobenius trace |t| < 2√p recovers all isomorphism classes of elliptic curves over F_p.",
        preset_generator: || MODULAR_HURWITZ,
        callouts: &[pin("hurwitz_id", [23.0, 7.0], [16.0, -22.0], "Kronecker-Hurwitz trace sum")],
        focus_observables: &["group_order", "frobenius_trace", "hasse_gap"],
    },
];

const NT_TORUS_STEPS: &[StoryStep] = &[
    StoryStep {
        title: "Ordinary Cyclic Subgroup on F_23 × F_23",
        narrative: "On y² ≡ x³ + x + 1 (mod 23), #E(F_23) = 28 (a_23 = -4) inside Hasse bound |a_23| ≤ 9, and geodesic secants wrap periodically across the torus.",
        preset_generator: || NT_TORUS_ORDINARY,
        callouts: &[pin("hasse_p23", [23.0, -4.0], [16.0, -22.0], "#E(F_23) = 28, a_23 = -4")],
        focus_observables: &["group_order", "frobenius_trace", "point_order", "cofactor"],
    },
    StoryStep {
        title: "Supersingular Midline Symmetry (a_23 = 0)",
        narrative: "Switching to y² ≡ x³ + 1 (mod 23) lands squarely on the Hasse midline #E(F_23) = p + 1 = 24 with trace a_23 = 0.",
        preset_generator: || NT_TORUS_SUPERSINGULAR,
        callouts: &[pin("hasse_ss", [23.0, 0.0], [16.0, -22.0], "Supersingular midline a_23 = 0")],
        focus_observables: &["j_invariant", "group_order", "frobenius_trace", "hasse_gap"],
    },
    StoryStep {
        title: "Harmonic Two-Torsion & Cross-Tab Handoff",
        narrative: "On y² ≡ x³ + x (mod 23), Q = (0, 0) is a 2-torsion point with vertical tangent, and one click hands (p, a, b, P, Q, k) directly into Elliptic Curves.",
        preset_generator: || NT_TORUS_J1728,
        callouts: &[pin("hasse_j1728", [23.0, 0.0], [16.0, -22.0], "J = 1728 harmonic curve over F_23")],
        focus_observables: &["j_invariant", "point_order", "cofactor", "is_singular"],
    },
];

inventory::submit! {
    PresetStoryMetadata {
        id: "weierstrass_discriminant_cusp_tour",
        title: "Discriminant Cusp & Node Bifurcation Tour",
        description: "Walk the 2D (a, b) discriminant locus Δ = -16(4a³ + 27b²) = 0 across two-component cubics, crunodes, origin cusps, and isolated acnodes.",
        category: "Guided Story",
        icon: "🧭",
        domain: "elliptic_curves",
        tool_name: "WeierstrassPlaygroundTool",
        steps: CUSP_TOUR_STEPS,
    }
}

inventory::submit! {
    PresetStoryMetadata {
        id: "modular_cm_and_hurwitz_tour",
        title: "CM Rays, Supersingular Traces & Hurwitz Sums",
        description: "Explore J = 0 and J = 1728 complex-multiplication rays, supersingular Frobenius traces, Breuer bounds, and Hurwitz-Kronecker identities.",
        category: "Guided Story",
        icon: "📐",
        domain: "elliptic_curves",
        tool_name: "ModularInvariantsTool",
        steps: MODULAR_TOUR_STEPS,
    }
}

inventory::submit! {
    PresetStoryMetadata {
        id: "fp_torus_hasse_arch_tour",
        title: "Finite-Field Torus Orbits & Hasse-Weil Arch",
        description: "Trace wrapped F_23 × F_23 geodesics, supersingular midline symmetry, and Hasse-Weil trace arches.",
        category: "Guided Story",
        icon: "🍩",
        domain: "number_theory",
        tool_name: "EllipticCurvePlayground",
        steps: NT_TORUS_STEPS,
    }
}

inventory::submit! {
    GoalChallengeMetadata {
        id: "elliptic_pinch_crunode",
        title: "Pinch the Discriminant Cusp Locus",
        description: "Steer (a, b) onto the singular discriminant locus Δ = -16(4a³ + 27b²) = 0 to pinch the smooth curve into a cusp, crunode, or acnode.",
        category: "Challenge",
        icon: "🎯",
        domain: "elliptic_curves",
        tool_name: "WeierstrassPlaygroundTool",
        initial_preset_generator: || WEIERSTRASS_SMOOTH_TWO_COMP,
        goal_summary: "Click the gold Δ = 0 cusp contour in the 2D Parameter Atlas or tune (a, b) so Δ = 0 and is_singular = 1.",
        predicate_expr: "is_singular > 0.5 && discriminant >= -0.001 && discriminant <= 0.001",
        hint: "Try (a, b) = (-3, 2) for a crunode at (1, 0), (-3, -2) for an acnode at (-1, 0), or (0, 0) for the origin cusp.",
        win_message: "Challenge solved: Discriminant vanished (Δ = 0) and the Weierstrass cubic degenerated onto its singular locus!",
        callouts: &[pin("cusp_target", [1.0, 0.0], [18.0, -24.0], "Pinch oval & branch onto Δ = 0")],
        focus_observables: &["discriminant", "real_components", "is_singular"],
    }
}

inventory::submit! {
    GoalChallengeMetadata {
        id: "elliptic_two_component_cm",
        title: "Two-Component Harmonic CM Curve (J = 1728)",
        description: "Configure a non-singular real curve with two connected real components (Δ > 0) and complex multiplication invariant J = 1728.",
        category: "Challenge",
        icon: "✨",
        domain: "elliptic_curves",
        tool_name: "WeierstrassPlaygroundTool",
        initial_preset_generator: || WEIERSTRASS_SINGLE_BRANCH,
        goal_summary: "Reach a smooth curve (is_singular = 0) with 2 real components and j_invariant = 1728.",
        predicate_expr: "is_singular < 0.5 && real_components > 1.5 && j_invariant >= 1727.5 && j_invariant <= 1728.5",
        hint: "Set b = 0 so J = 1728, and choose a < 0 (e.g. a = -2, b = 0) so x³ + ax factors into three real roots.",
        win_message: "Challenge solved: y² = x³ + ax with a < 0 has three real 2-torsion roots (Δ > 0) and harmonic CM invariant J = 1728!",
        callouts: &[pin("cm_target", [0.0, 0.0], [18.0, -24.0], "Target b = 0 with a < 0")],
        focus_observables: &["discriminant", "real_components", "j_invariant", "is_singular"],
    }
}

inventory::submit! {
    GoalChallengeMetadata {
        id: "modular_supersingular_j0",
        title: "Zero Frobenius Trace on the J = 0 CM Ray",
        description: "Find a smooth curve at p = 23 whose Frobenius trace a_p vanishes on the equianharmonic CM ray J = 0.",
        category: "Challenge",
        icon: "🧭",
        domain: "elliptic_curves",
        tool_name: "ModularInvariantsTool",
        initial_preset_generator: || MODULAR_ORDINARY_BASELINE,
        goal_summary: "Adjust integer coefficients (a, b) at inspect_prime = 23 so is_singular = 0, j_invariant = 0, and frobenius_trace = 0.",
        predicate_expr: "is_singular < 0.5 && frobenius_trace >= -0.001 && frobenius_trace <= 0.001 && j_invariant >= -0.001 && j_invariant <= 0.001",
        hint: "Set a = 0 with non-zero b mod 23 (e.g. a = 0, b = 1); since 23 ≡ 2 (mod 3), every y² = x³ + b is supersingular over F_23.",
        win_message: "Challenge solved: Equianharmonic CM curve y² = x³ + b is supersingular at p = 23 with #E(F_23) = 24 and a_23 = 0!",
        callouts: &[pin("ss_goal", [23.0, 0.0], [18.0, -24.0], "Drive a_23 to 0 on J = 0")],
        focus_observables: &["j_invariant", "group_order", "frobenius_trace", "is_singular"],
    }
}

inventory::submit! {
    GoalChallengeMetadata {
        id: "torus_prime_order_generator",
        title: "Prime-Order Cryptographic Torus Subgroup",
        description: "Find a smooth curve over F_p and generator point P with cofactor 1 and prime subgroup order at least 29.",
        category: "Challenge",
        icon: "🔐",
        domain: "number_theory",
        tool_name: "EllipticCurvePlayground",
        initial_preset_generator: || NT_TORUS_ORDINARY,
        goal_summary: "Tune (p, a, b) and select P on E(F_p) so is_singular = 0, cofactor = 1, and point_order ≥ 29.",
        predicate_expr: "is_singular < 0.5 && cofactor <= 1.001 && point_order >= 29.0",
        hint: "At p = 23, try (a, b) = (-1, 1) where #E(F_23) = 29 (prime!) or (a, b) = (4, 2) where #E(F_23) = 31, and select any finite point P.",
        win_message: "Challenge solved: Cofactor-1 prime-order cyclic torus group constructed!",
        callouts: &[pin("prime_ord_goal", [23.0, -6.0], [16.0, -22.0], "Reach prime #E ≥ 29 with cofactor = 1")],
        focus_observables: &["group_order", "point_order", "cofactor", "is_singular"],
    }
}

fn generate_hasse_weil_p23_baseline() -> String {
    let s = WeierstrassPlaygroundState {
        field_mode: CurveFieldMode::FiniteField,
        prime_p: 23,
        mod_a: 1,
        mod_b: 1,
        atlas_visible: true,
        atlas_axes: ["mod_a".to_string(), "mod_b".to_string()],
        ..WeierstrassPlaygroundState::default()
    };
    serde_json::to_string(&s).unwrap_or_default()
}

fn clamp_hasse_weil_p23_state(raw: &str) -> Result<String, String> {
    let mut tool = WeierstrassPlaygroundTool::default();
    tool.state_bounds().validate_json(raw)?;
    let mut parsed: WeierstrassPlaygroundState = serde_json::from_str(raw)
        .map_err(|e| format!("Invalid Hasse-Weil challenge state JSON: {e}"))?;
    parsed.field_mode = CurveFieldMode::FiniteField;
    parsed.prime_p = 23;
    parsed.mod_a = parsed.mod_a.clamp(-997, 997);
    parsed.mod_b = parsed.mod_b.clamp(-997, 997);
    parsed.scalar_k = parsed.scalar_k.clamp(1, 256);
    parsed.real_a = parsed.real_a.clamp(-8.0, 8.0);
    parsed.real_b = parsed.real_b.clamp(-8.0, 8.0);
    parsed.real_px = parsed.real_px.clamp(-10.0, 10.0);
    parsed.real_qx = parsed.real_qx.clamp(-10.0, 10.0);
    let candidate = serde_json::to_string(&parsed)
        .map_err(|e| format!("Challenge state serialize error: {e}"))?;
    tool.apply_challenge(Some(CHALLENGE_HASSE_WEIL_BOUNDARY_P23))?;
    tool.try_load_state(&candidate)?;
    tool.save_state()
        .ok_or_else(|| "Failed to serialize clamped challenge state".to_string())
}

inventory::submit! {
    ChallengeMetadata {
        id: CHALLENGE_HASSE_WEIL_BOUNDARY_P23,
        title: "Extremal Hasse-Weil Boundary over F_23",
        description: "With prime modulus locked at p = 23 (Hasse bound ⌊2√23⌋ = 9), find non-singular integers (a, b) maximizing or minimizing #E(F_23) so |a_23| = 9 (#E = 15 or 33).",
        category: "Arithmetic Geometry",
        icon: "🏆",
        domain: "elliptic_curves",
        tool_name: "WeierstrassPlaygroundTool",
        locked_inputs: "Field = F_23 (prime_p = 23 locked), Hasse bound floor(2*sqrt(23)) = 9",
        tunable_budget: "Integer Weierstrass coefficients (a, b) in [-15, 15] (or click 2D #E(F_23) Atlas)",
        target_summary: "Non-singular curve with hasse_gap == 0 (|a_23| = 9, #E(F_23) in {15, 33})",
        baseline_state_generator: generate_hasse_weil_p23_baseline,
        clamp_locked_state: clamp_hasse_weil_p23_state,
    }
}
