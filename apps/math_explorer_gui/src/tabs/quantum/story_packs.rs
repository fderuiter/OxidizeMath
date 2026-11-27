#![cfg_attr(any(), verified(opt_out = "gui_tool"))]
//! Guided `PresetStoryMetadata` tours and `GoalChallengeMetadata` AST target cards for `QuantumTab`.
// [cite:quantum_mechanics]

#[cfg(all(
    feature = "ai",
    feature = "applied",
    feature = "biology",
    feature = "physics"
))]
use super::volumetric_studio::presets::{
    PRESET_1S_2PZ_DIPOLE_JSON, PRESET_3D_IMPLICIT_GYROID_AST_JSON,
};
use crate::framework::CalloutSpec;
use crate::starter_gallery::{GoalChallengeMetadata, PresetStoryMetadata, StoryStep};

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

const FABRY_OFF_RESONANCE_JSON: &str = r#"{"metadata":{"schema_version":"1.0","title":"Off-Resonance Double Barrier","domain":"quantum","timestamp":"2026-11-23T00:00:00Z"},"config":{"preset":"DoubleBarrier","x0":-2.5,"k0":4.2,"sigma":0.5,"v0":28.0,"width":0.8,"dt":0.02,"steps_per_frame":2,"custom_expr":"v0 * exp(-(x / width)^4)","atlas_visible":true,"atlas_axes":["k0","v0"]}}"#;

const FABRY_GROUND_RESONANCE_JSON: &str = r#"{"metadata":{"schema_version":"1.0","title":"Ground Fabry-Perot Resonance","domain":"quantum","timestamp":"2026-11-23T00:00:00Z"},"config":{"preset":"DoubleBarrier","x0":-2.5,"k0":2.898,"sigma":0.5,"v0":28.0,"width":0.8,"dt":0.02,"steps_per_frame":2,"custom_expr":"v0 * exp(-(x / width)^4)","atlas_visible":true,"atlas_axes":["k0","v0"]}}"#;

const FABRY_EXCITED_RESONANCE_JSON: &str = r#"{"metadata":{"schema_version":"1.0","title":"Excited Fabry-Perot Resonance","domain":"quantum","timestamp":"2026-11-23T00:00:00Z"},"config":{"preset":"DoubleBarrier","x0":-2.5,"k0":5.671,"sigma":0.5,"v0":28.0,"width":0.8,"dt":0.02,"steps_per_frame":2,"custom_expr":"v0 * exp(-(x / width)^4)","atlas_visible":true,"atlas_axes":["k0","v0"]}}"#;

#[cfg(all(
    feature = "ai",
    feature = "applied",
    feature = "biology",
    feature = "physics"
))]
const VOLUMETRIC_4F_XYZ_AST_JSON: &str = r#"{"metadata":{"schema_version":"1.0","title":"4f_xyz Octant Nodal Orbital","domain":"quantum","timestamp":"2026-11-23T00:00:00Z"},"config":{"source":"custom_implicit_ast","render_mode":"hybrid_shell_and_cloud","grid_resolution":32,"n1":1,"l1":0,"m1":0,"n2":2,"l2":1,"m2":0,"superposition_mix":0.5,"relative_phase":0.0,"quantum_dt":0.08,"observable":"signed_wavefunction","iso_threshold":0.045,"splat_scale":0.95,"turing_a":0.1,"turing_b":0.9,"turing_du":1.0,"turing_dv":18.0,"turing_dt":0.008,"custom_expr":"x * y * z * exp(-a * sqrt(x^2 + y^2 + z^2))","custom_params":{"a":1.45},"cut_enabled":true,"cut_axis":"z","cut_offset":0.35,"probe_brush_enabled":true,"running":false,"camera":{"pitch":0.48,"yaw":0.62,"zoom":1.06}}}"#;

const SPIN_AXIAL_BASELINE_JSON: &str = r#"{"metadata":{"schema_version":"1.0","title":"Axial Zeeman Baseline","domain":"quantum","timestamp":"2026-11-23T00:00:00Z"},"config":{"b_field":[0.0,0.0,1.0],"paused":true,"camera":{"pitch":0.5,"yaw":0.5,"zoom":1.0}}}"#;

const CLEBSCH_STRETCHED_TRIPLET_JSON: &str = r#"{"metadata":{"schema_version":"1.0","title":"Stretched Spin-1/2 Triplet","domain":"quantum","timestamp":"2026-11-23T00:00:00Z"},"config":{"j1":0.5,"m1":0.5,"j2":0.5,"m2":0.5,"j":1.0,"m":1.0}}"#;

const FABRY_PEROT_STEPS: &[StoryStep] = &[
    StoryStep {
        title: "Off-Resonance Sub-Barrier Reflection",
        narrative: "At k0 = 4.20 (E0 = 8.82 << V0 = 28.0), destructive phase interference inside the central quantum well suppresses steady-state transmission below 2%.",
        preset_generator: || FABRY_OFF_RESONANCE_JSON,
        callouts: &[pin(
            "fp_off",
            [0.0, 0.14],
            [16.0, -24.0],
            "Off-resonance gap: destructive cavity reflection",
        )],
        focus_observables: &[
            "kinetic_energy",
            "subbarrier_margin",
            "steady_transmission",
        ],
    },
    StoryStep {
        title: "First Quasi-Bound Fabry-Perot Peak",
        narrative: "Tuning k0 = 2.898 matches the ground quasi-bound eigenstate of the [-0.4, 0.4] cavity: reflected amplitudes cancel and steady lattice transmission surges past 99%.",
        preset_generator: || FABRY_GROUND_RESONANCE_JSON,
        callouts: &[pin(
            "fp_first",
            [0.0, 0.14],
            [16.0, -24.0],
            "k0 = 2.898: ground cavity resonance (T >= 0.99)",
        )],
        focus_observables: &[
            "kinetic_energy",
            "subbarrier_margin",
            "steady_transmission",
        ],
    },
    StoryStep {
        title: "Excited Fabry-Perot Resonance Band",
        narrative: "Stepping up to the second bright ridge on the 2D (k0, V0) atlas at k0 = 5.671 excites the antisymmetric well mode, restoring near-unity sub-barrier transmission.",
        preset_generator: || FABRY_EXCITED_RESONANCE_JSON,
        callouts: &[pin(
            "fp_second",
            [0.0, 0.14],
            [16.0, -24.0],
            "k0 = 5.671: second Fabry-Perot resonance band",
        )],
        focus_observables: &[
            "kinetic_energy",
            "subbarrier_margin",
            "steady_transmission",
        ],
    },
];

#[cfg(all(
    feature = "ai",
    feature = "applied",
    feature = "biology",
    feature = "physics"
))]
const VOLUMETRIC_MANIFOLD_STEPS: &[StoryStep] = &[
    StoryStep {
        title: "1s + 2pz Electric-Dipole Lobes",
        narrative: "Superposing hydrogenic 1s and 2pz eigenstates creates opposite-phase polar lobes split across the z = 0 nodal plane.",
        preset_generator: || PRESET_1S_2PZ_DIPOLE_JSON,
        callouts: &[pin(
            "vol_dipole",
            [0.0, 0.45],
            [18.0, -22.0],
            "Opposite-phase 1s + 2pz dipole shells",
        )],
        focus_observables: &["grid_resolution", "mesh_triangles", "active_splats"],
    },
    StoryStep {
        title: "Schoen Gyroid 3D Implicit AST",
        narrative: "CompiledExpr::eval_3d evaluates sin(kx)cos(ky) + sin(ky)cos(kz) + sin(kz)cos(kx) across [-2.5, 2.5]^3 with zero heap allocations.",
        preset_generator: || PRESET_3D_IMPLICIT_GYROID_AST_JSON,
        callouts: &[pin(
            "vol_gyroid",
            [0.0, 0.0],
            [18.0, -22.0],
            "Triply periodic Schoen Gyroid minimal surface",
        )],
        focus_observables: &["grid_resolution", "mesh_triangles", "slice_contours"],
    },
    StoryStep {
        title: "4f_xyz Octant Nodal Orbital",
        narrative: "Evaluating x*y*z*exp(-a*r) carves eight alternating-sign octant lobes separated by all three Cartesian coordinate planes.",
        preset_generator: || VOLUMETRIC_4F_XYZ_AST_JSON,
        callouts: &[pin(
            "vol_4fxyz",
            [0.35, 0.35],
            [18.0, -22.0],
            "Eight alternating 4f_xyz octant lobes",
        )],
        focus_observables: &["iso_threshold", "mesh_triangles", "slice_contours"],
    },
];

inventory::submit! {
    PresetStoryMetadata {
        id: "quantum_fabry_perot_resonance_tour",
        title: "Fabry-Perot Resonant Tunneling Tour",
        description: "Explore how quasi-bound central-well eigenstates turn an opaque double barrier transparent on the 2D (k0, V0) lattice atlas.",
        category: "Guided Story",
        icon: "📖",
        domain: "quantum",
        tool_name: "WaveSimulator",
        steps: FABRY_PEROT_STEPS,
    }
}

#[cfg(all(
    feature = "ai",
    feature = "applied",
    feature = "biology",
    feature = "physics"
))]
inventory::submit! {
    PresetStoryMetadata {
        id: "quantum_3d_nodal_manifolds_tour",
        title: "3D Nodal Orbitals & Implicit Manifolds",
        description: "Journey from hydrogenic dipole shells to zero-allocation 3D Schoen Gyroid and 4f_xyz AST isosurfaces.",
        category: "Guided Story",
        icon: "🧊",
        domain: "quantum",
        tool_name: "VolumetricFieldStudioTool",
        steps: VOLUMETRIC_MANIFOLD_STEPS,
    }
}

inventory::submit! {
    GoalChallengeMetadata {
        id: "quantum_zeeman_south_inversion",
        title: "Bloch Southern Hemisphere Inversion",
        description: "Apply a transverse Zeeman field to rotate |0> past the Bloch equator into the southern hemisphere.",
        category: "Challenge",
        icon: "🧭",
        domain: "quantum",
        tool_name: "SpinVisualizer",
        initial_preset_generator: || SPIN_AXIAL_BASELINE_JSON,
        goal_summary: "Set a transverse Bx or By field (keeping |B| > 0.5) and evolve or step Larmor precession until <sigma_z> drops below -0.5.",
        predicate_expr: "b_magnitude > 0.5 && expectation_z < -0.5",
        hint: "Set Bx = 2.0, Bz = 0.0 and click Resume or Step (+0.05s) until the spin vector rotates near the |-z> south pole.",
        win_message: "Challenge solved: Transverse Zeeman torque rotated the spinor into the southern Bloch hemisphere!",
        callouts: &[pin("spin_north", [0.0, 1.0], [18.0, -20.0], "Initial |0> north pole (<sigma_z> = +1)")],
        focus_observables: &["b_magnitude", "expectation_x", "expectation_z", "time"],
    }
}

inventory::submit! {
    GoalChallengeMetadata {
        id: "quantum_singlet_bell_coupling",
        title: "Spin-0 Singlet Bell Coupling",
        description: "Decompose two spin-1/2 particles into the antisymmetric J = 0, M = 0 singlet eigenstate with |C|^2 = 1/2.",
        category: "Challenge",
        icon: "⚛️",
        domain: "quantum",
        tool_name: "ClebschGordanTool",
        initial_preset_generator: || CLEBSCH_STRETCHED_TRIPLET_JSON,
        goal_summary: "Configure opposite spin projections for j1 = 1/2, j2 = 1/2 coupled into total J = 0, M = 0 so j_total < 0.1 and cg_probability > 0.49.",
        predicate_expr: "j_total < 0.1 && cg_probability > 0.49",
        hint: "Flip m2 from +0.5 to -0.5 and lower both J and M from 1.0 to 0.0 to select <1/2, +1/2; 1/2, -1/2 | 0, 0> = +1/sqrt(2).",
        win_message: "Challenge solved: Coupled two spin-1/2 states into the rotationally invariant J = 0 singlet with |C|^2 = 0.5!",
        callouts: &[pin("cg_bar", [0.0, 0.707], [18.0, -20.0], "Target singlet amplitude +/-1/sqrt(2)")],
        focus_observables: &["j_total", "m_total", "cg_amplitude", "cg_probability"],
    }
}
