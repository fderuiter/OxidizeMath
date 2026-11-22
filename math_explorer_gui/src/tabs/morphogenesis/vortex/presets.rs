#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Academic theory metadata and federated Starter Gallery registrations for [`super::VortexMorphogenesisTool`].

use math_explorer::biology::morphogenesis::{GrayScottKinetics, SchnakenbergKinetics};
use math_explorer::physics::fluid_dynamics::lattice_boltzmann::{
    BgkCollision, LatticeBoltzmannD2Q9,
};
use scientific_metadata::theory::TheoryDescribable;
use std::collections::HashMap;

/// Composite [`TheoryDescribable`] metadata bridging D2Q9 BGK fluid mechanics and Turing/Gray-Scott kinetics.
#[derive(Clone, Debug, Default)]
pub struct VortexMorphogenesisTheory {
    gray_scott: GrayScottKinetics,
    schnakenberg: SchnakenbergKinetics,
}

impl TheoryDescribable for VortexMorphogenesisTheory {
    fn theory_description(&self) -> String {
        format!(
            "Coupled 2D Reaction-Advection-Diffusion Morphogenesis driven by a mesoscopic D2Q9 BGK \
             Lattice Boltzmann velocity field. {} {}",
            self.gray_scott.theory_description(),
            self.schnakenberg.theory_description()
        )
    }

    fn phonetic_description(&self) -> String {
        "Coupled Lattice Boltzmann vortex advection and two-species Turing reaction diffusion: \
         partial c sub i over partial t plus u dot grad c sub i equals D sub i Laplacian c sub i plus R sub i"
            .to_string()
    }

    fn theory_citation(&self) -> String {
        let lbm_cite = LatticeBoltzmannD2Q9::<BgkCollision>::new(1, 1, 1.0).theory_citation();
        format!(
            "{}; {}; {}",
            lbm_cite,
            self.gray_scott.theory_citation(),
            self.schnakenberg.theory_citation()
        )
    }

    fn available_descriptions(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert(
            "peclet_number".to_string(),
            "Dimensionless Peclet number Pe = U L / D_slow comparing fluid advection against morphogen diffusion."
                .to_string(),
        );
        map.insert(
            "damkohler_number".to_string(),
            "Dimensionless Damkohler number Da = k_react L / U comparing autocatalytic growth against convective flushing."
                .to_string(),
        );
        map.insert(
            "upwind_advection".to_string(),
            "First-order positivity-preserving 2D donor-cell upwind flux coupled with zero-flux Neumann obstacle reflection."
                .to_string(),
        );
        map.insert(
            "karman_vortex_trapping".to_string(),
            "Closed recirculation eddies shed behind bluff obstacles act as Lagrangian chemical micro-reactors."
                .to_string(),
        );
        map
    }

    fn theory_latex_equation(&self) -> Option<String> {
        Some(
            r"\frac{\partial c_i}{\partial t} + \mathbf{u}\cdot\nabla c_i = D_i \nabla^2 c_i + R_i(u, v), \quad \mathrm{Pe} = \frac{U L}{D_{\min}}, \quad \mathrm{Da} = \frac{k_{\mathrm{react}} L}{U}"
                .to_string(),
        )
    }
}

/// Starter preset JSON for **Karman Vortex Spots**.
pub const PRESET_VORTEX_KARMAN_SPOTS: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Karman Vortex Spots",
    "domain": "biology",
    "timestamp": "2026-11-22T00:00:00Z"
  },
  "config": {
    "advection_coupling": 0.45,
    "d_u": 0.16,
    "d_v": 0.06,
    "dt": 1.2,
    "dx": 1.0,
    "emitter_strength": 0.35,
    "fresh_inlet_feed": true,
    "gs_f": 0.038,
    "gs_k": 0.06,
    "height": 50,
    "inlet_ux": 0.1,
    "kinetics": "gray_scott",
    "paused": false,
    "preset": "karman_vortex_spots",
    "shear_profile": false,
    "sn_a": 0.1,
    "sn_b": 0.9,
    "tau": 0.56,
    "width": 100,
    "brush_mode": "obstacle",
    "brush_radius": 3,
    "display_mode": "coupled_wake"
  }
}"#;

/// Starter preset JSON for **Shear-Aligned Stripes**.
pub const PRESET_VORTEX_SHEAR_STRIPES: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Shear-Aligned Stripes",
    "domain": "biology",
    "timestamp": "2026-11-22T00:00:00Z"
  },
  "config": {
    "advection_coupling": 1.0,
    "d_u": 0.05,
    "d_v": 1.1,
    "dt": 3.0,
    "dx": 1.0,
    "emitter_strength": 0.25,
    "fresh_inlet_feed": false,
    "gs_f": 0.038,
    "gs_k": 0.06,
    "height": 50,
    "inlet_ux": 0.15,
    "kinetics": "schnakenberg",
    "paused": false,
    "preset": "shear_aligned_stripes",
    "shear_profile": true,
    "sn_a": 0.1,
    "sn_b": 0.9,
    "tau": 0.62,
    "width": 100,
    "brush_mode": "emitter",
    "brush_radius": 3,
    "display_mode": "morphogen_u"
  }
}"#;

/// Starter preset JSON for **Stagnation Plumes**.
pub const PRESET_VORTEX_STAGNATION_PLUMES: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Stagnation Plumes",
    "domain": "biology",
    "timestamp": "2026-11-22T00:00:00Z"
  },
  "config": {
    "advection_coupling": 0.75,
    "d_u": 0.16,
    "d_v": 0.06,
    "dt": 1.1,
    "dx": 1.0,
    "emitter_strength": 0.45,
    "fresh_inlet_feed": true,
    "gs_f": 0.036,
    "gs_k": 0.059,
    "height": 50,
    "inlet_ux": 0.09,
    "kinetics": "gray_scott",
    "paused": false,
    "preset": "stagnation_plumes",
    "shear_profile": false,
    "sn_a": 0.1,
    "sn_b": 0.9,
    "tau": 0.7,
    "width": 100,
    "brush_mode": "emitter",
    "brush_radius": 3,
    "display_mode": "coupled_wake"
  }
}"#;

/// Starter preset JSON for **Damkohler Washout**.
pub const PRESET_VORTEX_DAMKOHLER_WASHOUT: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Damkohler Washout",
    "domain": "biology",
    "timestamp": "2026-11-22T00:00:00Z"
  },
  "config": {
    "advection_coupling": 1.0,
    "d_u": 0.04,
    "d_v": 0.02,
    "dt": 2.6,
    "dx": 0.35,
    "emitter_strength": 0.0,
    "fresh_inlet_feed": true,
    "gs_f": 0.026,
    "gs_k": 0.06,
    "height": 50,
    "inlet_ux": 0.145,
    "kinetics": "gray_scott",
    "paused": false,
    "preset": "damkohler_washout",
    "shear_profile": false,
    "sn_a": 0.1,
    "sn_b": 0.9,
    "tau": 0.6,
    "width": 100,
    "brush_mode": "emitter",
    "brush_radius": 3,
    "display_mode": "coupled_wake"
  }
}"#;

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "VortexMorphogenesisTool",
        display_name: "Vortex Morphogenesis",
        domain: "biology",
        tags: &[
            "vortex",
            "lbm",
            "morphogenesis",
            "karman",
            "gray-scott",
            "schnakenberg",
            "peclet",
            "damkohler",
            "advection",
        ],
        build: || Box::new(super::VortexMorphogenesisTool::default()),
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "vortex_karman_spots",
        title: "Karman Vortex Spots",
        description: "∂c/∂t + u·∇c = D∇²c + R(u,v): Alternating Karman eddies trap autocatalytic Gray-Scott spots.",
        category: "Fluid-Morphogenesis",
        icon: "🌀",
        domain: "biology",
        tool_name: Some("VortexMorphogenesisTool"),
        preset_generator: || PRESET_VORTEX_KARMAN_SPOTS,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "vortex_shear_stripes",
        title: "Shear-Aligned Stripes",
        description: "Couette velocity shear damps streamwise gradients into flow-parallel Schnakenberg stripes.",
        category: "Fluid-Morphogenesis",
        icon: "🌊",
        domain: "biology",
        tool_name: Some("VortexMorphogenesisTool"),
        preset_generator: || PRESET_VORTEX_SHEAR_STRIPES,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "vortex_stagnation_plumes",
        title: "Stagnation Plumes",
        description: "Bluff-body wake recirculation zone pins a stationary autocatalytic chemical plume (Da > 1).",
        category: "Fluid-Morphogenesis",
        icon: "🧪",
        domain: "biology",
        tool_name: Some("VortexMorphogenesisTool"),
        preset_generator: || PRESET_VORTEX_STAGNATION_PLUMES,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "vortex_damkohler_washout",
        title: "Damkohler Washout",
        description: "High-speed flushing (Da < 1) sweeps autocatalysts past the extinction washout bifurcation.",
        category: "Fluid-Morphogenesis",
        icon: "💨",
        domain: "biology",
        tool_name: Some("VortexMorphogenesisTool"),
        preset_generator: || PRESET_VORTEX_DAMKOHLER_WASHOUT,
    }
}
