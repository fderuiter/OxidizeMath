#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

/// Starter preset JSON envelope for Gray-Scott Coral morphogenesis.
pub const PRESET_GRAY_SCOTT_CORAL: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Turing Morphogenesis - Gray-Scott Coral",
    "domain": "biology",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "a": 0.1,
    "b": 0.9,
    "d_u": 0.16,
    "d_v": 0.08,
    "dt": 1.0,
    "width": 100,
    "height": 100,
    "kinetics_mode": "GrayScott",
    "f": 0.0545,
    "k": 0.062
  }
}"#;

/// Starter preset JSON envelope for Vogel Golden-Angle Sunflower phyllotaxis.
pub const PRESET_PHYLLOTAXIS_SUNFLOWER: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Phyllotaxis - Golden-Angle Sunflower",
    "domain": "biology",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "divergence_angle_deg": 137.50776405003785,
    "scale_c": 0.45,
    "primordia_count": 220,
    "substrate": 2.5,
    "v_max": 1.0,
    "k_m": 1.0,
    "hill_n": 2.5,
    "kinetics": "MichaelisMenten",
    "lsystem_preset": "BotanicalShoot",
    "lsystem_iterations": 4,
    "branch_angle_deg": 25.0
  }
}"#;

/// Starter preset JSON envelope for Cooperative Hill Nutrient L-System Shoot.
pub const PRESET_PHYLLOTAXIS_HILL_SHOOT: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Phyllotaxis - Cooperative Hill Shoot",
    "domain": "biology",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "divergence_angle_deg": 137.50776405003785,
    "scale_c": 0.48,
    "primordia_count": 144,
    "substrate": 3.2,
    "v_max": 1.2,
    "k_m": 1.0,
    "hill_n": 3.2,
    "kinetics": "Hill",
    "lsystem_preset": "SympodialTree",
    "lsystem_iterations": 4,
    "branch_angle_deg": 22.5
  }
}"#;

/// Starter preset JSON envelope for Hawk-Dove Replicator ESS & Hill Kinetics.
pub const PRESET_EVOLUTION_KINETICS_ESS: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Evolution & Enzyme Kinetics - Hawk-Dove ESS",
    "domain": "biology",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "resource_value": 2.0,
    "cost_of_injury": 4.0,
    "initial_hawk_freq": 0.15,
    "time_horizon": 25.0,
    "dt": 0.05,
    "v_max": 10.0,
    "k_m": 1.5,
    "hill_n": 3.0,
    "substrate_probe": 2.0
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "gray_scott_coral",
        title: "Gray-Scott Coral",
        description: "∂u/∂t = D_u∇²u - uv² + f(1-u): Cubic autocatalysis forming self-replicating coral labyrinths.",
        category: "Reaction-Diffusion",
        icon: "🪸",
        domain: "biology",
        tool_name: Some("MorphogenesisTool"),
        preset_generator: || PRESET_GRAY_SCOTT_CORAL,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "phyllotaxis_golden_sunflower",
        title: "Golden-Angle Sunflower",
        description: "r_k = c√(k+0.5), θ_k = k·137.508°: Vogel floret packing with consecutive Fibonacci parastichies.",
        category: "Phyllotaxis",
        icon: "🌻",
        domain: "biology",
        tool_name: Some("PhyllotaxisLSystemTool"),
        preset_generator: || PRESET_PHYLLOTAXIS_SUNFLOWER,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "phyllotaxis_hill_shoot",
        title: "Cooperative Hill Shoot",
        description: "Parallel L-system branching coupled to sigmoidal Hill enzyme nutrient uptake kinetics.",
        category: "L-Systems",
        icon: "🌿",
        domain: "biology",
        tool_name: Some("PhyllotaxisLSystemTool"),
        preset_generator: || PRESET_PHYLLOTAXIS_HILL_SHOOT,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "evolution_hawk_dove_ess",
        title: "Hawk-Dove & Hill Kinetics",
        description: "dp/dt = p(1-p)(W_H - W_D) & v = V_max[S]^n/(K_m^n + [S]^n): ESS replicator & enzyme saturation.",
        category: "Evolution & Kinetics",
        icon: "🦅",
        domain: "biology",
        tool_name: Some("EvolutionKineticsTool"),
        preset_generator: || PRESET_EVOLUTION_KINETICS_ESS,
    }
}
