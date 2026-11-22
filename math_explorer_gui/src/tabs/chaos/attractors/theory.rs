#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

impl scientific_metadata::theory::TheoryDescribable for super::AttractorPlotter {
    fn theory_description(&self) -> String {
        format!(
            "Lorenz attractor dynamical system: $\\dot{{x}} = \\sigma(y - x)$, $\\dot{{y}} = x(\\rho - z) - y$, $\\dot{{z}} = xy - \\beta z$ with parameters $\\sigma = {:.2}$, $\\rho = {:.2}$, $\\beta = {:.2}$.",
            self.system.sigma, self.system.rho, self.system.beta
        )
    }
    fn phonetic_description(&self) -> String {
        "Lorenz strange attractor dynamical system theory".into()
    }
    fn theory_citation(&self) -> String {
        "Deterministic Nonperiodic Flow (Lorenz, 1963); Nonlinear Dynamics and Chaos (Strogatz, 2014)".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        self.system.available_descriptions()
    }
    fn theory_parameters(
        &self,
    ) -> std::collections::HashMap<String, scientific_metadata::theory::ParameterConstraint> {
        use scientific_metadata::theory::ParameterConstraint;
        std::collections::HashMap::from([
            (
                "sigma".to_string(),
                ParameterConstraint {
                    min: 0.0,
                    max: 50.0,
                    step: 0.1,
                },
            ),
            (
                "rho".to_string(),
                ParameterConstraint {
                    min: 0.0,
                    max: 100.0,
                    step: 0.5,
                },
            ),
            (
                "beta".to_string(),
                ParameterConstraint {
                    min: 0.0,
                    max: 10.0,
                    step: 0.1,
                },
            ),
            (
                "dt".to_string(),
                ParameterConstraint {
                    min: 0.001,
                    max: 0.05,
                    step: 0.001,
                },
            ),
        ])
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "3d_attractor",
        title: "3D Attractor",
        description: "ẋ=σ(y-x), ẏ=x(ρ-z)-y, ż=xy-βz: Canonical 3D chaotic strange attractor trajectory.",
        category: "Chaos Theory",
        icon: "🦋",
        domain: "chaos",
        tool_name: Some("AttractorPlotter"),
        preset_generator: || crate::starter_gallery::PRESET_3D_ATTRACTOR,
    }
}
