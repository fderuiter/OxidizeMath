#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{
    ElementType, PotentialFlowTool, SerializedFlowElement, FLOW_ANGLE_MAX, FLOW_ANGLE_MIN,
    FLOW_POS_MAX, FLOW_POS_MIN, FLOW_STRENGTH_MAX, FLOW_STRENGTH_MIN, FLOW_VIEW_MAX, FLOW_VIEW_MIN,
    MAX_CONTOUR_LEVELS, MAX_DYE_PROBES,
};

const fn default_element_strength() -> f64 {
    10.0
}
const fn default_true() -> bool {
    true
}
const fn default_contour_levels() -> usize {
    17
}
const fn default_contour_resolution() -> usize {
    48
}
const fn default_dye_rake_x() -> f64 {
    -8.0
}
const fn default_dye_rake_span() -> f64 {
    6.5
}
const fn default_dye_count() -> usize {
    9
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct PotentialFlowState {
    elements: Vec<SerializedFlowElement>,
    grid_size: usize,
    view_range: f64,
    #[serde(default)]
    selected_element_type: ElementType,
    #[serde(default = "default_element_strength")]
    new_element_strength: f64,
    #[serde(default)]
    new_element_x: f64,
    #[serde(default)]
    new_element_y: f64,
    #[serde(default)]
    new_element_angle: f64,
    #[serde(default = "default_true")]
    show_vectors: bool,
    #[serde(default = "default_true")]
    show_streamlines: bool,
    #[serde(default = "default_true")]
    show_dye: bool,
    #[serde(default = "default_contour_levels")]
    contour_levels: usize,
    #[serde(default = "default_contour_resolution")]
    contour_resolution: usize,
    #[serde(default = "default_dye_rake_x")]
    dye_rake_x: f64,
    #[serde(default)]
    dye_rake_y: f64,
    #[serde(default = "default_dye_rake_span")]
    dye_rake_span: f64,
    #[serde(default = "default_dye_count")]
    dye_count: usize,
    #[serde(default)]
    stamp_mode: bool,
}

impl PotentialFlowTool {
    pub(super) fn parameters_are_valid(&self) -> bool {
        (2..=math_commons::registry::MAX_GRID_SIZE).contains(&self.grid_size)
            && self.view_range.is_finite()
            && (FLOW_VIEW_MIN..=FLOW_VIEW_MAX).contains(&self.view_range)
            && self.new_element_strength.is_finite()
            && (FLOW_STRENGTH_MIN..=FLOW_STRENGTH_MAX).contains(&self.new_element_strength)
            && self.new_element_x.is_finite()
            && (FLOW_POS_MIN..=FLOW_POS_MAX).contains(&self.new_element_x)
            && self.new_element_y.is_finite()
            && (FLOW_POS_MIN..=FLOW_POS_MAX).contains(&self.new_element_y)
            && self.new_element_angle.is_finite()
            && (FLOW_ANGLE_MIN..=FLOW_ANGLE_MAX).contains(&self.new_element_angle)
            && (2..=MAX_CONTOUR_LEVELS).contains(&self.contour_levels)
            && (4..=math_commons::registry::MAX_GRID_SIZE).contains(&self.contour_resolution)
            && self.dye_rake_x.is_finite()
            && (FLOW_POS_MIN..=FLOW_POS_MAX).contains(&self.dye_rake_x)
            && self.dye_rake_y.is_finite()
            && (FLOW_POS_MIN..=FLOW_POS_MAX).contains(&self.dye_rake_y)
            && self.dye_rake_span.is_finite()
            && (0.0..=FLOW_VIEW_MAX * 2.0).contains(&self.dye_rake_span)
            && (1..=MAX_DYE_PROBES).contains(&self.dye_count)
            && self.elements.len() <= crate::framework::MAX_JSON_ARRAY_LEN
            && self.elements.iter().all(SerializedFlowElement::is_valid)
    }

    pub(super) fn save_preset_state(&self) -> Option<String> {
        serde_json::to_string(&PotentialFlowState {
            elements: self.elements.clone(),
            grid_size: self.grid_size,
            view_range: self.view_range,
            selected_element_type: self.selected_element_type,
            new_element_strength: self.new_element_strength,
            new_element_x: self.new_element_x,
            new_element_y: self.new_element_y,
            new_element_angle: self.new_element_angle,
            show_vectors: self.show_vectors,
            show_streamlines: self.show_streamlines,
            show_dye: self.show_dye,
            contour_levels: self.contour_levels,
            contour_resolution: self.contour_resolution,
            dye_rake_x: self.dye_rake_x,
            dye_rake_y: self.dye_rake_y,
            dye_rake_span: self.dye_rake_span,
            dye_count: self.dye_count,
            stamp_mode: self.stamp_mode,
        })
        .ok()
    }

    pub(super) fn try_load_preset_state(&mut self, state: &str) -> Result<(), String> {
        let loaded: PotentialFlowState = serde_json::from_str(state)
            .map_err(|e| format!("Invalid PotentialFlowTool state: {e}"))?;
        let mut candidate = Self {
            elements: loaded.elements,
            grid_size: loaded.grid_size,
            view_range: loaded.view_range,
            selected_element_type: loaded.selected_element_type,
            new_element_strength: loaded.new_element_strength,
            new_element_x: loaded.new_element_x,
            new_element_y: loaded.new_element_y,
            new_element_angle: loaded.new_element_angle,
            show_vectors: loaded.show_vectors,
            show_streamlines: loaded.show_streamlines,
            show_dye: loaded.show_dye,
            contour_levels: loaded.contour_levels,
            contour_resolution: loaded.contour_resolution,
            dye_rake_x: loaded.dye_rake_x,
            dye_rake_y: loaded.dye_rake_y,
            dye_rake_span: loaded.dye_rake_span,
            dye_count: loaded.dye_count,
            stamp_mode: loaded.stamp_mode,
            ..Default::default()
        };
        if !candidate.parameters_are_valid() {
            return Err(
                "PotentialFlowTool parameters must be finite and within valid grid/view bounds"
                    .to_string(),
            );
        }
        candidate.rebuild_field();
        *self = candidate;
        Ok(())
    }

    pub(super) fn collect_export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let mut out = Vec::new();
        if self.show_vectors {
            let (origins, tips) = self.compute_arrow_field();
            let mut pts = Vec::with_capacity(origins.len() * 2);
            for (o, t) in origins.into_iter().zip(tips) {
                pts.push(o);
                pts.push(t);
            }
            if !pts.is_empty() {
                out.push(("Vector Field".to_string(), pts));
            }
        }
        if self.show_streamlines {
            let r = self.view_range.max(0.5);
            let segs = self.field.extract_streamline_segments(
                (-r, r, -r, r),
                self.contour_resolution,
                &self.compute_iso_levels(),
            );
            let pts: Vec<[f64; 2]> = segs.into_iter().flat_map(|(a, b)| [a, b]).collect();
            if !pts.is_empty() {
                out.push(("Iso-ψ Streamlines".to_string(), pts));
            }
        }
        if self.show_dye {
            for (i, probe) in self.dye_probes.iter().enumerate() {
                if !probe.trail.is_empty() {
                    out.push((format!("Dye Streak {}", i + 1), probe.trail.clone()));
                }
            }
        }
        out
    }
}

impl scientific_metadata::theory::TheoryDescribable for PotentialFlowTool {
    fn theory_description(&self) -> String {
        "Two-dimensional incompressible irrotational potential flow governed by Laplace's equation $\\nabla^2 \\phi = 0$ and $\\nabla^2 \\psi = 0$, where streamlines satisfy $\\psi(x, y) = \\text{const}$, fluid velocity is $\\vec{V} = \\nabla \\phi = (\\partial_y \\psi, -\\partial_x \\psi)$, and inviscid pressure obeys Bernoulli's theorem.".into()
    }

    fn phonetic_description(&self) -> String {
        "Incompressible irrotational potential flow, Cauchy Riemann stream functions, and Kutta Joukowski lift".into()
    }

    fn theory_citation(&self) -> String {
        "Elementary Fluid Dynamics (Acheson, 1990); An Introduction to Fluid Dynamics (Batchelor, 1967)".into()
    }

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "Uniform".into(),
            "Uniform freestream flow phi = U (x cos(alpha) + y sin(alpha))".into(),
        );
        map.insert(
            "Source/Sink".into(),
            "Radial flow phi = (m / 2 pi) ln(r), psi = (m / 2 pi) theta".into(),
        );
        map.insert(
            "Vortex".into(),
            "Circulation flow psi = (Gamma / 2 pi) ln(r), Kutta-Joukowski lift L' = -rho U Gamma"
                .into(),
        );
        map.insert(
            "Doublet".into(),
            "Source-sink dipole limit psi = -(kappa y) / (2 pi r^2)".into(),
        );
        map
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "vector_field",
        title: "Vector Field",
        description: "ψ = U·y·(1 - R²/r²): Potential streamlines and 2D velocity vector field past a cylinder.",
        category: "Fluid Mechanics",
        icon: "🌊",
        domain: "fluid_dynamics",
        tool_name: Some("PotentialFlowTool"),
        preset_generator: || crate::starter_gallery::PRESET_VECTOR_FIELD,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "source_sink",
        title: "Source-Sink Pair",
        description: "ψ = (q/2π)·(θ₁ - θ₂): Superposition of equal source and sink singularities.",
        category: "Fluid Mechanics",
        icon: "🔄",
        domain: "fluid_dynamics",
        tool_name: Some("PotentialFlowTool"),
        preset_generator: || crate::starter_gallery::PRESET_SOURCE_SINK,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "lifting_cylinder",
        title: "Kutta-Joukowski Lifting Cylinder",
        description: "ψ = U·y·(1 - R²/r²) - (Γ/2π)·ln(r): Spinning cylinder circulation shifting stagnation points.",
        category: "Fluid Mechanics",
        icon: "✈",
        domain: "fluid_dynamics",
        tool_name: Some("PotentialFlowTool"),
        preset_generator: || crate::starter_gallery::PRESET_LIFTING_CYLINDER,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "rankine_oval",
        title: "Rankine Oval",
        description: "ψ = U·y + (Q/2π)·(θ₁ - θ₂): Closed streamline body formed by uniform flow and source-sink pair.",
        category: "Fluid Mechanics",
        icon: "🥚",
        domain: "fluid_dynamics",
        tool_name: Some("PotentialFlowTool"),
        preset_generator: || crate::starter_gallery::PRESET_RANKINE_OVAL,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "vortex_street",
        title: "Kelvin-Stuart Vortex Street",
        description: "Staggered counter-rotating vortex rows advecting meandering wake streamlines and RK4 dye streaks.",
        category: "Fluid Mechanics",
        icon: "🌀",
        domain: "fluid_dynamics",
        tool_name: Some("PotentialFlowTool"),
        preset_generator: || crate::starter_gallery::PRESET_VORTEX_STREET,
    }
}

pub(super) fn chain_segments(segments: &[([f64; 2], [f64; 2])]) -> Vec<Vec<[f64; 2]>> {
    let mut used = vec![false; segments.len()];
    let mut chains = Vec::new();
    let close = |a: [f64; 2], b: [f64; 2]| (a[0] - b[0]).hypot(a[1] - b[1]) <= 1e-3;
    for start in 0..segments.len() {
        if used[start] {
            continue;
        }
        used[start] = true;
        let mut poly = vec![segments[start].0, segments[start].1];
        for _ in 0..segments.len().min(512) {
            let tail = *poly.last().unwrap_or(&segments[start].1);
            let mut matched = false;
            for (idx, &(a, b)) in segments.iter().enumerate() {
                if used[idx] {
                    continue;
                }
                if close(tail, a) {
                    used[idx] = true;
                    poly.push(b);
                    matched = true;
                    break;
                } else if close(tail, b) {
                    used[idx] = true;
                    poly.push(a);
                    matched = true;
                    break;
                }
            }
            if !matched {
                break;
            }
        }
        chains.push(poly);
    }
    chains
}

impl PotentialFlowTool {
    pub(super) fn compute_iso_levels(&self) -> Vec<f64> {
        use math_explorer::physics::fluid_dynamics::potential_flow::FlowElement;
        let count = self.contour_levels.clamp(2, super::MAX_CONTOUR_LEVELS);
        let r = self.view_range.max(0.5);
        let mut min_psi = f64::INFINITY;
        let mut max_psi = f64::NEG_INFINITY;
        for ix in 0..=16 {
            let x = -r + (ix as f64) * (2.0 * r / 16.0);
            for iy in 0..=16 {
                let y = -r + (iy as f64) * (2.0 * r / 16.0);
                if !self.field.is_near_singularity(x, y, 0.25) {
                    let psi = self.field.stream_function(x, y);
                    if psi.is_finite() {
                        min_psi = min_psi.min(psi);
                        max_psi = max_psi.max(psi);
                    }
                }
            }
        }
        if !min_psi.is_finite() || !max_psi.is_finite() || (max_psi - min_psi).abs() <= 1e-6 {
            return vec![0.0];
        }
        let span = max_psi.abs().max(min_psi.abs()).max(1.0);
        let mut levels: Vec<f64> = (0..count)
            .map(|i| -span + (2.0 * span) * (i as f64) / ((count - 1) as f64))
            .collect();
        if levels.iter().all(|v| v.abs() > 1e-6) {
            levels.push(0.0);
            levels.sort_by(|a, b| a.total_cmp(b));
        }
        levels
    }
}
