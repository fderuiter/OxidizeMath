#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{InteractiveTool, SimulationFramework, ToolMetadata};
use crate::tabs::traceability::{
    embedded_traceability_report, paper_tex_url, render_paper_links, TraceabilityTab,
};
use eframe::egui;
use std::collections::BTreeSet;

#[path = "theory_links_bib.rs"]
mod bib;
pub(crate) use bib::resolve_citation_reference_links;

/// Staged academic paper PDF and `.tex` source URLs resolved for a tool or domain.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ResolvedPaperLink {
    pub paper_name: String,
    pub pdf_url: Option<String>,
    pub tex_url: Option<String>,
}

/// External DOI, arXiv, or web reference link matched from a tool citation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ExternalCitationLink {
    pub label: String,
    pub url: String,
}

fn camel_or_space_to_snake(input: &str) -> String {
    let mut out = String::with_capacity(input.len() + 4);
    let mut prev_lower_or_digit = false;
    for ch in input.trim().chars() {
        if ch.is_ascii_alphanumeric() {
            if ch.is_ascii_uppercase() && prev_lower_or_digit && !out.ends_with('_') {
                out.push('_');
            }
            out.push(ch.to_ascii_lowercase());
            prev_lower_or_digit = ch.is_ascii_lowercase() || ch.is_ascii_digit();
        } else if !out.is_empty() && !out.ends_with('_') {
            out.push('_');
            prev_lower_or_digit = false;
        }
    }
    out.trim_matches('_').to_string()
}

fn explicit_tool_papers(tool_id: &str) -> &'static [&'static str] {
    match tool_id.trim() {
        "AttentionMapsTool" => &["attention_is_all_you_need_rust"],
        "CeraTool" => &["cera_framework"],
        "Co2ProjectionsTool" => &["isosurface_extraction"],
        "TurbulenceTool" => &["generative_turbulence"],
        "PartitionsWidget" => &["partitions_implementation"],
        "UalbfWidget" => &["ualbf_quasiperfect"],
        "EllipticCurvePlayground" | "WeierstrassPlaygroundTool" | "ModularInvariantsTool" => {
            &["modular_polynomials_review", "hurwitz_kronecker"]
        }
        "RadarGatingTool" => &["mmwave_radiotherapy_setup"],
        "OpticsWaveTool" => &["mmwave_radiotherapy_setup", "quantum_mechanics"],
        "TokenizationTool"
        | "FactorizationTool"
        | "FractranMachineTool"
        | "AlgorithmicFractranTool"
        | "AlgorithmicComplexityTool" => &["algorithmic_information_rust"],
        "WinRatioWorkbenchTool" | "WinRatioStressTesterTool" => &["win_ratio"],
        "AlgorithmVisualizerTool" | "GraphEditorTool" => &["dijkstra"],
        "VolumetricFieldStudioTool" => &["isosurface_extraction", "gaussian_splatting"],
        "GeodesicPlaygroundTool" | "ChristoffelInspectorTool" => &["general_relativity_geodesics"],
        "DistributionExplorerTool" | "CopulaDependenceTool" => {
            &["clinical_trials_statistics", "clinical_trials"]
        }
        "BayesianInferenceTool" => &[
            "freesurfer",
            "clinical_trials_statistics",
            "clinical_trials",
        ],
        "MarkovSequenceWorkbenchTool" => &["clinical_trials_statistics"],
        "CorticalMorphometryTool" | "RiemannianPdeStudio" => &["freesurfer"],
        "PhyllotaxisLSystemTool" | "BiochemicalDynamicsTool" | "EvolutionaryBiologyTool" => {
            &["computational_biology", "biology"]
        }
        _ => &[],
    }
}

fn tool_module_tokens(tool_id: &str) -> Vec<String> {
    let snake = camel_or_space_to_snake(tool_id);
    let mut tokens = vec![tool_id.trim().to_ascii_lowercase(), snake.clone()];
    for suffix in ["_tool", "_widget", "_viewer", "_plotter", "_simulator"] {
        if let Some(stem) = snake.strip_suffix(suffix).filter(|s| !s.is_empty()) {
            tokens.push(stem.to_string());
        }
    }
    tokens.extend(
        explicit_tool_papers(tool_id)
            .iter()
            .map(|s| (*s).to_string()),
    );
    tokens
}

fn domain_module_tokens(domain: &str) -> Vec<String> {
    let clean = domain.trim().to_ascii_lowercase();
    let mut tokens: Vec<String> = (!clean.is_empty())
        .then(|| clean.clone())
        .into_iter()
        .collect();
    let aliases: &[&str] = match clean.as_str() {
        "algorithmic_information" => &["algorithmic_information_rust", "pure_math", "algorithms"],
        "chaos" => &["chaos_theory"],
        "elliptic_curves" => &[
            "modular_polynomials_review",
            "hurwitz_kronecker",
            "pure_math",
        ],
        "game_theory" => &["graph_parameters_rust"],
        "graph_theory" => &["dijkstra", "graph_parameters_rust"],
        "analysis" | "geometry_topology" | "number_theory" => &["pure_math", "math_commons"],
        "probability_statistics" => &[
            "clinical_trials_statistics",
            "clinical_trials",
            "freesurfer",
            "pure_math",
        ],
        "financial_math" => &["clinical_trials_statistics"],
        "biology" | "biochemical" => &["biology", "computational_biology"],
        "neuroscience" => &["biology", "computational_biology", "graph_parameters_rust"],
        "optics_waves" => &["domain_physics", "mmwave_radiotherapy_setup"],
        "solid_state" => &["stat_mech", "pure_math"],
        "ai" => &["graph_parameters_rust"],
        "paradox_lab" => &["paradox_lab", "game_theory", "graph_parameters_rust"],
        "relativity" => &[
            "general_relativity_geodesics",
            "general_relativity",
            "tensors",
        ],
        _ => &[],
    };
    tokens.extend(aliases.iter().map(|s| (*s).to_string()));
    tokens
}

/// Resolves hosted PDF and `.tex` links for `(tool_id, domain)` against `traceability_manifest.json`.
pub(crate) fn resolve_portal_paper_links(tool_id: &str, domain: &str) -> Vec<ResolvedPaperLink> {
    let Ok(report) = embedded_traceability_report() else {
        return Vec::new();
    };
    let tool_tokens = tool_module_tokens(tool_id);
    let domain_tokens = domain_module_tokens(domain);
    let mut primary = Vec::new();
    let mut secondary = Vec::new();

    for (paper_name, modules) in &report.paper_coverage {
        if paper_name.starts_with("spec:") || paper_name.starts_with("registry:") {
            continue;
        }
        let matches_tool = modules
            .iter()
            .any(|m| tool_tokens.iter().any(|t| m.eq_ignore_ascii_case(t)));
        let matches_domain = modules
            .iter()
            .any(|m| domain_tokens.iter().any(|d| m.eq_ignore_ascii_case(d)));
        if matches_tool {
            primary.push(paper_name.clone());
        } else if matches_domain {
            secondary.push(paper_name.clone());
        }
    }
    primary.sort();
    secondary.sort();

    let mut seen = BTreeSet::new();
    primary
        .into_iter()
        .chain(secondary)
        .filter(|name| seen.insert(name.clone()))
        .filter_map(|paper_name| {
            let pdf_url = TraceabilityTab::paper_url(&paper_name);
            let tex_url = paper_tex_url(&paper_name);
            (pdf_url.is_some() || tex_url.is_some()).then_some(ResolvedPaperLink {
                paper_name,
                pdf_url,
                tex_url,
            })
        })
        .collect()
}

/// Renders hosted paper PDF, `.tex` source, and external DOI/arXiv citation links in the Theory Context Portal.
pub(crate) fn render_portal_links_for_framework(
    framework: &SimulationFramework,
    ui: &mut egui::Ui,
    tool: &dyn InteractiveTool,
    citation: &str,
) {
    let active_meta: Option<&ToolMetadata> = framework
        .selected_tool_index
        .and_then(|idx| framework.available_tools.get(idx).copied());
    let tool_id = active_meta.map_or_else(|| tool.name(), |m| m.id);
    let domain = active_meta.map_or_else(
        || {
            if framework.domain.is_empty() {
                tool.domain()
            } else {
                framework.domain.as_str()
            }
        },
        |m| m.domain,
    );

    let paper_links = resolve_portal_paper_links(tool_id, domain);
    let ext_links = resolve_citation_reference_links(citation);
    if paper_links.is_empty() && ext_links.is_empty() {
        return;
    }

    ui.add_space(4.0);
    for paper in &paper_links {
        let stem = paper
            .paper_name
            .strip_suffix(".tex")
            .unwrap_or(&paper.paper_name);
        let pdf_label = format!("📄 View PDF ({stem}.pdf)");
        render_paper_links(
            ui,
            paper.pdf_url.as_deref(),
            paper.tex_url.as_deref(),
            &pdf_label,
        );
    }
    if !ext_links.is_empty() {
        ui.horizontal_wrapped(|ui| {
            for ext in &ext_links {
                ui.hyperlink_to(format!("🔗 {}", ext.label), &ext.url);
            }
        });
    }
}
