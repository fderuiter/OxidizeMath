#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

mod bib_links;

use super::{InteractiveTool, SimulationFramework, ToolMetadata};
use crate::tabs::traceability::{
    embedded_traceability_report, paper_tex_url, parse_spec_stem, render_paper_links,
    TraceabilityTab,
};
use eframe::egui;
use std::collections::BTreeSet;

const GITHUB_BIB_REGISTRY_URL: &str =
    "https://github.com/fderuiter/OxidizeMath/blob/main/papers/references.bib";

/// Staged academic paper PDF and `.tex` source URLs resolved for a tool or domain.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ResolvedPaperLink {
    pub paper_name: String,
    pub pdf_url: Option<String>,
    pub tex_url: Option<String>,
}

/// Verified `docs/math_specs/<id>.md` specification link resolved for a tool or domain.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ResolvedSpecLink {
    pub spec_stem: String,
    pub spec_url: String,
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

fn tool_module_tokens(tool_id: &str) -> Vec<String> {
    let snake = camel_or_space_to_snake(tool_id);
    let mut tokens = vec![tool_id.trim().to_ascii_lowercase(), snake.clone()];
    for suffix in ["_tool", "_widget", "_viewer", "_plotter", "_simulator"] {
        if let Some(stem) = snake.strip_suffix(suffix) {
            if !stem.is_empty() {
                tokens.push(stem.to_string());
            }
        }
    }
    let explicit: &[&str] = match tool_id.trim() {
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
        "WinRatioWorkbenchTool" | "WinRatioStressTesterTool" | "WinRatioTool" => &["win_ratio"],
        "AlgorithmVisualizerTool" | "GraphEditorTool" => &["dijkstra"],
        "VolumetricFieldStudioTool" => &["isosurface_extraction", "gaussian_splatting"],
        "GeodesicPlaygroundTool" | "ChristoffelInspectorTool" => &["general_relativity_geodesics"],
        "MarkovSequenceWorkbenchTool" => &["clinical_trials_statistics"],
        "CorticalMorphometryTool" | "RiemannianPdeStudio" => &["freesurfer"],
        "PhyllotaxisLSystemTool" | "BiochemicalDynamicsTool" | "EvolutionaryBiologyTool" => {
            &["computational_biology", "biology"]
        }
        _ => &[],
    };
    tokens.extend(explicit.iter().map(|s| (*s).to_string()));
    tokens
}

fn domain_module_tokens(domain: &str) -> Vec<String> {
    let clean = domain.trim().to_ascii_lowercase();
    let mut tokens = Vec::new();
    if !clean.is_empty() {
        tokens.push(clean.clone());
    }
    let aliases: &[&str] = match clean.as_str() {
        "algorithmic_information" => &["algorithmic_information_rust", "pure_math", "algorithms"],
        "chaos" => &["chaos_theory"],
        "clinical_trials" => &["clinical_trials_statistics"],
        "elliptic_curves" => &[
            "modular_polynomials_review",
            "hurwitz_kronecker",
            "pure_math",
        ],
        "game_theory" => &["graph_parameters_rust"],
        "graph_theory" => &["dijkstra", "graph_parameters_rust"],
        "analysis" | "geometry_topology" | "number_theory" => &["pure_math", "math_commons"],
        "financial_math" => &["clinical_trials_statistics"],
        "stochastic_processes" => &[
            "clinical_trials_statistics",
            "advanced_linear_algebra",
            "pure_math",
        ],
        "biology" | "biochemical" | "morphogenesis" => &["biology", "computational_biology"],
        "neuroscience" => &["biology", "computational_biology", "graph_parameters_rust"],
        "optics_waves" => &["optics", "domain_physics", "mmwave_radiotherapy_setup"],
        "quantum" => &["quantum_mechanics"],
        "solid_state" => &["stat_mech", "pure_math"],
        "ai" => &["attention_is_all_you_need_rust", "graph_parameters_rust"],
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

/// Resolves verified `docs/math_specs/<id>.md` specification links for `(tool_id, domain)`.
pub(crate) fn resolve_portal_spec_links(tool_id: &str, domain: &str) -> Vec<ResolvedSpecLink> {
    let Ok(report) = embedded_traceability_report() else {
        return Vec::new();
    };
    let tool_tokens = tool_module_tokens(tool_id);
    let domain_tokens = domain_module_tokens(domain);
    let mut matched = Vec::new();

    for (target_name, modules) in &report.paper_coverage {
        if !target_name.starts_with("spec:") {
            continue;
        }
        let hits = modules.iter().any(|m| {
            tool_tokens.iter().any(|t| m.eq_ignore_ascii_case(t))
                || domain_tokens.iter().any(|d| m.eq_ignore_ascii_case(d))
        });
        if hits {
            if let (Some(stem), Some(url)) = (
                parse_spec_stem(target_name),
                TraceabilityTab::spec_url(target_name),
            ) {
                matched.push(ResolvedSpecLink {
                    spec_stem: stem.to_string(),
                    spec_url: url,
                });
            }
        }
    }
    matched.sort_by(|a, b| a.spec_stem.cmp(&b.spec_stem));
    matched.dedup_by(|a, b| a.spec_stem == b.spec_stem);
    matched
}

/// Matches a citation string or BibTeX key against `papers/references.bib` DOI, arXiv, and URL entries.
pub(crate) fn resolve_citation_reference_links(citation: &str) -> Vec<ExternalCitationLink> {
    bib_links::resolve_bib_links_for_citation(citation)
}

/// Renders hosted paper PDF, `.tex` source, verified math specs, and external DOI/arXiv citation links.
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
    let fallback_domain = if framework.domain.is_empty() {
        tool.domain()
    } else {
        framework.domain.as_str()
    };
    let domain = active_meta.map_or(fallback_domain, |m| m.domain);

    let paper_links = resolve_portal_paper_links(tool_id, domain);
    let spec_links = resolve_portal_spec_links(tool_id, domain);
    let ext_links = resolve_citation_reference_links(citation);
    if paper_links.is_empty() && spec_links.is_empty() && ext_links.is_empty() {
        return;
    }

    ui.add_space(4.0);
    for paper in &paper_links {
        let stem = paper.paper_name.trim_end_matches(".tex");
        let pdf_label = format!("📄 View PDF ({stem}.pdf)");
        render_paper_links(
            ui,
            paper.pdf_url.as_deref(),
            paper.tex_url.as_deref(),
            &pdf_label,
        );
    }
    if !spec_links.is_empty() || !ext_links.is_empty() {
        ui.horizontal_wrapped(|ui| {
            for spec in &spec_links {
                ui.hyperlink_to(
                    format!("🔗 Math Spec ({}.md)", spec.spec_stem),
                    &spec.spec_url,
                );
            }
            for ext in &ext_links {
                ui.hyperlink_to(format!("🔗 {}", ext.label), &ext.url);
            }
            if paper_links.is_empty() && ext_links.is_empty() && spec_links.len() == 1 {
                ui.hyperlink_to(
                    "🔗 BibTeX Registry (references.bib)",
                    GITHUB_BIB_REGISTRY_URL,
                );
            }
        });
    }
}
