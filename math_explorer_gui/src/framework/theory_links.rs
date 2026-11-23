#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{InteractiveTool, SimulationFramework, ToolMetadata};
use crate::tabs::traceability::{
    embedded_traceability_report, paper_tex_url, render_paper_links, TraceabilityTab,
};
use eframe::egui;
use std::collections::BTreeSet;
use std::sync::OnceLock;

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

#[derive(Clone, Debug)]
struct BibEntry {
    norm_title_words: Vec<String>,
    first_author: String,
    year: String,
    label: String,
    url: String,
}

const STOP_WORDS: &[&str] = &[
    "a", "an", "and", "by", "for", "from", "in", "into", "of", "on", "the", "to", "via", "with",
];

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
        "TokenizationTool"
        | "FactorizationTool"
        | "FractranMachineTool"
        | "AlgorithmicFractranTool"
        | "AlgorithmicComplexityTool" => &["algorithmic_information_rust"],
        "AlgorithmVisualizerTool" | "GraphEditorTool" => &["dijkstra"],
        "VolumetricFieldStudioTool" => &["isosurface_extraction", "gaussian_splatting"],
        "BetSizeCalculatorTool"
        | "BankrollGrowthTool"
        | "CorrelatedRiskScenarioTool"
        | "Glicko2TournamentTool"
        | "MarketDeviggingTool"
        | "MatchScoreHeatmapTool"
        | "SgpCopulaTool"
        | "ZipCountPropsTool"
        | "LiveMomentumTool"
        | "SportsRatingsTool"
        | "ClvRuinTool" => &["sports_quantitative_modeling"],
        "GeodesicPlaygroundTool" | "ChristoffelInspectorTool" => &["general_relativity_geodesics"],
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
        "algorithmic_information" => &["algorithmic_information_rust", "pure_math"],
        "chaos" => &["chaos_theory"],
        "elliptic_curves" => &[
            "modular_polynomials_review",
            "hurwitz_kronecker",
            "pure_math",
        ],
        "game_theory" => &["graph_parameters_rust"],
        "graph_theory" => &["dijkstra", "graph_parameters_rust"],
        "analysis" | "geometry_topology" | "number_theory" => &["pure_math", "math_commons"],
        "financial_math" | "sports_analytics" => &["sports_quantitative_modeling"],
        "neuroscience" => &["biology", "graph_parameters_rust"],
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

fn content_words(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(|w| w.to_ascii_lowercase())
        .filter(|w| !STOP_WORDS.contains(&w.as_str()))
        .collect()
}

fn extract_arxiv_id(text: &str) -> Option<String> {
    let lower = text.to_ascii_lowercase();
    let idx = lower.find("arxiv:")?;
    let after = &text[idx + 6..];
    let id: String = after
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == '.' || *c == '/' || *c == '-')
        .collect();
    let trimmed = id.trim_end_matches('.');
    (trimmed.len() >= 7 && trimmed.chars().any(|c| c.is_ascii_digit())).then(|| trimmed.to_string())
}

fn extract_http_url(text: &str) -> Option<String> {
    let start = text.find("https://").or_else(|| text.find("http://"))?;
    let url: String = text[start..]
        .chars()
        .take_while(|c| {
            !c.is_whitespace() && !matches!(c, '}' | ')' | ']' | '>' | '"' | '\'' | '\\')
        })
        .collect();
    let trimmed = url.trim_end_matches(['.', ',', ';']);
    (trimmed.len() > 10).then(|| trimmed.to_string())
}

fn embedded_bib_entries() -> &'static [BibEntry] {
    static BIB_CACHE: OnceLock<Vec<BibEntry>> = OnceLock::new();
    BIB_CACHE.get_or_init(|| {
        const BIB_TSV: &str = include_str!(concat!(env!("OUT_DIR"), "/bib_links.tsv"));
        BIB_TSV
            .lines()
            .filter_map(|line| {
                let mut parts = line.splitn(5, '\t');
                Some(BibEntry {
                    first_author: parts.next()?.to_string(),
                    year: parts.next()?.to_string(),
                    label: parts.next()?.to_string(),
                    url: parts.next()?.to_string(),
                    norm_title_words: parts
                        .next()?
                        .split_whitespace()
                        .map(str::to_string)
                        .collect(),
                })
            })
            .collect()
    })
}

fn contains_word_subsequence(haystack: &[String], needle: &[String]) -> bool {
    if needle.len() < 3 || needle.len() > haystack.len() {
        return false;
    }
    haystack.windows(needle.len()).any(|w| w == needle)
}

fn extract_inline_citation_links(citation: &str) -> Vec<ExternalCitationLink> {
    let mut links = Vec::new();
    if let Some(id) = extract_arxiv_id(citation) {
        links.push(ExternalCitationLink {
            label: format!("arXiv:{id}"),
            url: format!("https://arxiv.org/abs/{id}"),
        });
    }
    if let Some(idx) = citation.find("10.") {
        let candidate: String = citation[idx..]
            .chars()
            .take_while(|c| !c.is_whitespace() && !matches!(c, ')' | ']' | ';' | ','))
            .collect();
        let doi = candidate.trim_end_matches('.');
        if doi.contains('/') && doi.len() >= 8 {
            links.push(ExternalCitationLink {
                label: format!("DOI: {doi}"),
                url: format!("https://doi.org/{doi}"),
            });
        }
    }
    if let Some(url) = extract_http_url(citation) {
        if !url.contains("doi.org/") && !url.contains("arxiv.org/") {
            links.push(ExternalCitationLink {
                label: "External Reference".to_string(),
                url,
            });
        }
    }
    links
}

/// Matches a citation string against `papers/references.bib` DOI, arXiv, and URL entries.
pub(crate) fn resolve_citation_reference_links(citation: &str) -> Vec<ExternalCitationLink> {
    if citation.trim().is_empty() {
        return Vec::new();
    }
    let cit_words = content_words(citation);
    let mut seen_urls = BTreeSet::new();
    let mut out = Vec::new();

    for link in extract_inline_citation_links(citation) {
        if seen_urls.insert(link.url.clone()) {
            out.push(link);
        }
    }
    for entry in embedded_bib_entries() {
        let title_match = contains_word_subsequence(&cit_words, &entry.norm_title_words);
        let author_year_match = entry.first_author.len() >= 4
            && entry.year.len() == 4
            && cit_words.iter().any(|w| w == &entry.first_author)
            && cit_words.iter().any(|w| w == &entry.year)
            && entry
                .norm_title_words
                .iter()
                .any(|tw| tw.len() >= 5 && cit_words.iter().any(|cw| cw == tw));
        if (title_match || author_year_match) && seen_urls.insert(entry.url.clone()) {
            out.push(ExternalCitationLink {
                label: entry.label.clone(),
                url: entry.url.clone(),
            });
        }
    }
    out
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
