#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Lazy one-time indexing of simulation tool starter presets and theory summaries for search.

use crate::framework::ToolMetadata;
use crate::starter_gallery::StarterPresetMetadata;
use std::collections::HashMap;
use std::sync::OnceLock;

/// Cached search corpus extracted lazily per `ToolMetadata.id`.
pub(super) struct ToolSearchCorpus {
    pub(super) theory_blob: String,
    pub(super) presets: Vec<&'static StarterPresetMetadata>,
}

static TOOL_CORPUS_INDEX: OnceLock<HashMap<&'static str, ToolSearchCorpus>> = OnceLock::new();

fn domains_match(preset_domain: &str, tool_domain: &str) -> bool {
    preset_domain == tool_domain
        || (preset_domain == "biology" && tool_domain == "morphogenesis")
        || (preset_domain == "morphogenesis" && tool_domain == "biology")
}

fn preset_matches_tool(
    preset: &StarterPresetMetadata,
    meta: &ToolMetadata,
    domain_tool_count: usize,
) -> bool {
    if !domains_match(preset.domain, meta.domain) {
        return false;
    }
    match preset.tool_name {
        Some(target) => {
            target.eq_ignore_ascii_case(meta.id) || target.eq_ignore_ascii_case(meta.display_name)
        }
        None => domain_tool_count == 1,
    }
}

fn build_theory_blob(meta: &ToolMetadata) -> String {
    let tool = (meta.build)();
    let theory = tool.theory();
    let mut blob = String::new();
    blob.push_str(&theory.theory_description());
    blob.push(' ');
    blob.push_str(&theory.theory_citation());
    if let Some(eq) = theory.theory_latex_equation() {
        blob.push(' ');
        blob.push_str(&eq);
    }
    for (key, desc) in theory.available_descriptions() {
        if !key.trim().eq_ignore_ascii_case("default") {
            blob.push(' ');
            blob.push_str(&key);
            blob.push(' ');
            blob.push_str(&desc);
        }
    }
    blob
}

fn init_tool_corpus_index() -> HashMap<&'static str, ToolSearchCorpus> {
    let all_tools: Vec<&'static ToolMetadata> = inventory::iter::<ToolMetadata>().collect();
    let all_presets: Vec<&'static StarterPresetMetadata> =
        inventory::iter::<StarterPresetMetadata>().collect();

    let mut domain_counts: HashMap<&'static str, usize> = HashMap::new();
    for meta in &all_tools {
        *domain_counts.entry(meta.domain).or_insert(0) += 1;
    }

    let mut map = HashMap::with_capacity(all_tools.len());
    for meta in all_tools {
        let count = domain_counts.get(meta.domain).copied().unwrap_or(1);
        let presets = all_presets
            .iter()
            .copied()
            .filter(|p| preset_matches_tool(p, meta, count))
            .collect();
        let theory_blob = build_theory_blob(meta);
        map.insert(
            meta.id,
            ToolSearchCorpus {
                theory_blob,
                presets,
            },
        );
    }
    map
}

/// Returns the lazily initialized preset and theory search corpus for `meta`.
pub(super) fn corpus_for_tool(meta: &ToolMetadata) -> Option<&'static ToolSearchCorpus> {
    TOOL_CORPUS_INDEX
        .get_or_init(init_tool_corpus_index)
        .get(meta.id)
}
