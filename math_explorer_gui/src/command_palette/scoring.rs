#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::theory_index::corpus_for_tool;
use std::borrow::Cow;

/// Formats a snake_case domain identifier into a user-facing domain title.
pub fn format_domain_title(domain: &str) -> Cow<'static, str> {
    match domain.trim() {
        "ai" => Cow::Borrowed("Artificial Intelligence"),
        "algorithmic_information" => Cow::Borrowed("Algorithmic Information"),
        "analysis" => Cow::Borrowed("Analysis & Calculus"),
        "astrophysics" => Cow::Borrowed("Astrophysics"),
        "battery_degradation" => Cow::Borrowed("Battery Degradation"),
        "biology" => Cow::Borrowed("Biology"),
        "chaos" => Cow::Borrowed("Chaos Theory"),
        "climate" => Cow::Borrowed("Climate"),
        "clinical_trials" => Cow::Borrowed("Clinical Trials"),
        "elliptic_curves" => Cow::Borrowed("Elliptic Curves"),
        "epidemiology" => Cow::Borrowed("Epidemiology"),
        "favoritism" => Cow::Borrowed("Favoritism"),
        "financial_math" => Cow::Borrowed("Financial Math"),
        "fluid_dynamics" => Cow::Borrowed("Fluid Dynamics"),
        "game_theory" => Cow::Borrowed("Game Theory"),
        "geometry_topology" => Cow::Borrowed("Geometry & Topology"),
        "graph_theory" => Cow::Borrowed("Graph Theory"),
        "medical" => Cow::Borrowed("Medical Physics"),
        "morphogenesis" => Cow::Borrowed("Morphogenesis"),
        "mri" => Cow::Borrowed("MRI Physics"),
        "neuroscience" => Cow::Borrowed("Neuroscience"),
        "number_theory" => Cow::Borrowed("Number Theory"),
        "paradox_lab" => Cow::Borrowed("Paradox Lab"),
        "quantum" => Cow::Borrowed("Quantum Mechanics"),
        "relativity" => Cow::Borrowed("General Relativity"),
        "solid_state" => Cow::Borrowed("Solid State Physics"),
        "traceability" => Cow::Borrowed("Traceability Portal"),
        other => Cow::Owned(title_case_identifier(other)),
    }
}

fn title_case_identifier(raw: &str) -> String {
    raw.split('_')
        .filter(|part| !part.is_empty())
        .map(|word| {
            let mut chars = word.chars();
            chars.next().map_or_else(String::new, |c| {
                format!("{}{}", c.to_uppercase(), chars.as_str())
            })
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Folds common Latin diacritics to ASCII lowercase so queries like `schrodinger` match `Schrödinger`.
pub(crate) fn fold_search_text(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for ch in input.chars() {
        for lower in ch.to_lowercase() {
            match lower {
                'à' | 'á' | 'â' | 'ã' | 'ä' | 'å' => out.push('a'),
                'ç' => out.push('c'),
                'è' | 'é' | 'ê' | 'ë' => out.push('e'),
                'ì' | 'í' | 'î' | 'ï' => out.push('i'),
                'ñ' => out.push('n'),
                'ò' | 'ó' | 'ô' | 'õ' | 'ö' | 'ø' => out.push('o'),
                'ù' | 'ú' | 'û' | 'ü' => out.push('u'),
                'ý' | 'ÿ' => out.push('y'),
                'ß' => out.push_str("ss"),
                other => out.push(other),
            }
        }
    }
    out
}

/// Extracts folded alphanumeric word tokens, splitting on punctuation/whitespace and CamelCase boundaries.
pub(crate) fn extract_search_words(input: &str) -> Vec<String> {
    let chars: Vec<char> = input.chars().collect();
    let mut words = Vec::new();
    let mut current = String::new();

    for idx in 0..chars.len() {
        let ch = chars[idx];
        if !ch.is_alphanumeric() {
            if !current.is_empty() {
                words.push(fold_search_text(&current));
                current.clear();
            }
            continue;
        }
        if idx > 0 && !current.is_empty() {
            let prev = chars[idx - 1];
            let next_is_lower = chars.get(idx + 1).is_some_and(|n| n.is_lowercase());
            let camel_split = (prev.is_lowercase() && ch.is_uppercase())
                || (prev.is_uppercase() && ch.is_uppercase() && next_is_lower);
            if camel_split {
                words.push(fold_search_text(&current));
                current.clear();
            }
        }
        current.push(ch);
    }
    if !current.is_empty() {
        words.push(fold_search_text(&current));
    }
    words
}

fn find_word_boundary_substring(haystack: &str, needle: &str) -> Option<usize> {
    if needle.is_empty() {
        return Some(0);
    }
    let mut offset = 0;
    while let Some(rel) = haystack[offset..].find(needle) {
        let pos = offset + rel;
        let at_boundary = pos == 0
            || haystack[..pos]
                .chars()
                .next_back()
                .is_none_or(|prev| !prev.is_alphanumeric());
        if at_boundary {
            return Some(pos);
        }
        let step = haystack[pos..].chars().next().map_or(1, char::len_utf8);
        offset = pos + step;
    }
    None
}

fn score_multi_token_words(
    words: &[String],
    text_folded: &str,
    q_tokens: &[String],
) -> Option<usize> {
    let mut in_order = true;
    let mut search_from = 0usize;
    for qt in q_tokens {
        let word_pos = words.iter().position(|w| w.starts_with(qt.as_str()));
        let Some(matched_idx) = word_pos else {
            find_word_boundary_substring(text_folded, qt)?;
            in_order = false;
            continue;
        };
        if matched_idx < search_from {
            in_order = false;
        } else {
            search_from = matched_idx + 1;
        }
    }
    Some(if in_order { 25 } else { 35 })
}

/// Scores `query` against `text` strictly on word boundaries (lower score = higher relevance).
///
/// Scattered character-subsequence matches (e.g. `"qspin"` in `"Quantum / Spin"`) are rejected.
pub fn fuzzy_match_score(text: &str, query: &str) -> Option<usize> {
    let query_folded = fold_search_text(query.trim());
    if query_folded.is_empty() {
        return Some(usize::MAX);
    }

    let text_folded = fold_search_text(text.trim());
    if text_folded == query_folded {
        return Some(0);
    }
    if text_folded.starts_with(&query_folded) {
        return Some(1);
    }

    let words = extract_search_words(text);
    if let Some(word_idx) = words.iter().position(|w| w == &query_folded) {
        return Some(2 + word_idx);
    }
    if let Some(pos) = find_word_boundary_substring(&text_folded, &query_folded) {
        return Some(10 + pos);
    }
    if let Some(word_idx) = words.iter().position(|w| w.starts_with(&query_folded)) {
        return Some(14 + word_idx * 2);
    }

    let q_tokens = extract_search_words(&query_folded);
    if q_tokens.len() > 1 {
        return score_multi_token_words(&words, &text_folded, &q_tokens);
    }
    None
}

fn merge_min(cur: Option<usize>, next: Option<usize>, penalty: usize) -> Option<usize> {
    match (cur, next.map(|s| s.saturating_add(penalty))) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    }
}

/// Scores a registered [`crate::framework::ToolMetadata`] across display name, ID, tags,
/// domain, starter presets, and lazy theory summary keywords on word boundaries.
pub(crate) fn score_tool_metadata(
    meta: &crate::framework::ToolMetadata,
    query: &str,
) -> Option<usize> {
    if query.trim().is_empty() {
        return Some(usize::MAX);
    }
    let domain_title = format_domain_title(meta.domain);
    let formatted_entry = format!("{} ({})", meta.display_name, domain_title);

    let mut score = fuzzy_match_score(meta.display_name, query);
    for candidate in [meta.id, formatted_entry.as_str()] {
        score = merge_min(score, fuzzy_match_score(candidate, query), 0);
    }
    for &tag in meta.tags {
        score = merge_min(score, fuzzy_match_score(tag, query), 4);
    }
    for domain_candidate in [meta.domain, domain_title.as_ref()] {
        score = merge_min(score, fuzzy_match_score(domain_candidate, query), 50);
    }
    if score.is_some_and(|s| s <= 2) {
        return score;
    }

    if let Some(corpus) = corpus_for_tool(meta) {
        for preset in &corpus.presets {
            score = merge_min(score, fuzzy_match_score(preset.title, query), 8);
            score = merge_min(score, fuzzy_match_score(preset.description, query), 16);
            score = merge_min(score, fuzzy_match_score(preset.category, query), 24);
        }
        if score.is_none() {
            score = merge_min(score, fuzzy_match_score(&corpus.theory_blob, query), 20);
        }
        if score.is_none() {
            score = score_cross_field_tokens(meta, &domain_title, corpus, query);
        }
    }
    score
}

fn score_cross_field_tokens(
    meta: &crate::framework::ToolMetadata,
    domain_title: &str,
    corpus: &super::theory_index::ToolSearchCorpus,
    query: &str,
) -> Option<usize> {
    let q_tokens = extract_search_words(query);
    if q_tokens.len() <= 1 {
        return None;
    }
    let all_matched = q_tokens.iter().all(|qt| {
        fuzzy_match_score(meta.display_name, qt).is_some()
            || fuzzy_match_score(meta.id, qt).is_some()
            || fuzzy_match_score(meta.domain, qt).is_some()
            || fuzzy_match_score(domain_title, qt).is_some()
            || meta
                .tags
                .iter()
                .any(|tag| fuzzy_match_score(tag, qt).is_some())
            || corpus.presets.iter().any(|p| {
                fuzzy_match_score(p.title, qt).is_some()
                    || fuzzy_match_score(p.description, qt).is_some()
            })
            || fuzzy_match_score(&corpus.theory_blob, qt).is_some()
    });
    all_matched.then_some(45)
}
