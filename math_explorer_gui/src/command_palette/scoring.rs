use std::borrow::Cow;

/// Formats a snake_case domain identifier into a user-facing domain title.
pub fn format_domain_title(domain: &str) -> Cow<'static, str> {
    match domain.trim() {
        "ai" => Cow::Borrowed("Artificial Intelligence"),
        "analysis" => Cow::Borrowed("Analysis & Calculus"),
        "battery_degradation" => Cow::Borrowed("Battery Degradation"),
        "biology" => Cow::Borrowed("Biology"),
        "chaos" => Cow::Borrowed("Chaos Theory"),
        "climate" => Cow::Borrowed("Climate"),
        "clinical_trials" => Cow::Borrowed("Clinical Trials"),
        "epidemiology" => Cow::Borrowed("Epidemiology"),
        "financial_math" => Cow::Borrowed("Financial Math"),
        "fluid_dynamics" => Cow::Borrowed("Fluid Dynamics"),
        "game_theory" => Cow::Borrowed("Game Theory"),
        "geometry_topology" => Cow::Borrowed("Geometry & Topology"),
        "graph_theory" => Cow::Borrowed("Graph Theory"),
        "medical" => Cow::Borrowed("Medical Physics"),
        "neuroscience" => Cow::Borrowed("Neuroscience"),
        "number_theory" => Cow::Borrowed("Number Theory"),
        "quantum" => Cow::Borrowed("Quantum Mechanics"),
        "solid_state" => Cow::Borrowed("Solid State Physics"),
        other => Cow::Owned(
            other
                .split('_')
                .filter(|part| !part.is_empty())
                .map(|word| {
                    let mut chars = word.chars();
                    chars.next().map_or_else(String::new, |c| {
                        format!("{}{}", c.to_uppercase(), chars.as_str())
                    })
                })
                .collect::<Vec<_>>()
                .join(" "),
        ),
    }
}

pub(super) fn score_tool_metadata(
    meta: &crate::framework::ToolMetadata,
    query: &str,
) -> Option<usize> {
    let domain_title = format_domain_title(meta.domain);
    let formatted_entry = format!("{} ({})", meta.display_name, domain_title);
    let mut score = fuzzy_match_score(meta.display_name, query);
    for candidate in [meta.id, formatted_entry.as_str()] {
        if let Some(s) = fuzzy_match_score(candidate, query) {
            score = Some(score.map_or(s, |cur| cur.min(s)));
        }
    }
    for &tag in meta.tags {
        if let Some(tag_score) = fuzzy_match_score(tag, query) {
            score = Some(score.map_or(tag_score, |cur| cur.min(tag_score)));
        }
    }
    for domain_candidate in [meta.domain, domain_title.as_ref()] {
        if let Some(domain_score) = fuzzy_match_score(domain_candidate, query) {
            let weighted = domain_score.saturating_add(50);
            score = Some(score.map_or(weighted, |cur| cur.min(weighted)));
        }
    }
    score
}

/// Helper to check if `query` matches `text` using case-insensitive fuzzy or substring filtering.
/// Returns a score if matched (lower score = better match / higher rank).
pub fn fuzzy_match_score(text: &str, query: &str) -> Option<usize> {
    let trimmed_query = query.trim().to_lowercase();
    if trimmed_query.is_empty() {
        return Some(usize::MAX);
    }

    let text_lower = text.to_lowercase();

    // 1. Exact match
    if text_lower == trimmed_query {
        return Some(0);
    }

    // 2. Exact prefix match
    if text_lower.starts_with(&trimmed_query) {
        return Some(1);
    }

    // 3. Exact contiguous substring match
    if let Some(pos) = text_lower.find(&trimmed_query) {
        return Some(10 + pos);
    }

    // 4. All whitespace-separated terms match as contiguous substrings
    let terms: Vec<&str> = trimmed_query.split_whitespace().collect();
    if terms.len() > 1 && terms.iter().all(|term| text_lower.contains(term)) {
        return Some(100);
    }

    // 5. Fuzzy character subsequence match (all non-space query characters appear in order)
    let query_chars: Vec<char> = trimmed_query
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect();
    if query_chars.is_empty() {
        return Some(usize::MAX);
    }

    let mut text_chars = text_lower.chars();
    let mut matched_count = 0;
    for q_char in &query_chars {
        if text_chars.any(|t_char| t_char == *q_char) {
            matched_count += 1;
        } else {
            break;
        }
    }

    if matched_count == query_chars.len() {
        Some(200)
    } else {
        None
    }
}
