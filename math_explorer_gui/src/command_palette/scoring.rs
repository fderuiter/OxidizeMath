use std::borrow::Cow;

/// Formats a snake_case domain identifier into a user-facing domain title.
pub fn format_domain_title(domain: &str) -> Cow<'static, str> {
    match domain.trim() {
        "ai" => Cow::Borrowed("Artificial Intelligence"),
        "analysis" => Cow::Borrowed("Analysis & Calculus"),
        "astrophysics" => Cow::Borrowed("Astrophysics"),
        "battery_degradation" => Cow::Borrowed("Battery Degradation"),
        "biology" => Cow::Borrowed("Biology"),
        "chaos" => Cow::Borrowed("Chaos Theory"),
        "climate" => Cow::Borrowed("Climate"),
        "clinical_trials" => Cow::Borrowed("Clinical Trials"),
        "epidemiology" => Cow::Borrowed("Epidemiology"),
        "favoritism" => Cow::Borrowed("Favoritism"),
        "financial_math" => Cow::Borrowed("Financial Math"),
        "fluid_dynamics" => Cow::Borrowed("Fluid Dynamics"),
        "game_theory" => Cow::Borrowed("Game Theory"),
        "geometry_topology" => Cow::Borrowed("Geometry & Topology"),
        "graph_theory" => Cow::Borrowed("Graph Theory"),
        "medical" => Cow::Borrowed("Medical Physics"),
        "mri" => Cow::Borrowed("MRI Physics"),
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

pub(crate) fn score_tool_metadata(
    meta: &crate::framework::ToolMetadata,
    query: &str,
) -> Option<usize> {
    let domain_title = format_domain_title(meta.domain);
    let formatted_entry = format!("{} ({})", meta.display_name, domain_title);
    let domain_prefixed = format!("{} {}", domain_title, meta.display_name);
    let mut score = fuzzy_match_score(meta.display_name, query);
    for candidate in [meta.id, formatted_entry.as_str(), domain_prefixed.as_str()] {
        if let Some(s) = fuzzy_match_score(candidate, query) {
            score = Some(score.map_or(s, |cur| cur.min(s)));
        }
    }
    for &tag in meta.tags {
        if let Some(tag_score) = fuzzy_match_score(tag, query) {
            score = Some(score.map_or(tag_score, |cur| cur.min(tag_score)));
        }
        let domain_tag = format!("{domain_title} {tag}");
        let words = split_normalized_words(&domain_tag);
        if let Some(gap_score) =
            match_bounded_gap_word_prefixes(&words, &query.trim().to_lowercase())
        {
            score = Some(score.map_or(gap_score, |cur| cur.min(gap_score)));
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

/// Scores `meta` within its already-selected simulation domain (display name, id, and curated tags).
pub(crate) fn score_tool_in_domain(
    meta: &crate::framework::ToolMetadata,
    query: &str,
) -> Option<usize> {
    let mut score = fuzzy_match_score(meta.display_name, query);
    if let Some(id_score) = fuzzy_match_score(meta.id, query) {
        score = Some(score.map_or(id_score, |cur| cur.min(id_score)));
    }
    for &tag in meta.tags {
        if let Some(tag_score) = fuzzy_match_score(tag, query) {
            let weighted = tag_score.saturating_add(2);
            score = Some(score.map_or(weighted, |cur| cur.min(weighted)));
        }
    }
    score
}

fn split_normalized_words(text: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut current = String::new();
    let mut prev_is_lower = false;

    for ch in text.chars() {
        if !ch.is_alphanumeric() {
            if !current.is_empty() {
                words.push(std::mem::take(&mut current));
            }
            prev_is_lower = false;
            continue;
        }
        let is_upper = ch.is_uppercase();
        if is_upper && prev_is_lower && !current.is_empty() {
            words.push(std::mem::take(&mut current));
        }
        for lower in ch.to_lowercase() {
            current.push(lower);
        }
        prev_is_lower = ch.is_lowercase();
    }
    if !current.is_empty() {
        words.push(current);
    }
    words
}

fn match_word_prefix_score(words: &[String], query: &str) -> Option<usize> {
    for (idx, word) in words.iter().enumerate() {
        if word == query {
            return Some(3 + idx);
        }
        if word.starts_with(query) {
            return Some(5 + idx);
        }
    }
    None
}

fn match_multi_term_tokens(text_lower: &str, words: &[String], query: &str) -> Option<usize> {
    let terms: Vec<&str> = query
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .collect();
    if terms.len() <= 1 {
        return None;
    }
    if terms
        .iter()
        .all(|term| words.iter().any(|w| w.starts_with(term)))
    {
        return Some(40);
    }
    if terms.iter().all(|term| text_lower.contains(term)) {
        return Some(85);
    }
    None
}

fn match_bounded_gap_word_prefixes(words: &[String], query: &str) -> Option<usize> {
    if query.len() < 4 || words.len() < 2 || query.chars().any(|c| !c.is_alphanumeric()) {
        return None;
    }
    for first_idx in 0..words.len() {
        let first_word = words[first_idx].as_str();
        let max_first = first_word.len().min(query.len().saturating_sub(1));
        for first_len in 1..=max_first {
            if !query.is_char_boundary(first_len) || !first_word.is_char_boundary(first_len) {
                continue;
            }
            if first_word[..first_len] != query[..first_len] {
                continue;
            }
            let rem = &query[first_len..];
            let max_second = (first_idx + 2).min(words.len().saturating_sub(1));
            for (second_idx, second_word_str) in words
                .iter()
                .enumerate()
                .take(max_second + 1)
                .skip(first_idx + 1)
            {
                let second_word = second_word_str.as_str();
                if second_word.starts_with(rem) && (first_len >= 3 || rem.len() >= 3) {
                    let gap = second_idx - first_idx - 1;
                    return Some(110 + first_idx * 4 + gap * 8);
                }
            }
        }
    }
    None
}

/// Checks if `query` matches `text` using case-insensitive word-prefix, contiguous substring,
/// multi-term token, and bounded-gap multi-word prefix scoring (lower score = higher relevance).
pub fn fuzzy_match_score(text: &str, query: &str) -> Option<usize> {
    let trimmed_query = query.trim().to_lowercase();
    if trimmed_query.is_empty() {
        return Some(usize::MAX);
    }

    let text_lower = text.to_lowercase();
    if text_lower == trimmed_query {
        return Some(0);
    }
    if text_lower.starts_with(&trimmed_query) {
        return Some(1);
    }

    let words = split_normalized_words(text);
    if let Some(word_score) = match_word_prefix_score(&words, &trimmed_query) {
        return Some(word_score);
    }
    if let Some(pos) = text_lower.find(&trimmed_query) {
        return Some(10 + pos);
    }
    if let Some(multi_score) = match_multi_term_tokens(&text_lower, &words, &trimmed_query) {
        return Some(multi_score);
    }
    match_bounded_gap_word_prefixes(&words, &trimmed_query)
}
