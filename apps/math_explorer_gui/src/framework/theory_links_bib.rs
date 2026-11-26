//! BibTeX, DOI, arXiv, and external reference link parsing for [`super::resolve_citation_reference_links`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::ExternalCitationLink;
use std::collections::{BTreeSet, HashMap};
use std::sync::OnceLock;

#[derive(Clone, Debug)]
struct BibEntry {
    key: String,
    norm_title_words: Vec<String>,
    authors: Vec<String>,
    year: String,
    ambiguous_author_year: bool,
    unique_coauthors: Vec<String>,
    unique_title_words: Vec<String>,
    label: String,
    url: String,
}

const STOP_WORDS: &[&str] = &[
    "a", "an", "and", "by", "for", "from", "in", "into", "of", "on", "the", "to", "via", "with",
];

fn fold_ascii_char(ch: char, out: &mut String) {
    match ch {
        'ä' | 'á' | 'à' | 'â' | 'ã' | 'å' | 'Ä' | 'Á' | 'À' | 'Â' | 'Ã' | 'Å' => {
            out.push('a')
        }
        'ë' | 'é' | 'è' | 'ê' | 'Ë' | 'É' | 'È' | 'Ê' => out.push('e'),
        'ï' | 'í' | 'ì' | 'î' | 'Ï' | 'Í' | 'Ì' | 'Î' => out.push('i'),
        'ö' | 'ó' | 'ò' | 'ô' | 'õ' | 'ø' | 'ő' | 'Ö' | 'Ó' | 'Ò' | 'Ô' | 'Õ' | 'Ø' | 'Ő' =>
        {
            out.push('o');
        }
        'ü' | 'ú' | 'ù' | 'û' | 'ű' | 'Ü' | 'Ú' | 'Ù' | 'Û' | 'Ű' => out.push('u'),
        'ç' | 'ć' | 'č' | 'Ç' | 'Ć' | 'Č' => out.push('c'),
        'ñ' | 'ń' | 'Ñ' | 'Ń' => out.push('n'),
        'ł' | 'Ł' => out.push('l'),
        'ś' | 'š' | 'Ś' | 'Š' => out.push('s'),
        'ź' | 'ż' | 'ž' | 'Ź' | 'Ż' | 'Ž' => out.push('z'),
        'ß' => out.push_str("ss"),
        other => out.push(other),
    }
}

fn fold_bibtex_accents(text: &str) -> String {
    let stripped = text
        .replace("\\\"", "")
        .replace("\\'", "")
        .replace("\\`", "")
        .replace("\\^", "")
        .replace("\\~", "")
        .replace("\\c", "")
        .replace("\\l", "l");
    let mut out = String::with_capacity(stripped.len());
    for ch in stripped.chars() {
        fold_ascii_char(ch, &mut out);
    }
    out
}

fn content_words(text: &str) -> Vec<String> {
    let folded = fold_bibtex_accents(text);
    folded
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_ascii_lowercase)
        .filter(|w| !STOP_WORDS.contains(&w.as_str()))
        .collect()
}

fn clause_tokens(clause: &str) -> Vec<String> {
    let folded = fold_bibtex_accents(clause);
    let mut tokens = content_words(&folded);
    for raw in folded.split_whitespace() {
        let joined: String = raw
            .chars()
            .filter(|c| c.is_ascii_alphanumeric())
            .collect::<String>()
            .to_ascii_lowercase();
        if joined.len() >= 3 && !tokens.contains(&joined) {
            tokens.push(joined);
        }
    }
    tokens
}

fn raw_key_tokens(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_ascii_alphanumeric() && c != '_' && c != '-')
        .filter(|w| w.len() >= 4)
        .map(str::to_ascii_lowercase)
        .collect()
}

fn read_balanced_braces(chars: &[char], mut idx: usize) -> Option<(String, usize)> {
    let mut depth = 1_usize;
    let start = idx;
    while idx < chars.len() {
        if chars[idx] == '{' {
            depth += 1;
        } else if chars[idx] == '}' {
            depth -= 1;
            if depth == 0 {
                let inner: String = chars[start..idx].iter().collect();
                return Some((inner, idx + 1));
            }
        }
        idx += 1;
    }
    None
}

fn read_bib_field_value(chars: &[char], i: &mut usize) -> Option<String> {
    let len = chars.len();
    if *i >= len {
        return None;
    }
    if chars[*i] == '{' {
        let (raw, next_i) = read_balanced_braces(chars, *i + 1)?;
        *i = next_i;
        Some(raw)
    } else if chars[*i] == '"' {
        let start = *i + 1;
        *i += 1;
        while *i < len && chars[*i] != '"' {
            *i += 1;
        }
        let raw: String = chars[start..(*i).min(len)].iter().collect();
        *i = (*i + 1).min(len);
        Some(raw)
    } else {
        let start = *i;
        while *i < len && !chars[*i].is_whitespace() && chars[*i] != ',' && chars[*i] != '}' {
            *i += 1;
        }
        Some(chars[start..*i].iter().collect())
    }
}

fn parse_bib_fields(body: &str) -> HashMap<String, String> {
    let chars: Vec<char> = body.chars().collect();
    let (mut i, len) = (0, chars.len());
    let mut fields = HashMap::new();
    while i < len {
        while i < len && !chars[i].is_ascii_alphabetic() {
            i += 1;
        }
        let key_start = i;
        while i < len && (chars[i].is_ascii_alphanumeric() || chars[i] == '_' || chars[i] == '-') {
            i += 1;
        }
        if key_start == i {
            break;
        }
        let field_name: String = chars[key_start..i].iter().collect();
        while i < len && chars[i].is_whitespace() {
            i += 1;
        }
        if i >= len || chars[i] != '=' {
            continue;
        }
        i += 1;
        while i < len && chars[i].is_whitespace() {
            i += 1;
        }
        let Some(val) = read_bib_field_value(&chars, &mut i) else {
            break;
        };
        let cleaned = val.replace(['{', '}'], "");
        fields.insert(
            field_name.to_ascii_lowercase(),
            cleaned.split_whitespace().collect::<Vec<_>>().join(" "),
        );
    }
    fields
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

fn resolve_bib_url(key: &str, fields: &HashMap<String, String>) -> Option<(String, String)> {
    if let Some(raw_doi) = fields.get("doi") {
        let doi = raw_doi
            .trim()
            .trim_start_matches("https://doi.org/")
            .trim_start_matches("http://doi.org/")
            .trim();
        if !doi.is_empty() {
            return Some((format!("DOI: {doi}"), format!("https://doi.org/{doi}")));
        }
    }
    if let Some(eprint) = fields.get("eprint") {
        let id = eprint.trim().trim_start_matches("arXiv:").trim();
        if !id.is_empty() {
            return Some((format!("arXiv:{id}"), format!("https://arxiv.org/abs/{id}")));
        }
    }
    for field in ["journal", "howpublished", "note", "pages", "url"] {
        if let Some(val) = fields.get(field) {
            if let Some(id) = extract_arxiv_id(val) {
                return Some((format!("arXiv:{id}"), format!("https://arxiv.org/abs/{id}")));
            }
            if let Some(url) = extract_http_url(val) {
                return Some((format!("Reference ({key})"), url));
            }
        }
    }
    None
}

fn extract_author_surnames(author_field: &str) -> Vec<String> {
    let folded = fold_bibtex_accents(author_field);
    let mut out = Vec::new();
    for raw_person in folded.split(" and ") {
        let person = raw_person.trim();
        if person.is_empty() || person.eq_ignore_ascii_case("others") {
            continue;
        }
        let raw_surname = if let Some((before_comma, _)) = person.split_once(',') {
            before_comma.split_whitespace().last().unwrap_or("")
        } else {
            person.split_whitespace().last().unwrap_or("")
        };
        let norm: String = raw_surname
            .chars()
            .filter(|c| c.is_ascii_alphabetic())
            .collect::<String>()
            .to_ascii_lowercase();
        if !norm.is_empty() && !out.contains(&norm) {
            out.push(norm);
        }
    }
    out
}

fn populate_disambiguation(entries: &mut [BibEntry]) {
    let mut groups: HashMap<(String, String), Vec<usize>> = HashMap::new();
    for (idx, entry) in entries.iter().enumerate() {
        if let Some(first) = entry.authors.first() {
            if !first.is_empty() && entry.year.len() == 4 {
                groups
                    .entry((first.clone(), entry.year.clone()))
                    .or_default()
                    .push(idx);
            }
        }
    }
    for indices in groups.values() {
        if indices.len() < 2 {
            continue;
        }
        for &idx in indices {
            let coauthors: Vec<String> = entries[idx]
                .authors
                .iter()
                .skip(1)
                .filter(|ca| {
                    indices
                        .iter()
                        .all(|&other| other == idx || !entries[other].authors.contains(ca))
                })
                .cloned()
                .collect();
            let title_words: Vec<String> = entries[idx]
                .norm_title_words
                .iter()
                .filter(|tw| {
                    tw.len() >= 4
                        && indices.iter().all(|&other| {
                            other == idx || !entries[other].norm_title_words.contains(tw)
                        })
                })
                .cloned()
                .collect();
            entries[idx].ambiguous_author_year = true;
            entries[idx].unique_coauthors = coauthors;
            entries[idx].unique_title_words = title_words;
        }
    }
}

fn embedded_bib_entries() -> &'static [BibEntry] {
    static BIB_CACHE: OnceLock<Vec<BibEntry>> = OnceLock::new();
    BIB_CACHE.get_or_init(|| {
        const RAW_BIB: &str = include_str!(concat!(env!("OUT_DIR"), "/references.bib"));
        let mut entries = Vec::new();
        for chunk in RAW_BIB.split('@').skip(1) {
            let Some((header, rest)) = chunk.split_once('{') else {
                continue;
            };
            if header.trim().eq_ignore_ascii_case("comment") {
                continue;
            }
            let Some((key, body)) = rest.split_once(',') else {
                continue;
            };
            let key = key.trim();
            let fields = parse_bib_fields(body);
            let Some((label, url)) = resolve_bib_url(key, &fields) else {
                continue;
            };
            let title = fields.get("title").map(String::as_str).unwrap_or("");
            let author = fields.get("author").map(String::as_str).unwrap_or("");
            let year = fields
                .get("year")
                .map(|y| y.trim().to_string())
                .unwrap_or_default();
            entries.push(BibEntry {
                key: key.to_ascii_lowercase(),
                norm_title_words: content_words(title),
                authors: extract_author_surnames(author),
                year,
                ambiguous_author_year: false,
                unique_coauthors: Vec::new(),
                unique_title_words: Vec::new(),
                label,
                url,
            });
        }
        populate_disambiguation(&mut entries);
        entries
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
    let mut search_from = 0_usize;
    while let Some(rel) = citation[search_from..].find("10.") {
        let idx = search_from + rel;
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
        search_from = idx + candidate.len().max(3);
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

fn clause_matches_author_year(entry: &BibEntry, clause: &[String]) -> bool {
    let Some(first) = entry.authors.first() else {
        return false;
    };
    if entry.year.len() != 4 || !clause.contains(&entry.year) || !clause.contains(first) {
        return false;
    }
    let short_surname_ok = first.len() >= 3
        || entry.authors.iter().skip(1).any(|ca| clause.contains(ca))
        || entry
            .norm_title_words
            .iter()
            .any(|tw| tw.len() >= 4 && clause.contains(tw));
    if !short_surname_ok {
        return false;
    }
    if !entry.ambiguous_author_year {
        return true;
    }
    entry.unique_coauthors.iter().any(|ca| clause.contains(ca))
        || entry
            .unique_title_words
            .iter()
            .any(|tw| clause.contains(tw))
}

fn entry_matches_citation(
    entry: &BibEntry,
    key_tokens: &[String],
    cit_words: &[String],
    clauses: &[Vec<String>],
) -> bool {
    let explicit_key = entry.key.len() >= 5
        && (entry.key.chars().any(|c| c.is_ascii_digit())
            || entry.key.contains('_')
            || entry.key.contains('-'))
        && key_tokens.contains(&entry.key);
    if explicit_key || contains_word_subsequence(cit_words, &entry.norm_title_words) {
        return true;
    }
    clauses
        .iter()
        .any(|clause| clause_matches_author_year(entry, clause))
}

/// Matches a citation string against `papers/references.bib` DOI, arXiv, and URL entries.
pub(crate) fn resolve_citation_reference_links(citation: &str) -> Vec<ExternalCitationLink> {
    if citation.trim().is_empty() {
        return Vec::new();
    }
    let key_tokens = raw_key_tokens(citation);
    let cit_words = content_words(citation);
    let clauses: Vec<Vec<String>> = citation.split(';').map(clause_tokens).collect();
    let mut seen_urls = BTreeSet::new();
    let mut out = Vec::new();

    for link in extract_inline_citation_links(citation) {
        if seen_urls.insert(link.url.clone()) {
            out.push(link);
        }
    }
    for entry in embedded_bib_entries() {
        if entry_matches_citation(entry, &key_tokens, &cit_words, &clauses)
            && seen_urls.insert(entry.url.clone())
        {
            out.push(ExternalCitationLink {
                label: entry.label.clone(),
                url: entry.url.clone(),
            });
        }
    }
    out
}
