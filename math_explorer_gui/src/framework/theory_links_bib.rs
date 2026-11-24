//! BibTeX, DOI, arXiv, and external reference link parsing for [`super::resolve_citation_reference_links`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::ExternalCitationLink;
use std::collections::{BTreeSet, HashMap};
use std::sync::OnceLock;

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

fn content_words(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(|w| w.to_ascii_lowercase())
        .filter(|w| !STOP_WORDS.contains(&w.as_str()))
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
        if i >= len {
            break;
        }
        let val = if chars[i] == '{' {
            let Some((raw, next_i)) = read_balanced_braces(&chars, i + 1) else {
                break;
            };
            i = next_i;
            raw
        } else if chars[i] == '"' {
            let start = i + 1;
            i += 1;
            while i < len && chars[i] != '"' {
                i += 1;
            }
            let raw: String = chars[start..i.min(len)].iter().collect();
            i = (i + 1).min(len);
            raw
        } else {
            let start = i;
            while i < len && !chars[i].is_whitespace() && chars[i] != ',' && chars[i] != '}' {
                i += 1;
            }
            chars[start..i].iter().collect()
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

fn extract_first_author_surname(author_field: &str) -> String {
    let first_author = author_field.split(" and ").next().unwrap_or("").trim();
    let surname = if let Some((before_comma, _)) = first_author.split_once(',') {
        before_comma.trim()
    } else {
        first_author.split_whitespace().last().unwrap_or("")
    };
    surname
        .chars()
        .filter(|c| c.is_ascii_alphabetic())
        .collect::<String>()
        .to_ascii_lowercase()
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
                norm_title_words: content_words(title),
                first_author: extract_first_author_surname(author),
                year,
                label,
                url,
            });
        }
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
