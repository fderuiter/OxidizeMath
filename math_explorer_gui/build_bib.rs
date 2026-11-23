//! Build-time generator for compact BibTeX citation link metadata (`bib_links.tsv`).
#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use std::collections::HashMap;
use std::fs;
use std::path::Path;

const BIB_LINKS_FILENAME: &str = "bib_links.tsv";

const STOP_WORDS: &[&str] = &[
    "a", "an", "and", "by", "for", "from", "in", "into", "of", "on", "the", "to", "via", "with",
];

/// Precomputes normalized citation link metadata from `../papers/references.bib` into `out_dir`.
pub fn generate_bib_links_tsv(out_dir: &Path) {
    println!("cargo:rerun-if-changed=build_bib.rs");
    let bib_path = Path::new("../papers/references.bib");
    println!("cargo:rerun-if-changed=../papers/references.bib");

    let raw_bib = fs::read_to_string(bib_path).unwrap_or_default();
    let mut tsv = String::new();

    for chunk in raw_bib.split('@').skip(1) {
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
        let year = fields.get("year").map(|y| y.trim()).unwrap_or("");
        let words = content_words(title).join(" ");
        let surname = extract_first_author_surname(author);
        tsv.push_str(&format!("{surname}\t{year}\t{label}\t{url}\t{words}\n"));
    }

    fs::write(out_dir.join(BIB_LINKS_FILENAME), tsv)
        .expect("Failed to write OUT_DIR/bib_links.tsv");
}

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
        let Some((val, next_i)) = read_bib_field_value(&chars, i) else {
            break;
        };
        i = next_i;
        let cleaned = val.replace(['{', '}'], "");
        fields.insert(
            field_name.to_ascii_lowercase(),
            cleaned.split_whitespace().collect::<Vec<_>>().join(" "),
        );
    }
    fields
}

fn read_bib_field_value(chars: &[char], mut i: usize) -> Option<(String, usize)> {
    let len = chars.len();
    if i >= len {
        return None;
    }
    if chars[i] == '{' {
        read_balanced_braces(chars, i + 1)
    } else if chars[i] == '"' {
        let start = i + 1;
        i += 1;
        while i < len && chars[i] != '"' {
            i += 1;
        }
        let raw: String = chars[start..i.min(len)].iter().collect();
        Some((raw, (i + 1).min(len)))
    } else {
        let start = i;
        while i < len && !chars[i].is_whitespace() && chars[i] != ',' && chars[i] != '}' {
            i += 1;
        }
        Some((chars[start..i].iter().collect(), i))
    }
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
