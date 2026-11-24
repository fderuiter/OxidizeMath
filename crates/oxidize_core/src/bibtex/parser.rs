use std::collections::BTreeMap;

/// A single parsed `@<entry_type>{<key>, ...}` record from a `.bib` bibliography.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BibtexEntry {
    /// Normalized lowercase entry type (e.g. `article`, `book`, `inproceedings`, `misc`).
    pub entry_type: String,
    /// Citation key immediately following `{`.
    pub key: String,
    /// 1-indexed line number in the `.bib` source where `@` appears.
    pub line: usize,
    /// Normalized lowercase field names mapped to brace-unwrapped field values.
    pub fields: BTreeMap<String, String>,
}

impl BibtexEntry {
    /// Returns the trimmed `doi` field value when present and non-empty.
    #[must_use]
    pub fn doi(&self) -> Option<&str> {
        self.fields
            .get("doi")
            .map(String::as_str)
            .map(str::trim)
            .filter(|v| !v.is_empty())
    }

    /// Returns the trimmed `eprint` field value when present and non-empty.
    #[must_use]
    pub fn eprint(&self) -> Option<&str> {
        self.fields
            .get("eprint")
            .map(String::as_str)
            .map(str::trim)
            .filter(|v| !v.is_empty())
    }

    /// Returns `true` when the entry defines a non-empty `doi` or `eprint` persistent identifier.
    #[must_use]
    pub fn has_persistent_id(&self) -> bool {
        self.doi().is_some() || self.eprint().is_some()
    }
}

/// Parsed bibliography index keyed by BibTeX citation identifier.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BibtexDatabase {
    entries: BTreeMap<String, BibtexEntry>,
}

impl BibtexDatabase {
    /// Parses a `.bib` file into a [`BibtexDatabase`], rejecting malformed braces or duplicate keys.
    pub fn parse(source: &str) -> Result<Self, String> {
        let bytes = source.as_bytes();
        let mut idx = 0;
        let mut line = 1_usize;
        let mut entries = BTreeMap::new();

        while idx < bytes.len() {
            if bytes[idx] == b'%' {
                while idx < bytes.len() && bytes[idx] != b'\n' {
                    idx += 1;
                }
                continue;
            }
            if bytes[idx] != b'@' {
                if bytes[idx] == b'\n' {
                    line += 1;
                }
                idx += 1;
                continue;
            }

            let entry_line = line;
            idx += 1;
            let (entry_type, body, next_idx, next_line) =
                read_at_block(source, idx, line, entry_line)?;
            idx = next_idx;
            line = next_line;

            if matches!(entry_type.as_str(), "comment" | "preamble" | "string") {
                continue;
            }
            let entry = parse_entry_record(entry_type, body, entry_line)?;
            if entries.contains_key(&entry.key) {
                return Err(format!(
                    "Duplicate BibTeX key `{}` at line {}",
                    entry.key, entry.line
                ));
            }
            entries.insert(entry.key.clone(), entry);
        }

        Ok(Self { entries })
    }

    /// Looks up a parsed [`BibtexEntry`] by exact citation key.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&BibtexEntry> {
        self.entries().get(key)
    }

    /// Returns the map of all parsed [`BibtexEntry`] records.
    #[must_use]
    pub fn entries(&self) -> &BTreeMap<String, BibtexEntry> {
        &self.entries
    }

    /// Verifies that `key` is syntactically valid, exists in the database, and defines `doi` or `eprint`.
    pub fn verify_cited_key(&self, key: &str) -> Result<&BibtexEntry, String> {
        let trimmed = key.trim();
        if !is_valid_bibtex_key(trimmed) {
            return Err(format!(
                "Invalid BibTeX key `{trimmed}`: expected compact citation key without whitespace"
            ));
        }
        let Some(entry) = self.get(trimmed) else {
            return Err(format!(
                "BibTeX key `{trimmed}` not found in `papers/references.bib`"
            ));
        };
        if entry.entry_type.is_empty() || entry.key.is_empty() || !entry.has_persistent_id() {
            return Err(format!(
                "BibTeX entry `{trimmed}` (line {}) must define a non-empty `doi` or `eprint` field",
                entry.line
            ));
        }
        Ok(entry)
    }
}

fn read_at_block(
    source: &str,
    mut idx: usize,
    mut line: usize,
    entry_line: usize,
) -> Result<(String, &str, usize, usize), String> {
    let bytes = source.as_bytes();
    let type_start = idx;
    while idx < bytes.len() && (bytes[idx].is_ascii_alphanumeric() || bytes[idx] == b'_') {
        idx += 1;
    }
    let entry_type = source[type_start..idx].trim().to_ascii_lowercase();
    if entry_type.is_empty() {
        return Err(format!("Missing BibTeX entry type at line {entry_line}"));
    }
    while idx < bytes.len() && bytes[idx].is_ascii_whitespace() {
        if bytes[idx] == b'\n' {
            line += 1;
        }
        idx += 1;
    }
    if idx >= bytes.len() || (bytes[idx] != b'{' && bytes[idx] != b'(') {
        return Err(format!(
            "Expected `{{` after `@{entry_type}` at line {entry_line}"
        ));
    }
    let close_delim = if bytes[idx] == b'{' { b'}' } else { b')' };
    scan_entry_body(source, idx + 1, line, entry_line, entry_type, close_delim)
}

fn scan_entry_body(
    source: &str,
    mut idx: usize,
    mut line: usize,
    entry_line: usize,
    entry_type: String,
    close_delim: u8,
) -> Result<(String, &str, usize, usize), String> {
    let bytes = source.as_bytes();
    let body_start = idx;
    let mut brace_depth = 1_usize;
    let mut in_quotes = false;

    while idx < bytes.len() {
        let b = bytes[idx];
        if b == b'\n' {
            line += 1;
        } else if b == b'\\' {
            idx += 1;
            if idx < bytes.len() && bytes[idx] == b'\n' {
                line += 1;
            }
        } else if b == b'"' && brace_depth == 1 {
            in_quotes = !in_quotes;
        } else if !in_quotes {
            if b == b'{' {
                brace_depth += 1;
            } else if b == b'}' && close_delim == b'}' {
                brace_depth -= 1;
                if brace_depth == 0 {
                    return Ok((entry_type, &source[body_start..idx], idx + 1, line));
                }
            } else if b == b')' && close_delim == b')' && brace_depth == 1 {
                return Ok((entry_type, &source[body_start..idx], idx + 1, line));
            }
        }
        idx += 1;
    }
    Err(format!(
        "Unclosed BibTeX entry `@{entry_type}` starting at line {entry_line}"
    ))
}

fn parse_entry_record(
    entry_type: String,
    body: &str,
    entry_line: usize,
) -> Result<BibtexEntry, String> {
    let Some((raw_key, fields_src)) = body.split_once(',') else {
        return Err(format!(
            "Missing citation key separator `,` in `@{entry_type}` at line {entry_line}"
        ));
    };
    let key = raw_key.trim().to_string();
    if !is_valid_bibtex_key(&key) {
        return Err(format!(
            "Invalid BibTeX citation key `{key}` at line {entry_line}"
        ));
    }
    let fields = parse_entry_fields(fields_src)?;
    Ok(BibtexEntry {
        entry_type,
        key,
        line: entry_line,
        fields,
    })
}

fn parse_entry_fields(src: &str) -> Result<BTreeMap<String, String>, String> {
    let bytes = src.as_bytes();
    let mut idx = 0;
    let mut fields = BTreeMap::new();

    while idx < bytes.len() {
        while idx < bytes.len() && (bytes[idx].is_ascii_whitespace() || bytes[idx] == b',') {
            idx += 1;
        }
        if idx >= bytes.len() {
            break;
        }
        if bytes[idx] == b'%' {
            while idx < bytes.len() && bytes[idx] != b'\n' {
                idx += 1;
            }
            continue;
        }
        let name_start = idx;
        while idx < bytes.len()
            && (bytes[idx].is_ascii_alphanumeric() || matches!(bytes[idx], b'_' | b'-' | b':'))
        {
            idx += 1;
        }
        if name_start == idx {
            idx += 1;
            continue;
        }
        let field_name = src[name_start..idx].to_ascii_lowercase();
        while idx < bytes.len() && bytes[idx].is_ascii_whitespace() {
            idx += 1;
        }
        if idx >= bytes.len() || bytes[idx] != b'=' {
            continue;
        }
        let (val, next_idx) = parse_field_value(src, idx + 1)?;
        idx = next_idx;
        fields.insert(field_name, normalize_bib_value(&val));
    }
    Ok(fields)
}

fn parse_field_value(src: &str, mut idx: usize) -> Result<(String, usize), String> {
    let bytes = src.as_bytes();
    while idx < bytes.len() && bytes[idx].is_ascii_whitespace() {
        idx += 1;
    }
    if idx >= bytes.len() {
        return Ok((String::new(), idx));
    }
    if bytes[idx] == b'{' {
        return read_braced_value(src, idx + 1);
    }
    if bytes[idx] == b'"' {
        return read_quoted_value(src, idx + 1);
    }
    let start = idx;
    while idx < bytes.len() && bytes[idx] != b',' && !bytes[idx].is_ascii_whitespace() {
        idx += 1;
    }
    Ok((src[start..idx].to_string(), idx))
}

fn read_braced_value(src: &str, mut idx: usize) -> Result<(String, usize), String> {
    let bytes = src.as_bytes();
    let start = idx;
    let mut depth = 1_usize;
    while idx < bytes.len() {
        if bytes[idx] == b'\\' {
            idx += 2;
            continue;
        }
        if bytes[idx] == b'{' {
            depth += 1;
        } else if bytes[idx] == b'}' {
            depth -= 1;
            if depth == 0 {
                return Ok((src[start..idx].to_string(), idx + 1));
            }
        }
        idx += 1;
    }
    Err("Unclosed `{` in BibTeX field value".to_string())
}

fn read_quoted_value(src: &str, mut idx: usize) -> Result<(String, usize), String> {
    let bytes = src.as_bytes();
    let start = idx;
    let mut brace_depth = 0_usize;
    while idx < bytes.len() {
        if bytes[idx] == b'\\' {
            idx += 2;
            continue;
        }
        if bytes[idx] == b'{' {
            brace_depth += 1;
        } else if bytes[idx] == b'}' {
            brace_depth = brace_depth.saturating_sub(1);
        } else if bytes[idx] == b'"' && brace_depth == 0 {
            return Ok((src[start..idx].to_string(), idx + 1));
        }
        idx += 1;
    }
    Err("Unclosed `\"` in BibTeX field value".to_string())
}

fn normalize_bib_value(raw: &str) -> String {
    let stripped: String = raw.chars().filter(|&c| c != '{' && c != '}').collect();
    stripped.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Checks whether `key` is a syntactically valid BibTeX citation identifier without spaces.
#[must_use]
pub fn is_valid_bibtex_key(key: &str) -> bool {
    !key.is_empty()
        && key.bytes().any(|b| b.is_ascii_alphabetic())
        && key.bytes().all(|b| {
            b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b':' | b'.' | b'+' | b'/')
        })
}
