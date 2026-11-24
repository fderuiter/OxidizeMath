use super::BibtexDatabase;
use std::collections::BTreeSet;

/// Extracts deduplicated BibTeX keys cited via `\cite{...}` / `\citep{...}` / `\citet{...}` in `.tex` sources.
#[must_use]
pub fn extract_tex_cite_keys(tex_source: &str) -> Vec<String> {
    let mut keys = BTreeSet::new();
    for raw_line in tex_source.lines() {
        let line = strip_tex_comment(raw_line);
        collect_line_tex_cites(line, &mut keys);
    }
    keys.into_iter().collect()
}

fn strip_tex_comment(line: &str) -> &str {
    let bytes = line.as_bytes();
    for i in 0..bytes.len() {
        if bytes[i] == b'%' {
            let mut backslashes = 0_usize;
            let mut j = i;
            while j > 0 && bytes[j - 1] == b'\\' {
                backslashes += 1;
                j -= 1;
            }
            if backslashes.is_multiple_of(2) {
                return &line[..i];
            }
        }
    }
    line
}

fn collect_line_tex_cites(line: &str, out: &mut BTreeSet<String>) {
    let bytes = line.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] != b'\\' {
            i += 1;
            continue;
        }
        let cmd_start = i + 1;
        let mut cmd_end = cmd_start;
        while cmd_end < bytes.len() && bytes[cmd_end].is_ascii_alphabetic() {
            cmd_end += 1;
        }
        let cmd = &line[cmd_start..cmd_end];
        if !(cmd.starts_with("cite") || cmd == "nocite") {
            i = cmd_end.max(i + 1);
            continue;
        }
        let mut cursor = skip_cite_options(bytes, cmd_end);
        if cursor < bytes.len()
            && bytes[cursor] == b'{'
            && let Some(close) = line[cursor + 1..].find('}')
        {
            for raw in line[cursor + 1..cursor + 1 + close].split(',') {
                let k = raw.trim();
                if !k.is_empty() && k != "*" {
                    out.insert(k.to_string());
                }
            }
            i = cursor + 1 + close + 1;
            continue;
        }
        cursor = cursor.max(i + 1);
        i = cursor;
    }
}

fn skip_cite_options(bytes: &[u8], mut cursor: usize) -> usize {
    while cursor < bytes.len() && (bytes[cursor].is_ascii_whitespace() || bytes[cursor] == b'*') {
        cursor += 1;
    }
    while cursor < bytes.len() && bytes[cursor] == b'[' {
        while cursor < bytes.len() && bytes[cursor] != b']' {
            cursor += 1;
        }
        cursor = (cursor + 1).min(bytes.len());
    }
    cursor
}

/// Extracts BibTeX keys referenced in Markdown specs or Rust docstrings (`@<key>` in backticks or `bib:<key>`).
#[must_use]
pub fn extract_doc_bib_keys(text: &str) -> Vec<String> {
    let mut seen = BTreeSet::new();
    let mut ordered = Vec::new();

    let mut search = text;
    while let Some(pos) = search.find("`@") {
        let rest = &search[pos + 2..];
        if let Some(end) = rest.find('`') {
            let candidate = rest[..end].trim();
            if !candidate.is_empty() && seen.insert(candidate.to_string()) {
                ordered.push(candidate.to_string());
            }
            search = &rest[end + 1..];
        } else {
            break;
        }
    }

    let mut idx = 0;
    while let Some(rel) = text[idx..].find("bib:") {
        let start = idx + rel + 4;
        let prev_ok = idx + rel == 0
            || !text.as_bytes()[idx + rel - 1].is_ascii_alphanumeric()
                && text.as_bytes()[idx + rel - 1] != b'_';
        let end = start
            + text[start..]
                .bytes()
                .take_while(|b| {
                    b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b':' | b'.' | b'+')
                })
                .count();
        let candidate = text[start..end].trim_end_matches('.');
        if prev_ok && !candidate.is_empty() && seen.insert(candidate.to_string()) {
            ordered.push(candidate.to_string());
        }
        idx = end.max(start);
    }

    ordered
}

/// Extracts `spec:<id>` and `docs/math_specs/<id>.md` identifiers from Markdown or Rust docstrings.
#[must_use]
pub fn extract_doc_spec_ids(text: &str) -> Vec<String> {
    let mut seen = BTreeSet::new();
    let mut ordered = Vec::new();

    let mut idx = 0;
    while let Some(rel) = text[idx..].find("spec:") {
        let start = idx + rel + 5;
        let prev_ok = idx + rel == 0
            || !text.as_bytes()[idx + rel - 1].is_ascii_alphanumeric()
                && text.as_bytes()[idx + rel - 1] != b'_';
        let end = start
            + text[start..]
                .bytes()
                .take_while(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-'))
                .count();
        let candidate = &text[start..end];
        if prev_ok && !candidate.is_empty() && seen.insert(candidate.to_string()) {
            ordered.push(candidate.to_string());
        }
        idx = end.max(start);
    }

    let mut path_idx = 0;
    while let Some(rel) = text[path_idx..].find("docs/math_specs/") {
        let start = path_idx + rel + "docs/math_specs/".len();
        let end = start
            + text[start..]
                .bytes()
                .take_while(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.'))
                .count();
        let file = text[start..end].trim_end_matches('.');
        if let Some(stem) = file.strip_suffix(".md")
            && !stem.is_empty()
            && stem != "README"
            && seen.insert(stem.to_string())
        {
            ordered.push(stem.to_string());
        }
        path_idx = end.max(start);
    }

    ordered
}

/// Validates a `docs/math_specs/<spec_id>.md` file for equations, stability, validity, and persistent BibTeX IDs.
pub fn validate_math_spec_markdown(
    spec_id: &str,
    markdown: &str,
    bib: &BibtexDatabase,
) -> Result<Vec<String>, String> {
    let lower = markdown.to_ascii_lowercase();
    if !(markdown.contains("$$") || markdown.contains('$')) {
        return Err(format!(
            "`docs/math_specs/{spec_id}.md` is missing a LaTeX governing equation (`$$ ... $$`)"
        ));
    }
    if !(lower.contains("stability") || lower.contains("error bound")) {
        return Err(format!(
            "`docs/math_specs/{spec_id}.md` is missing numerical stability / error bounds"
        ));
    }
    if !(lower.contains("validity") || lower.contains("precondition")) {
        return Err(format!(
            "`docs/math_specs/{spec_id}.md` is missing a domain of validity section"
        ));
    }
    let bib_keys = extract_doc_bib_keys(markdown);
    if bib_keys.is_empty() {
        return Err(format!(
            "`docs/math_specs/{spec_id}.md` must cite at least one BibTeX key (`@<key>` or `bib:<key>`)"
        ));
    }
    for key in &bib_keys {
        bib.verify_cited_key(key)
            .map_err(|err| format!("`docs/math_specs/{spec_id}.md`: {err}"))?;
    }
    Ok(bib_keys)
}
