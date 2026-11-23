//! Pure-`std` Cargo `--message-format=json` WebAssembly test artifact extraction.

use super::WorkspaceError;
use std::path::PathBuf;

fn decode_json_string_at(line: &str, start_quote: usize) -> Option<(String, usize)> {
    let bytes = line.as_bytes();
    if bytes.get(start_quote) != Some(&b'"') {
        return None;
    }
    let mut out = String::new();
    let mut chars = line[start_quote + 1..].char_indices();
    while let Some((rel_idx, ch)) = chars.next() {
        if ch == '"' {
            return Some((out, start_quote + 1 + rel_idx + 1));
        }
        if ch != '\\' {
            out.push(ch);
            continue;
        }
        let (_, esc) = chars.next()?;
        match esc {
            '"' | '\\' | '/' => out.push(esc),
            'b' => out.push('\u{0008}'),
            'f' => out.push('\u{000C}'),
            'n' => out.push('\n'),
            'r' => out.push('\r'),
            't' => out.push('\t'),
            'u' => {
                let mut hex = String::with_capacity(4);
                for _ in 0..4 {
                    hex.push(chars.next()?.1);
                }
                let code = u32::from_str_radix(&hex, 16).ok()?;
                out.push(char::from_u32(code)?);
            }
            _ => return None,
        }
    }
    None
}

fn extract_top_level_json_string_field(line: &str, target_key: &str) -> Option<String> {
    let bytes = line.as_bytes();
    let mut depth = 0usize;
    let mut idx = 0usize;
    while idx < bytes.len() {
        match bytes[idx] {
            b'{' | b'[' => {
                depth += 1;
                idx += 1;
            }
            b'}' | b']' => {
                depth = depth.saturating_sub(1);
                idx += 1;
            }
            b'"' => {
                let (decoded, next_idx) = decode_json_string_at(line, idx)?;
                idx = next_idx;
                if depth == 1 && decoded == target_key {
                    while idx < bytes.len() && bytes[idx].is_ascii_whitespace() {
                        idx += 1;
                    }
                    if bytes.get(idx) == Some(&b':') {
                        idx += 1;
                        while idx < bytes.len() && bytes[idx].is_ascii_whitespace() {
                            idx += 1;
                        }
                        if bytes.get(idx) == Some(&b'"') {
                            let (val, _) = decode_json_string_at(line, idx)?;
                            return Some(val);
                        }
                        return None;
                    }
                }
            }
            _ => idx += 1,
        }
    }
    None
}

fn line_has_test_profile_true(line: &str) -> bool {
    let Some(prof_pos) = line.find("\"profile\"") else {
        return false;
    };
    let tail = &line[prof_pos..];
    let Some(open) = tail.find('{') else {
        return false;
    };
    let Some(close) = tail[open..].find('}') else {
        return false;
    };
    let profile_obj = &tail[open..=open + close];
    let compact: String = profile_obj
        .chars()
        .filter(|c| !c.is_ascii_whitespace())
        .collect();
    compact.contains("\"test\":true")
}

/// Extracts compiled `.wasm` test executable paths from `cargo test --no-run --message-format=json` stdout.
pub fn extract_wasm_test_executables_from_cargo_json(
    stdout: &str,
) -> Result<Vec<PathBuf>, WorkspaceError> {
    let mut artifacts = Vec::new();
    for raw_line in stdout.lines() {
        let line = raw_line.trim();
        if !line.starts_with('{') {
            continue;
        }
        let Some(reason) = extract_top_level_json_string_field(line, "reason") else {
            continue;
        };
        if reason != "compiler-artifact" || !line_has_test_profile_true(line) {
            continue;
        }
        if let Some(exe) = extract_top_level_json_string_field(line, "executable")
            && exe.ends_with(".wasm")
        {
            let path = PathBuf::from(exe);
            if !artifacts.contains(&path) {
                artifacts.push(path);
            }
        }
    }
    if artifacts.is_empty() {
        return Err(WorkspaceError::NoWasmTestArtifacts);
    }
    Ok(artifacts)
}
