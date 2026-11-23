use super::DiffSource;
use super::git_range::{read_target_file, unified_zero_context_diff};
use super::is_core_path;
use std::collections::HashMap;
use std::path::Path;

/// Lexical state carried across Rust lines when scanning for non-comment code tokens.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct RustScanState {
    block_comment_depth: usize,
    in_string: bool,
    raw_string_hashes: Option<usize>,
}

fn step_block_comment(bytes: &[u8], idx: usize, depth: &mut usize) -> usize {
    if idx + 1 < bytes.len() && bytes[idx] == b'/' && bytes[idx + 1] == b'*' {
        *depth += 1;
        return idx + 2;
    }
    if idx + 1 < bytes.len() && bytes[idx] == b'*' && bytes[idx + 1] == b'/' {
        *depth -= 1;
        return idx + 2;
    }
    idx + 1
}

fn step_normal_string(bytes: &[u8], idx: usize, in_string: &mut bool) -> usize {
    if bytes[idx] == b'\\' {
        return (idx + 2).min(bytes.len());
    }
    if bytes[idx] == b'"' {
        *in_string = false;
    }
    idx + 1
}

fn step_raw_string(bytes: &[u8], idx: usize, raw_hashes: &mut Option<usize>) -> usize {
    if bytes[idx] == b'"'
        && let Some(needed) = *raw_hashes
    {
        let mut count = 0;
        while idx + 1 + count < bytes.len() && count < needed && bytes[idx + 1 + count] == b'#' {
            count += 1;
        }
        if count == needed {
            *raw_hashes = None;
            return idx + 1 + needed;
        }
    }
    idx + 1
}

fn try_raw_string_prefix(bytes: &[u8], start: usize, end: usize) -> Option<(usize, usize)> {
    let ident = &bytes[start..end];
    if ident != b"r" && ident != b"br" && ident != b"cr" {
        return None;
    }
    let mut hashes = 0;
    while end + hashes < bytes.len() && bytes[end + hashes] == b'#' {
        hashes += 1;
    }
    if end + hashes < bytes.len() && bytes[end + hashes] == b'"' {
        Some((hashes, end + hashes + 1))
    } else {
        None
    }
}

fn step_char_or_lifetime(bytes: &[u8], idx: usize) -> usize {
    if idx + 1 >= bytes.len() {
        return idx + 1;
    }
    if bytes[idx + 1] == b'\\' {
        let limit = (idx + 14).min(bytes.len());
        for (offset, &b) in bytes[idx + 2..limit].iter().enumerate() {
            if b == b'\'' {
                return idx + 2 + offset + 1;
            }
        }
        return idx + 1;
    }
    if let Ok(rest) = std::str::from_utf8(&bytes[idx + 1..])
        && let Some(ch) = rest.chars().next()
        && ch != '\''
        && ch != '\\'
    {
        let close = idx + 1 + ch.len_utf8();
        if close < bytes.len() && bytes[close] == b'\'' {
            return close + 1;
        }
    }
    idx + 1
}

fn step_normal_token(bytes: &[u8], idx: usize, state: &mut RustScanState) -> (usize, Option<bool>) {
    let b = bytes[idx];
    if b.is_ascii_whitespace() {
        return (idx + 1, Some(false));
    }
    if !b.is_ascii() {
        let ch_info = std::str::from_utf8(&bytes[idx..])
            .ok()
            .and_then(|s| s.chars().next().map(|c| (c.len_utf8(), !c.is_whitespace())));
        let (len, is_code) = ch_info.unwrap_or((1, true));
        return (idx + len, Some(is_code));
    }
    if b == b'/' {
        if idx + 1 < bytes.len() && bytes[idx + 1] == b'/' {
            return (bytes.len(), None);
        }
        if idx + 1 < bytes.len() && bytes[idx + 1] == b'*' {
            state.block_comment_depth = 1;
            return (idx + 2, Some(false));
        }
        return (idx + 1, Some(true));
    }
    if b == b'"' {
        state.in_string = true;
        return (idx + 1, Some(true));
    }
    if b == b'\'' {
        return (step_char_or_lifetime(bytes, idx), Some(true));
    }
    if b.is_ascii_alphabetic() || b == b'_' {
        let mut end = idx + 1;
        while end < bytes.len() && (bytes[end].is_ascii_alphanumeric() || bytes[end] == b'_') {
            end += 1;
        }
        if let Some((hashes, next)) = try_raw_string_prefix(bytes, idx, end) {
            state.raw_string_hashes = Some(hashes);
            return (next, Some(true));
        }
        return (end, Some(true));
    }
    (idx + 1, Some(true))
}

/// Scans a single line of Rust source code, updating `state` and returning `true` if any
/// non-whitespace, non-comment code token appears on the line.
pub(crate) fn scan_rust_line(state: &mut RustScanState, line: &str) -> bool {
    let bytes = line.as_bytes();
    let mut idx = 0;
    let mut has_code = false;

    while idx < bytes.len() {
        if state.block_comment_depth > 0 {
            idx = step_block_comment(bytes, idx, &mut state.block_comment_depth);
        } else if state.in_string {
            has_code = true;
            idx = step_normal_string(bytes, idx, &mut state.in_string);
        } else if state.raw_string_hashes.is_some() {
            has_code = true;
            idx = step_raw_string(bytes, idx, &mut state.raw_string_hashes);
        } else {
            let (next, action) = step_normal_token(bytes, idx, state);
            match action {
                None => break,
                Some(is_code) => {
                    if is_code {
                        has_code = true;
                    }
                    idx = next;
                }
            }
        }
    }
    has_code
}

fn parse_hunk_new_start(line: &str) -> Option<usize> {
    let rest = line.strip_prefix("@@ ")?;
    let header_end = rest.find(" @@")?;
    let specs = &rest[..header_end];
    let plus_spec = specs.split_whitespace().find_map(|s| s.strip_prefix('+'))?;
    plus_spec.split(',').next()?.parse::<usize>().ok()
}

fn state_before_line(file_content: Option<&str>, one_based_line: usize) -> RustScanState {
    let mut state = RustScanState::default();
    let Some(content) = file_content else {
        return state;
    };
    for line in content.lines().take(one_based_line.saturating_sub(1)) {
        let _ = scan_rust_line(&mut state, line);
    }
    state
}

fn parse_diff_core_target(line: &str) -> Option<(String, bool)> {
    let path = line.strip_prefix("+++ b/")?.replace('\\', "/");
    is_core_path(&path).then(|| {
        let is_rs = path.ends_with(".rs");
        (path, is_rs)
    })
}

pub(crate) fn count_added_core_code_lines<F>(diff: &str, mut load_file: F) -> usize
where
    F: FnMut(&str) -> Option<String>,
{
    let mut file_cache: HashMap<String, Option<String>> = HashMap::new();
    let mut current_core: Option<(String, bool)> = None;
    let mut scan_state = RustScanState::default();
    let mut in_hunk = false;
    let mut added_code_lines = 0;

    for line in diff.lines() {
        if line.starts_with("diff --git ") {
            current_core = None;
            in_hunk = false;
            scan_state = RustScanState::default();
        } else if line.starts_with("+++ ") {
            current_core = parse_diff_core_target(line);
            in_hunk = false;
        } else if line.starts_with("@@ ") {
            in_hunk = current_core.is_some();
            if let Some((ref path, true)) = current_core {
                let new_start = parse_hunk_new_start(line).unwrap_or(1);
                let cached = file_cache
                    .entry(path.clone())
                    .or_insert_with(|| load_file(path));
                scan_state = state_before_line(cached.as_deref(), new_start);
            } else {
                scan_state = RustScanState::default();
            }
        } else if in_hunk
            && let Some(added) = line.strip_prefix('+')
            && let Some((_, is_rs)) = current_core
        {
            let has_code = if is_rs {
                scan_rust_line(&mut scan_state, added)
            } else {
                !added.trim().is_empty()
            };
            if has_code {
                added_code_lines += 1;
            }
        }
    }
    added_code_lines
}

pub(crate) fn is_pure_deletion_for_source(repo_dir: &Path, source: &DiffSource) -> bool {
    let diff = unified_zero_context_diff(repo_dir, source);
    count_added_core_code_lines(&diff, |path| read_target_file(repo_dir, source, path)) == 0
}
