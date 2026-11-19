/// Advances past a single-quoted GitHub Actions string literal (`'...'` with `''` escapes).
fn skip_single_quoted(bytes: &[u8], start: usize) -> usize {
    let mut idx = start + 1;
    while idx < bytes.len() {
        if bytes[idx] == b'\'' {
            if idx + 1 < bytes.len() && bytes[idx + 1] == b'\'' {
                idx += 2;
                continue;
            }
            return idx + 1;
        }
        idx += 1;
    }
    bytes.len()
}

/// Extracts the next `${{ ... }}` expression body and remaining slice from `input`.
fn next_gha_expression(input: &str) -> Option<(&str, &str)> {
    let start = input.find("${{")?;
    let rest = &input[start + 3..];
    let bytes = rest.as_bytes();
    let mut idx = 0;
    while idx < bytes.len() {
        if bytes[idx] == b'\'' {
            idx = skip_single_quoted(bytes, idx);
        } else if idx + 1 < bytes.len() && bytes[idx] == b'}' && bytes[idx + 1] == b'}' {
            return Some((&rest[..idx], &rest[idx + 2..]));
        } else {
            idx += 1;
        }
    }
    Some((rest, ""))
}

/// Returns `true` if a GitHub Actions expression references the `github` context.
fn references_github_context(expr: &str) -> bool {
    let bytes = expr.as_bytes();
    let mut idx = 0;
    let mut prev_non_ws: Option<u8> = None;

    while idx < bytes.len() {
        let byte = bytes[idx];
        if byte == b'\'' {
            idx = skip_single_quoted(bytes, idx);
            prev_non_ws = Some(b'\'');
        } else if byte.is_ascii_alphabetic() || byte == b'_' {
            let ident_start = idx;
            while idx < bytes.len()
                && (bytes[idx].is_ascii_alphanumeric() || bytes[idx] == b'_' || bytes[idx] == b'-')
            {
                idx += 1;
            }
            if prev_non_ws != Some(b'.') && expr[ident_start..idx].eq_ignore_ascii_case("github") {
                return true;
            }
            prev_non_ws = Some(b'a');
        } else {
            if !byte.is_ascii_whitespace() {
                prev_non_ws = Some(byte);
            }
            idx += 1;
        }
    }

    false
}

/// Finds the first `${{ ... }}` expression in `script` that reads from the `github` context.
pub(super) fn find_github_expression_in_script(script: &str) -> Option<String> {
    let mut remaining = script;
    while let Some((expr, rest)) = next_gha_expression(remaining) {
        if references_github_context(expr) {
            return Some(expr.trim().to_string());
        }
        remaining = rest;
    }
    None
}
