//! AST/token-aware `#[cfg(test)]` stripper and `Color32` regression scanner for
//! `theme_palette_tests.rs` and `canvas_overlay_tests.rs`.

pub(crate) fn find_constructor_lines(
    stripped: &str,
    needle: &str,
    literal_only: bool,
) -> Vec<usize> {
    let mut lines = Vec::new();
    let mut offset = 0;
    while let Some(pos) = stripped[offset..].find(needle) {
        let abs_pos = offset + pos;
        let after = &stripped[abs_pos + needle.len()..];
        let first_arg_ch = after.chars().find(|c| !c.is_whitespace());
        let matched = if literal_only {
            first_arg_ch.is_some_and(|c| c.is_ascii_digit())
        } else {
            first_arg_ch.is_some()
        };
        if matched {
            let line_no = stripped[..abs_pos].bytes().filter(|&b| b == b'\n').count() + 1;
            lines.push(line_no);
        }
        offset = abs_pos + needle.len();
    }
    lines
}

fn blank_preserving_newlines(src: &str, out: &mut String) {
    for ch in src.chars() {
        out.push(if ch == '\n' { '\n' } else { ' ' });
    }
}

fn token_end(bytes: &[u8], start: usize) -> usize {
    let n = bytes.len();
    if start + 1 < n && bytes[start] == b'/' && bytes[start + 1] == b'/' {
        let mut j = start + 2;
        while j < n && bytes[j] != b'\n' {
            j += 1;
        }
        return j;
    }
    if start + 1 < n && bytes[start] == b'/' && bytes[start + 1] == b'*' {
        let (mut j, mut depth) = (start + 2, 1_usize);
        while j < n && depth > 0 {
            if j + 1 < n && bytes[j] == b'/' && bytes[j + 1] == b'*' {
                depth += 1;
                j += 2;
            } else if j + 1 < n && bytes[j] == b'*' && bytes[j + 1] == b'/' {
                depth -= 1;
                j += 2;
            } else {
                j += 1;
            }
        }
        return j;
    }
    if bytes[start] == b'"' {
        let mut j = start + 1;
        while j < n {
            if bytes[j] == b'\\' {
                j = (j + 2).min(n);
            } else if bytes[j] == b'"' {
                return j + 1;
            } else {
                j += 1;
            }
        }
        return n;
    }
    start + 1
}

fn is_cfg_test_attr(src: &str) -> Option<usize> {
    let rest = src.strip_prefix('#')?;
    let trimmed = rest.trim_start();
    let ws1 = rest.len() - trimmed.len();
    let inside = trimmed.strip_prefix('[')?;
    let close = inside.find(']')?;
    let compact: String = inside[..close]
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect();
    (compact == "cfg(test)").then_some(1 + ws1 + 1 + close + 1)
}

fn skip_annotated_item_end(src: &str, after_attr: usize) -> usize {
    let bytes = src.as_bytes();
    let n = bytes.len();
    let mut i = after_attr;
    while i < n {
        if bytes[i].is_ascii_whitespace() {
            i += 1;
        } else if (i + 1 < n && bytes[i] == b'/' && (bytes[i + 1] == b'/' || bytes[i + 1] == b'*'))
            || bytes[i] == b'"'
        {
            i = token_end(bytes, i);
        } else if bytes[i] == b'#' && src[i + 1..].trim_start().starts_with('[') {
            if let Some(end_b) = src[i..].find(']') {
                i += end_b + 1;
            } else {
                return n;
            }
        } else {
            break;
        }
    }
    let mut braces = 0_usize;
    let mut parens = 0_usize;
    while i < n {
        let b = bytes[i];
        if (i + 1 < n && b == b'/' && (bytes[i + 1] == b'/' || bytes[i + 1] == b'*')) || b == b'"' {
            i = token_end(bytes, i);
            continue;
        }
        match b {
            b'(' => parens += 1,
            b')' => parens = parens.saturating_sub(1),
            b'{' => braces += 1,
            b'}' if braces > 0 => {
                braces -= 1;
                if braces == 0 && parens == 0 {
                    return i + 1;
                }
            }
            b';' if braces == 0 && parens == 0 => return i + 1,
            _ => {}
        }
        i += 1;
    }
    n
}

pub(crate) fn strip_cfg_test_items_and_comments(src: &str) -> String {
    let bytes = src.as_bytes();
    let n = bytes.len();
    let mut out = String::with_capacity(n);
    let mut i = 0;
    while i < n {
        if i + 1 < n && bytes[i] == b'/' && (bytes[i + 1] == b'/' || bytes[i + 1] == b'*') {
            let end = token_end(bytes, i);
            blank_preserving_newlines(&src[i..end], &mut out);
            i = end;
        } else if bytes[i] == b'"' {
            let end = token_end(bytes, i);
            out.push_str(&src[i..end]);
            i = end;
        } else if bytes[i] == b'#' {
            if let Some(attr_len) = is_cfg_test_attr(&src[i..]) {
                let end = skip_annotated_item_end(src, i + attr_len);
                blank_preserving_newlines(&src[i..end], &mut out);
                i = end;
            } else {
                let ch = src[i..].chars().next().unwrap();
                out.push(ch);
                i += ch.len_utf8();
            }
        } else {
            let ch = src[i..].chars().next().unwrap();
            out.push(ch);
            i += ch.len_utf8();
        }
    }
    out
}

fn is_strict_canvas_module(rel_str: &str) -> bool {
    const STRICT_PREFIXES: &[&str] = &[
        "math_explorer_gui/src/accessibility",
        "math_explorer_gui/src/framework/raster_brush",
        "math_explorer_gui/src/async_sim/unified",
        "math_explorer_gui/src/tabs/number_theory/prime_spiral",
        "math_explorer_gui/src/tabs/solid_state/ising",
        "math_explorer_gui/src/tabs/fluid_dynamics/lattice_boltzmann",
        "math_explorer_gui/src/tabs/ai/attention_maps",
        "math_explorer_gui/src/tabs/ai/grid_world",
        "math_explorer_gui/src/tabs/morphogenesis/ui.rs",
        "math_explorer_gui/src/tabs/medical/dose",
    ];
    STRICT_PREFIXES.iter().any(|p| rel_str.starts_with(p))
}

pub(crate) fn audit_source_for_color32_violations(rel_str: &str, content: &str) -> Vec<String> {
    const BANNED: &[&str] = &[
        "Color32::YELLOW",
        "Color32::LIGHT_BLUE",
        "Color32::LIGHT_GREEN",
        "Color32::LIGHT_RED",
        "Color32::LIGHT_GRAY",
        "Color32::GOLD",
        "Color32::GREEN",
        "Color32::RED",
        "Color32::BLUE",
        "Color32::GRAY",
        "Color32::DARK_GRAY",
        "Color32::DARK_RED",
        "Color32::KHAKI",
        "Color32::BROWN",
    ];
    let prod = strip_cfg_test_items_and_comments(content);
    let mut violations = Vec::new();
    let lines: Vec<&str> = prod.lines().collect();
    let strict = is_strict_canvas_module(rel_str);
    for (idx, &line) in lines.iter().enumerate() {
        let ln = idx + 1;
        for banned in BANNED {
            if line.contains(banned) {
                violations.push(format!("{rel_str}:{ln}: {banned}"));
            }
        }
        if line.contains("Color32::from_gray(")
            && rel_str != "math_explorer_gui/src/tabs/morphogenesis/sculpt.rs"
            && rel_str != "egui_plot/src/items/mod.rs"
        {
            violations.push(format!("{rel_str}:{ln}: Color32::from_gray("));
        }
        if strict && line.contains("Color32::BLACK") {
            violations.push(format!("{rel_str}:{ln}: Color32::BLACK"));
        }
        if strict && line.contains("Color32::WHITE") {
            let ctx_start = idx.saturating_sub(5);
            let window = lines[ctx_start..=idx].join(" ");
            if !window.contains(".image(") {
                violations.push(format!("{rel_str}:{ln}: Color32::WHITE"));
            }
        }
    }
    if rel_str.starts_with("math_explorer_gui/") {
        audit_color32_constructors(rel_str, &prod, strict, &mut violations);
    }
    violations
}

fn audit_color32_constructors(
    rel_str: &str,
    prod: &str,
    strict: bool,
    violations: &mut Vec<String>,
) {
    for ln in find_constructor_lines(prod, "Color32::from_rgb(", !strict) {
        let kind = if strict {
            "Color32::from_rgb("
        } else {
            "Color32::from_rgb(literal)"
        };
        violations.push(format!("{rel_str}:{ln}: {kind}"));
    }
    if rel_str != "math_explorer_gui/src/tabs/neuroscience/cortical_morphometry/scene.rs" {
        for ctor in [
            "Color32::from_rgba_unmultiplied(",
            "Color32::from_rgba_premultiplied(",
        ] {
            for ln in find_constructor_lines(prod, ctor, true) {
                violations.push(format!("{rel_str}:{ln}: {ctor}literal)"));
            }
        }
    }
}
