use egui::{
    Align, Color32, FontId, Pos2, Response, Sense, Stroke, TextStyle, Ui, Vec2,
    text::{LayoutJob, TextFormat},
};

#[path = "latex_delims.rs"]
mod latex_delims;

pub use latex_delims::{MathAccent, MathDelimiter};

/// A 2D box node representing TeX mathematical layout elements.
#[derive(Clone, Debug, PartialEq)]
pub enum MathBox {
    /// Plain text or symbol fragment.
    Text {
        text: String,
        font_id: FontId,
        color: Color32,
        italics: bool,
    },
    /// Horizontal sequence of boxes.
    Row(Vec<MathBox>),
    /// Stacked fraction numerator over denominator.
    Fraction {
        numerator: Box<MathBox>,
        denominator: Box<MathBox>,
    },
    /// Square root expression with radicand.
    Sqrt(Box<MathBox>),
    /// Superscript and/or subscript attached to an optional base.
    Script {
        base: Option<Box<MathBox>>,
        sup: Option<Box<MathBox>>,
        sub: Option<Box<MathBox>>,
    },
    /// Height-scaled paired delimiters wrapping an inner `MathBox`.
    Delimited {
        open: MathDelimiter,
        body: Box<MathBox>,
        close: MathDelimiter,
        font_id: FontId,
        color: Color32,
    },
    /// Over-symbol accent (such as `\dot{x}` or `\ddot{x}`).
    Accent {
        base: Box<MathBox>,
        kind: MathAccent,
        color: Color32,
    },
}

const MAX_LATEX_INPUT_BYTES: usize = 2_048;
const MAX_LATEX_DEPTH: usize = 16;
const MAX_MATH_BOX_NODES: usize = 256;
const MAX_LAYOUT_CHARS: usize = 4_096;
const MAX_FALLBACK_CHARS: usize = 2_048;

fn truncate_chars(text: &str, max_chars: usize) -> &str {
    match text.char_indices().nth(max_chars) {
        Some((byte_idx, _)) => &text[..byte_idx],
        None => text,
    }
}

fn consume_node(remaining_nodes: &mut usize) -> Result<(), ()> {
    if *remaining_nodes == 0 {
        return Err(());
    }
    *remaining_nodes -= 1;
    Ok(())
}

/// Renders a mathematical LaTeX formula using 2D box layout in egui UI.
pub fn render_math_label(
    ui: &mut Ui,
    text: &str,
    style: TextStyle,
    default_color: Color32,
) -> Response {
    let font_id = style.resolve(ui.style());
    let color = default_color;
    let math_box = parse_latex_to_box(text, &font_id, color).unwrap_or_else(|_| MathBox::Text {
        text: truncate_chars(text, MAX_FALLBACK_CHARS).to_string(),
        font_id: font_id.clone(),
        color,
        italics: false,
    });

    render_math_box_ui(ui, &math_box)
}

/// Renders a `MathBox` tree directly onto egui UI canvas.
pub fn render_math_box_ui(ui: &mut Ui, math_box: &MathBox) -> Response {
    let geom = measure_math_box(ui, math_box);
    let (rect, response) = ui.allocate_exact_size(geom.size, Sense::hover());

    if ui.is_rect_visible(rect) {
        let painter = ui.painter();
        paint_math_box(painter, rect.min, math_box, &geom);
    }

    response
}

/// Parses a LaTeX string into an `egui::text::LayoutJob`.
pub fn parse_latex_to_layout_job(
    text: &str,
    style: TextStyle,
    ui: &Ui,
    default_color: Color32,
) -> LayoutJob {
    let font_id = style.resolve(ui.style());
    parse_latex_with_font(text, font_id, default_color)
}

/// Parses prose containing optional `$...$` or `$$...$$` inline LaTeX math into a `LayoutJob`.
///
/// Text outside `$...$` delimiters renders as upright prose without math italicization or
/// `_`/`^` script parsing, and inline `\frac{num}{den}` terms format on one line with `⁄`.
pub fn parse_prose_with_inline_math(
    text: &str,
    style: TextStyle,
    ui: &Ui,
    default_color: Color32,
) -> LayoutJob {
    let font_id = style.resolve(ui.style());
    parse_prose_with_font(text, font_id, default_color)
}

/// Parses prose with optional `$...$` inline math using an explicit `FontId` and `Color32`.
pub fn parse_prose_with_font(text: &str, font_id: FontId, color: Color32) -> LayoutJob {
    let mut job = LayoutJob::default();
    if text.is_empty() {
        return job;
    }
    if !text.contains('$') {
        let cap = if text.len() > MAX_LATEX_INPUT_BYTES {
            MAX_FALLBACK_CHARS
        } else {
            MAX_LAYOUT_CHARS
        };
        append_upright_fallback(&mut job, text, font_id, color, cap);
        return job;
    }
    match parse_prose_to_box(text, &font_id, color) {
        Ok(math_box) => {
            let mut remaining_chars = MAX_LAYOUT_CHARS;
            math_box_to_layout_job_mode(&math_box, &mut job, 0, &mut remaining_chars, true);
        }
        Err(()) => append_upright_fallback(&mut job, text, font_id, color, MAX_FALLBACK_CHARS),
    }
    job
}

fn append_upright_fallback(
    job: &mut LayoutJob,
    text: &str,
    font_id: FontId,
    color: Color32,
    max_chars: usize,
) {
    job.append(
        truncate_chars(text, max_chars),
        0.0,
        TextFormat {
            font_id,
            color,
            ..Default::default()
        },
    );
}

/// Parses a LaTeX string using a specific `FontId` and base `Color32`.
pub fn parse_latex_with_font(text: &str, font_id: FontId, color: Color32) -> LayoutJob {
    let mut job = LayoutJob::default();
    if text.is_empty() {
        return job;
    }

    match parse_latex_to_box(text, &font_id, color) {
        Ok(math_box) => {
            math_box_to_layout_job(&math_box, &mut job);
        }
        Err(_) => append_upright_fallback(&mut job, text, font_id, color, MAX_FALLBACK_CHARS),
    }

    job
}

fn split_prose_math_spans(text: &str) -> Result<Vec<(bool, &str)>, ()> {
    let mut spans = Vec::new();
    let mut cursor = 0;
    while let Some(rel_dollar) = text[cursor..].find('$') {
        let dollar_idx = cursor + rel_dollar;
        if dollar_idx > cursor {
            spans.push((false, &text[cursor..dollar_idx]));
        }
        if text[dollar_idx..].starts_with("$$") {
            let math_start = dollar_idx + 2;
            let rel_end = text[math_start..].find("$$").ok_or(())?;
            spans.push((true, &text[math_start..math_start + rel_end]));
            cursor = math_start + rel_end + 2;
        } else {
            let math_start = dollar_idx + 1;
            let rel_end = text[math_start..].find('$').ok_or(())?;
            spans.push((true, &text[math_start..math_start + rel_end]));
            cursor = math_start + rel_end + 1;
        }
    }
    if cursor < text.len() {
        spans.push((false, &text[cursor..]));
    }
    Ok(spans)
}

fn parse_prose_to_box(text: &str, base_font_id: &FontId, color: Color32) -> Result<MathBox, ()> {
    if text.len() > MAX_LATEX_INPUT_BYTES {
        return Err(());
    }
    let mut remaining_nodes = MAX_MATH_BOX_NODES;
    let spans = split_prose_math_spans(text)?;
    let mut boxes = Vec::new();
    for (is_math, part) in spans {
        if part.is_empty() {
            continue;
        }
        if is_math {
            let sub =
                parse_math_expression_boxes(part, base_font_id, color, 1, &mut remaining_nodes)?;
            boxes.extend(sub);
        } else {
            consume_node(&mut remaining_nodes)?;
            boxes.push(MathBox::Text {
                text: part.to_string(),
                font_id: base_font_id.clone(),
                color,
                italics: false,
            });
        }
    }
    simplify_boxes(boxes, &mut remaining_nodes)
}

/// Parses LaTeX string into `MathBox` AST tree.
#[allow(clippy::result_unit_err)]
pub fn parse_latex_to_box(
    text: &str,
    base_font_id: &FontId,
    color: Color32,
) -> Result<MathBox, ()> {
    if text.len() > MAX_LATEX_INPUT_BYTES {
        return Err(());
    }
    let mut remaining_nodes = MAX_MATH_BOX_NODES;
    parse_latex_to_box_bounded(text, base_font_id, color, 0, &mut remaining_nodes)
}

fn parse_latex_to_box_bounded(
    text: &str,
    base_font_id: &FontId,
    color: Color32,
    depth: usize,
    remaining_nodes: &mut usize,
) -> Result<MathBox, ()> {
    if text.len() > MAX_LATEX_INPUT_BYTES || depth > MAX_LATEX_DEPTH {
        return Err(());
    }
    if text.is_empty() {
        consume_node(remaining_nodes)?;
        return Ok(MathBox::Row(Vec::new()));
    }

    if text.contains('$') {
        let parts: Vec<&str> = text.split('$').collect();
        if parts.len() % 2 == 1 {
            let mut boxes = Vec::new();
            for (idx, part) in parts.iter().enumerate() {
                if part.is_empty() {
                    continue;
                }
                if idx % 2 == 1 {
                    let sub_boxes = parse_math_expression_boxes(
                        part,
                        base_font_id,
                        color,
                        depth + 1,
                        remaining_nodes,
                    )?;
                    boxes.extend(sub_boxes);
                } else {
                    consume_node(remaining_nodes)?;
                    boxes.push(MathBox::Text {
                        text: (*part).to_string(),
                        font_id: base_font_id.clone(),
                        color,
                        italics: false,
                    });
                }
            }
            return simplify_boxes(boxes, remaining_nodes);
        }
    }

    let boxes = parse_math_expression_boxes(text, base_font_id, color, depth, remaining_nodes)?;
    simplify_boxes(boxes, remaining_nodes)
}

fn simplify_boxes(boxes: Vec<MathBox>, remaining_nodes: &mut usize) -> Result<MathBox, ()> {
    if boxes.len() == 1 {
        Ok(boxes.into_iter().next().unwrap())
    } else {
        consume_node(remaining_nodes)?;
        Ok(MathBox::Row(boxes))
    }
}

fn attach_script_box(
    boxes: &mut Vec<MathBox>,
    script_box: MathBox,
    is_sup: bool,
    remaining_nodes: &mut usize,
) -> Result<(), ()> {
    let prev_box = boxes.pop();
    let (sup_val, sub_val) = if is_sup {
        (Some(Box::new(script_box)), None)
    } else {
        (None, Some(Box::new(script_box)))
    };
    match prev_box {
        Some(MathBox::Script { base, sup, sub }) => {
            boxes.push(MathBox::Script {
                base,
                sup: if is_sup { sup_val } else { sup },
                sub: if !is_sup { sub_val } else { sub },
            });
        }
        Some(b) => {
            consume_node(remaining_nodes)?;
            boxes.push(MathBox::Script {
                base: Some(Box::new(b)),
                sup: sup_val,
                sub: sub_val,
            });
        }
        None => {
            consume_node(remaining_nodes)?;
            boxes.push(MathBox::Script {
                base: None,
                sup: sup_val,
                sub: sub_val,
            });
        }
    }
    Ok(())
}

fn parse_math_expression_boxes(
    expr: &str,
    base_font_id: &FontId,
    color: Color32,
    depth: usize,
    remaining_nodes: &mut usize,
) -> Result<Vec<MathBox>, ()> {
    if depth > MAX_LATEX_DEPTH {
        return Err(());
    }
    let chars: Vec<char> = expr.chars().collect();
    let (mut i, len) = (0, chars.len());
    let mut boxes = Vec::new();
    let script_font_id = FontId::new(
        (base_font_id.size * 0.75).max(8.0),
        base_font_id.family.clone(),
    );

    while i < len {
        let c = chars[i];
        if c == '^' || c == '_' {
            let is_sup = c == '^';
            let (script_str, next_i) = read_script_arg(&chars, i + 1)?;
            i = next_i;
            let sbox = parse_latex_to_box_bounded(
                &script_str,
                &script_font_id,
                color,
                depth + 1,
                remaining_nodes,
            )?;
            attach_script_box(&mut boxes, sbox, is_sup, remaining_nodes)?;
        } else if c == '\\' {
            let (cmd, next_i) = read_command(&chars, i + 1)?;
            i = next_i;
            process_command_box(
                &cmd,
                &chars,
                &mut i,
                base_font_id,
                color,
                depth,
                remaining_nodes,
                &mut boxes,
            )?;
        } else if c == '{' {
            let (group_str, next_i) = read_group_arg(&chars, i + 1)?;
            i = next_i;
            let gbox = parse_latex_to_box_bounded(
                &group_str,
                base_font_id,
                color,
                depth + 1,
                remaining_nodes,
            )?;
            boxes.push(gbox);
        } else if c == '}' {
            return Err(());
        } else if c.is_ascii_alphabetic() {
            let start = i;
            while i < len && chars[i].is_ascii_alphabetic() {
                i += 1;
            }
            let word: String = chars[start..i].iter().collect();
            push_text_box(
                &mut boxes,
                word.clone(),
                base_font_id,
                color,
                word.len() == 1,
                remaining_nodes,
            )?;
        } else {
            push_text_box(
                &mut boxes,
                c.to_string(),
                base_font_id,
                color,
                false,
                remaining_nodes,
            )?;
            i += 1;
        }
    }
    Ok(latex_delims::upgrade_tall_plain_delimiters(boxes))
}

fn push_text_box(
    boxes: &mut Vec<MathBox>,
    text: String,
    font_id: &FontId,
    color: Color32,
    italics: bool,
    remaining_nodes: &mut usize,
) -> Result<(), ()> {
    consume_node(remaining_nodes)?;
    boxes.push(MathBox::Text {
        text,
        font_id: font_id.clone(),
        color,
        italics,
    });
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn process_command_box(
    cmd: &str,
    chars: &[char],
    i: &mut usize,
    base_font_id: &FontId,
    color: Color32,
    depth: usize,
    remaining_nodes: &mut usize,
    boxes: &mut Vec<MathBox>,
) -> Result<(), ()> {
    if let Some(sym) = map_latex_symbol(cmd) {
        return process_symbol_or_sqrt(
            cmd,
            sym,
            chars,
            i,
            base_font_id,
            color,
            depth,
            remaining_nodes,
            boxes,
        );
    }
    let parse_ctx = latex_delims::LatexParseCtx {
        chars,
        base_font_id,
        color,
        depth,
    };
    if let Some(kind) = latex_delims::accent_kind_from_cmd(cmd) {
        return latex_delims::process_accent_command(kind, &parse_ctx, i, remaining_nodes, boxes);
    }
    if is_upright_operator_name(cmd) {
        return push_text_box(
            boxes,
            cmd.to_string(),
            base_font_id,
            color,
            false,
            remaining_nodes,
        );
    }
    match cmd {
        "text" | "mathrm" | "mathbf" | "operatorname" | "mathcal" => {
            process_text_command(cmd, chars, i, base_font_id, color, remaining_nodes, boxes)
        }
        "mathbb" => process_mathbb_command(chars, i, base_font_id, color, remaining_nodes, boxes),
        "left" => latex_delims::process_left_command(&parse_ctx, i, remaining_nodes, boxes),
        "right" => latex_delims::process_stray_right_command(&parse_ctx, i, remaining_nodes, boxes),
        "frac" | "dfrac" | "tfrac" => {
            process_frac_command(chars, i, base_font_id, color, depth, remaining_nodes, boxes)
        }
        "pmod" => {
            process_pmod_command(chars, i, base_font_id, color, depth, remaining_nodes, boxes)
        }
        "not" => process_not_command(chars, i, base_font_id, color, remaining_nodes, boxes),
        "quad" => push_text_box(
            boxes,
            "  ".to_string(),
            base_font_id,
            color,
            false,
            remaining_nodes,
        ),
        "qquad" => push_text_box(
            boxes,
            "    ".to_string(),
            base_font_id,
            color,
            false,
            remaining_nodes,
        ),
        "," | ":" | ";" | " " => push_text_box(
            boxes,
            " ".to_string(),
            base_font_id,
            color,
            false,
            remaining_nodes,
        ),
        "!" | "big" | "Big" | "bigg" | "Bigg" | "bigl" | "bigr" | "Bigl" | "Bigr" | "biggl"
        | "biggr" | "Biggl" | "Biggr" => Ok(()),
        "\\" | "newline" => push_text_box(
            boxes,
            "\n".to_string(),
            base_font_id,
            color,
            false,
            remaining_nodes,
        ),
        _ => push_text_box(
            boxes,
            cmd.to_string(),
            base_font_id,
            color,
            false,
            remaining_nodes,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn process_symbol_or_sqrt(
    cmd: &str,
    sym: &str,
    chars: &[char],
    i: &mut usize,
    base_font_id: &FontId,
    color: Color32,
    depth: usize,
    remaining_nodes: &mut usize,
    boxes: &mut Vec<MathBox>,
) -> Result<(), ()> {
    if cmd == "sqrt" {
        skip_whitespace(chars, i);
        if *i < chars.len() && chars[*i] == '{' {
            let (arg, next_i) = read_group_arg(chars, *i + 1)?;
            *i = next_i;
            let radicand_box =
                parse_latex_to_box_bounded(&arg, base_font_id, color, depth + 1, remaining_nodes)?;
            consume_node(remaining_nodes)?;
            boxes.push(MathBox::Sqrt(Box::new(radicand_box)));
            return Ok(());
        }
    }
    push_text_box(
        boxes,
        sym.to_string(),
        base_font_id,
        color,
        false,
        remaining_nodes,
    )
}

fn resolve_text_escapes(raw: &str) -> String {
    let chars: Vec<char> = raw.chars().collect();
    let mut out = String::with_capacity(raw.len());
    let mut idx = 0;
    while idx < chars.len() {
        if chars[idx] != '\\' || idx + 1 >= chars.len() {
            out.push(chars[idx]);
            idx += 1;
            continue;
        }
        if let Ok((cmd, next_i)) = read_command(&chars, idx + 1) {
            idx = next_i;
            match cmd.as_str() {
                "AA" => out.push('Å'),
                "deg" => out.push('°'),
                "dots" | "ldots" => out.push('…'),
                "," | ":" | ";" | " " => out.push(' '),
                "%" | "_" | "#" | "&" | "{" | "}" => out.push_str(&cmd),
                other => out.push_str(map_latex_symbol(other).unwrap_or(other)),
            }
        } else {
            out.push(chars[idx]);
            idx += 1;
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn process_text_command(
    cmd: &str,
    chars: &[char],
    i: &mut usize,
    base_font_id: &FontId,
    color: Color32,
    remaining_nodes: &mut usize,
    boxes: &mut Vec<MathBox>,
) -> Result<(), ()> {
    skip_whitespace(chars, i);
    if *i < chars.len() && chars[*i] == '{' {
        let (arg, next_i) = read_group_arg(chars, *i + 1)?;
        *i = next_i;
        let resolved = resolve_text_escapes(&arg);
        push_text_box(
            boxes,
            resolved,
            base_font_id,
            color,
            cmd == "mathcal",
            remaining_nodes,
        )
    } else {
        Err(())
    }
}

fn process_mathbb_command(
    chars: &[char],
    i: &mut usize,
    base_font_id: &FontId,
    color: Color32,
    remaining_nodes: &mut usize,
    boxes: &mut Vec<MathBox>,
) -> Result<(), ()> {
    skip_whitespace(chars, i);
    if *i >= chars.len() || chars[*i] != '{' {
        return Err(());
    }
    let (arg, next_i) = read_group_arg(chars, *i + 1)?;
    *i = next_i;
    let mapped = latex_delims::map_blackboard_bold(&arg);
    push_text_box(boxes, mapped, base_font_id, color, false, remaining_nodes)
}

fn process_frac_command(
    chars: &[char],
    i: &mut usize,
    base_font_id: &FontId,
    color: Color32,
    depth: usize,
    remaining_nodes: &mut usize,
    boxes: &mut Vec<MathBox>,
) -> Result<(), ()> {
    skip_whitespace(chars, i);
    if *i >= chars.len() || chars[*i] != '{' {
        return Err(());
    }
    let (num, after_num) = read_group_arg(chars, *i + 1)?;
    *i = after_num;
    skip_whitespace(chars, i);
    if *i >= chars.len() || chars[*i] != '{' {
        return Err(());
    }
    let (den, after_den) = read_group_arg(chars, *i + 1)?;
    *i = after_den;
    let num_box =
        parse_latex_to_box_bounded(&num, base_font_id, color, depth + 1, remaining_nodes)?;
    let den_box =
        parse_latex_to_box_bounded(&den, base_font_id, color, depth + 1, remaining_nodes)?;
    consume_node(remaining_nodes)?;
    boxes.push(MathBox::Fraction {
        numerator: Box::new(num_box),
        denominator: Box::new(den_box),
    });
    Ok(())
}

fn process_pmod_command(
    chars: &[char],
    i: &mut usize,
    base_font_id: &FontId,
    color: Color32,
    depth: usize,
    remaining_nodes: &mut usize,
    boxes: &mut Vec<MathBox>,
) -> Result<(), ()> {
    skip_whitespace(chars, i);
    if *i >= chars.len() || chars[*i] != '{' {
        return Err(());
    }
    let (arg, next_i) = read_group_arg(chars, *i + 1)?;
    *i = next_i;
    push_text_box(
        boxes,
        " (mod ".to_string(),
        base_font_id,
        color,
        false,
        remaining_nodes,
    )?;
    let inner = parse_latex_to_box_bounded(&arg, base_font_id, color, depth + 1, remaining_nodes)?;
    boxes.push(inner);
    push_text_box(
        boxes,
        ")".to_string(),
        base_font_id,
        color,
        false,
        remaining_nodes,
    )
}

fn process_not_command(
    chars: &[char],
    i: &mut usize,
    base_font_id: &FontId,
    color: Color32,
    remaining_nodes: &mut usize,
    boxes: &mut Vec<MathBox>,
) -> Result<(), ()> {
    skip_whitespace(chars, i);
    if *i < chars.len() && chars[*i] == '\\' {
        let (next_cmd, next_i) = read_command(chars, *i + 1)?;
        let negated = match next_cmd.as_str() {
            "equiv" => Some("≢"),
            "in" => Some("∉"),
            "approx" => Some("≉"),
            _ => None,
        };
        if let Some(sym) = negated {
            *i = next_i;
            return push_text_box(
                boxes,
                sym.to_string(),
                base_font_id,
                color,
                false,
                remaining_nodes,
            );
        }
    } else if *i < chars.len() && chars[*i] == '=' {
        *i += 1;
        return push_text_box(
            boxes,
            "≠".to_string(),
            base_font_id,
            color,
            false,
            remaining_nodes,
        );
    }
    push_text_box(
        boxes,
        "¬".to_string(),
        base_font_id,
        color,
        false,
        remaining_nodes,
    )
}

fn is_upright_operator_name(cmd: &str) -> bool {
    matches!(
        cmd,
        "sin"
            | "cos"
            | "tan"
            | "sec"
            | "csc"
            | "cot"
            | "exp"
            | "ln"
            | "log"
            | "lim"
            | "max"
            | "min"
            | "sup"
            | "inf"
            | "det"
            | "gcd"
            | "lcm"
            | "mod"
            | "bmod"
            | "arg"
            | "dim"
            | "ker"
            | "erf"
            | "erfc"
            | "arcsin"
            | "arccos"
            | "arctan"
            | "atan2"
            | "sgn"
            | "sign"
            | "Pr"
            | "sinh"
            | "cosh"
            | "tanh"
    )
}

fn read_script_arg(chars: &[char], i: usize) -> Result<(String, usize), ()> {
    if i >= chars.len() {
        return Err(());
    }
    if chars[i] == '{' {
        read_group_arg(chars, i + 1)
    } else if chars[i] == '\\' {
        let (cmd, next_i) = read_command(chars, i + 1)?;
        Ok((format!("\\{}", cmd), next_i))
    } else {
        let s: String = std::iter::once(chars[i]).collect();
        Ok((s, i + 1))
    }
}

fn read_group_arg(chars: &[char], mut i: usize) -> Result<(String, usize), ()> {
    let mut depth = 1;
    let start = i;
    while i < chars.len() {
        if chars[i] == '{' {
            depth += 1;
        } else if chars[i] == '}' {
            depth -= 1;
            if depth == 0 {
                let arg: String = chars[start..i].iter().collect();
                return Ok((arg, i + 1));
            }
        }
        i += 1;
    }
    Err(())
}

fn read_command(chars: &[char], mut i: usize) -> Result<(String, usize), ()> {
    if i >= chars.len() {
        return Err(());
    }
    if !chars[i].is_ascii_alphabetic() {
        let s: String = std::iter::once(chars[i]).collect();
        return Ok((s, i + 1));
    }
    let start = i;
    while i < chars.len() && chars[i].is_ascii_alphabetic() {
        i += 1;
    }
    let cmd: String = chars[start..i].iter().collect();
    Ok((cmd, i))
}

fn skip_whitespace(chars: &[char], i: &mut usize) {
    while *i < chars.len() && chars[*i].is_whitespace() {
        *i += 1;
    }
}

fn map_greek_symbol(cmd: &str) -> Option<&'static str> {
    match cmd {
        "alpha" => Some("α"),
        "beta" => Some("β"),
        "gamma" => Some("γ"),
        "delta" => Some("δ"),
        "epsilon" | "varepsilon" => Some("ε"),
        "zeta" => Some("ζ"),
        "eta" => Some("η"),
        "theta" | "vartheta" => Some("θ"),
        "iota" => Some("ι"),
        "kappa" => Some("κ"),
        "lambda" => Some("λ"),
        "mu" => Some("μ"),
        "nu" => Some("ν"),
        "xi" => Some("ξ"),
        "pi" | "varpi" => Some("π"),
        "rho" | "varrho" => Some("ρ"),
        "sigma" | "varsigma" => Some("σ"),
        "tau" => Some("τ"),
        "upsilon" => Some("υ"),
        "phi" | "varphi" => Some("φ"),
        "chi" => Some("χ"),
        "psi" => Some("ψ"),
        "omega" => Some("ω"),
        "Gamma" => Some("Γ"),
        "Delta" => Some("Δ"),
        "Theta" => Some("Θ"),
        "Lambda" => Some("Λ"),
        "Xi" => Some("Ξ"),
        "Pi" => Some("Π"),
        "Sigma" => Some("Σ"),
        "Upsilon" => Some("Υ"),
        "Phi" => Some("Φ"),
        "Psi" => Some("Ψ"),
        "Omega" => Some("Ω"),
        _ => None,
    }
}

fn map_escaped_or_set_symbol(cmd: &str) -> Option<&'static str> {
    match cmd {
        "in" => Some("∈"),
        "notin" => Some("∉"),
        "subset" => Some("⊂"),
        "cup" => Some("∪"),
        "cap" => Some("∩"),
        "forall" => Some("∀"),
        "exists" => Some("∃"),
        "odot" => Some("⊙"),
        "oplus" => Some("⊕"),
        "otimes" => Some("⊗"),
        "|" => Some("‖"),
        "{" => Some("{"),
        "}" => Some("}"),
        "#" => Some("#"),
        "%" => Some("%"),
        "_" => Some("_"),
        "&" => Some("&"),
        _ => None,
    }
}

fn map_latex_symbol(cmd: &str) -> Option<&'static str> {
    if let Some(greek) = map_greek_symbol(cmd) {
        return Some(greek);
    }
    if let Some(sym) = map_escaped_or_set_symbol(cmd) {
        return Some(sym);
    }
    match cmd {
        "hbar" => Some("ℏ"),
        "ell" => Some("ℓ"),
        "AA" => Some("Å"),
        "deg" => Some("°"),
        "langle" => Some("⟨"),
        "rangle" => Some("⟩"),
        "lfloor" => Some("⌊"),
        "rfloor" => Some("⌋"),
        "lceil" => Some("⌈"),
        "rceil" => Some("⌉"),
        "dots" | "ldots" => Some("…"),
        "cdots" => Some("⋯"),
        "cdot" => Some("·"),
        "times" => Some("×"),
        "div" => Some("÷"),
        "pm" => Some("±"),
        "mp" => Some("∓"),
        "infty" => Some("∞"),
        "partial" => Some("∂"),
        "nabla" => Some("∇"),
        "sum" => Some("∑"),
        "prod" => Some("∏"),
        "int" => Some("∫"),
        "iint" => Some("∬"),
        "sqrt" => Some("√"),
        "leq" | "le" => Some("≤"),
        "geq" | "ge" => Some("≥"),
        "neq" | "ne" => Some("≠"),
        "approx" => Some("≈"),
        "equiv" => Some("≡"),
        "sim" => Some("~"),
        "propto" => Some("∝"),
        "perp" => Some("⊥"),
        "to" | "rightarrow" => Some("→"),
        "leftarrow" => Some("←"),
        "Rightarrow" => Some("⇒"),
        "Leftarrow" => Some("⇐"),
        "leftrightarrow" => Some("↔"),
        "vdots" => Some("⋮"),
        "ddots" => Some("⋱"),
        "lvert" | "rvert" | "mid" => Some("|"),
        "circ" => Some("∘"),
        "vee" => Some("∨"),
        _ => None,
    }
}

/// Converts a `MathBox` tree into an `egui::text::LayoutJob`.
pub fn math_box_to_layout_job(box_node: &MathBox, job: &mut LayoutJob) {
    let existing_chars = job.text.chars().take(MAX_LAYOUT_CHARS + 1).count();
    let mut remaining_chars = MAX_LAYOUT_CHARS.saturating_sub(existing_chars);
    math_box_to_layout_job_mode(box_node, job, 0, &mut remaining_chars, false);
}

fn append_bounded(
    job: &mut LayoutJob,
    text: &str,
    leading_space: f32,
    format: TextFormat,
    remaining_chars: &mut usize,
) {
    if *remaining_chars == 0 || text.is_empty() {
        return;
    }
    let truncated = truncate_chars(text, *remaining_chars);
    let added = truncated.chars().count();
    if added > 0 {
        job.append(truncated, leading_space, format);
        *remaining_chars = remaining_chars.saturating_sub(added);
    }
}

fn append_job_sections(
    src: &LayoutJob,
    dst: &mut LayoutJob,
    override_valign: Option<Align>,
    remaining_chars: &mut usize,
) {
    for section in &src.sections {
        if *remaining_chars == 0 {
            break;
        }
        let txt = &src.text[section.byte_range.clone()];
        let mut fmt = section.format.clone();
        if let Some(valign) = override_valign {
            fmt.valign = valign;
        }
        append_bounded(dst, txt, section.leading_space, fmt, remaining_chars);
    }
}

fn append_script_job(
    script_box: &MathBox,
    job: &mut LayoutJob,
    valign: Align,
    depth: usize,
    remaining_chars: &mut usize,
    inline_fractions: bool,
) {
    if *remaining_chars == 0 {
        return;
    }
    let mut sub_budget = *remaining_chars;
    let mut script_job = LayoutJob::default();
    math_box_to_layout_job_mode(
        script_box,
        &mut script_job,
        depth,
        &mut sub_budget,
        inline_fractions,
    );
    append_job_sections(&script_job, job, Some(valign), remaining_chars);
}

fn has_balanced_outer_parens(text: &str) -> bool {
    let trimmed = text.trim();
    if !trimmed.starts_with('(') || !trimmed.ends_with(')') {
        return false;
    }
    let mut depth = 0_i32;
    let total = trimmed.chars().count();
    for (idx, ch) in trimmed.chars().enumerate() {
        if ch == '(' {
            depth += 1;
        } else if ch == ')' {
            depth -= 1;
            if depth == 0 && idx + 1 < total {
                return false;
            }
        }
    }
    depth == 0
}

fn is_compound_inline_term(node: &MathBox, rendered_text: &str) -> bool {
    if has_balanced_outer_parens(rendered_text) {
        return false;
    }
    match node {
        MathBox::Fraction { .. } => true,
        MathBox::Row(items) => {
            let non_ws: Vec<&MathBox> = items
                .iter()
                .filter(|b| !matches!(b, MathBox::Text { text, .. } if text.trim().is_empty()))
                .collect();
            if non_ws.len() == 1 {
                is_compound_inline_term(non_ws[0], rendered_text)
            } else {
                non_ws.len() > 1
            }
        }
        MathBox::Text { text, .. } => {
            let trimmed = text.trim();
            trimmed.chars().count() > 1 && trimmed.contains(['+', '-', ' ', '/', '⁄'])
        }
        MathBox::Delimited {
            open, body, close, ..
        } => {
            if *open != MathDelimiter::None && *close != MathDelimiter::None {
                false
            } else {
                is_compound_inline_term(body, rendered_text)
            }
        }
        MathBox::Sqrt(_) | MathBox::Script { .. } | MathBox::Accent { .. } => false,
    }
}

fn append_inline_fraction(
    numerator: &MathBox,
    denominator: &MathBox,
    job: &mut LayoutJob,
    depth: usize,
    remaining_chars: &mut usize,
) {
    let mut sub_budget = *remaining_chars;
    let mut num_job = LayoutJob::default();
    math_box_to_layout_job_mode(numerator, &mut num_job, depth + 1, &mut sub_budget, true);
    let mut den_job = LayoutJob::default();
    math_box_to_layout_job_mode(denominator, &mut den_job, depth + 1, &mut sub_budget, true);

    let mut upright_fmt = num_job
        .sections
        .first()
        .or_else(|| den_job.sections.first())
        .map(|s| s.format.clone())
        .unwrap_or_default();
    upright_fmt.italics = false;
    upright_fmt.valign = Align::Center;

    let wrap_num = is_compound_inline_term(numerator, &num_job.text);
    let wrap_den = is_compound_inline_term(denominator, &den_job.text);
    if wrap_num {
        append_bounded(job, "(", 0.0, upright_fmt.clone(), remaining_chars);
    }
    append_job_sections(&num_job, job, None, remaining_chars);
    if wrap_num {
        append_bounded(job, ")", 0.0, upright_fmt.clone(), remaining_chars);
    }
    append_bounded(job, "⁄", 0.0, upright_fmt.clone(), remaining_chars);
    if wrap_den {
        append_bounded(job, "(", 0.0, upright_fmt.clone(), remaining_chars);
    }
    append_job_sections(&den_job, job, None, remaining_chars);
    if wrap_den {
        append_bounded(job, ")", 0.0, upright_fmt, remaining_chars);
    }
}

fn append_display_fraction(
    numerator: &MathBox,
    denominator: &MathBox,
    job: &mut LayoutJob,
    depth: usize,
    remaining_chars: &mut usize,
) {
    let mut sub_budget = *remaining_chars;
    let mut num_job = LayoutJob::default();
    math_box_to_layout_job_mode(numerator, &mut num_job, depth + 1, &mut sub_budget, false);
    let mut den_job = LayoutJob::default();
    math_box_to_layout_job_mode(denominator, &mut den_job, depth + 1, &mut sub_budget, false);

    let width_chars = num_job
        .text
        .chars()
        .count()
        .max(den_job.text.chars().count())
        .clamp(1, MAX_LAYOUT_CHARS);
    let first_fmt = num_job
        .sections
        .first()
        .or_else(|| den_job.sections.first())
        .map(|s| s.format.clone())
        .unwrap_or_default();

    append_bounded(job, &num_job.text, 0.0, first_fmt.clone(), remaining_chars);
    if *remaining_chars > 0 {
        let bar = "─".repeat(width_chars.min(*remaining_chars));
        append_bounded(
            job,
            &format!("\n{bar}\n"),
            0.0,
            first_fmt.clone(),
            remaining_chars,
        );
    }
    append_bounded(job, &den_job.text, 0.0, first_fmt, remaining_chars);
}

fn append_sqrt_job(
    radicand: &MathBox,
    job: &mut LayoutJob,
    depth: usize,
    remaining_chars: &mut usize,
    inline_fractions: bool,
) {
    let mut rad_budget = *remaining_chars;
    let mut rad_job = LayoutJob::default();
    math_box_to_layout_job_mode(
        radicand,
        &mut rad_job,
        depth + 1,
        &mut rad_budget,
        inline_fractions,
    );
    let first_fmt = rad_job
        .sections
        .first()
        .map(|s| s.format.clone())
        .unwrap_or_default();

    append_bounded(job, "√", 0.0, first_fmt.clone(), remaining_chars);
    let mut overbar_text = String::new();
    let mut count = 0;
    for c in rad_job.text.chars().filter(|&c| c != '\u{0305}') {
        if count >= *remaining_chars {
            break;
        }
        overbar_text.push(c);
        count += 1;
        if count < *remaining_chars {
            overbar_text.push('\u{0305}');
            count += 1;
        }
    }
    if !overbar_text.is_empty() {
        job.append(&overbar_text, 0.0, first_fmt);
        *remaining_chars = remaining_chars.saturating_sub(count);
    }
}

#[allow(clippy::too_many_arguments)]
fn append_script_box_job(
    base: Option<&MathBox>,
    sup: Option<&MathBox>,
    sub: Option<&MathBox>,
    job: &mut LayoutJob,
    depth: usize,
    remaining_chars: &mut usize,
    inline_fractions: bool,
) {
    if let Some(b) = base {
        math_box_to_layout_job_mode(b, job, depth + 1, remaining_chars, inline_fractions);
    }
    if let Some(s) = sup {
        append_script_job(
            s,
            job,
            Align::Min,
            depth + 1,
            remaining_chars,
            inline_fractions,
        );
    }
    if let Some(s) = sub {
        append_script_job(
            s,
            job,
            Align::Max,
            depth + 1,
            remaining_chars,
            inline_fractions,
        );
    }
}

fn math_box_to_layout_job_mode(
    box_node: &MathBox,
    job: &mut LayoutJob,
    depth: usize,
    remaining_chars: &mut usize,
    inline_fractions: bool,
) {
    if depth > MAX_LATEX_DEPTH || *remaining_chars == 0 {
        return;
    }
    match box_node {
        MathBox::Text {
            text,
            font_id,
            color,
            italics,
        } => {
            let fmt = TextFormat {
                font_id: font_id.clone(),
                color: *color,
                italics: *italics,
                valign: Align::Center,
                ..Default::default()
            };
            append_bounded(job, text, 0.0, fmt, remaining_chars);
        }
        MathBox::Row(boxes) => {
            for b in boxes {
                if *remaining_chars == 0 {
                    break;
                }
                math_box_to_layout_job_mode(b, job, depth + 1, remaining_chars, inline_fractions);
            }
        }
        MathBox::Fraction {
            numerator,
            denominator,
        } => {
            if inline_fractions {
                append_inline_fraction(numerator, denominator, job, depth, remaining_chars);
            } else {
                append_display_fraction(numerator, denominator, job, depth, remaining_chars);
            }
        }
        MathBox::Sqrt(radicand) => {
            append_sqrt_job(radicand, job, depth, remaining_chars, inline_fractions);
        }
        MathBox::Script { base, sup, sub } => append_script_box_job(
            base.as_deref(),
            sup.as_deref(),
            sub.as_deref(),
            job,
            depth,
            remaining_chars,
            inline_fractions,
        ),
        MathBox::Delimited {
            open,
            body,
            close,
            font_id,
            color,
        } => {
            let spec = latex_delims::DelimPaintSpec {
                open: *open,
                body,
                close: *close,
                font_id,
                color: *color,
            };
            latex_delims::delimited_to_layout_job(
                &spec,
                job,
                depth,
                remaining_chars,
                inline_fractions,
            );
        }
        MathBox::Accent { base, kind, color } => latex_delims::accent_to_layout_job(
            base,
            *kind,
            *color,
            job,
            depth,
            remaining_chars,
            inline_fractions,
        ),
    }
}

#[derive(Clone, Debug)]
struct BoxGeom {
    size: Vec2,
    children: Vec<BoxGeom>,
}

fn measure_math_box(ui: &Ui, b: &MathBox) -> BoxGeom {
    measure_math_box_bounded(ui, b, 0)
}

fn measure_script_box(
    ui: &Ui,
    base: Option<&MathBox>,
    sup: Option<&MathBox>,
    sub: Option<&MathBox>,
    depth: usize,
) -> BoxGeom {
    let mut children = Vec::new();
    let mut width: f32 = 0.0;
    let mut base_h: f32 = 12.0;
    if let Some(b) = base {
        let bg = measure_math_box_bounded(ui, b, depth + 1);
        width += bg.size.x;
        base_h = bg.size.y;
        children.push(bg);
    }
    let mut script_w: f32 = 0.0;
    if let Some(sp) = sup {
        let spg = measure_math_box_bounded(ui, sp, depth + 1);
        script_w = script_w.max(spg.size.x);
        children.push(spg);
    }
    if let Some(sb) = sub {
        let sbg = measure_math_box_bounded(ui, sb, depth + 1);
        script_w = script_w.max(sbg.size.x);
        children.push(sbg);
    }
    BoxGeom {
        size: Vec2::new(width + script_w, base_h * 1.2),
        children,
    }
}

fn measure_text_box(ui: &Ui, text: &str, font_id: &FontId, color: Color32) -> BoxGeom {
    let bounded_text = truncate_chars(text, MAX_LAYOUT_CHARS).to_owned();
    let galley = ui
        .painter()
        .layout_no_wrap(bounded_text.clone(), font_id.clone(), color);
    let raw_size = galley.size();
    let size = if !bounded_text.is_empty() && (raw_size.x <= 0.0 || raw_size.y <= 0.0) {
        Vec2::new(
            (bounded_text.chars().count() as f32 * font_id.size * 0.55).max(1.0),
            font_id.size.max(10.0),
        )
    } else {
        raw_size
    };
    BoxGeom {
        size,
        children: Vec::new(),
    }
}

fn measure_math_box_bounded(ui: &Ui, b: &MathBox, depth: usize) -> BoxGeom {
    if depth > MAX_LATEX_DEPTH {
        return BoxGeom {
            size: Vec2::ZERO,
            children: Vec::new(),
        };
    }
    match b {
        MathBox::Text {
            text,
            font_id,
            color,
            ..
        } => measure_text_box(ui, text, font_id, *color),
        MathBox::Row(boxes) => {
            let (mut width, mut height, mut children) = (0.0_f32, 0.0_f32, Vec::new());
            for child in boxes {
                let cg = measure_math_box_bounded(ui, child, depth + 1);
                width += cg.size.x;
                height = height.max(cg.size.y);
                children.push(cg);
            }
            BoxGeom {
                size: Vec2::new(width, height.max(12.0)),
                children,
            }
        }
        MathBox::Fraction {
            numerator,
            denominator,
        } => {
            let num_g = measure_math_box_bounded(ui, numerator, depth + 1);
            let den_g = measure_math_box_bounded(ui, denominator, depth + 1);
            let size = Vec2::new(
                num_g.size.x.max(den_g.size.x) + 6.0,
                num_g.size.y + den_g.size.y + 4.0,
            );
            BoxGeom {
                size,
                children: vec![num_g, den_g],
            }
        }
        MathBox::Sqrt(radicand) => {
            let rad_g = measure_math_box_bounded(ui, radicand, depth + 1);
            let size = Vec2::new(rad_g.size.y * 0.6 + rad_g.size.x + 4.0, rad_g.size.y + 3.0);
            BoxGeom {
                size,
                children: vec![rad_g],
            }
        }
        MathBox::Script { base, sup, sub } => {
            measure_script_box(ui, base.as_deref(), sup.as_deref(), sub.as_deref(), depth)
        }
        MathBox::Delimited {
            open,
            body,
            close,
            font_id,
            color,
        } => latex_delims::measure_delimited_box(ui, *open, body, *close, font_id, *color, depth),
        MathBox::Accent { base, .. } => latex_delims::measure_accent_box(ui, base, depth),
    }
}

fn paint_math_box(painter: &egui::Painter, origin: Pos2, math_box: &MathBox, geom: &BoxGeom) {
    paint_math_box_bounded(painter, origin, math_box, geom, 0);
}

#[allow(clippy::too_many_arguments)]
fn paint_fraction_box(
    painter: &egui::Painter,
    origin: Pos2,
    width: f32,
    numerator: &MathBox,
    denominator: &MathBox,
    num_g: &BoxGeom,
    den_g: &BoxGeom,
    depth: usize,
) {
    let num_x = origin.x + (width - num_g.size.x) / 2.0;
    paint_math_box_bounded(
        painter,
        Pos2::new(num_x, origin.y),
        numerator,
        num_g,
        depth + 1,
    );
    let line_y = origin.y + num_g.size.y + 2.0;
    let stroke_color = get_color(numerator, depth + 1).unwrap_or(Color32::WHITE);
    painter.line_segment(
        [
            Pos2::new(origin.x, line_y),
            Pos2::new(origin.x + width, line_y),
        ],
        Stroke::new(1.2_f32, stroke_color),
    );
    let den_x = origin.x + (width - den_g.size.x) / 2.0;
    paint_math_box_bounded(
        painter,
        Pos2::new(den_x, line_y + 2.0),
        denominator,
        den_g,
        depth + 1,
    );
}

fn paint_sqrt_box(
    painter: &egui::Painter,
    origin: Pos2,
    radicand: &MathBox,
    rad_g: &BoxGeom,
    depth: usize,
) {
    let sqrt_w = rad_g.size.y * 0.5;
    let color = get_color(radicand, depth + 1).unwrap_or(Color32::WHITE);
    let galley = painter.layout_no_wrap(
        "√".to_string(),
        FontId::proportional(rad_g.size.y.max(1.0)),
        color,
    );
    painter.galley(origin, galley, color);
    let rad_origin = Pos2::new(origin.x + sqrt_w + 2.0, origin.y + 3.0);
    paint_math_box_bounded(painter, rad_origin, radicand, rad_g, depth + 1);
    let bar_y = origin.y + 2.0;
    let bar_start = origin.x + sqrt_w + 1.0;
    painter.line_segment(
        [
            Pos2::new(bar_start, bar_y),
            Pos2::new(bar_start + rad_g.size.x + 2.0, bar_y),
        ],
        Stroke::new(1.2_f32, color),
    );
}

fn paint_script_box(
    painter: &egui::Painter,
    origin: Pos2,
    geom: &BoxGeom,
    base: Option<&MathBox>,
    sup: Option<&MathBox>,
    sub: Option<&MathBox>,
    depth: usize,
) {
    let mut idx = 0;
    let mut x = origin.x;
    if let (Some(b), Some(bg)) = (base, geom.children.get(idx)) {
        paint_math_box_bounded(painter, Pos2::new(x, origin.y), b, bg, depth + 1);
        x += bg.size.x;
        idx += 1;
    }
    if let (Some(sp), Some(spg)) = (sup, geom.children.get(idx)) {
        paint_math_box_bounded(painter, Pos2::new(x, origin.y), sp, spg, depth + 1);
        idx += 1;
    }
    if let (Some(sb), Some(sbg)) = (sub, geom.children.get(idx)) {
        let sub_y = origin.y + geom.size.y - sbg.size.y;
        paint_math_box_bounded(painter, Pos2::new(x, sub_y), sb, sbg, depth + 1);
    }
}

fn paint_row_boxes(
    painter: &egui::Painter,
    origin: Pos2,
    row_height: f32,
    boxes: &[MathBox],
    children: &[BoxGeom],
    depth: usize,
) {
    let mut x = origin.x;
    for (child_box, child_geom) in boxes.iter().zip(children) {
        let child_y = origin.y + (row_height - child_geom.size.y) * 0.5;
        paint_math_box_bounded(
            painter,
            Pos2::new(x, child_y),
            child_box,
            child_geom,
            depth + 1,
        );
        x += child_geom.size.x;
    }
}

fn paint_math_box_bounded(
    painter: &egui::Painter,
    origin: Pos2,
    math_box: &MathBox,
    geom: &BoxGeom,
    depth: usize,
) {
    if depth > MAX_LATEX_DEPTH {
        return;
    }
    match (math_box, geom.children.as_slice()) {
        (
            MathBox::Text {
                text,
                font_id,
                color,
                ..
            },
            _,
        ) => {
            let bounded_text = truncate_chars(text, MAX_LAYOUT_CHARS).to_owned();
            let galley = painter.layout_no_wrap(bounded_text, font_id.clone(), *color);
            painter.galley(origin, galley, *color);
        }
        (MathBox::Row(boxes), children) => {
            paint_row_boxes(painter, origin, geom.size.y, boxes, children, depth);
        }
        (
            MathBox::Fraction {
                numerator,
                denominator,
            },
            [num_g, den_g],
        ) => {
            paint_fraction_box(
                painter,
                origin,
                geom.size.x,
                numerator,
                denominator,
                num_g,
                den_g,
                depth,
            );
        }
        (MathBox::Sqrt(radicand), [rad_g]) => {
            paint_sqrt_box(painter, origin, radicand, rad_g, depth);
        }
        (MathBox::Script { base, sup, sub }, _) => {
            paint_script_box(
                painter,
                origin,
                geom,
                base.as_deref(),
                sup.as_deref(),
                sub.as_deref(),
                depth,
            );
        }
        (
            MathBox::Delimited {
                open,
                body,
                close,
                font_id,
                color,
            },
            [body_g],
        ) => {
            let spec = latex_delims::DelimPaintSpec {
                open: *open,
                body,
                close: *close,
                font_id,
                color: *color,
            };
            latex_delims::paint_delimited_box(painter, origin, &spec, geom, body_g, depth);
        }
        (MathBox::Accent { base, kind, color }, [base_g]) => {
            latex_delims::paint_accent_box(
                painter, origin, base, *kind, *color, geom, base_g, depth,
            );
        }
        _ => {}
    }
}

fn get_color(b: &MathBox, depth: usize) -> Option<Color32> {
    if depth > MAX_LATEX_DEPTH {
        return None;
    }
    match b {
        MathBox::Text { color, .. } => Some(*color),
        MathBox::Row(vec) => vec.first().and_then(|child| get_color(child, depth + 1)),
        MathBox::Fraction { numerator, .. } => get_color(numerator, depth + 1),
        MathBox::Sqrt(r) => get_color(r, depth + 1),
        MathBox::Script { base, .. } => base
            .as_deref()
            .and_then(|child| get_color(child, depth + 1)),
        MathBox::Delimited { color, .. } | MathBox::Accent { color, .. } => Some(*color),
    }
}

#[cfg(test)]
#[path = "latex_audit_tests.rs"]
mod latex_audit_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use egui::{Align, Color32, FontFamily, FontId};

    fn test_font() -> FontId {
        FontId::new(14.0, FontFamily::Proportional)
    }

    #[test]
    fn test_latex_box_ast_creation() {
        let font = test_font();
        let color = Color32::WHITE;

        let box_frac = parse_latex_to_box(r"\frac{a}{b}", &font, color).unwrap();
        match box_frac {
            MathBox::Fraction {
                numerator,
                denominator,
            } => {
                assert_eq!(
                    *numerator,
                    MathBox::Text {
                        text: "a".into(),
                        font_id: font.clone(),
                        color,
                        italics: true
                    }
                );
                assert_eq!(
                    *denominator,
                    MathBox::Text {
                        text: "b".into(),
                        font_id: font.clone(),
                        color,
                        italics: true
                    }
                );
            }
            _ => panic!("Expected MathBox::Fraction"),
        }

        let box_sqrt = parse_latex_to_box(r"\sqrt{x}", &font, color).unwrap();
        match box_sqrt {
            MathBox::Sqrt(radicand) => {
                assert_eq!(
                    *radicand,
                    MathBox::Text {
                        text: "x".into(),
                        font_id: font.clone(),
                        color,
                        italics: true
                    }
                );
            }
            _ => panic!("Expected MathBox::Sqrt"),
        }
    }

    #[test]
    fn test_latex_symbol_substitution() {
        let font = test_font();
        let job = parse_latex_with_font(r"\alpha + \beta = \gamma", font, Color32::WHITE);
        assert_eq!(job.text, "α + β = γ");
    }

    #[test]
    fn test_latex_superscript_and_subscript() {
        let font = test_font();
        let color = Color32::WHITE;

        let job_super = parse_latex_with_font("x^2", font.clone(), color);
        assert_eq!(job_super.text, "x2");
        assert_eq!(job_super.sections.len(), 2);
        assert_eq!(job_super.sections[0].format.valign, Align::Center);
        assert_eq!(job_super.sections[1].format.valign, Align::Min);

        let job_sub = parse_latex_with_font("y_1", font.clone(), color);
        assert_eq!(job_sub.text, "y1");
        assert_eq!(job_sub.sections.len(), 2);
        assert_eq!(job_sub.sections[0].format.valign, Align::Center);
        assert_eq!(job_sub.sections[1].format.valign, Align::Max);
    }

    #[test]
    fn test_latex_fraction() {
        let font = test_font();
        let job = parse_latex_with_font(r"\frac{dy}{dt}", font, Color32::WHITE);
        assert!(job.text.contains("dy") && job.text.contains("dt"));
        assert!(job.text.contains("──"));
    }

    #[test]
    fn test_latex_functions_and_text() {
        let font = test_font();

        let job_func = parse_latex_with_font(r"\sin(x)", font.clone(), Color32::WHITE);
        assert_eq!(job_func.text, "sin(x)");
        assert!(!job_func.sections[0].format.italics);

        let job_text = parse_latex_with_font(r"\text{Speed} v", font, Color32::WHITE);
        assert_eq!(job_text.text, "Speed v");
    }

    #[test]
    fn test_latex_inline_math() {
        let font = test_font();
        let job = parse_latex_with_font("Time $t$ (s)", font, Color32::WHITE);
        assert_eq!(job.text, "Time t (s)");
    }

    #[test]
    fn test_latex_fallback_invalid_syntax() {
        let font = test_font();

        let job_unclosed_brace = parse_latex_with_font(r"\frac{a}{b", font.clone(), Color32::WHITE);
        assert_eq!(job_unclosed_brace.text, r"\frac{a}{b");

        let job_unclosed_dollar = parse_latex_with_font("Time $t", font, Color32::WHITE);
        assert!(!job_unclosed_dollar.text.is_empty());
    }

    #[test]
    fn test_latex_empty_string() {
        let font = test_font();
        let job = parse_latex_with_font("", font, Color32::WHITE);
        assert!(job.text.is_empty());
        assert!(job.sections.is_empty());
    }

    #[test]
    fn test_generate_latex_2d_box_layout_screenshot() {
        let ctx = egui::Context::default();
        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                render_math_label(
                    ui,
                    r"\frac{a}{b} + \sqrt{x^2+1}",
                    TextStyle::Heading,
                    Color32::WHITE,
                );
                render_math_label(
                    ui,
                    r"\beta=0.20, \gamma=0.10, N=1000",
                    TextStyle::Body,
                    Color32::WHITE,
                );
            });
        });

        let datasets = vec![(
            r"\frac{a}{b} + \sqrt{x^2+1}".to_string(),
            vec![[0.0, 0.0], [1.0, 1.0]],
        )];
        let options = crate::export::ExportOptions::default();
        if let Ok(png_bytes) = crate::export::export_custom_png(&datasets, &options) {
            let _ = std::fs::write("/tmp/latex_2d_box_layout.png", png_bytes);
        }
    }

    #[test]
    fn test_nested_sqrt_and_csv_header_output_bounds() {
        let header = format!("{}x{}", r"\sqrt{".repeat(19), "}".repeat(19));
        let datasets = crate::parse_csv(&format!("x,{header}\n0,1\n"));
        let job = parse_latex_with_font(&datasets[0].name, test_font(), Color32::WHITE);
        assert!(job.text.chars().count() <= MAX_LAYOUT_CHARS);

        let nested_in_budget = format!("{}x{}", r"\sqrt{".repeat(16), "}".repeat(16));
        let in_budget_job = parse_latex_with_font(&nested_in_budget, test_font(), Color32::WHITE);
        assert_eq!(in_budget_job.text.chars().count(), 2 * 16 + 1);
        assert!(!in_budget_job.text.contains("\u{0305}\u{0305}"));
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn run_with_small_stack<F: FnOnce() + Send + 'static>(f: F) {
        std::thread::Builder::new()
            .stack_size(512 * 1024)
            .spawn(f)
            .unwrap()
            .join()
            .unwrap();
    }

    #[cfg(target_arch = "wasm32")]
    fn run_with_small_stack<F: FnOnce()>(f: F) {
        f();
    }

    #[test]
    fn test_deep_group_recursion_bounded_on_small_stack() {
        let deep = format!("{}x{}", "{".repeat(2500), "}".repeat(2500));
        run_with_small_stack(move || {
            let job = parse_latex_with_font(&deep, FontId::proportional(14.0), Color32::WHITE);
            assert!(job.text.chars().count() <= MAX_FALLBACK_CHARS);
        });

        let deep_under_input_cap = format!("{}x{}", "{".repeat(900), "}".repeat(900));
        assert!(deep_under_input_cap.len() <= MAX_LATEX_INPUT_BYTES);
        run_with_small_stack(move || {
            assert!(
                parse_latex_to_box(
                    &deep_under_input_cap,
                    &FontId::proportional(14.0),
                    Color32::WHITE,
                )
                .is_err()
            );
        });
    }

    #[test]
    fn test_node_budget_and_utf8_fallback_truncation() {
        let font = test_font();
        let many_nodes = "+".repeat(MAX_MATH_BOX_NODES + 10);
        assert!(many_nodes.len() <= MAX_LATEX_INPUT_BYTES);
        assert!(parse_latex_to_box(&many_nodes, &font, Color32::WHITE).is_err());

        let multibyte = "αβγ√∑".repeat(600);
        let job = parse_latex_with_font(&multibyte, font, Color32::WHITE);
        assert_eq!(job.text.chars().count(), MAX_FALLBACK_CHARS);

        let ctx = egui::Context::default();
        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                render_math_label(ui, &multibyte, TextStyle::Body, Color32::WHITE);
            });
        });
    }

    #[test]
    fn test_direct_math_box_traversal_and_layout_caps() {
        let font = test_font();
        let mut deep_box = MathBox::Text {
            text: "x".repeat(MAX_LAYOUT_CHARS * 2),
            font_id: font.clone(),
            color: Color32::WHITE,
            italics: false,
        };
        let mut job = LayoutJob::default();
        math_box_to_layout_job(&deep_box, &mut job);
        assert_eq!(job.text.chars().count(), MAX_LAYOUT_CHARS);

        let frac_box = MathBox::Fraction {
            numerator: Box::new(deep_box.clone()),
            denominator: Box::new(deep_box.clone()),
        };
        let mut frac_job = LayoutJob::default();
        math_box_to_layout_job(&frac_box, &mut frac_job);
        assert_eq!(frac_job.text.chars().count(), MAX_LAYOUT_CHARS);

        for _ in 0..64 {
            deep_box = MathBox::Sqrt(Box::new(deep_box));
        }
        let ctx = egui::Context::default();
        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                render_math_box_ui(ui, &deep_box);
            });
        });
    }

    #[test]
    fn test_latex_quantum_symbols_blackboard_and_text_angstrom() {
        let font = test_font();
        let sym_job = parse_latex_with_font(
            r"\hbar \langle S_x \rangle + \dots + 90\deg + \AA",
            font.clone(),
            Color32::WHITE,
        );
        assert_eq!(sym_job.text, "ℏ ⟨ Sx ⟩ + … + 90° + Å");

        let bb_job = parse_latex_with_font(
            r"\mathbb{R} \times \mathbb{N} \times \mathbb{Z} \times \mathbb{Q} \times \mathbb{C}",
            font.clone(),
            Color32::WHITE,
        );
        assert_eq!(bb_job.text, "ℝ × ℕ × ℤ × ℚ × ℂ");

        let crystal_job = parse_latex_with_font(r"a = 5.43\ \text{\AA}", font, Color32::WHITE);
        assert_eq!(crystal_job.text, "a = 5.43 Å");
    }

    #[test]
    fn test_latex_delimiters_and_combining_accents() {
        let font = test_font();
        let delim_job = parse_latex_with_font(
            r"\left( x \right) + \left[ y \right] + \left\{ z \right\} + \left. \frac{a}{b} \right|",
            font.clone(),
            Color32::WHITE,
        );
        assert!(delim_job.text.contains("( x ) + [ y ] + { z } +"));
        assert!(delim_job.text.ends_with('|'));
        assert!(!delim_job.text.contains("left") && !delim_job.text.contains("right"));
        assert!(!delim_job.text.contains('.'));

        let accent_job = parse_latex_with_font(
            r"-\vec{\mu}\cdot\vec{B} + \hat{z} + \bar{g}_{Na} + \dot{x} + \ddot{\mathbf{x}} + \tilde{\psi}",
            font,
            Color32::WHITE,
        );
        assert!(accent_job.text.contains("μ\u{20D7}·B\u{20D7}"));
        assert!(accent_job.text.contains("z\u{0302}"));
        assert!(accent_job.text.contains("g\u{0304}Na"));
        assert!(accent_job.text.contains("x\u{0307}"));
        assert!(accent_job.text.contains("x\u{0308}"));
        assert!(accent_job.text.contains("ψ\u{0303}"));
    }

    #[test]
    fn test_prose_with_inline_math_plain_and_fractions() {
        let font = test_font();
        let citation = "A Quantitative Description of g_na_bar (Hodgkin & Huxley, 1952)";
        let plain_job = parse_prose_with_font(citation, font.clone(), Color32::WHITE);
        assert_eq!(plain_job.text, citation);
        assert!(plain_job.sections.iter().all(|s| !s.format.italics));

        let inline_job = parse_prose_with_font(
            r"Slope $\frac{dy}{dt}$ and ratio $\frac{a + b}{c + d}$ with $\frac{(x + 1)}{2}$.",
            font,
            Color32::WHITE,
        );
        assert_eq!(
            inline_job.text,
            "Slope dy⁄dt and ratio (a + b)⁄(c + d) with (x + 1)⁄2."
        );
        assert!(!inline_job.text.contains('\n') && !inline_job.text.contains('─'));
        assert!(!inline_job.sections[0].format.italics);
    }

    #[test]
    fn test_prose_with_inline_math_fallbacks_and_caps() {
        let font = test_font();
        let unclosed =
            parse_prose_with_font("Unclosed $formula fallback", font.clone(), Color32::WHITE);
        assert_eq!(unclosed.text, "Unclosed $formula fallback");
        assert!(unclosed.sections.iter().all(|s| !s.format.italics));

        let bad_brace =
            parse_prose_with_font(r"Broken $\frac{a}{b$ end", font.clone(), Color32::WHITE);
        assert_eq!(bad_brace.text, r"Broken $\frac{a}{b$ end");

        let too_many_nodes = format!("Val ${}$", "+".repeat(MAX_MATH_BOX_NODES + 8));
        let capped_nodes = parse_prose_with_font(&too_many_nodes, font.clone(), Color32::WHITE);
        assert_eq!(capped_nodes.text, too_many_nodes);

        let huge_prose = format!("$x$ {}", "αβγ".repeat(900));
        let capped_bytes = parse_prose_with_font(&huge_prose, font, Color32::WHITE);
        assert_eq!(capped_bytes.text.chars().count(), MAX_FALLBACK_CHARS);

        let ctx = egui::Context::default();
        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                let job = parse_prose_with_inline_math(
                    r"Spin $H = -\vec{\mu}\cdot\vec{B}$ in $\mathbb{R}^3$.",
                    TextStyle::Body,
                    ui,
                    Color32::WHITE,
                );
                assert!(job.text.contains("ℝ3") && job.text.contains("μ\u{20D7}"));
            });
        });
    }
}
