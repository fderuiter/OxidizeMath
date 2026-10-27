use egui::{
    Align, Color32, FontId, Pos2, Response, Sense, Stroke, TextStyle, Ui, Vec2,
    text::{LayoutJob, TextFormat},
};

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
        text: text.to_string(),
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
        Err(_) => {
            job.append(
                text,
                0.0,
                TextFormat {
                    font_id,
                    color,
                    ..Default::default()
                },
            );
        }
    }

    job
}

/// Parses LaTeX string into `MathBox` AST tree.
#[allow(clippy::result_unit_err)]
pub fn parse_latex_to_box(
    text: &str,
    base_font_id: &FontId,
    color: Color32,
) -> Result<MathBox, ()> {
    if text.is_empty() {
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
                let is_math = idx % 2 == 1;
                if is_math {
                    let sub_boxes = parse_math_expression_boxes(part, base_font_id, color)?;
                    boxes.extend(sub_boxes);
                } else {
                    boxes.push(MathBox::Text {
                        text: (*part).to_string(),
                        font_id: base_font_id.clone(),
                        color,
                        italics: false,
                    });
                }
            }
            return Ok(simplify_boxes(boxes));
        }
    }

    let boxes = parse_math_expression_boxes(text, base_font_id, color)?;
    Ok(simplify_boxes(boxes))
}

fn simplify_boxes(boxes: Vec<MathBox>) -> MathBox {
    if boxes.len() == 1 {
        boxes.into_iter().next().unwrap()
    } else {
        MathBox::Row(boxes)
    }
}

fn parse_math_expression_boxes(
    expr: &str,
    base_font_id: &FontId,
    color: Color32,
) -> Result<Vec<MathBox>, ()> {
    let chars: Vec<char> = expr.chars().collect();
    let mut i = 0;
    let len = chars.len();
    let mut boxes = Vec::new();

    let script_font_size = (base_font_id.size * 0.75).max(8.0);
    let script_font_id = FontId::new(script_font_size, base_font_id.family.clone());

    while i < len {
        let c = chars[i];

        if c == '^' || c == '_' {
            let is_sup = c == '^';
            i += 1;
            let (script_str, next_i) = read_script_arg(&chars, i)?;
            i = next_i;

            let script_box = parse_latex_to_box(&script_str, &script_font_id, color)?;
            let prev_box = if !boxes.is_empty() { boxes.pop() } else { None };

            let (sup_val, sub_val) = if is_sup {
                (Some(Box::new(script_box)), None)
            } else {
                (None, Some(Box::new(script_box)))
            };

            match prev_box {
                Some(MathBox::Script { base, sup, sub }) => {
                    let new_sup = if is_sup { sup_val } else { sup };
                    let new_sub = if !is_sup { sub_val } else { sub };
                    boxes.push(MathBox::Script {
                        base,
                        sup: new_sup,
                        sub: new_sub,
                    });
                }
                Some(b) => {
                    boxes.push(MathBox::Script {
                        base: Some(Box::new(b)),
                        sup: sup_val,
                        sub: sub_val,
                    });
                }
                None => {
                    boxes.push(MathBox::Script {
                        base: None,
                        sup: sup_val,
                        sub: sub_val,
                    });
                }
            }
        } else if c == '\\' {
            i += 1;
            let (cmd, next_i) = read_command(&chars, i)?;
            i = next_i;
            process_command_box(&cmd, &chars, &mut i, base_font_id, color, &mut boxes)?;
        } else if c == '{' {
            i += 1;
            let (group_str, next_i) = read_group_arg(&chars, i)?;
            i = next_i;
            let group_box = parse_latex_to_box(&group_str, base_font_id, color)?;
            boxes.push(group_box);
        } else if c == '}' {
            return Err(());
        } else if c.is_ascii_alphabetic() {
            let start = i;
            while i < len && chars[i].is_ascii_alphabetic() {
                i += 1;
            }
            let word: String = chars[start..i].iter().collect();
            let italics = word.len() == 1;
            boxes.push(MathBox::Text {
                text: word,
                font_id: base_font_id.clone(),
                color,
                italics,
            });
        } else {
            let s: String = std::iter::once(c).collect();
            boxes.push(MathBox::Text {
                text: s,
                font_id: base_font_id.clone(),
                color,
                italics: false,
            });
            i += 1;
        }
    }

    Ok(boxes)
}

fn process_command_box(
    cmd: &str,
    chars: &[char],
    i: &mut usize,
    base_font_id: &FontId,
    color: Color32,
    boxes: &mut Vec<MathBox>,
) -> Result<(), ()> {
    if let Some(sym) = map_latex_symbol(cmd) {
        if cmd == "sqrt" {
            skip_whitespace(chars, i);
            if *i < chars.len() && chars[*i] == '{' {
                *i += 1;
                let (arg, next_i) = read_group_arg(chars, *i)?;
                *i = next_i;
                let radicand_box = parse_latex_to_box(&arg, base_font_id, color)?;
                boxes.push(MathBox::Sqrt(Box::new(radicand_box)));
                return Ok(());
            }
        }
        boxes.push(MathBox::Text {
            text: sym.to_string(),
            font_id: base_font_id.clone(),
            color,
            italics: false,
        });
        return Ok(());
    }

    match cmd {
        "text" | "mathrm" | "mathbf" => {
            skip_whitespace(chars, i);
            if *i < chars.len() && chars[*i] == '{' {
                *i += 1;
                let (arg, next_i) = read_group_arg(chars, *i)?;
                *i = next_i;
                boxes.push(MathBox::Text {
                    text: arg,
                    font_id: base_font_id.clone(),
                    color,
                    italics: false,
                });
            } else {
                return Err(());
            }
        }
        "frac" => {
            skip_whitespace(chars, i);
            if *i < chars.len() && chars[*i] == '{' {
                *i += 1;
                let (num, next_i) = read_group_arg(chars, *i)?;
                *i = next_i;
                skip_whitespace(chars, i);
                if *i < chars.len() && chars[*i] == '{' {
                    *i += 1;
                    let (den, next_i) = read_group_arg(chars, *i)?;
                    *i = next_i;

                    let num_box = parse_latex_to_box(&num, base_font_id, color)?;
                    let den_box = parse_latex_to_box(&den, base_font_id, color)?;
                    boxes.push(MathBox::Fraction {
                        numerator: Box::new(num_box),
                        denominator: Box::new(den_box),
                    });
                } else {
                    return Err(());
                }
            } else {
                return Err(());
            }
        }
        "sin" | "cos" | "tan" | "exp" | "ln" | "log" | "lim" | "max" | "min" | "det" | "arcsin"
        | "arccos" | "arctan" | "sinh" | "cosh" | "tanh" => {
            boxes.push(MathBox::Text {
                text: cmd.to_string(),
                font_id: base_font_id.clone(),
                color,
                italics: false,
            });
        }
        "quad" => {
            boxes.push(MathBox::Text {
                text: "  ".to_string(),
                font_id: base_font_id.clone(),
                color,
                italics: false,
            });
        }
        "qquad" => {
            boxes.push(MathBox::Text {
                text: "    ".to_string(),
                font_id: base_font_id.clone(),
                color,
                italics: false,
            });
        }
        "," | ":" | ";" => {
            boxes.push(MathBox::Text {
                text: " ".to_string(),
                font_id: base_font_id.clone(),
                color,
                italics: false,
            });
        }
        "\\" | "newline" => {
            boxes.push(MathBox::Text {
                text: "\n".to_string(),
                font_id: base_font_id.clone(),
                color,
                italics: false,
            });
        }
        _ => {
            boxes.push(MathBox::Text {
                text: cmd.to_string(),
                font_id: base_font_id.clone(),
                color,
                italics: false,
            });
        }
    }

    Ok(())
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

fn map_latex_symbol(cmd: &str) -> Option<&'static str> {
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
        "sqrt" => Some("√"),
        "leq" | "le" => Some("≤"),
        "geq" | "ge" => Some("≥"),
        "neq" | "ne" => Some("≠"),
        "approx" => Some("≈"),
        "equiv" => Some("≡"),
        "sim" => Some("~"),
        "to" | "rightarrow" => Some("→"),
        "leftarrow" => Some("←"),
        "Rightarrow" => Some("⇒"),
        "Leftarrow" => Some("⇐"),
        "in" => Some("∈"),
        "notin" => Some("∉"),
        "subset" => Some("⊂"),
        "forall" => Some("∀"),
        "exists" => Some("∃"),
        _ => None,
    }
}

/// Converts a `MathBox` tree into an `egui::text::LayoutJob`.
pub fn math_box_to_layout_job(box_node: &MathBox, job: &mut LayoutJob) {
    match box_node {
        MathBox::Text {
            text,
            font_id,
            color,
            italics,
        } => {
            job.append(
                text,
                0.0,
                TextFormat {
                    font_id: font_id.clone(),
                    color: *color,
                    italics: *italics,
                    valign: Align::Center,
                    ..Default::default()
                },
            );
        }
        MathBox::Row(boxes) => {
            for b in boxes {
                math_box_to_layout_job(b, job);
            }
        }
        MathBox::Fraction {
            numerator,
            denominator,
        } => {
            let mut num_job = LayoutJob::default();
            math_box_to_layout_job(numerator, &mut num_job);
            let mut den_job = LayoutJob::default();
            math_box_to_layout_job(denominator, &mut den_job);

            let num_text = &num_job.text;
            let den_text = &den_job.text;
            let width_chars = num_text
                .chars()
                .count()
                .max(den_text.chars().count())
                .max(1);
            let bar = "─".repeat(width_chars);

            let first_fmt = num_job
                .sections
                .first()
                .map(|s| s.format.clone())
                .unwrap_or_default();

            job.append(num_text, 0.0, first_fmt.clone());
            job.append(&format!("\n{}\n", bar), 0.0, first_fmt.clone());
            job.append(den_text, 0.0, first_fmt);
        }
        MathBox::Sqrt(radicand) => {
            let mut rad_job = LayoutJob::default();
            math_box_to_layout_job(radicand, &mut rad_job);
            let first_fmt = rad_job
                .sections
                .first()
                .map(|s| s.format.clone())
                .unwrap_or_default();

            job.append("√", 0.0, first_fmt.clone());
            let overbar_text: String = rad_job
                .text
                .chars()
                .flat_map(|c| vec![c, '\u{0305}'])
                .collect();
            job.append(&overbar_text, 0.0, first_fmt);
        }
        MathBox::Script { base, sup, sub } => {
            if let Some(b) = base {
                math_box_to_layout_job(b, job);
            }
            if let Some(s) = sup {
                let mut sup_job = LayoutJob::default();
                math_box_to_layout_job(s, &mut sup_job);
                for section in sup_job.sections {
                    let txt = &sup_job.text[section.byte_range.clone()];
                    let mut fmt = section.format;
                    fmt.valign = Align::Min;
                    job.append(txt, section.leading_space, fmt);
                }
            }
            if let Some(s) = sub {
                let mut sub_job = LayoutJob::default();
                math_box_to_layout_job(s, &mut sub_job);
                for section in sub_job.sections {
                    let txt = &sub_job.text[section.byte_range.clone()];
                    let mut fmt = section.format;
                    fmt.valign = Align::Max;
                    job.append(txt, section.leading_space, fmt);
                }
            }
        }
    }
}

#[derive(Clone, Debug)]
struct BoxGeom {
    size: Vec2,
    children: Vec<BoxGeom>,
}

fn measure_math_box(ui: &Ui, b: &MathBox) -> BoxGeom {
    match b {
        MathBox::Text {
            text,
            font_id,
            color,
            ..
        } => {
            let galley = ui
                .painter()
                .layout_no_wrap(text.clone(), font_id.clone(), *color);
            let size = galley.size();
            BoxGeom {
                size,
                children: Vec::new(),
            }
        }
        MathBox::Row(boxes) => {
            let mut width: f32 = 0.0;
            let mut height: f32 = 0.0;
            let mut children = Vec::new();

            for child in boxes {
                let cg = measure_math_box(ui, child);
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
            let num_g = measure_math_box(ui, numerator);
            let den_g = measure_math_box(ui, denominator);

            let width = num_g.size.x.max(den_g.size.x) + 6.0;
            let height = num_g.size.y + den_g.size.y + 4.0;

            BoxGeom {
                size: Vec2::new(width, height),
                children: vec![num_g, den_g],
            }
        }
        MathBox::Sqrt(radicand) => {
            let rad_g = measure_math_box(ui, radicand);
            let sqrt_w = rad_g.size.y * 0.6;
            let width = sqrt_w + rad_g.size.x + 4.0;
            let height = rad_g.size.y + 3.0;

            BoxGeom {
                size: Vec2::new(width, height),
                children: vec![rad_g],
            }
        }
        MathBox::Script { base, sup, sub } => {
            let mut children = Vec::new();
            let mut width: f32 = 0.0;
            let mut base_h: f32 = 12.0;

            if let Some(b) = base {
                let bg = measure_math_box(ui, b);
                width += bg.size.x;
                base_h = bg.size.y;
                children.push(bg);
            }

            let mut script_w: f32 = 0.0;
            if let Some(sp) = sup {
                let spg = measure_math_box(ui, sp);
                script_w = script_w.max(spg.size.x);
                children.push(spg);
            }
            if let Some(sb) = sub {
                let sbg = measure_math_box(ui, sb);
                script_w = script_w.max(sbg.size.x);
                children.push(sbg);
            }

            width += script_w;

            BoxGeom {
                size: Vec2::new(width, base_h * 1.2),
                children,
            }
        }
    }
}

fn paint_math_box(painter: &egui::Painter, origin: Pos2, math_box: &MathBox, geom: &BoxGeom) {
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
            let galley = painter.layout_no_wrap(text.clone(), font_id.clone(), *color);
            painter.galley(origin, galley, *color);
        }
        (MathBox::Row(boxes), children) => {
            let mut x = origin.x;
            for (child_box, child_geom) in boxes.iter().zip(children) {
                paint_math_box(painter, Pos2::new(x, origin.y), child_box, child_geom);
                x += child_geom.size.x;
            }
        }
        (
            MathBox::Fraction {
                numerator,
                denominator,
            },
            [num_g, den_g],
        ) => {
            let width = geom.size.x;

            let num_x = origin.x + (width - num_g.size.x) / 2.0;
            paint_math_box(painter, Pos2::new(num_x, origin.y), numerator, num_g);

            let line_y = origin.y + num_g.size.y + 2.0;
            let stroke_color = get_color(numerator).unwrap_or(Color32::WHITE);
            painter.line_segment(
                [
                    Pos2::new(origin.x, line_y),
                    Pos2::new(origin.x + width, line_y),
                ],
                Stroke::new(1.2, stroke_color),
            );

            let den_x = origin.x + (width - den_g.size.x) / 2.0;
            let den_y = line_y + 2.0;
            paint_math_box(painter, Pos2::new(den_x, den_y), denominator, den_g);
        }
        (MathBox::Sqrt(radicand), [rad_g]) => {
            let sqrt_w = rad_g.size.y * 0.5;
            let color = get_color(radicand).unwrap_or(Color32::WHITE);

            let galley =
                painter.layout_no_wrap("√".to_string(), FontId::proportional(rad_g.size.y), color);
            painter.galley(origin, galley, color);

            let rad_origin = Pos2::new(origin.x + sqrt_w + 2.0, origin.y + 3.0);
            paint_math_box(painter, rad_origin, radicand, rad_g);

            let bar_y = origin.y + 2.0;
            let bar_start = origin.x + sqrt_w + 1.0;
            let bar_end = bar_start + rad_g.size.x + 2.0;
            painter.line_segment(
                [Pos2::new(bar_start, bar_y), Pos2::new(bar_end, bar_y)],
                Stroke::new(1.2, color),
            );
        }
        (MathBox::Script { base, sup, sub }, children) => {
            let mut idx = 0;
            let mut x = origin.x;

            if let (Some(b), Some(bg)) = (base, children.get(idx)) {
                paint_math_box(painter, Pos2::new(x, origin.y), b, bg);
                x += bg.size.x;
                idx += 1;
            }

            let script_x = x;
            if let (Some(sp), Some(spg)) = (sup, children.get(idx)) {
                paint_math_box(painter, Pos2::new(script_x, origin.y), sp, spg);
                idx += 1;
            }

            if let (Some(sb), Some(sbg)) = (sub, children.get(idx)) {
                let sub_y = origin.y + geom.size.y - sbg.size.y;
                paint_math_box(painter, Pos2::new(script_x, sub_y), sb, sbg);
            }
        }
        _ => {}
    }
}

fn get_color(b: &MathBox) -> Option<Color32> {
    match b {
        MathBox::Text { color, .. } => Some(*color),
        MathBox::Row(vec) => vec.first().and_then(get_color),
        MathBox::Fraction { numerator, .. } => get_color(numerator),
        MathBox::Sqrt(r) => get_color(r),
        MathBox::Script { base, .. } => base.as_deref().and_then(get_color),
    }
}

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
}
