use super::{
    BoxGeom, MAX_LATEX_DEPTH, MathBox, append_bounded, consume_node, math_box_to_layout_job_mode,
    measure_math_box_bounded, paint_math_box_bounded, parse_latex_to_box_bounded, read_command,
    read_group_arg, skip_whitespace,
};
use egui::{
    Align, Color32, FontId, Pos2, Stroke, Ui, Vec2,
    text::{LayoutJob, TextFormat},
};

/// Paired or single mathematical delimiter rendered by `MathBox::Delimited`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MathDelimiter {
    /// Opening parenthesis `(`.
    ParenOpen,
    /// Closing parenthesis `)`.
    ParenClose,
    /// Opening square bracket `[`.
    BracketOpen,
    /// Closing square bracket `]`.
    BracketClose,
    /// Vertical bar `|`.
    Pipe,
    /// Double vertical norm bar `‖` (`\|`).
    DoublePipe,
    /// Opening curly brace `{`.
    BraceOpen,
    /// Closing curly brace `}`.
    BraceClose,
    /// Opening angle bracket `⟨`.
    AngleOpen,
    /// Closing angle bracket `⟩`.
    AngleClose,
    /// Opening floor bracket `⌊` (`\lfloor`).
    FloorOpen,
    /// Closing floor bracket `⌋` (`\rfloor`).
    FloorClose,
    /// Opening ceiling bracket `⌈` (`\lceil`).
    CeilOpen,
    /// Closing ceiling bracket `⌉` (`\rceil`).
    CeilClose,
    /// Null delimiter `.` (invisible boundary in `\left.` / `\right.`).
    None,
}

impl MathDelimiter {
    /// Textual glyph representation for single-line and `LayoutJob` rendering.
    #[must_use]
    pub const fn glyph(self) -> &'static str {
        match self {
            Self::ParenOpen => "(",
            Self::ParenClose => ")",
            Self::BracketOpen => "[",
            Self::BracketClose => "]",
            Self::Pipe => "|",
            Self::DoublePipe => "‖",
            Self::BraceOpen => "{",
            Self::BraceClose => "}",
            Self::AngleOpen => "⟨",
            Self::AngleClose => "⟩",
            Self::FloorOpen => "⌊",
            Self::FloorClose => "⌋",
            Self::CeilOpen => "⌈",
            Self::CeilClose => "⌉",
            Self::None => "",
        }
    }
}

/// Over-symbol accent kind rendered by `MathBox::Accent`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MathAccent {
    /// Single over-dot time derivative (`\dot{x}`).
    Dot,
    /// Double over-dot second derivative (`\ddot{x}`).
    Ddot,
    /// Circumflex hat (`\hat{x}`).
    Hat,
    /// Overbar (`\bar{x}`).
    Bar,
    /// Vector arrow (`\vec{x}`).
    VecArrow,
    /// Tilde wave (`\tilde{\psi}`).
    Tilde,
}

impl MathAccent {
    const fn combining_mark(self) -> char {
        match self {
            Self::Dot => '\u{0307}',
            Self::Ddot => '\u{0308}',
            Self::Hat => '\u{0302}',
            Self::Bar => '\u{0304}',
            Self::VecArrow => '\u{20D7}',
            Self::Tilde => '\u{0303}',
        }
    }
}

pub(super) fn accent_kind_from_cmd(cmd: &str) -> Option<MathAccent> {
    match cmd {
        "dot" => Some(MathAccent::Dot),
        "ddot" => Some(MathAccent::Ddot),
        "hat" => Some(MathAccent::Hat),
        "bar" | "overline" => Some(MathAccent::Bar),
        "vec" => Some(MathAccent::VecArrow),
        "tilde" => Some(MathAccent::Tilde),
        _ => None,
    }
}

pub(super) fn read_delimiter(chars: &[char], i: &mut usize) -> Result<MathDelimiter, ()> {
    skip_whitespace(chars, i);
    if *i >= chars.len() {
        return Err(());
    }
    let c = chars[*i];
    *i += 1;
    match c {
        '(' => Ok(MathDelimiter::ParenOpen),
        ')' => Ok(MathDelimiter::ParenClose),
        '[' => Ok(MathDelimiter::BracketOpen),
        ']' => Ok(MathDelimiter::BracketClose),
        '|' => Ok(MathDelimiter::Pipe),
        '.' => Ok(MathDelimiter::None),
        '\\' => {
            let (cmd, next_i) = read_command(chars, *i)?;
            *i = next_i;
            match cmd.as_str() {
                "{" | "lbrace" => Ok(MathDelimiter::BraceOpen),
                "}" | "rbrace" => Ok(MathDelimiter::BraceClose),
                "|" | "lVert" | "rVert" | "Vert" => Ok(MathDelimiter::DoublePipe),
                "lvert" | "rvert" | "mid" | "vert" => Ok(MathDelimiter::Pipe),
                "langle" => Ok(MathDelimiter::AngleOpen),
                "rangle" => Ok(MathDelimiter::AngleClose),
                "lfloor" => Ok(MathDelimiter::FloorOpen),
                "rfloor" => Ok(MathDelimiter::FloorClose),
                "lceil" => Ok(MathDelimiter::CeilOpen),
                "rceil" => Ok(MathDelimiter::CeilClose),
                _ => Ok(MathDelimiter::None),
            }
        }
        _ => Err(()),
    }
}

fn find_matching_right(chars: &[char], start: usize) -> Option<(String, usize)> {
    let mut pos = start;
    let mut brace_depth: usize = 0;
    let mut left_depth: usize = 1;

    while pos < chars.len() {
        match chars[pos] {
            '{' => {
                brace_depth += 1;
                pos += 1;
            }
            '}' => {
                if brace_depth == 0 {
                    return None;
                }
                brace_depth -= 1;
                pos += 1;
            }
            '\\' => {
                let cmd_start = pos;
                let Ok((cmd, next_pos)) = read_command(chars, pos + 1) else {
                    return None;
                };
                if brace_depth == 0 {
                    if cmd == "left" {
                        left_depth += 1;
                    } else if cmd == "right" {
                        left_depth -= 1;
                        if left_depth == 0 {
                            let inner: String = chars[start..cmd_start].iter().collect();
                            return Some((inner, next_pos));
                        }
                    }
                }
                pos = next_pos;
            }
            _ => pos += 1,
        }
    }
    None
}

pub(super) struct LatexParseCtx<'a> {
    pub chars: &'a [char],
    pub base_font_id: &'a FontId,
    pub color: Color32,
    pub depth: usize,
}

pub(super) fn process_left_command(
    ctx: &LatexParseCtx<'_>,
    i: &mut usize,
    remaining_nodes: &mut usize,
    boxes: &mut Vec<MathBox>,
) -> Result<(), ()> {
    let open = read_delimiter(ctx.chars, i)?;
    if let Some((body_str, after_right)) = find_matching_right(ctx.chars, *i) {
        let mut next_pos = after_right;
        let close = read_delimiter(ctx.chars, &mut next_pos).unwrap_or(MathDelimiter::None);
        *i = next_pos;
        let body_box = parse_latex_to_box_bounded(
            &body_str,
            ctx.base_font_id,
            ctx.color,
            ctx.depth + 1,
            remaining_nodes,
        )?;
        consume_node(remaining_nodes)?;
        boxes.push(MathBox::Delimited {
            open,
            body: Box::new(body_box),
            close,
            font_id: ctx.base_font_id.clone(),
            color: ctx.color,
        });
    } else if !open.glyph().is_empty() {
        consume_node(remaining_nodes)?;
        boxes.push(MathBox::Text {
            text: open.glyph().to_string(),
            font_id: ctx.base_font_id.clone(),
            color: ctx.color,
            italics: false,
        });
    }
    Ok(())
}

pub(super) fn process_stray_right_command(
    ctx: &LatexParseCtx<'_>,
    i: &mut usize,
    remaining_nodes: &mut usize,
    boxes: &mut Vec<MathBox>,
) -> Result<(), ()> {
    if let Ok(close) = read_delimiter(ctx.chars, i)
        && !close.glyph().is_empty()
    {
        consume_node(remaining_nodes)?;
        boxes.push(MathBox::Text {
            text: close.glyph().to_string(),
            font_id: ctx.base_font_id.clone(),
            color: ctx.color,
            italics: false,
        });
    }
    Ok(())
}

pub(super) fn process_accent_command(
    kind: MathAccent,
    ctx: &LatexParseCtx<'_>,
    i: &mut usize,
    remaining_nodes: &mut usize,
    boxes: &mut Vec<MathBox>,
) -> Result<(), ()> {
    skip_whitespace(ctx.chars, i);
    if *i >= ctx.chars.len() {
        return Err(());
    }
    let arg = if ctx.chars[*i] == '{' {
        let (group, next_i) = read_group_arg(ctx.chars, *i + 1)?;
        *i = next_i;
        group
    } else if ctx.chars[*i] == '\\' {
        let (cmd, next_i) = read_command(ctx.chars, *i + 1)?;
        *i = next_i;
        format!("\\{cmd}")
    } else {
        let ch = ctx.chars[*i];
        *i += 1;
        ch.to_string()
    };
    let base_box = parse_latex_to_box_bounded(
        &arg,
        ctx.base_font_id,
        ctx.color,
        ctx.depth + 1,
        remaining_nodes,
    )?;
    consume_node(remaining_nodes)?;
    boxes.push(MathBox::Accent {
        base: Box::new(base_box),
        kind,
        color: ctx.color,
    });
    Ok(())
}

pub(super) fn map_blackboard_bold(input: &str) -> String {
    input
        .chars()
        .map(|c| match c {
            'R' => 'ℝ',
            'Z' => 'ℤ',
            'N' => 'ℕ',
            'Q' => 'ℚ',
            'C' => 'ℂ',
            'E' => '𝔼',
            'P' => 'ℙ',
            'F' => '𝔽',
            'H' => 'ℍ',
            '1' => '𝟙',
            other => other,
        })
        .collect()
}

pub(super) fn contains_tall_math(b: &MathBox, depth: usize) -> bool {
    if depth > MAX_LATEX_DEPTH {
        return false;
    }
    match b {
        MathBox::Fraction { .. } | MathBox::Sqrt(_) => true,
        MathBox::Row(children) => children.iter().any(|c| contains_tall_math(c, depth + 1)),
        MathBox::Delimited { body, .. } => contains_tall_math(body, depth + 1),
        MathBox::Accent { base, .. } => contains_tall_math(base, depth + 1),
        MathBox::Script { base, sup, sub } => {
            base.as_deref()
                .is_some_and(|c| contains_tall_math(c, depth + 1))
                || sup
                    .as_deref()
                    .is_some_and(|c| contains_tall_math(c, depth + 1))
                || sub
                    .as_deref()
                    .is_some_and(|c| contains_tall_math(c, depth + 1))
        }
        MathBox::Text { .. } => false,
    }
}

fn plain_delim_token(b: &MathBox) -> Option<(MathDelimiter, &FontId, Color32)> {
    if let MathBox::Text {
        text,
        font_id,
        color,
        italics: false,
    } = b
    {
        match text.as_str() {
            "(" => Some((MathDelimiter::ParenOpen, font_id, *color)),
            ")" => Some((MathDelimiter::ParenClose, font_id, *color)),
            "[" => Some((MathDelimiter::BracketOpen, font_id, *color)),
            "]" => Some((MathDelimiter::BracketClose, font_id, *color)),
            _ => None,
        }
    } else {
        None
    }
}

fn find_matching_plain_close(
    boxes: &[MathBox],
    start: usize,
    open: MathDelimiter,
) -> Option<usize> {
    let target_close = match open {
        MathDelimiter::ParenOpen => MathDelimiter::ParenClose,
        MathDelimiter::BracketOpen => MathDelimiter::BracketClose,
        _ => return None,
    };
    let mut depth = 1usize;
    for (offset, item) in boxes[start..].iter().enumerate() {
        if let Some((d, _, _)) = plain_delim_token(item) {
            if d == open {
                depth += 1;
            } else if d == target_close {
                depth -= 1;
                if depth == 0 {
                    return Some(start + offset);
                }
            }
        }
    }
    None
}

/// Upgrades matched plain `(...)` or `[...]` sequences into `MathBox::Delimited`
/// whenever their interior contains a stacked fraction or radical.
pub(super) fn upgrade_tall_plain_delimiters(boxes: Vec<MathBox>) -> Vec<MathBox> {
    if boxes.len() < 3 {
        return boxes;
    }
    let mut out = Vec::with_capacity(boxes.len());
    let mut idx = 0;
    while idx < boxes.len() {
        if let Some((open, font_id, color)) = plain_delim_token(&boxes[idx])
            && matches!(open, MathDelimiter::ParenOpen | MathDelimiter::BracketOpen)
            && let Some(close_idx) = find_matching_plain_close(&boxes, idx + 1, open)
            && boxes[idx + 1..close_idx]
                .iter()
                .any(|b| contains_tall_math(b, 0))
        {
            let close = plain_delim_token(&boxes[close_idx])
                .map_or(MathDelimiter::ParenClose, |(d, _, _)| d);
            let font_clone = font_id.clone();
            let inner_vec = upgrade_tall_plain_delimiters(boxes[idx + 1..close_idx].to_vec());
            let body = if inner_vec.len() == 1 {
                inner_vec.into_iter().next().unwrap()
            } else {
                MathBox::Row(inner_vec)
            };
            out.push(MathBox::Delimited {
                open,
                body: Box::new(body),
                close,
                font_id: font_clone,
                color,
            });
            idx = close_idx + 1;
            continue;
        }
        out.push(boxes[idx].clone());
        idx += 1;
    }
    out
}

pub(super) fn delimited_to_layout_job(
    spec: &DelimPaintSpec<'_>,
    job: &mut LayoutJob,
    depth: usize,
    remaining_chars: &mut usize,
    inline_fractions: bool,
) {
    let tall = !inline_fractions && contains_tall_math(spec.body, depth + 1);
    let delim_size = if tall {
        spec.font_id.size * 1.3
    } else {
        spec.font_id.size
    };
    let fmt = TextFormat {
        font_id: FontId::new(delim_size, spec.font_id.family.clone()),
        color: spec.color,
        valign: Align::Center,
        ..Default::default()
    };
    if !spec.open.glyph().is_empty() {
        append_bounded(job, spec.open.glyph(), 0.0, fmt.clone(), remaining_chars);
    }
    math_box_to_layout_job_mode(spec.body, job, depth + 1, remaining_chars, inline_fractions);
    if !spec.close.glyph().is_empty() {
        append_bounded(job, spec.close.glyph(), 0.0, fmt, remaining_chars);
    }
}

fn format_accented_slice(slice: &str, kind: MathAccent) -> String {
    let mark = kind.combining_mark();
    if slice.is_empty() {
        return format!(" {mark}");
    }
    if kind == MathAccent::Bar && slice.chars().count() > 1 {
        let mut out = String::with_capacity(slice.len() * 2);
        for c in slice.chars() {
            out.push(c);
            out.push(mark);
        }
        out
    } else {
        let mut out = String::with_capacity(slice.len() + mark.len_utf8());
        out.push_str(slice);
        out.push(mark);
        out
    }
}

pub(super) fn accent_to_layout_job(
    base: &MathBox,
    kind: MathAccent,
    color: Color32,
    job: &mut LayoutJob,
    depth: usize,
    remaining_chars: &mut usize,
    inline_fractions: bool,
) {
    let mut sub_budget = *remaining_chars;
    let mut base_job = LayoutJob::default();
    math_box_to_layout_job_mode(
        base,
        &mut base_job,
        depth + 1,
        &mut sub_budget,
        inline_fractions,
    );
    if base_job.sections.is_empty() {
        return;
    }
    let last_idx = base_job.sections.len() - 1;
    for (idx, section) in base_job.sections.into_iter().enumerate() {
        if *remaining_chars == 0 {
            break;
        }
        let slice = &base_job.text[section.byte_range];
        if idx == last_idx && *remaining_chars > slice.chars().count() {
            let combined = format_accented_slice(slice, kind);
            let mut fmt = section.format;
            fmt.color = color;
            append_bounded(job, &combined, section.leading_space, fmt, remaining_chars);
        } else {
            append_bounded(
                job,
                slice,
                section.leading_space,
                section.format,
                remaining_chars,
            );
        }
    }
}

fn scaled_delim_width(delim: MathDelimiter, height: f32) -> f32 {
    match delim {
        MathDelimiter::None => 0.0,
        MathDelimiter::Pipe => (height * 0.16).clamp(4.0, 9.0),
        MathDelimiter::DoublePipe => (height * 0.22).clamp(6.0, 11.0),
        _ => (height * 0.24).clamp(6.0, 15.0),
    }
}

pub(super) fn measure_delimited_box(
    ui: &Ui,
    open: MathDelimiter,
    body: &MathBox,
    close: MathDelimiter,
    font_id: &FontId,
    color: Color32,
    depth: usize,
) -> BoxGeom {
    let body_g = measure_math_box_bounded(ui, body, depth + 1);
    let tall = contains_tall_math(body, depth + 1) || body_g.size.y > font_id.size * 1.55;
    if !tall {
        let measure_glyph = |d: MathDelimiter| -> f32 {
            let g = d.glyph();
            if g.is_empty() {
                0.0
            } else {
                ui.painter()
                    .layout_no_wrap(g.to_owned(), font_id.clone(), color)
                    .size()
                    .x
            }
        };
        let ow = measure_glyph(open);
        let cw = measure_glyph(close);
        return BoxGeom {
            size: Vec2::new(ow + body_g.size.x + cw, body_g.size.y.max(12.0)),
            children: vec![body_g],
        };
    }

    let height = body_g.size.y + 4.0;
    let ow = scaled_delim_width(open, height);
    let cw = scaled_delim_width(close, height);
    let pad = if ow > 0.0 { 1.5 } else { 0.0 } + if cw > 0.0 { 1.5 } else { 0.0 };
    BoxGeom {
        size: Vec2::new(ow + body_g.size.x + cw + pad, height),
        children: vec![body_g],
    }
}

pub(super) struct DelimPaintSpec<'a> {
    pub open: MathDelimiter,
    pub body: &'a MathBox,
    pub close: MathDelimiter,
    pub font_id: &'a FontId,
    pub color: Color32,
}

pub(super) fn paint_delimited_box(
    painter: &egui::Painter,
    origin: Pos2,
    spec: &DelimPaintSpec<'_>,
    geom: &BoxGeom,
    body_g: &BoxGeom,
    depth: usize,
) {
    let tall = contains_tall_math(spec.body, depth + 1) || body_g.size.y > spec.font_id.size * 1.55;
    if !tall {
        let mut x = origin.x;
        if !spec.open.glyph().is_empty() {
            let g = painter.layout_no_wrap(
                spec.open.glyph().to_owned(),
                spec.font_id.clone(),
                spec.color,
            );
            let w = g.size().x;
            let y = origin.y + (geom.size.y - g.size().y) * 0.5;
            painter.galley(Pos2::new(x, y), g, spec.color);
            x += w;
        }
        let body_y = origin.y + (geom.size.y - body_g.size.y) * 0.5;
        paint_math_box_bounded(painter, Pos2::new(x, body_y), spec.body, body_g, depth + 1);
        x += body_g.size.x;
        if !spec.close.glyph().is_empty() {
            let g = painter.layout_no_wrap(
                spec.close.glyph().to_owned(),
                spec.font_id.clone(),
                spec.color,
            );
            let y = origin.y + (geom.size.y - g.size().y) * 0.5;
            painter.galley(Pos2::new(x, y), g, spec.color);
        }
        return;
    }

    let h = geom.size.y;
    let ow = scaled_delim_width(spec.open, h);
    let cw = scaled_delim_width(spec.close, h);
    let left_pad = if ow > 0.0 { 1.5 } else { 0.0 };
    paint_vector_delimiter(painter, origin, Vec2::new(ow, h), spec.open, spec.color);
    let body_pos = Pos2::new(
        origin.x + ow + left_pad,
        origin.y + (h - body_g.size.y) * 0.5,
    );
    paint_math_box_bounded(painter, body_pos, spec.body, body_g, depth + 1);
    let close_pos = Pos2::new(origin.x + geom.size.x - cw, origin.y);
    paint_vector_delimiter(painter, close_pos, Vec2::new(cw, h), spec.close, spec.color);
}

fn paint_vector_delimiter(
    painter: &egui::Painter,
    pos: Pos2,
    size: Vec2,
    delim: MathDelimiter,
    color: Color32,
) {
    if size.x <= 0.0 || delim == MathDelimiter::None {
        return;
    }
    let stroke = Stroke::new(1.4_f32, color);
    let (top, bot) = (pos.y + 1.0, pos.y + size.y - 1.0);
    let (left, right) = (pos.x + 1.0, pos.x + size.x - 1.0);
    let (mid_x, mid_y) = ((left + right) * 0.5, (top + bot) * 0.5);

    match delim {
        MathDelimiter::Pipe => {
            painter.line_segment([Pos2::new(mid_x, top), Pos2::new(mid_x, bot)], stroke);
        }
        MathDelimiter::DoublePipe => {
            let sep = ((right - left) * 0.28).clamp(1.2, 2.4);
            painter.line_segment(
                [Pos2::new(mid_x - sep, top), Pos2::new(mid_x - sep, bot)],
                stroke,
            );
            painter.line_segment(
                [Pos2::new(mid_x + sep, top), Pos2::new(mid_x + sep, bot)],
                stroke,
            );
        }
        MathDelimiter::BracketOpen => {
            painter.line_segment([Pos2::new(right, top), Pos2::new(left, top)], stroke);
            painter.line_segment([Pos2::new(left, top), Pos2::new(left, bot)], stroke);
            painter.line_segment([Pos2::new(left, bot), Pos2::new(right, bot)], stroke);
        }
        MathDelimiter::BracketClose => {
            painter.line_segment([Pos2::new(left, top), Pos2::new(right, top)], stroke);
            painter.line_segment([Pos2::new(right, top), Pos2::new(right, bot)], stroke);
            painter.line_segment([Pos2::new(right, bot), Pos2::new(left, bot)], stroke);
        }
        MathDelimiter::FloorOpen => {
            painter.line_segment([Pos2::new(left, top), Pos2::new(left, bot)], stroke);
            painter.line_segment([Pos2::new(left, bot), Pos2::new(right, bot)], stroke);
        }
        MathDelimiter::FloorClose => {
            painter.line_segment([Pos2::new(right, top), Pos2::new(right, bot)], stroke);
            painter.line_segment([Pos2::new(right, bot), Pos2::new(left, bot)], stroke);
        }
        MathDelimiter::CeilOpen => {
            painter.line_segment([Pos2::new(right, top), Pos2::new(left, top)], stroke);
            painter.line_segment([Pos2::new(left, top), Pos2::new(left, bot)], stroke);
        }
        MathDelimiter::CeilClose => {
            painter.line_segment([Pos2::new(left, top), Pos2::new(right, top)], stroke);
            painter.line_segment([Pos2::new(right, top), Pos2::new(right, bot)], stroke);
        }
        MathDelimiter::ParenOpen | MathDelimiter::ParenClose => {
            let (tip_x, ctrl_x) = if delim == MathDelimiter::ParenOpen {
                (right, left - (right - left) * 0.35)
            } else {
                (left, right + (right - left) * 0.35)
            };
            paint_quad_curve(
                painter,
                Pos2::new(tip_x, top),
                Pos2::new(ctrl_x, mid_y),
                Pos2::new(tip_x, bot),
                stroke,
            );
        }
        MathDelimiter::BraceOpen | MathDelimiter::BraceClose => {
            paint_curly_brace(
                painter,
                (left, mid_x, right),
                (top, mid_y, bot),
                delim,
                stroke,
            );
        }
        MathDelimiter::AngleOpen => {
            painter.line_segment([Pos2::new(right, top), Pos2::new(left, mid_y)], stroke);
            painter.line_segment([Pos2::new(left, mid_y), Pos2::new(right, bot)], stroke);
        }
        MathDelimiter::AngleClose => {
            painter.line_segment([Pos2::new(left, top), Pos2::new(right, mid_y)], stroke);
            painter.line_segment([Pos2::new(right, mid_y), Pos2::new(left, bot)], stroke);
        }
        MathDelimiter::None => {}
    }
}

fn paint_curly_brace(
    painter: &egui::Painter,
    xs: (f32, f32, f32),
    ys: (f32, f32, f32),
    delim: MathDelimiter,
    stroke: Stroke,
) {
    let (left, mid_x, right) = xs;
    let (top, mid_y, bot) = ys;
    let (tip_x, cusp_x) = if delim == MathDelimiter::BraceOpen {
        (right, left)
    } else {
        (left, right)
    };
    painter.line_segment([Pos2::new(tip_x, top), Pos2::new(mid_x, top + 2.0)], stroke);
    painter.line_segment(
        [Pos2::new(mid_x, top + 2.0), Pos2::new(mid_x, mid_y - 2.5)],
        stroke,
    );
    painter.line_segment(
        [Pos2::new(mid_x, mid_y - 2.5), Pos2::new(cusp_x, mid_y)],
        stroke,
    );
    painter.line_segment(
        [Pos2::new(cusp_x, mid_y), Pos2::new(mid_x, mid_y + 2.5)],
        stroke,
    );
    painter.line_segment(
        [Pos2::new(mid_x, mid_y + 2.5), Pos2::new(mid_x, bot - 2.0)],
        stroke,
    );
    painter.line_segment([Pos2::new(mid_x, bot - 2.0), Pos2::new(tip_x, bot)], stroke);
}

fn paint_quad_curve(painter: &egui::Painter, p0: Pos2, p1: Pos2, p2: Pos2, stroke: Stroke) {
    let steps = 12usize;
    let mut prev = p0;
    for step in 1..=steps {
        let t = step as f32 / steps as f32;
        let omt = 1.0 - t;
        let x = omt * omt * p0.x + 2.0 * omt * t * p1.x + t * t * p2.x;
        let y = omt * omt * p0.y + 2.0 * omt * t * p1.y + t * t * p2.y;
        let next = Pos2::new(x, y);
        painter.line_segment([prev, next], stroke);
        prev = next;
    }
}

pub(super) fn measure_accent_box(ui: &Ui, base: &MathBox, depth: usize) -> BoxGeom {
    let base_g = measure_math_box_bounded(ui, base, depth + 1);
    let headroom = (base_g.size.y * 0.28).clamp(4.0, 7.0);
    BoxGeom {
        size: Vec2::new(
            truncate_nonneg(base_g.size.x).max(6.0),
            base_g.size.y + headroom,
        ),
        children: vec![base_g],
    }
}

fn truncate_nonneg(v: f32) -> f32 {
    if v.is_finite() && v > 0.0 { v } else { 0.0 }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn paint_accent_box(
    painter: &egui::Painter,
    origin: Pos2,
    base: &MathBox,
    kind: MathAccent,
    color: Color32,
    geom: &BoxGeom,
    base_g: &BoxGeom,
    depth: usize,
) {
    let headroom = (geom.size.y - base_g.size.y).max(3.0);
    let base_x = origin.x + (geom.size.x - base_g.size.x) * 0.5;
    paint_math_box_bounded(
        painter,
        Pos2::new(base_x, origin.y + headroom),
        base,
        base_g,
        depth + 1,
    );
    let center = Pos2::new(origin.x + geom.size.x * 0.5, origin.y + headroom * 0.48);
    paint_accent_mark(painter, center, base_g.size, kind, color);
}

fn paint_accent_mark(
    painter: &egui::Painter,
    center: Pos2,
    base_size: Vec2,
    kind: MathAccent,
    color: Color32,
) {
    let (cx, cy) = (center.x, center.y);
    let r = (base_size.y * 0.085).clamp(1.3, 2.3);
    let stroke = Stroke::new(1.2_f32, color);
    match kind {
        MathAccent::Dot => {
            painter.circle_filled(center, r, color);
        }
        MathAccent::Ddot => {
            let sep = (r * 1.75).max(2.2);
            painter.circle_filled(Pos2::new(cx - sep, cy), r, color);
            painter.circle_filled(Pos2::new(cx + sep, cy), r, color);
        }
        MathAccent::Bar => {
            let hw = (base_size.x * 0.42).max(2.5);
            painter.line_segment([Pos2::new(cx - hw, cy), Pos2::new(cx + hw, cy)], stroke);
        }
        MathAccent::Hat => {
            let hw = (base_size.x * 0.38).max(2.5);
            painter.line_segment(
                [Pos2::new(cx - hw, cy + 1.5), Pos2::new(cx, cy - 1.5)],
                stroke,
            );
            painter.line_segment(
                [Pos2::new(cx, cy - 1.5), Pos2::new(cx + hw, cy + 1.5)],
                stroke,
            );
        }
        MathAccent::VecArrow => {
            let hw = (base_size.x * 0.42).max(3.0);
            painter.line_segment([Pos2::new(cx - hw, cy), Pos2::new(cx + hw, cy)], stroke);
            painter.line_segment(
                [Pos2::new(cx + hw - 2.2, cy - 1.8), Pos2::new(cx + hw, cy)],
                stroke,
            );
            painter.line_segment(
                [Pos2::new(cx + hw - 2.2, cy + 1.8), Pos2::new(cx + hw, cy)],
                stroke,
            );
        }
        MathAccent::Tilde => {
            let hw = (base_size.x * 0.38).max(2.8);
            painter.line_segment(
                [
                    Pos2::new(cx - hw, cy + 0.8),
                    Pos2::new(cx - hw * 0.3, cy - 1.0),
                ],
                stroke,
            );
            painter.line_segment(
                [
                    Pos2::new(cx - hw * 0.3, cy - 1.0),
                    Pos2::new(cx + hw * 0.3, cy + 1.0),
                ],
                stroke,
            );
            painter.line_segment(
                [
                    Pos2::new(cx + hw * 0.3, cy + 1.0),
                    Pos2::new(cx + hw, cy - 0.8),
                ],
                stroke,
            );
        }
    }
}
