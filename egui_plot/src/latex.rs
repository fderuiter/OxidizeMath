use egui::{
    Align, Color32, FontId, TextStyle, Ui,
    text::{LayoutJob, TextFormat},
};

/// Parses a LaTeX string into an `egui::text::LayoutJob`.
///
/// Supports symbols (\alpha, \beta, \infty, etc.), superscripts (^), subscripts (_),
/// fractions (\frac{a}{b}), math functions (\sin, \cos, \exp), text blocks (\text{...}),
/// and variable italicization.
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

    // Attempt to parse text. If malformed, fall back to plain text.
    if parse_latex_into_job(text, &font_id, color, &mut job).is_err() {
        job = LayoutJob::default();
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

    job
}

fn parse_latex_into_job(
    text: &str,
    base_font_id: &FontId,
    color: Color32,
    job: &mut LayoutJob,
) -> Result<(), ()> {
    if text.contains('$') {
        let parts: Vec<&str> = text.split('$').collect();
        // If odd number of parts (even number of $), valid dollar pairs
        if parts.len() % 2 == 1 {
            for (idx, part) in parts.iter().enumerate() {
                if part.is_empty() {
                    continue;
                }
                let is_math = idx % 2 == 1;
                if is_math {
                    parse_math_expression(part, base_font_id, color, job)?;
                } else {
                    append_plain_text(part, base_font_id, color, job);
                }
            }
            return Ok(());
        }
    }

    // If no dollar signs or unbalanced, check if text has latex commands or tokens
    parse_math_expression(text, base_font_id, color, job)
}

fn append_plain_text(text: &str, font_id: &FontId, color: Color32, job: &mut LayoutJob) {
    job.append(
        text,
        0.0,
        TextFormat {
            font_id: font_id.clone(),
            color,
            valign: Align::Center,
            ..Default::default()
        },
    );
}

fn parse_math_expression(
    expr: &str,
    base_font_id: &FontId,
    color: Color32,
    job: &mut LayoutJob,
) -> Result<(), ()> {
    let chars: Vec<char> = expr.chars().collect();
    let mut i = 0;
    let len = chars.len();

    let script_font_size = (base_font_id.size * 0.75).max(8.0);
    let super_font_id = FontId::new(script_font_size, base_font_id.family.clone());
    let sub_font_id = FontId::new(script_font_size, base_font_id.family.clone());

    while i < len {
        let c = chars[i];

        if c == '^' {
            i += 1;
            let (script_str, next_i) = read_script_arg(&chars, i)?;
            i = next_i;
            append_script_job(&script_str, &super_font_id, color, Align::Min, job)?;
        } else if c == '_' {
            i += 1;
            let (script_str, next_i) = read_script_arg(&chars, i)?;
            i = next_i;
            append_script_job(&script_str, &sub_font_id, color, Align::Max, job)?;
        } else if c == '\\' {
            i += 1;
            let (cmd, next_i) = read_command(&chars, i)?;
            i = next_i;
            process_command(&cmd, &chars, &mut i, base_font_id, color, job)?;
        } else if c == '{' {
            // Grouping brace
            i += 1;
            let (group_str, next_i) = read_group_arg(&chars, i)?;
            i = next_i;
            parse_math_expression(&group_str, base_font_id, color, job)?;
        } else if c == '}' {
            // Unmatched closing brace is a syntax error
            return Err(());
        } else if c.is_ascii_alphabetic() {
            // Check if part of a word or single variable
            let start = i;
            while i < len && chars[i].is_ascii_alphabetic() {
                i += 1;
            }
            let word: String = chars[start..i].iter().collect();
            // Single letter variables in math mode are italicized
            let italics = word.len() == 1;
            job.append(
                &word,
                0.0,
                TextFormat {
                    font_id: base_font_id.clone(),
                    color,
                    italics,
                    valign: Align::Center,
                    ..Default::default()
                },
            );
        } else {
            // Other characters (digits, operators, spaces, punctuation)
            let s: String = std::iter::once(c).collect();
            job.append(
                &s,
                0.0,
                TextFormat {
                    font_id: base_font_id.clone(),
                    color,
                    valign: Align::Center,
                    ..Default::default()
                },
            );
            i += 1;
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

fn append_script_job(
    script_str: &str,
    font_id: &FontId,
    color: Color32,
    valign: Align,
    job: &mut LayoutJob,
) -> Result<(), ()> {
    let mut script_job = LayoutJob::default();
    parse_math_expression(script_str, font_id, color, &mut script_job)?;
    for section in script_job.sections {
        let section_text = &script_job.text[section.byte_range.clone()];
        let mut fmt = section.format;
        fmt.valign = valign;
        job.append(section_text, section.leading_space, fmt);
    }
    Ok(())
}

fn process_command(
    cmd: &str,
    chars: &[char],
    i: &mut usize,
    base_font_id: &FontId,
    color: Color32,
    job: &mut LayoutJob,
) -> Result<(), ()> {
    // Check symbol mappings
    if let Some(sym) = map_latex_symbol(cmd) {
        job.append(
            sym,
            0.0,
            TextFormat {
                font_id: base_font_id.clone(),
                color,
                valign: Align::Center,
                ..Default::default()
            },
        );
        return Ok(());
    }

    match cmd {
        "text" | "mathrm" | "mathbf" => {
            skip_whitespace(chars, i);
            if *i < chars.len() && chars[*i] == '{' {
                *i += 1;
                let (arg, next_i) = read_group_arg(chars, *i)?;
                *i = next_i;
                job.append(
                    &arg,
                    0.0,
                    TextFormat {
                        font_id: base_font_id.clone(),
                        color,
                        italics: false,
                        valign: Align::Center,
                        ..Default::default()
                    },
                );
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

                    // Render fraction as (num)/(den) or num/den
                    let num_has_ops = num.contains('+') || num.contains('-') || num.contains(' ');
                    let den_has_ops = den.contains('+') || den.contains('-') || den.contains(' ');

                    if num_has_ops {
                        job.append(
                            "(",
                            0.0,
                            TextFormat {
                                font_id: base_font_id.clone(),
                                color,
                                valign: Align::Center,
                                ..Default::default()
                            },
                        );
                    }
                    parse_math_expression(&num, base_font_id, color, job)?;
                    if num_has_ops {
                        job.append(
                            ")",
                            0.0,
                            TextFormat {
                                font_id: base_font_id.clone(),
                                color,
                                valign: Align::Center,
                                ..Default::default()
                            },
                        );
                    }

                    job.append(
                        " / ",
                        0.0,
                        TextFormat {
                            font_id: base_font_id.clone(),
                            color,
                            valign: Align::Center,
                            ..Default::default()
                        },
                    );

                    if den_has_ops {
                        job.append(
                            "(",
                            0.0,
                            TextFormat {
                                font_id: base_font_id.clone(),
                                color,
                                valign: Align::Center,
                                ..Default::default()
                            },
                        );
                    }
                    parse_math_expression(&den, base_font_id, color, job)?;
                    if den_has_ops {
                        job.append(
                            ")",
                            0.0,
                            TextFormat {
                                font_id: base_font_id.clone(),
                                color,
                                valign: Align::Center,
                                ..Default::default()
                            },
                        );
                    }
                } else {
                    return Err(());
                }
            } else {
                return Err(());
            }
        }
        "sin" | "cos" | "tan" | "exp" | "ln" | "log" | "lim" | "max" | "min" | "det" | "arcsin"
        | "arccos" | "arctan" | "sinh" | "cosh" | "tanh" => {
            job.append(
                cmd,
                0.0,
                TextFormat {
                    font_id: base_font_id.clone(),
                    color,
                    italics: false,
                    valign: Align::Center,
                    ..Default::default()
                },
            );
        }
        "quad" => {
            job.append(
                "  ",
                0.0,
                TextFormat {
                    font_id: base_font_id.clone(),
                    color,
                    valign: Align::Center,
                    ..Default::default()
                },
            );
        }
        "qquad" => {
            job.append(
                "    ",
                0.0,
                TextFormat {
                    font_id: base_font_id.clone(),
                    color,
                    valign: Align::Center,
                    ..Default::default()
                },
            );
        }
        "," | ":" | ";" => {
            job.append(
                " ",
                0.0,
                TextFormat {
                    font_id: base_font_id.clone(),
                    color,
                    valign: Align::Center,
                    ..Default::default()
                },
            );
        }
        "\\" | "newline" => {
            job.append(
                "\n",
                0.0,
                TextFormat {
                    font_id: base_font_id.clone(),
                    color,
                    valign: Align::Center,
                    ..Default::default()
                },
            );
        }
        _ => {
            // Unknown command: render command name as plain text
            job.append(
                cmd,
                0.0,
                TextFormat {
                    font_id: base_font_id.clone(),
                    color,
                    valign: Align::Center,
                    ..Default::default()
                },
            );
        }
    }

    Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;
    use egui::{Align, Color32, FontFamily, FontId};

    fn test_font() -> FontId {
        FontId::new(14.0, FontFamily::Proportional)
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

        // Superscript x^2
        let job_super = parse_latex_with_font("x^2", font.clone(), color);
        assert_eq!(job_super.text, "x2");
        assert_eq!(job_super.sections.len(), 2);
        assert_eq!(job_super.sections[0].format.valign, Align::Center);
        assert_eq!(job_super.sections[1].format.valign, Align::Min);
        assert!(job_super.sections[1].format.font_id.size < font.size);

        // Subscript y_1
        let job_sub = parse_latex_with_font("y_1", font.clone(), color);
        assert_eq!(job_sub.text, "y1");
        assert_eq!(job_sub.sections.len(), 2);
        assert_eq!(job_sub.sections[0].format.valign, Align::Center);
        assert_eq!(job_sub.sections[1].format.valign, Align::Max);
        assert!(job_sub.sections[1].format.font_id.size < font.size);

        // Grouped superscript x^{2t}
        let job_group = parse_latex_with_font("x^{2t}", font, color);
        assert_eq!(job_group.text, "x2t");
        assert!(job_group.sections.len() >= 2);
    }

    #[test]
    fn test_latex_fraction() {
        let font = test_font();
        let job = parse_latex_with_font(r"\frac{dy}{dt}", font, Color32::WHITE);
        assert_eq!(job.text, "dy / dt");
    }

    #[test]
    fn test_latex_functions_and_text() {
        let font = test_font();

        let job_func = parse_latex_with_font(r"\sin(x)", font.clone(), Color32::WHITE);
        assert_eq!(job_func.text, "sin(x)");
        // "sin" section should not be italicized
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

        // Unclosed brace should not panic, falls back to plain text
        let job_unclosed_brace = parse_latex_with_font(r"\frac{a}{b", font.clone(), Color32::WHITE);
        assert_eq!(job_unclosed_brace.text, r"\frac{a}{b");

        // Unclosed dollar sign should not panic, falls back to math or plain string
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
}
