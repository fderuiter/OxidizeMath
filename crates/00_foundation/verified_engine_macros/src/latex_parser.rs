#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MathOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
}

pub fn parse_latex_math(latex: &str) -> Vec<MathOp> {
    let mut ops = Vec::new();

    let mut chars = latex.chars().peekable();

    while let Some(c) = chars.next() {
        match c {
            '+' => ops.push(MathOp::Add),
            '-' => ops.push(MathOp::Sub),
            '*' => ops.push(MathOp::Mul),
            '/' => ops.push(MathOp::Div),
            '^' => ops.push(MathOp::Pow),
            '\\' => {
                let mut cmd = String::new();
                while let Some(&n) = chars.peek() {
                    if n.is_alphabetic() {
                        cmd.push(n);
                        chars.next();
                    } else {
                        break;
                    }
                }
                match cmd.as_str() {
                    "frac" => ops.push(MathOp::Div),
                    "times" | "cdot" => ops.push(MathOp::Mul),
                    "sum" | "int" => {} // ignore
                    "left" | "right" => {}
                    _ => {
                        // could be a greek letter or a math symbol, ignore for now
                        // or if it's \hat{f}, etc.
                    }
                }
            }
            _ => {}
        }
    }

    ops
}

#[cfg(test)]
mod tests {
    use super::{MathOp, parse_latex_math};

    #[test]
    fn test_parse_ascii_operators() {
        let ops = parse_latex_math("a + b - c * d / e ^ 2");
        assert_eq!(
            ops,
            vec![
                MathOp::Add,
                MathOp::Sub,
                MathOp::Mul,
                MathOp::Div,
                MathOp::Pow,
            ]
        );
    }

    #[test]
    fn test_parse_latex_commands_before_alphabetic_fallback() {
        let ops = parse_latex_math(r"\frac{\hat{f}_{FFT} \cdot c \times T_c}{2 B}");
        assert_eq!(ops, vec![MathOp::Div, MathOp::Mul, MathOp::Mul]);
    }

    #[test]
    fn test_ignores_structural_commands_and_identifiers() {
        let ops = parse_latex_math(r"\sum_{n=0}^{N-1} \int \left( \alpha x_n \right)");
        assert_eq!(ops, vec![MathOp::Pow, MathOp::Sub]);
        assert!(parse_latex_math("2 B R = 42").is_empty());
        assert!(parse_latex_math("").is_empty());
    }
}
