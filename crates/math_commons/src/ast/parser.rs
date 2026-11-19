//! Pratt parser for mathematical expressions with implicit multiplication and recursion bounds.

use super::expr::{BinaryOp, Expr, Func, UnaryOp};
use super::lexer::{LexError, Span, Token, TokenKind, tokenize};

/// Maximum allowed recursion nesting depth to prevent stack overflow.
pub const MAX_PARSE_DEPTH: usize = 64;

/// Error encountered during expression parsing.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ParseError {
    /// Error during tokenization.
    #[error("{}", .0.user_message())]
    Lex(#[from] LexError),

    /// Unexpected token encountered.
    #[error("At character {}: expected {expected}, found {}", span.start + 1, format_found_token(found))]
    UnexpectedToken {
        /// Description of expected syntax.
        expected: String,
        /// Description of actual token found.
        found: String,
        /// Position span.
        span: Span,
    },

    /// Unexpected end of input.
    #[error("Unexpected end of expression")]
    UnexpectedEof,

    /// Expression nesting depth exceeded safe threshold.
    #[error("Expression is nested too deeply (maximum depth {0})")]
    RecursionLimitExceeded(usize),

    /// Empty expression string provided.
    #[error("Expression cannot be empty")]
    EmptyExpression,
}

fn format_found_token(found: &str) -> String {
    if found == "<EOF>" || found == "end of expression" {
        "end of expression".to_string()
    } else {
        format!("'{found}'")
    }
}

impl ParseError {
    /// Formats this parse error in plain language using 1-indexed character positions
    /// without exposing internal `Span` structs or `<EOF>` markers.
    #[must_use]
    pub fn user_message(&self) -> String {
        self.to_string()
    }

    /// Formats this parse error in plain language with source context so trailing
    /// incomplete expressions also report a 1-indexed character position.
    #[must_use]
    pub fn user_message_for_input(&self, input: &str) -> String {
        if matches!(self, Self::UnexpectedEof) {
            let trimmed_chars = input.trim().chars().count();
            if trimmed_chars > 0 {
                return format!(
                    "Unexpected end of expression at character {}",
                    trimmed_chars + 1
                );
            }
        }
        self.user_message()
    }
}

/// Parses a mathematical string into an Abstract Syntax Tree (`Expr`).
///
/// Supports equation prefixes such as `y' = ...` or `w = ...` by stripping the LHS.
pub fn parse(input: &str) -> Result<Expr, ParseError> {
    let (_, expr) = parse_equation(input)?;
    Ok(expr)
}

/// Parses an expression that may include a leading equation binding such as `y' = expr` or `w = expr`.
///
/// Returns a tuple of `(Option<lhs_name>, Expr)`.
pub fn parse_equation(input: &str) -> Result<(Option<String>, Expr), ParseError> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Err(ParseError::EmptyExpression);
    }

    let tokens = tokenize(trimmed)?;
    if tokens.is_empty() || (tokens.len() == 1 && tokens[0].kind == TokenKind::Eof) {
        return Err(ParseError::EmptyExpression);
    }

    let mut parser = Parser::new(tokens);
    let lhs = parser.try_parse_equation_lhs();
    let (expr, _) = parser.parse_expr(0, 0)?;

    if !parser.is_at_end() {
        let tok = parser.peek();
        return Err(ParseError::UnexpectedToken {
            expected: "end of expression".into(),
            found: tok.kind.to_string(),
            span: tok.span,
        });
    }

    Ok((lhs, expr))
}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, pos: 0 }
    }

    fn peek(&self) -> &Token {
        if self.pos < self.tokens.len() {
            &self.tokens[self.pos]
        } else {
            self.tokens.last().expect("tokens should have EOF")
        }
    }

    fn advance(&mut self) -> &Token {
        let p = self.pos;
        if self.pos < self.tokens.len() {
            self.pos += 1;
        }
        &self.tokens[p]
    }

    fn is_at_end(&self) -> bool {
        self.peek().kind == TokenKind::Eof
    }

    /// Checks if input starts with an equation LHS like `y' =`, `w =`, or `f(t, y) =`.
    fn try_parse_equation_lhs(&mut self) -> Option<String> {
        let start_pos = self.pos;

        // Pattern 1: Ident '='
        if let TokenKind::Ident(name) = &self.peek().kind {
            let lhs_name = name.clone();
            if self.pos + 1 < self.tokens.len()
                && self.tokens[self.pos + 1].kind == TokenKind::Equals
            {
                self.pos += 2; // consume ident and '='
                return Some(lhs_name);
            }

            // Pattern 2: Ident '(' ... ')' '=' (e.g. f(t, y) =)
            if self.pos + 1 < self.tokens.len()
                && self.tokens[self.pos + 1].kind == TokenKind::LParen
            {
                let mut idx = self.pos + 2;
                while idx < self.tokens.len() && self.tokens[idx].kind != TokenKind::RParen {
                    idx += 1;
                }
                if idx + 1 < self.tokens.len() && self.tokens[idx + 1].kind == TokenKind::Equals {
                    self.pos = idx + 2; // consume through '='
                    return Some(lhs_name);
                }
            }
        }

        self.pos = start_pos;
        None
    }

    fn parse_expr(&mut self, min_bp: u8, depth: usize) -> Result<(Expr, usize), ParseError> {
        if depth > MAX_PARSE_DEPTH {
            return Err(ParseError::RecursionLimitExceeded(MAX_PARSE_DEPTH));
        }

        let (mut lhs, mut lhs_height) = self.parse_prefix(depth)?;

        loop {
            if self.is_at_end() {
                break;
            }

            // Check for explicit infix operator
            if let Some((l_bp, r_bp, op)) = self.peek_infix_op() {
                if l_bp < min_bp {
                    break;
                }
                self.advance(); // consume operator
                let (rhs, rhs_height) = self.parse_expr(r_bp, depth + 1)?;
                (lhs, lhs_height) = wrap_binary(op, lhs, lhs_height, rhs, rhs_height)?;
                continue;
            }

            // Check for implicit multiplication
            if self.can_start_atom() {
                let l_bp = 5;
                let r_bp = 6;
                if l_bp < min_bp {
                    break;
                }
                let (rhs, rhs_height) = self.parse_expr(r_bp, depth + 1)?;
                (lhs, lhs_height) = wrap_binary(BinaryOp::Mul, lhs, lhs_height, rhs, rhs_height)?;
                continue;
            }

            break;
        }

        Ok((lhs, lhs_height))
    }

    fn parse_prefix(&mut self, depth: usize) -> Result<(Expr, usize), ParseError> {
        let tok = self.advance().clone();
        match tok.kind {
            TokenKind::Number(n) => Ok((Expr::Number(n), 0)),
            TokenKind::Ident(name) => self.parse_ident_or_func(name, depth),
            TokenKind::LParen => {
                let (inner, inner_height) = self.parse_expr(0, depth + 1)?;
                if self.peek().kind != TokenKind::RParen {
                    return Err(ParseError::UnexpectedToken {
                        expected: "closing parenthesis ')'".into(),
                        found: self.peek().kind.to_string(),
                        span: self.peek().span,
                    });
                }
                self.advance(); // consume ')'
                Ok((inner, inner_height))
            }
            TokenKind::Plus => self.parse_unary(UnaryOp::Pos, depth),
            TokenKind::Minus => self.parse_unary(UnaryOp::Neg, depth),
            TokenKind::Eof => Err(ParseError::UnexpectedEof),
            _ => Err(ParseError::UnexpectedToken {
                expected: "number, variable, function, or '('".into(),
                found: tok.kind.to_string(),
                span: tok.span,
            }),
        }
    }

    fn parse_unary(&mut self, op: UnaryOp, depth: usize) -> Result<(Expr, usize), ParseError> {
        let (expr, inner_height) = self.parse_expr(7, depth + 1)?;
        let next_height = inner_height + 1;
        if next_height > MAX_PARSE_DEPTH {
            return Err(ParseError::RecursionLimitExceeded(MAX_PARSE_DEPTH));
        }
        Ok((Expr::unary(op, expr), next_height))
    }

    fn parse_ident_or_func(
        &mut self,
        name: String,
        depth: usize,
    ) -> Result<(Expr, usize), ParseError> {
        if let Some(func) = Func::from_name(&name) {
            if self.peek().kind == TokenKind::LParen {
                return self.parse_parenthesized_func(func, depth);
            }
            // Support function without parens e.g. "sin x" if followed by atom
            if self.can_start_atom() {
                let (arg, arg_height) = self.parse_expr(7, depth + 1)?;
                let next_height = arg_height + 1;
                if next_height > MAX_PARSE_DEPTH {
                    return Err(ParseError::RecursionLimitExceeded(MAX_PARSE_DEPTH));
                }
                return Ok((Expr::function(func, vec![arg]), next_height));
            }
        }

        Ok((Expr::Variable(name), 0))
    }

    fn parse_parenthesized_func(
        &mut self,
        func: Func,
        depth: usize,
    ) -> Result<(Expr, usize), ParseError> {
        self.advance(); // consume '('
        let mut args = Vec::new();
        let mut max_arg_height = 0;
        if self.peek().kind != TokenKind::RParen {
            loop {
                let (arg, arg_height) = self.parse_expr(0, depth + 1)?;
                max_arg_height = max_arg_height.max(arg_height);
                args.push(arg);
                if self.peek().kind == TokenKind::Comma {
                    self.advance();
                } else {
                    break;
                }
            }
        }
        if self.peek().kind != TokenKind::RParen {
            return Err(ParseError::UnexpectedToken {
                expected: "closing parenthesis ')' after function arguments".into(),
                found: self.peek().kind.to_string(),
                span: self.peek().span,
            });
        }
        self.advance(); // consume ')'
        let next_height = if args.is_empty() {
            0
        } else {
            max_arg_height + 1
        };
        if next_height > MAX_PARSE_DEPTH {
            return Err(ParseError::RecursionLimitExceeded(MAX_PARSE_DEPTH));
        }
        Ok((Expr::function(func, args), next_height))
    }

    fn peek_infix_op(&self) -> Option<(u8, u8, BinaryOp)> {
        match self.peek().kind {
            TokenKind::Plus => Some((1, 2, BinaryOp::Add)),
            TokenKind::Minus => Some((1, 2, BinaryOp::Sub)),
            TokenKind::Star => Some((3, 4, BinaryOp::Mul)),
            TokenKind::Slash => Some((3, 4, BinaryOp::Div)),
            TokenKind::Caret => Some((8, 7, BinaryOp::Pow)),
            _ => None,
        }
    }

    fn can_start_atom(&self) -> bool {
        matches!(
            self.peek().kind,
            TokenKind::Number(_) | TokenKind::Ident(_) | TokenKind::LParen
        )
    }
}

fn wrap_binary(
    op: BinaryOp,
    lhs: Expr,
    lhs_height: usize,
    rhs: Expr,
    rhs_height: usize,
) -> Result<(Expr, usize), ParseError> {
    let next_height = lhs_height.max(rhs_height) + 1;
    if next_height > MAX_PARSE_DEPTH {
        return Err(ParseError::RecursionLimitExceeded(MAX_PARSE_DEPTH));
    }
    Ok((Expr::binary(op, lhs, rhs), next_height))
}
