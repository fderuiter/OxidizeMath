//! Pratt parser for mathematical expressions with implicit multiplication and recursion bounds.

use super::expr::{BinaryOp, Expr, Func, UnaryOp};
use super::lexer::{LexError, Span, Token, TokenKind, tokenize};

/// Maximum allowed recursion nesting depth to prevent stack overflow.
pub const MAX_PARSE_DEPTH: usize = 64;

/// Error encountered during expression parsing.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ParseError {
    /// Error during tokenization.
    #[error("Lexical error: {0}")]
    Lex(#[from] LexError),

    /// Unexpected token encountered.
    #[error("Parse error at {span:?}: expected {expected}, found '{found}'")]
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
    #[error("Recursion limit exceeded (max depth {0})")]
    RecursionLimitExceeded(usize),

    /// Empty expression string provided.
    #[error("Expression cannot be empty")]
    EmptyExpression,
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
    let expr = parser.parse_expr(0, 0)?;

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

    fn parse_expr(&mut self, min_bp: u8, depth: usize) -> Result<Expr, ParseError> {
        if depth > MAX_PARSE_DEPTH {
            return Err(ParseError::RecursionLimitExceeded(MAX_PARSE_DEPTH));
        }

        let mut lhs = self.parse_prefix(depth)?;

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
                let rhs = self.parse_expr(r_bp, depth + 1)?;
                lhs = Expr::binary(op, lhs, rhs);
                continue;
            }

            // Check for implicit multiplication
            if self.can_start_implicit_mul() {
                let l_bp = 5;
                let r_bp = 6;
                if l_bp < min_bp {
                    break;
                }
                let rhs = self.parse_expr(r_bp, depth + 1)?;
                lhs = Expr::binary(BinaryOp::Mul, lhs, rhs);
                continue;
            }

            break;
        }

        Ok(lhs)
    }

    fn parse_prefix(&mut self, depth: usize) -> Result<Expr, ParseError> {
        let tok = self.advance().clone();
        match tok.kind {
            TokenKind::Number(n) => Ok(Expr::Number(n)),
            TokenKind::Ident(name) => self.parse_ident_or_func(name, depth),
            TokenKind::LParen => {
                let inner = self.parse_expr(0, depth + 1)?;
                if self.peek().kind != TokenKind::RParen {
                    return Err(ParseError::UnexpectedToken {
                        expected: "closing parenthesis ')'".into(),
                        found: self.peek().kind.to_string(),
                        span: self.peek().span,
                    });
                }
                self.advance(); // consume ')'
                Ok(inner)
            }
            TokenKind::Plus => {
                let expr = self.parse_expr(7, depth + 1)?;
                Ok(Expr::unary(UnaryOp::Pos, expr))
            }
            TokenKind::Minus => {
                let expr = self.parse_expr(7, depth + 1)?;
                Ok(Expr::unary(UnaryOp::Neg, expr))
            }
            TokenKind::Eof => Err(ParseError::UnexpectedEof),
            _ => Err(ParseError::UnexpectedToken {
                expected: "number, variable, function, or '('".into(),
                found: tok.kind.to_string(),
                span: tok.span,
            }),
        }
    }

    fn parse_ident_or_func(&mut self, name: String, depth: usize) -> Result<Expr, ParseError> {
        if let Some(func) = Func::from_name(&name) {
            if self.peek().kind == TokenKind::LParen {
                self.advance(); // consume '('
                let mut args = Vec::new();
                if self.peek().kind != TokenKind::RParen {
                    loop {
                        let arg = self.parse_expr(0, depth + 1)?;
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
                return Ok(Expr::function(func, args));
            }
            // Support function without parens e.g. "sin x" if followed by atom
            if self.can_start_prefix() {
                let arg = self.parse_expr(7, depth + 1)?;
                return Ok(Expr::function(func, vec![arg]));
            }
        }

        Ok(Expr::Variable(name))
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

    fn can_start_prefix(&self) -> bool {
        matches!(
            self.peek().kind,
            TokenKind::Number(_) | TokenKind::Ident(_) | TokenKind::LParen
        )
    }

    fn can_start_implicit_mul(&self) -> bool {
        matches!(
            self.peek().kind,
            TokenKind::Number(_) | TokenKind::Ident(_) | TokenKind::LParen
        )
    }
}
