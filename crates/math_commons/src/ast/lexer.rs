//! Lexical analyzer / tokenizer for mathematical expressions.

use std::fmt;

/// Spanned byte position range in the source string.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    /// Start byte index (inclusive).
    pub start: usize,
    /// End byte index (exclusive).
    pub end: usize,
}

impl Span {
    /// Creates a new span.
    #[must_use]
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }
}

/// Token kinds produced during lexical analysis.
#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    /// Numeric literal value.
    Number(f64),
    /// Identifier (variable, function, or constant name).
    Ident(String),
    /// Addition operator `+`.
    Plus,
    /// Subtraction operator `-`.
    Minus,
    /// Multiplication operator `*`.
    Star,
    /// Division operator `/`.
    Slash,
    /// Exponentiation operator `^`.
    Caret,
    /// Opening parenthesis `(`.
    LParen,
    /// Closing parenthesis `)`.
    RParen,
    /// Comma separator `,`.
    Comma,
    /// Equals sign `=`.
    Equals,
    /// End-of-file marker.
    Eof,
}

impl fmt::Display for TokenKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Number(n) => write!(f, "{n}"),
            Self::Ident(s) => write!(f, "{s}"),
            Self::Plus => write!(f, "+"),
            Self::Minus => write!(f, "-"),
            Self::Star => write!(f, "*"),
            Self::Slash => write!(f, "/"),
            Self::Caret => write!(f, "^"),
            Self::LParen => write!(f, "("),
            Self::RParen => write!(f, ")"),
            Self::Comma => write!(f, ","),
            Self::Equals => write!(f, "="),
            Self::Eof => write!(f, "<EOF>"),
        }
    }
}

/// Token with position span in the source input.
#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    /// Kind of token.
    pub kind: TokenKind,
    /// Source span.
    pub span: Span,
}

/// Error encountered during lexical analysis.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("Lexer error at offset {start}..{end}: {message}")]
pub struct LexError {
    /// Start byte index.
    pub start: usize,
    /// End byte index.
    pub end: usize,
    /// Descriptive error message.
    pub message: String,
}

/// Tokenizes a mathematical expression string into a vector of tokens.
pub fn tokenize(input: &str) -> Result<Vec<Token>, LexError> {
    let mut tokens = Vec::new();
    let bytes = input.as_bytes();
    let len = bytes.len();
    let mut i = 0;

    while i < len {
        let b = bytes[i];
        if b.is_ascii_whitespace() {
            i += 1;
            continue;
        }

        if let Some(kind) = match_single_char(b) {
            tokens.push(Token {
                kind,
                span: Span::new(i, i + 1),
            });
            i += 1;
        } else if b.is_ascii_digit() || b == b'.' {
            let (val, next_i) = parse_number(input, i)?;
            tokens.push(Token {
                kind: TokenKind::Number(val),
                span: Span::new(i, next_i),
            });
            i = next_i;
        } else if b.is_ascii_alphabetic() || b == b'_' {
            let (ident, next_i) = parse_ident(input, i);
            tokens.push(Token {
                kind: TokenKind::Ident(ident),
                span: Span::new(i, next_i),
            });
            i = next_i;
        } else {
            return Err(LexError {
                start: i,
                end: i + 1,
                message: format!(
                    "Unexpected character '{}'",
                    input[i..].chars().next().unwrap_or(' ')
                ),
            });
        }
    }

    tokens.push(Token {
        kind: TokenKind::Eof,
        span: Span::new(len, len),
    });
    Ok(tokens)
}

fn match_single_char(b: u8) -> Option<TokenKind> {
    match b {
        b'+' => Some(TokenKind::Plus),
        b'-' => Some(TokenKind::Minus),
        b'*' => Some(TokenKind::Star),
        b'/' => Some(TokenKind::Slash),
        b'^' => Some(TokenKind::Caret),
        b'(' => Some(TokenKind::LParen),
        b')' => Some(TokenKind::RParen),
        b',' => Some(TokenKind::Comma),
        b'=' => Some(TokenKind::Equals),
        _ => None,
    }
}

fn parse_ident(input: &str, start: usize) -> (String, usize) {
    let bytes = input.as_bytes();
    let mut end = start + 1;
    while end < bytes.len()
        && (bytes[end].is_ascii_alphanumeric() || bytes[end] == b'_' || bytes[end] == b'\'')
    {
        end += 1;
    }
    (input[start..end].to_string(), end)
}

fn parse_number(input: &str, start: usize) -> Result<(f64, usize), LexError> {
    let bytes = input.as_bytes();
    let len = bytes.len();
    let mut i = start;
    let mut has_digits = false;
    let mut has_dot = false;

    while i < len && bytes[i].is_ascii_digit() {
        has_digits = true;
        i += 1;
    }

    if i < len && bytes[i] == b'.' {
        has_dot = true;
        i += 1;
        while i < len && bytes[i].is_ascii_digit() {
            has_digits = true;
            i += 1;
        }
    }

    if !has_digits && has_dot {
        return Err(LexError {
            start,
            end: i,
            message: "Single period is not a valid number".into(),
        });
    }

    if i < len && (bytes[i] == b'e' || bytes[i] == b'E') {
        let e_start = i;
        i += 1;
        if i < len && (bytes[i] == b'+' || bytes[i] == b'-') {
            i += 1;
        }
        let exp_digits_start = i;
        while i < len && bytes[i].is_ascii_digit() {
            i += 1;
        }
        if i == exp_digits_start {
            i = e_start;
        }
    }

    let num_str = &input[start..i];
    match num_str.parse::<f64>() {
        Ok(val) => Ok((val, i)),
        Err(_) => Err(LexError {
            start,
            end: i,
            message: format!("Invalid numeric literal '{num_str}'"),
        }),
    }
}
