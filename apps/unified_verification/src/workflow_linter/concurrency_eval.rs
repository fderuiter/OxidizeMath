use super::expressions::{parse_single_quoted_literal, skip_ascii_ws, skip_single_quoted};

#[derive(Clone, Debug)]
pub(super) enum GhaVal {
    Null,
    Bool(bool),
    Num(i64),
    Str(String),
}

impl GhaVal {
    pub(super) fn is_truthy(&self) -> bool {
        match self {
            Self::Null => false,
            Self::Bool(b) => *b,
            Self::Num(n) => *n != 0,
            Self::Str(s) => !s.is_empty(),
        }
    }

    pub(super) fn as_gha_str(&self) -> String {
        match self {
            Self::Null => String::new(),
            Self::Bool(b) => b.to_string(),
            Self::Num(n) => n.to_string(),
            Self::Str(s) => s.clone(),
        }
    }

    fn as_f64(&self) -> f64 {
        match self {
            Self::Null | Self::Bool(false) => 0.0,
            Self::Bool(true) => 1.0,
            Self::Num(n) => *n as f64,
            Self::Str(s) => {
                let t = s.trim();
                if t.is_empty() {
                    0.0
                } else {
                    t.parse::<f64>().unwrap_or(f64::NAN)
                }
            }
        }
    }

    fn gha_eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Null, Self::Null) => true,
            (Self::Bool(a), Self::Bool(b)) => a == b,
            (Self::Num(a), Self::Num(b)) => a == b,
            (Self::Str(a), Self::Str(b)) => a.eq_ignore_ascii_case(b),
            (Self::Null, _) | (_, Self::Null) => false,
            _ => self.as_f64() == other.as_f64(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum FuncKind {
    StartsWith,
    EndsWith,
    Contains,
    Format,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum BinOp {
    Eq,
    Ne,
    And,
    Or,
}

impl BinOp {
    fn precedence(self) -> u8 {
        match self {
            Self::Eq | Self::Ne => 3,
            Self::And => 2,
            Self::Or => 1,
        }
    }
}

#[derive(Clone, Debug)]
pub(super) enum Token {
    Val(GhaVal),
    Context(Vec<String>),
    Func(FuncKind),
    LParen,
    RParen,
    Comma,
    Not,
    Bin(BinOp),
}

#[derive(Clone, Copy, Debug)]
enum OpItem {
    Group,
    Call(FuncKind, usize),
    UnaryNot,
    Binary(BinOp),
}

#[derive(Default, Clone, Copy, Debug)]
pub(super) struct ExprRefs {
    pub(super) uses_branch_ref: bool,
    pub(super) uses_unique_run_key: bool,
    pub(super) uses_pr_only_key: bool,
}

impl ExprRefs {
    pub(super) fn merge(&mut self, other: Self) {
        self.uses_branch_ref |= other.uses_branch_ref;
        self.uses_unique_run_key |= other.uses_unique_run_key;
        self.uses_pr_only_key |= other.uses_pr_only_key;
    }

    pub(super) fn record_context_path(&mut self, segments: &[String]) {
        if !segments
            .first()
            .is_some_and(|r| r.eq_ignore_ascii_case("github"))
        {
            return;
        }
        let Some(prop) = segments.get(1).map(String::as_str) else {
            return;
        };
        if segments.len() == 2
            && (prop.eq_ignore_ascii_case("ref") || prop.eq_ignore_ascii_case("ref_name"))
        {
            self.uses_branch_ref = true;
        } else if segments.len() == 2
            && (prop.eq_ignore_ascii_case("run_id") || prop.eq_ignore_ascii_case("sha"))
        {
            self.uses_unique_run_key = true;
        } else if prop.eq_ignore_ascii_case("head_ref")
            || (prop.eq_ignore_ascii_case("event")
                && segments
                    .get(2)
                    .is_some_and(|s| s.eq_ignore_ascii_case("pull_request")))
        {
            self.uses_pr_only_key = true;
        }
    }
}

pub(super) struct PushSimContext<'a> {
    pub(super) branch: &'a str,
    pub(super) run_id: &'a str,
    pub(super) sha: &'a str,
}

impl PushSimContext<'_> {
    fn resolve(&self, segments: &[String]) -> Result<GhaVal, ()> {
        if !segments
            .first()
            .is_some_and(|r| r.eq_ignore_ascii_case("github"))
        {
            return Err(());
        }
        let prop = segments.get(1).ok_or(())?.to_ascii_lowercase();
        if prop == "event" {
            return match segments.get(2).map(|s| s.to_ascii_lowercase()).as_deref() {
                Some("pull_request") => Ok(GhaVal::Null),
                Some("before") => Ok(GhaVal::Str("0".repeat(40))),
                _ => Err(()),
            };
        }
        if segments.len() != 2 {
            return Err(());
        }
        match prop.as_str() {
            "workflow" => Ok(GhaVal::Str("CI".to_string())),
            "event_name" => Ok(GhaVal::Str("push".to_string())),
            "ref" => Ok(GhaVal::Str(format!("refs/heads/{}", self.branch))),
            "ref_name" => Ok(GhaVal::Str(self.branch.to_string())),
            "ref_type" => Ok(GhaVal::Str("branch".to_string())),
            "head_ref" | "base_ref" => Ok(GhaVal::Str(String::new())),
            "run_id" | "run_number" => Ok(GhaVal::Str(self.run_id.to_string())),
            "sha" => Ok(GhaVal::Str(self.sha.to_string())),
            "repository" => Ok(GhaVal::Str("owner/repo".to_string())),
            "actor" => Ok(GhaVal::Str("actor".to_string())),
            _ => Err(()),
        }
    }
}

fn read_ident(expr: &str, start: usize) -> Option<(&str, usize)> {
    let bytes = expr.as_bytes();
    let first = *bytes.get(start)?;
    if !(first.is_ascii_alphabetic() || first == b'_') {
        return None;
    }
    let mut end = start + 1;
    while end < bytes.len()
        && (bytes[end].is_ascii_alphanumeric() || bytes[end] == b'_' || bytes[end] == b'-')
    {
        end += 1;
    }
    Some((&expr[start..end], end))
}

pub(super) fn parse_ident_or_context(expr: &str, start: usize) -> Result<(Token, usize), ()> {
    let (head, mut idx) = read_ident(expr, start).ok_or(())?;
    let bytes = expr.as_bytes();
    let after_head = skip_ascii_ws(bytes, idx);
    if bytes.get(after_head) == Some(&b'(') {
        let func = match head.to_ascii_lowercase().as_str() {
            "startswith" => FuncKind::StartsWith,
            "endswith" => FuncKind::EndsWith,
            "contains" => FuncKind::Contains,
            "format" => FuncKind::Format,
            _ => return Err(()),
        };
        return Ok((Token::Func(func), after_head));
    }
    match head.to_ascii_lowercase().as_str() {
        "true" => return Ok((Token::Val(GhaVal::Bool(true)), idx)),
        "false" => return Ok((Token::Val(GhaVal::Bool(false)), idx)),
        "null" => return Ok((Token::Val(GhaVal::Null), idx)),
        _ => {}
    }
    let mut segments = vec![head.to_string()];
    loop {
        let pos = skip_ascii_ws(bytes, idx);
        if bytes.get(pos) == Some(&b'.') {
            let (prop, next_idx) = read_ident(expr, skip_ascii_ws(bytes, pos + 1)).ok_or(())?;
            segments.push(prop.to_string());
            idx = next_idx;
        } else if bytes.get(pos) == Some(&b'[') {
            let (prop, after_lit) =
                parse_single_quoted_literal(expr, skip_ascii_ws(bytes, pos + 1)).ok_or(())?;
            let close = skip_ascii_ws(bytes, after_lit);
            if bytes.get(close) != Some(&b']') {
                return Err(());
            }
            segments.push(prop);
            idx = close + 1;
        } else {
            break;
        }
    }
    Ok((Token::Context(segments), idx))
}

fn read_punct_or_literal_token(expr: &str, idx: usize) -> Option<Result<(Token, usize), ()>> {
    let bytes = expr.as_bytes();
    match bytes[idx] {
        b'\'' => Some(
            parse_single_quoted_literal(expr, idx)
                .map(|(lit, next)| (Token::Val(GhaVal::Str(lit)), next))
                .ok_or(()),
        ),
        b'(' => Some(Ok((Token::LParen, idx + 1))),
        b')' => Some(Ok((Token::RParen, idx + 1))),
        b',' => Some(Ok((Token::Comma, idx + 1))),
        b'!' if bytes.get(idx + 1) == Some(&b'=') => Some(Ok((Token::Bin(BinOp::Ne), idx + 2))),
        b'!' => Some(Ok((Token::Not, idx + 1))),
        b'=' if bytes.get(idx + 1) == Some(&b'=') => Some(Ok((Token::Bin(BinOp::Eq), idx + 2))),
        b'&' if bytes.get(idx + 1) == Some(&b'&') => Some(Ok((Token::Bin(BinOp::And), idx + 2))),
        b'|' if bytes.get(idx + 1) == Some(&b'|') => Some(Ok((Token::Bin(BinOp::Or), idx + 2))),
        b if b.is_ascii_digit() => {
            let mut end = idx + 1;
            while end < bytes.len() && bytes[end].is_ascii_digit() {
                end += 1;
            }
            Some(
                expr[idx..end]
                    .parse::<i64>()
                    .map(|n| (Token::Val(GhaVal::Num(n)), end))
                    .map_err(|_| ()),
            )
        }
        _ => None,
    }
}

pub(super) fn tokenize_expr(expr: &str) -> Result<(Vec<Token>, ExprRefs), ()> {
    let bytes = expr.as_bytes();
    let (mut idx, mut tokens, mut refs) = (0, Vec::new(), ExprRefs::default());
    while idx < bytes.len() {
        idx = skip_ascii_ws(bytes, idx);
        if idx >= bytes.len() {
            break;
        }
        if let Some(res) = read_punct_or_literal_token(expr, idx) {
            let (tok, next) = res?;
            tokens.push(tok);
            idx = next;
        } else {
            let (tok, next) = parse_ident_or_context(expr, idx)?;
            if let Token::Context(segs) = &tok {
                refs.record_context_path(segs);
            }
            tokens.push(tok);
            idx = next;
        }
    }
    Ok((tokens, refs))
}

fn apply_func(kind: FuncKind, args: &[GhaVal]) -> Result<GhaVal, ()> {
    if let FuncKind::Format = kind {
        let mut rendered = args.first().ok_or(())?.as_gha_str();
        for (i, arg) in args.iter().skip(1).enumerate() {
            rendered = rendered.replace(&format!("{{{i}}}"), &arg.as_gha_str());
        }
        return Ok(GhaVal::Str(rendered));
    }
    if args.len() != 2 {
        return Err(());
    }
    let hay = args[0].as_gha_str().to_ascii_lowercase();
    let needle = args[1].as_gha_str().to_ascii_lowercase();
    let matched = match kind {
        FuncKind::StartsWith => hay.starts_with(&needle),
        FuncKind::EndsWith => hay.ends_with(&needle),
        FuncKind::Contains => hay.contains(&needle),
        FuncKind::Format => false,
    };
    Ok(GhaVal::Bool(matched))
}

fn pop_and_apply_op(ops: &mut Vec<OpItem>, vals: &mut Vec<GhaVal>) -> Result<(), ()> {
    match ops.pop().ok_or(())? {
        OpItem::UnaryNot => {
            let v = vals.pop().ok_or(())?;
            vals.push(GhaVal::Bool(!v.is_truthy()));
            Ok(())
        }
        OpItem::Binary(op) => {
            let (rhs, lhs) = (vals.pop().ok_or(())?, vals.pop().ok_or(())?);
            let out = match op {
                BinOp::Eq => GhaVal::Bool(lhs.gha_eq(&rhs)),
                BinOp::Ne => GhaVal::Bool(!lhs.gha_eq(&rhs)),
                BinOp::And => {
                    if lhs.is_truthy() {
                        rhs
                    } else {
                        lhs
                    }
                }
                BinOp::Or => {
                    if lhs.is_truthy() {
                        lhs
                    } else {
                        rhs
                    }
                }
            };
            vals.push(out);
            Ok(())
        }
        OpItem::Group | OpItem::Call(_, _) => Err(()),
    }
}

fn top_precedence(ops: &[OpItem]) -> Option<u8> {
    match ops.last()? {
        OpItem::UnaryNot => Some(4),
        OpItem::Binary(op) => Some(op.precedence()),
        OpItem::Group | OpItem::Call(_, _) => None,
    }
}

fn finish_rparen(ops: &mut Vec<OpItem>, vals: &mut Vec<GhaVal>) -> Result<(), ()> {
    while top_precedence(ops).is_some() {
        pop_and_apply_op(ops, vals)?;
    }
    match ops.pop().ok_or(())? {
        OpItem::Group => Ok(()),
        OpItem::Call(kind, argc) if vals.len() >= argc => {
            let args = vals.split_off(vals.len() - argc);
            vals.push(apply_func(kind, &args)?);
            Ok(())
        }
        _ => Err(()),
    }
}

fn step_eval_token(
    tokens: &[Token],
    idx: &mut usize,
    expect_operand: &mut bool,
    ops: &mut Vec<OpItem>,
    vals: &mut Vec<GhaVal>,
    ctx: &PushSimContext<'_>,
) -> Result<(), ()> {
    match (&tokens[*idx], *expect_operand) {
        (Token::Val(v), true) => {
            vals.push(v.clone());
            *expect_operand = false;
        }
        (Token::Context(segs), true) => {
            vals.push(ctx.resolve(segs)?);
            *expect_operand = false;
        }
        (Token::Not, true) => ops.push(OpItem::UnaryNot),
        (Token::LParen, true) => ops.push(OpItem::Group),
        (Token::Func(kind), true) if matches!(tokens.get(*idx + 1), Some(Token::LParen)) => {
            let zero_args = matches!(tokens.get(*idx + 2), Some(Token::RParen));
            ops.push(OpItem::Call(*kind, usize::from(!zero_args)));
            *idx += 1;
        }
        (Token::RParen, true) if matches!(ops.last(), Some(OpItem::Call(_, 0))) => {
            finish_rparen(ops, vals)?;
            *expect_operand = false;
        }
        (Token::RParen, false) => finish_rparen(ops, vals)?,
        (Token::Comma, false) => {
            while top_precedence(ops).is_some() {
                pop_and_apply_op(ops, vals)?;
            }
            let Some(OpItem::Call(_, argc)) = ops.last_mut() else {
                return Err(());
            };
            *argc += 1;
            *expect_operand = true;
        }
        (Token::Bin(op), false) => {
            while top_precedence(ops).is_some_and(|p| p >= op.precedence()) {
                pop_and_apply_op(ops, vals)?;
            }
            ops.push(OpItem::Binary(*op));
            *expect_operand = true;
        }
        _ => return Err(()),
    }
    Ok(())
}

pub(super) fn eval_tokens(tokens: &[Token], ctx: &PushSimContext<'_>) -> Result<GhaVal, ()> {
    if tokens.is_empty() {
        return Err(());
    }
    let (mut ops, mut vals, mut expect_operand, mut idx) = (Vec::new(), Vec::new(), true, 0);
    while idx < tokens.len() {
        step_eval_token(
            tokens,
            &mut idx,
            &mut expect_operand,
            &mut ops,
            &mut vals,
            ctx,
        )?;
        idx += 1;
    }
    if expect_operand {
        return Err(());
    }
    while !ops.is_empty() {
        pop_and_apply_op(&mut ops, &mut vals)?;
    }
    (vals.len() == 1)
        .then(|| vals.pop().unwrap_or(GhaVal::Null))
        .ok_or(())
}

pub(super) fn next_closed_gha_expr(input: &str) -> Result<Option<(&str, &str, &str)>, ()> {
    let Some(start) = input.find("${{") else {
        return Ok(None);
    };
    let rest = &input[start + 3..];
    let bytes = rest.as_bytes();
    let mut idx = 0;
    while idx < bytes.len() {
        if bytes[idx] == b'\'' {
            idx = skip_single_quoted(bytes, idx);
        } else if idx + 1 < bytes.len() && bytes[idx] == b'}' && bytes[idx + 1] == b'}' {
            return Ok(Some((&input[..start], &rest[..idx], &rest[idx + 2..])));
        } else {
            idx += 1;
        }
    }
    Err(())
}
