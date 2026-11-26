//! Tool-specific variable scope validation for `MathExpressionInputBar`.

use math_commons::ast::{is_universal_constant, Expr, MAX_EVAL_DEPTH};

/// Variable scope rules enforced on parsed expressions before compilation.
#[derive(Clone, Debug, Default)]
pub(super) struct VariableScopePolicy {
    allowed_vars: Option<Vec<String>>,
    disallowed_vars: Vec<(String, String)>,
}

impl VariableScopePolicy {
    /// Sets a closed list of allowed variable and parameter identifiers.
    pub(super) fn set_allowed_vars(&mut self, vars: &[&str]) {
        self.allowed_vars = Some(vars.iter().map(|&v| v.to_string()).collect());
    }

    /// Sets explicit disallowed variable identifiers with user-facing guidance.
    pub(super) fn set_disallowed_vars(&mut self, rules: &[(&str, &str)]) {
        self.disallowed_vars = rules
            .iter()
            .map(|&(name, hint)| (name.to_string(), hint.to_string()))
            .collect();
    }

    /// Validates all variable references in `expr` against configured tool symbols.
    pub(super) fn validate(
        &self,
        expr: &Expr,
        axes: &[String],
        complex_mode: bool,
    ) -> Result<(), String> {
        if self.allowed_vars.is_none() && self.disallowed_vars.is_empty() {
            return Ok(());
        }

        for var in collect_referenced_variables(expr) {
            if is_builtin_constant(&var, complex_mode) {
                continue;
            }
            if let Some((bad, hint)) = self
                .disallowed_vars
                .iter()
                .find(|(disallowed, _)| disallowed == &var)
            {
                return Err(format!("Unknown variable '{bad}' ({hint})"));
            }
            if let Some(allowed) = &self.allowed_vars {
                if !axes.iter().any(|a| a == &var) && !allowed.iter().any(|a| a == &var) {
                    let summary = format_allowed_summary(axes, allowed);
                    return Err(format!(
                        "Unknown variable '{var}'. Allowed variables: {summary}"
                    ));
                }
            }
        }
        Ok(())
    }
}

fn is_builtin_constant(name: &str, complex_mode: bool) -> bool {
    is_universal_constant(name) || (complex_mode && name.eq_ignore_ascii_case("i"))
}

fn format_allowed_summary(axes: &[String], allowed: &[String]) -> String {
    let mut combined: Vec<&str> = Vec::with_capacity(axes.len() + allowed.len());
    for item in axes.iter().chain(allowed.iter()) {
        if !combined.contains(&item.as_str()) {
            combined.push(item.as_str());
        }
    }
    if combined.is_empty() {
        "none (constants pi, e, tau only)".to_string()
    } else {
        combined.join(", ")
    }
}

fn collect_referenced_variables(root: &Expr) -> Vec<String> {
    let mut found = Vec::new();
    let mut stack: Vec<(&Expr, usize)> = vec![(root, 0)];
    while let Some((node, depth)) = stack.pop() {
        if depth > MAX_EVAL_DEPTH {
            continue;
        }
        match node {
            Expr::Number(_) => {}
            Expr::Variable(name) => {
                if !found.iter().any(|existing| existing == name) {
                    found.push(name.clone());
                }
            }
            Expr::Unary { expr, .. } => stack.push((expr, depth + 1)),
            Expr::Binary { left, right, .. } => {
                stack.push((right, depth + 1));
                stack.push((left, depth + 1));
            }
            Expr::Function { args, .. } => {
                for arg in args.iter().rev() {
                    stack.push((arg, depth + 1));
                }
            }
        }
    }
    found
}
