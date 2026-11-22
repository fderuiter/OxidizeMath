use crate::latex_parser::MathOp;
use std::collections::HashSet;
use syn::visit::Visit;

struct RustOpVisitor {
    ops: HashSet<MathOp>,
}

impl<'ast> Visit<'ast> for RustOpVisitor {
    fn visit_expr_binary(&mut self, node: &'ast syn::ExprBinary) {
        match node.op {
            syn::BinOp::Add(_) => {
                self.ops.insert(MathOp::Add);
            }
            syn::BinOp::Sub(_) => {
                self.ops.insert(MathOp::Sub);
            }
            syn::BinOp::Mul(_) => {
                self.ops.insert(MathOp::Mul);
            }
            syn::BinOp::Div(_) => {
                self.ops.insert(MathOp::Div);
            }
            syn::BinOp::BitXor(_) => {
                self.ops.insert(MathOp::Pow);
            }
            _ => {}
        }
        syn::visit::visit_expr_binary(self, node);
    }

    fn visit_expr_unary(&mut self, node: &'ast syn::ExprUnary) {
        if let syn::UnOp::Neg(_) = node.op {
            self.ops.insert(MathOp::Sub);
        }
        syn::visit::visit_expr_unary(self, node);
    }

    fn visit_expr_method_call(&mut self, node: &'ast syn::ExprMethodCall) {
        let method_name = node.method.to_string();
        if method_name == "powi" || method_name == "powf" || method_name == "exp" {
            self.ops.insert(MathOp::Pow);
        }
        syn::visit::visit_expr_method_call(self, node);
    }
}

pub fn verify_block_semantic_integrity(latex_str: &str, block: &syn::Block) -> Result<(), String> {
    let mut latex_ops: HashSet<MathOp> = crate::latex_parser::parse_latex_math(latex_str)
        .into_iter()
        .collect();

    if latex_str.contains("e^") || latex_str.contains("\\exp") {
        latex_ops.insert(MathOp::Pow);
    }

    let mut visitor = RustOpVisitor {
        ops: HashSet::new(),
    };
    visitor.visit_block(block);

    for op in &latex_ops {
        if !visitor.ops.contains(op) {
            return Err(format!(
                "Missing operation in Rust implementation: expected {:?} based on LaTeX formula",
                op
            ));
        }
    }

    Ok(())
}

#[cfg(test)]
pub fn verify_semantic_integrity(latex_str: &str, rust_fn: &syn::ItemFn) -> Result<(), String> {
    verify_block_semantic_integrity(latex_str, &rust_fn.block)
}

#[cfg(test)]
mod tests {
    use super::verify_semantic_integrity;

    #[test]
    fn test_verify_matching_operations() {
        let func: syn::ItemFn = syn::parse_quote! {
            fn compute(a: f64, b: f64, c: f64, d: f64) -> f64 {
                ((a * b + c - d) / 2.0).powi(2)
            }
        };
        let latex = r"\frac{a \times b + c - d}{2}^2";
        assert_eq!(verify_semantic_integrity(latex, &func), Ok(()));
    }

    #[test]
    fn test_verify_unary_negation_and_exponential() {
        let func: syn::ItemFn = syn::parse_quote! {
            fn decay(x: f64, k: f64) -> f64 {
                (-k * x).exp()
            }
        };
        assert_eq!(
            verify_semantic_integrity(r"\exp(-k \cdot x)", &func),
            Ok(())
        );
        assert_eq!(verify_semantic_integrity(r"e^{-k * x}", &func), Ok(()));
    }

    #[test]
    fn test_verify_missing_operation_fails() {
        let func: syn::ItemFn = syn::parse_quote! {
            fn wrong_range(beat: f64, bw: f64) -> f64 {
                beat * bw
            }
        };
        let err = verify_semantic_integrity(r"\frac{f}{2 B}", &func).unwrap_err();
        assert!(err.contains("Missing operation in Rust implementation"));
        assert!(err.contains("Div"));
    }
}
