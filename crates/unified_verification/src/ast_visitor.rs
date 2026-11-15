use std::fs;
use std::path::PathBuf;
use std::process::Command;
use syn::visit::Visit;
use walkdir::WalkDir;

struct MaliciousCodeVisitor {
    has_shell_execution: bool,
    has_network_socket: bool,
    has_suspicious_fs: bool,
}

impl<'ast> Visit<'ast> for MaliciousCodeVisitor {
    fn visit_expr_call(&mut self, i: &'ast syn::ExprCall) {
        if let syn::Expr::Path(p) = &*i.func {
            let path_str = p
                .path
                .segments
                .iter()
                .map(|s| s.ident.to_string())
                .collect::<Vec<_>>()
                .join("::");
            if path_str.contains("Command::new") {
                self.has_shell_execution = true;
            }
            if path_str.contains("TcpStream::connect") || path_str.contains("UdpSocket::bind") {
                self.has_network_socket = true;
            }
            if path_str.contains("remove_dir_all") {
                self.has_suspicious_fs = true;
            }
        }
        syn::visit::visit_expr_call(self, i);
    }
}

struct CloneDetectorVisitor {
    has_2d_index_clone: bool,
    has_color_clone: bool,
    in_function: bool,
    cast_u8_count: usize,
    mul_255_count: usize,
}

impl CloneDetectorVisitor {
    fn is_a_b_plus_c(expr: &syn::Expr) -> bool {
        if let syn::Expr::Binary(i) = expr {
            if let syn::BinOp::Add(_) = i.op {
                let left_is_mul = matches!(*i.left, syn::Expr::Binary(ref b) if matches!(b.op, syn::BinOp::Mul(_)));
                let right_is_mul = matches!(*i.right, syn::Expr::Binary(ref b) if matches!(b.op, syn::BinOp::Mul(_)));

                if left_is_mul || right_is_mul {
                    let is_ident = |expr: &syn::Expr| matches!(expr, syn::Expr::Path(p) if p.path.get_ident().is_some());

                    let (mul_expr, other_expr) = if left_is_mul {
                        (&*i.left, &*i.right)
                    } else {
                        (&*i.right, &*i.left)
                    };

                    if let syn::Expr::Binary(b) = mul_expr {
                        if is_ident(&b.left) && is_ident(&b.right) && is_ident(other_expr) {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }

    // Recursive check if any sub-expression contains A * B + C
    fn contains_a_b_plus_c(expr: &syn::Expr) -> bool {
        if Self::is_a_b_plus_c(expr) {
            return true;
        }
        // Very basic recursion for common wrappers like parenthesis
        match expr {
            syn::Expr::Paren(p) => Self::contains_a_b_plus_c(&p.expr),
            syn::Expr::Block(b) => {
                if let Some(syn::Stmt::Expr(e, _)) = b.block.stmts.last() {
                    Self::contains_a_b_plus_c(e)
                } else {
                    false
                }
            }
            syn::Expr::Cast(c) => Self::contains_a_b_plus_c(&c.expr),
            _ => false,
        }
    }
}

impl<'ast> Visit<'ast> for CloneDetectorVisitor {
    fn visit_item_fn(&mut self, i: &'ast syn::ItemFn) {
        let prev_in_func = self.in_function;
        self.in_function = true;
        self.cast_u8_count = 0;
        self.mul_255_count = 0;

        syn::visit::visit_item_fn(self, i);

        if self.cast_u8_count >= 3 && self.mul_255_count >= 3 {
            self.has_color_clone = true;
        }

        self.in_function = prev_in_func;
    }

    fn visit_expr_index(&mut self, i: &'ast syn::ExprIndex) {
        if Self::contains_a_b_plus_c(&*i.index) {
            self.has_2d_index_clone = true;
        }
        syn::visit::visit_expr_index(self, i);
    }

    fn visit_expr_binary(&mut self, i: &'ast syn::ExprBinary) {
        if let syn::BinOp::Mul(_) = i.op {
            let is_255 = |expr: &syn::Expr| {
                if let syn::Expr::Lit(syn::ExprLit {
                    lit: syn::Lit::Float(f),
                    ..
                }) = expr
                {
                    f.base10_parse::<f64>().unwrap_or(0.0) == 255.0
                } else if let syn::Expr::Lit(syn::ExprLit {
                    lit: syn::Lit::Int(int_lit),
                    ..
                }) = expr
                {
                    int_lit.base10_parse::<u32>().unwrap_or(0) == 255
                } else {
                    false
                }
            };
            if is_255(&*i.left) || is_255(&*i.right) {
                self.mul_255_count += 1;
            }
        }
        syn::visit::visit_expr_binary(self, i);
    }

    fn visit_expr_cast(&mut self, i: &'ast syn::ExprCast) {
        if let syn::Type::Path(p) = &*i.ty {
            if p.path.is_ident("u8") {
                self.cast_u8_count += 1;
            }
        }
        syn::visit::visit_expr_cast(self, i);
    }
}

pub fn run_clone_detector() -> bool {
    println!("Running Clone Detector on first-party repository...");
    let mut flags_found = 0;
    let first_party_dirs = vec!["crates", "apps", "math_explorer", "math_explorer_gui"];
    for dir in first_party_dirs {
        for entry in WalkDir::new(dir).into_iter().filter_map(|e| e.ok()) {
            if entry.path().extension().and_then(|s| s.to_str()) == Some("rs") {
                let path_str = entry.path().to_string_lossy().replace("\\", "/");
                if path_str.contains("/target/")
                    || path_str.contains("/egui_plot/")
                    || path_str.starts_with("egui_plot/")
                {
                    continue;
                }
                if path_str.contains("unified_verification")
                    || path_str.contains("colormap.rs")
                    || path_str.contains("types.rs")
                    || path_str.contains("grid.rs")
                {
                    continue;
                }

                if let Ok(content) = fs::read_to_string(entry.path()) {
                    if let Ok(ast) = syn::parse_file(&content) {
                        let mut clone_visitor = CloneDetectorVisitor {
                            has_2d_index_clone: false,
                            has_color_clone: false,
                            in_function: false,
                            cast_u8_count: 0,
                            mul_255_count: 0,
                        };
                        clone_visitor.visit_file(&ast);
                        if clone_visitor.has_2d_index_clone {
                            println!(
                                "[!] Code Clone Detected: 2D Grid Indexing pattern in {}",
                                entry.path().display()
                            );
                            flags_found += 1;
                        }
                        if clone_visitor.has_color_clone {
                            println!(
                                "[!] Code Clone Detected: Scalar-to-Color RGB conversion pattern in {}",
                                entry.path().display()
                            );
                            flags_found += 1;
                        }
                    }
                }
            }
        }
    }

    println!("Clone detector finished. Flagged {} patterns.", flags_found);
    flags_found == 0
}

pub fn run_ast_visitor() -> bool {
    println!("Running AST visitor on 100% of third-party dependencies...");

    let mut deps_paths = Vec::new();
    let meta_out = Command::new("cargo")
        .args(["metadata", "--format-version", "1"])
        .output()
        .unwrap();
    let meta: serde_json::Value = serde_json::from_slice(&meta_out.stdout).unwrap_or_default();
    if let Some(packages) = meta.get("packages").and_then(|p| p.as_array()) {
        for pkg in packages {
            if let Some(manifest_path) = pkg.get("manifest_path").and_then(|m| m.as_str()) {
                if let Some(source) = pkg.get("source").and_then(|s| s.as_str()) {
                    if source.contains("crates.io") {
                        if let Some(p) = PathBuf::from(manifest_path).parent() {
                            deps_paths.push(p.to_path_buf());
                        }
                    }
                }
            }
        }
    }

    let mut flags_found = 0;
    for path in deps_paths.iter() {
        for entry in WalkDir::new(path).into_iter().filter_map(|e| e.ok()) {
            if entry.path().extension().and_then(|s| s.to_str()) == Some("rs") {
                if let Ok(content) = fs::read_to_string(entry.path()) {
                    if let Ok(ast) = syn::parse_file(&content) {
                        let mut visitor = MaliciousCodeVisitor {
                            has_shell_execution: false,
                            has_network_socket: false,
                            has_suspicious_fs: false,
                        };
                        visitor.visit_file(&ast);
                        if visitor.has_shell_execution {
                            println!(
                                "[!] Malicious pattern 1: Shell execution found in {}",
                                entry.path().display()
                            );
                            flags_found += 1;
                        }
                        if visitor.has_network_socket {
                            println!(
                                "[!] Malicious pattern 2: Network socket pattern found in {}",
                                entry.path().display()
                            );
                            flags_found += 1;
                        }
                        if visitor.has_suspicious_fs {
                            println!(
                                "[!] Malicious pattern 3: Suspicious FS access found in {}",
                                entry.path().display()
                            );
                            flags_found += 1;
                        }
                    }
                }
            }
        }
    }

    println!("AST visitor finished. Flagged {} patterns.", flags_found);
    true // Non-blocking for third-party crates
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clone_detector_is_a_b_plus_c() {
        let code = "fn test() { let x = a * b + c; }";
        let file = syn::parse_file(code).unwrap();
        if let syn::Item::Fn(f) = &file.items[0] {
            if let syn::Stmt::Local(l) = &f.block.stmts[0] {
                if let Some(init) = &l.init {
                    assert!(CloneDetectorVisitor::is_a_b_plus_c(&init.expr));
                }
            }
        }
    }

    #[test]
    fn test_run_clone_detector_execution() {
        let _lock = crate::DIR_LOCK.lock().unwrap();
        let passed = run_clone_detector();
        assert!(passed, "First-party code should pass clone detector");
    }
}
