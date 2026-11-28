use oxidize_core::ast_visitor::extract_opt_out_reasons_from_attr;
use proc_macro2::TokenTree;
use syn::spanned::Spanned;
use syn::visit::{self, Visit};
use syn::{Attribute, UseTree};

const UI_CRATE_ROOTS: &[&str] = &[
    "eframe",
    "egui",
    "egui_plot",
    "js_sys",
    "muda",
    "rfd",
    "wasm_bindgen",
    "wasm_bindgen_futures",
    "wasm_bindgen_test",
    "web_sys",
];

const GUI_FRAMEWORK_MODULES: &[&str] = &[
    "accessibility",
    "app",
    "async_sim",
    "audio",
    "camera",
    "command_palette",
    "framework",
    "presets",
    "starter_gallery",
    "url_state",
    "volumetric_scene",
    "web_io",
    "webgl_recovery",
];

fn is_ui_crate_root(ident: &str) -> bool {
    UI_CRATE_ROOTS.contains(&ident)
}

fn is_gui_framework_module(ident: &str) -> bool {
    GUI_FRAMEWORK_MODULES.contains(&ident)
}

fn is_gui_tab_trait_ident(ident: &str) -> bool {
    matches!(ident, "ExplorerTab" | "TabDescriptor")
}

/// Returns `true` when `file_path` (or `package_name` + `src/`) targets `math_explorer_gui/src/`.
pub(crate) fn is_math_explorer_gui_src_file(file_path: &str, package_name: Option<&str>) -> bool {
    let norm = file_path.replace('\\', "/");
    let clean = norm.trim_start_matches("./");
    if clean.starts_with("apps/math_explorer_gui/src/")
        || clean.starts_with("math_explorer_gui/src/")
        || clean.contains("/math_explorer_gui/src/")
    {
        return true;
    }
    package_name == Some("math_explorer_gui")
        && (clean.starts_with("src/") || clean.contains("/src/"))
}

struct GuiOptOutAttrFinder {
    first_gui_tool_line: Option<usize>,
}

impl GuiOptOutAttrFinder {
    fn inspect_attrs(&mut self, attrs: &[Attribute]) {
        if self.first_gui_tool_line.is_some() {
            return;
        }
        for attr in attrs {
            let reasons = extract_opt_out_reasons_from_attr(attr);
            if reasons.iter().any(|r| r == "gui_tool") {
                self.first_gui_tool_line = Some(attr.span().start().line.max(1));
                return;
            }
        }
    }
}

impl<'ast> Visit<'ast> for GuiOptOutAttrFinder {
    fn visit_attribute(&mut self, attr: &'ast Attribute) {
        self.inspect_attrs(std::slice::from_ref(attr));
    }
}

struct UiImportDetector {
    has_ui_reference: bool,
}

impl UiImportDetector {
    fn check_segments(&mut self, segs: &[String]) {
        if self.has_ui_reference || segs.is_empty() {
            return;
        }
        if is_ui_crate_root(segs[0].as_str()) {
            self.has_ui_reference = true;
            return;
        }
        if segs[0] == "crate"
            && segs.len() >= 2
            && (is_gui_framework_module(segs[1].as_str())
                || (segs[1] == "tabs"
                    && segs.len() >= 3
                    && is_gui_tab_trait_ident(segs[2].as_str())))
        {
            self.has_ui_reference = true;
        }
    }

    fn walk_use_tree(&mut self, tree: &UseTree, prefix: &mut Vec<String>) {
        if self.has_ui_reference {
            return;
        }
        match tree {
            UseTree::Path(p) => {
                prefix.push(p.ident.to_string());
                self.check_segments(prefix);
                self.walk_use_tree(&p.tree, prefix);
                prefix.pop();
            }
            UseTree::Name(n) => {
                prefix.push(n.ident.to_string());
                self.check_segments(prefix);
                prefix.pop();
            }
            UseTree::Rename(r) => {
                prefix.push(r.ident.to_string());
                self.check_segments(prefix);
                prefix.pop();
            }
            UseTree::Glob(_) => {
                self.check_segments(prefix);
            }
            UseTree::Group(g) => {
                for item in &g.items {
                    self.walk_use_tree(item, prefix);
                }
            }
        }
    }

    fn scan_tokens(&mut self, tokens: proc_macro2::TokenStream) {
        for tt in tokens {
            if self.has_ui_reference {
                return;
            }
            match tt {
                TokenTree::Ident(id) if is_ui_crate_root(&id.to_string()) => {
                    self.has_ui_reference = true;
                    return;
                }
                TokenTree::Group(g) => self.scan_tokens(g.stream()),
                _ => {}
            }
        }
    }
}

impl<'ast> Visit<'ast> for UiImportDetector {
    fn visit_item_use(&mut self, node: &'ast syn::ItemUse) {
        let mut prefix = Vec::new();
        self.walk_use_tree(&node.tree, &mut prefix);
        visit::visit_item_use(self, node);
    }

    fn visit_path(&mut self, path: &'ast syn::Path) {
        if !self.has_ui_reference {
            let segs: Vec<String> = path.segments.iter().map(|s| s.ident.to_string()).collect();
            self.check_segments(&segs);
        }
        visit::visit_path(self, path);
    }

    fn visit_macro(&mut self, mac: &'ast syn::Macro) {
        if !self.has_ui_reference {
            let segs: Vec<String> = mac
                .path
                .segments
                .iter()
                .map(|s| s.ident.to_string())
                .collect();
            self.check_segments(&segs);
            self.scan_tokens(mac.tokens.clone());
        }
        visit::visit_macro(self, mac);
    }
}

/// Rejects `opt_out = "gui_tool"` on headless files inside `math_explorer_gui/src/` that lack UI imports.
pub(crate) fn audit_headless_gui_opt_out(
    ast: &syn::File,
    file_path: &str,
    package_name: Option<&str>,
) -> bool {
    if !is_math_explorer_gui_src_file(file_path, package_name) {
        return true;
    }
    let mut finder = GuiOptOutAttrFinder {
        first_gui_tool_line: None,
    };
    finder.visit_file(ast);
    let Some(line) = finder.first_gui_tool_line else {
        return true;
    };

    let mut detector = UiImportDetector {
        has_ui_reference: false,
    };
    detector.visit_file(ast);
    if detector.has_ui_reference {
        return true;
    }

    let loc = crate::ast_helpers::format_file_line(file_path, line);
    println!(
        "[!] Headless `opt_out = \"gui_tool\"` detected at {loc} without UI imports\n  Hint: Promote headless solvers and mathematical models into `pure_math` or `domain_*` (`crates/03_domains/`), or remove `opt_out = \"gui_tool\"` from `{file_path}`."
    );
    false
}
