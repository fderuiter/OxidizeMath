use super::{attrs_mark_test_fn, attrs_require_cfg_test, extract_opt_out_reasons_from_attr};
use syn::visit::{self, Visit};
use syn::{Attribute, ImplItem, ImplItemFn, Item, ItemFn, TraitItem, TraitItemFn};

fn all_item_attrs(node: &Item) -> &[Attribute] {
    match node {
        Item::Const(i) => &i.attrs,
        Item::Enum(i) => &i.attrs,
        Item::ExternCrate(i) => &i.attrs,
        Item::Fn(i) => &i.attrs,
        Item::ForeignMod(i) => &i.attrs,
        Item::Impl(i) => &i.attrs,
        Item::Macro(i) => &i.attrs,
        Item::Mod(i) => &i.attrs,
        Item::Static(i) => &i.attrs,
        Item::Struct(i) => &i.attrs,
        Item::Trait(i) => &i.attrs,
        Item::TraitAlias(i) => &i.attrs,
        Item::Type(i) => &i.attrs,
        Item::Union(i) => &i.attrs,
        Item::Use(i) => &i.attrs,
        _ => &[],
    }
}

struct FileDebtAstVisitor {
    in_test_scope: bool,
    entries: Vec<String>,
}

impl FileDebtAstVisitor {
    fn check_legacy_fn(&mut self, attrs: &[Attribute], ident: &syn::Ident) {
        if self.in_test_scope || attrs_mark_test_fn(attrs) {
            return;
        }
        let name = ident.to_string();
        if name.ends_with("_legacy") {
            self.entries.push(format!("Legacy function call: {name}"));
        }
    }
}

impl<'ast> Visit<'ast> for FileDebtAstVisitor {
    fn visit_attribute(&mut self, i: &'ast Attribute) {
        self.entries.extend(extract_opt_out_reasons_from_attr(i));
        visit::visit_attribute(self, i);
    }

    fn visit_item(&mut self, i: &'ast Item) {
        let prev = self.in_test_scope;
        if attrs_require_cfg_test(all_item_attrs(i)) {
            self.in_test_scope = true;
        }
        visit::visit_item(self, i);
        self.in_test_scope = prev;
    }

    fn visit_impl_item(&mut self, i: &'ast ImplItem) {
        let attrs: &[Attribute] = match i {
            ImplItem::Const(c) => &c.attrs,
            ImplItem::Fn(f) => &f.attrs,
            ImplItem::Type(t) => &t.attrs,
            ImplItem::Macro(m) => &m.attrs,
            _ => &[],
        };
        let prev = self.in_test_scope;
        if attrs_require_cfg_test(attrs) {
            self.in_test_scope = true;
        }
        visit::visit_impl_item(self, i);
        self.in_test_scope = prev;
    }

    fn visit_trait_item(&mut self, i: &'ast TraitItem) {
        let attrs: &[Attribute] = match i {
            TraitItem::Const(c) => &c.attrs,
            TraitItem::Fn(f) => &f.attrs,
            TraitItem::Type(t) => &t.attrs,
            TraitItem::Macro(m) => &m.attrs,
            _ => &[],
        };
        let prev = self.in_test_scope;
        if attrs_require_cfg_test(attrs) {
            self.in_test_scope = true;
        }
        visit::visit_trait_item(self, i);
        self.in_test_scope = prev;
    }

    fn visit_item_fn(&mut self, i: &'ast ItemFn) {
        self.check_legacy_fn(&i.attrs, &i.sig.ident);
        visit::visit_item_fn(self, i);
    }

    fn visit_impl_item_fn(&mut self, i: &'ast ImplItemFn) {
        self.check_legacy_fn(&i.attrs, &i.sig.ident);
        visit::visit_impl_item_fn(self, i);
    }

    fn visit_trait_item_fn(&mut self, i: &'ast TraitItemFn) {
        self.check_legacy_fn(&i.attrs, &i.sig.ident);
        visit::visit_trait_item_fn(self, i);
    }
}

/// Collects all AST `verified(opt_out = "...")` reasons (direct and `cfg_attr`) and non-test
/// `fn *_legacy` markers from a parsed `syn::File`, ignoring comments and string literals.
pub fn collect_ast_file_debt(ast: &syn::File) -> Vec<String> {
    let mut visitor = FileDebtAstVisitor {
        in_test_scope: attrs_require_cfg_test(&ast.attrs),
        entries: Vec::new(),
    };
    visitor.visit_file(ast);
    visitor.entries
}
