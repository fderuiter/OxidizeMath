use super::ApiItem;
use super::traversal::{
    ExportFilter, collect_facade_exports, flatten_use_tree, has_cfg_test, leaf_allowed_by_filter,
    resolve_child_mod_file,
};
use crate::ast_helpers::impl_base_ident;
use std::path::{Path, PathBuf};
use syn::visit::Visit;

const MAX_INLINE_MODULES: usize = 2_048;

pub(crate) fn extract_expiration_date(attrs: &[syn::Attribute]) -> Option<String> {
    for attr in attrs {
        if !attr.path().is_ident("deprecated") {
            continue;
        }
        let attr_str = quote::quote!(#attr).to_string();
        for key in ["expires", "expiry"] {
            if let Some(pos) = attr_str.find(key)
                && let Some(date) = find_date_pattern(&attr_str[pos..])
            {
                return Some(date);
            }
        }
        if let Some(date) = find_date_pattern(&attr_str) {
            return Some(date);
        }
    }
    None
}

pub(crate) fn find_date_pattern(s: &str) -> Option<String> {
    let bytes = s.as_bytes();
    if bytes.len() < 10 {
        return None;
    }
    for i in 0..=(bytes.len() - 10) {
        let is_digit = |b: u8| b.is_ascii_digit();
        if is_digit(bytes[i])
            && is_digit(bytes[i + 1])
            && is_digit(bytes[i + 2])
            && is_digit(bytes[i + 3])
            && bytes[i + 4] == b'-'
            && is_digit(bytes[i + 5])
            && is_digit(bytes[i + 6])
            && bytes[i + 7] == b'-'
            && is_digit(bytes[i + 8])
            && is_digit(bytes[i + 9])
        {
            return Some(s[i..i + 10].to_string());
        }
    }
    None
}

fn qualify_scope(scope: &[String], sig: String) -> String {
    if scope.is_empty() {
        return sig;
    }
    let mut wrapped = sig;
    for mod_name in scope.iter().rev() {
        wrapped = format!("pub mod {mod_name} {{ {wrapped} }}");
    }
    wrapped
}

#[derive(Default)]
pub(crate) struct ApiVisitor {
    pub(crate) items: Vec<ApiItem>,
    scope: Vec<String>,
    filter: Option<ExportFilter>,
}

struct ModContext<'a> {
    scope: &'a [String],
    filter: &'a ExportFilter,
    facades: &'a std::collections::BTreeMap<String, ExportFilter>,
    mod_dir: &'a Path,
}

impl ApiVisitor {
    fn active_allows(&self, ident: &str) -> bool {
        self.filter
            .as_ref()
            .is_none_or(|filter| filter.allows_ident(ident))
    }

    fn push_item(&mut self, raw_sig: String, attrs: &[syn::Attribute], fallback: Option<String>) {
        let signature = qualify_scope(&self.scope, raw_sig);
        let deprecation_expires = extract_expiration_date(attrs).or(fallback);
        self.items.push(ApiItem {
            signature,
            deprecation_expires,
        });
    }

    pub(crate) fn visit_module_ast(
        &mut self,
        file: &syn::File,
        root_filter: &ExportFilter,
        root_mod_dir: &Path,
    ) -> Vec<(PathBuf, ExportFilter)> {
        if has_cfg_test(&file.attrs) {
            return Vec::new();
        }

        let mut child_files = Vec::new();
        let mut stack: Vec<(Vec<String>, &[syn::Item], ExportFilter, PathBuf)> = vec![(
            Vec::new(),
            &file.items,
            root_filter.clone(),
            root_mod_dir.to_path_buf(),
        )];

        let mut steps = 0;
        while let Some((scope, mod_items, filter, mod_dir)) = stack.pop() {
            if steps >= MAX_INLINE_MODULES {
                break;
            }
            steps += 1;

            let facades = collect_facade_exports(mod_items, &filter);
            self.scope = scope.clone();
            self.filter = Some(filter.clone());

            let ctx = ModContext {
                scope: &scope,
                filter: &filter,
                facades: &facades,
                mod_dir: &mod_dir,
            };
            for item in mod_items {
                if let syn::Item::Mod(item_mod) = item {
                    self.handle_mod_item(item_mod, &ctx, &mut stack, &mut child_files);
                } else {
                    self.visit_item(item);
                }
            }
        }

        child_files
    }

    fn handle_mod_item<'a>(
        &mut self,
        item_mod: &'a syn::ItemMod,
        ctx: &ModContext<'_>,
        stack: &mut Vec<(Vec<String>, &'a [syn::Item], ExportFilter, PathBuf)>,
        child_files: &mut Vec<(PathBuf, ExportFilter)>,
    ) {
        if has_cfg_test(&item_mod.attrs) {
            return;
        }
        let mod_name = item_mod.ident.to_string();
        let is_pub_mod = matches!(item_mod.vis, syn::Visibility::Public(_));

        let child_filter = if is_pub_mod && ctx.filter.allows_ident(&mod_name) {
            Some(ExportFilter::AllPublic)
        } else {
            ctx.facades.get(&mod_name).cloned()
        };

        let Some(cf) = child_filter else {
            return;
        };

        if let Some((_, inline_items)) = &item_mod.content {
            let mut next_scope = ctx.scope.to_vec();
            if is_pub_mod && ctx.filter.allows_ident(&mod_name) {
                let ident = &item_mod.ident;
                self.push_item(
                    quote::quote!(pub mod #ident).to_string(),
                    &item_mod.attrs,
                    None,
                );
                next_scope.push(mod_name.clone());
            }
            stack.push((next_scope, inline_items, cf, ctx.mod_dir.join(mod_name)));
        } else if let Some(path) = resolve_child_mod_file(ctx.mod_dir, &mod_name, &item_mod.attrs) {
            child_files.push((path, cf));
        }
    }

    fn record_trait_items(&mut self, i: &syn::ItemTrait) {
        let trait_ident = &i.ident;
        let trait_gen = &i.generics;
        let parent_dep = extract_expiration_date(&i.attrs);

        for item in &i.items {
            match item {
                syn::TraitItem::Fn(m) if !has_cfg_test(&m.attrs) => {
                    let sig = &m.sig;
                    let raw = quote::quote!(pub trait #trait_ident #trait_gen { #sig }).to_string();
                    self.push_item(raw, &m.attrs, parent_dep.clone());
                }
                syn::TraitItem::Const(c) if !has_cfg_test(&c.attrs) => {
                    let c_id = &c.ident;
                    let c_gen = &c.generics;
                    let ty = &c.ty;
                    let raw = if let Some((_, expr)) = &c.default {
                        quote::quote!(
                            pub trait #trait_ident #trait_gen { const #c_id #c_gen : #ty = #expr }
                        )
                        .to_string()
                    } else {
                        quote::quote!(
                            pub trait #trait_ident #trait_gen { const #c_id #c_gen : #ty }
                        )
                        .to_string()
                    };
                    self.push_item(raw, &c.attrs, parent_dep.clone());
                }
                syn::TraitItem::Type(t) if !has_cfg_test(&t.attrs) => {
                    let t_id = &t.ident;
                    let t_gen = &t.generics;
                    let colon = &t.colon_token;
                    let bounds = &t.bounds;
                    let wc = &t.generics.where_clause;
                    let raw = quote::quote!(
                        pub trait #trait_ident #trait_gen { type #t_id #t_gen #colon #bounds #wc }
                    )
                    .to_string();
                    self.push_item(raw, &t.attrs, parent_dep.clone());
                }
                _ => {}
            }
        }
    }

    fn record_inherent_impl(&mut self, i: &syn::ItemImpl) {
        let impl_gen = &i.generics;
        let self_ty = &i.self_ty;
        let wc = &i.generics.where_clause;
        let parent_dep = extract_expiration_date(&i.attrs);

        for item in &i.items {
            match item {
                syn::ImplItem::Fn(m)
                    if matches!(m.vis, syn::Visibility::Public(_)) && !has_cfg_test(&m.attrs) =>
                {
                    let sig = &m.sig;
                    let raw = quote::quote!(impl #impl_gen #self_ty #wc { pub #sig }).to_string();
                    self.push_item(raw, &m.attrs, parent_dep.clone());
                }
                syn::ImplItem::Const(c)
                    if matches!(c.vis, syn::Visibility::Public(_)) && !has_cfg_test(&c.attrs) =>
                {
                    let c_id = &c.ident;
                    let c_gen = &c.generics;
                    let ty = &c.ty;
                    let expr = &c.expr;
                    let raw = quote::quote!(
                        impl #impl_gen #self_ty #wc { pub const #c_id #c_gen : #ty = #expr }
                    )
                    .to_string();
                    self.push_item(raw, &c.attrs, parent_dep.clone());
                }
                syn::ImplItem::Type(t)
                    if matches!(t.vis, syn::Visibility::Public(_)) && !has_cfg_test(&t.attrs) =>
                {
                    let t_id = &t.ident;
                    let t_gen = &t.generics;
                    let ty = &t.ty;
                    let t_wc = &t.generics.where_clause;
                    let raw = quote::quote!(
                        impl #impl_gen #self_ty #wc { pub type #t_id #t_gen #t_wc = #ty }
                    )
                    .to_string();
                    self.push_item(raw, &t.attrs, parent_dep.clone());
                }
                _ => {}
            }
        }
    }
}

impl<'ast> Visit<'ast> for ApiVisitor {
    fn visit_item_fn(&mut self, i: &'ast syn::ItemFn) {
        if matches!(i.vis, syn::Visibility::Public(_))
            && !has_cfg_test(&i.attrs)
            && self.active_allows(&i.sig.ident.to_string())
        {
            let sig = &i.sig;
            self.push_item(quote::quote!(pub #sig).to_string(), &i.attrs, None);
        }
    }

    fn visit_item_struct(&mut self, i: &'ast syn::ItemStruct) {
        if matches!(i.vis, syn::Visibility::Public(_))
            && !has_cfg_test(&i.attrs)
            && self.active_allows(&i.ident.to_string())
        {
            let ident = &i.ident;
            let generics = &i.generics;
            let wc = &i.generics.where_clause;
            self.push_item(
                quote::quote!(pub struct #ident #generics #wc).to_string(),
                &i.attrs,
                None,
            );
        }
    }

    fn visit_item_enum(&mut self, i: &'ast syn::ItemEnum) {
        if matches!(i.vis, syn::Visibility::Public(_))
            && !has_cfg_test(&i.attrs)
            && self.active_allows(&i.ident.to_string())
        {
            let ident = &i.ident;
            let generics = &i.generics;
            let wc = &i.generics.where_clause;
            self.push_item(
                quote::quote!(pub enum #ident #generics #wc).to_string(),
                &i.attrs,
                None,
            );
        }
    }

    fn visit_item_union(&mut self, i: &'ast syn::ItemUnion) {
        if matches!(i.vis, syn::Visibility::Public(_))
            && !has_cfg_test(&i.attrs)
            && self.active_allows(&i.ident.to_string())
        {
            let ident = &i.ident;
            let generics = &i.generics;
            let wc = &i.generics.where_clause;
            self.push_item(
                quote::quote!(pub union #ident #generics #wc).to_string(),
                &i.attrs,
                None,
            );
        }
    }

    fn visit_item_trait(&mut self, i: &'ast syn::ItemTrait) {
        if matches!(i.vis, syn::Visibility::Public(_))
            && !has_cfg_test(&i.attrs)
            && self.active_allows(&i.ident.to_string())
        {
            let unsafety = &i.unsafety;
            let ident = &i.ident;
            let generics = &i.generics;
            let colon = &i.colon_token;
            let supertraits = &i.supertraits;
            let wc = &i.generics.where_clause;
            self.push_item(
                quote::quote!(pub #unsafety trait #ident #generics #colon #supertraits #wc)
                    .to_string(),
                &i.attrs,
                None,
            );
            self.record_trait_items(i);
        }
    }

    fn visit_item_type(&mut self, i: &'ast syn::ItemType) {
        if matches!(i.vis, syn::Visibility::Public(_))
            && !has_cfg_test(&i.attrs)
            && self.active_allows(&i.ident.to_string())
        {
            let ident = &i.ident;
            let generics = &i.generics;
            let wc = &i.generics.where_clause;
            let ty = &i.ty;
            self.push_item(
                quote::quote!(pub type #ident #generics #wc = #ty).to_string(),
                &i.attrs,
                None,
            );
        }
    }

    fn visit_item_const(&mut self, i: &'ast syn::ItemConst) {
        if matches!(i.vis, syn::Visibility::Public(_))
            && !has_cfg_test(&i.attrs)
            && self.active_allows(&i.ident.to_string())
        {
            let ident = &i.ident;
            let generics = &i.generics;
            let ty = &i.ty;
            let expr = &i.expr;
            self.push_item(
                quote::quote!(pub const #ident #generics : #ty = #expr).to_string(),
                &i.attrs,
                None,
            );
        }
    }

    fn visit_item_static(&mut self, i: &'ast syn::ItemStatic) {
        if matches!(i.vis, syn::Visibility::Public(_))
            && !has_cfg_test(&i.attrs)
            && self.active_allows(&i.ident.to_string())
        {
            let mutability = &i.mutability;
            let ident = &i.ident;
            let ty = &i.ty;
            let expr = &i.expr;
            self.push_item(
                quote::quote!(pub static #mutability #ident : #ty = #expr).to_string(),
                &i.attrs,
                None,
            );
        }
    }

    fn visit_item_use(&mut self, i: &'ast syn::ItemUse) {
        if !matches!(i.vis, syn::Visibility::Public(_)) || has_cfg_test(&i.attrs) {
            return;
        }
        let active_filter = self.filter.clone().unwrap_or(ExportFilter::AllPublic);
        for leaf in flatten_use_tree(&i.tree) {
            if leaf_allowed_by_filter(&leaf, &active_filter) {
                self.push_item(leaf.format_signature(), &i.attrs, None);
            }
        }
    }

    fn visit_item_impl(&mut self, i: &'ast syn::ItemImpl) {
        if i.trait_.is_some() || has_cfg_test(&i.attrs) {
            return;
        }
        let allowed = match &self.filter {
            None | Some(ExportFilter::AllPublic) => true,
            Some(ExportFilter::Selected(set)) => impl_base_ident(&i.self_ty)
                .as_deref()
                .is_some_and(|name| set.contains(name)),
        };
        if allowed {
            self.record_inherent_impl(i);
        }
    }
}
