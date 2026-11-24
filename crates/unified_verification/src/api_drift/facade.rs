use super::ApiItem;
use super::traversal::{UseTarget, flatten_use_tree};
use oxidize_core::traceability::AstCache;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use syn::visit::Visit;

#[path = "facade_errors.rs"]
mod facade_errors;
#[path = "facade_signatures.rs"]
mod facade_signatures;

#[cfg(test)]
use facade_errors::DOMAIN_INNER_MODS;
use facade_errors::check_crate_error_reexports;
#[cfg(test)]
use facade_signatures::parse_signature_ast_counted;
use facade_signatures::{ParsedSignatureMap, build_parsed_signatures_counted};

const FACADE_REACHABLE_PREFIXES: &[&str] = &[
    "math_explorer/src/",
    "crates/00_foundation/math_scalars/src/",
    "crates/math_commons/src/",
    "crates/diagnostics/src/",
    "crates/pure_math/src/",
    "crates/oxidize_core/src/mesh/",
    "crates/oxidize_core/src/rng.rs",
    "crates/domain_ai/src/",
    "crates/domain_applied/src/",
    "crates/domain_biology/src/",
    "crates/domain_climate/src/",
    "crates/domain_epidemiology/src/",
    "crates/domain_physics/src/",
];

const REQUIRED_ROOT_EXPORTS: &[&str] = &[
    "ast",
    "constants",
    "diagnostics",
    "fit",
    "math_kernel",
    "math_scalars",
    "math_types",
    "mesh",
    "nalgebra",
    "num_complex",
    "petgraph",
    "poi",
    "primitives",
    "pure_math",
    "rand",
    "registry",
    "rng",
    "series",
    "wigner",
];

#[rustfmt::skip]
const STD_AND_REEXPORTED_TYPES: &[&str] = &[
    // Core / std / serde types & traits
    "Add", "AddAssign", "Any", "Arc", "AsMut", "AsRef", "BTreeMap", "BTreeSet", "BinaryHeap",
    "Borrow", "BorrowMut", "Box", "BufRead", "BufReader", "BufWriter", "Cell", "Clone", "Cmp",
    "Copy", "Cow", "Debug", "Default", "Deref", "DerefMut", "Deserialize", "Deserializer",
    "Display", "Div", "DivAssign", "DoubleEndedIterator", "Drop", "Duration", "Eq", "Error",
    "ExactSizeIterator", "Extend", "File", "Fn", "FnMut", "FnOnce", "Formatter", "From",
    "FromIterator", "FromStr", "Future", "Hash", "HashMap", "HashSet", "Hasher", "Index",
    "IndexMut", "Instant", "Into", "IntoIterator", "IoError", "Iter", "IterMut", "Iterator",
    "LowerExp", "Mul", "MulAssign", "Mutex", "Neg", "NonZeroU32", "NonZeroU64", "NonZeroUsize",
    "Not", "OnceLock", "Option", "Ord", "Ordering", "Output", "PartialEq", "PartialOrd", "Path",
    "PathBuf", "Peekable", "PhantomData", "Pin", "Product", "Range", "RangeBounds",
    "RangeInclusive", "Rc", "Read", "Ref", "RefCell", "RefMut", "Rem", "RemAssign", "Result",
    "RwLock", "Seek", "Self", "Send", "Serialize", "Serializer", "Sized", "SliceIndex", "String",
    "Sub", "SubAssign", "Sum", "Sync", "SystemTime", "ToOwned", "ToString", "TryFrom", "TryInto",
    "TypeId", "UpperExp", "Vec", "VecDeque", "Write",
    // Re-exported crate types (nalgebra, num_complex, petgraph, rand)
    "Allocator", "Cholesky", "Complex", "Complex32", "Complex64", "ComplexField", "Const",
    "CsMatrix", "DMatrix", "DMatrixSlice", "DMatrixView", "DMatrixViewMut", "DVector",
    "DVectorSlice", "DVectorView", "DVectorViewMut", "Data", "DefaultAllocator", "DiGraph", "Dim",
    "DimMin", "DimName", "Directed", "Dyn", "EdgeIndex", "EdgeRef", "Field", "Graph",
    "IntoEdgeReferences", "IntoEdges", "IntoNeighbors", "IntoNodeIdentifiers", "Isometry2",
    "Isometry3", "LU", "Matrix", "Matrix2", "Matrix3", "Matrix4", "Matrix6", "MatrixView",
    "MatrixViewMut", "NodeIndex", "OMatrix", "OVector", "Owned", "Point2", "Point3", "QR",
    "RealField", "Rng", "RngCore", "Rotation2", "Rotation3", "RowDVector", "RowSVector",
    "RowVector2", "RowVector3", "RowVector4", "SMatrix", "SVD", "SVector", "Scalar",
    "SeedableRng", "SimdComplexField", "SimdRealField", "Storage", "StorageMut", "StdRng",
    "SymmetricEigen", "Translation2", "Translation3", "U1", "U2", "U3", "U4", "U6", "UnGraph",
    "Undirected", "Unit", "UnitComplex", "UnitQuaternion", "UnitVector3", "Vector2", "Vector3",
    "Vector4", "Vector6", "VectorView", "VectorViewMut", "VisitMap", "Visitable",
];

fn is_facade_reachable_file(rel_path: &str) -> bool {
    FACADE_REACHABLE_PREFIXES
        .iter()
        .any(|prefix| rel_path.starts_with(prefix))
}

fn parse_exported_idents_in_file(path: &Path, cache: &mut AstCache) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    let Some(ast) = cache.get_or_load_fs(path).and_then(|e| e.ast.as_ref()) else {
        return out;
    };
    for item in &ast.items {
        match item {
            syn::Item::Enum(e) if matches!(e.vis, syn::Visibility::Public(_)) => {
                out.insert(e.ident.to_string());
            }
            syn::Item::Struct(s) if matches!(s.vis, syn::Visibility::Public(_)) => {
                out.insert(s.ident.to_string());
            }
            syn::Item::Trait(t) if matches!(t.vis, syn::Visibility::Public(_)) => {
                out.insert(t.ident.to_string());
            }
            syn::Item::Type(ty) if matches!(ty.vis, syn::Visibility::Public(_)) => {
                out.insert(ty.ident.to_string());
            }
            syn::Item::Mod(m) if matches!(m.vis, syn::Visibility::Public(_)) => {
                out.insert(m.ident.to_string());
            }
            syn::Item::Use(u) if matches!(u.vis, syn::Visibility::Public(_)) => {
                for leaf in flatten_use_tree(&u.tree) {
                    if let Some(id) = leaf.exported_ident() {
                        out.insert(id.to_string());
                    }
                }
            }
            _ => {}
        }
    }
    out
}

fn check_root_facade_modules(
    workspace_root: &Path,
    cache: &mut AstCache,
    violations: &mut Vec<String>,
) {
    let root_exports =
        parse_exported_idents_in_file(&workspace_root.join("math_explorer/src/lib.rs"), cache);
    for required in ["error", "prelude"] {
        if !root_exports.contains(required) {
            violations.push(format!(
                "Facade module gap: `math_explorer/src/lib.rs` must declare `pub mod {required};`"
            ));
        }
    }
    for required in REQUIRED_ROOT_EXPORTS {
        if !root_exports.contains(*required) {
            violations.push(format!(
                "Facade re-export gap: `math_explorer/src/lib.rs` must publicly re-export `{required}`"
            ));
        }
    }
}

fn collect_exported_type_names(
    apis: &BTreeMap<String, Vec<ApiItem>>,
    parsed_sigs: &ParsedSignatureMap<'_>,
) -> BTreeSet<String> {
    let mut exported: BTreeSet<String> = STD_AND_REEXPORTED_TYPES
        .iter()
        .map(|s| (*s).to_string())
        .collect();

    for (path, items) in apis {
        if !is_facade_reachable_file(path) {
            continue;
        }
        for api in items {
            if api.signature.starts_with("pub mod __doctest_facade") {
                continue;
            }
            if let Some(Some(file)) = parsed_sigs.get(api.signature.as_str()) {
                record_defined_or_reexported_ident(file, &mut exported);
            }
        }
    }
    exported
}

fn record_defined_or_reexported_ident(file: &syn::File, exported: &mut BTreeSet<String>) {
    for item in &file.items {
        match item {
            syn::Item::Struct(s) => {
                exported.insert(s.ident.to_string());
            }
            syn::Item::Enum(e) => {
                exported.insert(e.ident.to_string());
            }
            syn::Item::Union(u) => {
                exported.insert(u.ident.to_string());
            }
            syn::Item::Trait(t) => {
                exported.insert(t.ident.to_string());
            }
            syn::Item::Type(ty) => {
                exported.insert(ty.ident.to_string());
            }
            syn::Item::Use(u) => {
                for leaf in flatten_use_tree(&u.tree) {
                    match leaf.target {
                        UseTarget::Name(n) => {
                            exported.insert(n);
                        }
                        UseTarget::Rename { ident, alias } => {
                            exported.insert(ident);
                            exported.insert(alias);
                        }
                        UseTarget::Glob => {}
                    }
                }
            }
            _ => {}
        }
    }
}

#[derive(Default)]
struct SignatureTypeCollector {
    generics: BTreeSet<String>,
    referenced: BTreeSet<String>,
}

impl SignatureTypeCollector {
    fn add_generics(&mut self, generics: &syn::Generics) {
        for param in &generics.params {
            match param {
                syn::GenericParam::Type(tp) => {
                    self.generics.insert(tp.ident.to_string());
                }
                syn::GenericParam::Const(cp) => {
                    self.generics.insert(cp.ident.to_string());
                }
                syn::GenericParam::Lifetime(_) => {}
            }
        }
    }

    fn record_path(&mut self, path: &syn::Path) {
        if path.segments.len() >= 2
            && let Some(first) = path.segments.first()
            && (first.ident == "Self" || self.generics.contains(&first.ident.to_string()))
        {
            for seg in &path.segments {
                self.visit_path_arguments(&seg.arguments);
            }
            return;
        }
        if let Some(last) = path.segments.last() {
            let name = last.ident.to_string();
            if name.chars().next().is_some_and(char::is_uppercase)
                && !name
                    .chars()
                    .all(|c| c.is_ascii_uppercase() || c == '_' || c.is_ascii_digit())
                && !self.generics.contains(&name)
            {
                self.referenced.insert(name);
            }
        }
        for seg in &path.segments {
            self.visit_path_arguments(&seg.arguments);
        }
    }
}

impl<'ast> Visit<'ast> for SignatureTypeCollector {
    fn visit_item_struct(&mut self, i: &'ast syn::ItemStruct) {
        self.add_generics(&i.generics);
        syn::visit::visit_item_struct(self, i);
    }

    fn visit_item_enum(&mut self, i: &'ast syn::ItemEnum) {
        self.add_generics(&i.generics);
        syn::visit::visit_item_enum(self, i);
    }

    fn visit_item_union(&mut self, i: &'ast syn::ItemUnion) {
        self.add_generics(&i.generics);
        syn::visit::visit_item_union(self, i);
    }

    fn visit_item_fn(&mut self, i: &'ast syn::ItemFn) {
        self.add_generics(&i.sig.generics);
        syn::visit::visit_signature(self, &i.sig);
    }

    fn visit_item_impl(&mut self, i: &'ast syn::ItemImpl) {
        self.add_generics(&i.generics);
        for item in &i.items {
            if let syn::ImplItem::Fn(m) = item {
                self.add_generics(&m.sig.generics);
            } else if let syn::ImplItem::Type(t) = item {
                self.add_generics(&t.generics);
            }
        }
        syn::visit::visit_item_impl(self, i);
    }

    fn visit_item_trait(&mut self, i: &'ast syn::ItemTrait) {
        self.add_generics(&i.generics);
        for item in &i.items {
            if let syn::TraitItem::Fn(m) = item {
                self.add_generics(&m.sig.generics);
            } else if let syn::TraitItem::Type(t) = item {
                self.add_generics(&t.generics);
            }
        }
        syn::visit::visit_item_trait(self, i);
    }

    fn visit_item_type(&mut self, i: &'ast syn::ItemType) {
        self.add_generics(&i.generics);
        syn::visit::visit_item_type(self, i);
    }

    fn visit_type_path(&mut self, i: &'ast syn::TypePath) {
        if i.qself.is_some() {
            syn::visit::visit_type_path(self, i);
            return;
        }
        self.record_path(&i.path);
    }

    fn visit_trait_bound(&mut self, i: &'ast syn::TraitBound) {
        self.record_path(&i.path);
    }
}

fn collect_unexported_signature_types(
    file: &syn::File,
    exported: &BTreeSet<String>,
) -> Vec<String> {
    if file.items.iter().any(|it| matches!(it, syn::Item::Use(_))) {
        return Vec::new();
    }
    let mut collector = SignatureTypeCollector::default();
    collector.visit_file(file);
    collector
        .referenced
        .into_iter()
        .filter(|name| !exported.contains(name))
        .collect()
}

#[cfg(test)]
pub(crate) fn check_facade_completeness_in_dir(
    workspace_root: &Path,
    apis: &BTreeMap<String, Vec<ApiItem>>,
) -> Vec<String> {
    let mut cache = AstCache::new();
    check_facade_completeness_in_dir_with_cache(workspace_root, apis, &mut cache)
}

pub(crate) fn check_facade_completeness_in_dir_with_cache(
    workspace_root: &Path,
    apis: &BTreeMap<String, Vec<ApiItem>>,
    cache: &mut AstCache,
) -> Vec<String> {
    if !workspace_root.join("math_explorer/src/lib.rs").is_file() {
        return Vec::new();
    }

    let (parsed_sigs, _) = build_parsed_signatures_counted(apis);
    let mut violations = Vec::new();
    check_root_facade_modules(workspace_root, cache, &mut violations);
    check_crate_error_reexports(workspace_root, apis, &parsed_sigs, cache, &mut violations);

    let exported = collect_exported_type_names(apis, &parsed_sigs);
    for (path, items) in apis {
        if !is_facade_reachable_file(path) {
            continue;
        }
        for item in items {
            if item.signature.starts_with("pub mod __doctest_facade") {
                continue;
            }
            let Some(Some(file)) = parsed_sigs.get(item.signature.as_str()) else {
                continue;
            };
            for missing in collect_unexported_signature_types(file, &exported) {
                violations.push(format!(
                    "Un-reexported public signature type `{missing}` in `{path}`: `{}`",
                    item.signature
                ));
            }
        }
    }
    violations.sort();
    violations.dedup();
    violations
}

#[cfg(test)]
#[path = "facade_tests.rs"]
mod tests;
