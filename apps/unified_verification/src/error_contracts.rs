use crate::api_drift::collect_crate_roots;
use crate::api_drift::traversal::{ExportFilter, child_module_dir_for_file};
use crate::ast_helpers::format_file_line;
use oxidize_core::traceability::AstCache;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

mod visitor;

#[cfg(test)]
mod tests;

use visitor::{
    CrateContractState, FnOwner, ModuleContractAnalyzer, has_matching_companion, scan_module_items,
};

const MAX_MODULES_PER_CRATE: usize = 10_000;

const INTERNAL_HARNESS_CRATES: &[&str] = &["markdown_tests", "scientific_verification"];

const COMPANION_EXEMPT_CRATES: &[&str] = &["verified_engine", "verified_engine_macros"];

pub(super) const TRAIT_CONTRACT_ALLOWLIST: &[&str] = &[
    "AbelianGroup",
    "Add",
    "AddAssign",
    "AddRowVector",
    "AdvectedSpatialDiffusion",
    "BitAnd",
    "BitOr",
    "BitXor",
    "BlockRngCore",
    "BoundaryStrategy",
    "Clone",
    "CommutativeRing",
    "Continuous",
    "ContinuousCDF",
    "Debug",
    "Default",
    "Deref",
    "DerefMut",
    "Deserialize",
    "Diagnostic",
    "DiagnosticVisitor",
    "DiffusionModel",
    "Discrete",
    "DiscreteCDF",
    "Display",
    "Distribution",
    "Div",
    "DivAssign",
    "DoubleEndedIterator",
    "ElectrochemicalAgingModel",
    "Eq",
    "Error",
    "Euclid",
    "EuclideanDomain",
    "ExactSizeIterator",
    "Field",
    "Float",
    "From",
    "FromIterator",
    "FromPrimitive",
    "GenericOptimizer",
    "Group",
    "Hash",
    "Index",
    "IndexMut",
    "InnerProductSpace",
    "IntegralDomain",
    "Into",
    "IntoIterator",
    "Inv",
    "Iterator",
    "LangevinSystem",
    "MetricSpace",
    "Monoid",
    "Mul",
    "MulAdd",
    "MulAddAssign",
    "MulAssign",
    "Neg",
    "NormedSpace",
    "Not",
    "Num",
    "NumCast",
    "OdeSystem",
    "One",
    "Ord",
    "PairingStrategy",
    "PartialEq",
    "PartialOrd",
    "PointSeries",
    "Pow",
    "ReactionKinetics",
    "Rem",
    "RemAssign",
    "Ring",
    "RngCore",
    "Scalar",
    "SdeSystem",
    "SeedableRng",
    "Semigroup",
    "Serialize",
    "Shl",
    "Shr",
    "Signed",
    "SpatialDiffusion",
    "Sub",
    "SubAssign",
    "TheoryDescribable",
    "ToPrimitive",
    "TryFrom",
    "TryInto",
    "Unsigned",
    "VectorSpace",
    "VirtualFileSystem",
    "Zero",
];

fn crate_name_from_lib_path(crate_lib: &Path) -> String {
    let Some(crate_dir) = crate_lib.parent().and_then(Path::parent) else {
        return String::new();
    };
    if let Ok(raw) = std::fs::read_to_string(crate_dir.join("Cargo.toml"))
        && let Some(name) = oxidize_core::workspace::parse_package_name_std(&raw)
    {
        return name;
    }
    crate_dir
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default()
}

fn evaluate_companion_records(state: &CrateContractState, violations: &mut Vec<String>) {
    for rec in &state.panicking_fns {
        let has_companion = match &rec.owner {
            FnOwner::Free => has_matching_companion(&rec.name, &state.free_fns),
            FnOwner::Inherent(self_ty) | FnOwner::Trait { self_ty, .. } => state
                .inherent_methods
                .get(self_ty)
                .is_some_and(|methods| has_matching_companion(&rec.name, methods)),
        };
        if has_companion {
            continue;
        }
        let loc = format_file_line(&rec.rel_path, rec.line);
        let label = match &rec.owner {
            FnOwner::Free => format!("pub fn `{}`", rec.name),
            FnOwner::Inherent(ty) => format!("pub fn `{ty}::{}`", rec.name),
            FnOwner::Trait {
                trait_name,
                self_ty,
            } => format!("impl `{trait_name}` for `{self_ty}` fn `{}`", rec.name),
        };
        violations.push(format!(
            "[!] Uncompanioned panicking public library function at {loc}: {label} (expose `try_{}` or `checked_{}` returning typed `Result`)",
            rec.name, rec.name
        ));
    }
}

fn check_crate_contracts(
    crate_lib: &Path,
    workspace_root: &Path,
    cache: &mut AstCache,
    violations: &mut Vec<String>,
) {
    if !crate_lib.is_file() {
        return;
    }
    let crate_name = crate_name_from_lib_path(crate_lib);
    if INTERNAL_HARNESS_CRATES.contains(&crate_name.as_str()) {
        return;
    }
    let check_companions = !COMPANION_EXEMPT_CRATES.contains(&crate_name.as_str());
    let mut state = CrateContractState::default();
    let mut visited: BTreeMap<PathBuf, ExportFilter> = BTreeMap::new();
    let mut worklist = vec![crate_lib.to_path_buf()];
    visited.insert(crate_lib.to_path_buf(), ExportFilter::AllPublic);

    let mut steps = 0usize;
    while let Some(file_path) = worklist.pop() {
        if steps >= MAX_MODULES_PER_CRATE {
            break;
        }
        steps += 1;
        let Some(active_filter) = visited.get(&file_path).cloned() else {
            continue;
        };
        let mod_dir = child_module_dir_for_file(&file_path);
        let rel = file_path
            .strip_prefix(workspace_root)
            .unwrap_or(&file_path)
            .to_string_lossy()
            .replace('\\', "/");

        let child_files = {
            let Some(ast) = cache
                .get_or_load_fs(&file_path)
                .and_then(|entry| entry.ast.as_ref())
            else {
                continue;
            };
            let mut analyzer = ModuleContractAnalyzer {
                rel_path: &rel,
                check_companions,
                state: &mut state,
                violations,
            };
            scan_module_items(&ast.items, &active_filter, &mod_dir, &mut analyzer)
        };

        for (child_path, child_filter) in child_files {
            let should_enqueue = if let Some(prev) = visited.get_mut(&child_path) {
                prev.merge(child_filter)
            } else {
                visited.insert(child_path.clone(), child_filter);
                true
            };
            if should_enqueue {
                worklist.push(child_path);
            }
        }
    }

    if check_companions {
        evaluate_companion_records(&state, violations);
    }
}

pub(crate) fn collect_error_contract_violations_in_dir(
    workspace_root: &Path,
    cache: &mut AstCache,
) -> Vec<String> {
    let mut violations = Vec::new();
    for crate_lib in collect_crate_roots(workspace_root) {
        check_crate_contracts(&crate_lib, workspace_root, cache, &mut violations);
    }
    violations.sort();
    violations.dedup();
    violations
}

pub(crate) fn check_error_contracts_with_cache(cache: &mut AstCache) -> bool {
    let root = crate::manifest_cache::workspace_root_dir();
    let violations = collect_error_contract_violations_in_dir(&root, cache);
    for msg in &violations {
        println!("{msg}");
    }
    violations.is_empty()
}
