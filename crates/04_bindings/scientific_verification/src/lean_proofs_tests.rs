use super::{verify_lean_proofs, verify_lean_proofs_in_workspace};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

static COUNTER: AtomicUsize = AtomicUsize::new(0);

struct TempProofWorkspace {
    root: PathBuf,
}

impl TempProofWorkspace {
    fn new(label: &str, lean_body: &str, rust_theorems: &str) -> Self {
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "oxidize_lean_gate_{label}_{}_{id}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("proofs/OxidizeMath")).unwrap();
        fs::create_dir_all(root.join("crates/demo/src")).unwrap();

        let ws_root = super::host::resolve_workspace_root();
        let toolchain = fs::read_to_string(ws_root.join("proofs/lean-toolchain")).unwrap();
        fs::write(root.join("proofs/lean-toolchain"), toolchain).unwrap();
        fs::write(
            root.join("proofs/lakefile.lean"),
            "import Lake\nopen Lake DSL\npackage «demo_proofs» where\n@[default_target]\nlean_lib «OxidizeMath» where\n",
        )
        .unwrap();
        fs::write(
            root.join("proofs/OxidizeMath.lean"),
            "import OxidizeMath.Demo\n",
        )
        .unwrap();
        fs::write(root.join("proofs/OxidizeMath/Demo.lean"), lean_body).unwrap();

        fs::write(
            root.join("Cargo.toml"),
            "[workspace]\nmembers = [\"crates/demo\"]\n",
        )
        .unwrap();
        fs::write(
            root.join("crates/demo/Cargo.toml"),
            "[package]\nname = \"demo\"\nversion = \"0.1.0\"\nedition = \"2024\"\n",
        )
        .unwrap();
        fs::write(
            root.join("crates/demo/src/lib.rs"),
            format!(
                "formal_proof_verification!(\n    module = \"demo\",\n    proof = \"Demo.lean\",\n    theorems = [{rust_theorems}],\n    test = {{ assert_eq!(1 + 1, 2); }}\n);\n"
            ),
        )
        .unwrap();

        Self { root }
    }

    fn path(&self) -> &Path {
        &self.root
    }
}

impl Drop for TempProofWorkspace {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

#[test]
fn workspace_lean_proofs_compile_with_zero_banned_axioms_and_valid_bindings() {
    let summary = verify_lean_proofs().expect("workspace Lean 4.14.0 proofs must kernel-verify");
    assert_eq!(summary.digests.len(), 6);
    assert_eq!(summary.bindings.len(), 6);
    for (proof, binding) in &summary.bindings {
        assert!(binding.kernel_verified, "{proof} must be kernel_verified");
        assert!(
            !binding.rust_modules.is_empty(),
            "{proof} must bind at least one Rust module"
        );
        assert!(
            !binding.theorems.is_empty(),
            "{proof} must bind at least one theorem"
        );
    }
}

#[test]
fn verify_lean_proofs_fails_closed_when_lake_binary_is_missing() {
    let ws = TempProofWorkspace::new(
        "missing_lake",
        "namespace OxidizeMath.Demo\ntheorem ok (x : Int) : x = x := rfl\nend OxidizeMath.Demo\n",
        "\"ok\"",
    );
    let err = verify_lean_proofs_in_workspace(ws.path(), "__missing_lake_binary_oxidize__", 1)
        .expect_err("missing lake binary must fail closed");
    assert!(err.contains("lake"), "unexpected error: {err}");
}

#[test]
fn verify_lean_proofs_fails_closed_on_build_errors_and_banned_axioms_or_attrs() {
    let broken_tactic = TempProofWorkspace::new(
        "broken_build",
        "namespace OxidizeMath.Demo\ntheorem bad (x : Int) : x = x + 1 := by omega\nend OxidizeMath.Demo\n",
        "\"bad\"",
    );
    assert!(verify_lean_proofs_in_workspace(broken_tactic.path(), "lake", 1).is_err());

    for (label, snippet) in [
        (
            "sorry_kw",
            "namespace OxidizeMath.Demo\ntheorem bad (x : Int) : x = x + 1 := by sorry\nend OxidizeMath.Demo\n",
        ),
        (
            "opaque_decl",
            "namespace OxidizeMath.Demo\nopaque secret : Int\ntheorem ok (x : Int) : x = x := rfl\nend OxidizeMath.Demo\n",
        ),
        (
            "extern_attr",
            "namespace OxidizeMath.Demo\n@[extern \"c_fn\"] def extFn (x : UInt32) : UInt32 := x\ntheorem ok (x : Int) : x = x := rfl\nend OxidizeMath.Demo\n",
        ),
        (
            "impl_by_attr",
            "namespace OxidizeMath.Demo\ndef fast (x : Int) : Int := x + 1\n@[implemented_by fast] def spec (x : Int) : Int := x\ntheorem ok (x : Int) : x = x := rfl\nend OxidizeMath.Demo\n",
        ),
    ] {
        let ws = TempProofWorkspace::new(label, snippet, "\"ok\"");
        let res = verify_lean_proofs_in_workspace(ws.path(), "lake", 1);
        assert!(res.is_err(), "{label} must fail closed, got {res:?}");
    }

    let bad_binding = TempProofWorkspace::new(
        "unknown_thm",
        "namespace OxidizeMath.Demo\ntheorem real_thm (x : Int) : x = x := rfl\nend OxidizeMath.Demo\n",
        "\"nonexistent_thm\"",
    );
    let err = verify_lean_proofs_in_workspace(bad_binding.path(), "lake", 1)
        .expect_err("unknown bound theorem must fail closed");
    assert!(err.contains("nonexistent_thm"), "unexpected error: {err}");
}
