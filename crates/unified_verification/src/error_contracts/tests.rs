use super::collect_error_contract_violations_in_dir;
use oxidize_core::traceability::AstCache;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_ID: AtomicU64 = AtomicU64::new(0);

struct TempWorkspace {
    path: PathBuf,
}

impl TempWorkspace {
    fn new(label: &str) -> Self {
        let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "oxidize_math_error_contracts_{}_{}_{}",
            std::process::id(),
            label,
            id
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("create temp workspace");
        Self { path: dir }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempWorkspace {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn write_crate_lib(root: &Path, crate_name: &str, code: &str) {
    let src_dir = root.join("crates").join(crate_name).join("src");
    fs::create_dir_all(&src_dir).expect("create temp crate src");
    fs::write(src_dir.join("lib.rs"), code).expect("write temp lib.rs");
}

#[test]
fn rejects_untyped_result_signatures_string_str_and_unit() {
    let tmp = TempWorkspace::new("untyped_result");
    write_crate_lib(
        tmp.path(),
        "sample_domain",
        r#"
pub fn bad_string(x: f64) -> Result<f64, String> {
    Ok(x)
}

pub fn bad_static_str(x: f64) -> core::result::Result<f64, &'static str> {
    Ok(x)
}

pub struct Grid;
impl Grid {
    pub fn bad_unit(&self) -> Result<(), ()> {
        Ok(())
    }
}
"#,
    );

    let mut cache = AstCache::new();
    let violations = collect_error_contract_violations_in_dir(tmp.path(), &mut cache);
    assert_eq!(violations.len(), 3, "unexpected violations: {violations:?}");
    assert!(violations.iter().any(|v| v.contains("bad_string")));
    assert!(violations.iter().any(|v| v.contains("bad_static_str")));
    assert!(violations.iter().any(|v| v.contains("Grid::bad_unit")));
}

#[test]
fn rejects_uncompanioned_panicking_public_functions() {
    let tmp = TempWorkspace::new("uncompanioned");
    write_crate_lib(
        tmp.path(),
        "sample_engine",
        r#"
pub fn bare_step(dt: f64) -> f64 {
    assert!(dt > 0.0);
    dt
}

pub struct Stepper;
impl Stepper {
    pub fn try_advance(&self, n: Option<usize>) -> Option<usize> {
        n
    }

    pub fn advance(&self, n: Option<usize>) -> usize {
        n.unwrap()
    }
}
"#,
    );

    let mut cache = AstCache::new();
    let violations = collect_error_contract_violations_in_dir(tmp.path(), &mut cache);
    assert_eq!(violations.len(), 2, "unexpected violations: {violations:?}");
    assert!(violations.iter().any(|v| v.contains("bare_step")));
    assert!(violations.iter().any(|v| v.contains("Stepper::advance")));
}

#[test]
fn accepts_try_and_checked_companions_and_allowlisted_traits() {
    let tmp = TempWorkspace::new("companioned");
    write_crate_lib(
        tmp.path(),
        "sample_clean",
        r#"
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleError {
    NonPositive,
    DivisionByZero,
}

pub fn try_step(dt: f64) -> Result<f64, SampleError> {
    if dt <= 0.0 {
        return Err(SampleError::NonPositive);
    }
    Ok(dt)
}

pub fn step(dt: f64) -> f64 {
    try_step(dt).expect("valid dt")
}

#[derive(Clone, Copy)]
pub struct ScalarVal(pub f64);

impl ScalarVal {
    pub fn checked_div(self, rhs: Self) -> Result<Self, SampleError> {
        if rhs.0 == 0.0 {
            return Err(SampleError::DivisionByZero);
        }
        Ok(Self(self.0 / rhs.0))
    }
}

impl core::ops::Div for ScalarVal {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        self.checked_div(rhs).expect("non-zero divisor")
    }
}
"#,
    );

    let mut cache = AstCache::new();
    let violations = collect_error_contract_violations_in_dir(tmp.path(), &mut cache);
    assert!(
        violations.is_empty(),
        "expected zero violations, got: {violations:?}"
    );
}
