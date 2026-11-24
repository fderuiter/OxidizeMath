use super::*;
use crate::utils::test_utils::TempTestDir;
use std::fs;

fn setup_math_fixture(prefix: &str) -> TempTestDir {
    let dir = TempTestDir::new(prefix);
    fs::create_dir_all(dir.join("papers")).unwrap();
    fs::create_dir_all(dir.join("docs/math_specs")).unwrap();
    fs::write(
        dir.join("Cargo.toml"),
        "[workspace]\nmembers = [\"crates/pure_math\"]\n",
    )
    .unwrap();
    fs::write(
        dir.join("papers/references.bib"),
        concat!(
            "@article{dormand1980family,\n",
            "  title = {A family of embedded Runge-Kutta formulae},\n",
            "  author = {Dormand, J. R. and Prince, P. J.},\n",
            "  year = {1980},\n",
            "  doi = {10.1016/0771-050X(80)90013-3}\n",
            "}\n",
            "@article{bluestein1970linear,\n",
            "  title = {A linear filtering approach to the computation of discrete Fourier transform},\n",
            "  author = {Bluestein, Leo},\n",
            "  year = {1970},\n",
            "  eprint = {1001.0001}\n",
            "}\n",
            "@article{missingid2020,\n",
            "  title = {Unverified Reference},\n",
            "  author = {Doe, Jane},\n",
            "  year = {2020}\n",
            "}\n",
        ),
    )
    .unwrap();
    fs::write(
        dir.join("docs/math_specs/pure_math_v1.md"),
        concat!(
            "# Pure Math Spec (`spec:pure_math_v1`)\n\n",
            "$$y'(t) = f(t, y)$$\n\n",
            "Numerical stability: RK45 truncation error bound.\n",
            "Domain of validity: Lipschitz right-hand side.\n",
            "References: `@dormand1980family`.\n",
        ),
    )
    .unwrap();
    fs::write(dir.join("papers/bluestein_czt.tex"), "% manuscript\n").unwrap();
    fs::create_dir_all(dir.join("crates/pure_math/src")).unwrap();
    fs::write(
        dir.join("crates/pure_math/Cargo.toml"),
        "[package]\nname = \"pure_math\"\nversion = \"0.1.0\"\nedition = \"2024\"\n\n[lints]\nworkspace = true\n",
    )
    .unwrap();
    dir
}

#[test]
fn test_compliant_items_and_embed_theory_pass_with_zero_cap() {
    let dir = setup_math_fixture("math_doc_contract_ok");
    let whitelist_path = dir.join("verification_whitelist.toml");
    fs::write(&whitelist_path, "").unwrap();
    fs::write(
        dir.join("crates/pure_math/src/lib.rs"),
        concat!(
            "/// Integrator state $y'(t) = f(t, y)$ (Ref: `@dormand1980family`, `spec:pure_math_v1`).\n",
            "/// Numerical stability: fourth-order local truncation error bound $O(h^5)$.\n",
            "/// Domain of validity: Lipschitz-continuous right-hand side for finite $h > 0$.\n",
            "#[derive(Theory)]\n",
            "#[theory(citation = \"dormand1980family\")]\n",
            "pub struct OdeState;\n\n",
            "/// Evaluates Chirp-Z transform (Ref: `@bluestein1970linear`).\n",
            "/// Numerical stability: FFT convolution conditioning bounded by unit-circle contour.\n",
            "/// Domain of validity: non-empty finite input slice.\n",
            "#[embed_theory(\"papers/bluestein_czt.tex\", \"eq:czt\")]\n",
            "pub fn evaluate_czt() {}\n",
        ),
    )
    .unwrap();

    let wl = crate::whitelist::load_whitelist_strict(Some(&whitelist_path), &[] as &[&Path])
        .expect("valid whitelist");
    let mut cache = AstCache::new();
    let member = lossy_slash_path(&dir.join("crates/pure_math"));
    assert!(audit_members_in_root(
        &dir,
        &[member.as_str()],
        &wl,
        &mut cache
    ));
    assert_eq!(cache.parse_calls(), 1);
}

#[test]
fn test_missing_contract_parts_and_missing_doi_bibtex_fail() {
    let dir = setup_math_fixture("math_doc_contract_missing_parts");
    let bib = load_workspace_bib(&dir).unwrap();

    let src = concat!(
        "/// Missing equation, citation, stability, and validity.\n",
        "pub fn incomplete() {}\n\n",
        "/// Uses $$y = x^2$$ (Ref: `@missingid2020`, `spec:missing_spec`).\n",
        "/// Numerical stability: exact for small floats.\n",
        "/// Domain of validity: finite $x$.\n",
        "pub fn bad_refs() {}\n",
    );
    let ast = syn::parse_file(src).unwrap();
    let violations = collect_file_violations(&ast, &dir, &bib);
    assert_eq!(violations.len(), 2);
    assert_eq!(violations[0].line, 2);
    assert_eq!(violations[0].missing_parts.len(), 5);
    assert_eq!(violations[1].line, 7);
    assert!(
        violations[1]
            .missing_parts
            .iter()
            .any(|m| m.contains("missingid2020") && m.contains("doi"))
    );
    assert!(
        violations[1]
            .missing_parts
            .iter()
            .any(|m| m.contains("docs/math_specs/missing_spec.md"))
    );
}

#[test]
fn test_ratcheted_caps_enforce_single_ast_parse_and_over_cap_rejection() {
    let dir = setup_math_fixture("math_doc_contract_ratchet");
    let whitelist_path = dir.join("verification_whitelist.toml");
    let lib_rs = dir.join("crates/pure_math/src/lib.rs");
    fs::write(&lib_rs, "pub fn legacy_a() {}\npub fn legacy_b() {}\n").unwrap();

    fs::write(
        &whitelist_path,
        "[math_doc_contract.files]\n\"crates/pure_math/src/lib.rs\" = 2\n",
    )
    .unwrap();
    let wl = crate::whitelist::load_whitelist_strict(Some(&whitelist_path), &[] as &[&Path])
        .expect("valid whitelist");
    let mut cache = AstCache::new();
    let member = lossy_slash_path(&dir.join("crates/pure_math"));
    assert!(audit_members_in_root(
        &dir,
        &[member.as_str()],
        &wl,
        &mut cache
    ));
    assert_eq!(cache.parse_calls(), 1);

    assert!(audit_members_in_root(
        &dir,
        &[member.as_str()],
        &wl,
        &mut cache
    ));
    assert_eq!(cache.parse_calls(), 1);

    let mut cache_over = AstCache::new();
    fs::write(
        &lib_rs,
        "pub fn legacy_a() {}\npub fn legacy_b() {}\npub fn legacy_c() {}\n",
    )
    .unwrap();
    assert!(!audit_members_in_root(
        &dir,
        &[member.as_str()],
        &wl,
        &mut cache_over
    ));
}

#[test]
fn test_ratcheted_caps_reject_stale_under_cap_and_non_math_entries() {
    let dir = setup_math_fixture("math_doc_contract_stale_ratchet");
    let whitelist_path = dir.join("verification_whitelist.toml");
    let lib_rs = dir.join("crates/pure_math/src/lib.rs");
    fs::write(
        &whitelist_path,
        "[math_doc_contract.files]\n\"crates/pure_math/src/lib.rs\" = 2\n",
    )
    .unwrap();
    let wl = crate::whitelist::load_whitelist_strict(Some(&whitelist_path), &[] as &[&Path])
        .expect("valid whitelist");
    let member = lossy_slash_path(&dir.join("crates/pure_math"));

    let mut cache_under = AstCache::new();
    fs::write(
        &lib_rs,
        concat!(
            "/// Solves $$y' = f(t, y)$$ (Ref: `@dormand1980family`, `spec:pure_math_v1`).\n",
            "/// Numerical stability: bounded truncation error.\n",
            "/// Domain of validity: finite step size $h > 0$.\n",
            "pub fn documented_a() {}\n",
            "pub fn legacy_b() {}\n",
        ),
    )
    .unwrap();
    assert!(!audit_members_in_root(
        &dir,
        &[member.as_str()],
        &wl,
        &mut cache_under
    ));

    fs::write(
        &whitelist_path,
        "[math_doc_contract.files]\n\"crates/pure_math/src/deleted.rs\" = 1\n",
    )
    .unwrap();
    let stale_wl =
        crate::whitelist::load_whitelist_strict(Some(&whitelist_path), &[] as &[&Path]).unwrap();
    assert!(!audit_members_in_root(
        &dir,
        &[member.as_str()],
        &stale_wl,
        &mut cache_under
    ));

    fs::write(
        &whitelist_path,
        "[math_doc_contract.files]\n\"crates/oxidize_core/src/lib.rs\" = 1\n",
    )
    .unwrap();
    let non_math_wl =
        crate::whitelist::load_whitelist_strict(Some(&whitelist_path), &[] as &[&Path]).unwrap();
    assert!(!audit_members_in_root(
        &dir,
        &[],
        &non_math_wl,
        &mut cache_under
    ));
}
