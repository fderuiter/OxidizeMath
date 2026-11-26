use super::run_policy_audit_with_whitelist;
use super::tests::write_member_crate;
use crate::utils::test_utils::TempTestDir;
use std::fs;

#[test]
fn test_missing_docs_rejects_inner_allow_and_nested_cfg_attr() {
    let dir = TempTestDir::new("policy_audit_missing_docs_inner");
    let whitelist_file = dir.join("verification_whitelist.toml");

    // Inner #![allow(missing_docs)] fails even when a per-file cap is present
    let inner_allow = write_member_crate(
        &dir,
        "crates/inner_allow",
        "#![allow(missing_docs)]\npub struct Item;\n",
    );
    fs::write(
        &whitelist_file,
        "[missing_docs.files]\n\"crates/inner_allow/src/lib.rs\" = 1\n",
    )
    .unwrap();
    assert!(!run_policy_audit_with_whitelist(
        &[inner_allow.as_str()],
        &whitelist_file
    ));

    // Nested cfg_attr + rust::missing_docs inner bypass also fails
    let nested_inner = write_member_crate(
        &dir,
        "crates/nested_inner",
        "#![cfg_attr(not(test), expect(rust::missing_docs))]\npub struct Item;\n",
    );
    fs::write(&whitelist_file, "").unwrap();
    assert!(!run_policy_audit_with_whitelist(
        &[nested_inner.as_str()],
        &whitelist_file
    ));
}

#[test]
fn test_missing_docs_rejects_item_level_suppressions_in_library_crates() {
    let dir = TempTestDir::new("policy_audit_missing_docs_items");
    let whitelist_file = dir.join("verification_whitelist.toml");

    for (name, snippet) in [
        ("item_struct", "#[allow(missing_docs)]\npub struct S;\n"),
        ("item_enum", "#[allow(missing_docs)]\npub enum E { A }\n"),
        (
            "item_variant",
            "/// Doc.\npub enum E { #[allow(missing_docs)] A }\n",
        ),
        ("item_trait", "#[expect(missing_docs)]\npub trait T {}\n"),
        (
            "item_field",
            "/// Doc.\npub struct S { #[allow(missing_docs)] pub x: f64 }\n",
        ),
        ("item_fn", "#[allow(rust::missing_docs)]\npub fn f() {}\n"),
        (
            "impl_fn",
            "/// Doc.\npub struct S;\nimpl S { #[allow(missing_docs)] pub fn m(&self) {} }\n",
        ),
        (
            "macro_fn",
            "macro_rules! gen_fn { () => { #[allow(missing_docs)] pub fn f() {} }; }\n",
        ),
    ] {
        let rel = format!("crates/{name}");
        let member = write_member_crate(&dir, &rel, snippet);
        fs::write(
            &whitelist_file,
            format!("[missing_docs.files]\n\"{rel}/src/lib.rs\" = 1\n"),
        )
        .unwrap();
        assert!(
            !run_policy_audit_with_whitelist(&[member.as_str()], &whitelist_file),
            "Library crate item-level missing_docs bypass `{name}` must fail even when capped"
        );
    }

    // Binary-only crate (src/main.rs without src/lib.rs) remains governed by exact cap
    let bin_dir = dir.join("math_explorer_gui/src");
    fs::create_dir_all(&bin_dir).unwrap();
    fs::write(
        dir.join("math_explorer_gui/Cargo.toml"),
        "[package]\nname = \"math_explorer_gui\"\nversion = \"0.1.0\"\n\n[lints]\nworkspace = true\n",
    )
    .unwrap();
    fs::write(
        bin_dir.join("main.rs"),
        "#[allow(missing_docs)]\npub struct GuiState;\nfn main() {}\n",
    )
    .unwrap();
    fs::write(
        &whitelist_file,
        "[missing_docs.files]\n\"math_explorer_gui/src/main.rs\" = 1\n",
    )
    .unwrap();
    let bin_member = crate::whitelist::lossy_slash_path(&dir.join("math_explorer_gui"));
    assert!(run_policy_audit_with_whitelist(
        &[bin_member.as_str()],
        &whitelist_file
    ));
}

#[test]
fn test_missing_docs_rejects_outer_and_expect_above_cap() {
    let dir = TempTestDir::new("policy_audit_missing_docs_caps");
    let whitelist_file = dir.join("verification_whitelist.toml");
    fs::write(&whitelist_file, "").unwrap();

    // Unwhitelisted outer #[allow(missing_docs)] and #[expect(missing_docs)] fail
    let outer_allow = write_member_crate(
        &dir,
        "crates/outer_allow",
        "#[allow(missing_docs)]\npub mod a {}\n",
    );
    assert!(!run_policy_audit_with_whitelist(
        &[outer_allow.as_str()],
        &whitelist_file
    ));

    let outer_expect = write_member_crate(
        &dir,
        "crates/outer_expect",
        "#[expect(missing_docs, reason = \"legacy\")]\npub mod a {}\n",
    );
    assert!(!run_policy_audit_with_whitelist(
        &[outer_expect.as_str()],
        &whitelist_file
    ));

    // Exact cap on module-level attributes passes; adding one more above cap fails
    let capped = write_member_crate(
        &dir,
        "crates/capped",
        "#[allow(missing_docs)]\npub mod a {}\n#[allow(rust::missing_docs)]\npub mod b {}\n",
    );
    fs::write(
        &whitelist_file,
        "[missing_docs.files]\n\"crates/capped/src/lib.rs\" = 2\n",
    )
    .unwrap();
    assert!(run_policy_audit_with_whitelist(
        &[capped.as_str()],
        &whitelist_file
    ));

    fs::write(
        dir.join("crates/capped/src/lib.rs"),
        "#[allow(missing_docs)]\npub mod a {}\n#[allow(missing_docs)]\npub mod b {}\n#[expect(missing_docs)]\npub mod c {}\n",
    )
    .unwrap();
    assert!(!run_policy_audit_with_whitelist(
        &[capped.as_str()],
        &whitelist_file
    ));
}

#[test]
fn test_missing_docs_requires_downward_ratchet_and_rejects_stale_files() {
    let dir = TempTestDir::new("policy_audit_missing_docs_ratchet");
    let whitelist_file = dir.join("verification_whitelist.toml");

    let member = write_member_crate(
        &dir,
        "crates/ratchet",
        "#[allow(missing_docs)]\npub mod a {}\n#[allow(missing_docs)]\npub mod b {}\n",
    );
    fs::write(
        &whitelist_file,
        "[missing_docs.files]\n\"crates/ratchet/src/lib.rs\" = 2\n",
    )
    .unwrap();
    assert!(run_policy_audit_with_whitelist(
        &[member.as_str()],
        &whitelist_file
    ));

    // Reducing actual count from 2 to 1 without lowering cap fails downward ratchet
    fs::write(
        dir.join("crates/ratchet/src/lib.rs"),
        "/// Documented a.\npub mod a {}\n#[allow(missing_docs)]\npub mod b {}\n",
    )
    .unwrap();
    assert!(!run_policy_audit_with_whitelist(
        &[member.as_str()],
        &whitelist_file
    ));

    // Lowering the cap to 1 passes again
    fs::write(
        &whitelist_file,
        "[missing_docs.files]\n\"crates/ratchet/src/lib.rs\" = 1\n",
    )
    .unwrap();
    assert!(run_policy_audit_with_whitelist(
        &[member.as_str()],
        &whitelist_file
    ));

    // Reducing actual count to 0 without removing entry fails downward ratchet
    fs::write(
        dir.join("crates/ratchet/src/lib.rs"),
        "/// Documented a.\npub mod a {}\n/// Documented b.\npub mod b {}\n",
    )
    .unwrap();
    assert!(!run_policy_audit_with_whitelist(
        &[member.as_str()],
        &whitelist_file
    ));

    // Stale entry pointing to a deleted submodule inside the audited crate fails
    fs::write(
        &whitelist_file,
        "[missing_docs.files]\n\"crates/ratchet/src/deleted.rs\" = 1\n",
    )
    .unwrap();
    assert!(!run_policy_audit_with_whitelist(
        &[member.as_str()],
        &whitelist_file
    ));

    // Removing the entry once count reaches 0 passes cleanly
    fs::write(&whitelist_file, "").unwrap();
    assert!(run_policy_audit_with_whitelist(
        &[member.as_str()],
        &whitelist_file
    ));
}

#[test]
fn test_missing_docs_macro_rules_and_vendored_egui_plot_exemption() {
    let dir = TempTestDir::new("policy_audit_missing_docs_vendored");
    let whitelist_file = dir.join("verification_whitelist.toml");

    // macro_rules! module-level missing_docs attributes are counted/blocked
    let mac_member = write_member_crate(
        &dir,
        "crates/mac_demo",
        "macro_rules! gen_mod { () => { #[allow(missing_docs)] pub mod m {} }; }\n",
    );
    fs::write(&whitelist_file, "").unwrap();
    assert!(!run_policy_audit_with_whitelist(
        &[mac_member.as_str()],
        &whitelist_file
    ));
    fs::write(
        &whitelist_file,
        "[missing_docs.files]\n\"crates/mac_demo/src/lib.rs\" = 1\n",
    )
    .unwrap();
    assert!(run_policy_audit_with_whitelist(
        &[mac_member.as_str()],
        &whitelist_file
    ));

    // Vendored egui_plot with #![allow(missing_docs)] requires explicit vendored_crates entry
    let vendored = write_member_crate(
        &dir,
        "egui_plot",
        "#![allow(missing_docs)]\npub fn p() {}\n",
    );
    fs::write(
        dir.join("egui_plot/Cargo.toml"),
        "[package]\nname = \"egui_plot\"\nversion = \"0.34.0\"\n\n[lints.rust]\nmissing_docs = \"warn\"\n[lints.clippy]\ntoo_many_lines = \"deny\"\n",
    )
    .unwrap();
    fs::write(&whitelist_file, "").unwrap();
    assert!(!run_policy_audit_with_whitelist(
        &[vendored.as_str()],
        &whitelist_file
    ));

    fs::write(
        &whitelist_file,
        "[missing_docs]\nvendored_crates = [\"egui_plot\"]\n",
    )
    .unwrap();
    assert!(run_policy_audit_with_whitelist(
        &[vendored.as_str()],
        &whitelist_file
    ));
}
