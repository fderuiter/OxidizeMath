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
fn test_missing_docs_rejects_outer_and_expect_above_cap() {
    let dir = TempTestDir::new("policy_audit_missing_docs_caps");
    let whitelist_file = dir.join("verification_whitelist.toml");
    fs::write(&whitelist_file, "").unwrap();

    // Unwhitelisted outer #[allow(missing_docs)] and #[expect(missing_docs)] fail
    let outer_allow = write_member_crate(
        &dir,
        "crates/outer_allow",
        "#[allow(missing_docs)]\npub struct Item;\n",
    );
    assert!(!run_policy_audit_with_whitelist(
        &[outer_allow.as_str()],
        &whitelist_file
    ));

    let outer_expect = write_member_crate(
        &dir,
        "crates/outer_expect",
        "#[expect(missing_docs, reason = \"legacy\")]\npub struct Item;\n",
    );
    assert!(!run_policy_audit_with_whitelist(
        &[outer_expect.as_str()],
        &whitelist_file
    ));

    // Exact cap passes, adding one more attribute above cap fails
    let capped = write_member_crate(
        &dir,
        "crates/capped",
        "#[allow(missing_docs)]\npub struct A;\n#[allow(rust::missing_docs)]\npub struct B;\n",
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
        "#[allow(missing_docs)]\npub struct A;\n#[allow(missing_docs)]\npub struct B;\n#[expect(missing_docs)]\npub struct C;\n",
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
        "#[allow(missing_docs)]\npub struct A;\n#[allow(missing_docs)]\npub struct B;\n",
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
        "/// Documented A.\npub struct A;\n#[allow(missing_docs)]\npub struct B;\n",
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
        "/// Documented A.\npub struct A;\n/// Documented B.\npub struct B;\n",
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

    // macro_rules! outer and inner missing_docs attributes are counted/blocked
    let mac_member = write_member_crate(
        &dir,
        "crates/mac_demo",
        "macro_rules! gen_fn { () => { #[allow(missing_docs)] pub fn f() {} }; }\n",
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
