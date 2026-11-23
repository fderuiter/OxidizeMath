use super::*;

#[test]
fn test_egui_plot_wasm_paths_are_counted_and_covered() {
    let ws_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    for rel in [
        "egui_plot/src/file_io.rs",
        "egui_plot/src/export.rs",
        "egui_plot/src/export_modal.rs",
    ] {
        let path = ws_root.join(rel);
        let src = fs::read_to_string(&path).expect("egui_plot source must be readable");
        let (paths, covered) = count_file_wasm_coverage(&path, &src);
        assert!(paths > 0, "expected wasm32 paths in {rel}");
        assert_eq!(
            covered, paths,
            "expected all wasm32 paths in {rel} to be covered by #[wasm_bindgen_test]"
        );
        assert!(
            find_orphaned_wasm_unit_tests(&path, &src).is_empty(),
            "egui_plot must not contain #[cfg(target_arch = \"wasm32\")] #[test] items"
        );
    }
}

#[test]
fn test_rejects_orphaned_wasm32_unit_test_attributes() {
    let src = r#"
        #[cfg(target_arch = "wasm32")]
        #[test]
        fn orphaned_direct_item() {}

        #[cfg(all(test, target_arch = "wasm32"))]
        mod wasm_mod {
            #[test]
            fn orphaned_in_wasm_mod() {}

            #[wasm_bindgen_test::wasm_bindgen_test]
            fn valid_browser_test() {}
        }

        #[cfg_attr(target_arch = "wasm32", test)]
        fn orphaned_cfg_attr_test() {}

        #[cfg_attr(not(target_arch = "wasm32"), test)]
        #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
        fn valid_dual_target_test() {}
    "#;
    let found = find_orphaned_wasm_unit_tests(Path::new("probe.rs"), src);
    assert_eq!(
        found,
        vec![
            "orphaned_direct_item",
            "orphaned_in_wasm_mod",
            "orphaned_cfg_attr_test",
        ]
    );
}
