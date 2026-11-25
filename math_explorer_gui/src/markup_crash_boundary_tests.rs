//! Shell markup and script contract tests for the `'runtime'` crash boundary, panic/audio
//! bridge, `#reset_workspace_button`, and `sessionStorage` boot sentinel keys.

const INDEX_HTML: &str = include_str!("../index.html");
const RELOAD_JS: &str = include_str!("../reload.js");
const BOOTSTRAP_JS: &str = include_str!("../bootstrap.js");

#[test]
fn index_html_declares_runtime_recovery_css_and_reset_workspace_button() {
    assert!(
        INDEX_HTML
            .contains("#error_container[data-error-category=\"runtime\"] #error_recovery_steps"),
        "index.html must unhide #error_recovery_steps when data-error-category=\"runtime\""
    );
    assert!(
        INDEX_HTML.contains("id=\"reset_workspace_button\""),
        "index.html must include #reset_workspace_button inside .action-buttons"
    );
    let reset_pos = INDEX_HTML
        .find("id=\"reset_workspace_button\"")
        .expect("#reset_workspace_button must exist");
    let tag_start = INDEX_HTML[..reset_pos]
        .rfind("<button")
        .expect("#reset_workspace_button must be a <button>");
    let tag_end = INDEX_HTML[reset_pos..]
        .find('>')
        .map(|off| reset_pos + off)
        .expect("opening <button> tag must close");
    let button_tag = &INDEX_HTML[tag_start..=tag_end];
    assert!(
        button_tag.contains("type=\"button\""),
        "#reset_workspace_button must have type=\"button\" so Enter still submits Reload Page"
    );
    assert!(
        button_tag.contains("class=\"secondary-btn\""),
        "#reset_workspace_button must use .secondary-btn styling"
    );
    assert!(
        button_tag.contains("hidden"),
        "#reset_workspace_button must be hidden by default"
    );
}

#[test]
fn reload_and_bootstrap_define_runtime_category_and_preserve_compile_guards() {
    assert!(
        RELOAD_JS.contains("runtime: {") && RELOAD_JS.contains("Application Runtime Error"),
        "reload.js must define ERROR_CATEGORIES.runtime"
    );
    assert!(
        BOOTSTRAP_JS.contains("runtime: {") && BOOTSTRAP_JS.contains("Application Runtime Error"),
        "bootstrap.js must define FALLBACK_CATEGORY_COPY.runtime"
    );
    assert!(
        RELOAD_JS.contains("isWasmRuntimeTrap") && RELOAD_JS.contains("WebAssembly.RuntimeError"),
        "reload.js must classify WebAssembly.RuntimeError via isWasmRuntimeTrap"
    );
    assert!(
        BOOTSTRAP_JS.contains("isFallbackWasmRuntimeTrap")
            && BOOTSTRAP_JS.contains("WebAssembly.RuntimeError"),
        "bootstrap.js must classify WebAssembly.RuntimeError via isFallbackWasmRuntimeTrap"
    );
    assert!(
        RELOAD_JS.contains("isFirefoxInternalError")
            && RELOAD_JS.contains("isWasmCompileOrLinkError")
            && BOOTSTRAP_JS.contains("isFirefoxInternalError")
            && BOOTSTRAP_JS.contains("isFallbackCompileOrLinkError"),
        "InternalError, CompileError, and LinkError must remain guarded ahead of runtime traps"
    );
}

#[test]
fn reload_js_exposes_panic_audio_and_reset_hooks_and_keeps_post_boot_listeners() {
    for hook in [
        "recordFatalPanic:",
        "registerAudioStream:",
        "silenceActiveAudioStreams:",
        "resetSavedWorkspaceAndReload:",
    ] {
        assert!(
            RELOAD_JS.contains(hook),
            "window.__mathExplorerBoot in reload.js must expose {hook}"
        );
    }
    assert!(
        RELOAD_JS.contains("resolvedCategory !== 'webgl' && resolvedCategory !== 'runtime'"),
        "showError in reload.js must allow post-boot 'runtime' error transitions"
    );
    assert!(
        RELOAD_JS.contains("window.history.replaceState")
            && BOOTSTRAP_JS.contains("window.history.replaceState"),
        "reset handlers must clear share URL parameters via History.replaceState"
    );
    assert!(
        RELOAD_JS.contains("math_explorer_boot_sentinel")
            && RELOAD_JS.contains("math_explorer_quarantined_workspace")
            && RELOAD_JS.contains("math_explorer_preserved_theme"),
        "reload.js must share sentinel, quarantine, and preserved-theme storage keys with Rust"
    );
}
