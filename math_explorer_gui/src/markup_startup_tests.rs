#![cfg_attr(any(), verified(opt_out = "tests"))]

const INDEX_HTML: &str = include_str!("../index.html");
const RELOAD_JS: &str = include_str!("../reload.js");
const BOOTSTRAP_JS: &str = include_str!("../bootstrap.js");
const INITIALIZER_JS: &str = include_str!("../initializer.js");
const SW_JS: &str = include_str!("../sw.js");
const MAIN_RS: &str = include_str!("main.rs");

#[test]
fn test_service_worker_caches_shell_only_and_returns_valid_offline_responses() {
    assert!(
        INDEX_HTML.contains("<link data-trunk rel=\"copy-file\" href=\"sw.js\"/>"),
        "index.html must stage sw.js via Trunk copy-file"
    );
    for shell_entry in [
        "'./'",
        "'./index.html'",
        "'./reload.js'",
        "'./bootstrap.js'",
        "'./initializer.js'",
        "'./sw.js'",
    ] {
        assert!(
            SW_JS.contains(shell_entry),
            "sw.js must pre-cache pre-boot shell entry {shell_entry}"
        );
    }
    assert!(
        !SW_JS.contains("math_explorer_gui_bg.wasm"),
        "sw.js must not pre-cache the multi-MB math_explorer_gui_bg.wasm bundle"
    );
    assert!(
        SW_JS.contains("status: 503") && SW_JS.contains("new Response("),
        "sw.js must resolve uncached offline requests with an HTTP 503 Response"
    );
    assert!(
        RELOAD_JS.contains("serviceWorker.register('./sw.js')"),
        "reload.js must register ./sw.js asynchronously"
    );
}

#[test]
fn test_reload_button_form_and_delayed_css_fallback_in_index_html() {
    assert!(
        INDEX_HTML.contains("<form class=\"action-buttons\" method=\"get\" action=\"\">"),
        "index.html must wrap #webgl_reload_button in a same-origin GET <form>"
    );
    assert!(
        INDEX_HTML
            .contains("<button type=\"submit\" id=\"webgl_reload_button\" class=\"reload-btn\">"),
        "#webgl_reload_button must be type=\"submit\" inside the reload form"
    );
    assert!(
        INDEX_HTML.contains("html:not([data-boot-initialized]) #error_container")
            && INDEX_HTML.contains("html:not([data-boot-initialized]) #loading_indicator"),
        "index.html must include delayed pure-CSS fallback rules when startup JS fails to initialize"
    );
    assert!(
        INDEX_HTML
            .contains("#error_container[data-error-category=\"isolation\"] #error_recovery_steps"),
        "index.html must display #error_recovery_steps for data-error-category=\"isolation\""
    );
}

#[test]
fn test_reload_precedes_bootstrap_and_exposes_abort_and_split_timeouts() {
    let reload_pos = INDEX_HTML
        .find("<script src=\"reload.js\" defer></script>")
        .expect("reload.js deferred script tag must exist");
    let bootstrap_pos = INDEX_HTML
        .find("<script type=\"module\" src=\"bootstrap.js\" defer></script>")
        .expect("bootstrap.js deferred module script tag must exist");
    assert!(
        reload_pos < bootstrap_pos,
        "reload.js must appear before bootstrap.js in index.html"
    );

    for contract in [
        "DEFAULT_STALL_TIMEOUT_MS = 12000",
        "DEFAULT_COMPILE_TIMEOUT_MS = 45000",
        "stall_timeout_ms",
        "compile_timeout_ms",
        "new AbortController()",
        "isBootFailed",
        "enterPhase",
        "WebAssembly.RuntimeError",
        "RangeError",
        "ReferenceError",
        "TypeError",
        "sharedarraybuffer",
        "isolation:",
    ] {
        assert!(
            RELOAD_JS.contains(contract),
            "reload.js must define startup coordination contract: {contract}"
        );
    }

    assert!(
        INITIALIZER_JS.contains("enterPhase('compile'"),
        "initializer.js must transition the watchdog to the compile phase once all WASM bytes arrive"
    );
}

#[test]
fn test_bootstrap_dynamic_import_abort_signal_and_fallback_reload_bindings() {
    assert!(
        !BOOTSTRAP_JS.contains("import createInitializer from"),
        "bootstrap.js must dynamically import initializer.js inside bootstrap() rather than statically"
    );
    assert!(
        BOOTSTRAP_JS.contains("await import(INITIALIZER_PATH)"),
        "bootstrap.js must dynamically import INITIALIZER_PATH inside bootstrap()"
    );
    for contract in [
        "const WASM_SOURCE_SIZE = 0;",
        "fetchInit.signal = signal",
        "reader.cancel",
        "isBootAborted(signal)",
        "bindFallbackReloadButton",
    ] {
        assert!(
            BOOTSTRAP_JS.contains(contract),
            "bootstrap.js must include startup cancellation/recovery contract: {contract}"
        );
    }
    assert!(
        MAIN_RS.contains("bind_fallback_reload_listener")
            && MAIN_RS.contains("runner.destroy()")
            && MAIN_RS.contains("is_error_boundary_active"),
        "src/main.rs must guard WebRunner::start and bind a fallback reload listener"
    );
}
