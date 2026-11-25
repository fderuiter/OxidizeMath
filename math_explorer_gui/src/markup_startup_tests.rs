#![cfg_attr(any(), verified(opt_out = "tests"))]

const INDEX_HTML: &str = include_str!("../index.html");
const THEME_INIT_JS: &str = include_str!("../theme-init.js");
const RELOAD_JS: &str = include_str!("../reload.js");
const BOOTSTRAP_JS: &str = include_str!("../bootstrap.js");
const INITIALIZER_JS: &str = include_str!("../initializer.js");
const SW_JS: &str = include_str!("../sw.js");
const SITE_WEBMANIFEST: &str = include_str!("../site.webmanifest");
const MAIN_RS: &str = include_str!("main.rs");
const FINALIZE_WEB_DIST_SH: &str = include_str!("../../scripts/finalize_web_dist.sh");
const COMPRESS_DIST_SH: &str = include_str!("../../scripts/compress_dist.sh");

#[test]
fn test_service_worker_caches_shell_only_and_returns_valid_offline_responses() {
    assert!(
        INDEX_HTML.contains("<link data-trunk rel=\"copy-file\" href=\"sw.js\"/>"),
        "index.html must stage sw.js via Trunk copy-file"
    );
    for shell_entry in [
        "'./'",
        "'./index.html'",
        "'./theme-init.js'",
        "'./reload.js'",
        "'./bootstrap.js'",
        "'./initializer.js'",
        "'./sw.js'",
        "'./site.webmanifest'",
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
        !SW_JS.contains("status: 503") && !SW_JS.contains("buildOfflineUnavailableResponse"),
        "sw.js must pass uncached non-shell requests natively without synthetic HTTP 503 responses"
    );
    for forbidden_cache_override in ["cache: 'no-cache'", "cache: 'reload'"] {
        assert!(
            !SW_JS.contains(forbidden_cache_override),
            "sw.js must honor fresh HTTP cache entries without forcing {forbidden_cache_override}"
        );
    }
    for sw_defer_contract in [
        "serviceWorker.register('./sw.js')",
        "requestAnimationFrame",
        "requestIdleCallback",
        "scheduleAppShellServiceWorkerRegistration",
    ] {
        assert!(
            RELOAD_JS.contains(sw_defer_contract),
            "reload.js must defer ./sw.js registration until post-Frame-1 idle time: {sw_defer_contract}"
        );
    }
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
    for forbidden_serial in [
        "import createInitializer from",
        "await import(INITIALIZER_PATH)",
        "await import(MODULE_PATH)",
    ] {
        assert!(
            !BOOTSTRAP_JS.contains(forbidden_serial),
            "bootstrap.js must avoid static or serial module import '{forbidden_serial}'"
        );
    }
    for contract in [
        "const WASM_SOURCE_SIZE = 0;",
        "const WASM_BROTLI_SIZE = 0;",
        "const WASM_GZIP_SIZE = 0;",
        "import(INITIALIZER_PATH)",
        "import(MODULE_PATH)",
        "fetchWasmWithProgress(WASM_PATH, hooksProxy, signal, recordStreamError)",
        "await Promise.all([",
        "createStartupProgressProxy",
        "attachAssetPath(",
        "fetchInit.signal = signal",
        "reader.cancel",
        "isBootAborted(signal)",
        "bindFallbackReloadButton",
    ] {
        assert!(
            BOOTSTRAP_JS.contains(contract),
            "bootstrap.js must include parallel startup and recovery contract: {contract}"
        );
    }
    assert!(
        MAIN_RS.contains("bind_fallback_reload_listener")
            && MAIN_RS.contains("runner.destroy()")
            && MAIN_RS.contains("is_error_boundary_active"),
        "src/main.rs must guard WebRunner::start and bind a fallback reload listener"
    );
}

#[test]
fn test_finalize_web_dist_stamps_compressed_sizes_before_compressing_bootstrap() {
    assert!(
        COMPRESS_DIST_SH.contains("\"${@:2}\"") && COMPRESS_DIST_SH.contains("EXTS="),
        "compress_dist.sh must accept optional extension filters for staged compression passes"
    );

    let wasm_compress_pos = FINALIZE_WEB_DIST_SH
        .find("compress_dist.sh\" \"$STAGING_DIR\" wasm")
        .expect("finalize_web_dist.sh must compress .wasm before stamping bootstrap.js");
    let stamp_source_pos = FINALIZE_WEB_DIST_SH
        .find("const WASM_SOURCE_SIZE = ")
        .expect("finalize_web_dist.sh must stamp WASM_SOURCE_SIZE");
    let stamp_brotli_pos = FINALIZE_WEB_DIST_SH
        .find("const WASM_BROTLI_SIZE = ")
        .expect("finalize_web_dist.sh must stamp WASM_BROTLI_SIZE");
    let stamp_gzip_pos = FINALIZE_WEB_DIST_SH
        .find("const WASM_GZIP_SIZE = ")
        .expect("finalize_web_dist.sh must stamp WASM_GZIP_SIZE");
    let js_compress_pos = FINALIZE_WEB_DIST_SH
        .find("compress_dist.sh\" \"$STAGING_DIR\" js html css svg webmanifest json")
        .expect("finalize_web_dist.sh must compress bootstrap.js after stamping sizes");

    assert!(
        wasm_compress_pos < stamp_source_pos
            && stamp_source_pos < stamp_brotli_pos
            && stamp_brotli_pos < stamp_gzip_pos
            && stamp_gzip_pos < js_compress_pos,
        "finalize_web_dist.sh must compress .wasm first, stamp all three sizes, then compress JS assets"
    );
}

#[test]
fn test_bootstrap_guards_wasm_stream_header_length_and_wire_byte_counters() {
    for contract in [
        "WASM_MAGIC_BYTES = [0x00, 0x61, 0x73, 0x6d]",
        "evaluateStreamCompletion",
        "resolveWireByteTotal",
        "scaleInFlightWireProgress",
        "reconcileCompletedWireTotal",
        "encodedBodySize",
        "content-encoding",
        "content-length",
        "WASM_BROTLI_SIZE",
        "WASM_GZIP_SIZE",
    ] {
        assert!(
            BOOTSTRAP_JS.contains(contract),
            "bootstrap.js must enforce stream validation and wire-byte progress contract: {contract}"
        );
    }
}

#[test]
fn test_initializer_paces_polite_aria_buckets_and_clears_on_failure() {
    for contract in [
        "POLITE_MILESTONES = [25, 50, 75]",
        "window.setTimeout(drainNextPoliteAnnouncement",
        "window.clearTimeout(politeTimerId)",
        "enqueueCrossedMilestones(100)",
        "enqueueCrossedMilestones(pct)",
        "cancelPendingAnnouncements",
        "clearPendingPoliteAnnouncements",
        "signal.addEventListener('abort', clearPendingPoliteAnnouncements",
    ] {
        assert!(
            INITIALIZER_JS.contains(contract),
            "initializer.js must pace 25/50/75% polite updates and clear pending announcements: {contract}"
        );
    }
    assert!(
        BOOTSTRAP_JS.contains("hooks?.cancelPendingAnnouncements?.()"),
        "bootstrap.js must clear pending polite announcements when startup aborts or fails"
    );
}

#[test]
fn test_truncation_precedes_compile_error_and_internal_error_maps_to_compile() {
    for (name, script) in [("reload.js", RELOAD_JS), ("bootstrap.js", BOOTSTRAP_JS)] {
        let trunc_check = script
            .find("STREAM_TRUNCATION_PATTERN.test(")
            .unwrap_or_else(|| panic!("{name} must test STREAM_TRUNCATION_PATTERN"));
        let compile_err_check = script
            .find("WebAssembly.CompileError")
            .unwrap_or_else(|| panic!("{name} must check WebAssembly.CompileError"));
        assert!(
            trunc_check < compile_err_check,
            "{name} must classify stream truncation as network before WebAssembly.CompileError"
        );

        for token in [
            "InternalError",
            "out of memory",
            "too much recursion",
            "allocation size overflow",
        ] {
            assert!(
                script.contains(token),
                "{name} must route Firefox InternalError token '{token}' to compile recovery"
            );
        }
    }
}

#[test]
fn test_sw_swr_bounded_navigation_timeout_and_all_manifest_icons_cached() {
    let manifest: serde_json::Value =
        serde_json::from_str(SITE_WEBMANIFEST).expect("site.webmanifest must parse as valid JSON");
    let icons = manifest["icons"]
        .as_array()
        .expect("site.webmanifest must declare an icons array");
    assert_eq!(icons.len(), 7, "site.webmanifest must declare all 7 icons");
    for icon in icons {
        let src = icon["src"]
            .as_str()
            .expect("manifest icon must specify src");
        let sw_token = format!("'./{src}'");
        assert!(
            SW_JS.contains(&sw_token),
            "sw.js OPTIONAL_SHELL_PATHS must include manifest icon {sw_token}"
        );
    }
    for contract in [
        "NAVIGATION_FETCH_TIMEOUT_MS = 1200",
        "new AbortController()",
        "controller.abort()",
        "buildDeduplicatedPrecachePaths",
        "Promise.all(",
        "revalidateShellAsset",
        "event.waitUntil(",
    ] {
        assert!(
            SW_JS.contains(contract),
            "sw.js must implement SWR, 1,200 ms navigation abort, and parallel deduplicated precache: {contract}"
        );
    }
}

#[test]
fn test_preboot_theme_init_and_offline_recovery_contracts() {
    for token in [
        "window.localStorage",
        "getItem('app')",
        "JSON.parse",
        "schema_version",
        "app_state",
        "dataset.theme",
        "color-scheme",
        "meta[name=\"theme-color\"]",
        "#121214",
        "#f8f8f8",
        "#000000",
        "high-contrast",
    ] {
        assert!(
            THEME_INIT_JS.contains(token),
            "theme-init.js must synchronize saved theme before first paint: {token}"
        );
    }
    assert!(
        !THEME_INIT_JS.contains("data-boot-initialized"),
        "theme-init.js must not prematurely mark data-boot-initialized before reload.js/bootstrap.js"
    );
    for offline_contract in [
        "offline:",
        "navigator.onLine === false",
        "offline_connection_status",
        "addEventListener('online'",
        "addEventListener('offline'",
        "download_workspace_button",
    ] {
        assert!(
            RELOAD_JS.contains(offline_contract) && BOOTSTRAP_JS.contains(offline_contract),
            "reload.js and bootstrap.js must support dedicated offline recovery contract: {offline_contract}"
        );
    }
}
