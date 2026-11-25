#![cfg_attr(any(), verified(opt_out = "tests"))]

const INDEX_HTML: &str = include_str!("../index.html");
const RELOAD_JS: &str = include_str!("../reload.js");
const BOOTSTRAP_JS: &str = include_str!("../bootstrap.js");
const INITIALIZER_JS: &str = include_str!("../initializer.js");
const SW_JS: &str = include_str!("../sw.js");
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
    for forbidden_cache_override in ["cache: 'no-cache'", "cache: 'reload'"] {
        assert!(
            !SW_JS.contains(forbidden_cache_override),
            "sw.js must honor fresh HTTP cache entries without forcing {forbidden_cache_override}"
        );
    }
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
        "normalizeContentType",
        "isAllowedWasmContentType",
        "content-type",
        "application/wasm",
        "application/octet-stream",
        "response.body.cancel",
        "lacks \\\\0asm binary header",
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
        let mime_check = script
            .find("NON_WASM_RESPONSE_PATTERN.test(")
            .unwrap_or_else(|| panic!("{name} must test NON_WASM_RESPONSE_PATTERN"));
        let compile_err_check = script
            .find("WebAssembly.CompileError")
            .unwrap_or_else(|| panic!("{name} must check WebAssembly.CompileError"));
        assert!(
            trunc_check < compile_err_check && mime_check < compile_err_check,
            "{name} must classify stream truncation and non-WASM responses as network before WebAssembly.CompileError"
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
fn test_indeterminate_and_compile_progress_styles_across_themes_and_reduced_motion() {
    for selector in [
        "@keyframes loading-progress-sweep",
        ".loading-progress:indeterminate::-webkit-progress-bar",
        ".loading-progress:indeterminate::-moz-progress-bar",
        ".loading-progress[data-phase=\"compile\"]::-webkit-progress-bar",
        ".loading-progress[data-phase=\"compile\"]::-moz-progress-bar",
        "html:not([data-theme]) .loading-progress[data-phase=\"compile\"]",
        "html[data-theme=\"dark\"] .loading-progress[data-phase=\"compile\"]",
        "html[data-theme=\"light\"] .loading-progress[data-phase=\"compile\"]",
        "html[data-theme=\"high-contrast\"] .loading-progress[data-phase=\"compile\"]",
    ] {
        assert!(
            INDEX_HTML.contains(selector),
            "index.html must define startup progress selector: {selector}"
        );
    }

    let forced_start = INDEX_HTML
        .find("@media (forced-colors: active)")
        .expect("forced-colors block must exist");
    let dark_theme_start = INDEX_HTML
        .find("html[data-theme=\"dark\"]")
        .expect("dark data-theme block must exist");
    assert!(
        INDEX_HTML[forced_start..dark_theme_start]
            .contains(".loading-progress[data-phase=\"compile\"]"),
        "forced-colors block must style .loading-progress[data-phase=\"compile\"]"
    );

    let reduced_motion_start = INDEX_HTML
        .find("@media (prefers-reduced-motion: reduce)")
        .expect("prefers-reduced-motion block must exist");
    let reduced_motion_css = &INDEX_HTML[reduced_motion_start..];
    assert!(
        reduced_motion_css.contains("background-position: 50% 0 !important;")
            && reduced_motion_css.contains("background-image: none !important;"),
        "prefers-reduced-motion must freeze :indeterminate band and render solid [data-phase=\"compile\"] fill"
    );
}

#[test]
fn test_initializer_tags_and_clears_compile_phase_on_loading_progress() {
    for contract in [
        "function setIndeterminateProgress(progressEl, stageLabel, phase)",
        "progressEl.setAttribute('data-phase', phase)",
        "progressEl.removeAttribute('data-phase')",
        "setIndeterminateProgress(progressEl, 'Starting WebAssembly download')",
        "setIndeterminateProgress(progressEl, statusCopy)",
        "setIndeterminateProgress(progressEl, 'Compiling WebAssembly module', 'compile')",
        "setIndeterminateProgress(progressEl, 'Initializing WebGL and graphics engine', 'compile')",
    ] {
        assert!(
            INITIALIZER_JS.contains(contract),
            "initializer.js must manage #loading_progress data-phase across stages: {contract}"
        );
    }
}
