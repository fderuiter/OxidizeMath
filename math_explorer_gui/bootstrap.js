// External WebAssembly bootstrap module for Math Explorer GUI under strict CSP ('self').
// Dynamically imports initializer.js and math_explorer_gui.js, streams WebAssembly download
// progress with AbortSignal cancellation, and transitions cleanly to recovery on failure.

const WASM_SOURCE_SIZE = 0;
const WASM_BROTLI_SIZE = 0;
const WASM_GZIP_SIZE = 0;
const WASM_MAGIC_BYTES = [0x00, 0x61, 0x73, 0x6d];
const INITIALIZER_PATH = './initializer.js';
const MODULE_PATH = './math_explorer_gui.js';
const WASM_PATH = './math_explorer_gui_bg.wasm';
const FALLBACK_STALL_TIMEOUT_MS = 12000;
const FALLBACK_COMPILE_TIMEOUT_MS = 45000;
const DEFAULT_RECOVERY_FILENAME = 'math-explorer-recovery.mws';

const SHELL_THEME_TOKENS = {
    dark: { colorScheme: 'dark', themeColor: '#121214' },
    light: { colorScheme: 'light', themeColor: '#f8f8f8' },
    'high-contrast': { colorScheme: 'dark', themeColor: '#000000' },
};

const FALLBACK_CATEGORY_COPY = {
    network: {
        title: 'Network Error Loading Math Explorer',
        message:
            'Math Explorer could not finish downloading its WebAssembly application files. Your connection may have been interrupted or the server returned an error.',
        heading: 'Recovery Steps',
        steps: [
            'Check your internet connection and verify that firewalls, proxies, or content blockers allow .js and .wasm requests.',
            'Select Reload Page below (or perform a hard refresh) to retry downloading the application.',
            'If the server is temporarily unavailable, wait a moment and reload.',
        ],
        alert:
            'Network error loading Math Explorer. Check your internet connection and select Reload Page to retry.',
    },
    compile: {
        title: 'WebAssembly Startup Error',
        message:
            'Your browser downloaded the application bundle, but failed to compile, instantiate, or start the WebAssembly module. Cached files may be incomplete or incompatible.',
        heading: 'Recovery Steps',
        steps: [
            'Select Reload Page or perform a hard refresh (Ctrl+Shift+R / Cmd+Shift+R) to bypass stale or corrupted cached files.',
            'Ensure you are using a current release of Chrome, Edge, Firefox, or Safari with WebAssembly enabled.',
            'Verify that browser security extensions or enterprise policies do not block WebAssembly execution.',
        ],
        alert:
            'WebAssembly startup error. Cached files may be corrupted. Select Reload Page to retry.',
    },
    isolation: {
        title: 'Browser Isolation & Shared Memory Unavailable',
        message:
            'Math Explorer encountered an operation requiring SharedArrayBuffer or cross-origin isolation that is unavailable or blocked in this browser context.',
        heading: 'Recovery Steps',
        steps: [
            'Open Math Explorer over HTTPS or localhost in a standard browser window rather than a restricted embedded WebView.',
            'Allow site data, Service Workers, and standard WebAssembly memory features for this origin.',
            'Disable extensions or enterprise policies that strip isolation headers or block SharedArrayBuffer, then select Reload Page.',
        ],
        alert:
            'Browser isolation or SharedArrayBuffer is unavailable. Adjust browser settings and select Reload Page.',
    },
    timeout: {
        title: 'Startup Timed Out',
        message:
            'Loading Math Explorer stalled without progress before startup could complete. The network transfer may have frozen or stopped responding.',
        heading: 'Recovery Steps',
        steps: [
            'Confirm your network connection is active and not waiting on a captive portal or VPN.',
            'Select Reload Page below to restart the WebAssembly download cleanly.',
            'If you are on a slow or metered connection, pause competing downloads and try again.',
        ],
        alert:
            'Loading Math Explorer timed out due to a stalled transfer. Select Reload Page to retry.',
    },
    webgl: {
        title: 'WebGL Initialization Failed',
        message:
            'Math Explorer requires WebGL and hardware acceleration to render mathematics and interactive simulations. Your browser was unable to initialize a WebGL rendering context.',
        heading: 'Enable Hardware Acceleration & WebGL',
        steps: [],
        alert:
            'WebGL initialization failed. Please enable hardware acceleration in your browser settings.',
    },
};

const FALLBACK_COMPILE_TIMEOUT_COPY = {
    title: 'Startup Timed Out',
    message:
        'Math Explorer finished downloading its application bundle, but WebAssembly compilation or graphics initialization timed out before startup could complete.',
    heading: 'Recovery Steps',
    steps: [
        'Select Reload Page below to retry compiling WebAssembly and initializing WebGL.',
        'Close memory-intensive browser tabs or background applications if your device is low on CPU or memory.',
        'Verify that browser extensions or security policies are not stalling WebAssembly compilation or WebGL context creation.',
    ],
    alert:
        'Math Explorer timed out during WebAssembly compilation or graphics initialization. Select Reload Page to retry.',
};

// Start fallback twin-slot toggles at slot 1 so slot 0 stays empty when Rust's AtomicBool::new(false) takes over.
let fallbackPoliteSlotToggle = 1;
let fallbackAssertiveSlotToggle = 1;
const probedResourceStatuses = new Map();
let activeWatchdogCleanup = null;
let isPageUnloading = false;

function markBootInitialized() {
    const root = document.documentElement;
    if (root && !root.hasAttribute('data-boot-initialized')) {
        root.setAttribute('data-boot-initialized', 'true');
    }
}

function normalizeSavedTheme(rawTheme) {
    if (typeof rawTheme !== 'string') {
        return null;
    }
    const normalized = rawTheme.trim().toLowerCase();
    if (normalized === 'dark') {
        return 'dark';
    }
    if (normalized === 'light') {
        return 'light';
    }
    if (
        normalized === 'highcontrast' ||
        normalized === 'high-contrast' ||
        normalized === 'high_contrast'
    ) {
        return 'high-contrast';
    }
    return null;
}

function readSavedThemeFromStorage() {
    try {
        if (!window.localStorage) {
            return null;
        }
        const raw = window.localStorage.getItem('app');
        if (!raw) {
            return null;
        }
        const parsed = JSON.parse(raw.replace(/^\uFEFF/, ''));
        if (!parsed || typeof parsed !== 'object') {
            return null;
        }
        if (typeof parsed.schema_version === 'string') {
            const major = parseInt(parsed.schema_version.split('.')[0], 10);
            if (
                !Number.isFinite(major) ||
                major > 1 ||
                typeof parsed.app_version !== 'string' ||
                typeof parsed.timestamp !== 'string' ||
                !parsed.app_state ||
                typeof parsed.app_state !== 'object'
            ) {
                return null;
            }
            return normalizeSavedTheme(parsed.app_state.theme);
        }
        if (
            typeof parsed.selected_tab !== 'number' ||
            !parsed.tab_states ||
            typeof parsed.tab_states !== 'object'
        ) {
            return null;
        }
        return normalizeSavedTheme(parsed.theme);
    } catch (_) {
        // Fall back to OS media queries on SecurityError or invalid JSON
        return null;
    }
}

function syncShellThemeFromStorage() {
    const savedTheme = readSavedThemeFromStorage();
    const tokens = savedTheme && SHELL_THEME_TOKENS[savedTheme];
    if (!tokens) {
        return null;
    }
    const root = document.documentElement;
    if (root) {
        root.dataset.theme = savedTheme;
        if (root.style && typeof root.style.setProperty === 'function') {
            root.style.setProperty('color-scheme', tokens.colorScheme);
        }
    }
    const schemeMeta = document.querySelector('meta[name="color-scheme"]');
    if (schemeMeta) {
        schemeMeta.setAttribute('content', tokens.colorScheme);
    }
    const themeMetas = document.querySelectorAll('meta[name="theme-color"]');
    for (let i = 0; i < themeMetas.length; i += 1) {
        themeMetas[i].setAttribute('content', tokens.themeColor);
    }
    return savedTheme;
}

function announceFallback(message, priority) {
    if (!message || typeof document === 'undefined') {
        return;
    }
    const isAssertive =
        String(priority).toLowerCase() === 'assertive' ||
        String(priority).toLowerCase() === 'alert';
    const containerId = isAssertive ? 'aria-live-assertive' : 'aria-live-polite';
    const slot0 = document.getElementById(containerId + '-slot-0');
    const slot1 = document.getElementById(containerId + '-slot-1');
    let slotIdx = isAssertive ? fallbackAssertiveSlotToggle : fallbackPoliteSlotToggle;
    if (slot0 && slot1) {
        const slot0Busy = Boolean(slot0.textContent && slot0.textContent.length > 0);
        const slot1Busy = Boolean(slot1.textContent && slot1.textContent.length > 0);
        if (slot0Busy && !slot1Busy) {
            slotIdx = 1;
        } else if (slot1Busy && !slot0Busy) {
            slotIdx = 0;
        }
    }
    if (isAssertive) {
        fallbackAssertiveSlotToggle = 1 - slotIdx;
    } else {
        fallbackPoliteSlotToggle = 1 - slotIdx;
    }
    const activeSlot = slotIdx === 0 ? slot0 : slot1;
    const siblingSlot = slotIdx === 0 ? slot1 : slot0;
    if (siblingSlot) {
        siblingSlot.textContent = '';
    }
    if (activeSlot) {
        activeSlot.textContent = String(message);
    }
}

function readFallbackEnvelopeFromStorage() {
    try {
        if (!window.localStorage) {
            return '';
        }
        const raw = window.localStorage.getItem('app');
        return typeof raw === 'string' && raw.trim().length > 0 ? raw : '';
    } catch (_) {
        return '';
    }
}

function downloadFallbackWorkspace() {
    const payload = readFallbackEnvelopeFromStorage();
    if (!payload || typeof Blob === 'undefined' || !window.URL || !URL.createObjectURL) {
        return false;
    }
    const blob = new Blob([payload], { type: 'application/json' });
    const objectUrl = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = objectUrl;
    link.download = DEFAULT_RECOVERY_FILENAME;
    link.rel = 'noopener';
    link.style.display = 'none';
    if (document.body) {
        document.body.appendChild(link);
    }
    link.click();
    if (link.parentNode) {
        link.parentNode.removeChild(link);
    }
    window.setTimeout(() => {
        URL.revokeObjectURL(objectUrl);
    }, 0);
    announceFallback('Downloaded saved workspace envelope ' + link.download + '.', 'polite');
    return true;
}

function bindFallbackDownloadWorkspaceButton() {
    const downloadButton = document.getElementById('download_workspace_button');
    if (downloadButton && !downloadButton.dataset.downloadBound) {
        downloadButton.dataset.downloadBound = 'true';
        downloadButton.addEventListener('click', (event) => {
            if (event && typeof event.preventDefault === 'function') {
                event.preventDefault();
            }
            downloadFallbackWorkspace();
        });
    }
    return downloadButton;
}

function syncFallbackDownloadWorkspaceButton() {
    const btn = document.getElementById('download_workspace_button');
    if (!btn) {
        return null;
    }
    const hasPayload = readFallbackEnvelopeFromStorage().length > 0;
    btn.hidden = !hasPayload;
    if (btn.getAttribute('data-storage-blocked') !== 'true') {
        btn.setAttribute('data-storage-blocked', 'false');
    }
    return btn;
}

function bindFallbackReloadButton() {
    const reloadButton =
        document.getElementById('webgl_reload_button') ||
        document.querySelector('.reload-btn');
    if (reloadButton && !reloadButton.dataset.reloadBound) {
        reloadButton.dataset.reloadBound = 'true';
        reloadButton.addEventListener('click', (event) => {
            if (event && typeof event.preventDefault === 'function') {
                event.preventDefault();
            }
            window.location.reload();
        });
    }
    return reloadButton;
}

function formatError(err) {
    if (!err) {
        return 'Unknown initialization error.';
    }
    if (typeof err === 'string') {
        return err;
    }
    if (err instanceof Error) {
        return err.name ? `${err.name}: ${err.message}` : err.message;
    }
    if (typeof err.message === 'string' && err.message.length > 0) {
        return err.name ? `${err.name}: ${err.message}` : err.message;
    }
    try {
        return JSON.stringify(err);
    } catch (_) {
        return String(err);
    }
}

function assetBaseName(assetPath) {
    if (!assetPath || typeof assetPath !== 'string') {
        return '';
    }
    const cleaned = assetPath.replace(/^\.\//, '').split(/[?#]/)[0];
    const parts = cleaned.split('/');
    return parts[parts.length - 1] || cleaned;
}

function lookupResourceHttpStatus(assetPath) {
    const boot = window.__mathExplorerBoot;
    if (boot && typeof boot.findFailedStartupResource === 'function') {
        const entry = boot.findFailedStartupResource(assetPath);
        if (entry && entry.status >= 400) {
            return entry.status;
        }
    }
    const targetName = assetBaseName(assetPath);
    if (!targetName) {
        return 0;
    }
    const probed = probedResourceStatuses.get(targetName);
    if (typeof probed === 'number' && probed >= 400) {
        return probed;
    }
    if (typeof performance === 'undefined' || typeof performance.getEntriesByType !== 'function') {
        return 0;
    }
    const entries = performance.getEntriesByType('resource') || [];
    for (let i = entries.length - 1; i >= 0; i -= 1) {
        const entry = entries[i];
        if (
            entry &&
            typeof entry.name === 'string' &&
            entry.name.indexOf(targetName) !== -1 &&
            typeof entry.responseStatus === 'number' &&
            entry.responseStatus >= 400
        ) {
            return entry.responseStatus;
        }
    }
    return 0;
}

function cancelProbeResponseBody(resp) {
    if (resp && resp.body && typeof resp.body.cancel === 'function') {
        void resp.body.cancel().catch(() => {});
    }
}

async function probeAssetHttpStatus(assetPath) {
    const targetName = assetBaseName(assetPath);
    if (!targetName || typeof fetch !== 'function' || !/\.(?:js|mjs|wasm)$/i.test(targetName)) {
        return 0;
    }
    const cached = probedResourceStatuses.get(targetName);
    if (typeof cached === 'number' && cached >= 400) {
        return cached;
    }
    let resolvedUrl;
    try {
        resolvedUrl = new URL(assetPath, window.location.href);
        if (resolvedUrl.origin !== window.location.origin) {
            return 0;
        }
    } catch (_) {
        return 0;
    }
    let status = 0;
    try {
        const headResp = await fetch(resolvedUrl.href, {
            method: 'HEAD',
            credentials: 'same-origin',
            cache: 'no-store',
        });
        cancelProbeResponseBody(headResp);
        if (headResp.status === 405 || headResp.status === 501) {
            const getResp = await fetch(resolvedUrl.href, {
                method: 'GET',
                credentials: 'same-origin',
                cache: 'no-store',
            });
            cancelProbeResponseBody(getResp);
            status = getResp.status >= 400 ? getResp.status : 0;
        } else {
            status = headResp.status >= 400 ? headResp.status : 0;
        }
    } catch (_) {
        try {
            const getResp = await fetch(resolvedUrl.href, {
                method: 'GET',
                credentials: 'same-origin',
                cache: 'no-store',
            });
            cancelProbeResponseBody(getResp);
            status = getResp.status >= 400 ? getResp.status : 0;
        } catch (_inner) {
            status = 0;
        }
    }
    if (status >= 400) {
        probedResourceStatuses.set(targetName, status);
    }
    return status;
}

function enrichAssetError(err, assetPath) {
    const baseText = formatError(err);
    const name = assetBaseName(assetPath);
    const statusCode = lookupResourceHttpStatus(assetPath);
    if (statusCode >= 400 && name) {
        const statusToken = `HTTP ${statusCode}`;
        if (baseText.indexOf(statusToken) !== -1 && baseText.indexOf(name) !== -1) {
            return baseText;
        }
        return `Failed to load startup asset ${name} (${statusToken}): ${baseText}`;
    }
    if (
        name &&
        baseText.indexOf(name) === -1 &&
        /failed to fetch|dynamically imported module|importing a module|load failed|networkerror/i.test(
            baseText
        )
    ) {
        return `Failed to load startup asset ${name}: ${baseText}`;
    }
    return baseText;
}

const STREAM_TRUNCATION_PATTERN =
    /stream truncated|truncated stream|stream interrupted|stream closed early|ended early|empty response|0 bytes received|unexpected eof|unexpected end of|premature eof|incomplete download|incomplete transfer|err_incomplete_chunked_encoding|err_content_decoding_failed|err_http2_protocol_error|failed to read from a readablestream|buffersource argument is empty|fell off end|extends past end of the module/i;

const NON_WASM_RESPONSE_PATTERN =
    /unexpected content-type|unexpected response mime type|unsupported mime type|incorrect response mime type|disallowed mime type|not a valid javascript mime type|mime type of|expected ['"]?application\/wasm['"]?|non-webassembly response|lacks \\0asm binary header/i;

const FIREFOX_INTERNAL_ERROR_PATTERN =
    /internalerror|out of memory|too much recursion|allocation size overflow/i;

function isFirefoxInternalError(err, detailText) {
    if (typeof InternalError !== 'undefined' && err instanceof InternalError) {
        return true;
    }
    if (err && typeof err === 'object' && err.name === 'InternalError') {
        return true;
    }
    return FIREFOX_INTERNAL_ERROR_PATTERN.test(detailText || formatError(err));
}

function classifyFallbackCategory(err, detailText, assetPath) {
    if (/timeout|timed out|stalled/i.test(detailText)) {
        return 'timeout';
    }
    if (lookupResourceHttpStatus(assetPath) >= 400) {
        return 'network';
    }
    if (
        STREAM_TRUNCATION_PATTERN.test(detailText) ||
        NON_WASM_RESPONSE_PATTERN.test(detailText) ||
        /http\s+[45]\d\d|status code is not ok|failed to fetch|networkerror|network error|load failed|dynamically imported module|importing a module script failed|failed to load startup|aborted|aborterror|err_aborted|err_failed|err_connection|err_internet|err_content_length_mismatch/i.test(
            detailText,
        )
    ) {
        return 'network';
    }
    if (/webgl|hardware acceleration|rendering context/i.test(detailText)) {
        return 'webgl';
    }
    if (/sharedarraybuffer|cross-origin isolat|crossoriginisolated/i.test(detailText)) {
        return 'isolation';
    }
    if (
        isFirefoxInternalError(err, detailText) ||
        (typeof WebAssembly !== 'undefined' &&
            (err instanceof WebAssembly.CompileError ||
                err instanceof WebAssembly.LinkError ||
                err instanceof WebAssembly.RuntimeError)) ||
        err instanceof SyntaxError ||
        err instanceof RangeError ||
        err instanceof ReferenceError ||
        err instanceof TypeError
    ) {
        return 'compile';
    }
    return 'compile';
}

function selectFallbackCopy(category, detailText, phase) {
    if (
        category === 'timeout' &&
        (phase === 'compile' || /compil|graphics|webrunner/i.test(detailText))
    ) {
        return FALLBACK_COMPILE_TIMEOUT_COPY;
    }
    return FALLBACK_CATEGORY_COPY[category] || FALLBACK_CATEGORY_COPY.compile;
}

function buildDiagnosticDetails(err, assetPath) {
    if (!err) {
        return '';
    }
    return 'Initialization error: ' + enrichAssetError(err, assetPath);
}

function showBootstrapRecovery(err, assetPath, phase) {
    if (isPageUnloading) {
        return;
    }
    if (typeof activeWatchdogCleanup === 'function') {
        activeWatchdogCleanup();
    }
    const detailText = buildDiagnosticDetails(err, assetPath);
    const category = classifyFallbackCategory(err, detailText, assetPath);
    const copy = selectFallbackCopy(category, detailText, phase);

    const applyDomErrorState = () => {
        if (isPageUnloading) {
            return;
        }
        markBootInitialized();
        if (!window.__mathExplorerBoot) {
            syncShellThemeFromStorage();
        }
        const root = document.documentElement;
        if (root) {
            root.setAttribute('data-boot-failed', 'true');
        }
        try {
            delete window.wasmBindings;
        } catch (_) {
            window.wasmBindings = undefined;
        }

        const loading = document.getElementById('loading_indicator');
        if (loading) {
            loading.setAttribute('aria-busy', 'false');
            loading.remove();
        }

        const recoveryOverlay = document.getElementById('webgl_recovery_overlay');
        if (recoveryOverlay) {
            recoveryOverlay.setAttribute('aria-busy', 'false');
            recoveryOverlay.hidden = true;
            recoveryOverlay.setAttribute('style', 'display: none;');
        }

        const canvas = document.getElementById('the_canvas_id');
        if (canvas) {
            canvas.setAttribute('style', 'display: none;');
        }

        const errorContainer = document.getElementById('error_container');
        const alreadyActive =
            Boolean(errorContainer) &&
            errorContainer.getAttribute('data-error-active') === 'true';
        const previousCategory = errorContainer
            ? errorContainer.getAttribute('data-error-category')
            : null;

        if (errorContainer) {
            errorContainer.setAttribute('data-error-category', category);
            errorContainer.setAttribute('data-error-active', 'true');
            if (!errorContainer.hasAttribute('data-storage-blocked')) {
                errorContainer.setAttribute('data-storage-blocked', 'false');
            }
            errorContainer.setAttribute('style', 'display: flex;');
        }

        const titleEl = document.getElementById('error_title');
        if (titleEl) {
            titleEl.textContent = copy.title;
        }

        const messageEl = document.getElementById('error_message');
        if (messageEl) {
            messageEl.textContent = copy.message;
        }

        if (category !== 'webgl') {
            const headingEl = document.getElementById('error_recovery_heading');
            if (headingEl) {
                headingEl.textContent = copy.heading;
            }
            const listEl = document.getElementById('error_recovery_list');
            if (listEl && Array.isArray(copy.steps)) {
                while (listEl.firstChild) {
                    listEl.removeChild(listEl.firstChild);
                }
                for (let i = 0; i < copy.steps.length; i += 1) {
                    const li = document.createElement('li');
                    li.textContent = copy.steps[i];
                    listEl.appendChild(li);
                }
            }
        }

        const detailsEl = document.getElementById('error_details');
        if (detailsEl && detailText) {
            detailsEl.textContent = detailText;
            detailsEl.setAttribute('style', 'display: block;');
        }

        bindFallbackDownloadWorkspaceButton();
        syncFallbackDownloadWorkspaceButton();

        if (!alreadyActive || previousCategory !== category) {
            announceFallback(copy.alert, 'assertive');
            const reloadButton = bindFallbackReloadButton();
            if (reloadButton && typeof reloadButton.focus === 'function') {
                reloadButton.focus();
            }
        } else {
            bindFallbackReloadButton();
        }
    };

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', applyDomErrorState, { once: true });
    } else {
        applyDomErrorState();
    }

    if (
        category !== 'timeout' &&
        category !== 'webgl' &&
        category !== 'isolation' &&
        assetPath &&
        lookupResourceHttpStatus(assetPath) === 0 &&
        !/\bHTTP\s+[45]\d\d\b/i.test(detailText)
    ) {
        void probeAssetHttpStatus(assetPath).then((status) => {
            if (status >= 400 && !isPageUnloading) {
                showBootstrapRecovery(err, assetPath, phase);
            }
        });
    }
}

function isBootAborted(signal) {
    if (signal && signal.aborted) {
        return true;
    }
    const boot = window.__mathExplorerBoot;
    if (boot && typeof boot.isBootFailed === 'function' && boot.isBootFailed()) {
        return true;
    }
    const root = document.documentElement;
    if (root && root.getAttribute('data-boot-failed') === 'true') {
        return true;
    }
    const errorContainer = document.getElementById('error_container');
    if (
        errorContainer &&
        (errorContainer.getAttribute('data-error-active') === 'true' ||
            (errorContainer.style && errorContainer.style.display === 'flex'))
    ) {
        return true;
    }
    return false;
}

function resolveFallbackTimeout(globalOverrideName, paramName, fallbackMs, minimumMs) {
    const floorMs = typeof minimumMs === 'number' ? minimumMs : 100;
    const overrideVal =
        typeof window !== 'undefined' && globalOverrideName ? window[globalOverrideName] : undefined;
    if (
        typeof overrideVal === 'number' &&
        Number.isFinite(overrideVal) &&
        overrideVal >= floorMs
    ) {
        return Math.floor(overrideVal);
    }
    try {
        const params = new URLSearchParams(window.location.search);
        const raw = params.get(paramName);
        if (raw !== null) {
            const parsed = Number(raw);
            if (Number.isFinite(parsed) && parsed >= floorMs) {
                return Math.floor(parsed);
            }
        }
    } catch (_) {
        // Ignore URL parsing errors
    }
    return fallbackMs;
}

function normalizeContentEncoding(rawHeader) {
    return typeof rawHeader === 'string' ? rawHeader.trim().toLowerCase() : '';
}

function normalizeContentType(rawHeader) {
    if (typeof rawHeader !== 'string') {
        return '';
    }
    return rawHeader.split(';')[0].trim().toLowerCase();
}

function isAllowedWasmContentType(contentType) {
    return (
        !contentType ||
        contentType === 'application/wasm' ||
        contentType === 'application/octet-stream' ||
        contentType === 'binary/octet-stream'
    );
}

function isCompressedContentEncoding(encoding) {
    return Boolean(encoding && encoding !== 'identity');
}

function resolveWireByteTotal(encoding, rawContentLength) {
    const hasContentLength = Number.isFinite(rawContentLength) && rawContentLength > 0;
    if (!isCompressedContentEncoding(encoding)) {
        return hasContentLength ? Math.floor(rawContentLength) : WASM_SOURCE_SIZE;
    }
    if (hasContentLength) {
        return Math.floor(rawContentLength);
    }
    if (/\bbr\b/.test(encoding)) {
        return WASM_BROTLI_SIZE;
    }
    if (/\b(?:x-)?gzip\b/.test(encoding)) {
        return WASM_GZIP_SIZE;
    }
    return 0;
}

function lookupResourceEncodedBodySize(assetPath) {
    if (typeof performance === 'undefined' || typeof performance.getEntriesByType !== 'function') {
        return 0;
    }
    const targetName = assetBaseName(assetPath);
    if (!targetName) {
        return 0;
    }
    const entries = performance.getEntriesByType('resource') || [];
    for (let i = entries.length - 1; i >= 0; i -= 1) {
        const entry = entries[i];
        if (
            entry &&
            typeof entry.name === 'string' &&
            entry.name.indexOf(targetName) !== -1 &&
            typeof entry.encodedBodySize === 'number' &&
            Number.isFinite(entry.encodedBodySize) &&
            entry.encodedBodySize > 0
        ) {
            return Math.floor(entry.encodedBodySize);
        }
    }
    return 0;
}

function appendHeaderBytes(headerBytes, chunk) {
    if (!chunk || headerBytes.length >= 4) {
        return;
    }
    const needed = 4 - headerBytes.length;
    const count = Math.min(needed, chunk.byteLength || chunk.length || 0);
    for (let i = 0; i < count; i += 1) {
        headerBytes.push(chunk[i]);
    }
}

function evaluateStreamCompletion(
    decodedBytes,
    headerBytes,
    expectedIdentityBytes,
    expectedWasmBytes,
    contentType,
) {
    if (decodedBytes <= 0) {
        return {
            status: 'truncated',
            error: new Error(
                'WebAssembly download stream closed early: 0 bytes received (empty response).',
            ),
        };
    }
    if (expectedIdentityBytes > 0 && decodedBytes < expectedIdentityBytes) {
        return {
            status: 'truncated',
            error: new Error(
                `WebAssembly download stream truncated: received ${decodedBytes} of ${expectedIdentityBytes} bytes.`,
            ),
        };
    }
    if (headerBytes.length < 4) {
        return {
            status: 'truncated',
            error: new Error(
                `WebAssembly download stream truncated before 4-byte \\0asm header completed (${decodedBytes} bytes received).`,
            ),
        };
    }
    let hasWasmMagic = true;
    for (let i = 0; i < 4; i += 1) {
        if (headerBytes[i] !== WASM_MAGIC_BYTES[i]) {
            hasWasmMagic = false;
            break;
        }
    }
    if (!hasWasmMagic) {
        if (contentType === 'application/wasm') {
            return { status: 'invalid_header', error: null };
        }
        const mimeLabel = contentType || 'missing Content-Type';
        return {
            status: 'invalid_header',
            error: new Error(
                `Non-WebAssembly response payload (${mimeLabel}) lacks \\0asm binary header.`,
            ),
        };
    }
    if (expectedWasmBytes > 0 && decodedBytes < expectedWasmBytes) {
        return {
            status: 'truncated',
            error: new Error(
                `WebAssembly download stream truncated: received ${decodedBytes} of ${expectedWasmBytes} decoded bytes.`,
            ),
        };
    }
    if (decodedBytes < 8) {
        return {
            status: 'truncated',
            error: new Error(
                `WebAssembly download stream truncated: received ${decodedBytes} bytes (incomplete module header).`,
            ),
        };
    }
    return { status: 'complete', error: null };
}

function scaleInFlightWireProgress(decodedBytes, wireTotal, expectedWasmBytes, isCompressed) {
    if (wireTotal <= 0) {
        return {
            current: isCompressed ? 0 : decodedBytes,
            total: 0,
        };
    }
    const ceiling = wireTotal > 1 ? wireTotal - 1 : 0;
    if (!isCompressed) {
        return {
            current: Math.min(ceiling, decodedBytes),
            total: wireTotal,
        };
    }
    if (expectedWasmBytes <= 0) {
        return {
            current: 0,
            total: 0,
        };
    }
    const ratio = Math.min(1, decodedBytes / expectedWasmBytes);
    const scaledWireOctets = Math.floor(ratio * wireTotal);
    return {
        current: Math.min(ceiling, scaledWireOctets),
        total: wireTotal,
    };
}

function reconcileCompletedWireTotal(source, wireTotal, decodedBytes, isCompressed) {
    const encodedBodySize = lookupResourceEncodedBodySize(source);
    if (isCompressed) {
        if (encodedBodySize > 0) {
            return { total: encodedBodySize, encodedBodySize };
        }
        return { total: wireTotal > 0 ? wireTotal : decodedBytes, encodedBodySize: 0 };
    }
    if (wireTotal > 0) {
        return { total: wireTotal, encodedBodySize };
    }
    if (encodedBodySize > 0) {
        return { total: encodedBodySize, encodedBodySize };
    }
    return { total: decodedBytes, encodedBodySize: 0 };
}

async function fetchWasmWithProgress(source, hooks, signal, recordStreamError) {
    if (isBootAborted(signal)) {
        throw new DOMException('Startup aborted before WebAssembly fetch.', 'AbortError');
    }

    const fetchInit = { mode: 'cors', credentials: 'same-origin' };
    if (signal) {
        fetchInit.signal = signal;
    }

    const response = await fetch(source, fetchInit);
    if (isBootAborted(signal)) {
        if (response.body && typeof response.body.cancel === 'function') {
            void response.body.cancel().catch(() => {});
        }
        throw new DOMException('Startup aborted after WebAssembly response headers.', 'AbortError');
    }

    const status = response.status;
    const statusText = response.statusText;
    if (!response.ok) {
        throw new Error(
            `HTTP ${status} (${statusText || 'Request Failed'}) while fetching ${source}`,
        );
    }

    const contentType = normalizeContentType(response.headers.get('content-type'));
    if (!isAllowedWasmContentType(contentType)) {
        if (response.body && typeof response.body.cancel === 'function') {
            void response.body.cancel().catch(() => {});
        }
        hooks?.cancelPendingAnnouncements?.();
        const mimeErr = new Error(
            `Unexpected Content-Type "${contentType}" while fetching ${source} (expected application/wasm)`,
        );
        if (typeof recordStreamError === 'function') {
            recordStreamError(mimeErr);
        }
        throw mimeErr;
    }

    const contentEncoding = normalizeContentEncoding(response.headers.get('content-encoding'));
    const isCompressed = isCompressedContentEncoding(contentEncoding);
    const rawContentLength = Number(response.headers.get('content-length') || 0);
    const expectedIdentityBytes =
        !isCompressed && Number.isFinite(rawContentLength) && rawContentLength > 0
            ? Math.floor(rawContentLength)
            : 0;
    const expectedWasmBytes =
        WASM_SOURCE_SIZE > 0 ? WASM_SOURCE_SIZE : expectedIdentityBytes;
    const wireTotal = resolveWireByteTotal(contentEncoding, rawContentLength);

    if (!response.body || typeof response.body.getReader !== 'function') {
        const buffer = await response.arrayBuffer();
        const bytes = new Uint8Array(buffer);
        const headerBytes = [];
        appendHeaderBytes(headerBytes, bytes);
        const verdict = evaluateStreamCompletion(
            bytes.byteLength,
            headerBytes,
            expectedIdentityBytes,
            expectedWasmBytes,
            contentType,
        );
        if (verdict.error) {
            hooks?.cancelPendingAnnouncements?.();
            if (typeof recordStreamError === 'function') {
                recordStreamError(verdict.error);
            }
            throw verdict.error;
        }
        if (verdict.status === 'complete') {
            const reconciled = reconcileCompletedWireTotal(
                source,
                wireTotal,
                bytes.byteLength,
                isCompressed,
            );
            hooks?.onProgress?.({ current: reconciled.total, total: reconciled.total });
            window.__mathExplorerBoot?.enterPhase?.('compile', 'Compiling WebAssembly module');
        } else {
            hooks?.cancelPendingAnnouncements?.();
        }
        return new Response(buffer, {
            status,
            statusText,
            headers: response.headers,
        });
    }

    const reader = response.body.getReader();
    const headerBytes = [];
    let decodedBytes = 0;
    let settled = false;

    const stream = new ReadableStream({
        start(controller) {
            const cleanupAbortListener = () => {
                if (signal) {
                    signal.removeEventListener('abort', onAbort);
                }
            };

            function onAbort() {
                if (settled) {
                    return;
                }
                settled = true;
                cleanupAbortListener();
                hooks?.cancelPendingAnnouncements?.();
                const abortErr = new DOMException('WebAssembly stream aborted.', 'AbortError');
                void reader.cancel(abortErr).catch(() => {});
                try {
                    controller.error(abortErr);
                } catch (_) {
                    // Controller already closed or errored
                }
            }

            if (signal) {
                if (signal.aborted) {
                    onAbort();
                    return;
                }
                signal.addEventListener('abort', onAbort, { once: true });
            }

            function push() {
                if (isBootAborted(signal)) {
                    onAbort();
                    return;
                }

                reader
                    .read()
                    .then(({ done, value }) => {
                        if (settled) {
                            return;
                        }
                        if (isBootAborted(signal)) {
                            onAbort();
                            return;
                        }
                        if (done) {
                            settled = true;
                            cleanupAbortListener();
                            const verdict = evaluateStreamCompletion(
                                decodedBytes,
                                headerBytes,
                                expectedIdentityBytes,
                                expectedWasmBytes,
                                contentType,
                            );
                            if (verdict.status === 'truncated') {
                                hooks?.cancelPendingAnnouncements?.();
                                if (typeof recordStreamError === 'function') {
                                    recordStreamError(verdict.error);
                                }
                                controller.error(verdict.error);
                                return;
                            }
                            if (verdict.status === 'invalid_header') {
                                hooks?.cancelPendingAnnouncements?.();
                                if (verdict.error) {
                                    if (typeof recordStreamError === 'function') {
                                        recordStreamError(verdict.error);
                                    }
                                    controller.error(verdict.error);
                                    return;
                                }
                                controller.close();
                                return;
                            }
                            const reconciled = reconcileCompletedWireTotal(
                                source,
                                wireTotal,
                                decodedBytes,
                                isCompressed,
                            );
                            hooks?.onProgress?.({
                                current: reconciled.total,
                                total: reconciled.total,
                            });
                            if (
                                isCompressed &&
                                reconciled.encodedBodySize <= 0 &&
                                typeof window !== 'undefined'
                            ) {
                                window.setTimeout(() => {
                                    if (isBootAborted(signal) || hooks?.isSettled?.()) {
                                        return;
                                    }
                                    const lateEncoded = lookupResourceEncodedBodySize(source);
                                    if (lateEncoded > 0 && lateEncoded !== reconciled.total) {
                                        hooks?.onProgress?.({
                                            current: lateEncoded,
                                            total: lateEncoded,
                                        });
                                    }
                                }, 0);
                            }
                            window.__mathExplorerBoot?.enterPhase?.(
                                'compile',
                                'Compiling WebAssembly module',
                            );
                            controller.close();
                            return;
                        }

                        if (value && value.byteLength > 0) {
                            appendHeaderBytes(headerBytes, value);
                            decodedBytes += value.byteLength;
                            const scaled = scaleInFlightWireProgress(
                                decodedBytes,
                                wireTotal,
                                expectedWasmBytes,
                                isCompressed,
                            );
                            hooks?.onProgress?.(scaled);
                        }
                        controller.enqueue(value);
                        push();
                    })
                    .catch((err) => {
                        if (settled) {
                            return;
                        }
                        settled = true;
                        cleanupAbortListener();
                        hooks?.cancelPendingAnnouncements?.();
                        if (!isBootAborted(signal)) {
                            const streamErr = new Error(
                                `WebAssembly download stream interrupted after ${decodedBytes} bytes: ${formatError(err)}`,
                            );
                            if (typeof recordStreamError === 'function') {
                                recordStreamError(streamErr);
                            }
                            controller.error(streamErr);
                            return;
                        }
                        controller.error(err);
                    });
            }

            push();
        },
        cancel(reason) {
            settled = true;
            hooks?.cancelPendingAnnouncements?.();
            void reader.cancel(reason).catch(() => {});
        },
    });

    return new Response(stream, {
        status,
        statusText,
        headers: response.headers,
    });
}

function attachAssetPath(promise, assetPath) {
    return promise.then(
        (value) => value,
        (error) => Promise.reject({ __startupAssetFailure: true, error, assetPath }),
    );
}

function createStartupProgressProxy(boot, onTransferActivity) {
    let activeHooks = null;
    let cancelled = false;
    let transferComplete = false;
    let maxInFlightBytes = 0;
    let bufferedEvent = null;

    function normalizeProgressEvent(event) {
        const rawCurrent =
            event && Number.isFinite(event.current) && event.current > 0
                ? Math.floor(event.current)
                : 0;
        const rawTotal =
            event && Number.isFinite(event.total) && event.total > 0
                ? Math.floor(event.total)
                : 0;
        if (rawTotal > 0 && rawCurrent >= rawTotal) {
            transferComplete = true;
            maxInFlightBytes = Math.max(maxInFlightBytes, rawTotal);
            return { current: rawTotal, total: rawTotal };
        }
        maxInFlightBytes = Math.max(maxInFlightBytes, rawCurrent);
        return { current: maxInFlightBytes, total: rawTotal };
    }

    return {
        attach(hooksInstance) {
            if (cancelled) {
                hooksInstance?.cancelPendingAnnouncements?.();
                return;
            }
            activeHooks = hooksInstance || null;
            if (activeHooks && bufferedEvent !== null) {
                const pending = bufferedEvent;
                bufferedEvent = null;
                activeHooks.onProgress?.(pending);
            }
        },
        isCancelled() {
            return cancelled;
        },
        isTransferComplete() {
            return transferComplete;
        },
        onProgress(event) {
            if (cancelled) {
                return;
            }
            const normalized = normalizeProgressEvent(event);
            if (typeof onTransferActivity === 'function') {
                onTransferActivity(normalized);
            }
            if (activeHooks) {
                activeHooks.onProgress?.(normalized);
                return;
            }
            bufferedEvent = normalized;
            if (transferComplete) {
                boot?.enterPhase?.('compile', 'Compiling WebAssembly module');
            } else {
                boot?.recordProgress?.(`Downloading WebAssembly (${normalized.current} bytes)`);
            }
        },
        cancelPendingAnnouncements() {
            cancelled = true;
            bufferedEvent = null;
            activeHooks?.cancelPendingAnnouncements?.();
        },
        isSettled() {
            return activeHooks ? Boolean(activeHooks.isSettled?.()) : false;
        },
    };
}

async function bootstrap() {
    markBootInitialized();
    bindFallbackReloadButton();

    const boot = window.__mathExplorerBoot;
    if (!boot) {
        syncShellThemeFromStorage();
    }
    const localAbort =
        !boot && typeof AbortController !== 'undefined' ? new AbortController() : null;
    const signal =
        (boot && typeof boot.getAbortSignal === 'function' && boot.getAbortSignal()) ||
        (localAbort ? localAbort.signal : undefined);

    if (isBootAborted(signal)) {
        return;
    }

    let fallbackTimerId = null;
    let loadingObserver = null;
    let localPhase = 'transfer';
    let localStageLabel = WASM_PATH;

    const clearLocalFallbackTimer = () => {
        if (fallbackTimerId !== null) {
            window.clearTimeout(fallbackTimerId);
            fallbackTimerId = null;
        }
    };

    const disconnectLoadingObserver = () => {
        if (loadingObserver) {
            loadingObserver.disconnect();
            loadingObserver = null;
        }
    };

    const handleBeforeUnload = () => {
        isPageUnloading = true;
        clearLocalFallbackTimer();
        disconnectLoadingObserver();
    };

    const disposeLocalWatchdog = () => {
        clearLocalFallbackTimer();
        disconnectLoadingObserver();
        window.removeEventListener('beforeunload', handleBeforeUnload);
        if (activeWatchdogCleanup === disposeLocalWatchdog) {
            activeWatchdogCleanup = null;
        }
    };

    activeWatchdogCleanup = disposeLocalWatchdog;
    window.addEventListener('beforeunload', handleBeforeUnload);

    if (!boot && typeof MutationObserver !== 'undefined' && document.body) {
        loadingObserver = new MutationObserver(() => {
            if (!document.getElementById('loading_indicator') || isBootAborted(signal)) {
                disposeLocalWatchdog();
            }
        });
        loadingObserver.observe(document.body, { childList: true });
    }

    const armLocalFallbackTimer = (ms, label, phase = 'transfer') => {
        if (boot || !localAbort || isPageUnloading) {
            return;
        }
        localPhase = phase;
        localStageLabel = label;
        if (fallbackTimerId !== null) {
            window.clearTimeout(fallbackTimerId);
        }
        fallbackTimerId = window.setTimeout(() => {
            fallbackTimerId = null;
            if (
                isPageUnloading ||
                !document.getElementById('loading_indicator') ||
                isBootAborted(signal)
            ) {
                disposeLocalWatchdog();
                return;
            }
            const activeStage = localStageLabel || label;
            const err =
                localPhase === 'compile'
                    ? new Error(
                          `WebAssembly compilation or graphics initialization timed out after ${ms} ms (${activeStage}).`,
                      )
                    : new Error(`Startup stalled for ${ms} ms without progress (${activeStage}).`);
            try {
                localAbort.abort(err);
            } catch (_) {
                localAbort.abort();
            }
            showBootstrapRecovery(err, activeStage, localPhase);
        }, ms);
    };

    const stallTimeoutMs = resolveFallbackTimeout(
        '__MATH_EXPLORER_STALL_TIMEOUT_MS',
        'stall_timeout_ms',
        FALLBACK_STALL_TIMEOUT_MS,
        100,
    );
    const compileTimeoutMs = resolveFallbackTimeout(
        '__MATH_EXPLORER_COMPILE_TIMEOUT_MS',
        'compile_timeout_ms',
        FALLBACK_COMPILE_TIMEOUT_MS,
        100,
    );

    let hooks = null;
    let initCompleted = false;
    let currentAssetPath = WASM_PATH;
    let capturedStreamError = null;
    const recordStreamError = (streamErr) => {
        if (streamErr && !capturedStreamError) {
            capturedStreamError = streamErr;
        }
    };

    const hooksProxy = createStartupProgressProxy(boot, (progress) => {
        if (initCompleted) {
            return;
        }
        if (progress.total > 0 && progress.current >= progress.total) {
            armLocalFallbackTimer(compileTimeoutMs, 'Compiling WebAssembly module', 'compile');
        } else {
            armLocalFallbackTimer(stallTimeoutMs, WASM_PATH, 'transfer');
        }
    });

    try {
        boot?.enterPhase?.('transfer', `Downloading ${assetBaseName(WASM_PATH)}`);
        armLocalFallbackTimer(stallTimeoutMs, WASM_PATH, 'transfer');

        const initializerTask = attachAssetPath(
            import(INITIALIZER_PATH).then((initializerModule) => {
                if (!hooksProxy.isCancelled() && !capturedStreamError && !isBootAborted(signal)) {
                    const createInitializer = initializerModule?.default;
                    hooks =
                        typeof createInitializer === 'function'
                            ? createInitializer({ signal })
                            : null;
                    hooks?.onStart?.();
                    hooksProxy.attach(hooks);
                }
                return initializerModule;
            }),
            INITIALIZER_PATH,
        );
        const bindingsTask = attachAssetPath(import(MODULE_PATH), MODULE_PATH);
        const wasmResponseTask = attachAssetPath(
            fetchWasmWithProgress(WASM_PATH, hooksProxy, signal, recordStreamError),
            WASM_PATH,
        );

        const [, bindings, wasmResponse] = await Promise.all([
            initializerTask,
            bindingsTask,
            wasmResponseTask,
        ]);
        if (isBootAborted(signal)) {
            disposeLocalWatchdog();
            return;
        }

        currentAssetPath = MODULE_PATH;
        const init = bindings?.default;
        if (typeof init !== 'function') {
            throw new Error('WebAssembly module initializer export is missing.');
        }

        currentAssetPath = WASM_PATH;
        if (hooksProxy.isTransferComplete()) {
            armLocalFallbackTimer(compileTimeoutMs, 'Compiling WebAssembly module', 'compile');
        }

        const wasm = await init({ module_or_path: wasmResponse });
        initCompleted = true;

        if (isBootAborted(signal)) {
            disposeLocalWatchdog();
            try {
                delete window.wasmBindings;
            } catch (_) {
                window.wasmBindings = undefined;
            }
            return;
        }

        hooks?.onComplete?.();
        hooks?.onSuccess?.(wasm);

        if (isBootAborted(signal)) {
            disposeLocalWatchdog();
            return;
        }

        if (!document.getElementById('loading_indicator')) {
            disposeLocalWatchdog();
        } else if (!boot) {
            localStageLabel = 'Initializing WebGL graphics context';
            if (fallbackTimerId === null || localPhase !== 'compile') {
                armLocalFallbackTimer(
                    compileTimeoutMs,
                    'Initializing WebGL graphics context',
                    'compile',
                );
            }
        }

        window.wasmBindings = bindings;
        window.dispatchEvent(
            new CustomEvent('TrunkApplicationStarted', { detail: { wasm } }),
        );
    } catch (caught) {
        const failedPhase = localPhase;
        disposeLocalWatchdog();
        hooksProxy.cancelPendingAnnouncements();
        hooks?.cancelPendingAnnouncements?.();

        const unwrappedError =
            caught && caught.__startupAssetFailure ? caught.error : caught;
        const failedAssetPath = capturedStreamError
            ? WASM_PATH
            : (caught && caught.__startupAssetFailure && caught.assetPath) || currentAssetPath;
        const effectiveError = capturedStreamError || unwrappedError;

        if (
            !capturedStreamError &&
            isBootAborted(signal) &&
            (unwrappedError === (signal && signal.reason) ||
                (unwrappedError && unwrappedError.name === 'AbortError'))
        ) {
            return;
        }
        if (!window.__mathExplorerBoot && isBootAborted(signal)) {
            return;
        }
        if (localAbort && !localAbort.signal.aborted) {
            try {
                localAbort.abort(effectiveError);
            } catch (_) {
                localAbort.abort();
            }
        }

        console.error('Failed to bootstrap Math Explorer WebAssembly bundle:', effectiveError);

        const activeBoot = window.__mathExplorerBoot;
        const enrichedDetail = enrichAssetError(effectiveError, failedAssetPath);
        if (
            activeBoot &&
            typeof activeBoot.showError === 'function' &&
            typeof activeBoot.classifyError === 'function'
        ) {
            const category = activeBoot.classifyError(effectiveError, failedAssetPath);
            activeBoot.showError(category, enrichedDetail);
            if (lookupResourceHttpStatus(failedAssetPath) === 0) {
                window.setTimeout(() => {
                    const lateDetail = enrichAssetError(effectiveError, failedAssetPath);
                    if (lateDetail !== enrichedDetail) {
                        activeBoot.showError('network', lateDetail);
                        return;
                    }
                    if (category !== 'timeout' && category !== 'webgl' && category !== 'isolation') {
                        void probeAssetHttpStatus(failedAssetPath).then((status) => {
                            if (status >= 400 && !isPageUnloading) {
                                activeBoot.showError(
                                    'network',
                                    enrichAssetError(effectiveError, failedAssetPath),
                                );
                            }
                        });
                    }
                }, 0);
            }
        } else {
            showBootstrapRecovery(effectiveError, failedAssetPath, failedPhase);
        }
    }
}

markBootInitialized();

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        void bootstrap();
    }, { once: true });
} else {
    void bootstrap();
}
