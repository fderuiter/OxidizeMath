// External WebAssembly bootstrap module for Math Explorer GUI under strict CSP ('self').
// Dynamically imports initializer.js and math_explorer_gui.js, streams WebAssembly download
// progress with AbortSignal cancellation, records boot sentinels, and bridges startup traps.

const WASM_SOURCE_SIZE = 0;
const INITIALIZER_PATH = './initializer.js';
const MODULE_PATH = './math_explorer_gui.js';
const WASM_PATH = './math_explorer_gui_bg.wasm';
const FALLBACK_STALL_TIMEOUT_MS = 12000;
const FALLBACK_COMPILE_TIMEOUT_MS = 45000;
const BOOT_SENTINEL_STORAGE_KEY = 'math_explorer_boot_sentinel';

const FALLBACK_CATEGORY_COPY = {
    runtime: { title: 'Simulation Runtime Error', message: 'Math Explorer halted after an unexpected WebAssembly runtime trap during initialization.' },
    network: { title: 'Network Error Loading Math Explorer', message: 'Math Explorer could not finish downloading its WebAssembly application files.' },
    compile: { title: 'WebAssembly Startup Error', message: 'Your browser downloaded the bundle, but failed to compile or instantiate WebAssembly.' },
    isolation: { title: 'Browser Isolation & Shared Memory Unavailable', message: 'Math Explorer requires standard WebAssembly memory features blocked in this context.' },
    timeout: { title: 'Startup Timed Out', message: 'Loading Math Explorer timed out before startup could complete.' },
    webgl: { title: 'WebGL Initialization Failed', message: 'Math Explorer requires WebGL and hardware acceleration to render simulations.' },
};

function markBootInitialized() {
    const root = document.documentElement;
    if (root && !root.hasAttribute('data-boot-initialized')) {
        root.setAttribute('data-boot-initialized', 'true');
    }
}

function bindFallbackReloadButton() {
    const reloadButton = document.getElementById('webgl_reload_button') || document.querySelector('.reload-btn');
    if (reloadButton && !reloadButton.dataset.reloadBound) {
        reloadButton.dataset.reloadBound = 'true';
        reloadButton.addEventListener('click', (event) => {
            event?.preventDefault?.();
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
    if (err instanceof Error || (typeof err.message === 'string' && err.message.length > 0)) {
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
    const parts = assetPath.replace(/^\.\//, '').split(/[?#]/)[0].split('/');
    return parts[parts.length - 1] || assetPath;
}

function lookupResourceHttpStatus(assetPath) {
    const entry = window.__mathExplorerBoot?.findFailedStartupResource?.(assetPath);
    if (entry && entry.status >= 400) {
        return entry.status;
    }
    const targetName = assetBaseName(assetPath);
    if (!targetName || typeof performance === 'undefined' || typeof performance.getEntriesByType !== 'function') {
        return 0;
    }
    const entries = performance.getEntriesByType('resource') || [];
    for (let i = entries.length - 1; i >= 0; i -= 1) {
        const item = entries[i];
        if (item?.name?.indexOf(targetName) !== -1 && typeof item.responseStatus === 'number' && item.responseStatus >= 400) {
            return item.responseStatus;
        }
    }
    return 0;
}

function enrichAssetError(err, assetPath) {
    const baseText = formatError(err);
    const name = assetBaseName(assetPath);
    const statusCode = lookupResourceHttpStatus(assetPath);
    if (statusCode >= 400 && name) {
        const statusToken = `HTTP ${statusCode}`;
        return baseText.indexOf(statusToken) !== -1 && baseText.indexOf(name) !== -1
            ? baseText
            : `Failed to load startup asset ${name} (${statusToken}): ${baseText}`;
    }
    if (name && baseText.indexOf(name) === -1 && /failed to fetch|dynamically imported module|importing a module|load failed|networkerror/i.test(baseText)) {
        return `Failed to load startup asset ${name}: ${baseText}`;
    }
    return baseText;
}

function classifyFallbackCategory(err, detailText, assetPath) {
    if (/timeout|timed out|stalled/i.test(detailText)) {
        return 'timeout';
    }
    if (lookupResourceHttpStatus(assetPath) >= 400 || /http\s+[45]\d\d|status code is not ok|failed to fetch|networkerror|network error|load failed|dynamically imported module|importing a module script failed|failed to load startup|aborted|aborterror/i.test(detailText)) {
        return 'network';
    }
    if (/webgl|hardware acceleration|rendering context/i.test(detailText)) {
        return 'webgl';
    }
    if (/sharedarraybuffer|cross-origin isolat|crossoriginisolated/i.test(detailText)) {
        return 'isolation';
    }
    if ((typeof WebAssembly !== 'undefined' && (err instanceof WebAssembly.CompileError || err instanceof WebAssembly.LinkError)) || err instanceof SyntaxError) {
        return 'compile';
    }
    if ((typeof WebAssembly !== 'undefined' && err instanceof WebAssembly.RuntimeError) || /runtimeerror|unreachable|out of bounds|panicked|thread panicked/i.test(detailText)) {
        return 'runtime';
    }
    return 'compile';
}

function recordBootSentinelBeforeStart() {
    window.__mathExplorerRecovery?.quarantineAbortedBoot?.();
    if (window.__mathExplorerBoot?.markBootStarting) {
        window.__mathExplorerBoot.markBootStarting();
        return;
    }
    try {
        window.localStorage?.setItem(
            BOOT_SENTINEL_STORAGE_KEY,
            JSON.stringify({ status: 'starting', started_at: Date.now() }),
        );
    } catch (_) {
        // Ignore restricted storage environments
    }
}

function applyDomFallbackRecovery(category, copy, detailText) {
    markBootInitialized();
    document.documentElement?.setAttribute('data-boot-failed', 'true');
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
    document.getElementById('the_canvas_id')?.setAttribute('style', 'display: none;');
    const errorContainer = document.getElementById('error_container');
    if (errorContainer) {
        errorContainer.setAttribute('data-error-category', category);
        errorContainer.setAttribute('data-error-active', 'true');
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
    const detailsEl = document.getElementById('error_details');
    if (detailsEl && detailText) {
        detailsEl.textContent = detailText;
        detailsEl.setAttribute('style', 'display: block;');
    }
    bindFallbackReloadButton()?.focus?.();
}

function showBootstrapRecovery(err, assetPath) {
    const detailText = err ? 'Initialization error: ' + enrichAssetError(err, assetPath) : '';
    const category = classifyFallbackCategory(err, detailText, assetPath);
    if (window.__mathExplorerRecovery?.renderErrorBoundary) {
        window.__mathExplorerRecovery.renderErrorBoundary(category, detailText, 'compile');
        return;
    }
    const copy = FALLBACK_CATEGORY_COPY[category] || FALLBACK_CATEGORY_COPY.compile;
    const apply = () => applyDomFallbackRecovery(category, copy, detailText);
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', apply, { once: true });
    } else {
        apply();
    }
}

function isBootAborted(signal) {
    if (signal?.aborted || window.__mathExplorerBoot?.isBootFailed?.()) {
        return true;
    }
    if (document.documentElement?.getAttribute('data-boot-failed') === 'true') {
        return true;
    }
    const errorContainer = document.getElementById('error_container');
    return Boolean(errorContainer && (errorContainer.getAttribute('data-error-active') === 'true' || errorContainer.style?.display === 'flex'));
}

function resolveFallbackTimeout(paramName, fallbackMs) {
    try {
        const raw = new URLSearchParams(window.location.search).get(paramName);
        const parsed = raw !== null ? Number(raw) : NaN;
        if (Number.isFinite(parsed) && parsed >= 100) {
            return Math.floor(parsed);
        }
    } catch (_) {
        // Ignore URL parsing errors
    }
    return fallbackMs;
}

function pumpWasmReader(reader, controller, total, hooks, signal, onAbort, isSettled, markDone) {
    let current = 0;
    function push() {
        if (isBootAborted(signal)) {
            onAbort();
            return;
        }
        reader.read().then(({ done, value }) => {
            if (isSettled() || isBootAborted(signal)) {
                if (!isSettled()) {
                    onAbort();
                }
                return;
            }
            if (done) {
                markDone();
                const completed = total > 0 ? total : current;
                hooks?.onProgress?.({ current: completed, total: total > 0 ? total : completed });
                window.__mathExplorerBoot?.enterPhase?.('compile', 'Compiling WebAssembly module');
                controller.close();
                return;
            }
            current += value.byteLength;
            hooks?.onProgress?.({ current, total });
            controller.enqueue(value);
            push();
        }).catch((err) => {
            if (!isSettled()) {
                markDone();
                controller.error(err);
            }
        });
    }
    push();
}

function createWasmProgressStream(reader, total, hooks, signal) {
    let settled = false;
    return new ReadableStream({
        start(controller) {
            const cleanup = () => signal?.removeEventListener('abort', onAbort);
            const markDone = () => {
                settled = true;
                cleanup();
            };
            function onAbort() {
                if (settled) {
                    return;
                }
                markDone();
                const abortErr = new DOMException('WebAssembly stream aborted.', 'AbortError');
                void reader.cancel(abortErr).catch(() => {});
                try {
                    controller.error(abortErr);
                } catch (_) {
                    // Stream already closed
                }
            }
            if (signal?.aborted) {
                onAbort();
                return;
            }
            signal?.addEventListener('abort', onAbort, { once: true });
            pumpWasmReader(reader, controller, total, hooks, signal, onAbort, () => settled, markDone);
        },
        cancel(reason) {
            settled = true;
            void reader.cancel(reason).catch(() => {});
        },
    });
}

async function fetchWasmWithProgress(source, hooks, signal) {
    if (isBootAborted(signal)) {
        throw new DOMException('Startup aborted before WebAssembly fetch.', 'AbortError');
    }
    const fetchInit = { mode: 'cors', credentials: 'same-origin' };
    if (signal) {
        fetchInit.signal = signal;
    }
    const response = await fetch(source, fetchInit);
    if (isBootAborted(signal)) {
        void response.body?.cancel?.().catch(() => {});
        throw new DOMException('Startup aborted after WebAssembly response headers.', 'AbortError');
    }
    if (!response.ok) {
        throw new Error(`HTTP ${response.status} (${response.statusText || 'Request Failed'}) while fetching ${source}`);
    }
    const rawLen = Number(response.headers.get('content-length') || 0);
    const total = !response.headers.get('content-encoding') && Number.isFinite(rawLen) && rawLen > 0 ? rawLen : WASM_SOURCE_SIZE;
    if (!response.body || typeof response.body.getReader !== 'function') {
        window.__mathExplorerBoot?.enterPhase?.('compile', 'Compiling WebAssembly module');
        return response;
    }
    return new Response(createWasmProgressStream(response.body.getReader(), total, hooks, signal), {
        status: response.status,
        statusText: response.statusText,
        headers: response.headers,
    });
}

function handleBootstrapCatch(error, signal, currentAssetPath) {
    if (
        (isBootAborted(signal) && (error === signal?.reason || error?.name === 'AbortError')) ||
        (!window.__mathExplorerBoot && isBootAborted(signal))
    ) {
        return;
    }
    console.error('Failed to bootstrap Math Explorer WebAssembly bundle:', error);
    const activeBoot = window.__mathExplorerBoot;
    const enrichedDetail = enrichAssetError(error, currentAssetPath);
    if (activeBoot?.showError && activeBoot?.classifyError) {
        const category = activeBoot.classifyError(error, currentAssetPath);
        activeBoot.showError(category, enrichedDetail);
        if (lookupResourceHttpStatus(currentAssetPath) === 0) {
            window.setTimeout(() => {
                const lateDetail = enrichAssetError(error, currentAssetPath);
                if (lateDetail !== enrichedDetail || lookupResourceHttpStatus(currentAssetPath) >= 400) {
                    activeBoot.showError('network', lateDetail);
                }
            }, 0);
        }
        return;
    }
    showBootstrapRecovery(error, currentAssetPath);
}

function createLocalWatchdog(boot, localAbort) {
    let timerId = null;
    const clear = () => {
        if (timerId !== null) {
            window.clearTimeout(timerId);
            timerId = null;
        }
    };
    const arm = (ms, label) => {
        if (boot || !localAbort) {
            return;
        }
        clear();
        timerId = window.setTimeout(() => {
            timerId = null;
            if (!document.getElementById('loading_indicator')) {
                return;
            }
            const err = new Error(`Startup stalled for ${ms} ms without progress (${label}).`);
            try {
                localAbort.abort(err);
            } catch (_) {
                localAbort.abort();
            }
            showBootstrapRecovery(err, label);
        }, ms);
    };
    return { arm, clear };
}

async function instantiateAndStartBundle(bindings, wasmResponse, hooks, watchdog, signal) {
    recordBootSentinelBeforeStart();
    watchdog.arm(resolveFallbackTimeout('compile_timeout_ms', FALLBACK_COMPILE_TIMEOUT_MS), 'Compiling WebAssembly module');
    const wasm = await bindings.default({ module_or_path: wasmResponse });
    watchdog.clear();
    if (isBootAborted(signal)) {
        try {
            delete window.wasmBindings;
        } catch (_) {
            window.wasmBindings = undefined;
        }
        return;
    }
    hooks?.onComplete?.();
    hooks?.onSuccess?.(wasm);
    if (!isBootAborted(signal)) {
        window.wasmBindings = bindings;
        window.dispatchEvent(new CustomEvent('TrunkApplicationStarted', { detail: { wasm } }));
    }
}

async function bootstrap() {
    markBootInitialized();
    bindFallbackReloadButton();
    const boot = window.__mathExplorerBoot;
    const localAbort = !boot && typeof AbortController !== 'undefined' ? new AbortController() : null;
    const signal = boot?.getAbortSignal?.() || (localAbort ? localAbort.signal : undefined);
    if (isBootAborted(signal)) {
        return;
    }
    const watchdog = createLocalWatchdog(boot, localAbort);
    let currentAssetPath = INITIALIZER_PATH;
    try {
        boot?.enterPhase?.('transfer', `Loading ${assetBaseName(INITIALIZER_PATH)}`);
        watchdog.arm(resolveFallbackTimeout('stall_timeout_ms', FALLBACK_STALL_TIMEOUT_MS), INITIALIZER_PATH);
        const initializerModule = await import(INITIALIZER_PATH);
        if (isBootAborted(signal)) {
            watchdog.clear();
            return;
        }
        const hooks = typeof initializerModule?.default === 'function' ? initializerModule.default() : null;
        hooks?.onStart?.();

        currentAssetPath = MODULE_PATH;
        boot?.enterPhase?.('transfer', `Loading ${assetBaseName(MODULE_PATH)}`);
        watchdog.arm(resolveFallbackTimeout('stall_timeout_ms', FALLBACK_STALL_TIMEOUT_MS), MODULE_PATH);
        const bindings = await import(MODULE_PATH);
        if (isBootAborted(signal)) {
            watchdog.clear();
            return;
        }
        if (typeof bindings.default !== 'function') {
            throw new Error('WebAssembly module initializer export is missing.');
        }

        currentAssetPath = WASM_PATH;
        boot?.enterPhase?.('transfer', `Downloading ${assetBaseName(WASM_PATH)}`);
        watchdog.arm(resolveFallbackTimeout('stall_timeout_ms', FALLBACK_STALL_TIMEOUT_MS), WASM_PATH);
        const wasmResponse = await fetchWasmWithProgress(WASM_PATH, hooks, signal);
        if (isBootAborted(signal)) {
            watchdog.clear();
            return;
        }
        await instantiateAndStartBundle(bindings, wasmResponse, hooks, watchdog, signal);
    } catch (error) {
        watchdog.clear();
        handleBootstrapCatch(error, signal, currentAssetPath);
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
