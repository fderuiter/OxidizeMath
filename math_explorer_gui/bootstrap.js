// External WebAssembly bootstrap module for Math Explorer GUI under strict CSP ('self').
// Dynamically imports initializer.js and math_explorer_gui.js, streams WebAssembly download
// progress with AbortSignal cancellation, and transitions cleanly to recovery on failure.

const WASM_SOURCE_SIZE = 0;
const INITIALIZER_PATH = './initializer.js';
const MODULE_PATH = './math_explorer_gui.js';
const WASM_PATH = './math_explorer_gui_bg.wasm';
const FALLBACK_STALL_TIMEOUT_MS = 12000;
const FALLBACK_COMPILE_TIMEOUT_MS = 45000;

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
    },
    compile: {
        title: 'WebAssembly Startup Error',
        message:
            'Your browser downloaded the application bundle, but failed to compile, instantiate, or start the WebAssembly module.',
        heading: 'Recovery Steps',
        steps: [
            'Select Reload Page or perform a hard refresh to bypass stale or corrupted cached files.',
            'Ensure you are using a current browser release with WebAssembly enabled.',
            'Verify that browser extensions or enterprise policies do not block WebAssembly execution.',
        ],
    },
    isolation: {
        title: 'Browser Isolation & Shared Memory Unavailable',
        message:
            'Math Explorer encountered an operation requiring SharedArrayBuffer or browser isolation that is unavailable in this context.',
        heading: 'Recovery Steps',
        steps: [
            'Open Math Explorer over HTTPS or localhost in a standard browser window.',
            'Allow site data and standard WebAssembly memory features, then select Reload Page.',
        ],
    },
    timeout: {
        title: 'Startup Timed Out',
        message:
            'Loading Math Explorer timed out before startup could complete. Select Reload Page to retry.',
        heading: 'Recovery Steps',
        steps: [
            'Confirm your network connection is active and select Reload Page below to retry cleanly.',
        ],
    },
    webgl: {
        title: 'WebGL Initialization Failed',
        message:
            'Math Explorer requires WebGL and hardware acceleration to render interactive simulations.',
        heading: 'Enable Hardware Acceleration & WebGL',
        steps: [],
    },
};

function markBootInitialized() {
    const root = document.documentElement;
    if (root && !root.hasAttribute('data-boot-initialized')) {
        root.setAttribute('data-boot-initialized', 'true');
    }
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
            typeof entry.responseStatus === 'number' &&
            entry.responseStatus >= 400
        ) {
            return entry.responseStatus;
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

function classifyFallbackCategory(err, detailText, assetPath) {
    if (/timeout|timed out|stalled/i.test(detailText)) {
        return 'timeout';
    }
    if (lookupResourceHttpStatus(assetPath) >= 400) {
        return 'network';
    }
    if (
        /http\s+[45]\d\d|status code is not ok|failed to fetch|networkerror|network error|load failed|dynamically imported module|importing a module script failed|failed to load startup|aborted|aborterror/i.test(
            detailText
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

function buildDiagnosticDetails(err, assetPath) {
    if (!err) {
        return '';
    }
    return 'Initialization error: ' + enrichAssetError(err, assetPath);
}

function showBootstrapRecovery(err, assetPath) {
    const detailText = buildDiagnosticDetails(err, assetPath);
    const category = classifyFallbackCategory(err, detailText, assetPath);
    const copy = FALLBACK_CATEGORY_COPY[category] || FALLBACK_CATEGORY_COPY.compile;

    const applyDomErrorState = () => {
        markBootInitialized();
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

        const canvas = document.getElementById('the_canvas_id');
        if (canvas) {
            canvas.setAttribute('style', 'display: none;');
        }

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

        const reloadButton = bindFallbackReloadButton();
        if (reloadButton && typeof reloadButton.focus === 'function') {
            reloadButton.focus();
        }
    };

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', applyDomErrorState, { once: true });
    } else {
        applyDomErrorState();
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

function resolveFallbackTimeout(paramName, fallbackMs) {
    try {
        const params = new URLSearchParams(window.location.search);
        const raw = params.get(paramName);
        if (raw !== null) {
            const parsed = Number(raw);
            if (Number.isFinite(parsed) && parsed >= 100) {
                return Math.floor(parsed);
            }
        }
    } catch (_) {
        // Ignore URL parsing errors
    }
    return fallbackMs;
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

    const contentEncoding = response.headers.get('content-encoding');
    const rawContentLength = Number(response.headers.get('content-length') || 0);
    const total =
        !contentEncoding && Number.isFinite(rawContentLength) && rawContentLength > 0
            ? rawContentLength
            : WASM_SOURCE_SIZE;

    if (!response.body || typeof response.body.getReader !== 'function') {
        window.__mathExplorerBoot?.enterPhase?.('compile', 'Compiling WebAssembly module');
        return response;
    }

    const reader = response.body.getReader();
    let current = 0;
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
                            const completedBytes = total > 0 ? total : current;
                            hooks?.onProgress?.({
                                current: completedBytes,
                                total: total > 0 ? total : completedBytes,
                            });
                            window.__mathExplorerBoot?.enterPhase?.(
                                'compile',
                                'Compiling WebAssembly module',
                            );
                            controller.close();
                            return;
                        }

                        current += value.byteLength;
                        hooks?.onProgress?.({ current, total });
                        controller.enqueue(value);
                        push();
                    })
                    .catch((err) => {
                        if (settled) {
                            return;
                        }
                        settled = true;
                        cleanupAbortListener();
                        controller.error(err);
                    });
            }

            push();
        },
        cancel(reason) {
            settled = true;
            void reader.cancel(reason).catch(() => {});
        },
    });

    return new Response(stream, {
        status,
        statusText,
        headers: response.headers,
    });
}

async function bootstrap() {
    markBootInitialized();
    bindFallbackReloadButton();

    const boot = window.__mathExplorerBoot;
    const localAbort =
        !boot && typeof AbortController !== 'undefined' ? new AbortController() : null;
    const signal =
        (boot && typeof boot.getAbortSignal === 'function' && boot.getAbortSignal()) ||
        (localAbort ? localAbort.signal : undefined);

    if (isBootAborted(signal)) {
        return;
    }

    let fallbackTimerId = null;
    const armLocalFallbackTimer = (ms, label) => {
        if (boot || !localAbort) {
            return;
        }
        if (fallbackTimerId !== null) {
            window.clearTimeout(fallbackTimerId);
        }
        fallbackTimerId = window.setTimeout(() => {
            fallbackTimerId = null;
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

    const clearLocalFallbackTimer = () => {
        if (fallbackTimerId !== null) {
            window.clearTimeout(fallbackTimerId);
            fallbackTimerId = null;
        }
    };

    let hooks = null;
    let currentAssetPath = INITIALIZER_PATH;

    try {
        boot?.enterPhase?.('transfer', `Loading ${assetBaseName(INITIALIZER_PATH)}`);
        armLocalFallbackTimer(
            resolveFallbackTimeout('stall_timeout_ms', FALLBACK_STALL_TIMEOUT_MS),
            INITIALIZER_PATH,
        );

        const initializerModule = await import(INITIALIZER_PATH);
        if (isBootAborted(signal)) {
            clearLocalFallbackTimer();
            return;
        }

        const createInitializer = initializerModule?.default;
        hooks = typeof createInitializer === 'function' ? createInitializer() : null;
        hooks?.onStart?.();

        currentAssetPath = MODULE_PATH;
        boot?.enterPhase?.('transfer', `Loading ${assetBaseName(MODULE_PATH)}`);
        armLocalFallbackTimer(
            resolveFallbackTimeout('stall_timeout_ms', FALLBACK_STALL_TIMEOUT_MS),
            MODULE_PATH,
        );

        const bindings = await import(MODULE_PATH);
        if (isBootAborted(signal)) {
            clearLocalFallbackTimer();
            return;
        }

        const init = bindings.default;
        if (typeof init !== 'function') {
            throw new Error('WebAssembly module initializer export is missing.');
        }

        currentAssetPath = WASM_PATH;
        boot?.enterPhase?.('transfer', `Downloading ${assetBaseName(WASM_PATH)}`);
        armLocalFallbackTimer(
            resolveFallbackTimeout('stall_timeout_ms', FALLBACK_STALL_TIMEOUT_MS),
            WASM_PATH,
        );

        const wasmResponse = await fetchWasmWithProgress(WASM_PATH, hooks, signal);
        if (isBootAborted(signal)) {
            clearLocalFallbackTimer();
            return;
        }

        armLocalFallbackTimer(
            resolveFallbackTimeout('compile_timeout_ms', FALLBACK_COMPILE_TIMEOUT_MS),
            'Compiling WebAssembly module',
        );

        const wasm = await init({ module_or_path: wasmResponse });
        clearLocalFallbackTimer();

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

        if (isBootAborted(signal)) {
            return;
        }

        window.wasmBindings = bindings;
        window.dispatchEvent(
            new CustomEvent('TrunkApplicationStarted', { detail: { wasm } }),
        );
    } catch (error) {
        clearLocalFallbackTimer();

        if (error && error.name === 'AbortError' && isBootAborted(signal)) {
            return;
        }
        if (!window.__mathExplorerBoot && isBootAborted(signal)) {
            return;
        }

        console.error('Failed to bootstrap Math Explorer WebAssembly bundle:', error);

        const activeBoot = window.__mathExplorerBoot;
        const enrichedDetail = enrichAssetError(error, currentAssetPath);
        if (
            activeBoot &&
            typeof activeBoot.showError === 'function' &&
            typeof activeBoot.classifyError === 'function'
        ) {
            const category = activeBoot.classifyError(error, currentAssetPath);
            activeBoot.showError(category, enrichedDetail);
            if (lookupResourceHttpStatus(currentAssetPath) === 0) {
                window.setTimeout(() => {
                    const lateDetail = enrichAssetError(error, currentAssetPath);
                    if (lateDetail !== enrichedDetail) {
                        activeBoot.showError('network', lateDetail);
                    }
                }, 0);
            }
        } else {
            showBootstrapRecovery(error, currentAssetPath);
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
