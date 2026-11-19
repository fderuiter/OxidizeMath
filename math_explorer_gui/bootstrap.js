// External WebAssembly bootstrap module for Math Explorer GUI under strict CSP ('self').
// Delays ES module and WebAssembly requests while cross-origin isolation service worker
// registration is pending, streams download progress via initializer.js once ready, and
// transitions cleanly to the recovery boundary if initialization fails.
import createInitializer from './initializer.js';

const WASM_SOURCE_SIZE = 0;
const ISOLATION_TIMEOUT_MS = 5000;
const MODULE_PATH = './math_explorer_gui.js';
const WASM_PATH = './math_explorer_gui_bg.wasm';

const loadingStages = [];
window.__mathExplorerLoadingStages = loadingStages;

function recordLoadingStage(stage, loadedBytes, totalBytes) {
    const progress =
        typeof loadedBytes === 'number' && typeof totalBytes === 'number' && totalBytes > 0
            ? Math.min(100, Math.floor((loadedBytes * 100) / totalBytes))
            : null;
    loadingStages.push({
        stage,
        progress,
        loadedBytes: typeof loadedBytes === 'number' ? loadedBytes : null,
    });
}

function formatError(err) {
    if (!err) {
        return 'Unknown initialization error.';
    }
    if (typeof err === 'string') {
        return err;
    }
    if (err instanceof Error) {
        return err.stack || (err.name + ': ' + err.message);
    }
    try {
        return JSON.stringify(err);
    } catch (_) {
        return String(err);
    }
}

function buildDiagnosticDetails(err, isolationDecision) {
    const detailsLines = [];
    if (!window.crossOriginIsolated) {
        if (isolationDecision && isolationDecision.reason) {
            detailsLines.push('Isolation status: ' + isolationDecision.reason);
        } else {
            detailsLines.push('Isolation status: window.crossOriginIsolated is false.');
        }
    }
    if (err) {
        detailsLines.push('Initialization error: ' + formatError(err));
    }
    return detailsLines.join('\n');
}

function showBootstrapRecovery(err, isolationDecision) {
    const applyDomErrorState = () => {
        const loading = document.getElementById('loading_indicator');
        if (loading) {
            loading.setAttribute('aria-busy', 'false');
            loading.remove();
        }

        const canvas = document.getElementById('the_canvas_id');
        if (canvas) {
            canvas.setAttribute('style', 'display: none;');
        }

        const isolationUnavailable = !window.crossOriginIsolated;
        const errorContainer = document.getElementById('error_container');
        if (errorContainer) {
            errorContainer.setAttribute(
                'data-error-category',
                isolationUnavailable ? 'isolation' : 'compile',
            );
            errorContainer.setAttribute('style', 'display: flex;');
        }

        const titleEl = document.getElementById('error_title');
        if (titleEl) {
            titleEl.textContent = isolationUnavailable
                ? 'Cross-Origin Isolation & WebAssembly Initialization Failed'
                : 'WebAssembly Initialization Failed';
        }

        const messageEl = document.getElementById('error_message');
        if (messageEl) {
            messageEl.textContent = isolationUnavailable
                ? 'Math Explorer could not initialize its WebAssembly engine because cross-origin isolation (Service Workers or SharedArrayBuffer) is unavailable or blocked in this browser environment.'
                : 'Math Explorer could not load or instantiate its WebAssembly engine. Please verify your network connection and browser WebAssembly settings, then reload the page.';
        }

        const detailText = buildDiagnosticDetails(err, isolationDecision);
        const detailsEl = document.getElementById('error_details');
        if (detailsEl && detailText) {
            detailsEl.textContent = detailText;
            detailsEl.setAttribute('style', 'display: block;');
        }

        const reloadButton = document.getElementById('webgl_reload_button');
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

async function resolveIsolationGate() {
    if (window.crossOriginIsolated) {
        return { status: 'isolated', reason: null };
    }

    const controller = window.__mathExplorerIsolation;
    if (!controller) {
        return {
            status: 'unavailable',
            reason: 'Cross-origin isolation service worker controller did not initialize.',
        };
    }

    if (controller.status === 'reloading') {
        return { status: 'reloading', reason: controller.reason };
    }

    if (controller.status === 'isolated' || controller.status === 'unavailable') {
        return { status: controller.status, reason: controller.reason };
    }

    recordLoadingStage('activating-isolation', 0, WASM_SOURCE_SIZE);
    if (window.__mathExplorerBoot && typeof window.__mathExplorerBoot.recordProgress === 'function') {
        window.__mathExplorerBoot.recordProgress('Activating cross-origin isolation service worker');
    }

    let timeoutId = null;
    const timeoutPromise = new Promise((resolve) => {
        timeoutId = window.setTimeout(() => {
            resolve({
                status: 'unavailable',
                reason: 'Timed out waiting for cross-origin isolation service worker activation.',
            });
        }, ISOLATION_TIMEOUT_MS);
    });

    try {
        const decision = await Promise.race([
            controller.waitForDecision(),
            timeoutPromise,
        ]);
        if (controller.status === 'reloading') {
            return { status: 'reloading', reason: controller.reason };
        }
        return decision;
    } finally {
        if (timeoutId !== null) {
            window.clearTimeout(timeoutId);
        }
    }
}

async function fetchWasmWithProgress(source, hooks) {
    const response = await fetch(source, { mode: 'cors', credentials: 'same-origin' });
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
        recordLoadingStage('compiling-wasm', total, total);
        return response;
    }

    const reader = response.body.getReader();
    let current = 0;

    const stream = new ReadableStream({
        start(controller) {
            function push() {
                reader
                    .read()
                    .then(({ done, value }) => {
                        if (done) {
                            const finalBytes = total > 0 ? total : current;
                            recordLoadingStage('compiling-wasm', finalBytes, total);
                            hooks?.onProgress?.({ current: finalBytes, total });
                            controller.close();
                            return;
                        }

                        current += value.byteLength;
                        recordLoadingStage('downloading-wasm', current, total);
                        hooks?.onProgress?.({ current, total });
                        controller.enqueue(value);
                        push();
                    })
                    .catch((err) => {
                        controller.error(err);
                    });
            }

            push();
        },
        cancel(reason) {
            return reader.cancel(reason);
        },
    });

    const headers = new Headers(response.headers);
    if (!headers.has('Content-Type')) {
        headers.set('Content-Type', 'application/wasm');
    }

    return new Response(stream, {
        status,
        statusText,
        headers,
    });
}

async function bootstrap() {
    const decision = await resolveIsolationGate();

    // When a service worker activation reload is scheduled, exit cleanly
    // so zero module or WebAssembly transfers start or abort during reload.
    if (
        decision.status === 'reloading' ||
        (window.__mathExplorerIsolation && window.__mathExplorerIsolation.status === 'reloading')
    ) {
        if (window.__mathExplorerBoot && typeof window.__mathExplorerBoot.clearWatchdog === 'function') {
            window.__mathExplorerBoot.clearWatchdog();
        }
        return;
    }

    const hooks = typeof createInitializer === 'function' ? createInitializer() : null;
    hooks?.onStart?.();

    try {
        recordLoadingStage('loading-js', 0, WASM_SOURCE_SIZE);
        const bindings = await import(MODULE_PATH);
        const init = bindings.default;
        if (typeof init !== 'function') {
            throw new Error('WebAssembly module initializer export is missing.');
        }

        recordLoadingStage('fetching-wasm', 0, WASM_SOURCE_SIZE);
        const wasmResponse = await fetchWasmWithProgress(WASM_PATH, hooks);
        const wasm = await init({ module_or_path: wasmResponse });

        hooks?.onComplete?.();
        hooks?.onSuccess?.(wasm);

        window.wasmBindings = bindings;
        window.dispatchEvent(
            new CustomEvent('TrunkApplicationStarted', { detail: { wasm } }),
        );
    } catch (error) {
        if (
            window.__mathExplorerIsolation &&
            window.__mathExplorerIsolation.status === 'reloading'
        ) {
            return;
        }
        console.error('Failed to bootstrap Math Explorer WebAssembly bundle:', error);
        hooks?.onComplete?.();

        const boot = window.__mathExplorerBoot;
        if (boot && typeof boot.showError === 'function' && typeof boot.classifyError === 'function') {
            const category = boot.classifyError(error, WASM_PATH);
            if (!window.crossOriginIsolated && category !== 'network' && category !== 'timeout') {
                boot.showError('isolation', buildDiagnosticDetails(error, decision));
            } else if (hooks && typeof hooks.onFailure === 'function') {
                hooks.onFailure(error);
            } else if (typeof boot.handleFailure === 'function') {
                boot.handleFailure(error, WASM_PATH);
            } else {
                boot.showError(category, buildDiagnosticDetails(error, decision));
            }
        } else {
            showBootstrapRecovery(error, decision);
        }
    }
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        void bootstrap();
    }, { once: true });
} else {
    void bootstrap();
}
