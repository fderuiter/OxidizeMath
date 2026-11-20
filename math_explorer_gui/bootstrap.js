// External WebAssembly bootstrap module for Math Explorer GUI under strict CSP ('self').
// Streams WebAssembly download progress via initializer.js and transitions cleanly to
// the recovery boundary if initialization fails.
import createInitializer from './initializer.js';

const WASM_SOURCE_SIZE = 0;
const MODULE_PATH = './math_explorer_gui.js';
const WASM_PATH = './math_explorer_gui_bg.wasm';

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

function buildDiagnosticDetails(err) {
    if (!err) {
        return '';
    }
    return 'Initialization error: ' + formatError(err);
}

function showBootstrapRecovery(err) {
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

        const errorContainer = document.getElementById('error_container');
        if (errorContainer) {
            errorContainer.setAttribute('data-error-category', 'compile');
            errorContainer.setAttribute('style', 'display: flex;');
        }

        const titleEl = document.getElementById('error_title');
        if (titleEl) {
            titleEl.textContent = 'WebAssembly Initialization Failed';
        }

        const messageEl = document.getElementById('error_message');
        if (messageEl) {
            messageEl.textContent =
                'Math Explorer could not load or instantiate its WebAssembly engine. Please verify your network connection and browser WebAssembly settings, then reload the page.';
        }

        const detailText = buildDiagnosticDetails(err);
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
                            hooks?.onProgress?.({ current: total > 0 ? total : current, total });
                            controller.close();
                            return;
                        }

                        current += value.byteLength;
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
    });

    return new Response(stream, {
        status,
        statusText,
        headers: response.headers,
    });
}

async function bootstrap() {
    const hooks = typeof createInitializer === 'function' ? createInitializer() : null;
    hooks?.onStart?.();

    try {
        const bindings = await import(MODULE_PATH);
        const init = bindings.default;
        if (typeof init !== 'function') {
            throw new Error('WebAssembly module initializer export is missing.');
        }
        const wasmResponse = await fetchWasmWithProgress(WASM_PATH, hooks);
        const wasm = await init({ module_or_path: wasmResponse });

        hooks?.onComplete?.();
        hooks?.onSuccess?.(wasm);

        window.wasmBindings = bindings;
        window.dispatchEvent(
            new CustomEvent('TrunkApplicationStarted', { detail: { wasm } }),
        );
    } catch (error) {
        console.error('Failed to bootstrap Math Explorer WebAssembly bundle:', error);
        hooks?.onComplete?.();

        const boot = window.__mathExplorerBoot;
        if (boot && typeof boot.showError === 'function' && typeof boot.classifyError === 'function') {
            const category = boot.classifyError(error, WASM_PATH);
            if (hooks && typeof hooks.onFailure === 'function') {
                hooks.onFailure(error);
            } else if (typeof boot.handleFailure === 'function') {
                boot.handleFailure(error, WASM_PATH);
            } else {
                boot.showError(category, buildDiagnosticDetails(error));
            }
        } else {
            showBootstrapRecovery(error);
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
