// External bootstrap controller for Math Explorer GUI under strict CSP ('self').
// Delays ES module and WebAssembly requests while cross-origin isolation
// service worker registration is pending, proceeds once isolation is active
// or unavailable, and transitions to the recovery boundary if startup fails.
(function () {
    const ISOLATION_TIMEOUT_MS = 5000;
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

    function updateRecoveryContentForBootstrapFailure() {
        const container = document.getElementById('error_container');
        if (!container) {
            return;
        }

        const titleEl = container.querySelector('.error-title');
        const messageEl = container.querySelector('.error-message');
        const instructionsCard = container.querySelector('.instructions-card');
        const isolationUnavailable = !window.crossOriginIsolated;

        if (titleEl) {
            titleEl.textContent = isolationUnavailable
                ? 'Cross-Origin Isolation & WebAssembly Initialization Failed'
                : 'WebAssembly Initialization Failed';
        }

        if (messageEl) {
            messageEl.textContent = isolationUnavailable
                ? 'Math Explorer could not initialize its WebAssembly engine because cross-origin isolation (Service Workers or SharedArrayBuffer) is unavailable or blocked in this browser environment.'
                : 'Math Explorer could not load or instantiate its WebAssembly engine. Please verify your network connection and browser WebAssembly settings, then reload the page.';
        }

        if (instructionsCard && isolationUnavailable) {
            instructionsCard.replaceChildren();

            const heading = document.createElement('h2');
            heading.textContent = 'Restore Browser Isolation & Site Data Support';
            instructionsCard.appendChild(heading);

            const grid = document.createElement('div');
            grid.className = 'browser-grid';

            const steps = [
                {
                    title: 'Private / Incognito & Restricted Modes',
                    items: [
                        'Open Math Explorer in a standard browser window instead of Private or Incognito mode (Firefox Private Browsing and restricted WebViews disable Service Workers).',
                        'Ensure you are accessing the application over HTTPS or localhost so secure-context APIs are enabled.'
                    ]
                },
                {
                    title: 'Cookies, Site Data & Content Blockers',
                    items: [
                        'Allow site data and Service Worker registration for this site in your browser privacy settings.',
                        'Temporarily disable extensions or enterprise policies that block Service Workers, SharedArrayBuffer, or WebAssembly compilation, then click Reload Page.'
                    ]
                }
            ];

            for (const section of steps) {
                const itemDiv = document.createElement('div');
                itemDiv.className = 'browser-item';

                const subHeading = document.createElement('h3');
                subHeading.textContent = section.title;
                itemDiv.appendChild(subHeading);

                const list = document.createElement('ol');
                for (const text of section.items) {
                    const li = document.createElement('li');
                    li.textContent = text;
                    list.appendChild(li);
                }
                itemDiv.appendChild(list);
                grid.appendChild(itemDiv);
            }

            instructionsCard.appendChild(grid);
        }
    }

    function showBootstrapRecovery(err, isolationDecision) {
        const applyDomErrorState = () => {
            const loading = document.getElementById('loading_indicator');
            if (loading) {
                loading.remove();
            }

            const canvas = document.getElementById('the_canvas_id');
            if (canvas) {
                canvas.setAttribute('style', 'display: none;');
            }

            updateRecoveryContentForBootstrapFailure();

            const errorContainer = document.getElementById('error_container');
            if (errorContainer) {
                errorContainer.setAttribute('style', 'display: flex;');
            }

            const detailsLines = [];
            if (isolationDecision && isolationDecision.reason) {
                detailsLines.push('Isolation status: ' + isolationDecision.reason);
            } else if (!window.crossOriginIsolated) {
                detailsLines.push('Isolation status: window.crossOriginIsolated is false.');
            }
            if (err) {
                detailsLines.push('Initialization error: ' + formatError(err));
            }

            const detailsEl = document.getElementById('error_details');
            if (detailsEl && detailsLines.length > 0) {
                detailsEl.textContent = detailsLines.join('\n');
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
                reason: 'Cross-origin isolation service worker controller did not initialize.'
            };
        }

        if (controller.status === 'reloading') {
            return { status: 'reloading', reason: controller.reason };
        }

        if (controller.status === 'isolated' || controller.status === 'unavailable') {
            return { status: controller.status, reason: controller.reason };
        }

        let timeoutId = null;
        const timeoutPromise = new Promise((resolve) => {
            timeoutId = window.setTimeout(() => {
                resolve({
                    status: 'unavailable',
                    reason: 'Timed out waiting for cross-origin isolation service worker activation.'
                });
            }, ISOLATION_TIMEOUT_MS);
        });

        try {
            const decision = await Promise.race([
                controller.waitForDecision(),
                timeoutPromise
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

    async function bootstrapApplication() {
        const decision = await resolveIsolationGate();

        // When a service worker activation reload is scheduled, exit cleanly
        // so zero module or WebAssembly transfers start or abort during reload.
        if (
            decision.status === 'reloading' ||
            (window.__mathExplorerIsolation && window.__mathExplorerIsolation.status === 'reloading')
        ) {
            return;
        }

        try {
            const bindings = await import(MODULE_PATH);
            const init = bindings.default;
            if (typeof init !== 'function') {
                throw new Error('WebAssembly module initializer export is missing.');
            }
            const wasm = await init({ module_or_path: WASM_PATH });
            window.wasmBindings = bindings;
            window.dispatchEvent(
                new CustomEvent('TrunkApplicationStarted', { detail: { wasm } })
            );
        } catch (err) {
            console.error('Failed to bootstrap Math Explorer WebAssembly bundle:', err);
            showBootstrapRecovery(err, decision);
        }
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => {
            void bootstrapApplication();
        }, { once: true });
    } else {
        void bootstrapApplication();
    }
})();
