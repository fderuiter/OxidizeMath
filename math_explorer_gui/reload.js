// Pre-boot theme sync, error boundary, stall watchdog, ARIA announcer, and reload handler for Math Explorer GUI
(function () {
    const DEFAULT_STALL_TIMEOUT_MS = 12000;
    const SHELL_THEME_TOKENS = {
        dark: { colorScheme: 'dark', themeColor: '#121214' },
        light: { colorScheme: 'light', themeColor: '#f8f8f8' },
        'high-contrast': { colorScheme: 'dark', themeColor: '#000000' }
    };

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
        } catch (_err) {
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

    syncShellThemeFromStorage();

    const ERROR_CATEGORIES = {
        network: {
            title: 'Network Error Loading Math Explorer',
            message:
                'Math Explorer could not finish downloading its WebAssembly application files. Your connection may have been interrupted or the server returned an error.',
            heading: 'Recovery Steps',
            steps: [
                'Check your internet connection and verify that firewalls, proxies, or content blockers allow .js and .wasm requests.',
                'Select Reload Page below (or perform a hard refresh) to retry downloading the application.',
                'If the server is temporarily unavailable, wait a moment and reload.'
            ],
            alert:
                'Network error loading Math Explorer. Check your internet connection and select Reload Page to retry.'
        },
        compile: {
            title: 'WebAssembly Startup Error',
            message:
                'Your browser downloaded the application bundle, but failed to compile or instantiate the WebAssembly module. Cached files may be incomplete or corrupted.',
            heading: 'Recovery Steps',
            steps: [
                'Select Reload Page or perform a hard refresh (Ctrl+Shift+R / Cmd+Shift+R) to bypass stale or corrupted cached files.',
                'Ensure you are using a current release of Chrome, Edge, Firefox, or Safari with WebAssembly enabled.',
                'Verify that browser security extensions or enterprise policies do not block WebAssembly compilation.'
            ],
            alert:
                'WebAssembly startup error. Cached files may be corrupted. Select Reload Page to retry.'
        },
        timeout: {
            title: 'Startup Timed Out',
            message:
                'Loading Math Explorer stalled without progress before startup could complete. The network transfer may have frozen or stopped responding.',
            heading: 'Recovery Steps',
            steps: [
                'Confirm your network connection is active and not waiting on a captive portal or VPN.',
                'Select Reload Page below to restart the WebAssembly download cleanly.',
                'If you are on a slow or metered connection, pause competing downloads and try again.'
            ],
            alert:
                'Loading Math Explorer timed out due to a stalled transfer. Select Reload Page to retry.'
        },
        webgl: {
            title: 'WebGL Initialization Failed',
            message:
                'Math Explorer requires WebGL and hardware acceleration to render mathematics and interactive simulations. Your browser was unable to initialize a WebGL rendering context.',
            heading: 'Enable Hardware Acceleration & WebGL',
            steps: [],
            alert:
                'WebGL initialization failed. Please enable hardware acceleration in your browser settings.'
        }
    };

    // Start pre-boot twin-slot toggles at slot 1 so slot 0 starts empty when Rust's AtomicBool(false) takes over.
    let politeSlotToggle = 1;
    let assertiveSlotToggle = 1;
    let bootFailed = false;
    let isUnloading = false;
    let stallTimerId = null;
    let lastStageLabel = 'Initializing startup scripts';
    let pendingActions = [];

    function resolveStallTimeoutMs() {
        if (
            typeof window.__MATH_EXPLORER_STALL_TIMEOUT_MS === 'number' &&
            Number.isFinite(window.__MATH_EXPLORER_STALL_TIMEOUT_MS) &&
            window.__MATH_EXPLORER_STALL_TIMEOUT_MS >= 100
        ) {
            return Math.floor(window.__MATH_EXPLORER_STALL_TIMEOUT_MS);
        }
        try {
            const params = new URLSearchParams(window.location.search);
            const raw = params.get('stall_timeout_ms');
            if (raw !== null) {
                const parsed = Number(raw);
                if (Number.isFinite(parsed) && parsed >= 100) {
                    return Math.floor(parsed);
                }
            }
        } catch (_err) {
            // Ignore URL parsing errors in constrained environments
        }
        return DEFAULT_STALL_TIMEOUT_MS;
    }

    function runWhenDomReady(fn) {
        if (document.readyState === 'loading') {
            pendingActions.push(fn);
        } else {
            fn();
        }
    }

    function isBootPending() {
        if (bootFailed) {
            return false;
        }
        if (document.readyState === 'loading') {
            return true;
        }
        const loading = document.getElementById('loading_indicator');
        if (!loading) {
            return false;
        }
        if (loading.hidden || loading.style.display === 'none') {
            return false;
        }
        return loading.getAttribute('aria-busy') !== 'false';
    }

    function clearWatchdog() {
        if (stallTimerId !== null) {
            window.clearTimeout(stallTimerId);
            stallTimerId = null;
        }
    }

    function announce(message, priority) {
        if (!message) {
            return;
        }
        runWhenDomReady(function () {
            const isAssertive =
                String(priority).toLowerCase() === 'assertive' ||
                String(priority).toLowerCase() === 'alert';
            const containerId = isAssertive ? 'aria-live-assertive' : 'aria-live-polite';
            const slotIdx = isAssertive ? assertiveSlotToggle : politeSlotToggle;
            if (isAssertive) {
                assertiveSlotToggle = 1 - assertiveSlotToggle;
            } else {
                politeSlotToggle = 1 - politeSlotToggle;
            }
            const siblingIdx = 1 - slotIdx;
            const activeSlot = document.getElementById(containerId + '-slot-' + slotIdx);
            const siblingSlot = document.getElementById(containerId + '-slot-' + siblingIdx);
            if (siblingSlot) {
                siblingSlot.textContent = '';
            }
            if (activeSlot) {
                activeSlot.textContent = String(message);
            }
        });
    }

    function formatReason(reason) {
        if (!reason) {
            return '';
        }
        if (typeof reason === 'string') {
            return reason;
        }
        if (reason instanceof Error) {
            return reason.name ? reason.name + ': ' + reason.message : reason.message;
        }
        if (typeof reason.message === 'string' && reason.message.length > 0) {
            return reason.message;
        }
        try {
            return JSON.stringify(reason);
        } catch (_err) {
            return String(reason);
        }
    }

    function getFailedResourceStatus(urlHint) {
        if (typeof performance === 'undefined' || typeof performance.getEntriesByType !== 'function') {
            return 0;
        }
        const entries = performance.getEntriesByType('resource') || [];
        for (let i = entries.length - 1; i >= 0; i -= 1) {
            const entry = entries[i];
            if (!entry || typeof entry.name !== 'string') {
                continue;
            }
            const matchesTarget = urlHint
                ? entry.name.indexOf(urlHint) !== -1
                : /\.wasm(\?|$)|\.js(\?|$)/i.test(entry.name);
            if (matchesTarget && typeof entry.responseStatus === 'number' && entry.responseStatus >= 400) {
                return entry.responseStatus;
            }
        }
        return 0;
    }

    function hasFailedResourceStatus(urlHint) {
        return getFailedResourceStatus(urlHint) >= 400;
    }

    function classifyError(reason, resourceUrl) {
        const text = formatReason(reason);
        if (/timeout|timed out|stalled/i.test(text)) {
            return 'timeout';
        }
        if (hasFailedResourceStatus(resourceUrl)) {
            return 'network';
        }
        if (
            /http\s+[45]\d\d|status code is not ok|failed to fetch|networkerror|network error|load failed|aborted|aborterror|err_aborted|err_failed|err_connection|err_internet|err_content_length_mismatch|importing a module script failed|failed to load module|dynamically imported module|failed to load required startup/i.test(
                text
            )
        ) {
            return 'network';
        }
        if (
            (typeof WebAssembly !== 'undefined' &&
                (reason instanceof WebAssembly.CompileError || reason instanceof WebAssembly.LinkError)) ||
            reason instanceof SyntaxError ||
            /compileerror|linkerror|magic word|webassembly\.instantiate|webassembly\.compile|wasm|integrity|content security policy/i.test(
                text
            )
        ) {
            return 'compile';
        }
        if (/webgl|hardware acceleration|rendering context/i.test(text)) {
            return 'webgl';
        }
        return 'network';
    }

    function showError(category, details) {
        if (isUnloading) {
            return;
        }
        const resolvedCategory = Object.prototype.hasOwnProperty.call(ERROR_CATEGORIES, category)
            ? category
            : 'network';

        if (bootFailed && resolvedCategory !== 'webgl') {
            const extraDetail = formatReason(details);
            if (extraDetail && /http\s+[45]\d\d/i.test(extraDetail)) {
                runWhenDomReady(function () {
                    const detailsEl = document.getElementById('error_details');
                    if (detailsEl && !/http\s+[45]\d\d/i.test(detailsEl.textContent || '')) {
                        detailsEl.textContent = extraDetail;
                        detailsEl.style.display = 'block';
                    }
                });
            }
            return;
        }
        if (!isBootPending() && resolvedCategory !== 'webgl') {
            return;
        }

        bootFailed = true;
        clearWatchdog();

        const cfg = ERROR_CATEGORIES[resolvedCategory];
        const detailText = formatReason(details);

        runWhenDomReady(function () {
            const loading = document.getElementById('loading_indicator');
            if (loading) {
                loading.setAttribute('aria-busy', 'false');
                loading.hidden = true;
                loading.style.display = 'none';
            }

            const canvas = document.getElementById('the_canvas_id');
            if (canvas) {
                canvas.style.display = 'none';
            }

            const errorContainer = document.getElementById('error_container');
            if (errorContainer) {
                errorContainer.setAttribute('data-error-category', resolvedCategory);
            }

            const titleEl = document.getElementById('error_title');
            if (titleEl) {
                titleEl.textContent = cfg.title;
            }

            const msgEl = document.getElementById('error_message');
            if (msgEl) {
                msgEl.textContent = cfg.message;
            }

            if (resolvedCategory !== 'webgl') {
                const headingEl = document.getElementById('error_recovery_heading');
                if (headingEl) {
                    headingEl.textContent = cfg.heading;
                }
                const listEl = document.getElementById('error_recovery_list');
                if (listEl) {
                    while (listEl.firstChild) {
                        listEl.removeChild(listEl.firstChild);
                    }
                    for (let i = 0; i < cfg.steps.length; i += 1) {
                        const li = document.createElement('li');
                        li.textContent = cfg.steps[i];
                        listEl.appendChild(li);
                    }
                }
            }

            const detailsEl = document.getElementById('error_details');
            if (detailsEl) {
                if (detailText) {
                    detailsEl.textContent = detailText;
                    detailsEl.style.display = 'block';
                } else {
                    detailsEl.textContent = '';
                    detailsEl.style.display = 'none';
                }
            }

            if (errorContainer) {
                errorContainer.style.display = 'flex';
            }

            announce(cfg.alert, 'assertive');

            const reloadBtn = document.getElementById('webgl_reload_button');
            if (reloadBtn && typeof reloadBtn.focus === 'function') {
                reloadBtn.focus();
            }
        });
    }

    function handleFailure(reason, resourceUrl) {
        const category = classifyError(reason, resourceUrl);
        showError(category, reason);
    }

    function recordProgress(stageLabel) {
        if (stageLabel) {
            lastStageLabel = String(stageLabel);
        }
        if (bootFailed || isUnloading) {
            clearWatchdog();
            return;
        }
        clearWatchdog();
        const timeoutMs = resolveStallTimeoutMs();
        stallTimerId = window.setTimeout(function () {
            stallTimerId = null;
            if (!isBootPending() || isUnloading) {
                return;
            }
            showError(
                'timeout',
                'Startup stalled for ' + timeoutMs + ' ms without progress (' + lastStageLabel + ').'
            );
        }, timeoutMs);
    }

    function attachDomBindings() {
        const reloadButton = document.querySelector('.reload-btn');
        if (reloadButton && !reloadButton.dataset.reloadBound) {
            reloadButton.dataset.reloadBound = 'true';
            reloadButton.addEventListener('click', function () {
                window.location.reload();
            });
        }

        const queued = pendingActions.slice();
        pendingActions = [];
        for (let i = 0; i < queued.length; i += 1) {
            queued[i]();
        }

        if (typeof MutationObserver !== 'undefined' && document.body) {
            const observer = new MutationObserver(function () {
                if (!document.getElementById('loading_indicator')) {
                    clearWatchdog();
                    observer.disconnect();
                }
            });
            observer.observe(document.body, { childList: true });
        }
    }

    window.addEventListener('beforeunload', function () {
        isUnloading = true;
        clearWatchdog();
    });

    // Capture-phase listener catches <script> and <link rel="modulepreload"|"preload"> network/integrity failures
    window.addEventListener(
        'error',
        function (event) {
            if (!isBootPending() || isUnloading) {
                return;
            }
            const target = event && event.target;
            if (target && target !== window && target.tagName) {
                const tag = String(target.tagName).toUpperCase();
                const rel = String(target.rel || '').toLowerCase();
                if (tag === 'SCRIPT' || (tag === 'LINK' && (rel === 'modulepreload' || rel === 'preload'))) {
                    const assetUrl = target.src || target.href || 'startup resource';
                    const statusCode = getFailedResourceStatus(assetUrl);
                    const suffix = statusCode >= 400 ? ' (HTTP ' + statusCode + ')' : '';
                    showError('network', 'Failed to load required startup asset: ' + assetUrl + suffix);
                }
                return;
            }
            const reason = (event && (event.error || event.message)) || 'Unexpected startup error';
            handleFailure(reason);
        },
        true
    );

    window.addEventListener('unhandledrejection', function (event) {
        if (!isBootPending() || isUnloading) {
            return;
        }
        const reason = (event && event.reason) || 'Unhandled startup promise rejection';
        handleFailure(reason);
    });

    window.addEventListener('securitypolicyviolation', function (event) {
        if (!isBootPending() || isUnloading) {
            return;
        }
        const directive = (event && event.violatedDirective) || 'Content-Security-Policy';
        const blocked = (event && event.blockedURI) || 'inline';
        showError('compile', 'Content Security Policy blocked ' + blocked + ' (' + directive + ').');
    });

    window.__mathExplorerBoot = {
        announce: announce,
        recordProgress: recordProgress,
        classifyError: classifyError,
        showError: showError,
        handleFailure: handleFailure,
        clearWatchdog: clearWatchdog,
        syncShellThemeFromStorage: syncShellThemeFromStorage
    };

    recordProgress('Loading startup scripts');

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', attachDomBindings, { once: true });
    } else {
        attachDomBindings();
    }
})();
