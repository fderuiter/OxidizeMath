// Pre-boot theme sync, app-shell SW registration, error boundary, split stall/compile watchdog,
// shared AbortController cancellation, ARIA announcer, and reload handler for Math Explorer GUI
(function () {
    const DEFAULT_STALL_TIMEOUT_MS = 12000;
    const DEFAULT_COMPILE_TIMEOUT_MS = 45000;
    const STARTUP_ASSET_PATTERN =
        /(?:^|\/)(reload\.js|bootstrap\.js|initializer\.js|math_explorer_gui\.js|math_explorer_gui_bg\.wasm|worker\.js)(?:[?#]|$)/i;

    const SHELL_THEME_TOKENS = {
        dark: { colorScheme: 'dark', themeColor: '#121214' },
        light: { colorScheme: 'light', themeColor: '#f8f8f8' },
        'high-contrast': { colorScheme: 'dark', themeColor: '#000000' }
    };

    function markBootManaged() {
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

    function registerAppShellServiceWorker() {
        try {
            if (
                !window.isSecureContext ||
                !navigator.serviceWorker ||
                typeof navigator.serviceWorker.register !== 'function'
            ) {
                return;
            }
            navigator.serviceWorker.register('./sw.js').catch(function () {
                // Ignore SecurityError or unavailable SW in Private Browsing modes
            });
        } catch (_err) {
            // Ignore restricted navigator.serviceWorker accessors
        }
    }

    markBootManaged();
    syncShellThemeFromStorage();
    registerAppShellServiceWorker();

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
                'Your browser downloaded the application bundle, but failed to compile, instantiate, or start the WebAssembly module. Cached files may be incomplete or incompatible.',
            heading: 'Recovery Steps',
            steps: [
                'Select Reload Page or perform a hard refresh (Ctrl+Shift+R / Cmd+Shift+R) to bypass stale or corrupted cached files.',
                'Ensure you are using a current release of Chrome, Edge, Firefox, or Safari with WebAssembly enabled.',
                'Verify that browser security extensions or enterprise policies do not block WebAssembly execution.'
            ],
            alert:
                'WebAssembly startup error. Cached files may be corrupted. Select Reload Page to retry.'
        },
        isolation: {
            title: 'Browser Isolation & Shared Memory Unavailable',
            message:
                'Math Explorer encountered an operation requiring SharedArrayBuffer or cross-origin isolation that is unavailable or blocked in this browser context.',
            heading: 'Recovery Steps',
            steps: [
                'Open Math Explorer over HTTPS or localhost in a standard browser window rather than a restricted embedded WebView.',
                'Allow site data, Service Workers, and standard WebAssembly memory features for this origin.',
                'Disable extensions or enterprise policies that strip isolation headers or block SharedArrayBuffer, then select Reload Page.'
            ],
            alert:
                'Browser isolation or SharedArrayBuffer is unavailable. Adjust browser settings and select Reload Page.'
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

    const COMPILE_TIMEOUT_COPY = {
        title: 'Startup Timed Out',
        message:
            'Math Explorer finished downloading its application bundle, but WebAssembly compilation or graphics initialization timed out before startup could complete.',
        heading: 'Recovery Steps',
        steps: [
            'Select Reload Page below to retry compiling WebAssembly and initializing WebGL.',
            'Close memory-intensive browser tabs or background applications if your device is low on CPU or memory.',
            'Verify that browser extensions or security policies are not stalling WebAssembly compilation or WebGL context creation.'
        ],
        alert:
            'Math Explorer timed out during WebAssembly compilation or graphics initialization. Select Reload Page to retry.'
    };

    // Start pre-boot twin-slot toggles at slot 1 so slot 0 starts empty when Rust's AtomicBool(false) takes over.
    let politeSlotToggle = 1;
    let assertiveSlotToggle = 1;
    let bootFailed = false;
    let bootCompleted = false;
    let isUnloading = false;
    let stallTimerId = null;
    let currentPhase = 'transfer';
    let lastStageLabel = 'Initializing startup scripts';
    let pendingActions = [];
    const bootAbortController =
        typeof AbortController !== 'undefined' ? new AbortController() : null;

    function resolveTimeoutParam(globalOverrideName, queryParamName, defaultMs) {
        const overrideVal = window[globalOverrideName];
        if (typeof overrideVal === 'number' && Number.isFinite(overrideVal) && overrideVal >= 100) {
            return Math.floor(overrideVal);
        }
        try {
            const params = new URLSearchParams(window.location.search);
            const raw = params.get(queryParamName);
            if (raw !== null) {
                const parsed = Number(raw);
                if (Number.isFinite(parsed) && parsed >= 100) {
                    return Math.floor(parsed);
                }
            }
        } catch (_err) {
            // Ignore URL parsing errors in constrained environments
        }
        return defaultMs;
    }

    function resolveStallTimeoutMs() {
        return resolveTimeoutParam(
            '__MATH_EXPLORER_STALL_TIMEOUT_MS',
            'stall_timeout_ms',
            DEFAULT_STALL_TIMEOUT_MS
        );
    }

    function resolveCompileTimeoutMs() {
        return resolveTimeoutParam(
            '__MATH_EXPLORER_COMPILE_TIMEOUT_MS',
            'compile_timeout_ms',
            DEFAULT_COMPILE_TIMEOUT_MS
        );
    }

    function runWhenDomReady(fn) {
        if (document.readyState === 'loading') {
            pendingActions.push(fn);
        } else {
            fn();
        }
    }

    function isBootFailed() {
        return (
            bootFailed ||
            Boolean(bootAbortController && bootAbortController.signal && bootAbortController.signal.aborted)
        );
    }

    function isBootPending() {
        if (bootFailed || bootCompleted) {
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

    function abortBoot(reason) {
        if (bootAbortController && !bootAbortController.signal.aborted) {
            try {
                bootAbortController.abort(reason || 'Startup aborted');
            } catch (_err) {
                bootAbortController.abort();
            }
        }
    }

    function bindReloadButton() {
        const reloadButton =
            document.getElementById('webgl_reload_button') ||
            document.querySelector('.reload-btn');
        if (reloadButton && !reloadButton.dataset.reloadBound) {
            reloadButton.dataset.reloadBound = 'true';
            reloadButton.addEventListener('click', function (event) {
                if (event && typeof event.preventDefault === 'function') {
                    event.preventDefault();
                }
                window.location.reload();
            });
        }
        return reloadButton;
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
            return reason.name ? reason.name + ': ' + reason.message : reason.message;
        }
        try {
            return JSON.stringify(reason);
        } catch (_err) {
            return String(reason);
        }
    }

    function normalizeHintToken(urlHint) {
        if (!urlHint || typeof urlHint !== 'string') {
            return '';
        }
        const stripped = urlHint.replace(/^\.\//, '').split(/[?#]/)[0];
        const parts = stripped.split('/');
        return parts[parts.length - 1] || stripped;
    }

    function findFailedStartupResource(urlHint) {
        if (typeof performance === 'undefined' || typeof performance.getEntriesByType !== 'function') {
            return null;
        }
        const entries = performance.getEntriesByType('resource') || [];
        const hintToken = normalizeHintToken(urlHint);
        for (let i = entries.length - 1; i >= 0; i -= 1) {
            const entry = entries[i];
            if (!entry || typeof entry.name !== 'string') {
                continue;
            }
            const matchesTarget = hintToken
                ? entry.name.indexOf(hintToken) !== -1
                : STARTUP_ASSET_PATTERN.test(entry.name);
            if (matchesTarget && typeof entry.responseStatus === 'number' && entry.responseStatus >= 400) {
                const nameMatch = entry.name.split(/[?#]/)[0].split('/').pop() || entry.name;
                return {
                    url: entry.name,
                    name: nameMatch,
                    status: entry.responseStatus
                };
            }
        }
        return null;
    }

    function getFailedResourceStatus(urlHint) {
        const failed = findFailedStartupResource(urlHint);
        return failed ? failed.status : 0;
    }

    function hasFailedResourceStatus(urlHint) {
        return getFailedResourceStatus(urlHint) >= 400;
    }

    function classifyError(reason, resourceUrl) {
        const text = formatReason(reason);
        if (/timeout|timed out|stalled/i.test(text)) {
            return 'timeout';
        }
        if (resourceUrl && hasFailedResourceStatus(resourceUrl)) {
            return 'network';
        }
        if (
            /http\s+[45]\d\d|status code is not ok|failed to fetch|networkerror|network error|load failed|error loading dynamically imported module|importing a module script failed|failed to load module|dynamically imported module|aborted|aborterror|err_aborted|err_failed|err_connection|err_internet|err_content_length_mismatch|failed to load required startup|failed to load startup/i.test(
                text
            )
        ) {
            return 'network';
        }
        if (/webgl|hardware acceleration|rendering context/i.test(text)) {
            return 'webgl';
        }
        if (
            /sharedarraybuffer|cross-origin isolat|crossoriginisolated|atomics\.wait/i.test(text)
        ) {
            return 'isolation';
        }
        if (
            (typeof WebAssembly !== 'undefined' &&
                (reason instanceof WebAssembly.CompileError ||
                    reason instanceof WebAssembly.LinkError ||
                    reason instanceof WebAssembly.RuntimeError)) ||
            reason instanceof SyntaxError ||
            reason instanceof RangeError ||
            reason instanceof ReferenceError ||
            reason instanceof TypeError ||
            /compileerror|linkerror|runtimeerror|rangeerror|referenceerror|typeerror|unreachable|out of bounds|call stack|magic word|webassembly\.instantiate|webassembly\.compile|wasm|integrity|content security policy/i.test(
                text
            )
        ) {
            return 'compile';
        }
        if (hasFailedResourceStatus()) {
            return 'network';
        }
        return 'network';
    }

    function selectCategoryConfig(resolvedCategory, detailText) {
        if (
            resolvedCategory === 'timeout' &&
            (currentPhase === 'compile' || /compil|graphics|webrunner/i.test(detailText))
        ) {
            return COMPILE_TIMEOUT_COPY;
        }
        return ERROR_CATEGORIES[resolvedCategory];
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

        const detailText = formatReason(details);
        bootFailed = true;
        clearWatchdog();
        abortBoot(detailText || resolvedCategory);

        if (!bootCompleted) {
            try {
                delete window.wasmBindings;
            } catch (_err) {
                window.wasmBindings = undefined;
            }
        }

        const root = document.documentElement;
        if (root) {
            root.setAttribute('data-boot-initialized', 'true');
            root.setAttribute('data-boot-failed', 'true');
        }

        const cfg = selectCategoryConfig(resolvedCategory, detailText);

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
                errorContainer.setAttribute('data-error-active', 'true');
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

            const reloadBtn = bindReloadButton();
            if (reloadBtn && typeof reloadBtn.focus === 'function') {
                reloadBtn.focus();
            }
        });
    }

    function enrichFailureReason(reason, resourceUrl) {
        const text = formatReason(reason);
        const failed = findFailedStartupResource(resourceUrl);
        if (!failed) {
            return reason;
        }
        const statusLabel = 'HTTP ' + failed.status;
        if (text.indexOf(statusLabel) !== -1 && text.indexOf(failed.name) !== -1) {
            return reason;
        }
        return (
            'Failed to load required startup asset ' +
            failed.name +
            ' (' +
            statusLabel +
            ')' +
            (text ? ': ' + text : '')
        );
    }

    function handleFailure(reason, resourceUrl) {
        const enriched = enrichFailureReason(reason, resourceUrl);
        const category = classifyError(enriched, resourceUrl);
        showError(category, enriched);
    }

    function armWatchdog() {
        if (bootFailed || bootCompleted || isUnloading) {
            clearWatchdog();
            return;
        }
        clearWatchdog();
        const isCompilePhase = currentPhase === 'compile';
        const timeoutMs = isCompilePhase ? resolveCompileTimeoutMs() : resolveStallTimeoutMs();
        stallTimerId = window.setTimeout(function () {
            stallTimerId = null;
            if (!isBootPending() || isUnloading) {
                return;
            }
            const detail = isCompilePhase
                ? 'WebAssembly compilation or graphics initialization timed out after ' +
                  timeoutMs +
                  ' ms (' +
                  lastStageLabel +
                  ').'
                : 'Startup stalled for ' +
                  timeoutMs +
                  ' ms without progress (' +
                  lastStageLabel +
                  ').';
            showError('timeout', detail);
        }, timeoutMs);
    }

    function recordProgress(stageLabel) {
        if (stageLabel) {
            lastStageLabel = String(stageLabel);
        }
        armWatchdog();
    }

    function enterPhase(phase, stageLabel) {
        if (phase === 'complete') {
            markBootComplete();
            return;
        }
        if (bootCompleted || bootFailed) {
            clearWatchdog();
            return;
        }
        if (!document.getElementById('loading_indicator') && document.readyState !== 'loading') {
            markBootComplete();
            return;
        }
        if (stageLabel) {
            lastStageLabel = String(stageLabel);
        }
        currentPhase = phase === 'compile' ? 'compile' : 'transfer';
        armWatchdog();
    }

    function markBootComplete() {
        if (bootFailed) {
            return;
        }
        bootCompleted = true;
        currentPhase = 'complete';
        clearWatchdog();
        const root = document.documentElement;
        if (root) {
            root.setAttribute('data-boot-initialized', 'true');
        }
    }

    function scanStartupResourceErrors() {
        if (!isBootPending() || isUnloading) {
            return false;
        }
        const failed = findFailedStartupResource();
        if (!failed) {
            return false;
        }
        showError(
            'network',
            'Failed to load required startup asset: ' +
                failed.name +
                ' (HTTP ' +
                failed.status +
                ')'
        );
        return true;
    }

    function attachDomBindings() {
        markBootManaged();
        bindReloadButton();

        const queued = pendingActions.slice();
        pendingActions = [];
        for (let i = 0; i < queued.length; i += 1) {
            queued[i]();
        }

        scanStartupResourceErrors();

        if (typeof MutationObserver !== 'undefined' && document.body) {
            const observer = new MutationObserver(function () {
                if (!document.getElementById('loading_indicator')) {
                    if (!bootFailed) {
                        markBootComplete();
                    } else {
                        clearWatchdog();
                    }
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
                    if (statusCode < 400) {
                        window.setTimeout(function () {
                            const lateStatus = getFailedResourceStatus(assetUrl);
                            if (lateStatus >= 400) {
                                showError(
                                    'network',
                                    'Failed to load required startup asset: ' +
                                        assetUrl +
                                        ' (HTTP ' +
                                        lateStatus +
                                        ')'
                                );
                            }
                        }, 0);
                    }
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
        enterPhase: enterPhase,
        getPhase: function () {
            return currentPhase;
        },
        markBootComplete: markBootComplete,
        isBootFailed: isBootFailed,
        abortController: bootAbortController,
        getAbortSignal: function () {
            return bootAbortController ? bootAbortController.signal : undefined;
        },
        abortBoot: abortBoot,
        findFailedStartupResource: findFailedStartupResource,
        classifyError: classifyError,
        showError: showError,
        handleFailure: handleFailure,
        clearWatchdog: clearWatchdog,
        syncShellThemeFromStorage: syncShellThemeFromStorage
    };

    if (!scanStartupResourceErrors()) {
        enterPhase('transfer', 'Loading startup scripts');
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', attachDomBindings, { once: true });
    } else {
        attachDomBindings();
    }

    window.addEventListener('load', function () {
        scanStartupResourceErrors();
    }, { once: true });
})();
