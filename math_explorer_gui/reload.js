// Pre-boot theme sync, app-shell SW registration, split stall/compile watchdog,
// shared AbortController cancellation, and persistent runtime trap/rejection bridge.
(function () {
    const DEFAULT_STALL_TIMEOUT_MS = 12000;
    const DEFAULT_COMPILE_TIMEOUT_MS = 45000;
    const STARTUP_ASSET_PATTERN =
        /(?:^|\/)(recovery\.js|reload\.js|bootstrap\.js|initializer\.js|math_explorer_gui\.js|math_explorer_gui_bg\.wasm|worker\.js)(?:[?#]|$)/i;

    const SHELL_THEME_TOKENS = {
        dark: { colorScheme: 'dark', themeColor: '#121214' },
        light: { colorScheme: 'light', themeColor: '#f8f8f8' },
        'high-contrast': { colorScheme: 'dark', themeColor: '#000000' }
    };

    const recovery = window.__mathExplorerRecovery || null;

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
        if (normalized === 'dark' || normalized === 'light') {
            return normalized;
        }
        if (normalized === 'highcontrast' || normalized === 'high-contrast' || normalized === 'high_contrast') {
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
                return recovery ? recovery.readPreservedTheme() : null;
            }
            const parsed = JSON.parse(raw.replace(/^\uFEFF/, ''));
            if (!parsed || typeof parsed !== 'object') {
                return recovery ? recovery.extractThemePermissive(raw) : null;
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
                    return recovery ? recovery.extractThemePermissive(raw) : null;
                }
                return normalizeSavedTheme(parsed.app_state.theme);
            }
            if (typeof parsed.selected_tab !== 'number' || !parsed.tab_states || typeof parsed.tab_states !== 'object') {
                return recovery ? recovery.extractThemePermissive(raw) : null;
            }
            return normalizeSavedTheme(parsed.theme);
        } catch (_err) {
            return recovery ? recovery.readPreservedTheme() : null;
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
            root.style?.setProperty?.('color-scheme', tokens.colorScheme);
        }
        document.querySelector('meta[name="color-scheme"]')?.setAttribute('content', tokens.colorScheme);
        const themeMetas = document.querySelectorAll('meta[name="theme-color"]');
        for (let i = 0; i < themeMetas.length; i += 1) {
            themeMetas[i].setAttribute('content', tokens.themeColor);
        }
        return savedTheme;
    }

    function registerAppShellServiceWorker() {
        try {
            if (window.isSecureContext && navigator.serviceWorker?.register) {
                navigator.serviceWorker.register('./sw.js').catch(function () {});
            }
        } catch (_err) {
            // Ignore restricted Service Worker contexts
        }
    }

    markBootManaged();
    recovery?.quarantineAbortedBoot?.();
    syncShellThemeFromStorage();
    registerAppShellServiceWorker();

    let bootFailed = false;
    let bootCompleted = false;
    let errorDisplayed = false;
    let isUnloading = false;
    let stallTimerId = null;
    let currentPhase = 'transfer';
    let lastStageLabel = 'Initializing startup scripts';
    let lastFatalPanic = '';
    let pendingActions = [];
    const bootAbortController = typeof AbortController !== 'undefined' ? new AbortController() : null;

    function resolveTimeoutParam(globalOverrideName, queryParamName, defaultMs) {
        const overrideVal = window[globalOverrideName];
        if (typeof overrideVal === 'number' && Number.isFinite(overrideVal) && overrideVal >= 100) {
            return Math.floor(overrideVal);
        }
        try {
            const raw = new URLSearchParams(window.location.search).get(queryParamName);
            if (raw !== null) {
                const parsed = Number(raw);
                if (Number.isFinite(parsed) && parsed >= 100) {
                    return Math.floor(parsed);
                }
            }
        } catch (_err) {
            // Ignore URL parsing errors
        }
        return defaultMs;
    }

    function runWhenDomReady(fn) {
        if (document.readyState === 'loading') {
            pendingActions.push(fn);
        } else {
            fn();
        }
    }

    function isBootFailed() {
        return bootFailed || Boolean(bootAbortController?.signal?.aborted);
    }

    function isBootPending() {
        if (bootFailed || bootCompleted) {
            return false;
        }
        if (document.readyState === 'loading') {
            return true;
        }
        const loading = document.getElementById('loading_indicator');
        return Boolean(loading && !loading.hidden && loading.style.display !== 'none' && loading.getAttribute('aria-busy') !== 'false');
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

    function formatReason(reason) {
        if (!reason) {
            return '';
        }
        if (typeof reason === 'string') {
            return reason;
        }
        if (reason instanceof Error || (typeof reason.message === 'string' && reason.message.length > 0)) {
            return reason.name ? reason.name + ': ' + reason.message : reason.message;
        }
        try {
            return JSON.stringify(reason);
        } catch (_err) {
            return String(reason);
        }
    }

    function findFailedStartupResource(urlHint) {
        if (typeof performance === 'undefined' || typeof performance.getEntriesByType !== 'function') {
            return null;
        }
        const entries = performance.getEntriesByType('resource') || [];
        const hint = urlHint && typeof urlHint === 'string'
            ? urlHint.replace(/^\.\//, '').split(/[?#]/)[0].split('/').pop() || ''
            : '';
        for (let i = entries.length - 1; i >= 0; i -= 1) {
            const entry = entries[i];
            if (!entry || typeof entry.name !== 'string') {
                continue;
            }
            const matches = hint ? entry.name.indexOf(hint) !== -1 : STARTUP_ASSET_PATTERN.test(entry.name);
            if (matches && typeof entry.responseStatus === 'number' && entry.responseStatus >= 400) {
                const name = entry.name.split(/[?#]/)[0].split('/').pop() || entry.name;
                return { url: entry.name, name: name, status: entry.responseStatus };
            }
        }
        return null;
    }

    function classifyError(reason, resourceUrl) {
        const text = formatReason(reason);
        if (/timeout|timed out|stalled/i.test(text)) {
            return 'timeout';
        }
        if (resourceUrl && (findFailedStartupResource(resourceUrl)?.status || 0) >= 400) {
            return 'network';
        }
        if (/http\s+[45]\d\d|status code is not ok|failed to fetch|networkerror|network error|load failed|error loading dynamically imported module|importing a module script failed|failed to load module|dynamically imported module|aborted|aborterror|err_aborted|err_failed|err_connection|err_internet|err_content_length_mismatch|failed to load required startup|failed to load startup/i.test(text)) {
            return 'network';
        }
        if (/webgl|hardware acceleration|rendering context/i.test(text)) {
            return 'webgl';
        }
        if (/sharedarraybuffer|cross-origin isolat|crossoriginisolated|atomics\.wait|isolation:/i.test(text)) {
            return 'isolation';
        }
        if (
            (typeof WebAssembly !== 'undefined' && (reason instanceof WebAssembly.CompileError || reason instanceof WebAssembly.LinkError)) ||
            reason instanceof SyntaxError ||
            /compileerror|linkerror|syntaxerror|magic word|webassembly\.instantiate|webassembly\.compile|integrity|content security policy/i.test(text)
        ) {
            return 'compile';
        }
        if (
            (typeof WebAssembly !== 'undefined' && reason instanceof WebAssembly.RuntimeError) ||
            reason instanceof RangeError ||
            reason instanceof ReferenceError ||
            reason instanceof TypeError ||
            bootCompleted ||
            /runtimeerror|unreachable|out of bounds|call stack|panicked|thread panicked|divide by zero|integer overflow/i.test(text)
        ) {
            return 'runtime';
        }
        return findFailedStartupResource() ? 'network' : 'runtime';
    }

    function collectDiagnosticDetails(details) {
        const parts = [];
        const bridged = lastFatalPanic || (typeof window.__mathExplorerLastFatalPanic === 'string' ? window.__mathExplorerLastFatalPanic : '');
        if (bridged) {
            parts.push(bridged);
        }
        if (typeof window.__mathExplorerReadPanicSummary === 'function') {
            try {
                const summary = window.__mathExplorerReadPanicSummary();
                if (typeof summary === 'string' && summary.trim() && (!bridged || bridged.indexOf(summary.trim()) === -1)) {
                    parts.push(summary.trim());
                }
            } catch (_err) {
                // Ignore mid-trap borrow contention
            }
        }
        const primary = formatReason(details);
        if (primary && parts.join('\n').indexOf(primary) === -1) {
            parts.push(primary);
        }
        return parts.join('\n\n');
    }

    function showError(category, details) {
        if (isUnloading) {
            return;
        }
        const detailText = collectDiagnosticDetails(details);
        if (errorDisplayed && category !== 'webgl') {
            if (detailText) {
                runWhenDomReady(function () {
                    const el = document.getElementById('error_details');
                    if (el && (el.textContent || '').indexOf(detailText) === -1) {
                        el.textContent = detailText;
                        el.style.display = 'block';
                    }
                });
            }
            return;
        }
        errorDisplayed = true;
        bootFailed = true;
        clearWatchdog();
        if (!bootCompleted) {
            abortBoot(detailText || category);
            try {
                delete window.wasmBindings;
            } catch (_err) {
                window.wasmBindings = undefined;
            }
        }
        runWhenDomReady(function () {
            recovery?.renderErrorBoundary?.(category, detailText, currentPhase);
        });
    }

    function handleFailure(reason, resourceUrl) {
        const failed = findFailedStartupResource(resourceUrl);
        const text = formatReason(reason);
        const enriched = failed && text.indexOf('HTTP ' + failed.status) === -1
            ? 'Failed to load required startup asset ' + failed.name + ' (HTTP ' + failed.status + ')' + (text ? ': ' + text : '')
            : reason;
        showError(classifyError(enriched, resourceUrl), enriched);
    }

    function armWatchdog() {
        if (bootFailed || bootCompleted || isUnloading) {
            clearWatchdog();
            return;
        }
        clearWatchdog();
        const isCompile = currentPhase === 'compile';
        const timeoutMs = isCompile
            ? resolveTimeoutParam('__MATH_EXPLORER_COMPILE_TIMEOUT_MS', 'compile_timeout_ms', DEFAULT_COMPILE_TIMEOUT_MS)
            : resolveTimeoutParam('__MATH_EXPLORER_STALL_TIMEOUT_MS', 'stall_timeout_ms', DEFAULT_STALL_TIMEOUT_MS);
        stallTimerId = window.setTimeout(function () {
            stallTimerId = null;
            if (!isBootPending() || isUnloading) {
                return;
            }
            const detail = isCompile
                ? 'WebAssembly compilation or graphics initialization timed out after ' + timeoutMs + ' ms (' + lastStageLabel + ').'
                : 'Startup stalled for ' + timeoutMs + ' ms without progress (' + lastStageLabel + ').';
            showError('timeout', detail);
        }, timeoutMs);
    }

    function markBootComplete() {
        if (!bootFailed) {
            bootCompleted = true;
            currentPhase = 'complete';
            clearWatchdog();
            markBootManaged();
        }
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

    function attachDomBindings() {
        markBootManaged();
        if (recovery) {
            recovery.syncActionButtons?.();
            recovery.syncSafeModeBanner?.();
        } else {
            const btn = document.getElementById('webgl_reload_button');
            if (btn && !btn.dataset.reloadBound) {
                btn.dataset.reloadBound = 'true';
                btn.addEventListener('click', function (ev) {
                    ev.preventDefault();
                    window.location.reload();
                });
            }
        }
        const queued = pendingActions.slice();
        pendingActions = [];
        for (let i = 0; i < queued.length; i += 1) {
            queued[i]();
        }
        if (isBootPending() && !isUnloading) {
            const failed = findFailedStartupResource();
            if (failed) {
                showError('network', 'Failed to load required startup asset: ' + failed.name + ' (HTTP ' + failed.status + ')');
            }
        }
    }

    window.addEventListener('beforeunload', function () {
        isUnloading = true;
        clearWatchdog();
    });

    window.addEventListener('error', function (event) {
        if (isUnloading) {
            return;
        }
        const target = event && event.target;
        if (target && target !== window && target.tagName) {
            if (!isBootPending()) {
                return;
            }
            const tag = String(target.tagName).toUpperCase();
            const rel = String(target.rel || '').toLowerCase();
            if (tag === 'SCRIPT' || (tag === 'LINK' && (rel === 'modulepreload' || rel === 'preload'))) {
                const assetUrl = target.src || target.href || 'startup resource';
                const status = findFailedStartupResource(assetUrl)?.status || 0;
                showError('network', 'Failed to load required startup asset: ' + assetUrl + (status >= 400 ? ' (HTTP ' + status + ')' : ''));
            }
            return;
        }
        handleFailure((event && (event.error || event.message)) || 'Unexpected runtime error');
    }, true);

    window.addEventListener('unhandledrejection', function (event) {
        if (!isUnloading) {
            handleFailure((event && event.reason) || 'Unhandled promise rejection');
        }
    });

    window.addEventListener('securitypolicyviolation', function (event) {
        if (isBootPending() && !isUnloading) {
            const directive = (event && event.violatedDirective) || 'Content-Security-Policy';
            const blocked = (event && event.blockedURI) || 'inline';
            showError('compile', 'Content Security Policy blocked ' + blocked + ' (' + directive + ').');
        }
    });

    window.__mathExplorerBoot = {
        announce: function (msg, priority) {
            recovery?.announceLive?.(msg, priority);
        },
        recordProgress: function (stageLabel) {
            if (stageLabel) {
                lastStageLabel = String(stageLabel);
            }
            armWatchdog();
        },
        enterPhase: enterPhase,
        getPhase: function () {
            return currentPhase;
        },
        markBootStarting: function () {
            recovery?.markBootStarting?.();
        },
        markBootComplete: markBootComplete,
        markFirstFrameRendered: function () {
            markBootComplete();
            recovery?.markFirstFrameRendered?.();
        },
        reportFatalPanic: function (summary) {
            if (typeof summary === 'string' && summary.trim()) {
                lastFatalPanic = summary.trim();
            }
        },
        getFatalPanic: function () {
            return lastFatalPanic;
        },
        isBootFailed: isBootFailed,
        abortController: bootAbortController,
        getAbortSignal: function () {
            return bootAbortController ? bootAbortController.signal : undefined;
        },
        abortBoot: abortBoot,
        findFailedStartupResource: findFailedStartupResource,
        classifyError: classifyError,
        handleFailure: handleFailure,
        showError: showError
    };

    armWatchdog();
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', attachDomBindings, { once: true });
    } else {
        attachDomBindings();
    }
})();
