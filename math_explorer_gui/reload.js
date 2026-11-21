// Pre-boot theme sync, app-shell SW registration, split stall/compile watchdog,
// WebGL context recovery bridge, and persistent runtime trap/rejection bridge.
(function () {
    const DEFAULT_STALL_TIMEOUT_MS = 12000;
    const DEFAULT_COMPILE_TIMEOUT_MS = 45000;
    const DEFAULT_WEBGL_RESTORE_TIMEOUT_MS = 8000;
    const DEFAULT_RECOVERY_FILENAME = 'math-explorer-recovery.mws';
    const STARTUP_ASSET_PATTERN = /(?:^|\/)(recovery\.js|reload\.js|bootstrap\.js|initializer\.js|math_explorer_gui\.js|math_explorer_gui_bg\.wasm|worker\.js)(?:[?#]|$)/i;
    const SHELL_THEME_TOKENS = {
        dark: { colorScheme: 'dark', themeColor: '#121214' },
        light: { colorScheme: 'light', themeColor: '#f8f8f8' },
        'high-contrast': { colorScheme: 'dark', themeColor: '#000000' }
    };
    const WEBGL_CONTEXT_LOST_COPY = {
        title: 'WebGL Graphics Context Lost',
        message: 'Your browser lost its WebGL graphics context and could not restore hardware rendering automatically. Your workspace state is preserved—select Reload Page to resume or download a workspace backup (.mws) below.',
        blockedStorageMessage: 'Your browser lost its WebGL graphics context and blocked local storage writes. Select Download Saved Workspace (.mws) below before reloading so you do not lose your session.',
        alert: 'WebGL graphics context lost and could not be restored automatically. Select Reload Page to recover.',
        blockedStorageAlert: 'WebGL graphics context lost and browser storage blocked saving. Download your saved workspace envelope before selecting Reload Page.'
    };

    const recovery = window.__mathExplorerRecovery || null;
    let swRegistered = false, bootFailed = false, bootCompleted = false, errorDisplayed = false, isUnloading = false;
    let stallTimerId = null, webglRestoreTimerId = null, webglContextLostOccurred = false, webglRecoveryHandlers = null;
    let savedWorkspaceEnvelope = '', savedWorkspaceFilename = DEFAULT_RECOVERY_FILENAME, storageWriteBlocked = false;
    let currentPhase = 'transfer', lastStageLabel = 'Initializing startup scripts', lastFatalPanic = '', pendingActions = [];
    const bootAbortController = typeof AbortController !== 'undefined' ? new AbortController() : null;

    function markBootManaged() {
        const root = document.documentElement;
        if (root && !root.hasAttribute('data-boot-initialized')) root.setAttribute('data-boot-initialized', 'true');
    }
    function normalizeSavedTheme(rawTheme) {
        if (typeof rawTheme !== 'string') return null;
        const normalized = rawTheme.trim().toLowerCase();
        if (normalized === 'dark' || normalized === 'light') return normalized;
        return /^high[-_]?contrast$/.test(normalized) ? 'high-contrast' : null;
    }

    function readSavedThemeFromStorage() {
        try {
            if (!window.localStorage) return null;
            const raw = window.localStorage.getItem('app');
            if (!raw) return recovery ? recovery.readPreservedTheme() : null;
            const parsed = JSON.parse(raw.replace(/^\uFEFF/, ''));
            if (!parsed || typeof parsed !== 'object') return recovery ? recovery.extractThemePermissive(raw) : null;
            if (typeof parsed.schema_version === 'string') {
                const major = parseInt(parsed.schema_version.split('.')[0], 10);
                const validEnvelope = Number.isFinite(major) && major <= 1 && typeof parsed.app_version === 'string' &&
                    typeof parsed.timestamp === 'string' && parsed.app_state && typeof parsed.app_state === 'object';
                return validEnvelope ? normalizeSavedTheme(parsed.app_state.theme) : (recovery ? recovery.extractThemePermissive(raw) : null);
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
        if (!tokens) return null;
        const root = document.documentElement;
        if (root) {
            root.dataset.theme = savedTheme;
            root.style?.setProperty?.('color-scheme', tokens.colorScheme);
        }
        document.querySelector('meta[name="color-scheme"]')?.setAttribute('content', tokens.colorScheme);
        const themeMetas = document.querySelectorAll('meta[name="theme-color"]');
        for (let i = 0; i < themeMetas.length; i += 1) themeMetas[i].setAttribute('content', tokens.themeColor);
        return savedTheme;
    }

    function registerAppShellServiceWorker() {
        if (swRegistered) return;
        swRegistered = true;
        try {
            if (window.isSecureContext && navigator.serviceWorker?.register) {
                navigator.serviceWorker.register('./sw.js').catch(function () {});
            }
        } catch (_err) { /* ignore restricted SW contexts */ }
    }

    markBootManaged();
    recovery?.quarantineAbortedBoot?.();
    syncShellThemeFromStorage();

    function resolveTimeoutParam(globalOverrideName, queryParamName, defaultMs, minimumMs) {
        const floorMs = typeof minimumMs === 'number' ? minimumMs : 100;
        const overrideVal = window[globalOverrideName];
        if (typeof overrideVal === 'number' && Number.isFinite(overrideVal) && overrideVal >= floorMs) return Math.floor(overrideVal);
        try {
            const raw = new URLSearchParams(window.location.search).get(queryParamName);
            const parsed = raw !== null ? Number(raw) : NaN;
            if (Number.isFinite(parsed) && parsed >= floorMs) return Math.floor(parsed);
        } catch (_err) { /* ignore URL parse errors */ }
        return defaultMs;
    }

    function resolveWebglRestoreTimeoutMs() {
        return resolveTimeoutParam('__MATH_EXPLORER_WEBGL_RESTORE_TIMEOUT_MS', 'webgl_restore_timeout_ms', DEFAULT_WEBGL_RESTORE_TIMEOUT_MS, 50);
    }
    function runWhenDomReady(fn) {
        if (document.readyState === 'loading') pendingActions.push(fn);
        else fn();
    }
    function isBootFailed() {
        return bootFailed || document.documentElement?.getAttribute('data-boot-failed') === 'true' || Boolean(bootAbortController?.signal?.aborted);
    }
    function isBootPending() {
        if (isBootFailed() || bootCompleted) return false;
        if (document.readyState === 'loading') return true;
        const loading = document.getElementById('loading_indicator');
        return Boolean(loading && !loading.hidden && loading.style.display !== 'none' && loading.getAttribute('aria-busy') !== 'false');
    }
    function clearWatchdog() {
        if (stallTimerId !== null) window.clearTimeout(stallTimerId);
        stallTimerId = null;
    }
    function clearWebglRestoreTimer() {
        if (webglRestoreTimerId !== null) window.clearTimeout(webglRestoreTimerId);
        webglRestoreTimerId = null;
    }
    function abortBoot(reason) {
        if (bootAbortController && !bootAbortController.signal.aborted) {
            try { bootAbortController.abort(reason || 'Startup aborted'); } catch (_err) { bootAbortController.abort(); }
        }
    }
    function sanitizeRecoveryFilename(rawName) {
        if (typeof rawName !== 'string') return DEFAULT_RECOVERY_FILENAME;
        const leaf = rawName.trim().split(/[\\/]/).pop() || '';
        if (!leaf || leaf === '.' || leaf === '..') return DEFAULT_RECOVERY_FILENAME;
        return /\.(mws|json)$/i.test(leaf) ? leaf : leaf + '.mws';
    }
    function resolveSavedWorkspaceEnvelope() {
        if (savedWorkspaceEnvelope && savedWorkspaceEnvelope.length > 0) return savedWorkspaceEnvelope;
        try {
            const raw = window.localStorage?.getItem('app');
            return typeof raw === 'string' && raw.trim().length > 0 ? raw : '';
        } catch (_err) { return ''; }
    }
    function bindOnce(el, key, handler) {
        if (el && !el.dataset[key]) {
            el.dataset[key] = 'true';
            el.addEventListener('click', function (event) { event?.preventDefault?.(); handler(); });
        }
        return el;
    }
    function syncDownloadWorkspaceButton() {
        const btn = document.getElementById('download_workspace_button');
        if (!btn) return null;
        btn.hidden = resolveSavedWorkspaceEnvelope().length === 0;
        btn.setAttribute('data-storage-blocked', storageWriteBlocked ? 'true' : 'false');
        return btn;
    }
    function bindDownloadWorkspaceButton() {
        return bindOnce(document.getElementById('download_workspace_button'), 'downloadBound', downloadSavedWorkspace);
    }
    function bindReloadButton() {
        return bindOnce(document.getElementById('webgl_reload_button') || document.querySelector('.reload-btn'), 'reloadBound', function () { window.location.reload(); });
    }

    function setSavedWorkspaceEnvelope(envelopeJson, filename, storageBlocked) {
        if (typeof envelopeJson === 'string' && envelopeJson.length > 0) savedWorkspaceEnvelope = envelopeJson;
        if (filename) savedWorkspaceFilename = sanitizeRecoveryFilename(filename);
        storageWriteBlocked = Boolean(storageBlocked);
        runWhenDomReady(function () { bindDownloadWorkspaceButton(); syncDownloadWorkspaceButton(); });
    }

    function downloadSavedWorkspace() {
        const payload = resolveSavedWorkspaceEnvelope();
        if (!payload || typeof Blob === 'undefined' || !window.URL || !URL.createObjectURL) return false;
        const blob = new Blob([payload], { type: 'application/json' });
        const objectUrl = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = objectUrl;
        link.download = savedWorkspaceFilename || DEFAULT_RECOVERY_FILENAME;
        link.rel = 'noopener';
        link.style.display = 'none';
        document.body?.appendChild(link);
        link.click();
        link.parentNode?.removeChild(link);
        window.setTimeout(function () { URL.revokeObjectURL(objectUrl); }, 0);
        announce('Downloaded saved workspace envelope ' + link.download + '.', 'polite');
        return true;
    }

    function announce(message, priority) {
        if (!message) return;
        if (recovery?.announceLive) { recovery.announceLive(message, priority); return; }
        runWhenDomReady(function () {
            const slotId = /^(assertive|alert)$/i.test(String(priority)) ? 'aria-live-assertive-slot-0' : 'aria-live-polite-slot-0';
            const slot = document.getElementById(slotId);
            if (slot) slot.textContent = String(message);
        });
    }

    function formatReason(reason) {
        if (!reason) return '';
        if (typeof reason === 'string') return reason;
        if (reason instanceof Error || (typeof reason.message === 'string' && reason.message.length > 0)) {
            return reason.name ? reason.name + ': ' + reason.message : reason.message;
        }
        try { return JSON.stringify(reason); } catch (_err) { return String(reason); }
    }

    function findFailedStartupResource(urlHint) {
        if (typeof performance === 'undefined' || typeof performance.getEntriesByType !== 'function') return null;
        const entries = performance.getEntriesByType('resource') || [];
        const hint = urlHint && typeof urlHint === 'string' ? urlHint.replace(/^\.\//, '').split(/[?#]/)[0].split('/').pop() || '' : '';
        for (let i = entries.length - 1; i >= 0; i -= 1) {
            const entry = entries[i];
            if (!entry || typeof entry.name !== 'string') continue;
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
        if (/timeout|timed out|stalled/i.test(text)) return 'timeout';
        if (resourceUrl && (findFailedStartupResource(resourceUrl)?.status || 0) >= 400) return 'network';
        if (/http\s+[45]\d\d|status code is not ok|failed to fetch|networkerror|network error|load failed|error loading dynamically imported module|importing a module script failed|failed to load module|dynamically imported module|aborted|aborterror|err_aborted|err_failed|err_connection|err_internet|err_content_length_mismatch|failed to load required startup|failed to load startup/i.test(text)) {
            return 'network';
        }
        if (/webgl|hardware acceleration|rendering context/i.test(text)) return 'webgl';
        if (/sharedarraybuffer|cross-origin isolat|crossoriginisolated|atomics\.wait|isolation:/i.test(text)) return 'isolation';
        if ((typeof WebAssembly !== 'undefined' && (reason instanceof WebAssembly.CompileError || reason instanceof WebAssembly.LinkError)) ||
            reason instanceof SyntaxError || /compileerror|linkerror|syntaxerror|magic word|webassembly\.instantiate|webassembly\.compile|integrity|content security policy/i.test(text)) {
            return 'compile';
        }
        if ((typeof WebAssembly !== 'undefined' && reason instanceof WebAssembly.RuntimeError) ||
            reason instanceof RangeError || reason instanceof ReferenceError || reason instanceof TypeError || bootCompleted ||
            /runtimeerror|unreachable|out of bounds|call stack|panicked|thread panicked|divide by zero|integer overflow/i.test(text)) {
            return 'runtime';
        }
        return findFailedStartupResource() ? 'network' : 'runtime';
    }

    function collectDiagnosticDetails(details) {
        const parts = [];
        const bridged = lastFatalPanic || (typeof window.__mathExplorerLastFatalPanic === 'string' ? window.__mathExplorerLastFatalPanic : '');
        if (bridged) parts.push(bridged);
        if (typeof window.__mathExplorerReadPanicSummary === 'function') {
            try {
                const summary = window.__mathExplorerReadPanicSummary();
                if (typeof summary === 'string' && summary.trim() && (!bridged || bridged.indexOf(summary.trim()) === -1)) parts.push(summary.trim());
            } catch (_err) { /* ignore mid-trap borrow contention */ }
        }
        const primary = formatReason(details);
        if (primary && parts.join('\n').indexOf(primary) === -1) parts.push(primary);
        return parts.join('\n\n');
    }

    function setWebglOverlayState(visible) {
        runWhenDomReady(function () {
            const overlay = document.getElementById('webgl_recovery_overlay');
            if (!overlay) return;
            overlay.hidden = !visible;
            overlay.setAttribute('aria-busy', visible ? 'true' : 'false');
            overlay.style.display = visible ? 'flex' : 'none';
        });
    }

    function beginWebglContextRecovery(options) {
        if (isUnloading || bootFailed) return resolveWebglRestoreTimeoutMs();
        webglContextLostOccurred = true;
        const opts = options && typeof options === 'object' ? options : {};
        if (typeof opts.envelopeJson === 'string' || typeof opts.storageBlocked === 'boolean' || opts.filename) {
            setSavedWorkspaceEnvelope(opts.envelopeJson || '', opts.filename || DEFAULT_RECOVERY_FILENAME, Boolean(opts.storageBlocked));
        }
        clearWebglRestoreTimer();
        setWebglOverlayState(true);
        announce(opts.statusMessage || 'WebGL graphics context lost. Attempting automatic restoration; your workspace is preserved.', 'polite');
        const timeoutMs = resolveWebglRestoreTimeoutMs();
        webglRestoreTimerId = window.setTimeout(function () {
            webglRestoreTimerId = null;
            if (isUnloading) return;
            const detail = 'WebGL context restoration timed out after ' + timeoutMs + ' ms.';
            if (webglRecoveryHandlers && typeof webglRecoveryHandlers.onTimeout === 'function') webglRecoveryHandlers.onTimeout(detail);
            else failWebglContextRecovery(detail);
        }, timeoutMs);
        return timeoutMs;
    }

    function completeWebglContextRecovery(statusMessage) {
        clearWebglRestoreTimer();
        setWebglOverlayState(false);
        runWhenDomReady(function () {
            const canvas = document.getElementById('the_canvas_id');
            if (canvas) { canvas.style.display = ''; canvas.focus?.(); }
        });
        announce(statusMessage || 'WebGL graphics context restored. Simulation workspace resumed.', 'polite');
    }

    function failWebglContextRecovery(details) {
        webglContextLostOccurred = true;
        clearWebglRestoreTimer();
        setWebglOverlayState(false);
        showError('webgl', details || 'WebGL context was lost and could not be restored automatically.');
    }
    function registerWebglRecoveryHandlers(handlers) {
        webglRecoveryHandlers = handlers && typeof handlers === 'object' ? handlers : null;
    }
    function resolveBannerCopy(category, detailText) {
        const base = recovery?.resolveErrorCopy
            ? recovery.resolveErrorCopy(category, detailText, currentPhase)
            : { resolved: category || 'network', copy: { title: 'Startup Error', message: detailText || '', heading: 'Recovery Steps', steps: [], alert: 'Math Explorer encountered an error.' } };
        if (base.resolved !== 'webgl' || (!bootCompleted && !webglContextLostOccurred)) return base;
        const copy = {
            title: WEBGL_CONTEXT_LOST_COPY.title,
            message: storageWriteBlocked ? WEBGL_CONTEXT_LOST_COPY.blockedStorageMessage : WEBGL_CONTEXT_LOST_COPY.message,
            heading: 'Enable Hardware Acceleration & WebGL', steps: [],
            alert: storageWriteBlocked ? WEBGL_CONTEXT_LOST_COPY.blockedStorageAlert : WEBGL_CONTEXT_LOST_COPY.alert
        };
        return { resolved: 'webgl', copy: copy };
    }

    function showError(category, details) {
        if (isUnloading) return;
        const detailText = collectDiagnosticDetails(details);
        if (errorDisplayed && category !== 'webgl') {
            if (detailText) {
                runWhenDomReady(function () {
                    const el = document.getElementById('error_details');
                    if (el && (el.textContent || '').indexOf(detailText) === -1) { el.textContent = detailText; el.style.display = 'block'; }
                });
            }
            return;
        }
        const resolvedInfo = resolveBannerCopy(category, detailText);
        const resolvedCategory = resolvedInfo.resolved;
        const cfg = resolvedInfo.copy;
        errorDisplayed = true;
        bootFailed = true;
        clearWatchdog();
        clearWebglRestoreTimer();
        abortBoot(detailText || resolvedCategory);
        if (!bootCompleted) {
            try { delete window.wasmBindings; } catch (_err) { window.wasmBindings = undefined; }
        }
        runWhenDomReady(function () {
            if (recovery?.renderErrorDom) recovery.renderErrorDom(resolvedCategory, cfg, detailText);
            document.getElementById('error_container')?.setAttribute('data-storage-blocked', storageWriteBlocked ? 'true' : 'false');
            bindDownloadWorkspaceButton();
            if (resolvedCategory === 'webgl') syncDownloadWorkspaceButton();
            announce(cfg.alert, 'assertive');
            const reloadBtn = bindReloadButton();
            if (reloadBtn && typeof reloadBtn.focus === 'function') reloadBtn.focus();
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
        clearWatchdog();
        if (bootFailed || bootCompleted || isUnloading) return;
        const isCompile = currentPhase === 'compile';
        const timeoutMs = isCompile
            ? resolveTimeoutParam('__MATH_EXPLORER_COMPILE_TIMEOUT_MS', 'compile_timeout_ms', DEFAULT_COMPILE_TIMEOUT_MS, 100)
            : resolveTimeoutParam('__MATH_EXPLORER_STALL_TIMEOUT_MS', 'stall_timeout_ms', DEFAULT_STALL_TIMEOUT_MS, 100);
        stallTimerId = window.setTimeout(function () {
            stallTimerId = null;
            if (!isBootPending() || isUnloading) return;
            const detail = isCompile
                ? 'WebAssembly compilation or graphics initialization timed out after ' + timeoutMs + ' ms (' + lastStageLabel + ').'
                : 'Startup stalled for ' + timeoutMs + ' ms without progress (' + lastStageLabel + ').';
            showError('timeout', detail);
        }, timeoutMs);
    }

    function markBootComplete() {
        if (bootFailed) return;
        bootCompleted = true;
        currentPhase = 'complete';
        clearWatchdog();
        markBootManaged();
        registerAppShellServiceWorker();
    }

    function enterPhase(phase, stageLabel) {
        if (phase === 'complete') { markBootComplete(); return; }
        if (bootCompleted || bootFailed) { clearWatchdog(); return; }
        if (!document.getElementById('loading_indicator') && document.readyState !== 'loading') { markBootComplete(); return; }
        if (stageLabel) lastStageLabel = String(stageLabel);
        currentPhase = phase === 'compile' ? 'compile' : 'transfer';
        armWatchdog();
    }

    function bindCanvasContextGuard() {
        const canvas = document.getElementById('the_canvas_id');
        if (!canvas || canvas.dataset.webglGuardBound) return;
        canvas.dataset.webglGuardBound = 'true';
        canvas.addEventListener('webglcontextlost', function (event) {
            if (event && typeof event.preventDefault === 'function') event.preventDefault();
            if (!isUnloading && !isBootFailed() && !webglRecoveryHandlers && bootCompleted) beginWebglContextRecovery({});
        });
        canvas.addEventListener('webglcontextrestored', function () {
            if (!isUnloading && !isBootFailed() && !webglRecoveryHandlers && bootCompleted) completeWebglContextRecovery();
        });
    }

    function attachDomBindings() {
        markBootManaged();
        recovery?.syncActionButtons?.();
        recovery?.syncSafeModeBanner?.();
        bindReloadButton();
        bindDownloadWorkspaceButton();
        bindCanvasContextGuard();
        const queued = pendingActions.slice();
        pendingActions = [];
        for (let i = 0; i < queued.length; i += 1) queued[i]();
        if (isBootPending() && !isUnloading) {
            const failed = findFailedStartupResource();
            if (failed) showError('network', 'Failed to load required startup asset: ' + failed.name + ' (HTTP ' + failed.status + ')');
        }
        if (typeof MutationObserver !== 'undefined' && document.body) {
            const observer = new MutationObserver(function () {
                if (!document.getElementById('loading_indicator')) {
                    if (!isBootFailed()) markBootComplete();
                    else { bootFailed = true; clearWatchdog(); }
                    observer.disconnect();
                }
            });
            observer.observe(document.body, { childList: true });
        }
    }

    window.addEventListener('beforeunload', function () { isUnloading = true; clearWatchdog(); clearWebglRestoreTimer(); });
    window.addEventListener('error', function (event) {
        if (isUnloading) return;
        const target = event && event.target;
        if (target && target !== window && target.tagName) {
            if (!isBootPending()) return;
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
        if (!isUnloading) handleFailure((event && event.reason) || 'Unhandled promise rejection');
    });
    window.addEventListener('securitypolicyviolation', function (event) {
        if (isBootPending() && !isUnloading) {
            showError('compile', 'Content Security Policy blocked ' + ((event && event.blockedURI) || 'inline') + ' (' + ((event && event.violatedDirective) || 'Content-Security-Policy') + ').');
        }
    });

    window.__mathExplorerBoot = {
        announce: announce, enterPhase: enterPhase, markBootComplete: markBootComplete, isBootFailed: isBootFailed,
        abortController: bootAbortController, abortBoot: abortBoot, findFailedStartupResource: findFailedStartupResource,
        classifyError: classifyError, handleFailure: handleFailure, showError: showError, clearWatchdog: clearWatchdog,
        syncShellThemeFromStorage: syncShellThemeFromStorage, getWebglRestoreTimeoutMs: resolveWebglRestoreTimeoutMs,
        setSavedWorkspaceEnvelope: setSavedWorkspaceEnvelope, downloadSavedWorkspace: downloadSavedWorkspace,
        beginWebglContextRecovery: beginWebglContextRecovery, completeWebglContextRecovery: completeWebglContextRecovery,
        failWebglContextRecovery: failWebglContextRecovery, registerWebglRecoveryHandlers: registerWebglRecoveryHandlers,
        recordProgress: function (stageLabel) { if (stageLabel) lastStageLabel = String(stageLabel); armWatchdog(); },
        getPhase: function () { return currentPhase; },
        markBootStarting: function () { recovery?.markBootStarting?.(); },
        markFirstFrameRendered: function () { markBootComplete(); recovery?.markFirstFrameRendered?.(); },
        reportFatalPanic: function (summary) { if (typeof summary === 'string' && summary.trim()) lastFatalPanic = summary.trim(); },
        getFatalPanic: function () { return lastFatalPanic; },
        getAbortSignal: function () { return bootAbortController ? bootAbortController.signal : undefined; }
    };

    armWatchdog();
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', attachDomBindings, { once: true });
    else attachDomBindings();
})();
