// Recovery boundary UI, boot-in-progress sentinel, failed-boot workspace quarantine,
// Safe Mode notice, manual workspace reset, and local backup download for Math Explorer GUI.
(function () {
    const APP_STORAGE_KEY = 'app';
    const BOOT_SENTINEL_KEY = 'math_explorer_boot_sentinel';
    const QUARANTINE_APP_KEY = 'math_explorer_quarantined_app';
    const QUARANTINE_META_KEY = 'math_explorer_quarantine_meta';
    const PRESERVED_THEME_KEY = 'math_explorer_preserved_theme';

    const SHELL_THEME_TOKENS = {
        dark: { colorScheme: 'dark', themeColor: '#121214' },
        light: { colorScheme: 'light', themeColor: '#f8f8f8' },
        'high-contrast': { colorScheme: 'dark', themeColor: '#000000' }
    };

    const ERROR_CATEGORIES = {
        runtime: {
            title: 'Simulation Runtime Error',
            message: 'Math Explorer halted after an unexpected WebAssembly runtime trap or unhandled asynchronous error. Your browser tab remains responsive and recoverable.',
            heading: 'Recovery Steps',
            steps: [
                'Select Reload Page below to restart Math Explorer cleanly.',
                'If the error repeats during startup, select Reset Workspace & Reload to archive your saved workspace and restore clean defaults.',
                'Select Download Quarantined State (when available) to save a local JSON copy of your previous workspace.'
            ],
            alert: 'Simulation runtime error. Math Explorer halted safely. Select Reload Page or Reset Workspace and Reload to recover.'
        },
        network: {
            title: 'Network Error Loading Math Explorer',
            message: 'Math Explorer could not finish downloading its WebAssembly application files. Your connection may have been interrupted or the server returned an error.',
            heading: 'Recovery Steps',
            steps: [
                'Check your internet connection and verify that firewalls, proxies, or content blockers allow .js and .wasm requests.',
                'Select Reload Page below (or perform a hard refresh) to retry downloading the application.',
                'If the server is temporarily unavailable, wait a moment and reload.'
            ],
            alert: 'Network error loading Math Explorer. Check your internet connection and select Reload Page to retry.'
        },
        compile: {
            title: 'WebAssembly Startup Error',
            message: 'Your browser downloaded the application bundle, but failed to compile, instantiate, or start the WebAssembly module. Cached files may be incomplete or incompatible.',
            heading: 'Recovery Steps',
            steps: [
                'Select Reload Page or perform a hard refresh (Ctrl+Shift+R / Cmd+Shift+R) to bypass stale or corrupted cached files.',
                'Ensure you are using a current release of Chrome, Edge, Firefox, or Safari with WebAssembly enabled.',
                'Verify that browser security extensions or enterprise policies do not block WebAssembly execution.'
            ],
            alert: 'WebAssembly startup error. Cached files may be corrupted. Select Reload Page to retry.'
        },
        isolation: {
            title: 'Browser Isolation & Shared Memory Unavailable',
            message: 'Math Explorer encountered an operation requiring SharedArrayBuffer or cross-origin isolation that is unavailable or blocked in this browser context.',
            heading: 'Recovery Steps',
            steps: [
                'Open Math Explorer over HTTPS or localhost in a standard browser window rather than a restricted embedded WebView.',
                'Allow site data, Service Workers, and standard WebAssembly memory features for this origin.',
                'Disable extensions or enterprise policies that strip isolation headers or block SharedArrayBuffer, then select Reload Page.'
            ],
            alert: 'Browser isolation or SharedArrayBuffer is unavailable. Adjust browser settings and select Reload Page.'
        },
        timeout: {
            title: 'Startup Timed Out',
            message: 'Loading Math Explorer stalled without progress before startup could complete. The network transfer may have frozen or stopped responding.',
            heading: 'Recovery Steps',
            steps: [
                'Confirm your network connection is active and not waiting on a captive portal or VPN.',
                'Select Reload Page below to restart the WebAssembly download cleanly.',
                'If you are on a slow or metered connection, pause competing downloads and try again.'
            ],
            alert: 'Loading Math Explorer timed out due to a stalled transfer. Select Reload Page to retry.'
        },
        webgl: {
            title: 'WebGL Initialization Failed',
            message: 'Math Explorer requires WebGL and hardware acceleration to render mathematics and interactive simulations. Your browser was unable to initialize a WebGL rendering context.',
            heading: 'Enable Hardware Acceleration & WebGL',
            steps: [],
            alert: 'WebGL initialization failed. Please enable hardware acceleration in your browser settings.'
        }
    };

    const COMPILE_TIMEOUT_COPY = {
        title: 'Startup Timed Out',
        message: 'Math Explorer finished downloading its application bundle, but WebAssembly compilation or graphics initialization timed out before startup could complete.',
        heading: 'Recovery Steps',
        steps: [
            'Select Reload Page below to retry compiling WebAssembly and initializing WebGL.',
            'Close memory-intensive browser tabs or background applications if your device is low on CPU or memory.',
            'Verify that browser extensions or security policies are not stalling WebAssembly compilation or WebGL context creation.'
        ],
        alert: 'Math Explorer timed out during WebAssembly compilation or graphics initialization. Select Reload Page to retry.'
    };

    let politeSlotToggle = 1;
    let assertiveSlotToggle = 1;
    let quarantineChecked = false;
    let quarantinedOnThisLoad = false;
    let safeModeDismissed = false;
    let errorBoundaryActive = false;

    function readItem(key) {
        try {
            return window.localStorage ? window.localStorage.getItem(key) : null;
        } catch (_err) {
            return null;
        }
    }

    function writeItem(key, value) {
        try {
            if (window.localStorage) {
                window.localStorage.setItem(key, value);
            }
        } catch (_err) {
            // Ignore restricted storage contexts
        }
    }

    function removeItem(key) {
        try {
            if (window.localStorage) {
                window.localStorage.removeItem(key);
            }
        } catch (_err) {
            // Ignore restricted storage contexts
        }
    }

    function normalizeThemeToken(rawTheme) {
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

    function extractThemePermissive(rawPayload) {
        if (typeof rawPayload !== 'string' || rawPayload.length === 0) {
            return null;
        }
        try {
            const parsed = JSON.parse(rawPayload.replace(/^\uFEFF/, ''));
            if (parsed && typeof parsed === 'object') {
                const raw = parsed.app_state && typeof parsed.app_state === 'object'
                    ? parsed.app_state.theme
                    : parsed.theme;
                const valid = normalizeThemeToken(raw);
                if (valid) {
                    return valid;
                }
            }
        } catch (_err) {
            const match = /"theme"\s*:\s*"(dark|light|high[-_]?contrast)"/i.exec(rawPayload);
            if (match) {
                return normalizeThemeToken(match[1]);
            }
        }
        return null;
    }

    function applyShellTheme(themeName) {
        const tokens = themeName && SHELL_THEME_TOKENS[themeName];
        if (!tokens) {
            return null;
        }
        const root = document.documentElement;
        if (root) {
            root.dataset.theme = themeName;
            root.style?.setProperty?.('color-scheme', tokens.colorScheme);
        }
        document.querySelector('meta[name="color-scheme"]')?.setAttribute('content', tokens.colorScheme);
        const themeMetas = document.querySelectorAll('meta[name="theme-color"]');
        for (let i = 0; i < themeMetas.length; i += 1) {
            themeMetas[i].setAttribute('content', tokens.themeColor);
        }
        return themeName;
    }

    function readPreservedTheme() {
        return normalizeThemeToken(readItem(PRESERVED_THEME_KEY));
    }

    function archiveWorkspacePayload(rawPayload, reason) {
        const preservedTheme =
            extractThemePermissive(rawPayload) ||
            normalizeThemeToken(document.documentElement?.dataset?.theme) ||
            readPreservedTheme();
        if (typeof rawPayload === 'string' && rawPayload.length > 0) {
            writeItem(QUARANTINE_APP_KEY, rawPayload);
            writeItem(QUARANTINE_META_KEY, JSON.stringify({
                quarantined_at: new Date().toISOString(),
                reason: reason || 'recovery'
            }));
        }
        if (preservedTheme) {
            writeItem(PRESERVED_THEME_KEY, preservedTheme);
            applyShellTheme(preservedTheme);
        }
        removeItem(APP_STORAGE_KEY);
        return preservedTheme;
    }

    function quarantineAbortedBoot() {
        if (quarantineChecked) {
            return quarantinedOnThisLoad;
        }
        quarantineChecked = true;
        if (!readItem(BOOT_SENTINEL_KEY)) {
            return false;
        }
        removeItem(BOOT_SENTINEL_KEY);
        const rawApp = readItem(APP_STORAGE_KEY);
        if (rawApp && rawApp.length > 0) {
            archiveWorkspacePayload(rawApp, 'aborted_startup');
            quarantinedOnThisLoad = true;
            return true;
        }
        return false;
    }

    function markBootStarting() {
        quarantineAbortedBoot();
        writeItem(BOOT_SENTINEL_KEY, JSON.stringify({ status: 'starting', started_at: Date.now() }));
    }

    function markFirstFrameRendered() {
        removeItem(BOOT_SENTINEL_KEY);
        if (extractThemePermissive(readItem(APP_STORAGE_KEY))) {
            removeItem(PRESERVED_THEME_KEY);
        }
        syncSafeModeBanner();
    }

    function hasQuarantinedState() {
        return Boolean(readItem(QUARANTINE_APP_KEY));
    }

    function announceLive(message, priority) {
        if (!message) {
            return;
        }
        const update = function () {
            const p = String(priority).toLowerCase();
            const isAssertive = p === 'assertive' || p === 'alert';
            const prefix = isAssertive ? 'aria-live-assertive-slot-' : 'aria-live-polite-slot-';
            const idx = isAssertive ? assertiveSlotToggle : politeSlotToggle;
            if (isAssertive) {
                assertiveSlotToggle = 1 - assertiveSlotToggle;
            } else {
                politeSlotToggle = 1 - politeSlotToggle;
            }
            const sibling = document.getElementById(prefix + (1 - idx));
            const active = document.getElementById(prefix + idx);
            if (sibling) {
                sibling.textContent = '';
            }
            if (active) {
                active.textContent = String(message);
            }
        };
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', update, { once: true });
        } else {
            update();
        }
    }

    function downloadQuarantinedState() {
        const payload = readItem(QUARANTINE_APP_KEY) || readItem(APP_STORAGE_KEY);
        if (!payload) {
            return false;
        }
        const blob = new Blob([payload], { type: 'application/json;charset=utf-8' });
        const objectUrl = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = objectUrl;
        link.download = 'math-explorer-quarantined-workspace.json';
        link.style.display = 'none';
        document.body.appendChild(link);
        link.click();
        link.remove();
        window.setTimeout(function () {
            URL.revokeObjectURL(objectUrl);
        }, 1000);
        announceLive('Downloaded quarantined workspace JSON backup.', 'polite');
        return true;
    }

    function clearQuarantinedState() {
        removeItem(QUARANTINE_APP_KEY);
        removeItem(QUARANTINE_META_KEY);
        safeModeDismissed = true;
        syncSafeModeBanner();
        syncActionButtons();
        announceLive('Cleared quarantined workspace backup.', 'polite');
    }

    function resetWorkspaceAndReload() {
        const rawPayload = readItem(APP_STORAGE_KEY);
        document.documentElement?.setAttribute('data-workspace-resetting', 'true');
        try {
            window.__mathExplorerDestroyRunner?.();
        } catch (_err) {
            // Ignore when WebAssembly already trapped
        }
        archiveWorkspacePayload(rawPayload, 'manual_reset');
        removeItem(BOOT_SENTINEL_KEY);
        window.location.reload();
    }

    function bindButtonOnce(id, key, handler) {
        const btn = document.getElementById(id);
        if (btn && !btn.dataset[key]) {
            btn.dataset[key] = 'true';
            btn.addEventListener('click', function (event) {
                event?.preventDefault?.();
                handler();
            });
        }
        return btn;
    }

    function syncActionButtons() {
        const reloadBtn = bindButtonOnce('webgl_reload_button', 'reloadBound', function () {
            window.location.reload();
        });
        bindButtonOnce('workspace_reset_button', 'resetBound', resetWorkspaceAndReload);
        const dlBtn = bindButtonOnce('quarantine_download_button', 'downloadBound', downloadQuarantinedState);
        if (dlBtn) {
            dlBtn.hidden = !Boolean(readItem(QUARANTINE_APP_KEY) || readItem(APP_STORAGE_KEY));
        }
        bindButtonOnce('safe_mode_download_button', 'downloadBound', downloadQuarantinedState);
        bindButtonOnce('safe_mode_reset_button', 'clearBound', clearQuarantinedState);
        bindButtonOnce('safe_mode_dismiss_button', 'dismissBound', function () {
            safeModeDismissed = true;
            syncSafeModeBanner();
        });
        return reloadBtn;
    }

    function syncSafeModeBanner() {
        const banner = document.getElementById('safe_mode_banner');
        if (!banner) {
            return;
        }
        const shouldShow = !errorBoundaryActive && !safeModeDismissed && hasQuarantinedState();
        banner.hidden = !shouldShow;
        if (shouldShow && quarantinedOnThisLoad) {
            quarantinedOnThisLoad = false;
            announceLive(
                'Safe Mode active: previous workspace state was quarantined after an interrupted startup.',
                'polite'
            );
        }
    }

    function populateRecoveryList(resolved, copy) {
        if (resolved === 'webgl') {
            return;
        }
        const headingEl = document.getElementById('error_recovery_heading');
        if (headingEl) {
            headingEl.textContent = copy.heading;
        }
        const listEl = document.getElementById('error_recovery_list');
        if (!listEl) {
            return;
        }
        while (listEl.firstChild) {
            listEl.removeChild(listEl.firstChild);
        }
        for (let i = 0; i < copy.steps.length; i += 1) {
            const li = document.createElement('li');
            li.textContent = copy.steps[i];
            listEl.appendChild(li);
        }
    }

    function resolveErrorCopy(category, detailText, currentPhase) {
        const resolved = Object.prototype.hasOwnProperty.call(ERROR_CATEGORIES, category)
            ? category
            : 'network';
        const copy = resolved === 'timeout' && (currentPhase === 'compile' || /compil|graphics|webrunner/i.test(detailText || ''))
            ? COMPILE_TIMEOUT_COPY
            : ERROR_CATEGORIES[resolved];
        return { resolved: resolved, copy: copy };
    }

    function renderErrorDom(resolved, copy, detailText) {
        errorBoundaryActive = true;
        document.documentElement?.setAttribute('data-boot-initialized', 'true');
        document.documentElement?.setAttribute('data-boot-failed', 'true');
        const loading = document.getElementById('loading_indicator');
        if (loading) {
            loading.setAttribute('aria-busy', 'false');
            loading.hidden = true;
            loading.style.display = 'none';
        }
        const recoveryOverlay = document.getElementById('webgl_recovery_overlay');
        if (recoveryOverlay) {
            recoveryOverlay.setAttribute('aria-busy', 'false');
            recoveryOverlay.hidden = true;
            recoveryOverlay.style.display = 'none';
        }
        const banner = document.getElementById('safe_mode_banner');
        if (banner) {
            banner.hidden = true;
        }
        const canvas = document.getElementById('the_canvas_id');
        if (canvas) {
            canvas.style.display = 'none';
        }
        const errorContainer = document.getElementById('error_container');
        if (errorContainer) {
            errorContainer.setAttribute('data-error-category', resolved);
            errorContainer.setAttribute('data-error-active', 'true');
            errorContainer.style.display = 'flex';
        }
        const titleEl = document.getElementById('error_title');
        if (titleEl && copy) {
            titleEl.textContent = copy.title;
        }
        const msgEl = document.getElementById('error_message');
        if (msgEl && copy) {
            msgEl.textContent = copy.message;
        }
        if (copy) {
            populateRecoveryList(resolved, copy);
        }
        const detailsEl = document.getElementById('error_details');
        if (detailsEl) {
            detailsEl.textContent = detailText || '';
            detailsEl.style.display = detailText ? 'block' : 'none';
        }
        return syncActionButtons();
    }

    function renderErrorBoundary(category, detailText, currentPhase) {
        const resolvedInfo = resolveErrorCopy(category, detailText, currentPhase);
        const reloadBtn = renderErrorDom(resolvedInfo.resolved, resolvedInfo.copy, detailText);
        announceLive(resolvedInfo.copy.alert, 'assertive');
        reloadBtn?.focus?.();
    }

    const initDom = function () {
        syncActionButtons();
        syncSafeModeBanner();
    };
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initDom, { once: true });
    } else {
        initDom();
    }

    window.__mathExplorerRecovery = {
        ERROR_CATEGORIES: ERROR_CATEGORIES,
        normalizeThemeToken: normalizeThemeToken,
        extractThemePermissive: extractThemePermissive,
        applyShellTheme: applyShellTheme,
        readPreservedTheme: readPreservedTheme,
        quarantineAbortedBoot: quarantineAbortedBoot,
        markBootStarting: markBootStarting,
        markFirstFrameRendered: markFirstFrameRendered,
        hasQuarantinedState: hasQuarantinedState,
        downloadQuarantinedState: downloadQuarantinedState,
        clearQuarantinedState: clearQuarantinedState,
        resetWorkspaceAndReload: resetWorkspaceAndReload,
        announceLive: announceLive,
        syncActionButtons: syncActionButtons,
        syncSafeModeBanner: syncSafeModeBanner,
        resolveErrorCopy: resolveErrorCopy,
        renderErrorDom: renderErrorDom,
        renderErrorBoundary: renderErrorBoundary
    };
})();
