// Synchronous pre-boot theme synchronization script for Math Explorer GUI under strict CSP ('self').
// Executes in <head> immediately after critical styles so saved localStorage["app"] themes apply
// before <body> first-contentful-paint with zero OS theme flash.
(function () {
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

    window.__mathExplorerSyncShellTheme = syncShellThemeFromStorage;
    syncShellThemeFromStorage();
})();
