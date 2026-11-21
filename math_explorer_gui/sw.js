// Lightweight pre-boot app-shell Service Worker for Math Explorer GUI.
// Caches only lightweight shell HTML/JS/metadata (excluding the ~4.5 MB .wasm bundle)
// using network-first resolution so offline reloads render #error_container cleanly
// instead of failing navigation with net::ERR_FAILED.
const SHELL_CACHE_NAME = 'math-explorer-shell-v2';
const SHELL_CACHE_PREFIX = 'math-explorer-shell-';

const CORE_SHELL_PATHS = [
    './',
    './index.html',
    './recovery.js',
    './reload.js',
    './bootstrap.js',
    './initializer.js',
    './sw.js',
];

const OPTIONAL_SHELL_PATHS = [
    './site.webmanifest',
    './favicon.ico',
    './favicon.svg',
    './32x32.png',
    './180x180.png',
];

const SHELL_PATH_SET = new Set(
    CORE_SHELL_PATHS.concat(OPTIONAL_SHELL_PATHS).map((relPath) =>
        relPath.replace(/^\.\//, '')
    )
);

function scopeBaseUrl() {
    return (self.registration && self.registration.scope) || self.location.href;
}

function indexCacheKey() {
    return new URL('./index.html', scopeBaseUrl()).href;
}

function normalizeRelativeShellPath(requestUrl) {
    const scopePath = new URL('./', scopeBaseUrl()).pathname;
    if (!requestUrl.pathname.startsWith(scopePath)) {
        return null;
    }
    return requestUrl.pathname.slice(scopePath.length);
}

function isShellNavigationPath(relPath) {
    return relPath === '' || relPath === 'index.html';
}

function isShellAssetPath(relPath) {
    if (relPath === null) {
        return false;
    }
    if (
        /\.wasm$/i.test(relPath) ||
        /\.pdf$/i.test(relPath) ||
        /(^|\/)math_explorer_gui\.js$/i.test(relPath)
    ) {
        return false;
    }
    return SHELL_PATH_SET.has(relPath);
}

function buildOfflineUnavailableResponse() {
    return new Response('Offline: resource unavailable in Math Explorer app shell cache.', {
        status: 503,
        statusText: 'Service Unavailable',
        headers: {
            'Content-Type': 'text/plain; charset=utf-8',
            'Cache-Control': 'no-store',
        },
    });
}

async function precacheShell() {
    const cache = await caches.open(SHELL_CACHE_NAME);
    const base = scopeBaseUrl();

    await Promise.all(
        CORE_SHELL_PATHS.map(async (relPath) => {
            const targetUrl = new URL(relPath, base).href;
            try {
                const response = await fetch(targetUrl, { cache: 'reload' });
                if (response && response.ok) {
                    await cache.put(targetUrl, response);
                }
            } catch (_err) {
                // Best-effort during install; runtime network-first populates cache on online loads
            }
        })
    );

    await Promise.all(
        OPTIONAL_SHELL_PATHS.map(async (relPath) => {
            const targetUrl = new URL(relPath, base).href;
            try {
                const response = await fetch(targetUrl, { cache: 'no-cache' });
                if (response && response.ok) {
                    await cache.put(targetUrl, response);
                }
            } catch (_err) {
                // Optional icon/manifest errors never block service worker installation
            }
        })
    );
}

self.addEventListener('install', (event) => {
    event.waitUntil(
        precacheShell().then(() => self.skipWaiting())
    );
});

self.addEventListener('activate', (event) => {
    event.waitUntil(
        (async () => {
            const keys = await caches.keys();
            await Promise.all(
                keys
                    .filter((key) => key.startsWith(SHELL_CACHE_PREFIX) && key !== SHELL_CACHE_NAME)
                    .map((staleKey) => caches.delete(staleKey))
            );
            await self.clients.claim();
        })()
    );
});

async function handleShellOrNavigateFetch(request, isNavigate) {
    const cacheKey = isNavigate
        ? indexCacheKey()
        : new URL(request.url).origin + new URL(request.url).pathname;

    try {
        const networkRequest = new Request(request, { cache: 'no-cache' });
        const networkResponse = await fetch(networkRequest);
        if (networkResponse && networkResponse.ok) {
            const cache = await caches.open(SHELL_CACHE_NAME);
            await cache.put(cacheKey, networkResponse.clone());
        }
        return networkResponse;
    } catch (_networkErr) {
        const cache = await caches.open(SHELL_CACHE_NAME);
        const cached =
            (await cache.match(cacheKey)) ||
            (await cache.match(request, { ignoreSearch: true })) ||
            (isNavigate ? await cache.match(new URL('./', scopeBaseUrl()).href) : undefined);
        if (cached) {
            return cached;
        }
        return buildOfflineUnavailableResponse();
    }
}

async function handleUncachedRuntimeFetch(request) {
    try {
        const networkRequest = new Request(request, { cache: 'no-cache' });
        return await fetch(networkRequest);
    } catch (_networkErr) {
        return buildOfflineUnavailableResponse();
    }
}

self.addEventListener('fetch', (event) => {
    const request = event.request;
    if (!request || request.method !== 'GET') {
        return;
    }
    if (request.cache === 'only-if-cached' && request.mode !== 'same-origin') {
        return;
    }

    let requestUrl;
    try {
        requestUrl = new URL(request.url);
    } catch (_err) {
        return;
    }
    if (requestUrl.origin !== self.location.origin) {
        return;
    }

    const relPath = normalizeRelativeShellPath(requestUrl);
    const isShellNavigate =
        (request.mode === 'navigate' || request.destination === 'document') &&
        isShellNavigationPath(relPath);

    if (isShellNavigate || isShellAssetPath(relPath)) {
        event.respondWith(handleShellOrNavigateFetch(request, isShellNavigate));
        return;
    }

    event.respondWith(handleUncachedRuntimeFetch(request));
});
