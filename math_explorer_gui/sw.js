// Lightweight post-boot app-shell Service Worker for Math Explorer GUI.
// Serves shell scripts, manifest metadata, and all 7 PWA icons via stale-while-revalidate,
// bounds navigation and reload fetches with a 1,200 ms AbortController timeout, and passes
// non-shell bundle/document requests directly to the browser HTTP cache without synthetic 503s.
const SHELL_CACHE_NAME = 'math-explorer-shell-v1';
const SHELL_CACHE_PREFIX = 'math-explorer-shell-';
const NAVIGATION_FETCH_TIMEOUT_MS = 1200;
const NAV_FALLBACK_COOLDOWN_MS = 5000;

const CORE_SHELL_PATHS = [
    './',
    './index.html',
    './theme-init.js',
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
    './128x128.png',
    './180x180.png',
    './192x192.png',
    './256x256.png',
    './512x512.png',
    './512x512-maskable.png',
];

const SHELL_PATH_SET = new Set(
    CORE_SHELL_PATHS.concat(OPTIONAL_SHELL_PATHS).map((relPath) =>
        relPath.replace(/^\.\//, '')
    )
);

let lastNavigationFallbackAt = 0;

function scopeBaseUrl() {
    return (self.registration && self.registration.scope) || self.location.href;
}

function rootCacheKey() {
    return new URL('./', scopeBaseUrl()).href;
}

function indexCacheKey() {
    return new URL('./index.html', scopeBaseUrl()).href;
}

function assetCacheKey(requestUrl) {
    return requestUrl.origin + requestUrl.pathname;
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
    if (relPath === null || isShellNavigationPath(relPath)) {
        return false;
    }
    if (
        /\.wasm$/i.test(relPath) ||
        /\.pdf$/i.test(relPath) ||
        /\.json$/i.test(relPath) ||
        /(^|\/)math_explorer_gui\.js$/i.test(relPath)
    ) {
        return false;
    }
    return SHELL_PATH_SET.has(relPath);
}

function isExplicitReloadRequest(request) {
    return request.cache === 'reload' || request.cache === 'no-cache';
}

function isOfflineOrRecentNavFallback() {
    if (self.navigator && self.navigator.onLine === false) {
        return true;
    }
    return (
        lastNavigationFallbackAt > 0 &&
        Date.now() - lastNavigationFallbackAt < NAV_FALLBACK_COOLDOWN_MS
    );
}

function buildDeduplicatedPrecachePaths(preferIndexHtml) {
    const seen = new Set();
    const deduped = [];
    const canonicalDocPath = preferIndexHtml ? './index.html' : './';

    for (const relPath of CORE_SHELL_PATHS.concat(OPTIONAL_SHELL_PATHS)) {
        const stripped = relPath.replace(/^\.\//, '');
        if (isShellNavigationPath(stripped)) {
            if (!seen.has('__shell_document__')) {
                seen.add('__shell_document__');
                deduped.push(canonicalDocPath);
            }
            continue;
        }
        if (!seen.has(stripped)) {
            seen.add(stripped);
            deduped.push(relPath);
        }
    }
    return deduped;
}

async function detectClientPrefersIndexHtml() {
    if (!self.clients || typeof self.clients.matchAll !== 'function') {
        return false;
    }
    try {
        const windowClients = await self.clients.matchAll({
            type: 'window',
            includeUncontrolled: true,
        });
        for (const client of windowClients) {
            if (client && typeof client.url === 'string') {
                const parsed = new URL(client.url);
                if (parsed.pathname.endsWith('/index.html')) {
                    return true;
                }
            }
        }
    } catch (_err) {
        // Default to root document URL when client inspection fails
    }
    return false;
}

async function storeShellDocumentInCache(cache, response) {
    const rootUrl = rootCacheKey();
    const indexUrl = indexCacheKey();
    if (rootUrl === indexUrl) {
        await cache.put(indexUrl, response);
        return;
    }
    await Promise.all([
        cache.put(rootUrl, response.clone()),
        cache.put(indexUrl, response),
    ]);
}

async function precacheSinglePath(cache, base, relPath) {
    const stripped = relPath.replace(/^\.\//, '');
    const isDoc = isShellNavigationPath(stripped);
    const targetUrl = new URL(relPath, base).href;
    try {
        const response = await fetch(targetUrl);
        if (response && response.ok) {
            if (isDoc) {
                await storeShellDocumentInCache(cache, response);
            } else {
                await cache.put(targetUrl, response);
            }
        }
    } catch (_err) {
        // Best-effort during install; runtime fetch populates cache on online loads
    }
}

async function precacheShell() {
    const [cache, preferIndexHtml] = await Promise.all([
        caches.open(SHELL_CACHE_NAME),
        detectClientPrefersIndexHtml(),
    ]);
    const base = scopeBaseUrl();
    const paths = buildDeduplicatedPrecachePaths(preferIndexHtml);
    await Promise.all(paths.map((relPath) => precacheSinglePath(cache, base, relPath)));
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

function fetchWithTimeout(request, timeoutMs) {
    if (typeof AbortController === 'undefined') {
        return fetch(request);
    }
    const controller = new AbortController();
    const timerId = setTimeout(() => {
        controller.abort();
    }, timeoutMs);
    const fallbackUrlFetch = () =>
        fetch(request.url, {
            cache: request.cache || 'default',
            credentials: request.credentials || 'same-origin',
            signal: controller.signal,
        });
    let fetchPromise;
    try {
        fetchPromise = fetch(request, { signal: controller.signal }).catch((err) => {
            if (
                !controller.signal.aborted &&
                request.mode === 'navigate' &&
                err instanceof TypeError &&
                /navigate/i.test(String(err.message || ''))
            ) {
                return fallbackUrlFetch();
            }
            throw err;
        });
    } catch (_navigateModeErr) {
        fetchPromise = fallbackUrlFetch();
    }
    return fetchPromise.finally(() => {
        clearTimeout(timerId);
    });
}

async function matchCachedNavigation(cache, request) {
    return (
        (await cache.match(indexCacheKey())) ||
        (await cache.match(rootCacheKey())) ||
        (await cache.match(request, { ignoreSearch: true }))
    );
}

async function matchCachedShellAsset(cache, cacheKey, request) {
    return (
        (await cache.match(cacheKey)) ||
        (await cache.match(request, { ignoreSearch: true }))
    );
}

async function handleShellNavigationFetch(request) {
    try {
        const networkResponse = await fetchWithTimeout(request, NAVIGATION_FETCH_TIMEOUT_MS);
        if (networkResponse && networkResponse.ok) {
            lastNavigationFallbackAt = 0;
            try {
                const cache = await caches.open(SHELL_CACHE_NAME);
                await storeShellDocumentInCache(cache, networkResponse.clone());
            } catch (_cacheErr) {
                // Storage quota errors must never discard a valid network response
            }
        }
        return networkResponse;
    } catch (networkErr) {
        lastNavigationFallbackAt = Date.now();
        const cache = await caches.open(SHELL_CACHE_NAME);
        const cached = await matchCachedNavigation(cache, request);
        if (cached) {
            return cached;
        }
        throw networkErr;
    }
}

async function revalidateShellAsset(request, cacheKey) {
    if (self.navigator && self.navigator.onLine === false) {
        return null;
    }
    const networkResponse = await fetchWithTimeout(request, NAVIGATION_FETCH_TIMEOUT_MS);
    if (networkResponse && networkResponse.ok) {
        try {
            const cache = await caches.open(SHELL_CACHE_NAME);
            await cache.put(cacheKey, networkResponse.clone());
        } catch (_cacheErr) {
            // Ignore CacheStorage quota errors during background revalidation
        }
    }
    return networkResponse;
}

async function handleShellAssetFetch(event, request, requestUrl) {
    const cacheKey = assetCacheKey(requestUrl);

    if (isExplicitReloadRequest(request) && !isOfflineOrRecentNavFallback()) {
        try {
            const freshResponse = await fetchWithTimeout(request, NAVIGATION_FETCH_TIMEOUT_MS);
            if (freshResponse && freshResponse.ok) {
                try {
                    const cache = await caches.open(SHELL_CACHE_NAME);
                    await cache.put(cacheKey, freshResponse.clone());
                } catch (_cacheErr) {
                    // Storage quota errors must never discard a valid network response
                }
            }
            return freshResponse;
        } catch (networkErr) {
            const cache = await caches.open(SHELL_CACHE_NAME);
            const fallbackCached = await matchCachedShellAsset(cache, cacheKey, request);
            if (fallbackCached) {
                return fallbackCached;
            }
            throw networkErr;
        }
    }

    const cache = await caches.open(SHELL_CACHE_NAME);
    const cached = await matchCachedShellAsset(cache, cacheKey, request);
    if (cached) {
        if (!isOfflineOrRecentNavFallback()) {
            event.waitUntil(
                revalidateShellAsset(request, cacheKey).catch(() => {
                    // Background revalidation errors must never reject an already-served shell asset
                })
            );
        }
        return cached;
    }

    const networkResponse = await revalidateShellAsset(request, cacheKey);
    if (networkResponse) {
        return networkResponse;
    }
    return fetch(request);
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

    if (isShellNavigate) {
        event.respondWith(handleShellNavigationFetch(request));
        return;
    }

    if (isShellAssetPath(relPath)) {
        event.respondWith(handleShellAssetFetch(event, request, requestUrl));
        return;
    }

    // Pass non-shell requests (WebAssembly bundles, JSON manifests, PDFs) directly to the
    // browser HTTP cache and native network stack without synthetic HTTP 503 responses.
});
