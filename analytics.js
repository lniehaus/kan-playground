(function () {
var MEASUREMENT_ID = 'Add your own analytics ID here';

// Prevent double-initialization
if (window.__gaInitialized) return;
window.__gaInitialized = true;

// Setup dataLayer and gtag
window.dataLayer = window.dataLayer || [];
window.gtag = window.gtag || function () { window.dataLayer.push(arguments); };

// Queue initial events
window.gtag('js', new Date());
window.gtag('config', MEASUREMENT_ID);

// Load the GA library
var s = document.createElement('script');
s.async = true;
s.src = 'https://www.googletagmanager.com/gtag/js?id=' + encodeURIComponent(MEASUREMENT_ID);
(document.head || document.body || document.documentElement).appendChild(s);
})();