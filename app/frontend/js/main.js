/**
 * FallGuard - Shared Frontend JavaScript
 * Handles theme switching (Light/Dark mode), mobile navigation, and interactive UI states.
 */

document.addEventListener("DOMContentLoaded", () => {
  getOrCreateClientId();
  ensureAlertsNavLink();
  initThemeToggle();
  initMobileNav();
  highlightActiveNav();
  initSkeletonSimulation();
});

/**
 * Returns or creates persistent client UUID in localStorage.
 */
function getOrCreateClientId() {
  let cid = localStorage.getItem("fallguard_client_id");
  if (!cid) {
    if (typeof crypto !== "undefined" && crypto.randomUUID) {
      cid = crypto.randomUUID();
    } else {
      cid = "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, function (c) {
        const r = (Math.random() * 16) | 0;
        const v = c === "x" ? r : (r & 0x3) | 0x8;
        return v.toString(16);
      });
    }
    localStorage.setItem("fallguard_client_id", cid);
  }
  return cid;
}
window.getFallGuardClientId = getOrCreateClientId;

/**
 * Appends 'Alert History' link to navigation bar on all pages if not already present.
 */
function ensureAlertsNavLink() {
  const navUl = document.querySelector(".nav-links");
  if (!navUl) return;

  const exists = Array.from(navUl.querySelectorAll("a")).some(
    (a) => a.getAttribute("href") === "alerts.html"
  );

  if (!exists) {
    const li = document.createElement("li");
    const currentPath = window.location.pathname.split("/").pop() || "index.html";
    const activeClass = currentPath === "alerts.html" ? "nav-link active" : "nav-link";
    li.innerHTML = `<a href="alerts.html" class="${activeClass}">Alert History</a>`;
    navUl.appendChild(li);
  }
}

/**
 * Initializes light/dark mode with localStorage persistence and system sync.
 */
function initThemeToggle() {
  const themeToggleBtn = document.getElementById("theme-toggle");
  if (!themeToggleBtn) return;

  const savedTheme = localStorage.getItem("fallguard_theme");
  const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;

  // Apply saved theme or system default
  if (savedTheme) {
    document.documentElement.setAttribute("data-theme", savedTheme);
    updateThemeIcon(themeToggleBtn, savedTheme);
  } else {
    updateThemeIcon(themeToggleBtn, prefersDark ? "dark" : "light");
  }

  // Handle manual toggle
  themeToggleBtn.addEventListener("click", () => {
    const currentTheme = document.documentElement.getAttribute("data-theme") || 
      (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    
    const newTheme = currentTheme === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", newTheme);
    localStorage.setItem("fallguard_theme", newTheme);
    updateThemeIcon(themeToggleBtn, newTheme);
  });

  // Listen for system changes if no explicit local preference was stored
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", (e) => {
    if (!localStorage.getItem("fallguard_theme")) {
      updateThemeIcon(themeToggleBtn, e.matches ? "dark" : "light");
    }
  });
}

/**
 * Updates the theme button SVG icon.
 */
function updateThemeIcon(btn, theme) {
  if (theme === "dark") {
    // Show Sun icon (to switch to light)
    btn.innerHTML = `
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="5"></circle>
        <line x1="12" y1="1" x2="12" y2="3"></line>
        <line x1="12" y1="21" x2="12" y2="23"></line>
        <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line>
        <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line>
        <line x1="1" y1="12" x2="3" y2="12"></line>
        <line x1="21" y1="12" x2="23" y2="12"></line>
        <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line>
        <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>
      </svg>
    `;
    btn.setAttribute("aria-label", "Switch to light mode");
  } else {
    // Show Moon icon (to switch to dark)
    btn.innerHTML = `
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>
      </svg>
    `;
    btn.setAttribute("aria-label", "Switch to dark mode");
  }
}

/**
 * Toggles mobile navbar on hamburger click.
 */
function initMobileNav() {
  const toggleBtn = document.getElementById("mobile-menu-toggle");
  const navLinks = document.querySelector(".nav-links");
  if (!toggleBtn || !navLinks) return;

  toggleBtn.addEventListener("click", () => {
    navLinks.classList.toggle("show");
  });
}

/**
 * Marks active nav link according to current page URL.
 */
function highlightActiveNav() {
  const currentPath = window.location.pathname.split("/").pop() || "index.html";
  const links = document.querySelectorAll(".nav-link");

  links.forEach(link => {
    const href = link.getAttribute("href");
    if (href === currentPath || (currentPath === "" && href === "index.html")) {
      link.classList.add("active");
    } else {
      link.classList.remove("active");
    }
  });
}

/**
 * Subtle simulation for the hero pose preview to demonstrate pose coordinates.
 */
function initSkeletonSimulation() {
  const heroAngleVal = document.getElementById("hero-torso-angle");
  const heroSpeedVal = document.getElementById("hero-hip-speed");
  const heroRatioVal = document.getElementById("hero-aspect-ratio");

  if (!heroAngleVal || !heroSpeedVal || !heroRatioVal) return;

  // Gentle periodic fluctuation to give an "alive" sensing feel
  let tick = 0;
  setInterval(() => {
    tick += 0.1;
    const angle = (12.4 + Math.sin(tick) * 1.8).toFixed(1);
    const speed = (0.04 + Math.abs(Math.cos(tick * 0.8)) * 0.05).toFixed(2);
    const ratio = (0.48 + Math.sin(tick * 0.5) * 0.02).toFixed(2);

    heroAngleVal.textContent = `${angle}°`;
    heroSpeedVal.textContent = `${speed} m/s`;
    heroRatioVal.textContent = ratio;
  }, 1200);
}
