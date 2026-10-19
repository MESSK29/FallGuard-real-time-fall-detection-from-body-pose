/**
 * FallGuard - Alert History Page Logic
 * Fetches and renders alerts for the current client_id, auto-refreshes every 5s,
 * provides CSV download, and allows clearing history.
 */

document.addEventListener("DOMContentLoaded", () => {
  initAlertsPage();
});

let alertsAutoRefreshTimer = null;
let currentAlertsData = [];

function initAlertsPage() {
  const btnRefresh = document.getElementById("btn-refresh-alerts");
  const btnDownloadCsv = document.getElementById("btn-download-csv");
  const btnClearHistory = document.getElementById("btn-clear-history");

  if (btnRefresh) {
    btnRefresh.addEventListener("click", () => fetchAndRenderAlerts());
  }

  if (btnDownloadCsv) {
    btnDownloadCsv.addEventListener("click", () => downloadAlertsCSV());
  }

  if (btnClearHistory) {
    btnClearHistory.addEventListener("click", () => clearAlertsHistory());
  }

  // Initial fetch
  fetchAndRenderAlerts();

  // Auto-refresh every 5 seconds
  alertsAutoRefreshTimer = setInterval(() => {
    fetchAndRenderAlerts(true);
  }, 5000);
}

/**
 * Fetches alerts filtered to current client_id from GET /alerts.
 */
async function fetchAndRenderAlerts(isAutoRefresh = false) {
  const clientId = window.getFallGuardClientId ? window.getFallGuardClientId() : "anonymous";
  const loadingEl = document.getElementById("alerts-loading-state");
  const emptyEl = document.getElementById("alerts-empty-state");
  const tableContainer = document.getElementById("alerts-table-container");
  const tableBody = document.getElementById("alerts-table-body");
  const countBadge = document.getElementById("alerts-count-badge");

  if (!isAutoRefresh && loadingEl && (!currentAlertsData || currentAlertsData.length === 0)) {
    loadingEl.style.display = "flex";
    if (emptyEl) emptyEl.style.display = "none";
    if (tableContainer) tableContainer.style.display = "none";
  }

  try {
    const resp = await fetch("/alerts", {
      headers: {
        "X-Client-Id": clientId
      }
    });

    if (!resp.ok) {
      throw new Error(`HTTP ${resp.status}`);
    }

    const alerts = await resp.json();
    currentAlertsData = alerts;

    if (loadingEl) loadingEl.style.display = "none";

    if (countBadge) {
      countBadge.textContent = `${alerts.length} ${alerts.length === 1 ? "Alert" : "Alerts"}`;
    }

    if (!alerts || alerts.length === 0) {
      if (emptyEl) emptyEl.style.display = "block";
      if (tableContainer) tableContainer.style.display = "none";
      return;
    }

    if (emptyEl) emptyEl.style.display = "none";
    if (tableContainer) tableContainer.style.display = "block";

    renderAlertRows(alerts, tableBody);
  } catch (err) {
    console.error("[FallGuard] Failed to fetch alerts:", err);
    if (loadingEl) loadingEl.style.display = "none";
    if (!currentAlertsData || currentAlertsData.length === 0) {
      if (emptyEl) {
        emptyEl.style.display = "block";
        const emptyMsg = emptyEl.querySelector("p");
        if (emptyMsg) {
          emptyMsg.textContent = "Could not load alerts. Server may be unreachable.";
        }
      }
    }
  }
}

/**
 * Renders HTML table rows for alert records.
 */
function renderAlertRows(alerts, tbody) {
  if (!tbody) return;

  tbody.innerHTML = alerts
    .map((alert) => {
      const probPct = Math.round(Number(alert.probability || 0) * 100);

      // Status pill
      let statusClass = "badge-status confirmed";
      let statusLabel = "Confirmed";
      if (alert.is_test) {
        statusClass = "badge-status test";
        statusLabel = "Test Alert";
      } else if (alert.status === "dismissed") {
        statusClass = "badge-status dismissed";
        statusLabel = "Dismissed";
      }

      // Telegram pill
      const tgClass = alert.telegram_sent ? "badge-tg sent" : "badge-tg unsent";
      const tgLabel = alert.telegram_sent ? "Sent" : "No";

      // Incident Clip pill
      const clipRaw = String(alert.clip_status || "disabled");
      let clipClass = "badge-clip disabled";
      let clipLabel = "Disabled";

      if (clipRaw === "sent") {
        clipClass = "badge-clip sent";
        clipLabel = "Sent";
      } else if (clipRaw === "pending" || clipRaw === "processing") {
        clipClass = "badge-clip pending";
        clipLabel = "Pending";
      } else if (clipRaw.startsWith("skipped")) {
        clipClass = "badge-clip skipped";
        clipLabel = "Skipped";
      } else if (clipRaw.startsWith("failed")) {
        clipClass = "badge-clip failed";
        clipLabel = "Failed";
      } else if (clipRaw.startsWith("expired")) {
        clipClass = "badge-clip skipped";
        clipLabel = "Expired";
      }

      // Dismiss action button (if confirmed and not dismissed)
      let actionBtn = `<span style="color: var(--text-faint); font-size: 0.8rem;">--</span>`;
      if (alert.status === "confirmed" && !alert.is_test) {
        actionBtn = `
          <button class="btn btn-outline btn-xs btn-dismiss-table" onclick="dismissTableAlert(${alert.id})">
            Dismiss
          </button>
        `;
      }

      return `
        <tr>
          <td style="font-weight: 500;">
            <div style="display:flex; align-items:center; gap:0.5rem;">
              <span>${escapeHtml(alert.local_time || alert.time || "Unknown")}</span>
            </div>
            <span style="font-size: 0.72rem; color: var(--text-faint); font-family: var(--font-mono);">${escapeHtml(alert.time || "")}</span>
          </td>
          <td>
            <span class="prob-tag ${probPct >= 80 ? "high" : "med"}">${probPct}%</span>
          </td>
          <td>
            <span class="${statusClass}">${statusLabel}</span>
          </td>
          <td>
            <span class="${tgClass}">${tgLabel}</span>
          </td>
          <td>
            <span class="${clipClass}" title="${escapeHtml(clipRaw)}">${clipLabel}</span>
          </td>
          <td>
            ${actionBtn}
          </td>
        </tr>
      `;
    })
    .join("");
}

/**
 * Dismisses an alert directly from the table.
 */
window.dismissTableAlert = async function (alertId) {
  const clientId = window.getFallGuardClientId ? window.getFallGuardClientId() : "anonymous";
  try {
    const resp = await fetch(`/alerts/${alertId}/dismiss`, {
      method: "POST",
      headers: {
        "X-Client-Id": clientId
      }
    });

    if (resp.ok) {
      fetchAndRenderAlerts();
    } else {
      alert(`Could not dismiss alert #${alertId}.`);
    }
  } catch (err) {
    console.error("Failed to dismiss alert:", err);
    alert("Backend unreachable.");
  }
};

/**
 * Clears alert history for the calling client_id.
 */
async function clearAlertsHistory() {
  if (!confirm("Are you sure you want to clear your alert history? This cannot be undone.")) {
    return;
  }

  const clientId = window.getFallGuardClientId ? window.getFallGuardClientId() : "anonymous";
  try {
    const resp = await fetch("/alerts/clear", {
      method: "POST",
      headers: {
        "X-Client-Id": clientId
      }
    });

    if (resp.ok) {
      currentAlertsData = [];
      fetchAndRenderAlerts();
    } else {
      alert("Could not clear alert history.");
    }
  } catch (err) {
    console.error("Clear alerts failed:", err);
    alert("Backend unreachable.");
  }
}

/**
 * Generates and triggers browser download of alerts as CSV.
 */
function downloadAlertsCSV() {
  if (!currentAlertsData || currentAlertsData.length === 0) {
    alert("No alerts to export.");
    return;
  }

  const headers = ["ID", "Time (UTC)", "Local Time", "Peak Probability (%)", "Status", "Telegram Sent"];
  const rows = currentAlertsData.map((a) => [
    a.id,
    `"${(a.time || "").replace(/"/g, '""')}"`,
    `"${(a.local_time || "").replace(/"/g, '""')}"`,
    Math.round(Number(a.probability || 0) * 100),
    a.is_test ? "Test" : a.status,
    a.telegram_sent ? "Yes" : "No"
  ]);

  const csvContent = [headers.join(","), ...rows.map((r) => r.join(","))].join("\r\n");
  const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  const filename = `fallguard_alerts_${new Date().toISOString().slice(0, 10)}.csv`;

  link.setAttribute("href", url);
  link.setAttribute("download", filename);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

function escapeHtml(text) {
  const map = {
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;"
  };
  return String(text).replace(/[&<>"']/g, (m) => map[m]);
}
