/**
 * components.js — Componentes modulares da UI
 * Cada função é responsável por um único pedaço da interface.
 * Programação defensiva: optional chaining + fallbacks em todos os dados.
 */

// ── Utilitários ──────────────────────────────────────────────────────────────

/**
 * Formata número com separadores de milhar (pt-BR)
 * @param {number|null|undefined} value
 * @param {string} fallback
 */
export function fmtNumber(value, fallback = "—") {
  if (value == null || isNaN(value)) return fallback;
  return value.toLocaleString("pt-BR");
}

/**
 * Retorna cor de barra baseada na % de nulos
 * @param {number} pct
 */
function nullColor(pct) {
  if (pct === 0)    return "var(--accent-green)";
  if (pct < 5)      return "var(--accent-blue)";
  if (pct < 20)     return "var(--accent-yellow)";
  return "var(--accent-red)";
}

// ── Componente: Header Chips ────────────────────────────────────────────────
/**
 * Preenche os chips do header com as métricas principais.
 * @param {{ auc_roc, threshold, total_features }} model
 */
export function renderHeaderChips(model) {
  const safeGet = (val, prefix = "", suffix = "", decimals = 4) => {
    if (val == null) return "—";
    return `${prefix}${Number(val).toFixed(decimals)}${suffix}`;
  };

  const aucEl       = document.getElementById("auc-chip");
  const thrEl       = document.getElementById("threshold-chip");
  const featEl      = document.getElementById("features-chip");

  if (aucEl)  aucEl.textContent  = `AUC-ROC: ${safeGet(model?.auc_roc)}`;
  if (thrEl)  thrEl.textContent  = `Threshold: ${safeGet(model?.threshold, "", "", 3)}`;
  if (featEl) featEl.textContent = `Features: ${model?.total_features ?? "—"}`;
}

// ── Componente: Summary Cards ───────────────────────────────────────────────
/**
 * Renderiza cards de métricas resumo.
 * @param {Array} cards
 */
export function renderSummaryCards(cards) {
  const container = document.getElementById("summary-cards");
  if (!container) return;

  container.innerHTML = (cards ?? []).map(card => `
    <div class="summary-card" role="listitem">
      <span class="summary-card__label">${card?.label ?? "—"}</span>
      <span class="summary-card__value ${card?.color ?? ""}">${card?.value ?? "—"}</span>
      <span class="summary-card__sub">${card?.sub ?? ""}</span>
    </div>
  `).join("");
}

// ── Componente: Confusion Matrix ────────────────────────────────────────────
/**
 * Renderiza a matriz de confusão 2×2.
 * @param {{ tn, fp, fn, tp, threshold }} cm
 */
export function renderConfusionMatrix(cm) {
  const el  = document.getElementById("confusion-matrix");
  const tag = document.getElementById("confusion-threshold-tag");
  const leg = document.getElementById("confusion-legend");
  if (!el) return;

  const tn = fmtNumber(cm?.tn);
  const fp = fmtNumber(cm?.fp);
  const fn = fmtNumber(cm?.fn);
  const tp = fmtNumber(cm?.tp);
  const total = (cm?.tn ?? 0) + (cm?.fp ?? 0) + (cm?.fn ?? 0) + (cm?.tp ?? 0);

  if (tag) tag.textContent = `Threshold = ${cm?.threshold?.toFixed(3) ?? "—"}`;

  el.innerHTML = `
    <div></div>
    <div class="cm-header cm-header--col">Predito: Adimplente</div>
    <div class="cm-header cm-header--col">Predito: Inadimplente</div>

    <div class="cm-header cm-header--row">Real: Adimplente</div>
    <div class="cm-cell cm-cell--tn" role="cell">
      <span class="cm-cell__value color--green">${tn}</span>
      <span class="cm-cell__label color--green">TN</span>
      <span class="cm-cell__sub">Adimplente correto</span>
    </div>
    <div class="cm-cell cm-cell--fp" role="cell">
      <span class="cm-cell__value color--red">${fp}</span>
      <span class="cm-cell__label color--red">FP</span>
      <span class="cm-cell__sub">Bom cliente rejeitado</span>
    </div>

    <div class="cm-header cm-header--row">Real: Inadimplente</div>
    <div class="cm-cell cm-cell--fn" role="cell">
      <span class="cm-cell__value color--yellow">${fn}</span>
      <span class="cm-cell__label color--yellow">FN</span>
      <span class="cm-cell__sub">Inadimplente aprovado</span>
    </div>
    <div class="cm-cell cm-cell--tp" role="cell">
      <span class="cm-cell__value color--blue">${tp}</span>
      <span class="cm-cell__label color--blue">TP</span>
      <span class="cm-cell__sub">Inadimplente bloqueado</span>
    </div>
  `;

  if (leg) {
    leg.innerHTML = [
      { color: "var(--accent-green)",  label: "TN — Acerto (adimplente)" },
      { color: "var(--accent-blue)",   label: "TP — Acerto (inadimplente)" },
      { color: "var(--accent-red)",    label: "FP — Receita perdida" },
      { color: "var(--accent-yellow)", label: "FN — Crédito em risco" },
    ].map(({ color, label }) => `
      <div class="legend-item">
        <span class="legend-dot" style="background:${color}"></span>
        <span>${label}</span>
      </div>
    `).join("");
  }
}

// ── Componente: Dataset Table ───────────────────────────────────────────────
/**
 * Renderiza a tabela de visão geral do dataset.
 * @param {Array} rows
 */
export function renderDatasetTable(rows) {
  const tbody = document.getElementById("dataset-tbody");
  if (!tbody) return;

  tbody.innerHTML = (rows ?? []).map(row => {
    const nullPct = row?.nullPct ?? 0;
    const color   = nullColor(nullPct);

    return `
      <tr>
        <td class="file-name">${row?.file ?? "—"}</td>
        <td class="text-right">${fmtNumber(row?.rows)}</td>
        <td class="text-right">${fmtNumber(row?.cols)}</td>
        <td><span class="key-badge">${row?.key ?? "—"}</span></td>
        <td>
          <div class="null-bar-wrap">
            <div class="null-bar-bg">
              <div
                class="null-bar-fill"
                style="width:${Math.min(nullPct, 100)}%; background:${color};"
                role="progressbar"
                aria-valuenow="${nullPct}"
                aria-valuemin="0"
                aria-valuemax="100"
                aria-label="${nullPct}% de valores nulos"
              ></div>
            </div>
            <span class="null-bar-pct" style="color:${color}">${nullPct.toFixed(1)}%</span>
          </div>
        </td>
        <td class="desc-text">${row?.desc ?? "—"}</td>
      </tr>
    `;
  }).join("");
}

// ── Componente: Finance Cards ───────────────────────────────────────────────
/**
 * Renderiza os cards de premissas financeiras de negócio.
 * @param {Array} items
 */
export function renderFinanceCards(items) {
  const container = document.getElementById("finance-cards");
  if (!container) return;

  container.innerHTML = (items ?? []).map(item => `
    <div class="finance-card" role="listitem">
      <span class="finance-card__icon" aria-hidden="true">${item?.icon ?? ""}</span>
      <span class="finance-card__label">${item?.label ?? "—"}</span>
      <span class="finance-card__value" style="color:${item?.color ?? "inherit"}">${item?.value ?? "—"}</span>
      <span class="finance-card__desc">${item?.desc ?? ""}</span>
    </div>
  `).join("");
}

// ── Componente: Chart — Distribuição TARGET ─────────────────────────────────
/**
 * Renderiza o gráfico de barras da distribuição de classes.
 * @param {{ adimplente, inadimplente, total }} dist
 */
export function renderTargetChart(dist) {
  const canvas = document.getElementById("chart-target");
  if (!canvas || typeof Chart === "undefined") return;

  const adm = dist?.adimplente?.count ?? 0;
  const inad = dist?.inadimplente?.count ?? 0;
  const total = dist?.total || (adm + inad) || 1;

  new Chart(canvas, {
    type: "bar",
    data: {
      labels: [
        `Adimplente (0)\n${((adm/total)*100).toFixed(1)}%`,
        `Inadimplente (1)\n${((inad/total)*100).toFixed(1)}%`,
      ],
      datasets: [{
        data: [adm, inad],
        backgroundColor: ["rgba(63,185,80,.7)", "rgba(248,81,73,.7)"],
        borderColor:     ["#3fb950", "#f85149"],
        borderWidth: 1,
        borderRadius: 6,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => ` ${fmtNumber(ctx.raw)} registros (${((ctx.raw/total)*100).toFixed(2)}%)`,
          },
        },
      },
      scales: {
        x: {
          ticks: { color: "#8b949e", font: { size: 11 } },
          grid:  { color: "#21262d" },
        },
        y: {
          ticks: {
            color: "#8b949e",
            font: { size: 11 },
            callback: v => fmtNumber(v),
          },
          grid: { color: "#21262d" },
        },
      },
    },
  });
}

// ── Componente: Chart — Feature Importance ──────────────────────────────────
let featureChartInstance = null;

/**
 * Renderiza o gráfico de importância de features com suporte a filtro.
 * @param {Array} features — lista de { feature, importance, source }
 * @param {string} filter — "all" | "Periférica" | "Principal"
 */
export function renderFeatureChart(features, filter = "all") {
  const canvas = document.getElementById("chart-features");
  if (!canvas || typeof Chart === "undefined") return;

  const filtered = (features ?? []).filter(f =>
    filter === "all" || f?.source === filter
  ).slice(0, 25);

  const labels     = filtered.map(f => f?.feature ?? "—");
  const values     = filtered.map(f => f?.importance ?? 0);
  const colors     = filtered.map(f =>
    f?.source === "Periférica" ? "rgba(248,81,73,.75)" : "rgba(88,166,255,.75)"
  );
  const borders    = filtered.map(f =>
    f?.source === "Periférica" ? "#f85149" : "#58a6ff"
  );

  if (featureChartInstance) {
    featureChartInstance.destroy();
    featureChartInstance = null;
  }

  featureChartInstance = new Chart(canvas, {
    type: "bar",
    data: {
      labels,
      datasets: [{
        data: values,
        backgroundColor: colors,
        borderColor: borders,
        borderWidth: 1,
        borderRadius: 4,
      }],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => ` Importância: ${fmtNumber(ctx.raw)}`,
            afterLabel: ctx => {
              const src = filtered[ctx.dataIndex]?.source ?? "—";
              return ` Origem: ${src}`;
            },
          },
        },
      },
      scales: {
        x: {
          ticks: { color: "#8b949e", font: { size: 10 } },
          grid:  { color: "#21262d" },
        },
        y: {
          ticks: { color: "#c9d1d9", font: { size: 11 } },
          grid:  { color: "#21262d" },
        },
      },
    },
  });
}

// ── Componente: Filter Buttons ──────────────────────────────────────────────
/**
 * Inicializa os botões de filtro de features.
 * @param {Array} features
 */
export function initFeatureFilters(features) {
  const buttons = document.querySelectorAll(".filter-btn");

  buttons.forEach(btn => {
    btn.addEventListener("click", () => {
      buttons.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      renderFeatureChart(features, btn.dataset.filter ?? "all");
    });
  });
}
