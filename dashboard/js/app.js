/**
 * app.js — Entry point da aplicação
 * Responsável pelo ciclo de vida: loading → fetch de dados → render → error
 *
 * Fluxo:
 *  1. Exibe Loading State
 *  2. Simula fetch assíncrono (substituível por fetch() real de API)
 *  3. Monta todos os componentes
 *  4. Oculta Loading State, exibe App
 *  5. Em caso de erro: exibe Error State
 */

import { pipelineData } from "./data.js";
import {
  renderHeaderChips,
  renderSummaryCards,
  renderConfusionMatrix,
  renderDatasetTable,
  renderFinanceCards,
  renderTargetChart,
  renderFeatureChart,
  initFeatureFilters,
} from "./components.js";

// ── State Management ─────────────────────────────────────────────────────────

const AppState = {
  loading: true,
  error:   null,
  data:    null,
};

// ── DOM Helpers ───────────────────────────────────────────────────────────────

function showLoading() {
  document.getElementById("loading-state")?.classList.remove("hidden");
  document.getElementById("error-state")?.classList.add("hidden");
  document.getElementById("app")?.classList.add("hidden");
}

function showError(message) {
  const errorEl  = document.getElementById("error-state");
  const detailEl = document.getElementById("error-detail");
  if (detailEl) detailEl.textContent = message ?? "Erro desconhecido.";
  document.getElementById("loading-state")?.classList.add("hidden");
  document.getElementById("app")?.classList.add("hidden");
  errorEl?.classList.remove("hidden");
}

function showApp() {
  document.getElementById("loading-state")?.classList.add("hidden");
  document.getElementById("error-state")?.classList.add("hidden");
  document.getElementById("app")?.classList.remove("hidden");
}

// ── Data Fetching ─────────────────────────────────────────────────────────────

/**
 * Simula carregamento assíncrono dos dados.
 * Em produção: substitua por fetch('/api/pipeline-results')
 * @returns {Promise<object>}
 */
async function fetchPipelineData() {
  // Simula latência de rede (300ms) para demonstrar o Loading State
  await new Promise(resolve => setTimeout(resolve, 800));

  // Valida estrutura mínima dos dados (programação defensiva)
  if (!pipelineData?.model || !pipelineData?.feature_importance) {
    throw new Error("Estrutura de dados inválida ou incompleta.");
  }

  return pipelineData;
}

// ── Rendering Orchestrator ───────────────────────────────────────────────────

/**
 * Monta todos os componentes da página em sequência.
 * Chart.js pode não estar carregado ainda — aguarda via polling.
 * @param {object} data
 */
async function mountApp(data) {
  // Aguarda Chart.js estar disponível (carregado via defer no HTML)
  await waitForChart();

  // Componentes síncronos (não dependem de Chart.js)
  renderHeaderChips(data.model);
  renderSummaryCards(data.summary_cards);
  renderConfusionMatrix(data.confusion_matrix);
  renderDatasetTable(data.dataset_overview);
  renderFinanceCards(data.business_assumptions);

  // Componentes com Canvas/Chart.js
  renderTargetChart(data.target_distribution);
  renderFeatureChart(data.feature_importance, "all");

  // Interatividade dos filtros de features
  initFeatureFilters(data.feature_importance);
}

/**
 * Aguarda Chart.js estar disponível no window global.
 * @returns {Promise<void>}
 */
function waitForChart() {
  return new Promise((resolve, reject) => {
    const MAX_TRIES = 50;
    let tries = 0;
    const check = setInterval(() => {
      if (typeof Chart !== "undefined") {
        clearInterval(check);
        resolve();
      } else if (++tries >= MAX_TRIES) {
        clearInterval(check);
        reject(new Error("Chart.js não carregou. Verifique sua conexão."));
      }
    }, 100);
  });
}

// ── Bootstrap ────────────────────────────────────────────────────────────────

async function bootstrap() {
  showLoading();

  try {
    const data    = await fetchPipelineData();
    AppState.data = data;
    await mountApp(data);
    showApp();
  } catch (err) {
    AppState.error = err;
    console.error("[Squad3 Dashboard] Erro fatal:", err);
    showError(err?.message ?? "Falha ao inicializar o dashboard.");
  } finally {
    AppState.loading = false;
  }
}

// Inicia quando o DOM estiver pronto
document.addEventListener("DOMContentLoaded", bootstrap);
