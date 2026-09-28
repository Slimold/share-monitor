"use strict";

const PAGE_DIRECTORY = new URL("./", window.location.href);
const DATA_DIRECTORY = new URL(
  PAGE_DIRECTORY.pathname.endsWith("/web/") ? "../data/" : "./data/",
  PAGE_DIRECTORY
);
const SNAPSHOT_URL = new URL("us_tech_snapshot.json", DATA_DIRECTORY).href;
const BACKTEST_URL = new URL("backtest_results.json", DATA_DIRECTORY).href;
const SITE_STALE_AFTER_HOURS = 72;
const FILE_FETCH_TIMEOUT_MS = 20_000;

const LABELS = {
  regimes: {
    risk_on: "风险偏好开启",
    neutral: "中性环境",
    risk_off: "风险规避",
    unavailable: "数据不足",
  },
  actions: {
    buy_eligible_with_stock_confirmation: "个股确认后允许建仓",
    selective_or_smaller_positions: "精选标的，控制仓位",
    reduce_risk_and_avoid_new_buys: "降低风险，暂停新买入",
    wait_for_data: "等待完整数据",
  },
  reasons: {
    insufficient_history: "历史数据不足，尚不能形成可靠分数",
    macro_conditions_tight: "实际利率或金融条件处于偏紧区间",
    macro_conditions_supportive: "宏观金融条件相对支持风险资产",
    volatility_elevated: "隐含或实现波动率处于偏高区间",
    volatility_subdued: "市场波动率处于相对低位",
    technology_trend_weak: "科技板块趋势和相对强弱偏弱",
    technology_trend_supportive: "科技板块趋势和相对强弱偏强",
    qqq_above_200dma: "QQQ 位于 200 日均线上方",
    qqq_below_200dma: "QQQ 位于 200 日均线下方",
    semiconductors_leading_qqq: "半导体近 60 日相对领先 QQQ",
    semiconductors_lagging_qqq: "半导体近 60 日相对落后 QQQ",
    market_risk_off: "市场风险闸门处于关闭状态",
    weak_stock_confirmation: "个股趋势或相对强弱未能确认",
    stock_trend_still_intact: "个股长期趋势仍然完整",
    trend_aligned: "20、50、200 日趋势顺序一致",
    relative_strength_positive: "相对 QQQ 的 60 日强弱为正",
    long_term_trend_intact: "价格仍位于 50 日和 200 日均线上方",
    relative_strength_not_confirmed: "相对强弱尚未确认",
    below_200dma: "价格跌破 200 日均线",
    relative_strength_negative: "相对 QQQ 的 60 日强弱为负",
    mixed_stock_signals: "个股信号相互冲突",
    less_than_200_sessions: "有效历史少于 200 个交易日",
  },
  stockStatuses: {
    buy_eligible: "允许建仓",
    small_position_only: "仅限小仓",
    hold: "持有观察",
    hold_no_new_position: "持有，不开新仓",
    reduce_watch: "减仓关注",
    observe_or_reduce: "观察或减仓",
    exit_condition_triggered: "退出条件触发",
    insufficient_data: "数据不足",
  },
  strategyDecisions: {
    buy_mu_starter: "MU 试探仓候选",
    add_mu_after_reversal: "MU 反转加仓候选",
    reduce_mu_buy_soxs: "减持 MU / SOXS 候选",
    exit_mu_if_held: "若持有 MU：退出条件满足",
    exit_soxs_if_held: "若持有 SOXS：退出条件满足",
    wait_for_confirmation: "等待确认",
    data_unavailable: "策略数据不可用",
  },
  strategyReasons: {
    mu_rsi_oversold: "MU RSI(6) 进入超卖区",
    downside_stretch_extreme: "MU 相对自身趋势与 SMH 明显超跌",
    mu_reversal_confirmed: "MU RSI 与价格出现反转确认",
    sector_rebound_confirmed: "半导体板块同步回升",
    mu_overbought_rolled_over: "MU 从近期超买状态回落",
    smh_weakness_confirmed: "SMH 板块弱势得到确认",
    shock_cooldown_active: "冲击日后的 5 个交易日冷却仍在生效",
    mu_exit_rsi_above_50: "MU RSI(6) 已高于原策略退出线 50",
    soxs_exit_condition_met: "SOXS 退出条件已经满足",
    smh_below_ema10_and_5d_negative: "SMH 低于 EMA10 且 5 日收益为负",
    soxs_above_ema5: "SOXS 位于 EMA5 上方",
    market_risk_elevated: "市场温度计处于较高风险区",
    market_risk_not_extreme: "市场风险尚未进入极端区",
    waiting_for_extreme_and_confirmation: "等待 RSI 极值与板块方向共同确认",
    strategy_inputs_not_aligned: "MU、SMH、SOXS 与市场风险数据日期未对齐",
  },
  components: {
    macro_risk: "宏观风险",
    volatility_risk: "波动风险",
    trend_risk: "趋势风险",
  },
  latest: {
    QQQ: "QQQ",
    VXN: "VXN",
    VXNCLS: "VXN",
    REAL10Y: "10年实际利率",
    DFII10: "10年实际利率",
    NFCI: "NFCI",
    CASH3M: "3月国债收益率",
    DGS3MO: "3月国债收益率",
  },
};

const state = {
  riskHistory: [],
  equityHistory: [],
  resizeObserver: null,
  loading: false,
  hasRendered: false,
  snapshotRaw: null,
  backtestRaw: null,
};

const byId = (id) => document.getElementById(id);

function firstDefined(...values) {
  return values.find((value) => value !== undefined && value !== null);
}

function finiteNumber(value) {
  if (value === null || value === undefined || value === "") return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

function asArray(value) {
  if (Array.isArray(value)) return value;
  if (value && typeof value === "object") {
    return Object.entries(value).map(([key, item]) => {
      if (item && typeof item === "object" && !Array.isArray(item)) {
        return { key, ...item };
      }
      return { key, value: item };
    });
  }
  return [];
}

function text(element, value, fallback = "--") {
  element.textContent = value === null || value === undefined || value === "" ? fallback : String(value);
}

function formatNumber(value, digits = 2) {
  const number = finiteNumber(value);
  if (number === null) return "--";
  return new Intl.NumberFormat("zh-CN", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(number);
}

function formatFlexibleNumber(value) {
  const number = finiteNumber(value);
  if (number === null) return "--";
  const magnitude = Math.abs(number);
  const digits = magnitude >= 100 ? 2 : magnitude >= 10 ? 2 : 3;
  return formatNumber(number, digits);
}

function formatPercent(value, digits = 1) {
  const number = finiteNumber(value);
  if (number === null) return "--";
  return new Intl.NumberFormat("zh-CN", {
    style: "percent",
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(number);
}

function formatDate(value) {
  if (!value) return "--";
  const raw = String(value);
  const match = raw.match(/^\d{4}-\d{2}-\d{2}/);
  return match ? match[0] : raw;
}

function formatDateTime(value) {
  if (!value) return "--";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return new Intl.DateTimeFormat("zh-CN", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    timeZoneName: "short",
  }).format(date);
}

function ageInHours(value) {
  if (!value) return null;
  const timestamp = new Date(value).getTime();
  if (!Number.isFinite(timestamp)) return null;
  return Math.max(0, (Date.now() - timestamp) / 3_600_000);
}

function riskTone(score) {
  const value = finiteNumber(score);
  if (value === null) return "unknown";
  if (value < 40) return "positive";
  if (value < 65) return "warning";
  return "negative";
}

function stockTone(status) {
  if (["buy_eligible", "hold"].includes(status)) return "positive";
  if (["small_position_only", "hold_no_new_position", "observe_or_reduce"].includes(status)) {
    return "warning";
  }
  if (["reduce_watch", "exit_condition_triggered"].includes(status)) return "negative";
  return "unknown";
}

function sourceLabel(source) {
  const labels = {
    network: "网络获取",
    cache: "缓存命中",
    cache_fallback: "缓存回退",
  };
  return labels[source] || source || "已读取";
}

function createElement(tag, className, content) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (content !== undefined) text(element, content, "");
  return element;
}

function normalizeSnapshot(raw = {}) {
  const market = raw.market || raw.snapshot || raw.latest_snapshot || raw;
  const components = market.components || raw.component_scores || {
    macro_risk: firstDefined(market.macro_risk, raw.macro_risk),
    volatility_risk: firstDefined(market.volatility_risk, raw.volatility_risk),
    trend_risk: firstDefined(market.trend_risk, raw.trend_risk),
  };
  const quality = raw.quality || raw.data_quality || {};
  return {
    generatedAt: firstDefined(raw.generated_at_utc, raw.generated_at, raw.updated_at),
    asOf: firstDefined(raw.as_of_market_date, market.as_of, raw.as_of),
    window: raw.window || {},
    riskScore: finiteNumber(firstDefined(market.risk_score, raw.risk_score)),
    regime: String(firstDefined(market.regime, raw.regime, "unavailable")),
    action: String(firstDefined(market.action, raw.action, "wait_for_data")),
    targetPosition: finiteNumber(firstDefined(market.target_position, raw.target_position)),
    reasons: asArray(firstDefined(market.reasons, raw.reasons, [])).map((item) =>
      typeof item === "string" ? item : firstDefined(item.reason, item.key, item.value)
    ),
    components,
    latest: market.latest || raw.latest || raw.latest_values || {},
    history: asArray(raw.history || market.history || raw.risk_history),
    indicators: raw.indicators || market.indicators || null,
    watchlist: asArray(raw.watchlist || raw.stocks || market.watchlist),
    muSoxsStrategy: firstDefined(
      raw.strategies?.mu_soxs,
      raw.mu_soxs_strategy,
      market.strategies?.mu_soxs
    ),
    quality: {
      status: String(firstDefined(quality.status, raw.status, "unknown")),
      staleSources: asArray(quality.stale_sources || quality.staleSources).map((item) =>
        typeof item === "string" ? item : firstDefined(item.name, item.key, item.value)
      ),
      sourceMetadata: quality.source_metadata || quality.sources || raw.source_metadata || {},
    },
  };
}

function unwrapMetrics(value) {
  if (!value || typeof value !== "object") return {};
  return value.metrics && typeof value.metrics === "object" ? value.metrics : value;
}

function normalizePeriod(period = {}) {
  const nested = period.metrics || {};
  return {
    start: firstDefined(period.start, nested.start),
    end: firstDefined(period.end, nested.end),
    annualization: firstDefined(period.annualization, nested.annualization),
    transactionCostBps: firstDefined(period.transaction_cost_bps, nested.transaction_cost_bps),
    strategy: unwrapMetrics(period.strategy || nested.strategy),
    benchmark: unwrapMetrics(period.benchmark || nested.benchmark),
    comparison: period.comparison || nested.comparison || {},
  };
}

function normalizeEquity(points) {
  return asArray(points)
    .map((point) => ({
      date: firstDefined(point.date, point.timestamp, point.as_of, point.key),
      strategy: finiteNumber(
        firstDefined(point.strategy, point.strategy_equity, point.strategy_value, point.value)
      ),
      benchmark: finiteNumber(
        firstDefined(point.benchmark, point.benchmark_equity, point.qqq, point.qqq_equity)
      ),
    }))
    .filter((point) => point.date && (point.strategy !== null || point.benchmark !== null));
}

function normalizeBacktest(raw = {}) {
  const full = normalizePeriod(raw.full || raw.metrics?.full || raw);
  const holdout = normalizePeriod(raw.holdout || raw.metrics?.holdout || {});
  const equity = normalizeEquity(
    raw.equity_curve || raw.full_equity_curve || raw.full?.equity_curve || raw.full?.equity
  );
  const holdoutEquity = normalizeEquity(
    raw.holdout_equity_curve || raw.holdout?.equity_curve || raw.holdout?.equity
  );
  const experiments = Object.entries(raw.experiments || {}).map(([name, experiment]) => ({
    name,
    full: normalizePeriod(experiment.full || {}),
    holdout: normalizePeriod(experiment.holdout || {}),
  }));
  return {
    generatedAt: firstDefined(raw.generated_at_utc, raw.generated_at),
    window: raw.window || {},
    assumptions: raw.assumptions || {},
    full,
    holdout,
    equity,
    holdoutEquity,
    experiments,
    validation: raw.validation || {},
  };
}

async function fetchJson(url) {
  const requestUrl = new URL(url, window.location.href);
  const fileName = requestUrl.pathname.split("/").filter(Boolean).at(-1) || "数据文件";
  requestUrl.searchParams.set("refresh", String(Date.now()));
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), FILE_FETCH_TIMEOUT_MS);
  try {
    const response = await fetch(requestUrl, { cache: "no-store", signal: controller.signal });
    if (!response.ok) throw new Error(`${fileName} 返回 HTTP ${response.status}`);
    try {
      return await response.json();
    } catch (error) {
      if (error?.name === "AbortError") throw error;
      throw new Error(`${fileName} 不是有效的 JSON: ${error.message}`);
    }
  } catch (error) {
    if (error?.name === "AbortError") {
      throw new Error(`${fileName} 读取超过 ${FILE_FETCH_TIMEOUT_MS / 1000} 秒`);
    }
    if (error?.message?.startsWith(`${fileName} `)) throw error;
    throw new Error(`${fileName} 读取失败: ${error?.message || String(error)}`);
  } finally {
    window.clearTimeout(timeout);
  }
}

async function loadDashboard() {
  if (state.loading) return;
  state.loading = true;

  const isInitialLoad = !state.hasRendered;
  const reloadButton = byId("reload-button");
  const retryButton = byId("error-retry-button");
  const headerQuality = byId("header-quality");
  reloadButton.disabled = true;
  retryButton.disabled = true;
  text(reloadButton, "读取中…");
  byId("main-content").setAttribute("aria-busy", "true");
  byId("loading-panel").hidden = !isInitialLoad;
  byId("error-panel").hidden = true;
  if (isInitialLoad) byId("dashboard").hidden = true;
  headerQuality.dataset.tone = "loading";
  text(headerQuality, "正在读取数据");
  headerQuality.removeAttribute("title");
  hideLoadStatus();

  try {
    const [snapshotResult, backtestResult] = await Promise.allSettled([
      fetchJson(SNAPSHOT_URL),
      fetchJson(BACKTEST_URL),
    ]);

    if (snapshotResult.status === "rejected" && backtestResult.status === "rejected") {
      const message = `市场快照与回测结果均读取失败。${snapshotResult.reason.message}；${backtestResult.reason.message}`;
      byId("loading-panel").hidden = true;
      headerQuality.dataset.tone = "error";
      headerQuality.title = message;
      if (state.hasRendered) {
        text(headerQuality, "刷新失败，保留上次数据");
        byId("dashboard").hidden = false;
        showLoadStatus("error", "刷新失败，已保留上次数据", message);
      } else {
        byId("error-panel").hidden = false;
        text(byId("error-message"), message);
        text(headerQuality, "数据不可用");
      }
      return;
    }

    const errors = [];
    if (snapshotResult.status === "fulfilled") state.snapshotRaw = snapshotResult.value;
    else errors.push(`市场快照：${snapshotResult.reason.message}`);
    if (backtestResult.status === "fulfilled") state.backtestRaw = backtestResult.value;
    else errors.push(`回测结果：${backtestResult.reason.message}`);

    renderDashboard(
      normalizeSnapshot(state.snapshotRaw || {}),
      normalizeBacktest(state.backtestRaw || {}),
      errors
    );
    state.hasRendered = true;
    byId("loading-panel").hidden = true;
    byId("dashboard").hidden = false;
  } catch (error) {
    const message = `仪表盘处理失败：${error?.message || String(error)}`;
    byId("loading-panel").hidden = true;
    headerQuality.dataset.tone = "error";
    headerQuality.title = message;
    if (state.hasRendered) {
      byId("dashboard").hidden = false;
      text(headerQuality, "刷新失败，保留上次数据");
      showLoadStatus("error", "页面更新失败，已保留上次数据", message);
    } else {
      byId("error-panel").hidden = false;
      text(byId("error-message"), message);
      text(headerQuality, "数据不可用");
    }
  } finally {
    state.loading = false;
    reloadButton.disabled = false;
    retryButton.disabled = false;
    text(reloadButton, "重新读取");
    byId("main-content").removeAttribute("aria-busy");
  }
}

function renderDashboard(snapshot, backtest, loadErrors) {
  const generatedAt = snapshot.generatedAt || backtest.generatedAt;
  renderHeaderQuality(snapshot.quality, loadErrors, generatedAt);
  renderLoadStatus(loadErrors);
  renderSiteStaleAlert(generatedAt);
  renderOverview(snapshot);
  renderRiskHistory(snapshot.history);
  renderBacktest(backtest);
  renderIndicators(snapshot);
  renderMuSoxsStrategy(snapshot.muSoxsStrategy);
  renderWatchlist(snapshot.watchlist);
  renderFreshness(snapshot, backtest, loadErrors);
  installChartResizeObserver();
}

function hideLoadStatus() {
  const alert = byId("load-status-alert");
  alert.hidden = true;
  alert.removeAttribute("data-tone");
}

function showLoadStatus(tone, title, message) {
  const alert = byId("load-status-alert");
  alert.dataset.tone = tone;
  alert.setAttribute("role", tone === "error" ? "alert" : "status");
  alert.setAttribute("aria-live", tone === "error" ? "assertive" : "polite");
  text(byId("load-status-title"), title);
  text(byId("load-status-message"), message);
  alert.hidden = false;
}

function renderLoadStatus(loadErrors) {
  if (!loadErrors.length) {
    hideLoadStatus();
    return;
  }
  showLoadStatus(
    "warning",
    "部分数据刷新失败",
    `${loadErrors.join("；")}。页面正在显示其余可用数据或上次成功读取的结果。`
  );
}

function renderHeaderQuality(quality, loadErrors, generatedAt) {
  const badge = byId("header-quality");
  const staleCount = quality.staleSources.length;
  const normalizedStatus = quality.status.toLowerCase();
  const siteAgeHours = ageInHours(generatedAt);
  if (["error", "failed", "unavailable"].includes(normalizedStatus)) {
    badge.dataset.tone = "error";
    text(badge, "数据质量异常");
    return;
  }
  if (
    loadErrors.length ||
    staleCount ||
    (siteAgeHours !== null && siteAgeHours > SITE_STALE_AFTER_HOURS) ||
    ["unknown", "stale", "warning", "degraded"].includes(normalizedStatus)
  ) {
    badge.dataset.tone = "warning";
    if (siteAgeHours !== null && siteAgeHours > SITE_STALE_AFTER_HOURS) {
      text(badge, "站点数据已过期");
    } else {
      text(badge, staleCount ? `${staleCount} 个来源陈旧` : "部分数据不可用");
    }
    return;
  }
  badge.dataset.tone = "ok";
  text(badge, "数据读取完成");
}

function renderSiteStaleAlert(generatedAt) {
  const alert = byId("site-stale-alert");
  const message = byId("site-stale-message");
  const siteAgeHours = ageInHours(generatedAt);
  const isStale = siteAgeHours !== null && siteAgeHours > SITE_STALE_AFTER_HOURS;
  alert.hidden = !isStale;
  if (!isStale) return;
  const wholeDays = Math.max(3, Math.floor(siteAgeHours / 24));
  text(
    message,
    `最近一次发布距今约 ${wholeDays} 天（${formatDateTime(generatedAt)}）。请不要把页面上的信号当作当前行情。`
  );
}

function renderOverview(snapshot) {
  text(byId("market-date"), `市场日期 ${formatDate(snapshot.asOf)}`);
  const score = snapshot.riskScore;
  text(byId("risk-score"), score === null ? "--" : formatNumber(score, 1));

  const gauge = byId("risk-gauge");
  const safeScore = score === null ? 0 : clamp(score, 0, 100);
  gauge.style.setProperty("--risk", String(safeScore));
  gauge.style.setProperty("--risk-color", `var(--${riskTone(score)})`);
  if (score === null) {
    gauge.removeAttribute("aria-valuenow");
    gauge.setAttribute("aria-valuetext", "风险分不可用");
  } else {
    gauge.setAttribute("aria-valuenow", String(score));
    gauge.setAttribute("aria-valuetext", `${formatNumber(score, 1)} 分，${LABELS.regimes[snapshot.regime] || snapshot.regime}`);
  }

  const regime = byId("regime-badge");
  regime.dataset.regime = snapshot.regime;
  text(regime, LABELS.regimes[snapshot.regime] || snapshot.regime);
  text(
    byId("target-position"),
    snapshot.targetPosition === null ? "候选仓位 --" : `候选仓位 ${formatPercent(snapshot.targetPosition, 0)}`
  );
  text(byId("action-label"), LABELS.actions[snapshot.action] || humanizeKey(snapshot.action));

  const reasonList = byId("reason-list");
  reasonList.replaceChildren();
  const reasons = snapshot.reasons.filter(Boolean).slice(0, 5);
  if (!reasons.length) reasons.push("insufficient_history");
  reasons.forEach((reason) => {
    reasonList.append(createElement("li", "", LABELS.reasons[reason] || humanizeKey(reason)));
  });

  renderComponents(snapshot.components);
  renderLatest(snapshot.latest);
}

function renderComponents(components) {
  const root = byId("component-list");
  root.replaceChildren();
  Object.entries(LABELS.components).forEach(([key, label]) => {
    const value = finiteNumber(components?.[key]);
    const row = createElement("div", "component-row");
    row.append(createElement("strong", "", label));
    row.append(createElement("span", "", value === null ? "--" : `${formatNumber(value, 1)} / 100`));
    const track = createElement("div", "component-track");
    track.setAttribute("role", "img");
    track.setAttribute(
      "aria-label",
      value === null ? `${label}不可用` : `${label}${formatNumber(value, 1)}分`
    );
    const fill = createElement("span", "component-fill");
    fill.style.width = `${value === null ? 0 : clamp(value, 0, 100)}%`;
    fill.style.setProperty("--risk-color", `var(--${riskTone(value)})`);
    track.append(fill);
    row.append(track);
    root.append(row);
  });
}

function renderLatest(latest) {
  const root = byId("latest-values");
  root.replaceChildren();
  const preferredKeys = ["QQQ", "VXN", "REAL10Y", "NFCI"];
  const entries = preferredKeys
    .map((key) => [key, firstDefined(latest?.[key], latest?.[key === "VXN" ? "VXNCLS" : key])])
    .filter(([, value]) => value !== undefined);
  const fallbackEntries = Object.entries(latest || {}).slice(0, 4);
  (entries.length ? entries : fallbackEntries).forEach(([key, raw]) => {
    const item = createElement("div", "latest-item");
    item.append(createElement("dt", "", LABELS.latest[key] || humanizeKey(key)));
    const value = raw && typeof raw === "object" ? firstDefined(raw.value, raw.latest, raw.close) : raw;
    const unit = raw && typeof raw === "object" ? raw.unit : null;
    item.append(createElement("dd", "", `${formatFlexibleNumber(value)}${unit || ""}`));
    root.append(item);
  });
  if (!root.children.length) {
    ["QQQ", "VXN", "10年实际利率", "NFCI"].forEach((label) => {
      const item = createElement("div", "latest-item");
      item.append(createElement("dt", "", label));
      item.append(createElement("dd", "", "--"));
      root.append(item);
    });
  }
}

function normalizeRiskHistory(history) {
  return history
    .map((point) => ({
      date: firstDefined(point.date, point.as_of, point.timestamp, point.key),
      risk: finiteNumber(firstDefined(point.risk_score, point.risk, point.score)),
      position: finiteNumber(firstDefined(point.target_position, point.position, point.exposure)),
      qqq: finiteNumber(firstDefined(point.qqq, point.QQQ)),
    }))
    .filter((point) => point.date && point.risk !== null)
    .sort((left, right) => String(left.date).localeCompare(String(right.date)));
}

function renderRiskHistory(history) {
  state.riskHistory = normalizeRiskHistory(history);
  const points = state.riskHistory;
  text(
    byId("risk-history-range"),
    points.length ? `${formatDate(points[0].date)} 至 ${formatDate(points.at(-1).date)}` : "--"
  );

  const table = byId("risk-history-table");
  table.replaceChildren();
  sampleForTable(points, 120).forEach((point) => {
    const row = document.createElement("tr");
    row.append(createElement("td", "", formatDate(point.date)));
    row.append(createElement("td", "", formatNumber(point.risk, 1)));
    row.append(createElement("td", "", formatPercent(point.position, 0)));
    table.append(row);
  });
  if (!points.length) appendEmptyRow(table, 3, "暂无风险历史数据");

  const summary = byId("risk-chart-summary");
  if (points.length) {
    const values = points.map((point) => point.risk);
    text(
      summary,
      `当前 ${formatNumber(values.at(-1), 1)} 分，区间最低 ${formatNumber(Math.min(...values), 1)} 分，最高 ${formatNumber(Math.max(...values), 1)} 分。`
    );
  } else {
    text(summary, "生成市场快照后，这里将显示风险分随时间的变化。", "");
  }
  drawRiskChart();
}

function renderBacktest(backtest) {
  renderMetricGroup("full", backtest.full);
  renderMetricGroup("holdout", backtest.holdout);
  renderAssumptions(backtest);
  renderSensitivity(backtest.experiments);

  state.equityHistory = backtest.equity.length ? backtest.equity : backtest.holdoutEquity;
  const equity = state.equityHistory;
  text(
    byId("backtest-range"),
    equity.length ? `${formatDate(equity[0].date)} 至 ${formatDate(equity.at(-1).date)}` : "--"
  );

  const table = byId("equity-history-table");
  table.replaceChildren();
  sampleForTable(equity, 120).forEach((point) => {
    const row = document.createElement("tr");
    row.append(createElement("td", "", formatDate(point.date)));
    row.append(createElement("td", "", formatNumber(point.strategy, 3)));
    row.append(createElement("td", "", formatNumber(point.benchmark, 3)));
    table.append(row);
  });
  if (!equity.length) appendEmptyRow(table, 3, "暂无净值曲线数据");

  const strategyEnd = equity.at(-1)?.strategy;
  const benchmarkEnd = equity.at(-1)?.benchmark;
  text(
    byId("equity-chart-summary"),
    equity.length
      ? `期末净值：策略 ${formatNumber(strategyEnd, 3)}，QQQ ${formatNumber(benchmarkEnd, 3)}。`
      : "运行回测后，这里将显示策略与 QQQ 的同期净值。",
    ""
  );
  renderBacktestCallout(backtest);
  drawEquityChart();
}

function experimentLabel(name) {
  const labels = {
    conservative: "conservative",
    conservative_ma10: "conservative + MA10",
    balanced: "balanced",
    balanced_ma10: "balanced + MA10",
    growth: "growth",
    growth_ma10: "growth + MA10",
    qqq_200dma: "QQQ 200DMA 基线",
  };
  return labels[name] || humanizeKey(name);
}

function renderSensitivity(experiments) {
  const root = byId("sensitivity-table");
  root.replaceChildren();
  experiments.forEach((experiment) => {
    const row = document.createElement("tr");
    row.dataset.default = String(experiment.name === "conservative");
    const nameCell = document.createElement("td");
    nameCell.append(createElement("span", "policy-name", experimentLabel(experiment.name)));
    if (experiment.name === "conservative") {
      nameCell.append(createElement("span", "default-policy-label", "默认"));
    }
    row.append(nameCell);
    row.append(createElement("td", "", formatPercent(experiment.full.strategy.cagr)));
    row.append(createElement("td", "", formatPercent(experiment.full.strategy.max_drawdown)));
    row.append(createElement("td", "", formatPercent(experiment.holdout.strategy.cagr)));
    row.append(createElement("td", "", formatPercent(experiment.holdout.strategy.max_drawdown)));
    row.append(createElement("td", "", formatNumber(experiment.holdout.strategy.turnover, 2)));
    root.append(row);
  });
  if (!experiments.length) appendEmptyRow(root, 6, "回测结果未提供敏感性实验");
}

function renderAssumptions(backtest) {
  const root = byId("backtest-assumptions");
  root.replaceChildren();
  const cost = firstDefined(
    backtest.assumptions.transaction_cost_bps,
    backtest.full.transactionCostBps
  );
  const annualization = firstDefined(
    backtest.assumptions.annualization,
    backtest.full.annualization,
    252
  );
  const items = [
    ["成本", finiteNumber(cost) === null ? "--" : `${formatNumber(cost, 0)} bp`],
    ["年化", finiteNumber(annualization) === null ? "--" : `${formatNumber(annualization, 0)} 日`],
    ["执行", "次日收盘"],
  ];
  items.forEach(([label, value]) => root.append(createElement("span", "", `${label} ${value}`)));
}

function renderMetricGroup(prefix, period) {
  const periodLabel = period.start || period.end ? `${formatDate(period.start)} 至 ${formatDate(period.end)}` : "--";
  text(byId(`${prefix}-period`), periodLabel);
  const root = byId(`${prefix}-metrics`);
  root.replaceChildren();
  const header = ["指标", "策略", "QQQ"];
  header.forEach((label) => root.append(createElement("div", "metric-cell metric-header", label)));
  const definitions = [
    ["累计收益", "total_return", (value) => formatPercent(value)],
    ["年化收益", "cagr", (value) => formatPercent(value)],
    ["年化波动", "volatility", (value) => formatPercent(value)],
    ["Sharpe", "sharpe", (value) => formatNumber(value, 2)],
    ["最大回撤", "max_drawdown", (value) => formatPercent(value)],
    ["Calmar", "calmar", (value) => formatNumber(value, 2)],
    ["单边换手", "turnover", (value) => formatNumber(value, 2)],
  ];
  definitions.forEach(([label, key, formatter]) => {
    root.append(createElement("div", "metric-cell", label));
    root.append(createElement("div", "metric-cell metric-value", formatter(period.strategy[key])));
    root.append(createElement("div", "metric-cell metric-value", formatter(period.benchmark[key])));
  });
}

function renderBacktestCallout(backtest) {
  const root = byId("backtest-callout");
  const holdout = backtest.holdout;
  const strategyCagr = finiteNumber(holdout.strategy.cagr);
  const benchmarkCagr = finiteNumber(holdout.benchmark.cagr);
  const strategyDrawdown = finiteNumber(holdout.strategy.max_drawdown);
  const benchmarkDrawdown = finiteNumber(holdout.benchmark.max_drawdown);
  if ([strategyCagr, benchmarkCagr, strategyDrawdown, benchmarkDrawdown].some((value) => value === null)) {
    text(root, "尾部评估指标尚不完整。请先运行回测，再根据近期结果评价模型。", "");
    return;
  }
  const drawdownImprovement = strategyDrawdown - benchmarkDrawdown;
  const direction = drawdownImprovement > 0 ? "较浅" : drawdownImprovement < 0 ? "更深" : "相同";
  const gates = backtest.validation.holdout_gates || {};
  const retention = finiteNumber(gates.cagr_retention);
  const gateStatus = gates.all_passed === true ? "PASS" : gates.all_passed === false ? "FAIL" : "未判定";
  root.dataset.tone = gates.all_passed === false ? "warning" : "default";
  text(
    root,
    `尾部评估门槛 ${gateStatus}：策略 CAGR 为 ${formatPercent(strategyCagr)}，同期 QQQ 为 ${formatPercent(benchmarkCagr)}${retention === null ? "" : `，保留率 ${formatPercent(retention)}`}；策略最大回撤为 ${formatPercent(strategyDrawdown)}，比 QQQ ${direction} ${formatPercent(Math.abs(drawdownImprovement))}；Sharpe 为 ${formatNumber(holdout.strategy.sharpe, 2)}，QQQ 为 ${formatNumber(holdout.benchmark.sharpe, 2)}。这是样本描述，不是未来收益保证。`,
    ""
  );
}

function normalizeIndicators(snapshot) {
  if (snapshot.indicators) {
    return asArray(snapshot.indicators).map((item) => ({
      key: firstDefined(item.key, item.name, item.symbol),
      name: firstDefined(item.label, item.display_name, item.name, item.key, item.symbol),
      value: finiteNumber(firstDefined(item.value, item.latest, item.close)),
      unit: firstDefined(item.unit, ""),
      risk: finiteNumber(firstDefined(item.risk_score, item.score, item.percentile)),
      riskLabel: firstDefined(item.risk_label, item.component),
      date: firstDefined(item.observed_at, item.as_of, item.date, snapshot.asOf),
      source: firstDefined(item.source, item.provider),
      stale: Boolean(firstDefined(item.stale, false)),
    }));
  }

  const sourceMetadata = snapshot.quality.sourceMetadata || {};
  const componentByKey = {
    QQQ: "trend_risk",
    SPY: "trend_risk",
    XLK: "trend_risk",
    SMH: "trend_risk",
    VXN: "volatility_risk",
    VXNCLS: "volatility_risk",
    REAL10Y: "macro_risk",
    DFII10: "macro_risk",
    NFCI: "macro_risk",
  };
  const sourceAliases = {
    VXN: "VXNCLS",
    REAL10Y: "DFII10",
    CASH3M: "DGS3MO",
  };
  const rows = [];
  Object.entries(snapshot.latest || {}).forEach(([key, raw]) => {
    const detail = raw && typeof raw === "object" ? raw : {};
    const sourceKey = sourceAliases[key] || key;
    const matchingMetadataKey = Object.keys(sourceMetadata).find(
      (candidate) => candidate === key || candidate.endsWith(`:${sourceKey}`)
    );
    const metadata = sourceMetadata[matchingMetadataKey] || {};
    const componentKey = componentByKey[key];
    rows.push({
      key,
      name: LABELS.latest[key] || humanizeKey(key),
      value: finiteNumber(raw && typeof raw === "object" ? firstDefined(raw.value, raw.latest, raw.close) : raw),
      unit: firstDefined(detail.unit, ""),
      risk: finiteNumber(
        firstDefined(
          detail.risk_score,
          detail.score,
          detail.percentile,
          componentKey ? snapshot.components?.[componentKey] : null
        )
      ),
      riskLabel: componentKey ? LABELS.components[componentKey] : null,
      date: firstDefined(
        detail.observed_at,
        detail.as_of,
        detail.date,
        metadata.last_observation_date,
        snapshot.asOf
      ),
      source: firstDefined(detail.source, detail.provider, metadata.source, metadata.provider),
      stale: Boolean(detail.stale || metadata.stale || metadata.observation_stale),
      staleLabel: metadata.observation_stale ? "观测陈旧" : "缓存陈旧",
    });
  });
  return rows;
}

function renderIndicators(snapshot) {
  const root = byId("indicator-table");
  root.replaceChildren();
  const indicators = normalizeIndicators(snapshot);
  indicators.forEach((indicator) => {
    const row = document.createElement("tr");
    row.append(createElement("td", "", indicator.name || humanizeKey(indicator.key)));
    row.append(createElement("td", "", `${formatFlexibleNumber(indicator.value)}${indicator.unit || ""}`));
    row.append(
      createElement(
        "td",
        "",
        indicator.risk === null
          ? "--"
          : `${indicator.riskLabel ? `${humanizeKey(indicator.riskLabel)} ` : ""}${formatNumber(indicator.risk, 1)}`
      )
    );
    row.append(createElement("td", "", formatDate(indicator.date)));
    const sourceCell = document.createElement("td");
    const sourceState = createElement(
      "span",
      "source-state",
      indicator.stale ? indicator.staleLabel || "数据陈旧" : sourceLabel(indicator.source)
    );
    sourceState.dataset.tone = indicator.stale ? "warning" : "ok";
    sourceCell.append(sourceState);
    row.append(sourceCell);
    root.append(row);
  });
  if (!indicators.length) appendEmptyRow(root, 5, "快照未提供核心指标明细");
}

function strategyDecisionTone(decision) {
  if (["buy_mu_starter", "add_mu_after_reversal"].includes(decision)) return "positive";
  if (decision === "reduce_mu_buy_soxs") return "negative";
  if (["exit_mu_if_held", "exit_soxs_if_held", "wait_for_confirmation"].includes(decision)) {
    return "warning";
  }
  return "unknown";
}

function confidenceLevel(score, threshold) {
  const value = finiteNumber(score);
  if (value === null) return { label: "数据不足", tone: "unknown" };
  if (value >= threshold) return { label: "已达执行阈值", tone: threshold >= 70 ? "negative" : "positive" };
  if (value >= 50) return { label: "中等，等待确认", tone: "warning" };
  return { label: "较低", tone: "unknown" };
}

function renderConfidence(prefix, score, threshold, meterLabel) {
  const value = finiteNumber(score);
  const safeValue = value === null ? 0 : clamp(value, 0, 100);
  const level = confidenceLevel(value, threshold);
  text(byId(`${prefix}-confidence-value`), value === null ? "--" : formatNumber(value, 1));
  const levelElement = byId(`${prefix}-confidence-level`);
  levelElement.dataset.tone = level.tone;
  text(levelElement, level.label);
  const meter = byId(`${prefix}-confidence-meter`);
  meter.style.setProperty("--confidence", String(safeValue));
  if (value === null) {
    meter.removeAttribute("aria-valuenow");
    meter.setAttribute("aria-valuetext", `${meterLabel}数据不足`);
  } else {
    meter.setAttribute("aria-valuenow", String(value));
    meter.setAttribute("aria-valuetext", `${meterLabel}${formatNumber(value, 1)}分，${level.label}`);
  }
}

function renderMuSoxsStrategy(strategy) {
  const card = byId("mu-soxs-card");
  if (!strategy || typeof strategy !== "object") {
    card.hidden = true;
    return;
  }
  card.hidden = false;
  const decision = String(firstDefined(strategy.decision, "data_unavailable"));
  const decisionBadge = byId("mu-soxs-decision");
  decisionBadge.dataset.tone = strategyDecisionTone(decision);
  text(decisionBadge, LABELS.strategyDecisions[decision] || humanizeKey(decision));

  const instruments = strategy.instruments || {};
  const version = firstDefined(strategy.version, "mu-soxs-confidence");
  text(
    byId("mu-soxs-meta"),
    `${formatDate(strategy.as_of)} · ${version} · ${firstDefined(instruments.long, "MU")} / ${firstDefined(instruments.inverse, "SOXS")}`
  );
  text(byId("mu-soxs-mu-close"), formatNumber(strategy.mu_close, 2));
  text(byId("mu-soxs-rsi"), formatNumber(strategy.mu_rsi6, 1));
  text(byId("mu-soxs-market-risk"), formatNumber(strategy.market_risk, 1));

  const thresholds = strategy.thresholds || {};
  renderConfidence("buy", strategy.buy_confidence, finiteNumber(thresholds.buy) ?? 65, "买入 MU 置信度");
  renderConfidence(
    "sell",
    strategy.sell_confidence,
    finiteNumber(thresholds.sell) ?? 70,
    "卖出 MU 并买入 SOXS 置信度"
  );

  const reasonList = byId("mu-soxs-reasons");
  reasonList.replaceChildren();
  const reasons = asArray(strategy.reasons).map((item) =>
    typeof item === "string" ? item : firstDefined(item.reason, item.key, item.value)
  );
  (reasons.length ? reasons : ["waiting_for_extreme_and_confirmation"]).slice(0, 4).forEach((reason) => {
    reasonList.append(
      createElement("li", "", LABELS.strategyReasons[reason] || humanizeKey(reason))
    );
  });

  const backtest = strategy.recent_backtest || {};
  const metrics = backtest.metrics || {};
  const window = backtest.window || {};
  const completed = finiteNumber(metrics.completed_trades);
  const wins = finiteNumber(metrics.winning_trades);
  const summary =
    finiteNumber(metrics.total_return) === null
      ? "近期回测 --"
      : `${formatDate(window.start)}–${formatDate(window.end)}：${formatPercent(metrics.total_return, 2)}；日收盘最大回撤 ${formatPercent(metrics.max_drawdown, 2)}；${wins ?? 0}/${completed ?? 0} 笔盈利（含开发样本）`;
  text(byId("mu-soxs-backtest"), summary);
}

function renderWatchlist(watchlist) {
  const root = byId("watchlist-table");
  root.replaceChildren();
  text(byId("watchlist-count"), `${watchlist.length} 只股票`);
  watchlist.forEach((stock) => {
    const symbol = String(firstDefined(stock.symbol, stock.ticker, stock.key, "--"));
    const status = String(firstDefined(stock.status, stock.state, "insufficient_data"));
    const row = document.createElement("tr");
    row.append(createElement("td", "stock-symbol", symbol));
    const statusCell = document.createElement("td");
    const badge = createElement("span", "stock-status", LABELS.stockStatuses[status] || humanizeKey(status));
    badge.dataset.tone = stockTone(status);
    statusCell.append(badge);
    row.append(statusCell);
    row.append(createElement("td", "", formatNumber(firstDefined(stock.close, stock.price), 2)));
    row.append(
      createElement(
        "td",
        "",
        formatPercent(firstDefined(stock.relative_strength_60d, stock.rs60, stock.relative_strength))
      )
    );
    row.append(
      createElement(
        "td",
        "",
        formatPercent(firstDefined(stock.drawdown_from_52w_high, stock.drawdown, stock.distance_from_high))
      )
    );
    const reasonCell = document.createElement("td");
    const reasonTags = createElement("div", "reason-tags");
    const reasons = asArray(stock.reasons).map((item) =>
      typeof item === "string" ? item : firstDefined(item.reason, item.key, item.value)
    );
    (reasons.length ? reasons : ["无原因信息"]).slice(0, 4).forEach((reason) => {
      reasonTags.append(
        createElement("span", "reason-tag", LABELS.reasons[reason] || humanizeKey(reason))
      );
    });
    reasonCell.append(reasonTags);
    row.append(reasonCell);
    root.append(row);
  });
  if (!watchlist.length) appendEmptyRow(root, 6, "尚未生成自选股状态");
}

function normalizeSourceMetadata(metadata) {
  return asArray(metadata).map((item) => ({
    name: String(firstDefined(item.name, item.symbol, item.series, item.key, "未知来源")),
    provider: firstDefined(item.provider, item.source, "未标记"),
    source: firstDefined(item.source, "unknown"),
    fetchedAt: firstDefined(item.fetched_at, item.updated_at),
    observedAt: firstDefined(item.last_observation_date, item.observed_at),
    ageSessions: finiteNumber(item.observation_age_sessions),
    cacheStale: Boolean(item.stale),
    observationStale: Boolean(item.observation_stale),
    stale: Boolean(item.stale || item.observation_stale),
    error: firstDefined(item.error_code, item.error, null),
  }));
}

function sourceErrorLabel(value) {
  const labels = {
    upstream_refresh_failed_using_cache: "上游刷新失败，已使用缓存",
  };
  return labels[value] || value;
}

function renderFreshness(snapshot, backtest, loadErrors) {
  text(byId("generated-at"), formatDateTime(snapshot.generatedAt || backtest.generatedAt));
  const root = byId("freshness-list");
  root.replaceChildren();
  const sources = normalizeSourceMetadata(snapshot.quality.sourceMetadata);
  sources.forEach((source) => {
    const item = createElement("article", "freshness-item");
    const heading = createElement("div", "freshness-item-heading");
    heading.append(createElement("strong", "", source.name));
    const staleLabel = source.cacheStale ? "陈旧缓存" : "观测陈旧";
    const badge = createElement("span", "source-state", source.stale ? staleLabel : "正常");
    badge.dataset.tone = source.stale ? "warning" : "ok";
    heading.append(badge);
    item.append(heading);
    item.append(
      createElement(
        "p",
        "freshness-meta",
        `${source.provider} | 观测至 ${formatDate(source.observedAt)} | 抓取 ${formatDateTime(source.fetchedAt)}` +
          `${source.ageSessions === null ? "" : ` | ${formatNumber(source.ageSessions, 0)} 个工作日`}` +
          `${source.error ? ` | ${sourceErrorLabel(source.error)}` : ""}`
      )
    );
    root.append(item);
  });
  loadErrors.forEach((error) => {
    const item = createElement("article", "freshness-item");
    const heading = createElement("div", "freshness-item-heading");
    heading.append(createElement("strong", "", "文件读取异常"));
    const badge = createElement("span", "source-state", "部分失败");
    badge.dataset.tone = "error";
    heading.append(badge);
    item.append(heading);
    item.append(createElement("p", "freshness-meta", error));
    root.append(item);
  });
  if (!sources.length && !loadErrors.length) {
    const item = createElement("article", "freshness-item");
    const heading = createElement("div", "freshness-item-heading");
    heading.append(createElement("strong", "", "来源元数据"));
    const badge = createElement("span", "source-state", "未提供");
    badge.dataset.tone = "warning";
    heading.append(badge);
    item.append(heading);
    item.append(createElement("p", "freshness-meta", "快照可读取，但没有逐来源的新鲜度记录。"));
    root.append(item);
  }
}

function appendEmptyRow(root, columns, message) {
  const row = document.createElement("tr");
  row.className = "table-empty";
  const cell = createElement("td", "", message);
  cell.colSpan = columns;
  row.append(cell);
  root.append(row);
}

function humanizeKey(value) {
  if (value === null || value === undefined) return "--";
  return String(value).replaceAll("_", " ");
}

function sampleForTable(points, maximum) {
  if (points.length <= maximum) return points;
  const step = Math.ceil(points.length / maximum);
  const sampled = points.filter((_, index) => index % step === 0);
  if (sampled.at(-1) !== points.at(-1)) sampled.push(points.at(-1));
  return sampled;
}

function sampleForChart(points, maximum = 360) {
  if (points.length <= maximum) return points;
  const step = Math.ceil(points.length / maximum);
  const sampled = points.filter((_, index) => index % step === 0);
  if (sampled.at(-1) !== points.at(-1)) sampled.push(points.at(-1));
  return sampled;
}

function chartColors() {
  const styles = getComputedStyle(document.documentElement);
  const get = (name) => styles.getPropertyValue(name).trim();
  return {
    surface: get("--surface"),
    border: get("--border"),
    text: get("--text-secondary"),
    muted: get("--text-muted"),
    accent: get("--accent"),
    positive: get("--positive"),
    warning: get("--warning"),
    negative: get("--negative"),
  };
}

function setupCanvas(canvas) {
  const width = Math.max(canvas.clientWidth, 280);
  const height = Math.max(canvas.clientHeight, 220);
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);
  return { context, width, height };
}

function drawRiskChart() {
  const canvas = byId("risk-chart");
  const empty = byId("risk-chart-empty");
  const points = sampleForChart(state.riskHistory);
  if (points.length < 2) {
    canvas.hidden = true;
    empty.hidden = false;
    return;
  }
  canvas.hidden = false;
  empty.hidden = true;
  const { context, width, height } = setupCanvas(canvas);
  const colors = chartColors();
  const plot = { left: 42, top: 12, right: width - 12, bottom: height - 30 };
  const plotHeight = plot.bottom - plot.top;
  const plotWidth = plot.right - plot.left;
  const y = (value) => plot.bottom - (clamp(value, 0, 100) / 100) * plotHeight;
  const x = (index) => plot.left + (index / (points.length - 1)) * plotWidth;

  context.fillStyle = "rgba(85, 214, 158, 0.055)";
  context.fillRect(plot.left, y(40), plotWidth, y(0) - y(40));
  context.fillStyle = "rgba(244, 196, 94, 0.055)";
  context.fillRect(plot.left, y(65), plotWidth, y(40) - y(65));
  context.fillStyle = "rgba(255, 125, 134, 0.055)";
  context.fillRect(plot.left, y(100), plotWidth, y(65) - y(100));

  context.font = '11px "Cascadia Code", Consolas, monospace';
  context.textBaseline = "middle";
  [0, 40, 65, 100].forEach((tick) => {
    context.beginPath();
    context.strokeStyle = colors.border;
    context.lineWidth = 1;
    context.moveTo(plot.left, y(tick));
    context.lineTo(plot.right, y(tick));
    context.stroke();
    context.fillStyle = colors.muted;
    context.textAlign = "right";
    context.fillText(String(tick), plot.left - 8, y(tick));
  });

  context.beginPath();
  context.strokeStyle = colors.accent;
  context.lineWidth = 2;
  context.lineJoin = "round";
  points.forEach((point, index) => {
    const px = x(index);
    const py = y(point.risk);
    if (index === 0) context.moveTo(px, py);
    else context.lineTo(px, py);
  });
  context.stroke();

  const last = points.at(-1);
  context.fillStyle = colors.accent;
  context.beginPath();
  context.arc(x(points.length - 1), y(last.risk), 4, 0, Math.PI * 2);
  context.fill();
  drawDateLabels(context, points, plot, colors);
}

function drawEquityChart() {
  const canvas = byId("equity-chart");
  const empty = byId("equity-chart-empty");
  const points = sampleForChart(state.equityHistory).filter(
    (point) => point.strategy !== null && point.benchmark !== null
  );
  if (points.length < 2) {
    canvas.hidden = true;
    empty.hidden = false;
    return;
  }
  canvas.hidden = false;
  empty.hidden = true;
  const { context, width, height } = setupCanvas(canvas);
  const colors = chartColors();
  const plot = { left: 48, top: 12, right: width - 12, bottom: height - 30 };
  const plotWidth = plot.right - plot.left;
  const plotHeight = plot.bottom - plot.top;
  const allValues = points.flatMap((point) => [point.strategy, point.benchmark]);
  const rawMin = Math.min(...allValues);
  const rawMax = Math.max(...allValues);
  const padding = Math.max((rawMax - rawMin) * 0.08, 0.02);
  const minimum = Math.max(0, rawMin - padding);
  const maximum = rawMax + padding;
  const range = maximum - minimum || 1;
  const x = (index) => plot.left + (index / (points.length - 1)) * plotWidth;
  const y = (value) => plot.bottom - ((value - minimum) / range) * plotHeight;

  context.font = '11px "Cascadia Code", Consolas, monospace';
  context.textBaseline = "middle";
  for (let index = 0; index <= 4; index += 1) {
    const value = minimum + (range * index) / 4;
    const py = y(value);
    context.beginPath();
    context.strokeStyle = colors.border;
    context.lineWidth = 1;
    context.moveTo(plot.left, py);
    context.lineTo(plot.right, py);
    context.stroke();
    context.fillStyle = colors.muted;
    context.textAlign = "right";
    context.fillText(formatNumber(value, 2), plot.left - 8, py);
  }

  drawSeries(context, points, x, y, "strategy", colors.accent, []);
  drawSeries(context, points, x, y, "benchmark", colors.text, [6, 5]);
  drawDateLabels(context, points, plot, colors);
}

function drawSeries(context, points, x, y, key, color, dash) {
  context.beginPath();
  context.strokeStyle = color;
  context.lineWidth = 2;
  context.lineJoin = "round";
  context.setLineDash(dash);
  points.forEach((point, index) => {
    const px = x(index);
    const py = y(point[key]);
    if (index === 0) context.moveTo(px, py);
    else context.lineTo(px, py);
  });
  context.stroke();
  context.setLineDash([]);
}

function drawDateLabels(context, points, plot, colors) {
  const indices = [0, Math.floor((points.length - 1) / 2), points.length - 1];
  const alignments = ["left", "center", "right"];
  context.fillStyle = colors.muted;
  context.textBaseline = "top";
  indices.forEach((index, position) => {
    const ratio = index / (points.length - 1);
    const px = plot.left + ratio * (plot.right - plot.left);
    context.textAlign = alignments[position];
    context.fillText(formatDate(points[index].date), px, plot.bottom + 10);
  });
}

function installChartResizeObserver() {
  if (state.resizeObserver) state.resizeObserver.disconnect();
  let frame = null;
  const redraw = () => {
    if (frame) cancelAnimationFrame(frame);
    frame = requestAnimationFrame(() => {
      drawRiskChart();
      drawEquityChart();
    });
  };
  if ("ResizeObserver" in window) {
    state.resizeObserver = new ResizeObserver(redraw);
    state.resizeObserver.observe(byId("risk-chart").parentElement);
    state.resizeObserver.observe(byId("equity-chart").parentElement);
  } else {
    window.addEventListener("resize", redraw, { passive: true });
  }
}

byId("reload-button").addEventListener("click", loadDashboard);
byId("error-retry-button").addEventListener("click", loadDashboard);
loadDashboard();
