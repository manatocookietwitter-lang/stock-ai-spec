import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AppShell } from "./components";
import { DecisionReviewPage } from "./pages/DecisionReviewPage";
import { ExecutionPage } from "./pages/ExecutionPage";
import { SettingsPage } from "./pages/SettingsPage";
import { StockDetailPage } from "./pages/StockDetailPage";
import { TodayPage } from "./pages/TodayPage";
import type { TodayResponse } from "./types";

const proposal = {
  proposalId: "proposal-1",
  asOf: "2026-08-24T11:30:00+09:00",
  generatedAt: "2026-08-24T11:36:00+09:00",
  modelBundleVersion: "fixture-model",
  decisionEngineVersion: "engine-v1",
  status: "READY",
  isResearchOnly: false,
  isOrderInstruction: false as const,
  estimatedSellValue: "100000",
  estimatedBuyValue: "100000",
  estimatedTransactionCost: "200",
  estimatedTaxEffect: "0",
  estimatedCashAfter: { bucket: "100000" },
  changeCount: 2,
  holdExpectedValue: "1000",
  proposedExpectedValue: "1200",
  netImprovement: "200",
  noTradeReason: null,
  lines: [
    {
      lineId: "sell-line",
      symbol: "A",
      companyName: "売却会社",
      accountBucketId: "bucket",
      currentShares: 100,
      recommendedShares: 0,
      shareDifference: -100,
      action: "SELL" as const,
      referencePrice: "1000",
      currentMarketValue: "100000",
      recommendedMarketValue: "0",
      estimatedRequiredOrReleasedCash: "-100000",
      holdExpectedValue: "0",
      proposedExpectedValue: "0",
      estimatedTransactionCost: "100",
      estimatedTaxEffect: "0",
      netExpectedImprovement: "100",
      downsideLevel: 0.2,
      uncertaintyLevel: 0.1,
      positiveReasons: ["全体最適化で売却"],
      negativeReasons: [],
    },
    {
      lineId: "buy-line",
      symbol: "B",
      companyName: "購入会社",
      accountBucketId: "bucket",
      currentShares: 0,
      recommendedShares: 100,
      shareDifference: 100,
      action: "BUY" as const,
      referencePrice: "1000",
      currentMarketValue: "0",
      recommendedMarketValue: "100000",
      estimatedRequiredOrReleasedCash: "100000",
      holdExpectedValue: "0",
      proposedExpectedValue: "0",
      estimatedTransactionCost: "100",
      estimatedTaxEffect: "0",
      netExpectedImprovement: "100",
      downsideLevel: 0.1,
      uncertaintyLevel: 0.1,
      positiveReasons: ["予測順位が高い"],
      negativeReasons: [],
    },
    {
      lineId: "hold-line",
      symbol: "C",
      companyName: "維持会社",
      accountBucketId: "bucket",
      currentShares: 100,
      recommendedShares: 100,
      shareDifference: 0,
      action: "HOLD" as const,
      referencePrice: "1000",
      currentMarketValue: "100000",
      recommendedMarketValue: "100000",
      estimatedRequiredOrReleasedCash: "0",
      holdExpectedValue: "0",
      proposedExpectedValue: "0",
      estimatedTransactionCost: "0",
      estimatedTaxEffect: "0",
      netExpectedImprovement: "0",
      downsideLevel: 0.1,
      uncertaintyLevel: 0.1,
      positiveReasons: [],
      negativeReasons: [],
    },
    {
      lineId: "skip-line",
      symbol: "D",
      companyName: "見送会社",
      accountBucketId: "bucket",
      currentShares: 0,
      recommendedShares: 0,
      shareDifference: 0,
      action: "SKIP" as const,
      referencePrice: "1000",
      currentMarketValue: "0",
      recommendedMarketValue: "0",
      estimatedRequiredOrReleasedCash: "0",
      holdExpectedValue: "0",
      proposedExpectedValue: "0",
      estimatedTransactionCost: "0",
      estimatedTaxEffect: "0",
      netExpectedImprovement: "0",
      downsideLevel: 0.1,
      uncertaintyLevel: 0.1,
      positiveReasons: [],
      negativeReasons: [],
    },
  ],
};

const today: TodayResponse = {
  businessDate: "2026-08-24",
  pipelineState: "PROPOSAL_READY",
  isStale: false,
  blockingReason: null,
  runtimeMode: "DETERMINISTIC_FIXTURE_ONLY",
  proposal,
};

function jsonResponse(value: unknown): Response {
  return new Response(JSON.stringify(value), { status: 200, headers: { "Content-Type": "application/json" } });
}

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("Goal 5 PWA", () => {
  it("renders five fixed navigation items", () => {
    render(<MemoryRouter><AppShell /></MemoryRouter>);
    const nav = screen.getByRole("navigation", { name: "メインナビゲーション" });
    expect(nav.querySelectorAll("a")).toHaveLength(5);
    for (const label of ["ホーム", "今日", "ランキング", "検証", "設定"]) {
      expect(within(nav).getAllByText(label).length).toBeGreaterThan(0);
    }
  });

  it("shows SELL before BUY, folds HOLD, and excludes SKIP from Today", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => jsonResponse(today)));
    render(<MemoryRouter><TodayPage /></MemoryRouter>);
    await screen.findByText("売却会社");
    const content = document.body.textContent || "";
    expect(content.indexOf("売却会社")).toBeLessThan(content.indexOf("購入会社"));
    expect(screen.getByText("HOLD 1件を確認")).toBeInTheDocument();
    expect(screen.queryByText("見送会社")).not.toBeInTheDocument();
    expect(screen.getByText("保存しても証券会社への注文は行われません")).toBeInTheDocument();
  });

  it("recalculates 100-share decisions and saves only to the manual-record API", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url === "/api/v1/today") return jsonResponse(today);
      if (url.endsWith("/review")) {
        return jsonResponse({
          proposal_id: "proposal-1",
          selected_targets: { "sell-line": 0, "buy-line": 100, "hold-line": 100, "skip-line": 0 },
          estimated_cash_after: { bucket: "100000" },
          estimated_buy_value: "100000",
          estimated_sell_value: "100000",
          estimated_transaction_cost: "200",
          estimated_tax_effect: "0",
          resulting_positions: 2,
          constraint_violations: [],
          is_order_instruction: false,
        });
      }
      if (url.endsWith("/decision") && init?.method !== "POST") return jsonResponse(null);
      if (url.endsWith("/decisions")) return jsonResponse({ version: 1 });
      throw new Error(`unexpected URL ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);
    vi.stubGlobal("crypto", { randomUUID: () => "fixed-id" });
    render(<MemoryRouter><DecisionReviewPage /></MemoryRouter>);
    const selector = await screen.findByLabelText("購入会社の目標株数");
    expect(selector).toHaveDisplayValue("AI提案 100株");
    expect((await screen.findAllByText("100,000円", {}, { timeout: 2000 })).length).toBeGreaterThan(0);
    await userEvent.click(screen.getByRole("button", { name: "判断を保存" }));
    await waitFor(() => expect(fetchMock.mock.calls.some(([url]) => String(url).endsWith("/decisions"))).toBe(true));
    const saveCall = fetchMock.mock.calls.find(([url]) => String(url).endsWith("/decisions"));
    expect(saveCall?.[1]?.headers).toMatchObject({ "X-Stock-AI-Intent": "manual-record" });
    expect(fetchMock.mock.calls.some(([url]) => String(url).includes("order"))).toBe(false);
  });

  it("records editable actual fill fields and can apply them to the next portfolio", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url === "/api/v1/today") return jsonResponse(today);
      if (url.endsWith("/decision")) return jsonResponse({
        decision_id: "decision-1",
        version: 1,
        lines: proposal.lines.map((line) => ({
          proposal_line_id: line.lineId,
          selected_target_shares: line.recommendedShares,
        })),
      });
      if (url.includes("/executions") || url === "/api/v1/portfolio/apply-executions") {
        return jsonResponse({ ok: true });
      }
      throw new Error(`unexpected URL ${url} ${init?.method || "GET"}`);
    });
    vi.stubGlobal("fetch", fetchMock);
    vi.stubGlobal("crypto", { randomUUID: () => "fixed-id" });
    render(<MemoryRouter><ExecutionPage /></MemoryRouter>);
    const company = await screen.findByText("購入会社");
    const card = company.closest("section");
    expect(card).not.toBeNull();
    const cardView = within(card as HTMLElement);
    await userEvent.selectOptions(cardView.getByLabelText("実行状態"), "partially_filled");
    await userEvent.clear(cardView.getByLabelText("約定株数"));
    await userEvent.type(cardView.getByLabelText("約定株数"), "100");
    await userEvent.clear(cardView.getByLabelText("源泉徴収税額"));
    await userEvent.type(cardView.getByLabelText("源泉徴収税額"), "50");
    await userEvent.click(cardView.getByRole("button", { name: "この結果を記録" }));
    await waitFor(() => expect(fetchMock.mock.calls.some(([url]) => String(url).includes("/executions"))).toBe(true));
    const executionCall = fetchMock.mock.calls.find(([url]) => String(url).includes("/executions"));
    const executionBody = JSON.parse(String(executionCall?.[1]?.body)) as Record<string, unknown>;
    expect(executionBody).toMatchObject({ filledShares: 100, taxWithheld: "50", side: "BUY" });
    await userEvent.selectOptions(cardView.getByLabelText("実行状態"), "cancelled");
    await userEvent.clear(cardView.getByLabelText("約定株数"));
    await userEvent.type(cardView.getByLabelText("約定株数"), "0");
    await userEvent.click(cardView.getByRole("button", { name: "この結果を記録" }));
    await waitFor(() => expect(
      fetchMock.mock.calls.filter(([url]) => String(url).includes("/executions")),
    ).toHaveLength(2));
    const cancellationCall = fetchMock.mock.calls.filter(
      ([url]) => String(url).includes("/executions"),
    ).at(-1);
    const cancellationBody = JSON.parse(String(cancellationCall?.[1]?.body)) as Record<string, unknown>;
    expect(cancellationBody).toMatchObject({
      status: "cancelled",
      filledShares: 0,
      averageFillPrice: null,
      actualCommission: "0",
      actualOtherCost: "0",
      taxWithheld: "0",
    });
    await userEvent.click(screen.getByRole("button", { name: "記録済み約定を実保有へ反映" }));
    await waitFor(() => expect(fetchMock.mock.calls.some(([url]) => String(url) === "/api/v1/portfolio/apply-executions")).toBe(true));
  });

  it("renders the formal capital and model settings routes without Notifications API", async () => {
    vi.stubGlobal("Notification", undefined);
    vi.stubGlobal("fetch", vi.fn(async () => jsonResponse({
      runtimeMode: "FIXTURE",
      runtime: { remoteAccess: "LOCALHOST_ONLY", orderSubmission: "OUT_OF_SCOPE" },
      data: { jQuantsApiKeyConfigured: false, credentialValueExposed: false },
      capital: {
        availableCash: "100000",
        reservedCash: "10000",
        asOf: proposal.asOf,
        minimumCashRatio: null,
        dailyProposalLimit: null,
      },
      accounts: [],
      decisionPolicies: {
        decisionEngineVersion: "engine-v1",
        costPolicyId: "cost",
        costPolicyVersion: "1",
        taxPolicyId: "tax",
        taxPolicyVersion: "1",
        roundLotShares: 100,
        maximumPositions: null,
        maximumSymbolWeight: null,
        maximumSectorWeight: null,
        maximumTurnoverRatio: null,
        maximumTradeAdvRatio: null,
        minimumImprovementYen: null,
        uncertaintyBufferYen: null,
      },
      notifications: { inApp: "AVAILABLE", webPush: "BLOCKED_BY_CONFIGURATION" },
      decision: {
        freezeTime: "11:30 JST",
        recommendedTradeTime: "12:30 JST 後場寄り",
        method: "Daily Portfolio Decision Engine",
      },
      model: {
        morningChampion: "BLOCKED_BY_DATA_CAPABILITY",
        trainedAt: null,
        trainingDataEnd: null,
        validationStatus: "BLOCKED_BY_DATA_CAPABILITY",
        automaticPromotion: false,
      },
    })));
    const capitalView = render(
      <MemoryRouter initialEntries={["/settings/capital"]}>
        <Routes><Route path="/settings/:section" element={<SettingsPage />} /></Routes>
      </MemoryRouter>,
    );
    await screen.findByText("運用対象資金");
    capitalView.unmount();
    render(
      <MemoryRouter initialEntries={["/settings/model"]}>
        <Routes><Route path="/settings/:section" element={<SettingsPage />} /></Routes>
      </MemoryRouter>,
    );
    await screen.findByText("検証状態");
    expect(screen.queryByText("不明な設定項目です")).not.toBeInTheDocument();
  });

  it("switches same-symbol account buckets without mixing prices or shares", async () => {
    const detail = {
      symbol: "A",
      positions: [
        { symbol: "A", account_bucket_id: "taxable", shares: 200, average_acquisition_price: "900", market_price: "1000" },
        { symbol: "A", account_bucket_id: "nisa", shares: 100, average_acquisition_price: "800", market_price: "1000" },
      ],
      proposalLines: [
        { ...proposal.lines[0], lineId: "tax-line", symbol: "A", accountBucketId: "taxable", currentShares: 200, recommendedShares: 100, shareDifference: -100 },
        { ...proposal.lines[2], lineId: "nisa-line", symbol: "A", accountBucketId: "nisa", currentShares: 100, recommendedShares: 100 },
      ],
      proposal,
    };
    vi.stubGlobal("fetch", vi.fn(async () => jsonResponse(detail)));
    render(<MemoryRouter initialEntries={["/stocks/A"]}><StockDetailPage /></MemoryRouter>);
    await screen.findByText("実保有合計");
    expect(screen.getByText("300株")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("tab", { name: "nisa" }));
    expect(screen.getByText("実保有 / nisa")).toBeInTheDocument();
    expect(screen.getAllByText("100株").length).toBeGreaterThan(0);
    expect(screen.getByText("800円")).toBeInTheDocument();
  });
});
