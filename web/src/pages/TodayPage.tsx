import { type ReactNode } from "react";
import { Link } from "react-router-dom";
import { ActionBadge, Metric, PageHeader, RuntimeBanner, StatePanel, dateTime, yen } from "../components";
import { useApiData } from "../hooks";
import type { ProposalLine, TodayResponse } from "../types";

const priority = new Map([
  ["SELL", 0],
  ["REDUCE", 1],
  ["BUY", 2],
  ["HOLD", 3],
  ["SKIP", 4],
]);

function ProposalRow({ line }: { line: ProposalLine }): ReactNode {
  return (
    <Link className="proposal-row" to={`/stocks/${line.symbol}`}>
      <ActionBadge action={line.action} />
      <div className="proposal-row__body">
        <strong>{line.companyName}</strong>
        <span>{line.symbol} ・ {line.accountBucketId}</span>
      </div>
      <div className="proposal-row__shares">
        <strong>{line.currentShares}株 → {line.recommendedShares}株</strong>
        <span>{line.shareDifference === 0 ? "変更なし" : `${line.shareDifference > 0 ? "+" : ""}${line.shareDifference}株`}</span>
      </div>
    </Link>
  );
}

export function TodayPage(): ReactNode {
  const { data, error, loading } = useApiData<TodayResponse>("/api/v1/today");
  if (loading) return <StatePanel title="今日の状態を確認しています" />;
  if (error) return <StatePanel tone="error" title="今日の提案を取得できません" detail={error} />;
  if (!data) return <StatePanel tone="error" title="状態を確認できません" />;
  const proposal = data.proposal;
  const visible = proposal?.lines
    .filter((line) => line.action !== "SKIP" && line.action !== "HOLD")
    .sort((a, b) => (priority.get(a.action) ?? 9) - (priority.get(b.action) ?? 9));
  const holds = proposal?.lines.filter((line) => line.action === "HOLD") || [];

  return (
    <div className="page page--with-sticky">
      <PageHeader eyebrow={data.businessDate} title="今日" meta={proposal ? `更新 ${dateTime(proposal.generatedAt)}` : undefined} />
      <RuntimeBanner mode={data.runtimeMode} />
      {!proposal ? (
        <StatePanel
          tone={data.isStale ? "warning" : "neutral"}
          title={data.isStale ? "今日は提案を表示できません" : "次回の分析を待っています"}
          detail={data.blockingReason || `現在の状態: ${data.pipelineState}`}
        />
      ) : (
        <>
          <section className="proposal-hero">
            <div><span>11:30 FREEZE</span><strong>{proposal.changeCount}件の変更</strong></div>
            <div><span>モデル</span><strong>{proposal.modelBundleVersion}</strong></div>
          </section>
          <section className="section">
            <div className="section-title"><div><span>TODAY'S ACTIONS</span><h2>株数の変更</h2></div></div>
            {visible && visible.length > 0 ? visible.map((line) => <ProposalRow key={line.lineId} line={line} />) : (
              <StatePanel title="今日は株数変更なし" detail={proposal.noTradeReason || "HOLD比較後の純改善が基準に届きませんでした。"} />
            )}
            {holds.length > 0 && (
              <details className="hold-section">
                <summary>HOLD {holds.length}件を確認</summary>
                {holds.map((line) => <ProposalRow key={line.lineId} line={line} />)}
              </details>
            )}
          </section>
          <section className="section summary-panel">
            <div className="section-title"><div><span>ESTIMATE</span><h2>提案サマリー</h2></div></div>
            <div className="metric-grid metric-grid--2">
              <Metric label="推定売却額" value={yen(proposal.estimatedSellValue)} />
              <Metric label="推定購入額" value={yen(proposal.estimatedBuyValue)} />
              <Metric label="推定取引コスト" value={yen(proposal.estimatedTransactionCost)} />
              <Metric label="推定税影響" value={yen(proposal.estimatedTaxEffect)} />
            </div>
            <div className="summary-total"><span>純改善</span><strong>{yen(proposal.netImprovement)}</strong></div>
          </section>
          <div className="sticky-action">
            <p>保存しても証券会社への注文は行われません</p>
            <Link className="button button--primary" to="/today/review">判断を確認</Link>
          </div>
        </>
      )}
    </div>
  );
}
