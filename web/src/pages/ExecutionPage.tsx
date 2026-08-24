import { type ReactNode, useEffect, useState } from "react";
import { apiGet, apiPost } from "../api";
import { PageHeader, StatePanel, yen } from "../components";
import { useApiData } from "../hooks";
import type { ProposalLine, TodayResponse } from "../types";

interface DecisionLine {
  proposal_line_id: string;
  selected_target_shares: number;
}

interface SavedDecision {
  decision_id: string;
  version: number;
  lines: DecisionLine[];
}

interface EntryState {
  status: string;
  side: "BUY" | "SELL";
  orderedShares: number;
  filledShares: number;
  executedAt: string;
  price: string;
  commission: string;
  otherCost: string;
  taxWithheld: string;
  confirmDifference: boolean;
}

function localDateTimeValue(value: Date): string {
  const pad = (part: number): string => part.toString().padStart(2, "0");
  const milliseconds = value.getMilliseconds().toString().padStart(3, "0");
  return `${value.getFullYear()}-${pad(value.getMonth() + 1)}-${pad(value.getDate())}T${pad(value.getHours())}:${pad(value.getMinutes())}:${pad(value.getSeconds())}.${milliseconds}`;
}

const initialEntry: EntryState = {
  status: "filled",
  side: "BUY",
  orderedShares: 0,
  filledShares: 0,
  executedAt: "",
  price: "",
  commission: "0",
  otherCost: "0",
  taxWithheld: "0",
  confirmDifference: false,
};

export function ExecutionPage(): ReactNode {
  const today = useApiData<TodayResponse>("/api/v1/today");
  const [decision, setDecision] = useState<SavedDecision | null>(null);
  const [loadingDecision, setLoadingDecision] = useState(true);
  const [entries, setEntries] = useState<Record<string, EntryState>>({});
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const tomorrow = new Date(Date.now() + 86_400_000);
  const [nextAsOf, setNextAsOf] = useState(localDateTimeValue(tomorrow));
  const proposal = today.data?.proposal;

  useEffect(() => {
    if (!proposal) {
      setLoadingDecision(false);
      return;
    }
    apiGet<SavedDecision | null>(`/api/v1/proposals/${proposal.proposalId}/decision`)
      .then((value) => setDecision(value))
      .catch((reason: unknown) => setError(reason instanceof Error ? reason.message : "判断を取得できません"))
      .finally(() => setLoadingDecision(false));
  }, [proposal]);

  if (today.loading || loadingDecision) return <StatePanel title="実行結果入力を準備しています" />;
  if (today.error || error) return <StatePanel tone="error" title="入力を準備できません" detail={today.error || error} />;
  if (!proposal || !decision) return <StatePanel title="実行待ちの保存済み判断がありません" />;

  const savedDecision = decision;
  const decisionTargets = new Map(savedDecision.lines.map((line) => [line.proposal_line_id, line.selected_target_shares]));
  const changed = proposal.lines.filter((line) => decisionTargets.get(line.lineId) !== line.currentShares);

  function entryFor(line: ProposalLine): EntryState {
    const target = decisionTargets.get(line.lineId) ?? line.currentShares;
    return entries[line.lineId] || {
      ...initialEntry,
      side: target > line.currentShares ? "BUY" : "SELL",
      orderedShares: Math.abs(target - line.currentShares),
      filledShares: Math.abs(target - line.currentShares),
      executedAt: localDateTimeValue(new Date()),
      price: line.referencePrice,
    };
  }

  function patchEntry(line: ProposalLine, patch: Partial<EntryState>): void {
    setEntries((current) => ({ ...current, [line.lineId]: { ...entryFor(line), ...patch } }));
  }

  async function record(line: ProposalLine): Promise<void> {
    const entry = entryFor(line);
    const zeroFill = ["not_ordered", "open"].includes(entry.status)
      || (["cancelled", "expired"].includes(entry.status) && entry.filledShares === 0);
    setError(null);
    setMessage(null);
    try {
      await apiPost(`/api/v1/decisions/${savedDecision.decision_id}/executions`, {
        executionId: `execution-${crypto.randomUUID()}`,
        executedAt: new Date(entry.executedAt).toISOString(),
        symbol: line.symbol,
        accountBucketId: line.accountBucketId,
        status: entry.status,
        side: entry.side,
        orderedShares: entry.orderedShares,
        filledShares: zeroFill ? 0 : entry.filledShares,
        averageFillPrice: zeroFill ? null : entry.price,
        actualCommission: zeroFill ? "0" : entry.commission,
        actualOtherCost: zeroFill ? "0" : entry.otherCost,
        taxWithheld: zeroFill ? "0" : entry.taxWithheld,
        confirmDifference: entry.confirmDifference,
      });
      setMessage(`${line.companyName}の実行結果を記録しました。`);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "実行結果を記録できませんでした");
    }
  }

  async function applyRecorded(): Promise<void> {
    setError(null);
    setMessage(null);
    try {
      await apiPost("/api/v1/portfolio/apply-executions", {
        nextAsOf: new Date(nextAsOf).toISOString(),
        nextPortfolioId: `actual-${crypto.randomUUID()}`,
      });
      setMessage("記録済み約定から次の実保有状態を追加しました。");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "次の実保有状態へ反映できませんでした");
    }
  }

  return (
    <div className="page">
      <PageHeader eyebrow="証券会社での手動操作後" title="実行結果を記録" />
      <section className="notice-box"><strong>実際の結果だけを入力</strong><span>提案株数ではなく、証券会社で確認した約定株数・単価を記録してください。</span></section>
      {changed.length === 0 ? <StatePanel title="取引しない判断です" /> : changed.map((line) => {
        const target = decisionTargets.get(line.lineId) ?? line.currentShares;
        const orderedShares = Math.abs(target - line.currentShares);
        const entry = entryFor(line);
        return (
          <section className="execution-card" key={line.lineId}>
            <div className="execution-card__title"><div><strong>{line.companyName}</strong><span>{line.accountBucketId}</span></div><b>{target > line.currentShares ? "BUY" : "SELL"} {orderedShares}株</b></div>
            <label>実行状態<select value={entry.status} onChange={(event) => patchEntry(line, { status: event.target.value })}>
              <option value="not_ordered">未注文</option><option value="open">注文済み・未約定</option><option value="partially_filled">一部約定</option><option value="filled">全約定</option><option value="cancelled">取消</option><option value="expired">失効</option>
            </select></label>
            <div className="form-grid">
              <label>売買方向<select value={entry.side} onChange={(event) => patchEntry(line, { side: event.target.value as "BUY" | "SELL" })}><option value="BUY">BUY</option><option value="SELL">SELL</option></select></label>
              <label>注文株数<input type="number" min="0" step="100" value={entry.orderedShares} onChange={(event) => patchEntry(line, { orderedShares: Number(event.target.value) })} /></label>
              <label>実行日時<input type="datetime-local" step="0.001" value={entry.executedAt} onChange={(event) => patchEntry(line, { executedAt: event.target.value })} /></label>
            </div>
            {!(["not_ordered", "open"].includes(entry.status)) && <div className="form-grid">
              <label>約定株数<input type="number" min="0" step="100" value={entry.filledShares} onChange={(event) => patchEntry(line, { filledShares: Number(event.target.value) })} /></label>
              <label>平均約定単価<input inputMode="decimal" value={entry.price} onChange={(event) => patchEntry(line, { price: event.target.value })} /></label>
              <label>実手数料<input inputMode="decimal" value={entry.commission} onChange={(event) => patchEntry(line, { commission: event.target.value })} /></label>
              <label>その他費用<input inputMode="decimal" value={entry.otherCost} onChange={(event) => patchEntry(line, { otherCost: event.target.value })} /></label>
              <label>源泉徴収税額<input inputMode="decimal" value={entry.taxWithheld} onChange={(event) => patchEntry(line, { taxWithheld: event.target.value })} /></label>
            </div>}
            <p className="reference">参考価格 {yen(line.referencePrice)} / 判断 {line.currentShares}株 → {target}株</p>
            <label className="checkbox"><input type="checkbox" checked={entry.confirmDifference} onChange={(event) => patchEntry(line, { confirmDifference: event.target.checked })} />保存済み判断と異なる実行を明示確認する</label>
            <button className="button button--secondary" onClick={() => void record(line)}>この結果を記録</button>
          </section>
        );
      })}
      <section className="section preview-panel">
        <h2>翌日以降の実保有へ反映</h2>
        <p className="explain">すべての手動注文結果を記録した後、反映時点を確認して実行します。提案株数ではなく、記録済み約定だけを使用します。</p>
        <label>次の保有状態の時点<input type="datetime-local" value={nextAsOf} onChange={(event) => setNextAsOf(event.target.value)} /></label>
        <button className="button button--primary" onClick={() => void applyRecorded()}>記録済み約定を実保有へ反映</button>
      </section>
      {message && <StatePanel title={message} detail="翌日状態は確定した実約定だけから更新されます。" />}
      {error && <StatePanel tone="error" title="記録できませんでした" detail={error} />}
    </div>
  );
}
