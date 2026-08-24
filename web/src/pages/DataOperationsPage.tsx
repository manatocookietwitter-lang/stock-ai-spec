import { type ReactNode, useState } from "react";
import { Link } from "react-router-dom";
import { apiPost } from "../api";
import { PageHeader, StatePanel } from "../components";
import { useApiData } from "../hooks";
import type { AppStatus } from "../types";

interface Conflict { conflict_id: string; code: string; message: string; row_number: number }
interface ExecutionPreview { preview_id: string; conflicts: Conflict[]; execution_payloads: string[] }
interface PositionDifference { symbol: string; account_bucket_id: string; ledger_shares: number; imported_shares: number }
interface CashDifference { account_bucket_id: string; ledger_available_cash: string; imported_available_cash: string; ledger_reserved_cash: string; imported_reserved_cash: string }
interface PositionPreview { preview_id: string; differences: PositionDifference[]; cash_differences: CashDifference[]; imported_as_of: string }

export function DataOperationsPage(): ReactNode {
  const status = useApiData<AppStatus>("/api/v1/status");
  const [kind, setKind] = useState<"executions" | "positions">("executions");
  const [csv, setCsv] = useState("");
  const [executionPreview, setExecutionPreview] = useState<ExecutionPreview | null>(null);
  const [positionPreview, setPositionPreview] = useState<PositionPreview | null>(null);
  const [accepted, setAccepted] = useState<string[]>([]);
  const [confirmed, setConfirmed] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function preview(): Promise<void> {
    setError(null); setMessage(null); setExecutionPreview(null); setPositionPreview(null); setConfirmed(false);
    try {
      if (kind === "executions") {
        setExecutionPreview(await apiPost<ExecutionPreview>("/api/v1/imports/executions/preview", { csvText: csv }));
      } else {
        setPositionPreview(await apiPost<PositionPreview>("/api/v1/imports/positions/preview", { csvText: csv }));
      }
    } catch (reason) { setError(reason instanceof Error ? reason.message : "CSVを確認できませんでした"); }
  }

  async function confirmImport(): Promise<void> {
    setError(null);
    try {
      if (executionPreview) {
        const result = await apiPost<{ imported: number; skipped: number }>(`/api/v1/imports/executions/${executionPreview.preview_id}/confirm`, { acceptedConflictIds: accepted });
        setMessage(`${result.imported}件を追加し、${result.skipped}件を既存のまま保持しました。`);
      } else if (positionPreview) {
        await apiPost(`/api/v1/imports/positions/${positionPreview.preview_id}/confirm`, { nextPortfolioId: `reconciled-${crypto.randomUUID()}`, confirmAllDifferences: confirmed });
        setMessage("差分確認済みの口座状態を新しい実状態として追加しました。");
      }
    } catch (reason) { setError(reason instanceof Error ? reason.message : "確認を保存できませんでした"); }
  }

  return (
    <div className="page">
      <Link className="back-link" to="/settings">← 設定へ戻る</Link>
      <PageHeader eyebrow="IMPORT / RECONCILIATION" title="データ・運用状態" />
      {status.loading ? <StatePanel title="運用状態を確認しています" /> : status.error ? <StatePanel tone="error" title="運用状態を取得できません" detail={status.error} /> : status.data && <section className={`pipeline-state ${status.data.isStale ? "pipeline-state--warning" : ""}`}>
        <span>{status.data.businessDate}</span><strong>{status.data.pipelineState}</strong><p>{status.data.blockingReason || "直近jobの状態を表示しています。"}</p>
      </section>}

      <section className="section">
        <div className="section-title"><div><span>PREVIEW FIRST</span><h2>CSV差分確認</h2></div></div>
        <div className="segmented segmented--2"><button className={kind === "executions" ? "active" : ""} onClick={() => setKind("executions")}>約定CSV</button><button className={kind === "positions" ? "active" : ""} onClick={() => setKind("positions")}>口座状態CSV</button></div>
        <textarea aria-label="CSV内容" className="csv-input" value={csv} onChange={(event) => setCsv(event.target.value)} placeholder={kind === "executions" ? "execution_id,decision_id,executed_at,..." : "as_of,record_type,symbol,account_bucket_id,...,available_cash,reserved_cash"} />
        <button className="button button--secondary" disabled={!csv.trim()} onClick={() => void preview()}>書き込まずに差分を確認</button>
      </section>

      {executionPreview && <section className="section preview-panel">
        <h2>約定preview</h2><p>候補 {executionPreview.execution_payloads.length}件 / conflict {executionPreview.conflicts.length}件</p>
        {executionPreview.conflicts.map((conflict) => <label className="conflict-row" key={conflict.conflict_id}><input type="checkbox" checked={accepted.includes(conflict.conflict_id)} onChange={(event) => setAccepted((current) => event.target.checked ? [...current, conflict.conflict_id] : current.filter((item) => item !== conflict.conflict_id))} /><span><strong>{conflict.code}</strong> 行{conflict.row_number}: {conflict.message}</span></label>)}
        <button className="button button--primary" onClick={() => void confirmImport()}>確認した行だけ追加</button>
      </section>}

      {positionPreview && <section className="section preview-panel">
        <h2>保有差分preview</h2><p>取込時点 {positionPreview.imported_as_of}</p>
        {positionPreview.differences.map((difference) => <div className="difference-row" key={`${difference.symbol}-${difference.account_bucket_id}`}><span>{difference.symbol} / {difference.account_bucket_id}</span><strong>{difference.ledger_shares}株 → {difference.imported_shares}株</strong></div>)}
        {positionPreview.cash_differences.map((difference) => <div className="difference-row" key={`cash-${difference.account_bucket_id}`}><span>現金 / {difference.account_bucket_id}</span><strong>{difference.ledger_available_cash}円（予約 {difference.ledger_reserved_cash}円） → {difference.imported_available_cash}円（予約 {difference.imported_reserved_cash}円）</strong></div>)}
        <label className="checkbox"><input type="checkbox" checked={confirmed} onChange={(event) => setConfirmed(event.target.checked)} />全差分を確認し、新しい実状態として追加する</label>
        <button className="button button--primary" disabled={!confirmed} onClick={() => void confirmImport()}>差分確認済みで反映</button>
      </section>}
      {message && <StatePanel title={message} detail="既存recordは上書きしていません。" />}
      {error && <StatePanel tone="error" title="処理を完了できません" detail={error} />}
    </div>
  );
}
