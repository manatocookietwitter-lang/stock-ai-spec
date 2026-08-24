import { type ReactNode } from "react";
import { Link } from "react-router-dom";
import { PageHeader, StatePanel } from "../components";
import { useApiData } from "../hooks";
import type { TodayResponse } from "../types";

interface SavedDecision {
  decision_id: string;
  version: number;
  saved_at: string;
  confirms_manual_order_only: boolean;
}

export function DecisionSavedPage(): ReactNode {
  const today = useApiData<TodayResponse>("/api/v1/today");
  const proposalId = today.data?.proposal?.proposalId;
  const decision = useApiData<SavedDecision | null>(proposalId ? `/api/v1/proposals/${proposalId}/decision` : "/api/v1/health");
  if (today.loading || decision.loading) return <StatePanel title="保存済み判断を確認しています" />;
  if (!proposalId || !decision.data) return <StatePanel title="保存済み判断はありません" detail="今日の提案から判断を保存してください。" />;
  return (
    <div className="page">
      <PageHeader eyebrow={`Version ${decision.data.version}`} title="判断を保存しました" />
      <section className="success-panel">
        <span aria-hidden="true">✓</span>
        <strong>ローカル台帳へ保存済み</strong>
        <p>AI提案は変更されていません。証券会社への注文・通信は行っていません。</p>
      </section>
      <div className="button-stack">
        <Link className="button button--primary" to="/today/executions">実行結果を記録</Link>
        <Link className="button button--secondary" to="/today/review">判断を修正して新versionで保存</Link>
        <Link className="text-link" to="/today">今日へ戻る</Link>
      </div>
    </div>
  );
}
