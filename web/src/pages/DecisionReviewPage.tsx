import { type ReactNode, useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { apiGet, apiPost } from "../api";
import { ActionBadge, PageHeader, StatePanel, yen } from "../components";
import { useApiData } from "../hooks";
import type { DecisionReview, TodayResponse } from "../types";

interface SavedDecision {
  version: number;
}

export function DecisionReviewPage(): ReactNode {
  const navigate = useNavigate();
  const { data, error, loading } = useApiData<TodayResponse>("/api/v1/today");
  const [targets, setTargets] = useState<Record<string, number>>({});
  const [review, setReview] = useState<DecisionReview | null>(null);
  const [reviewError, setReviewError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const proposal = data?.proposal;

  useEffect(() => {
    if (!proposal) return;
    setTargets(Object.fromEntries(proposal.lines.map((line) => [line.lineId, line.recommendedShares])));
  }, [proposal]);

  useEffect(() => {
    if (!proposal || Object.keys(targets).length !== proposal.lines.length) return;
    let active = true;
    apiPost<DecisionReview>(`/api/v1/proposals/${proposal.proposalId}/review`, { selectedTargets: targets })
      .then((value) => {
        if (active) {
          setReview(value);
          setReviewError(null);
        }
      })
      .catch((reason: unknown) => {
        if (active) setReviewError(reason instanceof Error ? reason.message : "再計算に失敗しました");
      });
    return () => { active = false; };
  }, [proposal, targets]);

  if (loading) return <StatePanel title="判断内容を準備しています" />;
  if (error) return <StatePanel tone="error" title="提案を取得できません" detail={error} />;
  if (!proposal) return <StatePanel tone="warning" title="有効な当日提案がありません" detail={data?.blockingReason} />;

  async function save(): Promise<void> {
    if (!proposal || !review || review.constraint_violations.length > 0) return;
    setSaving(true);
    setReviewError(null);
    try {
      const existing = await apiGet<SavedDecision | null>(`/api/v1/proposals/${proposal.proposalId}/decision`);
      await apiPost(`/api/v1/proposals/${proposal.proposalId}/decisions`, {
        decisionId: `decision-${crypto.randomUUID()}`,
        version: (existing?.version || 0) + 1,
        savedAt: new Date().toISOString(),
        confirmsManualOrderOnly: true,
        lines: proposal.lines.map((line) => ({
          proposalLineId: line.lineId,
          selectedTargetShares: targets[line.lineId],
        })),
      });
      navigate("/today/decision");
    } catch (reason) {
      setReviewError(reason instanceof Error ? reason.message : "判断を保存できませんでした");
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="page page--with-sticky">
      <PageHeader eyebrow="AI提案とは別に保存" title="判断を確認" />
      <section className="notice-box"><strong>これは注文画面ではありません</strong><span>証券会社での操作はご自身で行い、結果を後から記録します。</span></section>
      <section className="section review-list">
        {proposal.lines.filter((line) => line.action !== "SKIP").map((line) => (
          <article className="review-row" key={line.lineId}>
            <div className="review-row__title">
              <div><strong>{line.companyName}</strong><span>{line.accountBucketId}</span></div>
              <ActionBadge action={line.action} />
            </div>
            <div className="ai-target"><span>AI提案</span><strong>{line.currentShares}株 → {line.recommendedShares}株</strong></div>
            <label>
              あなたの判断（100株単位）
              <select
                aria-label={`${line.companyName}の目標株数`}
                value={targets[line.lineId] ?? line.recommendedShares}
                onChange={(event) => setTargets((current) => ({ ...current, [line.lineId]: Number(event.target.value) }))}
              >
                <option value={line.recommendedShares}>AI提案 {line.recommendedShares}株</option>
                {line.currentShares !== line.recommendedShares && <option value={line.currentShares}>取引しない {line.currentShares}株</option>}
              </select>
            </label>
            <small>参考価格 {yen(line.referencePrice)}</small>
          </article>
        ))}
      </section>
      <section className="section summary-panel">
        <div className="section-title"><div><span>RECALCULATED</span><h2>判断後の見込み</h2></div></div>
        {review ? (
          <>
            <dl className="summary-list">
              <div><dt>推定購入額</dt><dd>{yen(review.estimated_buy_value)}</dd></div>
              <div><dt>推定売却額</dt><dd>{yen(review.estimated_sell_value)}</dd></div>
              <div><dt>推定取引コスト</dt><dd>{yen(review.estimated_transaction_cost)}</dd></div>
              <div><dt>推定税影響</dt><dd>{yen(review.estimated_tax_effect)}</dd></div>
              <div><dt>判断後の保有明細</dt><dd>{review.resulting_positions}件</dd></div>
            </dl>
            {Object.entries(review.estimated_cash_after).map(([bucket, cash]) => (
              <div className="cash-line" key={bucket}><span>{bucket} 推定現金</span><strong>{yen(cash)}</strong></div>
            ))}
            {review.constraint_violations.length > 0 && (
              <div className="violation" role="alert">
                <strong>このままでは保存できません</strong>
                {review.constraint_violations.map((item) => <span key={item}>{item}</span>)}
              </div>
            )}
          </>
        ) : <p>再計算中…</p>}
        {reviewError && <StatePanel tone="error" title="再計算または保存に失敗しました" detail={reviewError} />}
      </section>
      <div className="sticky-action">
        <p>保存しても証券会社への注文は行われません</p>
        <button className="button button--primary" disabled={saving || !review || review.constraint_violations.length > 0} onClick={() => void save()}>
          {saving ? "保存中…" : "判断を保存"}
        </button>
        <Link className="text-link" to="/today">提案へ戻る</Link>
      </div>
    </div>
  );
}
