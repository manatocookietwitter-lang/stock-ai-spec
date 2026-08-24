import { type ReactNode, useState } from "react";
import { Link } from "react-router-dom";
import { ActionBadge, PageHeader, StatePanel } from "../components";
import { useApiData } from "../hooks";
import type { RankingRow } from "../types";

const tabs = [
  ["overall", "総合"],
  ["5d", "5日"],
  ["20d", "20日"],
  ["morning", "前場"],
] as const;

export function RankingPage(): ReactNode {
  const [rankType, setRankType] = useState("overall");
  const [search, setSearch] = useState("");
  const { data, error, loading } = useApiData<RankingRow[]>(`/api/v1/ranking?rankType=${rankType}&limit=50`);
  const rows = (data || []).filter((row) => `${row.symbol}${row.company_name}`.toLowerCase().includes(search.toLowerCase()));
  return (
    <div className="page">
      <PageHeader eyebrow="予測順位 ≠ 最終売買" title="ランキング" />
      <div className="segmented" role="tablist" aria-label="ランキング期間">
        {tabs.map(([value, label]) => <button role="tab" aria-selected={rankType === value} className={rankType === value ? "active" : ""} key={value} onClick={() => setRankType(value)}>{label}</button>)}
      </div>
      <label className="search"><span className="sr-only">銘柄名・コードで検索</span><input placeholder="銘柄名・コードで検索" value={search} onChange={(event) => setSearch(event.target.value)} /></label>
      <p className="explain">順位は予測評価です。最終提案は保有、現金、税、コスト、リスクも考慮します。</p>
      {loading ? <StatePanel title="ランキングを読み込んでいます" /> : error ? <StatePanel tone="error" title="ランキングを取得できません" detail={error} /> : rows.length === 0 ? <StatePanel title="このランキングはまだありません" detail="認証済み予測を運用台帳へ登録すると表示されます。" /> : (
        <div className="ranking-list">
          {rows.map((row) => <Link className="ranking-row" key={row.ranking_id} to={`/stocks/${row.symbol}`}>
            <strong className="rank-number">{row.rank}</strong>
            <div><strong>{row.company_name}</strong><span>{row.symbol} ・ {row.rank}位 / {row.total_universe}</span></div>
            <div className="ranking-row__labels"><span className="candidate-label">{row.candidate_status}</span>{row.portfolio_action && <ActionBadge action={row.portfolio_action} />}</div>
          </Link>)}
        </div>
      )}
    </div>
  );
}
