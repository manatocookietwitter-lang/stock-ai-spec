import { type ReactNode } from "react";
import { NavLink, Outlet } from "react-router-dom";
import type { Action } from "./types";

export function yen(value: string | number | null | undefined): string {
  if (value === null || value === undefined || value === "") return "—";
  const number = Number(value);
  if (!Number.isFinite(number)) return "—";
  return `${Math.round(number).toLocaleString("ja-JP")}円`;
}

export function pct(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  const sign = value > 0 ? "+" : "";
  return `${sign}${(value * 100).toFixed(2)}%`;
}

export function dateTime(value: string | null | undefined): string {
  if (!value) return "—";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.valueOf())) return "—";
  return new Intl.DateTimeFormat("ja-JP", {
    month: "numeric",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(parsed);
}

export function ActionBadge({ action }: { action: Action }): ReactNode {
  return <span className={`action action--${action.toLowerCase()}`}>{action}</span>;
}

export function PageHeader({
  eyebrow,
  title,
  meta,
}: {
  eyebrow?: string;
  title: string;
  meta?: string;
}): ReactNode {
  return (
    <header className="page-header">
      {eyebrow && <p className="eyebrow">{eyebrow}</p>}
      <div className="page-header__line">
        <h1>{title}</h1>
        {meta && <span className="page-header__meta">{meta}</span>}
      </div>
    </header>
  );
}

export function StatePanel({
  title,
  detail,
  tone = "neutral",
}: {
  title: string;
  detail?: string | null;
  tone?: "neutral" | "error" | "warning";
}): ReactNode {
  return (
    <section className={`state-panel state-panel--${tone}`} role={tone === "error" ? "alert" : "status"}>
      <strong>{title}</strong>
      {detail && <p>{detail}</p>}
    </section>
  );
}

export function RuntimeBanner({ mode }: { mode: string }): ReactNode {
  if (mode === "LIVE") return null;
  const fixture = mode === "DETERMINISTIC_FIXTURE_ONLY";
  return (
    <div className="runtime-banner" role="status">
      <strong>{fixture ? "Fixture表示" : "運用能力を確認してください"}</strong>
      <span>
        {fixture
          ? "この数値は画面確認用です。実データや収益性を示しません。"
          : "liveデータ・採用modelが揃うまで提案は停止します。"}
      </span>
    </div>
  );
}

const nav = [
  ["/", "資産", "ホーム"],
  ["/today", "判断", "今日"],
  ["/ranking", "探索", "ランキング"],
  ["/validation", "検証", "検証"],
  ["/settings", "管理", "設定"],
] as const;

export function AppShell(): ReactNode {
  return (
    <div className="app-shell">
      <a className="skip-link" href="#main">本文へ移動</a>
      <main id="main" className="app-main">
        <Outlet />
      </main>
      <nav className="bottom-nav" aria-label="メインナビゲーション">
        {nav.map(([to, overline, label]) => (
          <NavLink key={to} to={to} end={to === "/"}>
            <span>{overline}</span>
            <strong>{label}</strong>
          </NavLink>
        ))}
      </nav>
    </div>
  );
}

export function Metric({ label, value, tone }: { label: string; value: string; tone?: string }): ReactNode {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong className={tone ? `text-${tone}` : undefined}>{value}</strong>
    </div>
  );
}
