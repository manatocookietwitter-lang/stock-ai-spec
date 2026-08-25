import { getSafetyAcknowledgement } from '../db/operator-settings';
import { updateSafetyAcknowledgement } from './actions';
import {
  chatGPTSignInPath,
  chatGPTSignOutPath,
  getChatGPTUser,
} from './chatgpt-auth';

export const dynamic = 'force-dynamic';

const readiness = [
  {
    label: 'J-Quants V2 日足',
    detail: '2017〜2020年の取得・検証済み範囲',
    state: '準備済み',
    tone: 'ready',
  },
  {
    label: '研究対象期間',
    detail: '2021年以降の履歴取得が未完了',
    state: '未完了',
    tone: 'waiting',
  },
  {
    label: 'Production Dataset',
    detail: '必要期間の取得後に生成・検証',
    state: '待機中',
    tone: 'waiting',
  },
  {
    label: '11:30 前場データ',
    detail: '利用可能なデータ提供元が未接続',
    state: 'データ待ち',
    tone: 'blocked',
  },
  {
    label: '実口座・税状態',
    detail: 'SBI実CSVと口座情報が未登録',
    state: '入力待ち',
    tone: 'blocked',
  },
];

const stopReasons = [
  '研究対象期間の実データが揃っていません',
  'Championモデルはまだ固定されていません',
  '11:30時点の前場データが利用できません',
  '実口座・NISA・税状態が未登録です',
];

export default async function Home() {
  const user = await getChatGPTUser();
  const safety = user
    ? await getSafetyAcknowledgement(user.userId)
    : { acknowledged: false, updatedAt: null };
  const now = new Intl.DateTimeFormat('ja-JP', {
    timeZone: 'Asia/Tokyo',
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(new Date());

  return (
    <main className="shell">
      <header className="topbar">
        <a className="brand" href="#home" aria-label="株AI ホーム">
          <span className="brandMark" aria-hidden="true">株</span>
          <span><strong>株AI</strong><small>Decision Support</small></span>
        </a>
        <div className="identity">
          <span className="statusDot" aria-hidden="true" />
          <span>{user?.displayName ?? 'ローカルプレビュー'}</span>
          {user ? (
            <a href={chatGPTSignOutPath('/')} className="accountLink">ログアウト</a>
          ) : (
            <a href={chatGPTSignInPath('/')} className="accountLink">ログイン</a>
          )}
        </div>
      </header>

      <div className="content" id="home">
        <section className="hero">
          <div>
            <span className="eyebrow">認証付きホスト版</span>
            <h1>今日の判断を、<br />安全に一か所へ。</h1>
            <p>実データの準備状況を確認し、利用可能な根拠だけで12:30の保有案を組み立てます。</p>
          </div>
          <div className="timeCard">
            <small>現在時刻（JST）</small>
            <strong>{now}</strong>
            <span>実注文は行いません</span>
          </div>
        </section>

        <section className="proposal card" id="today">
          <div className="sectionHead">
            <div>
              <span className="eyebrow">今日の提案</span>
              <h2>提案を安全に停止しています</h2>
            </div>
            <span className="pill danger">NO PROPOSAL</span>
          </div>
          <p className="lead">不足データを推測で補わず、条件が揃うまでBUY・HOLD・REDUCE・SELL・SKIPを生成しません。</p>
          <div className="reasonGrid">
            {stopReasons.map((reason, index) => (
              <div className="reason" key={reason}>
                <span>{String(index + 1).padStart(2, '0')}</span>
                <p>{reason}</p>
              </div>
            ))}
          </div>
        </section>

        <div className="twoColumn">
          <section className="card" id="validation">
            <div className="sectionHead compact">
              <div>
                <span className="eyebrow">Data readiness</span>
                <h2>実データ準備状況</h2>
              </div>
              <span className="count">1 / 5</span>
            </div>
            <div className="readinessList">
              {readiness.map((item) => (
                <div className="readinessItem" key={item.label}>
                  <span className={`check ${item.tone}`} aria-hidden="true" />
                  <div><strong>{item.label}</strong><small>{item.detail}</small></div>
                  <span className={`state ${item.tone}`}>{item.state}</span>
                </div>
              ))}
            </div>
          </section>

          <aside className="card security" id="settings">
            <span className="shield" aria-hidden="true">✓</span>
            <span className="eyebrow">Security boundary</span>
            <h2>機密データはローカルに残します</h2>
            <p>JQUANTS_API_KEY、取得済み研究データ、実口座CSVはこのホスト版へ送信しません。</p>
            <ul>
              <li>本人限定の認証アクセス</li>
              <li>自動発注・Broker接続なし</li>
              <li>架空の本番データを表示しない</li>
            </ul>
            <div className="acknowledgement">
              <strong>{safety.acknowledged ? '安全境界を確認済み' : '安全境界の確認が必要です'}</strong>
              <small>
                {safety.updatedAt
                  ? `最終更新: ${new Intl.DateTimeFormat('ja-JP', { timeZone: 'Asia/Tokyo', dateStyle: 'short', timeStyle: 'short' }).format(new Date(safety.updatedAt))}`
                  : 'この確認状態だけを本人別に保存します。'}
              </small>
              {user ? (
                <form action={updateSafetyAcknowledgement}>
                  <input
                    type="hidden"
                    name="acknowledgement"
                    value={safety.acknowledged ? 'revoked' : 'confirmed'}
                  />
                  <button type="submit">
                    {safety.acknowledged ? '確認を取り消す' : '境界を理解して確認する'}
                  </button>
                </form>
              ) : (
                <a className="ackButton" href={chatGPTSignInPath('/')}>ログインして確認</a>
              )}
            </div>
          </aside>
        </div>
      </div>

      <nav className="bottomNav" aria-label="主要ナビゲーション">
        <a href="#home" className="active"><span>⌂</span>ホーム</a>
        <a href="#today"><span>◎</span>今日</a>
        <a href="#validation"><span>↗</span>検証</a>
        <a href="#validation"><span>▤</span>順位</a>
        <a href="#settings"><span>⚙</span>設定</a>
      </nav>
    </main>
  );
}
