import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  metadataBase: new URL('https://stock-ai-decision-support.manato0618.chatgpt.site'),
  title: '株AI | 日本株意思決定支援',
  description: '実注文を行わない、日本株の意思決定支援アプリ。',
  openGraph: {
    title: '株AI | 日本株意思決定支援',
    description: '実注文を行わない、日本株の意思決定支援アプリ。',
    locale: 'ja_JP',
    type: 'website',
    images: [
      {
        url: 'https://stock-ai-decision-support.manato0618.chatgpt.site/og.png',
        width: 1536,
        height: 1024,
        alt: '株AI — 実注文を行わない、日本株の意思決定支援',
      },
    ],
  },
  twitter: {
    card: 'summary_large_image',
    title: '株AI | 日本株意思決定支援',
    description: '実注文を行わない、日本株の意思決定支援アプリ。',
    images: ['https://stock-ai-decision-support.manato0618.chatgpt.site/og.png'],
  },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="ja">
      <body>{children}</body>
    </html>
  );
}
