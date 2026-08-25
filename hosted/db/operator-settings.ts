import { env } from 'cloudflare:workers';
import { eq } from 'drizzle-orm';
import { getDb } from './index';
import { operatorSettings } from './schema';

export type SafetyAcknowledgement = {
  acknowledged: boolean;
  updatedAt: string | null;
};

async function ensureSchema() {
  await env.DB.prepare(
    `CREATE TABLE IF NOT EXISTS operator_settings (
      user_id TEXT PRIMARY KEY NOT NULL,
      safety_acknowledged INTEGER NOT NULL DEFAULT 0,
      updated_at TEXT NOT NULL
    )`,
  ).run();
}

export async function getSafetyAcknowledgement(
  userId: string,
): Promise<SafetyAcknowledgement> {
  await ensureSchema();
  const row = await getDb().query.operatorSettings.findFirst({
    where: eq(operatorSettings.userId, userId),
  });
  return {
    acknowledged: row?.safetyAcknowledged ?? false,
    updatedAt: row?.updatedAt ?? null,
  };
}

export async function setSafetyAcknowledgement(
  userId: string,
  acknowledged: boolean,
) {
  await ensureSchema();
  const updatedAt = new Date().toISOString();
  await getDb()
    .insert(operatorSettings)
    .values({ userId, safetyAcknowledged: acknowledged, updatedAt })
    .onConflictDoUpdate({
      target: operatorSettings.userId,
      set: { safetyAcknowledged: acknowledged, updatedAt },
    });
}
