'use server';

import { revalidatePath } from 'next/cache';
import { getChatGPTUser } from './chatgpt-auth';
import { setSafetyAcknowledgement } from '../db/operator-settings';

export async function updateSafetyAcknowledgement(formData: FormData) {
  const user = await getChatGPTUser();
  if (!user) {
    throw new Error('認証済みユーザーのみ更新できます。');
  }

  const requested = formData.get('acknowledgement');
  if (requested !== 'confirmed' && requested !== 'revoked') {
    throw new Error('不正な更新内容です。');
  }

  await setSafetyAcknowledgement(user.userId, requested === 'confirmed');
  revalidatePath('/');
}
