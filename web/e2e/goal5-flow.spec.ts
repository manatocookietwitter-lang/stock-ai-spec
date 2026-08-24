import { expect, test } from "@playwright/test";

test("Today review change, partial fill, apply, and next Home use the live local API", async ({ page }) => {
  await page.goto("/today");
  await expect(page.getByRole("heading", { name: "今日", exact: true })).toBeVisible();
  await page.getByRole("link", { name: "判断を確認" }).click();

  const buyRow = page.locator(".review-row").filter({ hasText: "BUY" }).first();
  await expect(buyRow).toBeVisible();
  const buyChoice = buyRow.getByRole("combobox");
  const choices = await buyChoice.locator("option").all();
  expect(choices.length).toBeGreaterThan(1);
  await buyChoice.selectOption({ index: 1 });
  await expect(page.getByText("判断後の見込み")).toBeVisible();
  await page.getByRole("button", { name: "判断を保存" }).click();

  await expect(page.getByRole("heading", { name: "判断を保存しました" })).toBeVisible();
  await page.getByRole("link", { name: "実行結果を記録" }).click();
  const execution = page.locator(".execution-card").filter({ hasText: "NTT" }).first();
  await expect(execution).toBeVisible();
  expect(Number(await execution.getByLabel("注文株数").inputValue())).toBe(500);
  await execution.getByLabel("実行状態").selectOption("partially_filled");
  await execution.getByLabel("約定株数").fill("100");
  await execution.getByRole("button", { name: "この結果を記録" }).click();
  await expect(page.getByText(/実行結果を記録しました/)).toBeVisible();

  await page.getByRole("button", { name: "記録済み約定を実保有へ反映" }).click();
  await expect(page.getByText("記録済み約定から次の実保有状態を追加しました。")).toBeVisible();
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "ホーム" })).toBeVisible();
  await expect(page.getByText("実保有", { exact: true })).toBeVisible();
  await expect(page.getByText("9432 ・ 合計 400株", { exact: true })).toBeVisible();
  await expect(page.getByText("実状態を正本にしています")).toBeVisible();
});
