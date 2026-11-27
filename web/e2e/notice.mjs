// The first-visit disclaimer and privacy notice covers the app until it is acknowledged, and the
// acknowledgement is kept in localStorage. The e2e scripts acknowledge it once per browser
// context the way a user does, on a mock-backend page of the same origin.
export async function acknowledgeNotice(context, base) {
  const page = await context.newPage();
  await page.goto(`${base}?backend=mock&speed=0`);
  await page.getByTestId("notice-accept").click();
  await page.getByTestId("notice-dialog").waitFor({ state: "detached" });
  await page.close();
}
