import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { packageOfModule, readLicenseFiles, renderNotices, resolveEntry } from "../scripts/thirdPartyNotices";
import { ABOUT, NOTICE_KEY, PRIVACY, acknowledgeNotice, buildLabel, noticeAcknowledged, privacyParagraphs, textHash, useNotice } from "../src/components/help/notice";

describe("third-party notices", () => {
  it("maps module ids to their npm package", () => {
    expect(packageOfModule("/w/web/node_modules/react-dom/cjs/react-dom-client.production.js")).toEqual({ name: "react-dom", dir: "/w/web/node_modules/react-dom" });
    expect(packageOfModule("\0D:\\w\\node_modules\\@scope\\pkg\\index.js?commonjs-exports")).toEqual({ name: "@scope/pkg", dir: "D:/w/node_modules/@scope/pkg" });
    expect(packageOfModule("/w/node_modules/a/node_modules/b/x.js")?.name).toBe("b");
    expect(packageOfModule("/w/web/src/App.tsx")).toBeNull();
    expect(packageOfModule("\0vite/preload-helper.js")).toBeNull();
  });

  it("reads license files and falls back to the override directory", () => {
    const root = mkdtempSync(join(tmpdir(), "notices-"));
    const withLicense = join(root, "node_modules", "has-license");
    const without = join(root, "node_modules", "no-license");
    const overrides = join(root, "licenses");
    for (const dir of [withLicense, without, overrides]) mkdirSync(dir, { recursive: true });
    writeFileSync(join(withLicense, "LICENSE.md"), "MIT text\n");
    writeFileSync(join(withLicense, "package.json"), JSON.stringify({ version: "1.2.3", license: "MIT", repository: { url: "git+https://github.com/x/y.git" } }));
    writeFileSync(join(without, "package.json"), JSON.stringify({ version: "2.0.0", license: "MPL-2.0" }));
    expect(readLicenseFiles(withLicense)).toBe("MIT text");

    const entry = resolveEntry({ name: "has-license", packageDir: withLicense, usedIn: "bundle" }, overrides);
    expect(entry).toMatchObject({ version: "1.2.3", license: "MIT", homepage: "https://github.com/x/y", text: "MIT text" });
    expect(() => resolveEntry({ name: "no-license", packageDir: without, usedIn: "bundle" }, overrides)).toThrow(/No license text for no-license/);

    writeFileSync(join(overrides, "no-license.txt"), "MPL text");
    const fallback = resolveEntry({ name: "no-license", packageDir: without, usedIn: "bundle" }, overrides);
    expect(fallback).toMatchObject({ version: "2.0.0", license: "MPL-2.0", text: "MPL text" });

    const text = renderNotices([entry, fallback]);
    expect(text).toContain("  - has-license 1.2.3 (MIT)");
    expect(text).toContain("Contained in: bundle");
    expect(text).toContain("MPL text");
  });
});

describe("disclaimer and privacy notice", () => {
  it("uses the texts of pyRadPlan/app/about.json", () => {
    expect(ABOUT.disclaimer_title).toBe("NOT FOR CLINICAL USE");
    expect(ABOUT.disclaimer).toContain("not a medical device");
  });

  it("links the imprint and the privacy policy in the documentation", () => {
    expect(ABOUT.imprint_url).toMatch(/^https:\/\/docs\.pyradplan\.org\/.*\/legal\/imprint\.html$/);
    expect(ABOUT.privacy_url).toMatch(/^https:\/\/docs\.pyradplan\.org\/.*\/legal\/privacy\.html$/);
  });

  it("states where data and downloads go", () => {
    const text = privacyParagraphs("https://example.org").join(" ");
    expect(text).toContain("https://example.org");
    expect(text).toContain("cdn.jsdelivr.net");
    expect(text).toContain("no telemetry");
    expect(PRIVACY.join(" ")).not.toContain("https://example.org");
  });

  it("derives the storage key from the texts", () => {
    expect(textHash("a")).not.toBe(textHash("b"));
    expect(textHash("")).toBe("811c9dc5");
    expect(NOTICE_KEY).toMatch(/^pyradplan\.notice-acknowledged\.[0-9a-f]{8}$/);
  });

  it("acknowledges without storage and resolves the waiters", async () => {
    useNotice.setState({ acknowledged: false });
    const waiting = noticeAcknowledged();
    acknowledgeNotice();
    await expect(waiting).resolves.toBeUndefined();
    expect(useNotice.getState().acknowledged).toBe(true);
  });

  it("labels the build from the Vite env", () => {
    expect(buildLabel({})).toBeNull();
    expect(buildLabel({ VITE_PYRADPLAN_CHANNEL: "develop", VITE_PYRADPLAN_VERSION: "v0.9.0-3-gabc1234", VITE_PYRADPLAN_COMMIT: "abc1234" })).toBe(
      "develop channel · v0.9.0-3-gabc1234 · commit abc1234",
    );
  });
});
