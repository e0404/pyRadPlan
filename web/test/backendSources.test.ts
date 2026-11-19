import { afterEach, describe, expect, it, vi } from "vitest";
import { backendKindFromUrl, backendSources, createBackend, trustedSource, type SourceTrust } from "../src/api";

const deployed: SourceTrust = {
  dev: false,
  origin: "https://plan.example.org",
  base: "https://plan.example.org/app/",
  scope: "https://plan.example.org/app/",
};

const pages: SourceTrust = {
  dev: false,
  origin: "https://e0404.github.io",
  base: "https://e0404.github.io/pyRadPlan/develop/index.html",
  scope: "https://e0404.github.io/pyRadPlan/develop/",
};

describe("backend locations from URL parameters", () => {
  afterEach(() => vi.restoreAllMocks());

  it("honours every value on the development server", () => {
    const dev = { ...deployed, dev: true };
    expect(trustedSource("https://elsewhere.example.com/pyodide/", dev)).toBe("https://elsewhere.example.com/pyodide/");
  });

  it("honours only URLs below the app's directory on a deployment", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    expect(trustedSource("https://plan.example.org/app/mirror/", deployed)).toBe("https://plan.example.org/app/mirror/");
    expect(trustedSource("/app/mirror/", deployed)).toBe("/app/mirror/");
    expect(trustedSource("wheels/", deployed)).toBe("wheels/");
    expect(trustedSource("./", deployed)).toBe("./");
    expect(trustedSource("/mirror/", deployed)).toBeUndefined();
    expect(trustedSource("/application/", deployed)).toBeUndefined();
    expect(trustedSource("/app/../other/", deployed)).toBeUndefined();
    expect(trustedSource("../other/", deployed)).toBeUndefined();
    expect(trustedSource("/app/%2e%2e/other/", deployed)).toBeUndefined();
    expect(trustedSource("https://evil.example.com/pyodide/", deployed)).toBeUndefined();
    expect(trustedSource("//evil.example.com/pyrp/", deployed)).toBeUndefined();
    expect(trustedSource("http://plan.example.org/app/", deployed)).toBeUndefined();
    expect(trustedSource("javascript:alert(1)", deployed)).toBeUndefined();
    expect(trustedSource(null, deployed)).toBeUndefined();
    expect(trustedSource("", deployed)).toBeUndefined();
    expect(warn).toHaveBeenCalledTimes(9);
  });

  it("keeps a page of a shared origin (GitHub Pages) to its own channel", () => {
    vi.spyOn(console, "warn").mockImplementation(() => undefined);
    expect(trustedSource("pyrp-next/", pages)).toBe("pyrp-next/");
    expect(trustedSource("/pyRadPlan/develop/mirror/", pages)).toBe("/pyRadPlan/develop/mirror/");
    expect(trustedSource("/other-repo/", pages)).toBeUndefined();
    expect(trustedSource("https://e0404.github.io/other-repo/pyrp/", pages)).toBeUndefined();
    expect(trustedSource("/pyRadPlan/latest/pyrp/", pages)).toBeUndefined();
    expect(trustedSource("/pyRadPlan/", pages)).toBeUndefined();
  });

  it("allows the whole origin where the app is served from its root", () => {
    const root = { ...deployed, base: "https://plan.example.org/", scope: "https://plan.example.org/" };
    expect(trustedSource("/mirror/", root)).toBe("/mirror/");
    vi.spyOn(console, "warn").mockImplementation(() => undefined);
    expect(trustedSource("https://evil.example.com/", root)).toBeUndefined();
  });

  it("applies the check to pyodide, packages and pyrp; cdn names the official runtime", () => {
    vi.spyOn(console, "warn").mockImplementation(() => undefined);
    const foreign = "https://evil.example.com/x/";
    const search = `?pyodide=${foreign}&packages=${foreign}&pyrp=${foreign}`;
    expect(backendSources(search, deployed)).toEqual({ pyodide: undefined, packages: undefined, pyrp: undefined });
    expect(backendSources(search, { ...deployed, dev: true })).toEqual({ pyodide: foreign, packages: foreign, pyrp: foreign });
    expect(backendSources("?pyodide=cdn&pyrp=pyrp-next/", deployed)).toEqual({
      pyodide: "cdn",
      packages: undefined,
      pyrp: "pyrp-next/",
    });
  });
});

describe("mock backend selection", () => {
  afterEach(() => vi.restoreAllMocks());

  it("offers the mock only to builds that contain it", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    expect(backendKindFromUrl("?backend=mock", true)).toBe("mock");
    expect(backendKindFromUrl("?backend=MOCK", true)).toBe("mock");
    expect(backendKindFromUrl("?backend=mock", false)).toBe("pyodide");
    expect(warn).toHaveBeenCalledTimes(1);
    expect(backendKindFromUrl("", false)).toBe("pyodide");
    expect(backendKindFromUrl("?backend=pyodide", true)).toBe("pyodide");
  });

  it("builds the mock in development builds (the tests run in one)", async () => {
    const backend = await createBackend("mock", "?speed=0");
    expect(backend.kind).toBe("mock");
  });
});
