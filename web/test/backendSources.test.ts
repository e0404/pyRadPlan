import { afterEach, describe, expect, it, vi } from "vitest";
import { backendSources, trustedSource, type SourceTrust } from "../src/api";

const deployed: SourceTrust = { dev: false, origin: "https://plan.example.org", base: "https://plan.example.org/app/" };

describe("backend locations from URL parameters", () => {
  afterEach(() => vi.restoreAllMocks());

  it("honours every value on the development server", () => {
    const dev = { ...deployed, dev: true };
    expect(trustedSource("https://elsewhere.example.com/pyodide/", dev)).toBe("https://elsewhere.example.com/pyodide/");
  });

  it("honours only the page's own origin on a deployment", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    expect(trustedSource("https://plan.example.org/mirror/", deployed)).toBe("https://plan.example.org/mirror/");
    expect(trustedSource("/mirror/", deployed)).toBe("/mirror/");
    expect(trustedSource("wheels/", deployed)).toBe("wheels/");
    expect(trustedSource("https://evil.example.com/pyodide/", deployed)).toBeUndefined();
    expect(trustedSource("//evil.example.com/pyrp/", deployed)).toBeUndefined();
    expect(trustedSource("http://plan.example.org/", deployed)).toBeUndefined();
    expect(trustedSource("javascript:alert(1)", deployed)).toBeUndefined();
    expect(trustedSource(null, deployed)).toBeUndefined();
    expect(trustedSource("", deployed)).toBeUndefined();
    expect(warn).toHaveBeenCalledTimes(4);
  });

  it("applies the check to pyodide, packages and pyrp; cdn names the official runtime", () => {
    vi.spyOn(console, "warn").mockImplementation(() => undefined);
    const foreign = "https://evil.example.com/x/";
    const search = `?pyodide=${foreign}&packages=${foreign}&pyrp=${foreign}`;
    expect(backendSources(search, deployed)).toEqual({ pyodide: undefined, packages: undefined, pyrp: undefined });
    expect(backendSources(search, { ...deployed, dev: true })).toEqual({ pyodide: foreign, packages: foreign, pyrp: foreign });
    expect(backendSources("?pyodide=cdn&pyrp=/pyrp-next/", deployed)).toEqual({
      pyodide: "cdn",
      packages: undefined,
      pyrp: "/pyrp-next/",
    });
  });
});
