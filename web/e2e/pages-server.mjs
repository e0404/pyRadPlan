// Static server that behaves like GitHub Pages, for testing a static deployment locally:
// no COOP/COEP headers, `Access-Control-Allow-Origin: *`, `Cache-Control: max-age=600`, ETag,
// the content types GitHub Pages sends (.wasm application/wasm, .mjs text/javascript,
// .whl application/octet-stream), a 404 page, and a mount prefix like /<repo>/.
// Usage: node e2e/pages-server.mjs <dir> [--prefix=/pyRadPlan/] [--port=8080]
import { createReadStream, existsSync, statSync } from "node:fs";
import { createServer } from "node:http";
import { createHash } from "node:crypto";
import { extname, join, normalize, resolve } from "node:path";

const args = process.argv.slice(2);
const flag = (name, fallback) => (args.find((a) => a.startsWith(`--${name}=`)) ?? `--${name}=${fallback}`).split("=")[1];
const root = resolve(args.find((a) => !a.startsWith("--")) ?? "dist");
const prefix = flag("prefix", "/");
const port = Number(flag("port", "8080"));

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "application/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".wasm": "application/wasm",
  ".svg": "image/svg+xml",
  ".zip": "application/zip",
  ".whl": "application/octet-stream",
};

function notFound(res) {
  const body = Buffer.from("<h1>404</h1>");
  res.writeHead(404, { "Content-Type": "text/html; charset=utf-8", "Content-Length": body.length });
  res.end(body);
}

createServer((req, res) => {
  const url = decodeURIComponent((req.url ?? "/").split("?")[0]);
  if (!url.startsWith(prefix)) return notFound(res);
  let file = join(root, normalize(url.slice(prefix.length)));
  if (!file.startsWith(root)) return notFound(res);
  if (existsSync(file) && statSync(file).isDirectory()) file = join(file, "index.html");
  if (!existsSync(file)) return notFound(res);
  const st = statSync(file);
  const etag = `"${createHash("md5").update(`${st.mtimeMs}:${st.size}`).digest("hex").slice(0, 12)}"`;
  const headers = {
    "Content-Type": MIME[extname(file)] ?? "application/octet-stream",
    "Cache-Control": "max-age=600",
    ETag: etag,
    "Last-Modified": st.mtime.toUTCString(),
    "Access-Control-Allow-Origin": "*",
  };
  if (req.headers["if-none-match"] === etag) {
    res.writeHead(304, headers);
    return res.end();
  }
  res.writeHead(200, { ...headers, "Content-Length": st.size });
  createReadStream(file).pipe(res);
}).listen(port, () => console.log(`serving ${root} at http://localhost:${port}${prefix}`));
