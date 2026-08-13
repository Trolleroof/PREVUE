/**
 * Record the experiment page to an MP4.
 *
 * The page renders as a pure function of (scenario, t) — `window.demo.seek(t)` — so
 * frames are captured deterministically rather than by racing a real-time screen
 * recorder. Drives headless Chrome over the DevTools Protocol using Node's built-in
 * WebSocket: no npm install, no Playwright.
 *
 *   node waddle_wm/ui/record.mjs --page results/demo/experiment.html \
 *                                --out results/demo/experiment.mp4 [--fps 30] [--scenario grasp_miss]
 */
import { spawn, spawnSync } from "node:child_process";
import { mkdtempSync, writeFileSync, rmSync, mkdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

const argv = process.argv.slice(2);
const arg = (name, fallback) => {
  const i = argv.indexOf(`--${name}`);
  return i >= 0 ? argv[i + 1] : fallback;
};

const PAGE = resolve(arg("page", "results/demo/experiment.html"));
const OUT = resolve(arg("out", "results/demo/experiment.mp4"));
const FPS = Number(arg("fps", 30));
const SCENARIO = arg("scenario", "grasp_miss");
const WIDTH = Number(arg("width", 1600));
const HEIGHT = Number(arg("height", 900));
const CHROME = arg("chrome", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome");
const PORT = Number(arg("port", 9222 + (process.pid % 500)));

const sleep = ms => new Promise(r => setTimeout(r, ms));

/* ---- launch headless Chrome ---- */
const profile = mkdtempSync(join(tmpdir(), "waddle-rec-"));
const chrome = spawn(CHROME, [
  "--headless=new", `--remote-debugging-port=${PORT}`, `--user-data-dir=${profile}`,
  `--window-size=${WIDTH},${HEIGHT}`, "--hide-scrollbars", "--force-device-scale-factor=1",
  "--allow-file-access-from-files", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
  `file://${PAGE}`,
], { stdio: "ignore" });

async function target() {
  for (let attempt = 0; attempt < 60; attempt++) {
    try {
      const list = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json();
      const page = list.find(t => t.type === "page" && t.webSocketDebuggerUrl);
      if (page) return page.webSocketDebuggerUrl;
    } catch { /* chrome not up yet */ }
    await sleep(250);
  }
  throw new Error("Chrome did not expose a debugging target");
}

const socket = new WebSocket(await target());
await new Promise(r => socket.addEventListener("open", r, { once: true }));

let nextId = 0;
const pending = new Map();
socket.addEventListener("message", event => {
  const message = JSON.parse(event.data);
  const waiter = pending.get(message.id);
  if (!waiter) return;
  pending.delete(message.id);
  message.error ? waiter.reject(new Error(message.error.message)) : waiter.resolve(message.result);
});
const send = (method, params = {}) => new Promise((resolve, reject) => {
  const id = ++nextId;
  pending.set(id, { resolve, reject });
  socket.send(JSON.stringify({ id, method, params }));
});

const evaluate = async expression => {
  const { result, exceptionDetails } = await send("Runtime.evaluate", { expression, awaitPromise: true });
  if (exceptionDetails) throw new Error(exceptionDetails.exception?.description || "page error");
  return result.value;
};

/* ---- wait for the page, then step the clock ---- */
await send("Page.enable");
await send("Runtime.enable");
await send("Emulation.setDeviceMetricsOverride", { width: WIDTH, height: HEIGHT, deviceScaleFactor: 1, mobile: false });
for (let attempt = 0; attempt < 60; attempt++) {
  if (await evaluate("Boolean(window.demo)")) break;
  await sleep(200);
}
const duration = await evaluate("window.demo.duration");

/* --stills 5.6,12,26 writes single PNGs instead of a video — poster frames, and the
   quickest way to eyeball the layout at a given beat. */
const stills = arg("stills", null);
if (stills) {
  const into = resolve(arg("out", "results/demo"), stills.includes(".png") ? ".." : ".");
  mkdirSync(into, { recursive: true });
  for (const mark of stills.split(",").map(Number)) {
    await evaluate(`window.demo.seek(${mark}, ${JSON.stringify(SCENARIO)})`);
    const { data } = await send("Page.captureScreenshot", { format: "png" });
    const path = join(into, `still-${SCENARIO}-${String(mark).replace(".", "_")}s.png`);
    writeFileSync(path, Buffer.from(data, "base64"));
    process.stdout.write(`wrote ${path}\n`);
  }
  socket.close();
  chrome.kill();
  rmSync(profile, { recursive: true, force: true });
  process.exit(0);
}

const frames = Math.round(duration * FPS);
const dir = mkdtempSync(join(tmpdir(), "waddle-frames-"));
process.stdout.write(`recording ${frames} frames (${duration.toFixed(1)}s @ ${FPS}fps) at ${WIDTH}x${HEIGHT}\n`);

for (let frame = 0; frame < frames; frame++) {
  await evaluate(`window.demo.seek(${(frame / FPS).toFixed(4)}, ${JSON.stringify(SCENARIO)})`);
  const { data } = await send("Page.captureScreenshot", { format: "png", captureBeyondViewport: false });
  writeFileSync(join(dir, `f${String(frame).padStart(5, "0")}.png`), Buffer.from(data, "base64"));
  if (frame % (FPS * 5) === 0) process.stdout.write(`  ${(frame / FPS).toFixed(0)}s\n`);
}

socket.close();
chrome.kill();

/* ---- encode ---- */
mkdirSync(resolve(OUT, ".."), { recursive: true });
const ffmpeg = spawnSync("ffmpeg", [
  "-y", "-framerate", String(FPS), "-i", join(dir, "f%05d.png"),
  "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
  "-movflags", "+faststart", "-vf", "scale=1920:1080:flags=lanczos", OUT,
], { stdio: "inherit" });
rmSync(dir, { recursive: true, force: true });
rmSync(profile, { recursive: true, force: true });
if (ffmpeg.status !== 0) process.exit(ffmpeg.status ?? 1);
process.stdout.write(`wrote ${OUT}\n`);
