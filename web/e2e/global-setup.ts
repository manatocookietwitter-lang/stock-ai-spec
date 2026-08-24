import { spawn, spawnSync } from "node:child_process";
import { join } from "node:path";
import process from "node:process";

export default async function globalSetup(): Promise<() => void> {
  const python = join(process.cwd(), "..", ".venv", "Scripts", "python.exe");
  const server = spawn(python, ["e2e/server.py"], {
    cwd: process.cwd(),
    stdio: "inherit",
    windowsHide: true,
  });
  const deadline = Date.now() + 30_000;
  while (Date.now() < deadline) {
    if (server.exitCode !== null) {
      throw new Error(`Goal 5 E2E server exited with ${server.exitCode}`);
    }
    try {
      const response = await fetch("http://127.0.0.1:8766/api/v1/health");
      if (response.ok) break;
    } catch {
      // The local server is still starting.
    }
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  try {
    const response = await fetch("http://127.0.0.1:8766/api/v1/health");
    if (!response.ok) throw new Error("health response is not successful");
  } catch (error) {
    server.kill();
    throw new Error("Goal 5 E2E server did not become ready", { cause: error });
  }
  return () => {
    if (process.platform === "win32" && server.pid !== undefined) {
      spawnSync("taskkill", ["/PID", String(server.pid), "/T", "/F"], {
        stdio: "ignore",
        windowsHide: true,
      });
    } else {
      server.kill("SIGTERM");
    }
  };
}
