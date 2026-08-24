const MANUAL_HEADERS = {
  "Content-Type": "application/json",
  "X-Stock-AI-Intent": "manual-record",
};

export class ApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function parse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let detail = `API error ${response.status}`;
    try {
      const payload = (await response.json()) as { detail?: string };
      detail = payload.detail || detail;
    } catch {
      // The stable status is still useful if a proxy returned a non-JSON page.
    }
    throw new ApiError(response.status, detail);
  }
  return (await response.json()) as T;
}

export async function apiGet<T>(path: string): Promise<T> {
  return parse<T>(
    await fetch(path, {
      headers: { Accept: "application/json" },
      cache: "no-store",
    }),
  );
}

export async function apiPost<T>(path: string, body: unknown): Promise<T> {
  return parse<T>(
    await fetch(path, {
      method: "POST",
      headers: MANUAL_HEADERS,
      body: JSON.stringify(body),
      cache: "no-store",
    }),
  );
}
