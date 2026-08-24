import { useCallback, useEffect, useState } from "react";
import { apiGet } from "./api";

export function useApiData<T>(path: string): {
  data: T | null;
  error: string | null;
  loading: boolean;
  reload: () => void;
} {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [revision, setRevision] = useState(0);
  const reload = useCallback(() => setRevision((value) => value + 1), []);

  useEffect(() => {
    let active = true;
    setLoading(true);
    setError(null);
    apiGet<T>(path)
      .then((value) => {
        if (active) setData(value);
      })
      .catch((reason: unknown) => {
        if (active) {
          setData(null);
          setError(reason instanceof Error ? reason.message : "データを取得できませんでした");
        }
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [path, revision]);

  useEffect(() => {
    const refreshVisible = (): void => {
      if (document.visibilityState === "visible" && navigator.onLine) reload();
    };
    const invalidateOffline = (): void => {
      setData(null);
      setError("オフラインです。古い提案は表示しません。");
      setLoading(false);
    };
    document.addEventListener("visibilitychange", refreshVisible);
    window.addEventListener("online", refreshVisible);
    window.addEventListener("offline", invalidateOffline);
    const timer = window.setInterval(refreshVisible, 60_000);
    return () => {
      document.removeEventListener("visibilitychange", refreshVisible);
      window.removeEventListener("online", refreshVisible);
      window.removeEventListener("offline", invalidateOffline);
      window.clearInterval(timer);
    };
  }, [reload]);

  return { data, error, loading, reload };
}
