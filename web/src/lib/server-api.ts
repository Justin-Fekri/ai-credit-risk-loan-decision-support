import type { Application, ModelResponse, QueueResponse } from "./api";

const API = process.env.API_URL || "http://127.0.0.1:45232";

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${API}${path}`, { cache: "no-store" });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `API ${path} failed (${res.status})`);
  }
  return res.json() as Promise<T>;
}

export const serverApi = {
  queue: () => getJson<QueueResponse>("/queue"),
  application: (id: string) => getJson<Application>(`/applications/${id}`),
  model: () => getJson<ModelResponse>("/model"),
};
