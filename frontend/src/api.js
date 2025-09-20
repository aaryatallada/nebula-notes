
const API_BASE =
  (typeof import.meta !== "undefined" && import.meta.env && import.meta.env.VITE_API_BASE) ||
  (typeof window !== "undefined" && window.__API_BASE__) ||
  "http://localhost:8000";

const USER_KEY = "nebulaUserId";

export function getUserId() {
  let id = null;
  try {
    id = localStorage.getItem(USER_KEY);
  } catch (_) {}
  return id || null;
}

export function setUserId(id) {
  try {
    localStorage.setItem(USER_KEY, id);
  } catch (_) {}
  return id;
}

export function ensureUserId() {
  let id = getUserId();
  if (!id) {
    const gen =
      (typeof crypto !== "undefined" && crypto.randomUUID && crypto.randomUUID()) ||
      ("u_" + Math.random().toString(36).slice(2) + Date.now().toString(36));
    id = setUserId(gen);
  }
  return id;
}

ensureUserId();

function userHeaders() {
  const uid = getUserId();
  return uid ? { "X-User": uid } : {};
}

async function request(path, { timeout = 15000, headers = {}, ...opts } = {}) {
  const controller = new AbortController();
  const t = setTimeout(() => controller.abort(), timeout);

  const res = await fetch(`${API_BASE}${path}`, {
    ...opts,
    headers: { ...headers, ...userHeaders() },
    signal: controller.signal,
  }).catch((err) => {
    clearTimeout(t);
    throw new Error(`Network error calling ${path}: ${err?.message || err}`);
  });

  clearTimeout(t);

  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try {
      const data = await res.json();
      msg = data?.detail || JSON.stringify(data);
    } catch {
      try {
        msg = await res.text();
      } catch {}
    }
    throw new Error(`API ${path} failed: ${msg}`);
  }

  const ct = res.headers.get("content-type") || "";
  return ct.includes("application/json") ? res.json() : res.text();
}

export async function listNotes() {
  return request(`/api/notes`);
}

export async function createNote(payload) {
  return request(`/api/notes`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function updateNote(id, payload) {
  return request(`/api/notes/${id}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function deleteNote(id) {
  return request(`/api/notes/${id}`, { method: "DELETE" });
}

// Search
export async function search(q, { k = 10, mode = "hybrid", alpha = undefined } = {}) {
  const params = new URLSearchParams({ q, k: String(k), mode });
  if (alpha !== undefined && alpha !== null) params.set("alpha", String(alpha));
  return request(`/api/search?${params.toString()}`);
}

// Idea map
export async function mapPoints() {
  return request(`/api/map`);
}

// Embeddings rebuild (idempotent)
export async function rebuildEmbeddings() {
  return request(`/api/embed/rebuild`, { method: "POST" });
}

// Health (optional)
export async function health() {
  return request(`/api/health`);
}

export const API = {
  BASE: API_BASE,
  getUserId,
  setUserId,
  ensureUserId,
  listNotes,
  createNote,
  updateNote,
  deleteNote,
  search,
  mapPoints,
  rebuildEmbeddings,
  health,
};
