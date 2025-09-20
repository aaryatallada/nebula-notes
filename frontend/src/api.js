export const BASE = 'http://localhost:8000';

export async function listNotes() {
  const r = await fetch(`${BASE}/api/notes`);
  return r.json();
}

export async function createNote(payload) {
  const r = await fetch(`${BASE}/api/notes`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload)
  });
  return r.json();
}

export async function updateNote(id, payload) {
  const r = await fetch(`${BASE}/api/notes/${id}`, {
    method: 'PUT',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload)
  });
  return r.json();
}

export async function deleteNote(id) {
  return fetch(`${BASE}/api/notes/${id}`, { method: 'DELETE' });
}

export async function search(q) {
  const r = await fetch(`${BASE}/api/search?q=${encodeURIComponent(q)}`);
  return r.json();
}

export async function mapPoints() {
  const r = await fetch(`${BASE}/api/map`);
  return r.json();
}

export async function rebuildEmbeddings() {
  const r = await fetch(`${BASE}/api/embed/rebuild`, { method: 'POST' });
  return r.json();
}
