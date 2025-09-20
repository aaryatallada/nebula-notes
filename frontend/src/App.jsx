import React, { useEffect, useState } from "react";
import GalaxyBG from "./components/GalaxyBG";
import Navbar from "./components/Navbar";
import SearchBar from "./components/SearchBar";
import NoteEditor from "./components/NoteEditor";
import NoteList from "./components/NoteList";
import IdeaMap from "./components/IdeaMap";
import { listNotes, createNote, updateNote, deleteNote, search, mapPoints, rebuildEmbeddings } from "./api";

export default function App() {
  const [notes, setNotes] = useState([]);
  const [selected, setSelected] = useState(null);
  const [q, setQ] = useState("");
  const [results, setResults] = useState([]);
  const [points, setPoints] = useState([]);

  const load = async () => {
    setNotes(await listNotes());
    setPoints(await mapPoints());
  };

  useEffect(() => { load(); }, []);

  const onSave = async (payload) => {
    try {
      if (selected) await updateNote(selected.id, payload);
      else await createNote(payload);
      setSelected(null);
      await load();
    } catch (err) {
      alert(err.message);
      console.error(err);
    }
  };

  const onDeleteNote = async (id) => {
    await deleteNote(id);
    if (selected?.id === id) setSelected(null);
    await load();
  };

  const onSearch = async () => {
    const r = await search(q);
    setResults(r);
  };

  const onPick = (id) => {
    const n = notes.find((x) => x.id === id);
    if (n) setSelected(n);
  };

  const onRebuild = async () => {
    await rebuildEmbeddings();
    await load();
  };

  return (
    <div className="min-h-screen relative">
      <GalaxyBG />
      <Navbar onRebuild={onRebuild} />

      <main className="max-w-6xl mx-auto px-4 py-6 space-y-6">
        <SearchBar q={q} setQ={setQ} onSearch={onSearch} />

        <div className="glass rounded-xl p-4">
          <div className="text-sm text-gray-400 mb-2">Semantic results</div>
          {results.length === 0 ? (
            <div className="text-xs text-gray-500">
              No matches yet. Try a richer phrase (e.g., “chicken rice dinner”) or click <span className="underline">Rebuild Embeddings</span> above.
            </div>
          ) : (
            <ul className="space-y-2">
              {results.map((r, idx) => (
                <li key={idx} className="p-3 rounded-lg border border-white/10 hover:bg-white/5 transition flex justify-between">
                  <div>
                    <div className="font-medium text-gray-100">{r.note.title}</div>
                    <div className="text-xs text-gray-400 truncate">
                      {r.note.content.replace(/\n/g, " ").slice(0, 160)}
                    </div>
                  </div>
                  <div className="text-xs text-gray-400">score: {(r.score * 100).toFixed(0)}%</div>
                </li>
              ))}
            </ul>
          )}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <div className="lg:col-span-1">
            <NoteList
              notes={notes}
              selectedId={selected?.id}
              onSelect={setSelected}
              onDelete={onDeleteNote}
            />
          </div>
          <div className="lg:col-span-2 space-y-4">
            <NoteEditor selected={selected} onSave={onSave} />
            <IdeaMap points={points} onPick={onPick} />
          </div>
        </div>
      </main>
    </div>
  );
}
