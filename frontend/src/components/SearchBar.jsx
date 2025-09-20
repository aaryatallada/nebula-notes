import React from "react";

export default function SearchBar({ q, setQ, onSearch }) {
  return (
    <div className="glass max-w-6xl mx-auto mt-6 px-4 py-3 rounded-xl shadow-glow">
      <div className="flex gap-3">
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Search the nebula… try 'vector databases' or 'protein dinner'"
          className="input-dark flex-1"
        />
        <button onClick={onSearch} className="btn">
          Semantic Search
        </button>
      </div>
    </div>
  );
}
