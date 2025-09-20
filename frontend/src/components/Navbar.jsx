import React from "react";

export default function Navbar({ onRebuild }) {
  return (
    <div className="w-full sticky top-0 z-20 glass shadow-glow">
      <div className="max-w-6xl mx-auto px-4 py-3 flex items-center justify-between">
        <div className="font-bold text-xl title-gradient">🌌 Nebula Notes</div>
        <button onClick={onRebuild} className="btn" title="Recompute all embeddings">
          Rebuild Embeddings
        </button>
      </div>
    </div>
  );
}
