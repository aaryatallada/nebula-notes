import React from "react";

export default function NoteList({ notes, onSelect, onDelete, selectedId }) {
  return (
    <div className="glass rounded-xl p-3">
      <div className="text-sm text-gray-400 mb-2">Notes ({notes.length})</div>
      <ul className="space-y-2">
        {notes.map((n) => (
          <li
            key={n.id}
            className={`p-3 rounded-lg border transition
              ${selectedId === n.id ? "border-purple-400 bg-white/10" : "border-white/10 hover:bg-white/5"}`}
          >
            <div className="flex items-start justify-between gap-3">
              {/* IMPORTANT: min-w-0 allows truncation inside flex */}
              <button onClick={() => onSelect(n)} className="text-left flex-1 min-w-0">
                {/* single-line title, ellipsis */}
                <div className="font-medium text-gray-100 truncate break-words">
                  {n.title}
                </div>
                {/* two-line clamp for preview; won’t push the delete button */}
                <div className="text-xs text-gray-400 line-clamp-2 break-words">
                  {n.content}
                </div>
              </button>

              {/* prevent shrinking so it never gets pushed offscreen */}
              <button
                onClick={() => onDelete(n.id)}
                className="shrink-0 text-red-300 hover:text-red-200 text-sm"
              >
                Delete
              </button>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}
