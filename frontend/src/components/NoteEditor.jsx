import React, { useState, useEffect } from "react";
import { marked } from "marked";

export default function NoteEditor({ selected, onSave }) {
  const [title, setTitle] = useState("");
  const [content, setContent] = useState("");

  useEffect(() => {
    setTitle(selected?.title || "");
    setContent(selected?.content || "");
  }, [selected]);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      <div className="glass rounded-xl p-4">
        <input
          className="input-dark mb-3"
          placeholder="Title"
          value={title}
          onChange={(e) => setTitle(e.target.value)}
        />
        <textarea
          className="textarea-dark min-h-[260px]"
          placeholder="Markdown content…"
          value={content}
          onChange={(e) => setContent(e.target.value)}
        />
        <button onClick={() => onSave({ title, content })} className="btn mt-3">
          {selected ? "Update Note" : "Create Note"}
        </button>
      </div>

      <div className="glass rounded-xl p-4 overflow-auto">
        <div className="text-xs text-gray-400 mb-2">Preview</div>
        <div
          className="prose prose-invert max-w-none"
          dangerouslySetInnerHTML={{ __html: marked.parse(content || "") }}
        />
      </div>
    </div>
  );
}
