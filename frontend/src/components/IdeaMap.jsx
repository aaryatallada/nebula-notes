import React, { useEffect, useRef, useState } from "react";

export default function IdeaMap({ points, onPick }) {
  const ref = useRef(null);
  const [hover, setHover] = useState(null); 

  useEffect(() => {
    const canvas = ref.current;
    const ctx = canvas.getContext("2d");
    const DPR = Math.min(window.devicePixelRatio || 1, 2);

    const resize = () => {
      const { clientWidth, clientHeight } = canvas.parentElement;
      canvas.width = clientWidth * DPR;
      canvas.height = clientHeight * DPR;
      canvas.style.width = clientWidth + "px";
      canvas.style.height = clientHeight + "px";
      draw();
    };

    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      if (!points.length) return;

      const xs = points.map(p => p.x), ys = points.map(p => p.y);
      const minX = Math.min(...xs), maxX = Math.max(...xs);
      const minY = Math.min(...ys), maxY = Math.max(...ys);
      const pad = 30 * DPR;

      for (const p of points) {
        const nx = (p.x - minX) / (maxX - minX + 1e-6);
        const ny = (p.y - minY) / (maxY - minY + 1e-6);
        const cx = pad + nx * (canvas.width - 2 * pad);
        const cy = pad + ny * (canvas.height - 2 * pad);

        ctx.beginPath();
        ctx.arc(cx, cy, 7 * DPR, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(139,92,246,0.18)";
        ctx.fill();
        ctx.beginPath();
        ctx.arc(cx, cy, 3.5 * DPR, 0, Math.PI * 2);
        ctx.fillStyle = "white";
        ctx.fill();

        ctx.font = `${10 * DPR}px ui-sans-serif, system-ui`;
        ctx.fillStyle = "rgba(226,232,240,0.9)";
        ctx.fillText(p.title.slice(0, 22), cx + 8 * DPR, cy - 6 * DPR);

        p._cx = cx; p._cy = cy;
      }
    };

    const pick = (x, y) => {
      const hit = points.find(p => {
        const dx = p._cx - x, dy = p._cy - y;
        return dx * dx + dy * dy < (8 * 8);
      });
      return hit || null;
    };

    const onClick = (e) => {
      const rect = canvas.getBoundingClientRect();
      const x = (e.clientX - rect.left) * (window.devicePixelRatio || 1);
      const y = (e.clientY - rect.top) * (window.devicePixelRatio || 1);
      const hit = pick(x, y);
      if (hit) onPick(hit.id);
    };

    const onMove = (e) => {
      const rect = canvas.getBoundingClientRect();
      const x = (e.clientX - rect.left) * (window.devicePixelRatio || 1);
      const y = (e.clientY - rect.top) * (window.devicePixelRatio || 1);
      const hit = pick(x, y);
      if (hit) {
        setHover({ x: e.clientX + 12, y: e.clientY + 8, title: hit.title });
        canvas.style.cursor = "pointer";
      } else {
        setHover(null);
        canvas.style.cursor = "default";
      }
    };

    resize();
    window.addEventListener("resize", resize);
    canvas.addEventListener("click", onClick);
    canvas.addEventListener("mousemove", onMove);
    return () => {
      window.removeEventListener("resize", resize);
      canvas.removeEventListener("click", onClick);
      canvas.removeEventListener("mousemove", onMove);
    };
  }, [points, onPick]);

  return (
    <div className="glass rounded-xl p-2">
      <div className="px-2 pt-1 text-xs text-gray-400">Idea Map · click a dot to open the note</div>
      <div className="h-[380px]">
        <canvas ref={ref} style={{ width: "100%", height: "100%" }} />
      </div>
      {hover && (
        <div
          className="pointer-events-none fixed z-30 px-2 py-1 rounded-md text-xs text-gray-900 bg-white/90 shadow"
          style={{ left: hover.x, top: hover.y }}
        >
          {hover.title}
        </div>
      )}
    </div>
  );
}
