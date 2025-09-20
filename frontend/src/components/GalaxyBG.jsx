import React, { useEffect, useRef } from "react";

export default function GalaxyBG() {
  const ref = useRef(null);

  useEffect(() => {
    const canvas = ref.current;
    const ctx = canvas.getContext("2d");
    const DPR = Math.min(window.devicePixelRatio || 1, 2);

    let stars = [];
    const STAR_COUNT = 220; 
    const SPEED = 0.02;

    function resize() {
      const { innerWidth, innerHeight } = window;
      canvas.width = innerWidth * DPR;
      canvas.height = innerHeight * DPR;
      canvas.style.width = innerWidth + "px";
      canvas.style.height = innerHeight + "px";
      initStars();
    }

    function initStars() {
      stars = Array.from({ length: STAR_COUNT }, () => ({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        r: (Math.random() * 1.4 + 0.4) * DPR,
        tw: Math.random() * Math.PI * 2,
        v: Math.random() * SPEED + SPEED * 0.3
      }));
    }

    function tick() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // soft vignette
      const grad = ctx.createRadialGradient(
        canvas.width * 0.5, canvas.height * 0.55, 0,
        canvas.width * 0.5, canvas.height * 0.55, Math.max(canvas.width, canvas.height) * 0.6
      );
      grad.addColorStop(0, "rgba(20, 24, 54, 0.15)");
      grad.addColorStop(1, "rgba(5, 7, 14, 0.65)");
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // stars
      for (const s of stars) {
        s.tw += s.v;
        const a = 0.6 + Math.sin(s.tw) * 0.35; 
        ctx.globalAlpha = a;
        ctx.beginPath();
        ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
        ctx.fillStyle = "#ffffff";
        ctx.fill();

        // subtle drift
        s.x += s.v * 0.6;
        if (s.x > canvas.width + 2) s.x = -2;
      }
      ctx.globalAlpha = 1;
      requestAnimationFrame(tick);
    }

    resize();
    window.addEventListener("resize", resize);
    const id = requestAnimationFrame(tick);
    return () => {
      window.removeEventListener("resize", resize);
      cancelAnimationFrame(id);
    };
  }, []);

  return (
    <canvas
      ref={ref}
      style={{ position: "fixed", inset: 0, zIndex: -3 }}
      aria-hidden="true"
    />
  );
}
