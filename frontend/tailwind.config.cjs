module.exports = {
  darkMode: 'class',
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        space: {
          900: "#05070e",
          800: "#0b1020",
          700: "#0e1630",
          600: "#131a3a",
          accent: "#8b5cf6", // violet
          accent2: "#06b6d4" // cyan
        }
      },
      boxShadow: {
        glow: "0 0 30px rgba(139, 92, 246, 0.35), 0 0 60px rgba(6, 182, 212, 0.25)"
      },
      backdropBlur: {
        xs: "2px"
      }
    }
  },
  plugins: []
}
