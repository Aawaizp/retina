/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  darkMode: "class",                 // the 3D store page is always dark
  theme: {
    extend: {
      fontFamily: { sans: ["Inter", "system-ui", "sans-serif"] },
      colors: { brand: { 50: "#eef6ff", 100: "#d9eaff", 500: "#2f6fed", 600: "#2259c9", 700: "#1d48a3" } },
    },
  },
  plugins: [],
};
