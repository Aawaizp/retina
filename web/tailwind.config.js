/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      fontFamily: { sans: ["Inter", "system-ui", "sans-serif"] },
      colors: { brand: { 50: "#edf8f5", 100: "#d6eee6", 500: "#137c70", 600: "#0f675e", 700: "#0b514a" } },
    },
  },
  plugins: [],
};
