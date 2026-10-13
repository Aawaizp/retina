import { lazy, Suspense } from "react";
import { NavLink, Route, Routes } from "react-router-dom";
import Home from "./pages/Home.jsx";

// Heavy pages load only when opened: the home page stays fast on a phone.
const Dashboards = lazy(() => import("./pages/Dashboards.jsx"));
const WhatIf = lazy(() => import("./pages/WhatIf.jsx"));

const nav = [
  ["/", "Home"],
  ["/dashboards", "Dashboards"],
  ["/what-if", "What-if"],
];

export default function App() {
  return (
    <div className="flex min-h-screen flex-col">
      <header className="sticky top-0 z-10 border-b border-slate-200 bg-white/80 backdrop-blur dark:border-slate-800 dark:bg-slate-950/80">
        <div className="mx-auto flex max-w-6xl items-center gap-4 px-4 py-3">
          <NavLink to="/" className="flex items-center gap-2 font-bold tracking-tight">
            <span className="grid h-7 w-7 place-items-center rounded-lg bg-brand-500 text-sm text-white">R</span>
            RETINA
          </NavLink>
          <nav className="ml-auto flex gap-1">
            {nav.map(([to, label]) => (
              <NavLink
                key={to}
                to={to}
                end
                className={({ isActive }) =>
                  `btn whitespace-nowrap px-2 sm:px-3 ${isActive ? "bg-brand-50 text-brand-700 dark:bg-brand-700/30 dark:text-brand-100" : "hover:bg-slate-100 dark:hover:bg-slate-800"}`
                }
              >
                {label}
              </NavLink>
            ))}
          </nav>
        </div>
      </header>
      <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-8">
        <Suspense fallback={<p className="muted">Loading...</p>}>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/dashboards" element={<Dashboards />} />
            <Route path="/what-if" element={<WhatIf />} />
            <Route path="*" element={<Home />} />
          </Routes>
        </Suspense>
      </main>
      <footer className="border-t border-slate-200 py-6 text-center text-xs muted dark:border-slate-800">
        Every number on this site is <span className="label">SIMULATED</span> - not real sales. Code MIT.
        Product data: Open Food Facts / Open Prices (ODbL). Instacart 2017: aggregate statistics only, non-commercial.
      </footer>
    </div>
  );
}
