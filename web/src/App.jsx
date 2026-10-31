import { useEffect, useState } from "react";
import Landing from "./Landing.jsx";
import Studio from "./Studio.jsx";

const currentPage = () => window.location.hash === "#studio" ? "studio" : "demo";

export default function App() {
  const [page, setPage] = useState(currentPage);

  useEffect(() => {
    const onHash = () => setPage(currentPage());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  useEffect(() => { window.scrollTo({ top: 0, behavior: "auto" }); }, [page]);

  return (
    <div className="min-h-screen bg-[#f8faf9]">
      {page === "studio" && <header className="border-b border-slate-200 bg-white px-4 py-4">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center gap-4">
          <a href="#demo" className="flex items-center gap-3 text-slate-900">
            <span className="grid h-9 w-9 place-items-center rounded-xl bg-slate-900 text-lg font-bold text-white">R</span>
            <span className="leading-tight"><span className="block text-base font-extrabold tracking-tight">RETINA</span><span className="block text-[11px] font-medium tracking-wide text-slate-500">Retail simulation studio</span></span>
          </a>
          <nav className="ml-auto flex items-center gap-2 text-sm font-semibold" aria-label="Main navigation">
            <a href="#demo" aria-current={page === "demo" ? "page" : undefined}
              className={`rounded-lg px-3 py-2 ${page === "demo" ? "bg-brand-50 text-brand-700" : "text-slate-600 hover:bg-slate-100"}`}>Demo</a>
            <a href="#studio" aria-current={page === "studio" ? "page" : undefined}
              className={`rounded-lg px-3 py-2 ${page === "studio" ? "bg-brand-50 text-brand-700" : "text-slate-600 hover:bg-slate-100"}`}>Data studio</a>
          </nav>
        </div>
      </header>}

      {page === "demo" ? <Landing /> : <Studio />}

      {page === "studio" && <footer className="border-t border-slate-200 bg-white px-4 py-6 text-center text-xs leading-5 text-slate-500">
        RETINA creates synthetic retail data. The landing demo is illustrative; data studio runs are labelled simulated. · Code MIT
      </footer>}
    </div>
  );
}
