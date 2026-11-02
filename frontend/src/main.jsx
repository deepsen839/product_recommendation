import React, { useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import { Search, Gift, Sparkles, Heart, Flower2, PartyPopper, LoaderCircle, ArrowUpRight, PackageCheck } from "lucide-react";
import "./styles.css";

const API_BASE = import.meta.env.VITE_API_BASE || "/api";
const occasions = [
  { label: "Birthday", icon: PartyPopper },
  { label: "Wedding", icon: Heart },
  { label: "Anniversary", icon: Heart },
  { label: "Diwali", icon: Sparkles },
  { label: "Valentine's Day", icon: Heart },
  { label: "Office Party", icon: Gift },
  { label: "Housewarming", icon: Flower2 },
  { label: "Farewell", icon: Gift }
];

function money(value) {
  const n = Number(value);
  return Number.isFinite(n) ? `₹${n.toLocaleString("en-IN", { maximumFractionDigits: 0 })}` : "Price unavailable";
}

function ProductCard({ product, index }) {
  const description = product.shortDescription || product.description || "A thoughtful pick for a special moment.";
  return (
    <article className="product-card">
      <div className="product-image-wrap">
        {product.mainImage
          ? <img className="product-image" src={product.mainImage} alt={product.name || "Product"} loading="lazy" onError={(e) => { e.currentTarget.style.display = "none"; }} />
          : <div className="image-placeholder"><Gift size={34} /></div>}
        <span className="rank">#{index + 1} match</span>
      </div>
      <div className="product-content">
        <div className="product-meta"><span>{product.category || "Gifts & Other"}</span><span>{product.brand || "Unbranded"}</span></div>
        <h3>{product.name || "Untitled product"}</h3>
        <p className="description">{description}</p>
        <div className="price-row">
          <strong>{money(product.price)}</strong>
          {Number(product.discountPercentage) > 0 && <span className="discount">{Number(product.discountPercentage)}% off</span>}
        </div>
        <div className="match-row"><span className="match-label">Relevance match</span><strong>{Number(product.relevance_score || 0).toFixed(1)}%</strong></div>
        <div className="progress"><span style={{ width: `${Math.max(0, Math.min(100, Number(product.relevance_score || 0)))}%` }} /></div>
        <details className="why">
          <summary>Why this recommendation?</summary>
          <p>{product.match_reason || "Matched against the occasion and product catalog details."}</p>
          <small>SKU: {product.sku || "Not provided"} · Stock: {product.stock ?? "N/A"}</small>
        </details>
      </div>
    </article>
  );
}

function App() {
  const [occasion, setOccasion] = useState("");
  const [active, setActive] = useState("");
  const [limit, setLimit] = useState(12);
  const [results, setResults] = useState([]);
  const [searchedFor, setSearchedFor] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [hasSearched, setHasSearched] = useState(false);

  const trimmed = occasion.trim();
  const canSearch = trimmed.length > 0 && !loading;

  async function search(value = occasion) {
    const query = value.trim();
    if (!query) {
      setError("Please enter an occasion to get recommendations.");
      return;
    }
    setLoading(true);
    setError("");
    setHasSearched(true);
    setSearchedFor(query);
    try {
      const response = await fetch(`${API_BASE}/recommendations`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ occasion: query, limit })
      });
      if (!response.ok) throw new Error(`Recommendation service returned ${response.status}.`);
      const data = await response.json();
      setResults(data.recommendations || []);
    } catch (err) {
      setResults([]);
      setError(`${err.message} Check that the API container is running.`);
    } finally {
      setLoading(false);
    }
  }

  function chooseOccasion(label) {
    setOccasion(label);
    setActive(label);
    search(label);
  }

  const resultLabel = useMemo(() => `${results.length} ${results.length === 1 ? "gift" : "gifts"} found`, [results.length]);

  return (
    <main>
      <nav className="topbar">
        <a className="brand" href="#"><span className="brand-mark"><Gift size={19} /></span> occasionly</a>
        <span className="top-note"><span className="live-dot" /> Thoughtful finds, for every moment</span>
      </nav>

      <section className="hero">
        <div className="hero-copy">
          <div className="eyebrow"><Sparkles size={14} /> THE ART OF GIFTING</div>
          <h1>Make every occasion<br /><em>feel extraordinary.</em></h1>
          <p>Tell us what you're celebrating. We'll find thoughtful gifts from our curated product collection.</p>
        </div>
        <div className="hero-art" aria-hidden="true">
          <div className="art-circle circle-one" /><div className="art-circle circle-two" />
          <div className="gift-illustration"><Gift size={104} strokeWidth={1.1} /></div>
          <span className="spark spark-a">✦</span><span className="spark spark-b">✧</span><span className="spark spark-c">✳</span>
        </div>
      </section>

      <section className="search-panel">
        <label htmlFor="occasion">WHAT ARE YOU CELEBRATING?</label>
        <div className="search-line">
          <div className="input-wrap"><Search size={19} /><input id="occasion" value={occasion} onChange={(e) => { setOccasion(e.target.value); setActive(""); }} onKeyDown={(e) => e.key === "Enter" && search()} placeholder="Try “Birthday”, “Diwali” or “Anniversary”" /></div>
          <select aria-label="Number of recommendations" value={limit} onChange={(e) => setLimit(Number(e.target.value))}>
            <option value={4}>4 gifts</option><option value={8}>8 gifts</option><option value={12}>12 gifts</option><option value={16}>16 gifts</option><option value={20}>20 gifts</option>
          </select>
          <button className="search-button" onClick={() => search()} disabled={!canSearch}>{loading ? <LoaderCircle className="spin" size={18} /> : <Search size={18} />} Find gifts</button>
        </div>
        <div className="popular-row"><span>POPULAR:</span>{occasions.map(({label}) => <button key={label} className={`chip ${active === label ? "selected" : ""}`} onClick={() => chooseOccasion(label)}>{label}</button>)}</div>
        {error && <p className="error-message">{error}</p>}
      </section>

      <section className="results-section">
        <div className="section-heading">
          <div><span className="eyebrow muted-eyebrow">CURATED FOR YOUR MOMENT</span><h2>{hasSearched ? `Gifts for ${searchedFor}` : "A little inspiration"}</h2></div>
          {hasSearched && <span className="result-count">{resultLabel}</span>}
        </div>
        {loading ? <div className="state-box"><LoaderCircle className="spin" size={28} /><p>Finding thoughtful matches…</p></div>
          : hasSearched && results.length > 0 ? <div className="product-grid">{results.map((p, i) => <ProductCard key={p.id || p.sku || i} product={p} index={i} />)}</div>
          : hasSearched && !error ? <div className="state-box"><PackageCheck size={32} /><h3>No matching products yet</h3><p>Try a broader occasion such as celebration, gifting, wedding or festival.</p></div>
          : !hasSearched ? <div className="empty-inspiration"><div className="empty-icon"><Flower2 size={25} /></div><div><h3>Your next perfect gift is a search away.</h3><p>Choose a popular occasion above or type your own to see ranked recommendations.</p></div><ArrowUpRight size={20} /></div> : null}
      </section>

      <footer><span>occasionly</span><span>Groq-assisted occasion understanding · Catalog-grounded recommendations · Not sponsored placements</span></footer>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
