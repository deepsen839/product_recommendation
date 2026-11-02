import streamlit as st
from app.recommender import OccasionRecommender

st.set_page_config(page_title="Occasionly | Gift finder", page_icon="🎁", layout="wide")
@st.cache_resource
def load_engine():
    return OccasionRecommender()
engine = load_engine()

st.markdown("""
<style>
.main {background: #faf9f6;}
.hero {padding: 1.5rem 1.8rem; border-radius: 18px; background: linear-gradient(120deg,#263b5a,#725b78); color:white; margin-bottom:1.2rem;}
.hero h1 {color:white; margin:0; font-size:2.15rem;}
.hero p {color:#f1edf5; margin:.45rem 0 0;}
.product-card {border:1px solid #e9e5df; border-radius:14px; padding:12px; background:white; min-height:160px;}
.score {display:inline-block; background:#eaf4ed; color:#21633b; padding:4px 9px; border-radius:20px; font-size:12px; font-weight:700;}
.muted {color:#6c6c6c; font-size:13px;}
</style>
<div class="hero"><h1>Find the perfect gift for every occasion</h1><p>Tell us what you’re celebrating. We’ll find thoughtful picks from the product collection.</p></div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("About this catalog")
    stats = engine.catalog_stats
    st.metric("Products supplied", stats["total"])
    st.metric("Eligible products", stats["eligible"])
    st.caption("Only ACTIVE, approved, in-stock products are shown. Draft and review-required listings are excluded.")
    st.divider()
    st.caption("Recommendation method: Hugging Face Sentence Transformers semantic embeddings + TF-IDF + keyword matching; Groq optionally expands free-form queries.")

st.subheader("What are you celebrating?")
examples = ["Birthday", "Wedding", "Anniversary", "Diwali", "Valentine's Day", "Office Party", "Housewarming", "Farewell"]
selected = st.selectbox("Try a popular occasion", ["Choose an occasion…"] + examples, label_visibility="collapsed")
with st.form("occasion_search", clear_on_submit=False):
    occasion = st.text_input("Enter an occasion", value="" if selected == "Choose an occasion…" else selected, placeholder="e.g. Birthday, Diwali, a romantic anniversary…")
    c1,c2 = st.columns([1,4])
    with c1: limit = st.select_slider("Number of results", options=[4,8,12,16,20], value=12)
    with c2: submitted = st.form_submit_button("✨ Find gifts", type="primary", use_container_width=True)

if submitted:
    if not occasion.strip():
        st.warning("Please enter an occasion to get recommendations.")
    else:
        results = engine.recommend(occasion, limit=limit)
        st.markdown(f"### Recommendations for **{occasion.strip()}**")
        st.caption(f"Showing {len(results)} ranked product(s) from {stats['eligible']} eligible listings.")
        if not results:
            st.info("No matching products found. Try a broader occasion such as celebration, gifting, wedding or festival.")
        else:
            cols = st.columns(3)
            for i, product in enumerate(results):
                with cols[i % 3]:
                    with st.container(border=True):
                        img = product.get("mainImage")
                        if img:
                            st.image(img, use_container_width=True)
                        else:
                            st.markdown("🎁")
                        st.markdown(f"**{product.get('name') or 'Untitled product'}**")
                        st.caption(f"{product['category']} · {product.get('brand') or 'Unbranded'}")
                        desc = product.get("shortDescription") or product.get("description") or "No description available."
                        st.write(desc[:180] + ("…" if len(desc) > 180 else ""))
                        try:
                            price = float(product.get("price"))
                            discount = float(product.get("discountPercentage") or 0)
                            st.markdown(f"**₹{price:,.0f}**" + (f"  ·  {discount:g}% off" if discount else ""))
                        except (TypeError, ValueError):
                            st.markdown("**Price not available**")
                        st.markdown(f"<span class='score'>Match {product['relevance_score']:.1f}%</span>", unsafe_allow_html=True)
                        with st.expander("Why this recommendation?"):
                            st.write(product["match_reason"])
                            st.caption(f"SKU: {product.get('sku') or 'Not provided'} · Stock: {product.get('stock', 'N/A')}")
                        if product.get("id"):
                            st.caption(f"Product ID: {product['id']}")
else:
    st.markdown("### Popular occasions")
    cols = st.columns(4)
    for col, label, emoji in zip(cols, examples[:4], ["🎂","💍","❤️","🪔"]):
        with col:
            if st.button(f"{emoji} {label}", use_container_width=True):
                st.session_state["quick_occasion"] = label
                st.rerun()
    if st.session_state.get("quick_occasion"):
        q = st.session_state.pop("quick_occasion")
        st.markdown(f"### Recommendations for **{q}**")
        results = engine.recommend(q, 12)
        if not results: st.info("No matching products found. Try another occasion.")
        else:
            cols = st.columns(3)
            for i,p in enumerate(results):
                with cols[i%3]:
                    with st.container(border=True):
                        if p.get("mainImage"): st.image(p["mainImage"], use_container_width=True)
                        st.markdown(f"**{p.get('name','Untitled')}**")
                        st.caption(f"{p['category']} · {p.get('brand','')}")
                        st.write((p.get("shortDescription") or p.get("description") or "")[:160])
                        st.markdown(f"**₹{p.get('price','N/A')}** · Match {p['relevance_score']}%")

st.divider()
st.caption("Occasionly · Product recommendations are ranked from catalog text, not sponsored placements. Category labels are inferred from product names and descriptions because the supplied JSON contains category IDs but no category descriptions.")
