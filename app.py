import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="What's My Drink?", page_icon="🍸", layout="wide")

st.markdown("""
<style>
/* ---- Typography (safe) ---- */
.stApp {
  font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "SF Pro Display",
               "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
}
.stApp h1, .stApp h2, .stApp h3 {
  letter-spacing: -0.015em !important;
  font-weight: 650 !important;
}
[data-testid="stCaptionContainer"] { opacity: 0.8; }

/* ---- Sidebar dark ---- */
[data-testid="stSidebar"] > div:first-child { background: #111827 !important; }
[data-testid="stSidebar"] * { color: #E5E7EB !important; }
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span { color: #D1D5DB !important; }

/* ---- Sidebar inputs ---- */
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea {
  background: #0B1220 !important;
  color: #E5E7EB !important;
  border: 1px solid #334155 !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div {
  background: #0B1220 !important;
  color: #E5E7EB !important;
  border: 1px solid #E5E7EB !important;
  box-shadow: none !important;
}

/* ---- Sidebar button ---- */
[data-testid="stSidebar"] .stButton > button {
  background: #0B1220 !important;
  color: #E5E7EB !important;
  border: 1px solid #E5E7EB !important;
}
[data-testid="stSidebar"] .stButton > button:hover,
[data-testid="stSidebar"] .stButton > button:hover * {
  background: #E5E7EB !important;
  color: #111827 !important;
  border-color: #E5E7EB !important;
}

/* Remove the inner input-box look inside the select */
[data-testid="stSidebar"] [data-baseweb="select"] input {
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
  outline: none !important;
  caret-color: transparent !important;
  padding: 0 !important;
}

</style>
""", unsafe_allow_html=True)

st.title("What's My Drink? 🍸")
st.caption("Find drinks with customizable parameters + an optional AI-powered “Vibe Match”.")

# -----------------------------
# Reset mechanism
# -----------------------------
if "reset_counter" not in st.session_state:
    st.session_state.reset_counter = 0
if "feedback" not in st.session_state:
    st.session_state.feedback = []

def do_reset():
    st.session_state.reset_counter += 1
    st.session_state.feedback = []

# -----------------------------
# Data loading
# -----------------------------
DEFAULT_CSV = "drinks.csv"
st.sidebar.header("Data")

csv_path = st.sidebar.text_input("CSV filename in this folder", value=DEFAULT_CSV)
csv_file = Path(csv_path)

if not csv_file.exists():
    st.error(
        f"Couldn't find `{csv_path}` in this folder. Files here: "
        f"{[p.name for p in Path('.').iterdir()]}"
    )
    st.stop()

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # Robustness: add missing columns if needed
    needed_text_cols = [
        "id", "name", "description", "tags", "contains",
        "recipe", "zero_proof_swap", "glass", "base_spirit", "image_key"
    ]
    needed_num_cols = ["abv", "sweet", "bitter", "sour", "boozy", "fizzy"]

    for c in needed_text_cols:
        if c not in df.columns:
            df[c] = ""
    for c in needed_num_cols:
        if c not in df.columns:
            df[c] = 0

    # Lists
    df["tags_list"] = df["tags"].fillna("").apply(lambda x: [t.strip() for t in str(x).split("|") if t.strip()])
    df["contains_list"] = df["contains"].fillna("").apply(lambda x: [t.strip() for t in str(x).split("|") if t.strip()])

    # Numeric normalization
    for c in needed_num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # ID fallback
    df["id"] = df["id"].replace("", np.nan)
    df["id"] = df["id"].fillna(df["name"]).fillna("unknown")

    # Normalize base_spirit to lowercase
    df["base_spirit"] = df["base_spirit"].fillna("").astype(str).str.strip().str.lower()

    return df

df = load_data(csv_path)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Recommendation settings")
st.sidebar.button("Reset preferences", on_click=do_reset)

rc = st.session_state.reset_counter

mode = st.sidebar.radio(
    "Recommendation mode",
    ["Parameters (fast)", "Vibe Match (AI-enhanced)"],
    index=1,
    key=f"mode_{rc}",
)
show_compare = st.sidebar.checkbox("Compare Parameters vs Vibe Match", value=False, key=f"show_compare_{rc}")


# Mixed/Unmixed/Any control
drink_type_pref = st.sidebar.radio(
    "Drink type",
    ["Any", "Mixed (cocktails)", "Unmixed (beer/wine)"],
    index=0,
    key=f"drink_type_{rc}",
)

st.sidebar.subheader("Taste sliders")

# Sliders in two columns
cL, cR = st.sidebar.columns(2)
sweet_pref = cL.slider("Sweet", 0, 10, 5, key=f"sweet_{rc}")
bitter_pref = cR.slider("Bitter", 0, 10, 5, key=f"bitter_{rc}")
sour_pref = cL.slider("Sour", 0, 10, 5, key=f"sour_{rc}")
boozy_pref = cR.slider("Boozy", 0, 10, 5, key=f"boozy_{rc}")
fizzy_pref = cL.slider("Fizzy", 0, 10, 5, key=f"fizzy_{rc}")

base_spirit = st.sidebar.selectbox(
    "Preferred base",
    ["any", "gin", "vodka", "rum", "whiskey", "tequila", "wine", "beer", "liqueur", "none"],
    index=0,
    key=f"base_{rc}",
)

st.sidebar.subheader("Constraints")
max_abv = st.sidebar.slider("Max ABV (%)", 0, 30, 30, key=f"max_abv_{rc}")

# Constraint checkboxes
avoid_citrus = st.sidebar.checkbox("Avoid citrus", value=False, key=f"avoid_citrus_{rc}")
avoid_dairy  = st.sidebar.checkbox("Avoid dairy",  value=False, key=f"avoid_dairy_{rc}")
avoid_egg    = st.sidebar.checkbox("Avoid egg",    value=False, key=f"avoid_egg_{rc}")
avoid_gluten = st.sidebar.checkbox("Avoid gluten", value=False, key=f"avoid_gluten_{rc}")

st.sidebar.caption("Tip: Use Drink type to force beer/wine vs cocktails. Vibe description is optional and powers AI-matching.")

# -----------------------------
# Describe-what-you-want
# -----------------------------
st.markdown("## Tell me what you’re craving")
vibe_text = st.text_input(
    "Describe the drink you want in a few words",
    value="",
    placeholder="e.g., crisp, refreshing, not sweet; or bitter, spirit-forward, classic",
    help="Vibe Match uses embeddings to match your text to drink descriptions/tags. If blank/short, it falls back to the parameters.",
    key=f"vibe_{rc}",
)

preset_cols = st.columns(4)
presets = [
    ("🍋 Tart + crisp", "tart, crisp, citrusy, not sweet"),
    ("🌴 Tropical", "tropical, fruity, vacation vibes"),
    ("🥃 Spirit-forward", "strong, spirit-forward, bitter, classic"),
    ("✨ Light + fizzy", "light, refreshing, fizzy, easy to sip"),
]

def set_vibe(text: str):
    st.session_state[f"vibe_{st.session_state.reset_counter}"] = text

for i, ((label, text), col) in enumerate(zip(presets, preset_cols)):
    with col:
        st.button(label, key=f"preset_{rc}_{i}", on_click=set_vibe, args=(text,))

# -----------------------------
# Mixed vs Unmixed determination (FIRST)
# -----------------------------
UNMIXED_BASES = {"beer", "wine"}

def infer_drink_type() -> str:
    """
    Returns: 'any' | 'mixed' | 'unmixed'
    Priority:
      1) explicit sidebar Drink type
      2) base_spirit if user selected beer/wine
      3) vibe text keywords
      4) fallback any
    """
    if drink_type_pref.startswith("Mixed"):
        return "mixed"
    if drink_type_pref.startswith("Unmixed"):
        return "unmixed"

    # If user explicitly picked beer/wine base, treat as unmixed
    if base_spirit in UNMIXED_BASES:
        return "unmixed"

    # If vibe text strongly indicates beer/wine
    vt = (vibe_text or "").lower()
    unmixed_keywords = ["beer", "ipa", "lager", "pilsner", "stout", "porter", "wheat beer", "cider", "seltzer", "wine", "prosecco", "champagne", "cabernet", "sauvignon", "riesling", "rosé", "rose", "spritz"]
    mixed_keywords = ["cocktail", "martini", "margarita", "old fashioned", "negroni", "daiquiri", "mojito", "sour", "highball"]

    if any(k in vt for k in unmixed_keywords) and not any(k in vt for k in mixed_keywords):
        return "unmixed"
    if any(k in vt for k in mixed_keywords) and not any(k in vt for k in unmixed_keywords):
        return "mixed"

    return "any"

desired_type = infer_drink_type()

def is_unmixed_row(row) -> bool:
    return str(row["base_spirit"]).strip().lower() in UNMIXED_BASES

def filter_by_type(df_in: pd.DataFrame) -> pd.DataFrame:
    if desired_type == "mixed":
        return df_in[~df_in.apply(is_unmixed_row, axis=1)].copy()
    if desired_type == "unmixed":
        return df_in[df_in.apply(is_unmixed_row, axis=1)].copy()
    return df_in.copy()

st.caption(f"**Drink type:** {desired_type.upper()} (based on your selections)")

# -----------------------------
# Scoring: Parameters (after filtering)
# -----------------------------
def passes_constraints(row) -> bool:
    if float(row["abv"]) > max_abv:
        return False

    contains = set(row["contains_list"])
    if avoid_citrus and "citrus" in contains:
        return False
    if avoid_dairy and "dairy" in contains:
        return False
    if avoid_egg and "egg" in contains:
        return False
    if avoid_gluten and "gluten" in contains:
        return False

    # If user set base_spirit (and not "any"), respect it
    if base_spirit != "any":
        if str(row["base_spirit"]).strip().lower() != base_spirit:
            return False

    return True

def rules_score(row) -> float:
    taste = np.array([row["sweet"], row["bitter"], row["sour"], row["boozy"], row["fizzy"]], dtype=float)
    pref = np.array([sweet_pref, bitter_pref, sour_pref, boozy_pref, fizzy_pref], dtype=float)
    dist = np.linalg.norm(taste - pref)

    score = 100 - dist * 7

    # tiny nudge if drink type is forced
    if desired_type == "unmixed" and is_unmixed_row(row):
        score += 2
    if desired_type == "mixed" and (not is_unmixed_row(row)):
        score += 2

    return float(score)

def diversify_top(work: pd.DataFrame, n=3) -> pd.DataFrame:
    """
    Diversity by base_spirit to avoid 3 beers or 3 gins in a row.
    """
    picks = []
    used = set()
    for _, r in work.iterrows():
        s = str(r["base_spirit"]).strip().lower()
        if len(picks) < n and (s not in used or len(picks) >= 2):
            picks.append(r)
            used.add(s)
    if len(picks) < n:
        picks = [r for _, r in work.head(n).iterrows()]
    return pd.DataFrame(picks)

def get_top_rules(n=3) -> pd.DataFrame:
    pool = filter_by_type(df)
    work = pool[pool.apply(passes_constraints, axis=1)].copy()
    if work.empty:
        return work
    work["score_custom"] = work.apply(rules_score, axis=1)
    work = work.sort_values("score_custom", ascending=False)
    return diversify_top(work, n=n).copy()

# -----------------------------
# VIBE MATCH: embeddings + parameters (after filtering)
# -----------------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

def drink_profile_text(row) -> str:
    tags = " ".join(row["tags_list"])
    base = str(row["base_spirit"])
    desc = str(row.get("description", ""))
    return f"{row['name']}. Base: {base}. Tags: {tags}. {desc}"

@st.cache_data
def build_drink_embeddings_from_df(df_subset: pd.DataFrame):
    model = load_embedder()
    texts = df_subset.apply(drink_profile_text, axis=1).tolist()
    emb = model.encode(texts, normalize_embeddings=True)
    return emb

def cosine_sim(vec, mat):
    return mat @ vec

def get_top_hybrid(n=3) -> pd.DataFrame:
    pool = filter_by_type(df)
    work = pool[pool.apply(passes_constraints, axis=1)].copy()
    if work.empty:
        return work

    work["score_custom"] = work.apply(rules_score, axis=1)

    # Fallback when vibe text is empty/too short
    if not vibe_text or len(vibe_text.strip()) < 8:
        work = work.sort_values("score_custom", ascending=False)
        out = diversify_top(work, n=n).copy()
        out["score_vibes"] = out["score_custom"]
        out["sim"] = 0.0
        return out

    try:
        model = load_embedder()
        user_vec = model.encode([vibe_text.strip()], normalize_embeddings=True)[0]

        work = work.reset_index(drop=True)
        drink_emb = build_drink_embeddings_from_df(work)
        work["sim"] = cosine_sim(user_vec, drink_emb)

    except Exception as e:
        st.warning(f"Vibe Match mode unavailable (embedding error): {e}. Falling back to Parameters.")
        work = work.sort_values("score_custom", ascending=False)
        out = diversify_top(work, n=n).copy()
        out["score_vibes"] = out["score_custom"]
        out["sim"] = 0.0
        return out

    # Blend scores
    custom_scaled = (work["score_custom"] - work["score_custom"].min()) / (
        work["score_custom"].max() - work["score_custom"].min() + 1e-9
    )
    work["custom_scaled"] = custom_scaled
    work["score_vibes"] = 0.7 * work["sim"] + 0.3 * work["custom_scaled"]

    work = work.sort_values("score_vibes", ascending=False)
    return diversify_top(work, n=n).copy()

# -----------------------------
# Explainability helper
# -----------------------------
def top_tag_matches(r):
    matches = []
    if r["sweet"] >= 7 and sweet_pref >= 7: matches.append("sweet")
    if r["bitter"] >= 7 and bitter_pref >= 7: matches.append("bitter")
    if r["sour"] >= 7 and sour_pref >= 7: matches.append("sour")
    if r["fizzy"] >= 7 and fizzy_pref >= 7: matches.append("fizzy")
    if r["boozy"] >= 7 and boozy_pref >= 7: matches.append("boozy")
    if not matches:
        matches = list(r["tags_list"])[:3]
    return matches[:4]

# -----------------------------
# Render cards
# -----------------------------
def render_cards(subdf: pd.DataFrame, score_col: str):
    if subdf is None or subdf.empty:
        st.warning("No matches. Try loosening constraints (ABV, base, avoid tags) or switch Drink type to Any.")
        return

    for _, r in subdf.iterrows():
        drink_id = str(r.get("id", r.get("name", "unknown")))
        name = str(r.get("name", "Unknown Drink"))
        tags_list = r.get("tags_list", [])
        contains_list = r.get("contains_list", [])
        base = str(r.get("base_spirit", "unknown"))
        abv = float(r.get("abv", 0))
        desc = str(r.get("description", ""))
        recipe = str(r.get("recipe", ""))
        zp = str(r.get("zero_proof_swap", ""))

        with st.container(border=True):
            st.subheader(name)
            if desc:
                st.write(desc)

            st.write(f"**Type:** {'UNMIXED' if base in UNMIXED_BASES else 'MIXED'}")
            st.write(f"**Base:** {base}  •  **ABV:** {abv:.1f}%")
            st.write(f"**Tags:** {', '.join(tags_list[:6]) if tags_list else '—'}")
            st.write(f"**Contains:** {', '.join(contains_list) if contains_list else 'none'}")

            if "sim" in r and r.get("sim") is not None:
                try:
                    st.write(f"**Vibe match score:** {float(r.get('sim')):.3f}")
                except Exception:
                    pass

            score_val = r.get(score_col, None)
            if score_val is not None:
                try:
                    st.write(f"**Score:** {float(score_val):.3f}")
                except Exception:
                    st.write(f"**Score:** {score_val}")

            with st.expander("Explain my pick"):
                st.write(f"**Step 1 — Picked drink type:** {desired_type.upper()}")
                st.write("**Step 2 — Matched your preferences:**")
                st.write(
                    f"- Sweet: {sweet_pref}, Bitter: {bitter_pref}, Sour: {sour_pref}, "
                    f"Boozy: {boozy_pref}, Fizzy: {fizzy_pref}"
                )
                st.write("**Matched signals:**")
                st.write(f"- Matches: {', '.join(top_tag_matches(r))}")
                st.write("**Constraints applied:**")
                st.write(
                    f"- ABV ≤ {max_abv}%"
                    f"{' | avoid citrus' if avoid_citrus else ''}"
                    f"{' | avoid dairy' if avoid_dairy else ''}"
                    f"{' | avoid egg' if avoid_egg else ''}"
                    f"{' | avoid gluten' if avoid_gluten else ''}"
                )
                st.write("**Your vibe description (Vibe Match):**")
                st.write(vibe_text.strip() if vibe_text else "— (not provided)")
                if "sim" in r:
                    st.write("**Vibe match score:**")
                    st.write(float(r.get("sim", 0.0)))

            with st.expander("Recipe + zero-proof swap"):
                if recipe:
                    st.write("**Recipe:** " + recipe)
                if zp:
                    st.write("**Zero-proof swap:** " + zp)

            fb_key = f"{score_col}_{drink_id}_{rc}"
            c1, c2, c3 = st.columns([1, 1, 6])
            with c1:
                if st.button("👍", key=fb_key + "_up"):
                    st.session_state.feedback.append(
                        {"drink_id": drink_id, "drink_name": name, "vote": "up", "mode": score_col, "type": desired_type}
                    )
                    st.toast("Saved 👍", icon="✅")
            with c2:
                if st.button("👎", key=fb_key + "_down"):
                    st.session_state.feedback.append(
                        {"drink_id": drink_id, "drink_name": name, "vote": "down", "mode": score_col, "type": desired_type}
                    )
                    st.toast("Saved 👎", icon="✅")
            with c3:
                st.caption("Rate this recommendation (stored for this session).")

# -----------------------------
# Compute recommendations
# -----------------------------
rules_top = get_top_rules(3)
hybrid_top = get_top_hybrid(3)

# -----------------------------
# Display recommendations
# -----------------------------
if show_compare:
    left, right = st.columns(2)
    with left:
        st.markdown("## Parameters picks")
        render_cards(rules_top, "score_custom")
    with right:
        st.markdown("## Vibe Match picks")
        render_cards(hybrid_top, "score_vibes")

    if (rules_top is not None and not rules_top.empty) and (hybrid_top is not None and not hybrid_top.empty):
        st.divider()
        st.markdown("### Compare summary")
        overlap = len(set(rules_top["id"]) & set(hybrid_top["id"]))
        st.write(f"**Overlap in Top 3:** {overlap}/3")

        cmp = pd.merge(
            rules_top[["id", "name", "base_spirit", "score_custom"]],
            hybrid_top[["id", "score_vibes", "sim"]],
            on="id",
            how="outer"
        ).fillna("")
        st.dataframe(cmp, use_container_width=True)

else:
    st.markdown(f"## {mode} picks")
    if "Custom" in mode:
        render_cards(rules_top, "score_custom")
    else:
        render_cards(hybrid_top, "score_vibes")

# -----------------------------
# Feedback view + footer
# -----------------------------
st.divider()
st.markdown("### Session feedback")
fb = st.session_state.get("feedback", [])
if fb:
    fb_df = pd.DataFrame(fb)
    st.dataframe(fb_df, use_container_width=True)
    st.write("👍:", (fb_df["vote"] == "up").sum(), " | 👎:", (fb_df["vote"] == "down").sum())
else:
    st.write("No feedback yet—click 👍/👎 on a recommendation.")

st.caption(f"Loaded {len(df)} drinks from `{csv_path}`.")
