"""The Kirk Files — Dashboard (Fase E).

Principio de diseño: el dashboard NO opina. Presenta la información organizada y
deja que el usuario saque sus propias conclusiones (inspirado en JMail.world).

Stack: Streamlit puro + Plotly para el timeline. Todas las queries a SQLite pasan
por ``@st.cache_data`` (no se reconecta en cada interacción).

Correr con:
    streamlit run ui/app.py

Vistas:
    📅 Timeline       — los 104 episodios en el tiempo, coloreados por relevancia
    📺 Episodios      — claims de un episodio, agrupados por tipo  (paso 2)
    ⚡ Contradicciones — tabla filtrable con el tipo de cambio narrativo
    ✅ Verificaciones  — claims verificados contra fuentes externas  (paso 2)
    👤 Personas        — quién aparece, dónde y en cuántas contradicciones (paso 2)
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------- #
# Configuración
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "db" / "kirkfiles.db"

# Paleta — tema oscuro, rojo oscuro del proyecto como color primario.
RED = "#A4161A"        # primario (high · silent · contradicted)
AMBER = "#E0A106"      # secundario (partial · acknowledged · ambiguous)
GREEN = "#2EA043"      # evidence_based · supported
GRAY = "#9AA0A6"       # low
FAINT = "#3A3F47"      # none
INK = "#E6EDF3"
PANEL = "#161B22"

RELEVANCE_COLORS = {"high": RED, "partial": AMBER, "low": GRAY, "none": FAINT}
RELEVANCE_ORDER = ["high", "partial", "low", "none"]

CHANGE_COLORS = {"silent": RED, "evidence_based": GREEN, "acknowledged": AMBER}
CHANGE_LABEL_ES = {"silent": "silencioso", "evidence_based": "con evidencia",
                   "acknowledged": "reconocido"}

VERDICT_COLORS = {"supported": GREEN, "contradicted": RED,
                  "ambiguous": AMBER, "unverifiable": GRAY}

CONTRA_TYPE_LABEL = {"direct": "directa", "evolution": "evolución",
                     "abandoned": "abandonada", "reinforced": "reforzada"}

st.set_page_config(
    page_title="The Kirk Files",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------- #
# Capa de datos — todo cacheado, una conexión read-only por query
# --------------------------------------------------------------------------- #
def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _read_sql(query: str) -> pd.DataFrame:
    conn = _connect()
    try:
        return pd.read_sql_query(query, conn)
    finally:
        conn.close()


@st.cache_data(show_spinner=False)
def load_episodes() -> pd.DataFrame:
    df = _read_sql(
        """
        SELECT id, episode_number, title, published_at, word_count,
               relevance_label, relevance_score, relevance_summary, charlie_topics
        FROM episodes
        ORDER BY published_at
        """
    )
    df["published_at"] = pd.to_datetime(df["published_at"])
    df["relevance_label"] = df["relevance_label"].fillna("none")
    return df


@st.cache_data(show_spinner=False)
def load_claims() -> pd.DataFrame:
    return _read_sql(
        """
        SELECT c.id, c.episode_id, c.claim_text, c.claim_type, c.speaker_confidence,
               c.evidence_provided, c.persons_mentioned, c.dates_mentioned,
               c.quote_verbatim, e.episode_number, e.published_at, e.title AS episode_title
        FROM claims c
        JOIN episodes e ON e.id = c.episode_id
        ORDER BY c.id
        """
    )


@st.cache_data(show_spinner=False)
def load_contradictions() -> pd.DataFrame:
    return _read_sql(
        """
        SELECT
            x.id, x.contradiction_type, x.severity, x.explanation,
            x.confidence_score, x.data_artifact, x.change_type,
            a.id AS a_id, a.claim_text AS a_text, a.quote_verbatim AS a_quote,
            a.claim_type AS a_type, a.persons_mentioned AS a_persons,
            ea.episode_number AS a_ep, ea.published_at AS a_date, ea.title AS a_title,
            b.id AS b_id, b.claim_text AS b_text, b.quote_verbatim AS b_quote,
            b.claim_type AS b_type, b.persons_mentioned AS b_persons,
            eb.episode_number AS b_ep, eb.published_at AS b_date, eb.title AS b_title
        FROM contradictions x
        JOIN claims a   ON a.id = x.claim_a_id
        JOIN episodes ea ON ea.id = a.episode_id
        JOIN claims b   ON b.id = x.claim_b_id
        JOIN episodes eb ON eb.id = b.episode_id
        ORDER BY x.id
        """
    )


@st.cache_data(show_spinner=False)
def load_verifications() -> pd.DataFrame:
    return _read_sql(
        """
        SELECT v.id, v.claim_id, v.verdict, v.sources_supporting, v.sources_contradicting,
               v.primary_documents, v.llm_reasoning, v.confidence,
               c.claim_text, c.claim_type, e.episode_number, e.published_at, e.title AS episode_title
        FROM verifications v
        JOIN claims c   ON c.id = v.claim_id
        JOIN episodes e ON e.id = c.episode_id
        ORDER BY v.id
        """
    )


@st.cache_data(show_spinner=False)
def corpus_counts() -> dict:
    eps = load_episodes()
    return {
        "episodes": len(eps),
        "claims": len(load_claims()),
        "contradictions": len(load_contradictions()),
        "verifications": len(load_verifications()),
    }


@st.cache_data(show_spinner=False)
def verdict_by_claim() -> dict:
    """Mapa claim_id → (verdict, confidence) para el badge de la Vista Episodio."""
    v = load_verifications()
    return {int(r.claim_id): (r.verdict, r.confidence) for r in v.itertuples()}


# --------------------------------------------------------------------------- #
# Utilidades de presentación
# --------------------------------------------------------------------------- #
def parse_json_list(raw) -> list:
    if not raw or (isinstance(raw, float) and pd.isna(raw)):
        return []
    try:
        val = json.loads(raw)
        return val if isinstance(val, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def ep_label(num, date=None) -> str:
    if num is None or (isinstance(num, float) and pd.isna(num)):
        return "Ep s/n"
    return f"Ep {int(num)}"


def badge(text: str, color: str) -> str:
    return (
        f"<span style='background:{color}22;color:{color};border:1px solid {color}66;"
        f"padding:2px 9px;border-radius:11px;font-size:0.78rem;font-weight:600;"
        f"white-space:nowrap'>{text}</span>"
    )


def go_to(view: str, **state) -> None:
    """Cambia de vista (y opcionalmente fija filtros) y refresca."""
    st.session_state.view = view
    for k, val in state.items():
        st.session_state[k] = val
    st.rerun()


# --------------------------------------------------------------------------- #
# Sidebar — navegación + resumen del corpus
# --------------------------------------------------------------------------- #
NAV = [
    ("📅 Timeline", "timeline"),
    ("📺 Episodios", "episode"),
    ("⚡ Contradicciones", "contradictions"),
    ("✅ Verificaciones", "verifications"),
    ("👤 Personas", "people"),
]


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### 🔍 The Kirk Files")
        st.caption("Consistencia narrativa · Candace Owens × Charlie Kirk")
        st.divider()

        current = st.session_state.view
        for label, key in NAV:
            if st.button(
                label,
                key=f"nav_{key}",
                use_container_width=True,
                type="primary" if current == key else "secondary",
            ):
                if current != key:
                    go_to(key)

        st.divider()
        c = corpus_counts()
        st.markdown("#### 📊 Resumen del corpus")
        st.markdown(
            f"""
<div style='line-height:1.9;font-size:0.95rem'>
<b>{c['episodes']}</b> episodios<br>
<b>{c['claims']:,}</b> claims<br>
<b>{c['contradictions']}</b> contradicciones<br>
<b>{c['verifications']}</b> verificadas
</div>
""",
            unsafe_allow_html=True,
        )
        st.divider()
        st.caption("Los datos se presentan sin juicio editorial.\nLas conclusiones son del lector.")


# --------------------------------------------------------------------------- #
# Vista: Timeline
# --------------------------------------------------------------------------- #
def view_timeline() -> None:
    st.title("📅 Timeline")
    st.caption(
        "Los 104 episodios desde septiembre 2025. Color = relevancia sobre Charlie Kirk. "
        "Pasa el cursor para ver el detalle; haz clic en un punto para abrir el episodio."
    )

    eps = load_episodes()
    min_d, max_d = eps["published_at"].min().date(), eps["published_at"].max().date()

    c1, c2 = st.columns([3, 1])
    with c1:
        rango = st.slider(
            "Rango de fechas",
            min_value=min_d, max_value=max_d, value=(min_d, max_d), format="DD MMM YYYY",
        )
    with c2:
        labels_on = st.multiselect(
            "Relevancia", RELEVANCE_ORDER, default=RELEVANCE_ORDER,
            help="Filtra qué niveles de relevancia se muestran.",
        )

    lo, hi = pd.Timestamp(rango[0]), pd.Timestamp(rango[1])
    view = eps[
        (eps["published_at"] >= lo)
        & (eps["published_at"] <= hi)
        & (eps["relevance_label"].isin(labels_on))
    ]

    # Métrica rápida del rango filtrado.
    dist = view["relevance_label"].value_counts().to_dict()
    cols = st.columns(5)
    cols[0].metric("Episodios", len(view))
    for i, lab in enumerate(RELEVANCE_ORDER):
        cols[i + 1].metric(lab, dist.get(lab, 0))

    if view.empty:
        st.info("No hay episodios en el rango/filtro seleccionado.")
        return

    fig = go.Figure()
    for lab in RELEVANCE_ORDER:
        sub = view[view["relevance_label"] == lab]
        if sub.empty:
            continue
        customdata = list(zip(
            sub["id"],
            sub["episode_number"].apply(lambda n: "s/n" if pd.isna(n) else int(n)),
            sub["title"],
            sub["published_at"].dt.strftime("%d %b %Y"),
            sub["word_count"].fillna(0).astype(int),
            sub["relevance_score"].fillna(0).round(2),
            sub["relevance_summary"].fillna("—"),
        ))
        fig.add_trace(go.Scatter(
            x=sub["published_at"],
            y=[lab] * len(sub),
            mode="markers",
            name=lab,
            marker=dict(
                size=12, color=RELEVANCE_COLORS[lab],
                line=dict(width=1, color="#0E1117"), opacity=0.9,
            ),
            customdata=customdata,
            hovertemplate=(
                "<b>%{customdata[2]}</b><br>"
                "Ep %{customdata[1]} · %{customdata[3]}<br>"
                "Palabras: %{customdata[4]:,}<br>"
                "Relevancia: %{customdata[5]} (%{y})<br>"
                "<i>%{customdata[6]}</i><extra></extra>"
            ),
        ))

    fig.update_layout(
        height=420,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None),
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis=dict(
            title=None, categoryorder="array", categoryarray=RELEVANCE_ORDER[::-1],
            showgrid=True, gridcolor="#21262D",
        ),
        xaxis=dict(title=None, showgrid=True, gridcolor="#21262D"),
        hoverlabel=dict(bgcolor=PANEL, font_size=13),
    )

    event = st.plotly_chart(
        fig, use_container_width=True, on_select="rerun",
        selection_mode="points", key="timeline_chart",
    )

    selected = (event or {}).get("selection", {}).get("points", [])
    if selected:
        ep_id = int(selected[0]["customdata"][0])
        go_to("episode", episode_id=ep_id)


# --------------------------------------------------------------------------- #
# Vista: Contradicciones
# --------------------------------------------------------------------------- #
def _claim_block(role: str, ep, date, ctype, text, quote) -> None:
    st.markdown(
        f"**{role}** · {ep_label(ep, date)} · {pd.to_datetime(date).date()} "
        f"&nbsp; {badge(ctype, GRAY)}",
        unsafe_allow_html=True,
    )
    st.markdown(text)
    if quote:
        st.caption(f"“{quote}”")


def _verdict_line(claim_id: int, vmap: dict) -> None:
    if claim_id in vmap:
        verdict, conf = vmap[claim_id]
        st.markdown(
            f"Verificación externa: {badge(verdict, VERDICT_COLORS.get(verdict, GRAY))} "
            f"&nbsp;<span style='color:{GRAY};font-size:0.8rem'>confianza {conf:.2f}</span>",
            unsafe_allow_html=True,
        )


def view_contradictions() -> None:
    st.title("⚡ Contradicciones")
    st.caption(
        "Posiciones que cambiaron entre episodios. El *tipo de cambio* distingue un giro "
        "silencioso de uno construido sobre evidencia intermedia."
    )

    df = load_contradictions()
    vmap = verdict_by_claim()

    # --- Métrica destacada: tipo de cambio en las contradicciones 'direct' limpias.
    direct_clean = df[(df["contradiction_type"] == "direct") & (df["data_artifact"] == 0)]
    n_silent = int((direct_clean["change_type"] == "silent").sum())
    n_evidence = int((direct_clean["change_type"] == "evidence_based").sum())
    n_ack = int((direct_clean["change_type"] == "acknowledged").sum())
    m1, m2, m3 = st.columns(3)
    m1.markdown(
        f"<div style='text-align:center'><span style='font-size:2.2rem;color:{RED};"
        f"font-weight:700'>{n_silent}</span><br>cambios silenciosos</div>",
        unsafe_allow_html=True,
    )
    m2.markdown(
        f"<div style='text-align:center'><span style='font-size:2.2rem;color:{GREEN};"
        f"font-weight:700'>{n_evidence}</span><br>con evidencia</div>",
        unsafe_allow_html=True,
    )
    m3.markdown(
        f"<div style='text-align:center'><span style='font-size:2.2rem;color:{AMBER};"
        f"font-weight:700'>{n_ack}</span><br>reconocidos</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    # --- Filtros.
    f1, f2, f3, f4 = st.columns([1.3, 1.3, 1, 1.2])
    with f1:
        types = st.multiselect(
            "Tipo de contradicción",
            ["direct", "evolution", "abandoned", "reinforced"],
            default=["direct", "evolution", "abandoned"],
        )
    with f2:
        changes = st.multiselect(
            "Tipo de cambio", ["silent", "evidence_based", "acknowledged"], default=[],
            help="Solo aplica a contradicciones 'direct' clasificadas (Fase C.2).",
        )
    with f3:
        sevs = st.multiselect("Severidad", ["high", "medium", "low"], default=[])
    with f4:
        show_artifacts = st.toggle(
            "Incluir artefactos de datos", value=False,
            help="Por defecto se ocultan las contradicciones marcadas como ruido de datos.",
        )

    # Búsqueda libre opcional (por persona/tema). Útil para el cruce desde otras vistas.
    person_focus = st.session_state.get("contra_person")
    q = st.text_input(
        "Buscar en el texto de los claims",
        value=person_focus or "",
        placeholder="p. ej. Tyler Robinson, FBI, Erika Kirk…",
    )
    if person_focus:
        st.session_state.contra_person = None  # se consume una vez

    res = df.copy()
    if not show_artifacts:
        res = res[res["data_artifact"] == 0]
    if types:
        res = res[res["contradiction_type"].isin(types)]
    if changes:
        res = res[res["change_type"].isin(changes)]
    if sevs:
        res = res[res["severity"].isin(sevs)]
    if q:
        ql = q.lower()
        mask = (
            res["a_text"].str.lower().str.contains(ql, na=False)
            | res["b_text"].str.lower().str.contains(ql, na=False)
            | res["a_persons"].str.lower().str.contains(ql, na=False)
            | res["b_persons"].str.lower().str.contains(ql, na=False)
        )
        res = res[mask]

    # --- Paginación (la lista por defecto es grande: ~191 filas limpias).
    st.markdown(f"**{len(res)}** contradicciones coinciden con el filtro.")
    if res.empty:
        st.info("Ninguna contradicción coincide. Ajusta los filtros.")
        return

    per_page = 20
    n_pages = (len(res) - 1) // per_page + 1
    page = 1
    if n_pages > 1:
        page = st.number_input(
            f"Página (de {n_pages})", min_value=1, max_value=n_pages, value=1, step=1
        )
    chunk = res.iloc[(page - 1) * per_page: page * per_page]

    for r in chunk.itertuples():
        change_b = ""
        if pd.notna(r.change_type):
            change_b = badge(
                f"cambio {CHANGE_LABEL_ES.get(r.change_type, r.change_type)}",
                CHANGE_COLORS.get(r.change_type, GRAY),
            )
        sev = r.severity or "—"
        artifact_tag = " · ⚠ artefacto" if r.data_artifact else ""
        title = (
            f"{CONTRA_TYPE_LABEL.get(r.contradiction_type, r.contradiction_type)} · "
            f"{ep_label(r.a_ep, r.a_date)} ↔ {ep_label(r.b_ep, r.b_date)} · "
            f"sev {sev} · conf {r.confidence_score:.2f}{artifact_tag}"
        )
        with st.expander(title):
            if change_b:
                st.markdown(f"Tipo de cambio narrativo: {change_b}", unsafe_allow_html=True)
            ca, cb = st.columns(2)
            with ca:
                _claim_block("Claim A", r.a_ep, r.a_date, r.a_type, r.a_text, r.a_quote)
                _verdict_line(int(r.a_id), vmap)
            with cb:
                _claim_block("Claim B", r.b_ep, r.b_date, r.b_type, r.b_text, r.b_quote)
                _verdict_line(int(r.b_id), vmap)
            st.divider()
            st.markdown(f"**Análisis del modelo:** {r.explanation}")


# --------------------------------------------------------------------------- #
# Vistas paso 2 — placeholders navegables (se implementan en la segunda entrega)
# --------------------------------------------------------------------------- #
def view_episode() -> None:
    st.title("📺 Episodio")
    ep_id = st.session_state.get("episode_id")
    if ep_id is None:
        st.info("Abre un episodio desde el Timeline (haz clic en un punto).")
        return
    eps = load_episodes()
    row = eps[eps["id"] == ep_id]
    if row.empty:
        st.warning("Episodio no encontrado.")
        return
    r = row.iloc[0]
    st.subheader(f"{ep_label(r.episode_number, r.published_at)} — {r.title}")
    st.caption(
        f"{pd.to_datetime(r.published_at).date()} · {r.relevance_label} · "
        f"{int(r.word_count or 0):,} palabras"
    )
    st.info(
        "Vista de Episodio completa (claims agrupados, badges de verificación, links a "
        "contradicciones) llega en el **paso 2**.",
        icon="🚧",
    )


def view_verifications() -> None:
    st.title("✅ Verificaciones")
    st.info("Vista de Verificaciones llega en el **paso 2**.", icon="🚧")


def view_people() -> None:
    st.title("👤 Personas")
    st.info("Vista de Personas llega en el **paso 2**.", icon="🚧")


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #
VIEWS = {
    "timeline": view_timeline,
    "episode": view_episode,
    "contradictions": view_contradictions,
    "verifications": view_verifications,
    "people": view_people,
}


def main() -> None:
    if "view" not in st.session_state:
        st.session_state.view = "timeline"
    render_sidebar()
    VIEWS.get(st.session_state.view, view_timeline)()


if __name__ == "__main__":
    main()
