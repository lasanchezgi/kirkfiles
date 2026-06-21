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
    📊 Coherencia      — scorecard narrativo de 4 dimensiones, sin veredicto  (E.2)
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
GRAY = "#9AA0A6"       # low  (secundario, pero visible)
FAINT = "#6E7681"      # none (secundario, pero visible)
INK = "#E6EDF3"
PANEL = "#161B22"

RELEVANCE_COLORS = {"high": RED, "partial": AMBER, "low": GRAY, "none": FAINT}
RELEVANCE_ORDER = ["high", "partial", "low", "none"]

CHANGE_COLORS = {"silent": RED, "evidence_based": GREEN, "acknowledged": AMBER}
CHANGE_LABEL_ES = {"silent": "silencioso", "evidence_based": "con evidencia",
                   "acknowledged": "reconocido"}

VERDICT_COLORS = {"supported": GREEN, "contradicted": RED,
                  "ambiguous": AMBER, "unverifiable": GRAY}
_MD_VERDICT_COLOR = {"supported": "green", "contradicted": "red",
                     "ambiguous": "orange", "unverifiable": "gray"}

CONTRA_TYPE_LABEL = {"direct": "directa", "evolution": "evolución",
                     "abandoned": "abandonada", "reinforced": "reforzada"}

# Claim metadata (Vista Episodio).
CLAIM_TYPE_ORDER = ["fact", "chronology", "interpretation", "speculation", "relation"]
CLAIM_TYPE_LABEL = {"fact": "Hechos", "chronology": "Cronología",
                    "interpretation": "Interpretación", "speculation": "Especulación",
                    "relation": "Relación"}
EVIDENCE_COLORS = {"source_cited": GREEN, "document": GREEN, "anecdotal": AMBER, "none": GRAY}
CONF_COLORS = {"high": GREEN, "medium": AMBER, "low": GRAY, "unknown": FAINT}

# E.2 — scorecard de coherencia. Nivel 2 de verificación (Fase D) apunta a ~192
# claims (backlog). Mientras `verifications < NIVEL_2_TARGET`, el scorecard es parcial.
NIVEL_2_TARGET = 192

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
def total_cost() -> float:
    """Costo acumulado de todo el pipeline (todas las fases en api_usage)."""
    df = _read_sql("SELECT COALESCE(SUM(cost_usd), 0) AS total FROM api_usage")
    return float(df["total"].iloc[0])


@st.cache_data(show_spinner=False)
def verdict_by_claim() -> dict:
    """Mapa claim_id → (verdict, confidence) para el badge de la Vista Episodio."""
    v = load_verifications()
    return {int(r.claim_id): (r.verdict, r.confidence) for r in v.itertuples()}


@st.cache_data(show_spinner=False)
def claim_contradiction_index() -> dict:
    """Mapa claim_id → lista de ids de contradicciones (limpias) que lo involucran."""
    contra = load_contradictions()
    idx: dict[int, list[int]] = {}
    for r in contra[contra["data_artifact"] == 0].itertuples():
        idx.setdefault(int(r.a_id), []).append(int(r.id))
        idx.setdefault(int(r.b_id), []).append(int(r.id))
    return idx


@st.cache_data(show_spinner=False)
def scorecard() -> dict:
    """Métricas del scorecard de coherencia (E.2).

    No emite veredicto: solo el porcentaje por dimensión calculado sobre los datos
    existentes. Se recalcula solo cuando avancen las verificaciones (Nivel 2), así
    que el respaldo externo deja de ser parcial sin tocar este código.
    """
    contra = load_contradictions()
    direct = contra[(contra["contradiction_type"] == "direct") & (contra["data_artifact"] == 0)]
    n_direct = len(direct)
    ct = direct["change_type"].value_counts().to_dict()
    silent = int(ct.get("silent", 0))
    acknowledged = int(ct.get("acknowledged", 0))
    evidence_based = int(ct.get("evidence_based", 0))

    v = load_verifications()
    vd = v["verdict"].value_counts().to_dict()
    supported = int(vd.get("supported", 0))
    contradicted = int(vd.get("contradicted", 0))
    n_verified = len(v)
    total_claims = len(load_claims())

    def pct(num: int, den: int) -> float:
        return (num / den) if den else 0.0

    dims = [
        {
            "label": "Consistencia interna",
            "measure": "Cambios de posición con sustento frente a cambios silenciosos",
            "source": "tabla contradictions",
            "pct": pct(evidence_based, n_direct),
            "fraction": f"{evidence_based}/{n_direct}",
            "detail": f"{evidence_based} con evidencia · {silent} silenciosos · {acknowledged} reconocidos",
            "color": GREEN,
        },
        {
            "label": "Respaldo externo",
            "measure": "Claims verificados respaldados por fuentes externas",
            "source": "tabla verifications",
            "pct": pct(supported, n_verified),
            "fraction": f"{supported}/{n_verified}",
            "detail": f"{supported} supported · {contradicted} contradicted",
            "color": GREEN,
        },
        {
            "label": "Evolución con evidencia",
            "measure": "Cambios respaldados por evidencia en episodios intermedios",
            "source": "contradictions.change_type",
            "pct": pct(evidence_based, n_direct),
            "fraction": f"{evidence_based}/{n_direct}",
            "detail": f"{evidence_based} de {n_direct} cambios directos",
            "color": GREEN,
        },
        {
            "label": "Transparencia narrativa",
            "measure": "Cambios reconocidos explícitamente por Candace",
            "source": "contradictions.change_type",
            "pct": pct(acknowledged, n_direct),
            "fraction": f"{acknowledged}/{n_direct}",
            "detail": f"{acknowledged} reconocidos de {n_direct} cambios directos",
            "color": AMBER,
        },
    ]
    return {
        "dims": dims,
        "n_verified": n_verified,
        "total_claims": total_claims,
        "nivel2_target": NIVEL_2_TARGET,
        "nivel2_complete": n_verified >= NIVEL_2_TARGET,
    }


@st.cache_data(show_spinner=False)
def people_stats() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Agrega menciones de personas. Devuelve (resumen_por_persona, claims_por_persona).

    resumen: person · claims · episodes · contradictions (ordenable).
    detalle: filas (person, claim_id, episode_id) para el cross-filter."""
    claims = load_claims()
    contra = load_contradictions()

    rows = []
    for r in claims.itertuples():
        for person in parse_json_list(r.persons_mentioned):
            name = str(person).strip()
            if name:
                rows.append((name, int(r.id), int(r.episode_id)))
    detail = pd.DataFrame(rows, columns=["person", "claim_id", "episode_id"])

    contra_count: dict[str, int] = {}
    for r in contra[contra["data_artifact"] == 0].itertuples():
        ppl = {str(p).strip() for p in parse_json_list(r.a_persons) + parse_json_list(r.b_persons)}
        for p in ppl:
            if p:
                contra_count[p] = contra_count.get(p, 0) + 1

    if detail.empty:
        return pd.DataFrame(columns=["person", "claims", "episodes", "contradicciones"]), detail
    summary = (
        detail.groupby("person")
        .agg(claims=("claim_id", "nunique"), episodes=("episode_id", "nunique"))
        .reset_index()
    )
    summary["contradicciones"] = summary["person"].map(lambda p: contra_count.get(p, 0))
    summary = summary.sort_values(["claims", "episodes"], ascending=False).reset_index(drop=True)
    return summary, detail


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
    ("📊 Coherencia", "coherence"),
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
        st.markdown(
            f"<div style='margin-top:0.6rem;color:{GRAY};font-size:0.9rem'>"
            f"💸 Costo total del proyecto<br>"
            f"<b style='color:{INK};font-size:1.05rem'>${total_cost():,.2f}</b> en llamadas a la API"
            f"</div>",
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

    # Tamaño del punto ∝ word_count (densidad del episodio), escalado a píxeles
    # sobre el rango global para que el contexto no cambie al filtrar.
    wmin = float(eps["word_count"].fillna(0).min())
    wmax = float(eps["word_count"].fillna(0).max()) or 1.0

    def _size(wc) -> float:
        wc = 0.0 if pd.isna(wc) else float(wc)
        frac = (wc - wmin) / (wmax - wmin) if wmax > wmin else 0.5
        return 9 + frac * 23  # 9–32 px

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
            [lab] * len(sub),
        ))
        fig.add_trace(go.Scatter(
            x=sub["published_at"],
            y=[0] * len(sub),                       # un solo carril cronológico
            mode="markers",
            name=lab,
            marker=dict(
                size=[_size(w) for w in sub["word_count"]],
                color=RELEVANCE_COLORS[lab],
                line=dict(width=1, color="#0E1117"), opacity=0.88,
            ),
            customdata=customdata,
            hovertemplate=(
                "<b>%{customdata[2]}</b><br>"
                "Ep %{customdata[1]} · %{customdata[3]}<br>"
                "Palabras: %{customdata[4]:,}<br>"
                "Relevancia: %{customdata[5]} (%{customdata[7]})"
                "<br><i>%{customdata[6]}</i><extra></extra>"
            ),
        ))

    fig.update_layout(
        height=300,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None),
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis=dict(
            title=None, showgrid=False, zeroline=True, zerolinecolor="#21262D",
            showticklabels=False, range=[-1, 1],
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


_MD_CHANGE_COLOR = {"silent": "red", "evidence_based": "green", "acknowledged": "orange"}


def render_contradiction_rows(res: pd.DataFrame, vmap: dict, per_page: int = 20) -> None:
    """Lista paginada de contradicciones como expanders, con el badge de change_type
    visible en el título."""
    if res.empty:
        st.info("Ninguna contradicción coincide. Ajusta los filtros.")
        return

    n_pages = (len(res) - 1) // per_page + 1
    page = 1
    if n_pages > 1:
        page = st.number_input(
            f"Página (de {n_pages})", min_value=1, max_value=n_pages, value=1, step=1
        )
    chunk = res.iloc[(page - 1) * per_page: page * per_page]

    for r in chunk.itertuples():
        change_b = ""
        title_change = ""
        if pd.notna(r.change_type):
            change_b = badge(
                f"cambio {CHANGE_LABEL_ES.get(r.change_type, r.change_type)}",
                CHANGE_COLORS.get(r.change_type, GRAY),
            )
            color = _MD_CHANGE_COLOR.get(r.change_type, "gray")
            title_change = f" · :{color}[● {CHANGE_LABEL_ES.get(r.change_type, r.change_type)}]"
        sev = r.severity or "—"
        artifact_tag = " · ⚠ artefacto" if r.data_artifact else ""
        title = (
            f"{CONTRA_TYPE_LABEL.get(r.contradiction_type, r.contradiction_type)} · "
            f"{ep_label(r.a_ep, r.a_date)} ↔ {ep_label(r.b_ep, r.b_date)} · "
            f"sev {sev} · conf {r.confidence_score:.2f}{title_change}{artifact_tag}"
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


def view_contradictions() -> None:
    st.title("⚡ Contradicciones")
    st.caption(
        "Posiciones que cambiaron entre episodios. El *tipo de cambio* distingue un giro "
        "silencioso de uno construido sobre evidencia intermedia."
    )

    df = load_contradictions()
    vmap = verdict_by_claim()

    # --- Métrica destacada (lo más importante del proyecto): tipo de cambio en las
    # contradicciones 'direct' limpias. st.metric nativo, arriba de todo.
    direct_clean = df[(df["contradiction_type"] == "direct") & (df["data_artifact"] == 0)]
    n_silent = int((direct_clean["change_type"] == "silent").sum())
    n_evidence = int((direct_clean["change_type"] == "evidence_based").sum())
    n_ack = int((direct_clean["change_type"] == "acknowledged").sum())
    m1, m2, m3 = st.columns(3)
    m1.metric("🔴 Cambios silenciosos", n_silent)
    m2.metric("🟢 Con evidencia", n_evidence)
    m3.metric("🟡 Reconocidos", n_ack)
    st.divider()

    # --- ¿Llegamos aquí enfocando un claim concreto (desde la Vista Episodio)?
    claim_focus = st.session_state.get("contra_claim_id")
    if claim_focus:
        focus_row = df[(df["a_id"] == claim_focus) | (df["b_id"] == claim_focus)]
        c1, c2 = st.columns([4, 1])
        c1.info(f"Mostrando las contradicciones del claim #{claim_focus}.", icon="🔗")
        if c2.button("Quitar enfoque", use_container_width=True):
            del st.session_state["contra_claim_id"]
            st.rerun()
        render_contradiction_rows(focus_row, vmap)
        return

    # --- Filtros.
    f1, f2, f3, f4 = st.columns([1.3, 1.3, 1, 1.2])
    with f1:
        types = st.multiselect(
            "Tipo de contradicción",
            ["direct", "evolution", "abandoned", "reinforced"],
            default=["direct"],
            help="Las 'direct' (mutuamente excluyentes) son las más relevantes; arrancamos ahí.",
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

    st.markdown(f"**{len(res)}** contradicciones coinciden con el filtro.")
    render_contradiction_rows(res, vmap)


# --------------------------------------------------------------------------- #
# Vista: Episodio
# --------------------------------------------------------------------------- #
def _episode_picker(eps: pd.DataFrame) -> int | None:
    """Selector de episodio (cuando se entra sin venir del Timeline)."""
    opts = eps.sort_values("published_at", ascending=False)
    labels = {
        int(r.id): f"{ep_label(r.episode_number, r.published_at)} · "
                   f"{pd.to_datetime(r.published_at).date()} · {r.title}"
        for r in opts.itertuples()
    }
    ids = list(labels.keys())
    current = st.session_state.get("episode_id")
    idx = ids.index(current) if current in ids else 0
    return st.selectbox("Episodio", ids, index=idx, format_func=lambda i: labels[i])


def view_episode() -> None:
    st.title("📺 Episodio")
    eps = load_episodes()

    ep_id = _episode_picker(eps)
    st.session_state.episode_id = ep_id
    row = eps[eps["id"] == ep_id]
    if row.empty:
        st.warning("Episodio no encontrado.")
        return
    r = row.iloc[0]

    # --- Header.
    st.subheader(r.title)
    label_color = RELEVANCE_COLORS.get(r.relevance_label, GRAY)
    header = (
        f"{ep_label(r.episode_number, r.published_at)} · "
        f"{pd.to_datetime(r.published_at).date()} &nbsp; "
        f"{badge(r.relevance_label, label_color)} &nbsp; "
        f"{int(r.word_count or 0):,} palabras"
    )
    if pd.notna(r.relevance_score):
        header += f" &nbsp;·&nbsp; score {r.relevance_score:.2f}"
    st.markdown(header, unsafe_allow_html=True)
    topics = parse_json_list(r.charlie_topics)
    if topics:
        st.markdown(
            " ".join(badge(t, FAINT) for t in topics), unsafe_allow_html=True
        )
    if isinstance(r.relevance_summary, str) and r.relevance_summary:
        st.caption(r.relevance_summary)
    st.divider()

    # --- Claims agrupados por tipo.
    claims = load_claims()
    ep_claims = claims[claims["episode_id"] == ep_id]
    vmap = verdict_by_claim()
    cidx = claim_contradiction_index()

    st.markdown(f"### Claims extraídos · {len(ep_claims)}")
    if ep_claims.empty:
        st.info("Este episodio no tiene claims extraídos.")
        return

    present_types = [t for t in CLAIM_TYPE_ORDER if (ep_claims["claim_type"] == t).any()]
    present_types += [t for t in ep_claims["claim_type"].dropna().unique() if t not in CLAIM_TYPE_ORDER]

    for ctype in present_types:
        sub = ep_claims[ep_claims["claim_type"] == ctype]
        st.markdown(f"#### {CLAIM_TYPE_LABEL.get(ctype, ctype)} · {len(sub)}")
        for c in sub.itertuples():
            with st.container(border=True):
                meta = []
                if c.speaker_confidence:
                    meta.append(badge(f"confianza {c.speaker_confidence}",
                                      CONF_COLORS.get(c.speaker_confidence, GRAY)))
                if c.evidence_provided:
                    meta.append(badge(f"evidencia {c.evidence_provided}",
                                      EVIDENCE_COLORS.get(c.evidence_provided, GRAY)))
                cid = int(c.id)
                if cid in vmap:
                    verd, conf = vmap[cid]
                    meta.append(badge(f"✓ {verd}", VERDICT_COLORS.get(verd, GRAY)))
                if cid in cidx:
                    meta.append(badge(f"⚡ {len(cidx[cid])} contradicción(es)", RED))
                if meta:
                    st.markdown(" ".join(meta), unsafe_allow_html=True)

                st.markdown(c.claim_text)

                cols = st.columns([1, 1, 3])
                if c.quote_verbatim:
                    with cols[0].popover("Cita textual"):
                        st.markdown(f"“{c.quote_verbatim}”")
                if cid in cidx:
                    if cols[1].button("⚡ Ver contradicciones", key=f"goc_{cid}"):
                        go_to("contradictions", contra_claim_id=cid)


# --------------------------------------------------------------------------- #
# Vista: Verificaciones
# --------------------------------------------------------------------------- #
def _render_sources(label: str, raw, color: str) -> None:
    urls = parse_json_list(raw)
    if not urls:
        return
    st.markdown(f"**{label}:**")
    for u in urls:
        st.markdown(f"- [{u}]({u})")


def view_verifications() -> None:
    st.title("✅ Verificaciones")
    st.caption("Claims contrastados contra fuentes externas (Fase D). Solo claims con registro.")

    v = load_verifications()
    if v.empty:
        st.info("Aún no hay verificaciones.")
        return

    # --- Métrica: distribución de veredictos.
    order = ["supported", "contradicted", "ambiguous", "unverifiable"]
    dist = v["verdict"].value_counts().to_dict()
    cols = st.columns(len(order))
    icons = {"supported": "🟢", "contradicted": "🔴", "ambiguous": "🟡", "unverifiable": "⚪"}
    for col, verd in zip(cols, order):
        col.metric(f"{icons[verd]} {verd}", dist.get(verd, 0))
    st.divider()

    chosen = st.multiselect("Filtrar por veredicto", order, default=[])
    res = v[v["verdict"].isin(chosen)] if chosen else v

    st.markdown(f"**{len(res)}** verificaciones.")
    for r in res.itertuples():
        color = VERDICT_COLORS.get(r.verdict, GRAY)
        title = (
            f"{ep_label(r.episode_number, r.published_at)} · "
            f":{_MD_VERDICT_COLOR.get(r.verdict, 'gray')}[● {r.verdict}] · "
            f"conf {r.confidence:.2f}"
        )
        with st.expander(title):
            st.markdown(
                f"{badge(r.verdict, color)} &nbsp;"
                f"<span style='color:{GRAY};font-size:0.8rem'>confianza {r.confidence:.2f}</span>",
                unsafe_allow_html=True,
            )
            st.markdown(f"**{r.claim_text}**")
            st.caption(f"{ep_label(r.episode_number, r.published_at)} · "
                       f"{pd.to_datetime(r.published_at).date()} · {r.episode_title}")
            _render_sources("Fuentes que lo respaldan", r.sources_supporting, GREEN)
            _render_sources("Fuentes que lo contradicen", r.sources_contradicting, RED)
            _render_sources("Documentos primarios", r.primary_documents, AMBER)
            if isinstance(r.llm_reasoning, str) and r.llm_reasoning:
                with st.popover("Razonamiento del modelo"):
                    st.markdown(r.llm_reasoning)


# --------------------------------------------------------------------------- #
# Vista: Personas
# --------------------------------------------------------------------------- #
def view_people() -> None:
    st.title("👤 Personas")
    st.caption("Quién aparece en el corpus, en cuántos claims/episodios y en cuántas contradicciones.")

    summary, detail = people_stats()
    if summary.empty:
        st.info("No hay personas extraídas.")
        return

    sort_by = st.radio(
        "Ordenar por", ["claims", "episodes", "contradicciones"],
        horizontal=True, format_func=lambda s: {"claims": "claims", "episodes": "episodios",
                                                 "contradicciones": "contradicciones"}[s],
    )
    table = summary.sort_values(sort_by, ascending=False).reset_index(drop=True)

    st.markdown(f"**{len(table)}** personas únicas mencionadas.")
    st.dataframe(
        table.rename(columns={"person": "persona", "episodes": "episodios"}),
        use_container_width=True, hide_index=True, height=380,
    )

    st.divider()
    # --- Drill-down: explorar una persona → sus claims + cross-link a contradicciones.
    person = st.selectbox("Explorar persona", table["person"].tolist())
    prow = summary[summary["person"] == person].iloc[0]
    m1, m2, m3, m4 = st.columns([1, 1, 1, 1.4])
    m1.metric("Claims", int(prow.claims))
    m2.metric("Episodios", int(prow.episodes))
    m3.metric("Contradicciones", int(prow.contradicciones))
    with m4:
        st.write("")
        if st.button("⚡ Ver en Contradicciones", use_container_width=True):
            go_to("contradictions", contra_person=person)

    claims = load_claims()
    pids = detail[detail["person"] == person]["claim_id"].tolist()
    pclaims = claims[claims["id"].isin(pids)].sort_values("published_at")
    st.markdown(f"#### Claims que mencionan a **{person}** · {len(pclaims)}")
    cidx = claim_contradiction_index()
    for c in pclaims.itertuples():
        tag = badge(f"⚡ {len(cidx[int(c.id)])}", RED) if int(c.id) in cidx else ""
        st.markdown(
            f"- {ep_label(c.episode_number, c.published_at)} · "
            f"{badge(c.claim_type or '—', FAINT)} {tag} &nbsp; {c.claim_text}",
            unsafe_allow_html=True,
        )


# --------------------------------------------------------------------------- #
# Vista: Coherencia (scorecard narrativo — E.2)
# --------------------------------------------------------------------------- #
def _score_bar(d: dict) -> str:
    """Una barra horizontal: label · porcentaje · fracción · qué mide · fuente."""
    pct = d["pct"]
    width = max(pct * 100, 1.5)  # deja siempre un sliver visible aunque sea 0%
    return f"""
<div style='margin:0 0 1.6rem 0'>
  <div style='display:flex;justify-content:space-between;align-items:baseline;margin-bottom:0.35rem'>
    <span style='font-weight:600;font-size:1.05rem;color:{INK}'>{d['label']}</span>
    <span style='font-weight:700;font-size:1.2rem;color:{d['color']}'>{pct*100:.0f}%
      <span style='color:{GRAY};font-weight:500;font-size:0.85rem'>&nbsp;({d['fraction']})</span>
    </span>
  </div>
  <div style='background:{PANEL};border:1px solid {FAINT}44;border-radius:7px;height:16px;overflow:hidden'>
    <div style='width:{width:.1f}%;height:100%;background:{d['color']};border-radius:7px 0 0 7px'></div>
  </div>
  <div style='color:{GRAY};font-size:0.86rem;margin-top:0.35rem'>{d['measure']}
    <span style='color:{FAINT}'> · {d['detail']} · fuente: {d['source']}</span>
  </div>
</div>
"""


def view_coherence() -> None:
    st.title("📊 Coherencia")
    st.caption(
        "¿La investigación es internamente coherente y externamente respaldada? "
        "Esta vista no emite un veredicto: presenta cuatro dimensiones para que el "
        "lector conecte los puntos."
    )

    sc = scorecard()

    if not sc["nivel2_complete"]:
        st.warning(
            f"**Scorecard parcial.** La verificación externa está en Nivel 1 — "
            f"{sc['n_verified']} de los ~{sc['nivel2_target']} claims objetivo del Nivel 2. "
            f"La dimensión de *respaldo externo* es preliminar y puede cambiar al "
            f"completar la Fase D.",
            icon="⚠️",
        )

    st.write("")
    for d in sc["dims"]:
        st.markdown(_score_bar(d), unsafe_allow_html=True)

    st.divider()
    st.caption(
        f"Basado en {sc['n_verified']} claims verificados de {sc['total_claims']:,} totales — "
        f"scorecard parcial hasta completar Nivel 2 de verificación. Sin score agregado "
        f"ni veredicto final: las conclusiones son del lector."
    )


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #
VIEWS = {
    "timeline": view_timeline,
    "episode": view_episode,
    "contradictions": view_contradictions,
    "verifications": view_verifications,
    "people": view_people,
    "coherence": view_coherence,
}


def main() -> None:
    if "view" not in st.session_state:
        st.session_state.view = "timeline"
    render_sidebar()
    VIEWS.get(st.session_state.view, view_timeline)()


if __name__ == "__main__":
    main()
