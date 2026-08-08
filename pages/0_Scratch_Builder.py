from __future__ import annotations

import streamlit as st

from model_builder import (
    ShapeError,
    analyze_architecture,
    export_architecture_json,
    generate_model_code,
    validate_forward_pass,
)
from model_builder.scratch_component import render_scratch_builder


st.set_page_config(
    page_title="Scratch Builder | AI Lab Junior",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="collapsed",
)


DEFAULTS = {
    "lang": "ar",
    "input_mode": "vector",
    "features": 2,
    "channels": 1,
    "height": 28,
    "width": 28,
    "output_size": 2,
    "layers": [],
}
for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)


TEXT = {
    "ar": {
        "title": "🧩 مختبر الشبكات — أسلوب Scratch",
        "subtitle": "اسحب اللبنات، رتبها، غيّر إعداداتها، ثم شغّل نفس الشبكة بـPyTorch.",
        "simple": "⚡ مثال بسيط",
        "cnn": "🖼️ مثال صور",
        "clear": "🗑 شبكة جديدة",
        "valid": "الشبكة سليمة وجاهزة للتشغيل.",
        "invalid": "جرّب إصلاح الشبكة",
        "params": "المعاملات",
        "stages": "المراحل",
        "output": "شكل الخرج",
        "test": "🧪 اختبر الشبكة",
        "train": "🚀 انتقل إلى التدريب والنتائج",
        "code": "الكود والحفظ",
        "download_code": "تحميل PyTorch",
        "download_json": "حفظ المشروع",
        "tip": "الفكرة للتلميذ: لا تبدأ بالكود. ابنِ الشبكة مثل لعبة، شاهد إن كانت صحيحة، ثم درّبها وقارن النتائج.",
    },
    "fr": {
        "title": "🧩 Laboratoire réseau — style Scratch",
        "subtitle": "Glisse les blocs, réorganise-les, règle-les, puis exécute le même réseau avec PyTorch.",
        "simple": "⚡ Exemple simple",
        "cnn": "🖼️ Exemple image",
        "clear": "🗑 Nouveau réseau",
        "valid": "Le réseau est valide et prêt à fonctionner.",
        "invalid": "Corrige le réseau",
        "params": "Paramètres",
        "stages": "Étapes",
        "output": "Sortie",
        "test": "🧪 Tester le réseau",
        "train": "🚀 Entraîner et voir les résultats",
        "code": "Code et sauvegarde",
        "download_code": "Télécharger PyTorch",
        "download_json": "Sauvegarder le projet",
        "tip": "Idée pédagogique : ne commence pas par le code. Construis le réseau comme un jeu, vérifie-le, entraîne-le puis compare les résultats.",
    },
}


def t(key: str) -> str:
    return TEXT[st.session_state.lang][key]


def project_from_session() -> dict:
    return {
        "mode": st.session_state.input_mode,
        "features": int(st.session_state.features),
        "channels": int(st.session_state.channels),
        "height": int(st.session_state.height),
        "width": int(st.session_state.width),
        "output_size": int(st.session_state.output_size),
        "layers": st.session_state.layers,
    }


def apply_project(project: dict) -> None:
    st.session_state.input_mode = "image" if project.get("mode") == "image" else "vector"
    st.session_state.features = max(1, int(project.get("features", 2)))
    st.session_state.channels = max(1, int(project.get("channels", 1)))
    st.session_state.height = max(4, int(project.get("height", 28)))
    st.session_state.width = max(4, int(project.get("width", 28)))
    st.session_state.output_size = max(2, int(project.get("output_size", 2)))
    st.session_state.layers = list(project.get("layers", []))
    st.session_state.pop("lab_result", None)
    st.session_state.pop("lab_dataset", None)
    st.session_state.pop("lab_signature", None)


def sync_from_component() -> None:
    state = st.session_state.get("scratch_canvas")
    project = getattr(state, "project", None) if state is not None else None
    if project:
        apply_project(dict(project))


def input_shape() -> tuple[int, ...]:
    if st.session_state.input_mode == "image":
        return (
            int(st.session_state.channels),
            int(st.session_state.height),
            int(st.session_state.width),
        )
    return (int(st.session_state.features),)


def load_simple() -> None:
    apply_project(
        {
            "mode": "vector",
            "features": 2,
            "channels": 1,
            "height": 28,
            "width": 28,
            "output_size": 2,
            "layers": [
                {"type": "linear", "params": {"out_features": 8, "activation": "ReLU"}},
                {"type": "linear", "params": {"out_features": 8, "activation": "ReLU"}},
            ],
        }
    )


def load_cnn() -> None:
    apply_project(
        {
            "mode": "image",
            "features": 2,
            "channels": 1,
            "height": 28,
            "width": 28,
            "output_size": 4,
            "layers": [
                {
                    "type": "conv2d",
                    "params": {
                        "out_channels": 8,
                        "kernel_size": 3,
                        "stride": 1,
                        "padding": 1,
                        "activation": "ReLU",
                    },
                },
                {"type": "maxpool2d", "params": {"kernel_size": 2, "stride": 2}},
                {"type": "flatten", "params": {}},
                {"type": "linear", "params": {"out_features": 16, "activation": "ReLU"}},
            ],
        }
    )


st.markdown(
    """
    <style>
      .block-container {max-width: 1320px; padding-top: 1rem; padding-bottom: 3rem;}
      .scratch-hero {border:1px solid rgba(128,128,128,.2); border-radius:18px; padding:1rem 1.2rem; margin-bottom:.8rem;}
      .scratch-hero h1 {margin:0; font-size:1.75rem;}
      .scratch-hero p {margin:.28rem 0 0 0; opacity:.75;}
    </style>
    """,
    unsafe_allow_html=True,
)

lang_col, preset1, preset2, reset_col = st.columns([1.1, 1.3, 1.3, 1.3])
with lang_col:
    st.selectbox(
        "Language / اللغة",
        options=["ar", "fr"],
        format_func=lambda value: "العربية" if value == "ar" else "Français",
        key="lang",
        label_visibility="collapsed",
    )
with preset1:
    if st.button(t("simple"), use_container_width=True):
        load_simple()
        st.rerun()
with preset2:
    if st.button(t("cnn"), use_container_width=True):
        load_cnn()
        st.rerun()
with reset_col:
    if st.button(t("clear"), use_container_width=True):
        apply_project({"mode": "vector", "features": 2, "output_size": 2, "layers": []})
        st.rerun()

st.markdown(
    f'<div class="scratch-hero"><h1>{t("title")}</h1><p>{t("subtitle")}</p></div>',
    unsafe_allow_html=True,
)

render_scratch_builder(
    project_from_session(),
    lang=st.session_state.lang,
    key="scratch_canvas",
    on_change=sync_from_component,
)

shape = input_shape()
analysis = None
error = None
try:
    analysis = analyze_architecture(st.session_state.layers, shape, int(st.session_state.output_size))
except ShapeError as exc:
    error = str(exc)

st.divider()
if analysis is not None:
    st.success(t("valid"))
    m1, m2, m3 = st.columns(3)
    m1.metric(t("params"), f"{analysis.total_parameters:,}")
    m2.metric(t("stages"), len(analysis.reports))
    m3.metric(t("output"), " × ".join(map(str, analysis.output_shape)))

    a1, a2 = st.columns(2)
    with a1:
        if st.button(t("test"), use_container_width=True):
            try:
                result_shape = validate_forward_pass(
                    st.session_state.layers,
                    shape,
                    int(st.session_state.output_size),
                )
                st.success(f"PyTorch ✓  →  {result_shape}")
            except (ShapeError, RuntimeError) as exc:
                st.error(str(exc))
    with a2:
        if st.button(t("train"), type="primary", use_container_width=True):
            st.switch_page("pages/1_Train_and_Visualize.py")

    st.caption(t("tip"))

    with st.expander(t("code")):
        code = generate_model_code(st.session_state.layers, shape, int(st.session_state.output_size))
        project_json = export_architecture_json(
            st.session_state.layers,
            shape,
            int(st.session_state.output_size),
        )
        st.code(code, language="python")
        d1, d2 = st.columns(2)
        d1.download_button(
            t("download_code"),
            data=code,
            file_name="student_model.py",
            mime="text/x-python",
            use_container_width=True,
        )
        d2.download_button(
            t("download_json"),
            data=project_json,
            file_name="ai_lab_project.json",
            mime="application/json",
            use_container_width=True,
        )
else:
    st.warning(f'{t("invalid")}: {error or ""}')
    st.caption(t("tip"))
