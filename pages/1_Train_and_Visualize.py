from __future__ import annotations

import json

import pandas as pd
import streamlit as st
import torch

from model_builder import ShapeError, analyze_architecture, build_torch_model
from model_builder.training import make_image_dataset, make_vector_dataset, train_classifier


st.set_page_config(
    page_title="Train & Visualize | AI Lab Junior",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)


if "lang" not in st.session_state:
    st.session_state.lang = "ar"
if "layers" not in st.session_state:
    st.session_state.layers = []
if "input_mode" not in st.session_state:
    st.session_state.input_mode = "vector"
if "features" not in st.session_state:
    st.session_state.features = 2
if "channels" not in st.session_state:
    st.session_state.channels = 1
if "height" not in st.session_state:
    st.session_state.height = 28
if "width" not in st.session_state:
    st.session_state.width = 28
if "output_size" not in st.session_state:
    st.session_state.output_size = 2


AR = st.session_state.lang == "ar"
TEXT = {
    "title": "🧪 درّب الشبكة وشاهدها" if AR else "🧪 Entraîner et visualiser le réseau",
    "subtitle": (
        "هنا تتحول البنية التي صنعتها إلى تجربة حقيقية: بيانات → تدريب → نتائج."
        if AR
        else "Ici, ton architecture devient une vraie expérience : données → entraînement → résultats."
    ),
    "visual": "شكل الشبكة" if AR else "Vue du réseau",
    "training": "مختبر التدريب" if AR else "Laboratoire d'entraînement",
    "train": "🚀 ابدأ التدريب" if AR else "🚀 Lancer l'entraînement",
    "demo": "⚡ حمّل مثالاً بسيطاً للتدريب" if AR else "⚡ Charger un exemple simple",
    "loss": "الخسارة Loss" if AR else "Perte (Loss)",
    "accuracy": "الدقة Accuracy" if AR else "Précision (Accuracy)",
    "test_accuracy": "دقة الاختبار" if AR else "Précision test",
    "train_accuracy": "دقة التدريب" if AR else "Précision entraînement",
    "epochs": "عدد دورات التدريب Epochs" if AR else "Nombre d'epochs",
    "learning_rate": "سرعة التعلم" if AR else "Learning rate",
    "samples": "عدد الأمثلة" if AR else "Nombre d'exemples",
    "noise": "صعوبة البيانات" if AR else "Bruit / difficulté",
    "dataset": "نوع التجربة" if AR else "Type d'expérience",
    "results": "نتائج التدريب" if AR else "Résultats de l'entraînement",
    "confusion": "جدول القرارات" if AR else "Matrice de confusion",
    "true_data": "البيانات الحقيقية" if AR else "Données réelles",
    "model_view": "ما الذي تعلّمه النموذج؟" if AR else "Ce que le modèle a appris",
    "no_model": (
        "البنية الحالية غير صالحة للتدريب. ارجع للصفحة الرئيسية وصححها، أو حمّل المثال البسيط."
        if AR
        else "L'architecture actuelle n'est pas entraînable. Corrige-la sur la page principale ou charge l'exemple simple."
    ),
    "too_many_classes": (
        "للحفاظ على سرعة المختبر على الهاتف، التدريب التعليمي محدود إلى 12 فئة."
        if AR
        else "Pour garder le laboratoire rapide sur mobile, l'entraînement pédagogique est limité à 12 classes."
    ),
    "synthetic_note": (
        "البيانات هنا مولّدة داخل التطبيق للتعلّم والتجريب؛ ليست مجموعة بيانات حقيقية مثل MNIST."
        if AR
        else "Les données sont générées dans l'application pour apprendre et expérimenter ; ce n'est pas un dataset réel comme MNIST."
    ),
}

LAYER_LABELS = {
    "ar": {
        "input": "المدخلات",
        "linear": "Fully Connected",
        "conv2d": "Conv2D",
        "maxpool2d": "MaxPool",
        "dropout": "Dropout",
        "flatten": "Flatten",
        "output": "القرار",
    },
    "fr": {
        "input": "Entrée",
        "linear": "Fully Connected",
        "conv2d": "Conv2D",
        "maxpool2d": "MaxPool",
        "dropout": "Dropout",
        "flatten": "Flatten",
        "output": "Décision",
    },
}

ICONS = {
    "input": "📥",
    "linear": "●",
    "conv2d": "▦",
    "maxpool2d": "▤",
    "dropout": "◌",
    "flatten": "⇢",
    "output": "🎯",
}


st.markdown(
    """
    <style>
      .block-container {max-width: 1180px; padding-top: 1.4rem; padding-bottom: 3rem;}
      .lab-hero {border:1px solid rgba(128,128,128,.22); border-radius:18px; padding:1rem 1.2rem; margin-bottom:1rem;}
      .lab-hero h1 {margin:0; font-size:1.9rem;}
      .lab-hero p {margin:.35rem 0 0 0; opacity:.78;}
      .network-flow {display:flex; flex-direction:column; align-items:center; width:100%; margin:.5rem auto 1.4rem auto;}
      .network-node {width:min(92%, 620px); border:1px solid rgba(128,128,128,.28); border-radius:16px; padding:.8rem 1rem; text-align:center; background:rgba(127,127,127,.045);}
      .network-icon {font-size:1.45rem; margin-bottom:.15rem;}
      .network-name {font-weight:750; font-size:1.02rem;}
      .network-shape {font-family:monospace; opacity:.78; margin-top:.2rem;}
      .network-params {font-size:.82rem; opacity:.62; margin-top:.15rem;}
      .network-arrow {font-size:1.65rem; line-height:1.2; opacity:.55; padding:.12rem 0;}
      .neuron-row {display:flex; justify-content:center; gap:.26rem; flex-wrap:wrap; margin-top:.42rem;}
      .neuron-dot {width:14px; height:14px; border-radius:50%; border:2px solid currentColor; opacity:.62;}
      .feature-square {width:15px; height:15px; border:2px solid currentColor; opacity:.58; transform:rotate(4deg);}
      .tiny-note {opacity:.72; font-size:.9rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def shape_text(shape: tuple[int, ...]) -> str:
    return " × ".join(str(v) for v in shape)


def miniature(kind: str, shape: tuple[int, ...]) -> str:
    count = shape[0] if shape else 1
    shown = min(int(count), 10)
    css_class = "feature-square" if kind in {"conv2d", "maxpool2d"} else "neuron-dot"
    dots = "".join(f'<span class="{css_class}"></span>' for _ in range(shown))
    suffix = f"<span>×{count}</span>" if count > shown else ""
    return f'<div class="neuron-row">{dots}{suffix}</div>'


def render_network(analysis) -> None:
    lang = "ar" if AR else "fr"
    nodes = [
        {
            "kind": "input",
            "shape": analysis.input_shape,
            "parameters": 0,
        }
    ]
    nodes.extend(
        {
            "kind": report.kind,
            "shape": report.output_shape,
            "parameters": report.parameters,
        }
        for report in analysis.reports
    )

    html_parts = ['<div class="network-flow">']
    for index, node in enumerate(nodes):
        kind = node["kind"]
        html_parts.append(
            f"""
            <div class="network-node">
              <div class="network-icon">{ICONS[kind]}</div>
              <div class="network-name">{LAYER_LABELS[lang][kind]}</div>
              <div class="network-shape">{shape_text(node['shape'])}</div>
              {miniature(kind, node['shape'])}
              <div class="network-params">{node['parameters']:,} params</div>
            </div>
            """
        )
        if index < len(nodes) - 1:
            html_parts.append('<div class="network-arrow">↓</div>')
    html_parts.append("</div>")
    st.markdown("".join(html_parts), unsafe_allow_html=True)


def current_input_shape() -> tuple[int, ...]:
    if st.session_state.input_mode == "image":
        return (
            int(st.session_state.channels),
            int(st.session_state.height),
            int(st.session_state.width),
        )
    return (int(st.session_state.features),)


def architecture_signature(input_shape: tuple[int, ...]) -> str:
    return json.dumps(
        {
            "layers": st.session_state.layers,
            "input_shape": input_shape,
            "output_size": int(st.session_state.output_size),
        },
        sort_keys=True,
    )


def load_training_demo() -> None:
    st.session_state.input_mode = "vector"
    st.session_state.features = 2
    st.session_state.output_size = 2
    st.session_state.layers = [
        {"type": "linear", "params": {"out_features": 8, "activation": "ReLU"}},
        {"type": "linear", "params": {"out_features": 8, "activation": "ReLU"}},
    ]
    st.session_state.pop("lab_result", None)
    st.session_state.pop("lab_dataset", None)
    st.session_state.pop("lab_signature", None)


st.markdown(
    f'<div class="lab-hero"><h1>{TEXT["title"]}</h1><p>{TEXT["subtitle"]}</p></div>',
    unsafe_allow_html=True,
)

if st.button(TEXT["demo"], use_container_width=True):
    load_training_demo()
    st.rerun()

input_shape = current_input_shape()
output_size = int(st.session_state.output_size)
analysis = None
architecture_error = None
try:
    analysis = analyze_architecture(st.session_state.layers, input_shape, output_size)
except ShapeError as exc:
    architecture_error = str(exc)

st.subheader(TEXT["visual"])
if analysis is None:
    st.warning(f'{TEXT["no_model"]}\n\n{architecture_error or ""}')
else:
    render_network(analysis)

st.divider()
st.subheader(TEXT["training"])
st.caption(TEXT["synthetic_note"])

if output_size > 12:
    st.warning(TEXT["too_many_classes"])

train_disabled = analysis is None or output_size > 12

controls = st.columns(4)
with controls[0]:
    samples = st.slider(TEXT["samples"], 120, 600, 300, 30)
with controls[1]:
    epochs = st.slider(TEXT["epochs"], 10, 150, 50, 10)
with controls[2]:
    learning_rate = st.select_slider(
        TEXT["learning_rate"],
        options=[0.0005, 0.001, 0.003, 0.01, 0.03],
        value=0.01,
    )
with controls[3]:
    noise = st.slider(TEXT["noise"], 0.05, 0.80, 0.35, 0.05)

if st.session_state.input_mode == "vector":
    dataset_options = ["clusters"]
    if input_shape == (2,) and output_size == 2:
        dataset_options.append("xor")
    dataset_kind = st.selectbox(
        TEXT["dataset"],
        dataset_options,
        format_func=lambda value: (
            "مجموعات منفصلة" if AR and value == "clusters" else
            "XOR — تجربة غير خطية" if AR else
            "Groupes séparés" if value == "clusters" else
            "XOR — non linéaire"
        ),
    )
else:
    dataset_kind = "image_patterns"
    st.info(
        "سنولد صوراً صغيرة فيها إشارة مضيئة في أماكن مختلفة حسب الفئة."
        if AR
        else "Des images synthétiques contiendront un motif lumineux placé différemment pour chaque classe."
    )

if st.button(TEXT["train"], type="primary", use_container_width=True, disabled=train_disabled):
    try:
        if st.session_state.input_mode == "vector":
            dataset = make_vector_dataset(
                input_shape[0],
                output_size,
                samples=samples,
                noise=noise,
                kind=dataset_kind,
            )
        else:
            dataset = make_image_dataset(
                input_shape,
                output_size,
                samples=samples,
                noise=min(noise, 0.5),
            )

        with st.spinner("النموذج يتعلّم..." if AR else "Le modèle apprend..."):
            result = train_classifier(
                st.session_state.layers,
                input_shape,
                output_size,
                dataset,
                epochs=epochs,
                learning_rate=float(learning_rate),
            )
        st.session_state.lab_dataset = dataset
        st.session_state.lab_result = result
        st.session_state.lab_signature = architecture_signature(input_shape)
    except (ValueError, RuntimeError, ShapeError) as exc:
        st.error(str(exc))

signature = architecture_signature(input_shape)
result = st.session_state.get("lab_result")
dataset = st.session_state.get("lab_dataset")
result_is_current = result is not None and dataset is not None and st.session_state.get("lab_signature") == signature

if result_is_current:
    st.divider()
    st.subheader(TEXT["results"])

    final_loss = result.losses[-1]
    m1, m2, m3 = st.columns(3)
    m1.metric(TEXT["test_accuracy"], f"{100 * result.final_test_accuracy:.1f}%")
    m2.metric(TEXT["train_accuracy"], f"{100 * result.final_train_accuracy:.1f}%")
    m3.metric(TEXT["loss"], f"{final_loss:.4f}")

    history = pd.DataFrame(
        {
            "Epoch": range(1, len(result.losses) + 1),
            "Loss": result.losses,
            "Train accuracy": [100 * value for value in result.train_accuracies],
            "Test accuracy": [100 * value for value in result.test_accuracies],
        }
    )

    chart1, chart2 = st.columns(2)
    with chart1:
        st.caption("Loss ↓")
        st.line_chart(history, x="Epoch", y="Loss", height=300)
    with chart2:
        st.caption("Accuracy ↑")
        st.line_chart(history, x="Epoch", y=["Train accuracy", "Test accuracy"], height=300)

    model = build_torch_model(st.session_state.layers, input_shape, output_size)
    model.load_state_dict(result.state_dict)
    model.eval()

    if st.session_state.input_mode == "vector":
        with torch.no_grad():
            predictions = model(dataset.test_x).argmax(dim=1)

        if input_shape == (2,):
            all_x = torch.cat([dataset.train_x, dataset.test_x], dim=0)
            xmin, xmax = float(all_x[:, 0].min()) - 0.6, float(all_x[:, 0].max()) + 0.6
            ymin, ymax = float(all_x[:, 1].min()) - 0.6, float(all_x[:, 1].max()) + 0.6
            xs = torch.linspace(xmin, xmax, 42)
            ys = torch.linspace(ymin, ymax, 42)
            gx, gy = torch.meshgrid(xs, ys, indexing="xy")
            grid = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)
            with torch.no_grad():
                grid_prediction = model(grid).argmax(dim=1)

            grid_df = pd.DataFrame(
                {
                    "x1": grid[:, 0].numpy(),
                    "x2": grid[:, 1].numpy(),
                    "prediction": [str(int(v)) for v in grid_prediction],
                }
            )
            points_df = pd.DataFrame(
                {
                    "x1": dataset.test_x[:, 0].numpy(),
                    "x2": dataset.test_x[:, 1].numpy(),
                    "class": [str(int(v)) for v in dataset.test_y],
                }
            )
            v1, v2 = st.columns(2)
            with v1:
                st.caption(TEXT["true_data"])
                st.scatter_chart(points_df, x="x1", y="x2", color="class", height=360)
            with v2:
                st.caption(TEXT["model_view"])
                st.scatter_chart(grid_df, x="x1", y="x2", color="prediction", height=360)
        else:
            comparison = pd.DataFrame(
                {
                    "True": dataset.test_y[:40].numpy(),
                    "Prediction": predictions[:40].numpy(),
                }
            )
            st.dataframe(comparison, use_container_width=True, hide_index=True)
    else:
        with torch.no_grad():
            predictions = model(dataset.test_x).argmax(dim=1)
        count = min(6, len(dataset.test_x))
        image_cols = st.columns(3)
        for index in range(count):
            image = dataset.test_x[index].detach().cpu()
            if image.shape[0] == 1:
                view = image[0].numpy()
            else:
                view = image[:3].permute(1, 2, 0).numpy()
            minimum = float(view.min())
            maximum = float(view.max())
            view = (view - minimum) / (maximum - minimum + 1e-8)
            truth = int(dataset.test_y[index].item())
            predicted = int(predictions[index].item())
            caption = (
                f"الصحيح: {truth} | توقع: {predicted}"
                if AR
                else f"Vrai : {truth} | Prédit : {predicted}"
            )
            image_cols[index % 3].image(view, caption=caption)

    st.caption(TEXT["confusion"])
    labels = [f"→ {i}" for i in range(output_size)]
    index_labels = [f"{i} ↓" for i in range(output_size)]
    confusion_df = pd.DataFrame(
        result.confusion_matrix.numpy(),
        index=index_labels,
        columns=labels,
    )
    st.dataframe(confusion_df, use_container_width=True)

    if AR:
        st.info("راقب شيئين: من المفروض أن تنخفض Loss مع التدريب، وأن ترتفع Accuracy. جرّب تغيير عدد الطبقات ثم أعد التدريب وقارن النتيجة.")
    else:
        st.info("Observe deux choses : la Loss devrait diminuer et l'Accuracy augmenter. Modifie ensuite le réseau, réentraîne-le et compare.")
