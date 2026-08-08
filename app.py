from __future__ import annotations

import json

import streamlit as st

from model_builder import (
    ShapeError,
    analyze_architecture,
    analyze_layers,
    export_architecture_json,
    generate_model_code,
    import_architecture_json,
    validate_forward_pass,
)


st.set_page_config(
    page_title="AI Lab Junior | PyTorch Model Builder",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


COPY = {
    "ar": {
        "app_title": "🧠 AI Lab Junior",
        "tagline": "مختبر مبسّط لبناء الشبكات العصبية وفهم كيف يفكر النموذج خطوة بخطوة.",
        "builder": "مختبر البناء",
        "learn": "تعلّم",
        "challenges": "تحديات",
        "teacher": "ركن الأستاذ",
        "language": "لغة الواجهة",
        "preset": "ابدأ بمثال جاهز",
        "load": "تحميل المثال",
        "import": "استيراد مشروع JSON",
        "import_btn": "فتح المشروع",
        "input": "1) معطيات الدخول",
        "input_type": "نوع المعطيات",
        "vector": "أرقام / خصائص",
        "image": "صورة",
        "features": "عدد الخصائص",
        "channels": "القنوات",
        "height": "الارتفاع",
        "width": "العرض",
        "classes": "عدد الفئات التي نريد التنبؤ بها",
        "add_layer": "2) أضف طبقة",
        "layer_type": "نوع الطبقة",
        "add": "➕ أضف الطبقة",
        "architecture": "بنية الشبكة",
        "empty": "ابدأ بإضافة طبقة من القائمة الجانبية، أو حمّل مثالاً جاهزاً.",
        "valid": "البنية سليمة ويمكن تشغيلها.",
        "needs_fix": "البنية تحتاج تصحيحاً",
        "parameters": "المعاملات القابلة للتعلّم",
        "layers": "عدد المراحل",
        "output": "شكل الخرج",
        "test": "🧪 اختبر النموذج بعينة وهمية",
        "clear": "مسح الكل",
        "code": "كود PyTorch",
        "download_code": "تحميل كود Python",
        "download_json": "حفظ المشروع JSON",
        "explain_shape": "كل رقم بين الأقواس يصف شكل المعلومات بعد مرورها من الطبقة.",
        "output_layer": "طبقة القرار النهائية",
        "solution": "إظهار الحل",
    },
    "fr": {
        "app_title": "🧠 AI Lab Junior",
        "tagline": "Un laboratoire simple pour construire un réseau de neurones et suivre les données étape par étape.",
        "builder": "Construire",
        "learn": "Apprendre",
        "challenges": "Défis",
        "teacher": "Espace enseignant",
        "language": "Langue",
        "preset": "Exemple de départ",
        "load": "Charger l'exemple",
        "import": "Importer un projet JSON",
        "import_btn": "Ouvrir le projet",
        "input": "1) Entrée du modèle",
        "input_type": "Type de données",
        "vector": "Nombres / caractéristiques",
        "image": "Image",
        "features": "Nombre de caractéristiques",
        "channels": "Canaux",
        "height": "Hauteur",
        "width": "Largeur",
        "classes": "Nombre de classes à prédire",
        "add_layer": "2) Ajouter une couche",
        "layer_type": "Type de couche",
        "add": "➕ Ajouter",
        "architecture": "Architecture du réseau",
        "empty": "Ajoute une couche depuis la barre latérale ou charge un exemple.",
        "valid": "L'architecture est valide et exécutable.",
        "needs_fix": "L'architecture doit être corrigée",
        "parameters": "Paramètres entraînables",
        "layers": "Étapes",
        "output": "Forme de sortie",
        "test": "🧪 Tester avec une entrée fictive",
        "clear": "Tout effacer",
        "code": "Code PyTorch",
        "download_code": "Télécharger le code Python",
        "download_json": "Sauvegarder le projet JSON",
        "explain_shape": "Les nombres entre parenthèses décrivent la forme des données après chaque couche.",
        "output_layer": "Couche de décision finale",
        "solution": "Afficher la solution",
    },
}

LAYER_NAMES = {
    "ar": {
        "linear": "Fully Connected — طبقة كاملة الاتصال",
        "conv2d": "Conv2D — اكتشاف الأشكال في الصورة",
        "maxpool2d": "MaxPool2D — تصغير الصورة",
        "dropout": "Dropout — تقليل الحفظ",
        "flatten": "Flatten — تحويل الصورة إلى أرقام",
        "output": "Output — القرار النهائي",
    },
    "fr": {
        "linear": "Fully Connected — relier les informations",
        "conv2d": "Conv2D — détecter des motifs",
        "maxpool2d": "MaxPool2D — réduire l'image",
        "dropout": "Dropout — limiter la mémorisation",
        "flatten": "Flatten — transformer en vecteur",
        "output": "Output — décision finale",
    },
}

LAYER_HELP = {
    "ar": {
        "linear": "تجمع المعلومات السابقة وتتعلم كيف تمزجها لاتخاذ قرار.",
        "conv2d": "تبحث داخل الصورة عن حواف وأشكال صغيرة باستعمال مرشحات قابلة للتعلّم.",
        "maxpool2d": "تصغّر الخريطة مع الاحتفاظ بأقوى الإشارات.",
        "dropout": "تطفئ بعض الوصلات أثناء التدريب حتى لا يعتمد النموذج على مسار واحد فقط.",
        "flatten": "ترتب خرائط الصورة في لائحة واحدة من الأرقام قبل طبقات القرار.",
        "output": "تعطي درجة لكل فئة ممكنة. أثناء التدريب نتعلم أي درجة يجب أن تكون أكبر.",
    },
    "fr": {
        "linear": "Combine les informations précédentes pour apprendre une décision.",
        "conv2d": "Cherche des bords et petits motifs dans une image avec des filtres appris.",
        "maxpool2d": "Réduit la taille des cartes tout en gardant les signaux importants.",
        "dropout": "Désactive temporairement certaines connexions pendant l'entraînement.",
        "flatten": "Transforme les cartes d'une image en une seule liste de nombres.",
        "output": "Produit un score pour chaque classe possible.",
    },
}

PRESETS = {
    "—": None,
    "🔢 Mini classifieur": {
        "mode": "vector",
        "features": 8,
        "output_size": 3,
        "layers": [
            {"type": "linear", "params": {"out_features": 16, "activation": "ReLU"}},
            {"type": "dropout", "params": {"p": 0.2}},
            {"type": "linear", "params": {"out_features": 8, "activation": "ReLU"}},
        ],
    },
    "🖼️ Reconnaître des chiffres 28×28": {
        "mode": "image",
        "channels": 1,
        "height": 28,
        "width": 28,
        "output_size": 10,
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
            {"type": "linear", "params": {"out_features": 32, "activation": "ReLU"}},
        ],
    },
}


def tr(key: str) -> str:
    return COPY[st.session_state.get("lang", "ar")][key]


def shape_text(shape: tuple[int, ...]) -> str:
    return " × ".join(str(v) for v in shape)


def apply_project(mode: str, layers: list[dict], output_size: int, **dims: int) -> None:
    st.session_state.layers = layers
    st.session_state.input_mode = mode
    st.session_state.output_size = output_size
    for key, value in dims.items():
        st.session_state[key] = value


def rerun() -> None:
    st.rerun()


if "layers" not in st.session_state:
    st.session_state.layers = []
if "lang" not in st.session_state:
    st.session_state.lang = "ar"
if "input_mode" not in st.session_state:
    st.session_state.input_mode = "vector"
if "features" not in st.session_state:
    st.session_state.features = 8
if "channels" not in st.session_state:
    st.session_state.channels = 1
if "height" not in st.session_state:
    st.session_state.height = 28
if "width" not in st.session_state:
    st.session_state.width = 28
if "output_size" not in st.session_state:
    st.session_state.output_size = 3


st.markdown(
    """
    <style>
      .block-container {padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1200px;}
      .hero {padding: 1.1rem 1.25rem; border: 1px solid rgba(128,128,128,.22); border-radius: 18px; margin-bottom: 1rem;}
      .hero h1 {margin: 0 0 .25rem 0; font-size: 2rem;}
      .hero p {margin: 0; opacity: .82; font-size: 1.02rem;}
      .layer-card {border: 1px solid rgba(128,128,128,.22); border-radius: 14px; padding: .85rem 1rem; margin: .45rem 0;}
      .layer-title {font-weight: 700; font-size: 1.02rem; margin-bottom: .2rem;}
      .shape-pill {display: inline-block; padding: .18rem .5rem; border-radius: 999px; background: rgba(127,127,127,.12); font-family: monospace; margin-right: .35rem;}
      .muted {opacity: .72; font-size: .92rem;}
      .concept-card {border-left: 4px solid currentColor; padding: .8rem 1rem; background: rgba(127,127,127,.06); border-radius: 8px; margin-bottom: .7rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.selectbox(
        "Language / اللغة",
        options=["ar", "fr"],
        format_func=lambda x: "العربية" if x == "ar" else "Français",
        key="lang",
    )

    st.divider()
    st.subheader(tr("preset"))
    preset_name = st.selectbox(tr("preset"), list(PRESETS.keys()), label_visibility="collapsed")
    if st.button(tr("load"), use_container_width=True, disabled=PRESETS[preset_name] is None):
        preset = PRESETS[preset_name]
        assert preset is not None
        dims = {k: v for k, v in preset.items() if k in {"features", "channels", "height", "width"}}
        apply_project(preset["mode"], preset["layers"], preset["output_size"], **dims)
        rerun()

    uploaded = st.file_uploader(tr("import"), type=["json"])
    if uploaded is not None and st.button(tr("import_btn"), use_container_width=True):
        try:
            payload = import_architecture_json(uploaded.getvalue().decode("utf-8"))
            input_shape = payload["input_shape"]
            if len(input_shape) == 1:
                apply_project(
                    "vector",
                    payload["layers"],
                    payload["output_size"],
                    features=input_shape[0],
                )
            else:
                apply_project(
                    "image",
                    payload["layers"],
                    payload["output_size"],
                    channels=input_shape[0],
                    height=input_shape[1],
                    width=input_shape[2],
                )
            rerun()
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            st.error(str(exc))

    st.divider()
    st.subheader(tr("input"))
    st.radio(
        tr("input_type"),
        options=["vector", "image"],
        format_func=lambda x: tr(x),
        key="input_mode",
        horizontal=True,
    )

    if st.session_state.input_mode == "vector":
        st.number_input(tr("features"), min_value=1, max_value=10000, step=1, key="features")
        input_shape = (int(st.session_state.features),)
    else:
        c1, c2 = st.columns(2)
        c1.number_input(tr("channels"), min_value=1, max_value=4, step=1, key="channels")
        c2.number_input(tr("height"), min_value=4, max_value=512, step=1, key="height")
        st.number_input(tr("width"), min_value=4, max_value=512, step=1, key="width")
        input_shape = (
            int(st.session_state.channels),
            int(st.session_state.height),
            int(st.session_state.width),
        )

    st.number_input(tr("classes"), min_value=2, max_value=1000, step=1, key="output_size")

    st.divider()
    st.subheader(tr("add_layer"))
    layer_type = st.selectbox(
        tr("layer_type"),
        options=["linear", "conv2d", "maxpool2d", "dropout", "flatten"],
        format_func=lambda x: LAYER_NAMES[st.session_state.lang][x],
    )

    params: dict[str, int | float | str] = {}
    if layer_type == "linear":
        params["out_features"] = st.number_input("Neurons", min_value=1, max_value=4096, value=32, step=1)
        params["activation"] = st.selectbox("Activation", ["ReLU", "GELU", "Tanh", "Sigmoid", "None"])
    elif layer_type == "conv2d":
        params["out_channels"] = st.number_input("Filters", min_value=1, max_value=256, value=8, step=1)
        params["kernel_size"] = st.number_input("Kernel", min_value=1, max_value=11, value=3, step=1)
        params["stride"] = st.number_input("Stride", min_value=1, max_value=8, value=1, step=1)
        params["padding"] = st.number_input("Padding", min_value=0, max_value=8, value=1, step=1)
        params["activation"] = st.selectbox("Activation", ["ReLU", "GELU", "Tanh", "Sigmoid", "None"])
    elif layer_type == "maxpool2d":
        params["kernel_size"] = st.number_input("Pool size", min_value=1, max_value=8, value=2, step=1)
        params["stride"] = st.number_input("Pool stride", min_value=1, max_value=8, value=2, step=1)
    elif layer_type == "dropout":
        params["p"] = st.slider("Dropout", min_value=0.0, max_value=0.9, value=0.2, step=0.05)

    if st.button(tr("add"), type="primary", use_container_width=True):
        st.session_state.layers.append({"type": layer_type, "params": dict(params)})
        rerun()


st.markdown(
    f"""
    <div class="hero">
      <h1>{tr('app_title')}</h1>
      <p>{tr('tagline')}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

builder_tab, learn_tab, challenge_tab, teacher_tab = st.tabs(
    [tr("builder"), tr("learn"), tr("challenges"), tr("teacher")]
)


with builder_tab:
    st.subheader(tr("architecture"))
    st.caption(tr("explain_shape"))

    stack_analysis = None
    stack_error = None
    try:
        stack_analysis = analyze_layers(st.session_state.layers, input_shape)
    except ShapeError as exc:
        stack_error = str(exc)

    if not st.session_state.layers:
        st.info(tr("empty"))
    else:
        valid_reports = list(stack_analysis.reports) if stack_analysis else []
        for index, layer in enumerate(st.session_state.layers):
            report = valid_reports[index] if index < len(valid_reports) else None
            cols = st.columns([8, 1, 1, 1])
            with cols[0]:
                if report:
                    shape_markup = (
                        f'<span class="shape-pill">{shape_text(report.input_shape)}</span> → '
                        f'<span class="shape-pill">{shape_text(report.output_shape)}</span>'
                    )
                    param_text = f"{report.parameters:,} params"
                else:
                    shape_markup = '<span class="shape-pill">?</span>'
                    param_text = "—"
                st.markdown(
                    f"""
                    <div class="layer-card">
                      <div class="layer-title">{index + 1}. {LAYER_NAMES[st.session_state.lang][layer['type']]}</div>
                      <div>{shape_markup}</div>
                      <div class="muted">{LAYER_HELP[st.session_state.lang][layer['type']]} · {param_text}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            if cols[1].button("↑", key=f"up_{index}", disabled=index == 0):
                st.session_state.layers[index - 1], st.session_state.layers[index] = (
                    st.session_state.layers[index],
                    st.session_state.layers[index - 1],
                )
                rerun()
            if cols[2].button("↓", key=f"down_{index}", disabled=index == len(st.session_state.layers) - 1):
                st.session_state.layers[index + 1], st.session_state.layers[index] = (
                    st.session_state.layers[index],
                    st.session_state.layers[index + 1],
                )
                rerun()
            if cols[3].button("✕", key=f"del_{index}"):
                del st.session_state.layers[index]
                rerun()

    full_analysis = None
    full_error = stack_error
    if full_error is None:
        try:
            full_analysis = analyze_architecture(
                st.session_state.layers, input_shape, int(st.session_state.output_size)
            )
        except ShapeError as exc:
            full_error = str(exc)

    if full_analysis:
        output_report = full_analysis.reports[-1]
        st.markdown(
            f"""
            <div class="layer-card">
              <div class="layer-title">🎯 {tr('output_layer')}</div>
              <div><span class="shape-pill">{shape_text(output_report.input_shape)}</span> →
              <span class="shape-pill">{shape_text(output_report.output_shape)}</span></div>
              <div class="muted">{LAYER_HELP[st.session_state.lang]['output']} · {output_report.parameters:,} params</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.success(tr("valid"))
        m1, m2, m3 = st.columns(3)
        m1.metric(tr("parameters"), f"{full_analysis.total_parameters:,}")
        m2.metric(tr("layers"), len(full_analysis.reports))
        m3.metric(tr("output"), shape_text(full_analysis.output_shape))
    elif full_error:
        st.warning(f"{tr('needs_fix')}: {full_error}")

    action1, action2 = st.columns([2, 1])
    if action1.button(tr("test"), use_container_width=True, disabled=full_analysis is None):
        try:
            output_shape = validate_forward_pass(
                st.session_state.layers, input_shape, int(st.session_state.output_size)
            )
            st.success(f"PyTorch forward pass ✓  →  {output_shape}")
        except (ShapeError, RuntimeError) as exc:
            st.error(str(exc))
    if action2.button(tr("clear"), use_container_width=True):
        st.session_state.layers = []
        rerun()

    if full_analysis:
        st.divider()
        st.subheader(tr("code"))
        code = generate_model_code(
            st.session_state.layers, input_shape, int(st.session_state.output_size)
        )
        with st.expander(tr("code")):
            st.code(code, language="python")

        project_json = export_architecture_json(
            st.session_state.layers, input_shape, int(st.session_state.output_size)
        )
        d1, d2 = st.columns(2)
        d1.download_button(
            tr("download_code"),
            data=code,
            file_name="student_model.py",
            mime="text/x-python",
            use_container_width=True,
        )
        d2.download_button(
            tr("download_json"),
            data=project_json,
            file_name="ai_lab_project.json",
            mime="application/json",
            use_container_width=True,
        )


with learn_tab:
    if st.session_state.lang == "ar":
        st.subheader("الفكرة قبل الكود")
        st.markdown(
            """
            <div class="concept-card"><b>1. ما هي الشبكة العصبية؟</b><br>
            هي سلسلة من عمليات حسابية صغيرة. كل طبقة تستقبل أرقاماً، تغيّرها، ثم ترسل النتيجة للطبقة التالية.</div>
            <div class="concept-card"><b>2. ما معنى التدريب؟</b><br>
            في البداية تكون الأوزان شبه عشوائية. نقارن جواب النموذج بالجواب الصحيح ثم نعدّل الأوزان قليلاً. تكرار هذه العملية هو التعلم.</div>
            <div class="concept-card"><b>3. لماذا نهتم بشكل Tensor؟</b><br>
            لأن كل طبقة تنتظر شكلاً محدداً من المعلومات. الصورة مثلاً لها قنوات وارتفاع وعرض، بينما Fully Connected تنتظر لائحة أرقام.</div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("#### قاموس سريع")
        st.markdown(
            "- **Neuron:** عملية صغيرة تجمع أرقاماً وتنتج رقماً جديداً.\n"
            "- **Weight:** رقم يتعلمه النموذج ويحدد أهمية اتصال معين.\n"
            "- **Activation:** دالة تسمح للشبكة بتعلم علاقات غير خطية.\n"
            "- **Epoch:** مرور كامل على بيانات التدريب.\n"
            "- **Loss:** رقم يقيس مقدار خطأ النموذج."
        )
    else:
        st.subheader("Comprendre avant de coder")
        st.markdown(
            """
            <div class="concept-card"><b>1. Réseau de neurones</b><br>
            Une suite d'opérations mathématiques. Chaque couche reçoit des nombres, les transforme et passe le résultat à la suivante.</div>
            <div class="concept-card"><b>2. Entraînement</b><br>
            On compare la prédiction à la bonne réponse, puis on ajuste progressivement les poids du modèle.</div>
            <div class="concept-card"><b>3. Forme d'un tensor</b><br>
            Chaque couche attend une forme précise. Une image a canaux × hauteur × largeur; une couche Fully Connected attend un vecteur.</div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("#### Petit lexique")
        st.markdown(
            "- **Neurone :** petite opération qui combine plusieurs nombres.\n"
            "- **Poids :** valeur apprise qui règle l'importance d'une connexion.\n"
            "- **Activation :** fonction qui permet d'apprendre des relations complexes.\n"
            "- **Epoch :** un passage complet sur les données.\n"
            "- **Loss :** nombre qui mesure l'erreur du modèle."
        )


with challenge_tab:
    if st.session_state.lang == "ar":
        st.subheader("جرّب بنفسك")
        st.markdown("**التحدي 1:** لديك 6 خصائص وتريد تصنيفها إلى فئتين. ابنِ شبكة فيها طبقة مخفية من 12 عصبوناً.")
        with st.expander(tr("solution")):
            st.code("Input(6) → Linear(12) + ReLU → Output(2)")
        st.markdown("**التحدي 2:** صورة 28×28 تمر من Conv2D ثم MaxPool2D. ماذا تحتاج قبل Fully Connected؟")
        with st.expander(tr("solution")):
            st.write("Flatten، لأنها تحول خرائط الصورة إلى vector من الأرقام.")
        st.markdown("**التحدي 3:** جرّب Kernel أكبر من الصورة. اقرأ رسالة الخطأ ثم أصلح الإعدادات.")
    else:
        st.subheader("À toi de jouer")
        st.markdown("**Défi 1 :** 6 caractéristiques, 2 classes. Construis une couche cachée de 12 neurones.")
        with st.expander(tr("solution")):
            st.code("Input(6) → Linear(12) + ReLU → Output(2)")
        st.markdown("**Défi 2 :** après Conv2D et MaxPool2D, que faut-il ajouter avant Fully Connected ?")
        with st.expander(tr("solution")):
            st.write("Flatten, pour transformer les cartes de l'image en vecteur.")
        st.markdown("**Défi 3 :** choisis un kernel plus grand que l'image, lis l'erreur puis corrige le modèle.")


with teacher_tab:
    if st.session_state.lang == "ar":
        st.subheader("اقتراح حصة من 45 دقيقة")
        st.markdown(
            "1. **5 د:** ما الفرق بين برنامج بقواعد ثابتة ونموذج يتعلم من أمثلة؟\n"
            "2. **10 د:** افتح مثال Mini classifieur وتتبع shapes.\n"
            "3. **10 د:** غيّر عدد العصبونات ولاحظ عدد parameters.\n"
            "4. **10 د:** افتح مثال الصور واشرح Conv2D → Pool → Flatten.\n"
            "5. **10 د:** تحدي جماعي: أصلح architecture غير صالحة وفسّر سبب الخطأ."
        )
        st.info("الهدف هو فهم الفكرة والتجريب، وليس حفظ أسماء الطبقات أو كتابة الكود من الذاكرة.")
    else:
        st.subheader("Séance proposée — 45 minutes")
        st.markdown(
            "1. **5 min :** programme à règles fixes vs modèle qui apprend.\n"
            "2. **10 min :** explorer le Mini classifieur et suivre les formes.\n"
            "3. **10 min :** changer le nombre de neurones et observer les paramètres.\n"
            "4. **10 min :** explorer Conv2D → Pool → Flatten sur une image.\n"
            "5. **10 min :** corriger en groupe une architecture invalide et expliquer l'erreur."
        )
        st.info("L'objectif est de comprendre et expérimenter, pas de mémoriser la syntaxe PyTorch.")
