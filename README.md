# AI Lab Junior — Visual PyTorch Model Builder

An educational, browser-based neural-network builder designed for **middle-school learners**. Students build small PyTorch architectures, follow tensor shapes step by step, discover why an architecture is valid or invalid, and export the resulting Python code.

> **الهدف التربوي:** نخلي التلميذ يشوف كيفاش كتتحول المعطيات داخل الشبكة العصبية، يجرب، يغلط، ويفهم سبب الخطأ قبل ما نهتمو بكتابة الكود.

## What students can do

- Build vector classifiers and simple image classifiers without writing code first.
- Add `Linear`, `Conv2D`, `MaxPool2D`, `Dropout`, and `Flatten` layers.
- See the **input shape → output shape** after every layer.
- See how many trainable parameters each layer introduces.
- Get a clear explanation when two layers are incompatible.
- Test the architecture with a synthetic PyTorch forward pass.
- Start from classroom-friendly presets.
- Export a standalone `student_model.py` file.
- Save and reload projects as JSON.
- Use the interface in **Arabic or French**.

## Why this version is safer for learning

The original prototype generated model code but did not propagate CNN spatial dimensions. That could produce a `Linear` layer with the wrong number of inputs after `Conv2D`, pooling, and `Flatten`.

The educational edition moves architecture reasoning into a tested core engine. It validates each transition before code generation, so a student sees an explanatory error instead of learning from a silently invalid architecture.

## Classroom flow

A suggested 45-minute session is included directly in the **Teacher** tab:

1. Compare fixed rules with learning from examples.
2. Explore a tiny vector classifier.
3. Change the number of neurons and observe parameter counts.
4. Explore `Conv2D → MaxPool2D → Flatten` on a 28×28 image.
5. Fix an intentionally invalid architecture and explain the error.

The focus is conceptual understanding, experimentation, and vocabulary—not memorizing PyTorch syntax.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the Streamlit URL shown in your terminal.

## Run tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

The tests cover vector networks, CNN shape inference, invalid layer transitions, JSON round-trips, generated-code syntax, and real PyTorch forward passes.

## Project structure

```text
.
├── app.py                       # Streamlit educational interface
├── model_builder/
│   ├── __init__.py
│   └── core.py                  # validation, shape inference, PyTorch export
├── tests/
│   └── test_core.py
├── .streamlit/config.toml       # classroom-friendly theme
├── .devcontainer/devcontainer.json
├── .github/workflows/tests.yml
├── requirements.txt
└── requirements-dev.txt
```

## Educational design principles

- **Immediate feedback:** errors explain what shape a layer expected and what it received.
- **Visible state:** students can follow tensor shapes rather than treating the model as a black box.
- **Small steps:** the default examples are intentionally compact.
- **No student data required:** the app uses synthetic inputs for architecture validation and does not need names, accounts, or uploaded datasets.
- **Code comes second:** students can inspect/export PyTorch after they understand the architecture.

## Current scope

This is an architecture-learning tool, not a full training platform. It does not yet train models on datasets or visualize gradients. That separation keeps the first classroom experience fast and focused.

Possible next modules:

- interactive training on tiny built-in datasets;
- loss and accuracy visualization;
- confusion-matrix activities;
- neuron/weight animations;
- teacher-created challenges and printable worksheets;
- additional language support.

## Technical references

- [PyTorch `nn.Linear`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html)
- [PyTorch `nn.Conv2d`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
- [PyTorch `nn.MaxPool2d`](https://docs.pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html)
- [PyTorch `nn.Flatten`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Flatten.html)
- [Streamlit documentation](https://docs.streamlit.io/)
