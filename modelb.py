import streamlit as st
import torch.nn as nn
import torch

# --- Session state for layers
if 'layers' not in st.session_state:
    st.session_state.layers = []

# --- Sidebar UI: Layer Creation & Model Input
st.sidebar.title("🛠️ Build a Layer")
st.sidebar.markdown("Add layers one at a time")

# Input/output size
st.sidebar.subheader("📏 Model Dimensions")
input_size = st.sidebar.number_input("Input size (e.g., 784 for MNIST)", value=784)
output_size = st.sidebar.number_input("Output size", value=10)

# Layer type selection
st.sidebar.subheader("➕ Add New Layer")
layer_type = st.sidebar.selectbox("Layer Type", ["Fully Connected", "Conv2D", "MaxPool2D", "Dropout", "Flatten"])
params = {}

if layer_type == "Fully Connected":
    params['out_features'] = st.sidebar.number_input("Output Features", min_value=1, value=128)
    params['activation'] = st.sidebar.selectbox("Activation", ["ReLU", "Sigmoid", "Tanh", "None"])

elif layer_type == "Conv2D":
    params['out_channels'] = st.sidebar.number_input("Out Channels", min_value=1, value=32)
    params['kernel_size'] = st.sidebar.number_input("Kernel Size", min_value=1, value=3)
    params['stride'] = st.sidebar.number_input("Stride", min_value=1, value=1)
    params['padding'] = st.sidebar.number_input("Padding", min_value=0, value=0)
    params['activation'] = st.sidebar.selectbox("Activation", ["ReLU", "Sigmoid", "Tanh", "None"])

elif layer_type == "MaxPool2D":
    params['kernel_size'] = st.sidebar.number_input("Kernel Size", min_value=1, value=2)
    params['stride'] = st.sidebar.number_input("Stride", min_value=1, value=2)

elif layer_type == "Dropout":
    params['p'] = st.sidebar.slider("Dropout Probability", 0.0, 1.0, 0.5)

elif layer_type == "Flatten":
    pass

# Add layer button
if st.sidebar.button("➕ Add Layer"):
    st.session_state.layers.append({"type": layer_type, "params": params})

# --- Main UI
st.title("🧠 Visual PyTorch Model Builder")
st.caption("Drag-and-drop (simulated) | Define input/output size | Generate full PyTorch model code")

st.subheader("📚 Model Layers")

# Display each layer
for i, layer in enumerate(st.session_state.layers):
    col1, col2, col3 = st.columns([6, 1, 1])
    with col1:
        st.markdown(f"**{i+1}. {layer['type']}** — `{layer['params']}`")
    with col2:
        if st.button("⬆️", key=f"up_{i}") and i > 0:
            st.session_state.layers[i - 1], st.session_state.layers[i] = st.session_state.layers[i], st.session_state.layers[i - 1]
    with col3:
        if st.button("⬇️", key=f"down_{i}") and i < len(st.session_state.layers) - 1:
            st.session_state.layers[i], st.session_state.layers[i + 1] = st.session_state.layers[i + 1], st.session_state.layers[i]

# Remove and clear controls
col_clear, col_remove = st.columns(2)
with col_clear:
    if st.button("🧹 Clear All Layers"):
        st.session_state.layers = []

with col_remove:
    if st.session_state.layers:
        index_to_remove = st.number_input("Remove layer #", min_value=1, max_value=len(st.session_state.layers), step=1, format="%d")
        if st.button("❌ Remove Selected Layer"):
            del st.session_state.layers[index_to_remove - 1]

# --- Build model as nn.Sequential
def build_model(layers, input_size):
    model = []
    in_features = input_size
    in_channels = 1

    for layer in layers:
        ltype = layer["type"]
        p = layer["params"]

        if ltype == "Fully Connected":
            model.append(nn.Linear(in_features, p['out_features']))
            in_features = p['out_features']
            if p['activation'] != "None":
                model.append(getattr(nn, p['activation'])())

        elif ltype == "Conv2D":
            model.append(nn.Conv2d(in_channels, p['out_channels'], kernel_size=p['kernel_size'],
                                   stride=p['stride'], padding=p['padding']))
            in_channels = p['out_channels']
            if p['activation'] != "None":
                model.append(getattr(nn, p['activation'])())

        elif ltype == "MaxPool2D":
            model.append(nn.MaxPool2d(kernel_size=p['kernel_size'], stride=p['stride']))

        elif ltype == "Dropout":
            model.append(nn.Dropout(p=p['p']))

        elif ltype == "Flatten":
            model.append(nn.Flatten())

    model.append(nn.Linear(in_features, output_size))
    return nn.Sequential(*model)

# --- Export PyTorch source code
def generate_model_code(layers, input_size, output_size):
    lines = [
        "import torch",
        "import torch.nn as nn\n",
        "class CustomModel(nn.Module):",
        "    def __init__(self):",
        "        super(CustomModel, self).__init__()",
        "        self.model = nn.Sequential(",
    ]
    
    indent = " " * 12
    in_features = input_size
    in_channels = 1

    for layer in layers:
        ltype = layer["type"]
        p = layer["params"]
        
        if ltype == "Flatten":
            lines.append(f"{indent}nn.Flatten(),")
        elif ltype == "Dropout":
            lines.append(f"{indent}nn.Dropout({p['p']}),")
        elif ltype == "Fully Connected":
            lines.append(f"{indent}nn.Linear({in_features}, {p['out_features']}),")
            in_features = p['out_features']
            if p['activation'] != "None":
                lines.append(f"{indent}nn.{p['activation']}(),")
        elif ltype == "Conv2D":
            lines.append(
                f"{indent}nn.Conv2d({in_channels}, {p['out_channels']}, kernel_size={p['kernel_size']}, stride={p['stride']}, padding={p['padding']}),"
            )
            in_channels = p['out_channels']
            if p['activation'] != "None":
                lines.append(f"{indent}nn.{p['activation']}(),")
        elif ltype == "MaxPool2D":
            lines.append(
                f"{indent}nn.MaxPool2d(kernel_size={p['kernel_size']}, stride={p['stride']}),"
            )

    lines.append(f"{indent}nn.Linear({in_features}, {output_size})")
    lines.append("        )\n")
    lines.append("    def forward(self, x):")
    lines.append("        return self.model(x)\n")

    return "\n".join(lines)

# --- Final build and export
if st.button("🚀 Build Model"):
    torch_model = build_model(st.session_state.layers, input_size)
    st.subheader("🔍 PyTorch `nn.Sequential` Structure")
    st.code(str(torch_model), language="python")

    code = generate_model_code(st.session_state.layers, input_size, output_size)
    st.subheader("📝 Exportable PyTorch Model Code")
    st.code(code, language="python")
    st.download_button("💾 Download Model Code", data=code, file_name="custom_model.py")
