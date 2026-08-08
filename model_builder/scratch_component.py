"""Scratch-like drag-and-drop builder for AI Lab Junior.

The component uses Streamlit Components v2 so the browser handles the visual
interaction while Python keeps the canonical project state for PyTorch.
"""

from __future__ import annotations

from typing import Any, Callable

import streamlit as st


HTML = """
<div class="scratch-root">
  <div class="scratch-app"></div>
</div>
"""


CSS = r"""
.scratch-root {
  font-family: var(--st-font, sans-serif);
  color: var(--st-text-color);
  width: 100%;
}

.scratch-toolbar {
  display: flex;
  flex-wrap: wrap;
  gap: .65rem;
  align-items: end;
  padding: .8rem;
  margin-bottom: .8rem;
  border: 1px solid color-mix(in srgb, var(--st-text-color) 16%, transparent);
  border-radius: 16px;
  background: color-mix(in srgb, var(--st-background-color) 94%, var(--st-primary-color) 6%);
}

.scratch-field {
  display: flex;
  flex-direction: column;
  gap: .2rem;
  min-width: 110px;
  flex: 1 1 110px;
}

.scratch-field label {
  font-size: .76rem;
  font-weight: 700;
  opacity: .72;
}

.scratch-field input,
.scratch-field select {
  width: 100%;
  box-sizing: border-box;
  border: 1px solid color-mix(in srgb, var(--st-text-color) 18%, transparent);
  background: var(--st-background-color);
  color: var(--st-text-color);
  border-radius: 10px;
  padding: .52rem .6rem;
  font: inherit;
}

.scratch-layout {
  display: grid;
  grid-template-columns: 220px minmax(300px, 1fr) 270px;
  gap: .8rem;
  align-items: start;
}

.scratch-panel {
  border: 1px solid color-mix(in srgb, var(--st-text-color) 16%, transparent);
  border-radius: 18px;
  background: color-mix(in srgb, var(--st-background-color) 97%, var(--st-text-color) 3%);
  overflow: hidden;
}

.scratch-panel-title {
  padding: .78rem .9rem;
  font-size: .84rem;
  font-weight: 800;
  border-bottom: 1px solid color-mix(in srgb, var(--st-text-color) 12%, transparent);
  opacity: .82;
}

.scratch-palette {
  padding: .75rem;
  display: flex;
  flex-direction: column;
  gap: .55rem;
}

.palette-block,
.model-block {
  position: relative;
  border: 0;
  border-radius: 12px;
  font: inherit;
  font-weight: 760;
  color: white;
  box-shadow: 0 3px 0 rgba(0,0,0,.16);
  user-select: none;
  -webkit-user-select: none;
}

.palette-block {
  padding: .7rem .75rem;
  cursor: grab;
  touch-action: none;
  text-align: start;
}

.palette-block:active { cursor: grabbing; transform: translateY(1px); box-shadow: 0 2px 0 rgba(0,0,0,.16); }
.palette-block small { display: block; opacity: .82; font-weight: 520; margin-top: .1rem; }

.kind-linear { background: #4c97ff; }
.kind-conv2d { background: #9966ff; }
.kind-maxpool2d { background: #59c059; }
.kind-flatten { background: #ffab19; }
.kind-dropout { background: #ff6680; }
.kind-input { background: #0f9d8a; }
.kind-output { background: #7b61a8; }

.workspace-wrap { min-width: 0; }
.workspace-lane {
  min-height: 410px;
  padding: 1rem;
  background-image: radial-gradient(color-mix(in srgb, var(--st-text-color) 14%, transparent) 1px, transparent 1px);
  background-size: 18px 18px;
  transition: background-color .12s ease, outline .12s ease;
}
.workspace-lane.drop-active {
  background-color: color-mix(in srgb, var(--st-primary-color) 9%, transparent);
  outline: 2px dashed var(--st-primary-color);
  outline-offset: -8px;
}

.fixed-block,
.model-block {
  max-width: 520px;
  margin: 0 auto;
}

.fixed-block {
  border-radius: 14px;
  padding: .72rem .85rem;
  color: white;
  box-shadow: 0 3px 0 rgba(0,0,0,.14);
}

.fixed-title { font-weight: 800; }
.fixed-sub { font-size: .8rem; opacity: .86; margin-top: .14rem; }

.chain-arrow {
  text-align: center;
  font-size: 1.4rem;
  line-height: 1.2;
  opacity: .42;
  margin: .22rem 0;
}

.model-block {
  display: grid;
  grid-template-columns: 34px 1fr 32px;
  align-items: center;
  gap: .45rem;
  padding: .7rem .65rem;
  cursor: pointer;
}
.model-block.selected { outline: 3px solid color-mix(in srgb, white 72%, transparent); outline-offset: -5px; }
.drag-handle {
  font-size: 1.25rem;
  line-height: 1;
  cursor: grab;
  touch-action: none;
  opacity: .88;
  text-align: center;
}
.drag-handle:active { cursor: grabbing; }
.block-main { min-width: 0; }
.block-title { font-weight: 820; }
.block-summary { font-size: .78rem; opacity: .86; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.delete-mini {
  width: 28px;
  height: 28px;
  padding: 0;
  border: 0;
  border-radius: 50%;
  background: rgba(0,0,0,.18);
  color: white;
  cursor: pointer;
  font-size: 1rem;
}

.empty-workspace {
  max-width: 520px;
  min-height: 110px;
  margin: .35rem auto;
  border: 2px dashed color-mix(in srgb, var(--st-text-color) 22%, transparent);
  border-radius: 16px;
  display: grid;
  place-items: center;
  padding: 1rem;
  text-align: center;
  opacity: .58;
  font-weight: 700;
}

.inspector-body { padding: .8rem; }
.inspector-help { font-size: .84rem; opacity: .68; line-height: 1.45; margin-bottom: .75rem; }
.inspector-fields { display: flex; flex-direction: column; gap: .62rem; }
.inspector-fields .scratch-field { min-width: 0; }
.inspector-actions { display: grid; grid-template-columns: 1fr auto; gap: .5rem; margin-top: .8rem; }
.apply-btn,
.remove-btn {
  border: 0;
  border-radius: 11px;
  padding: .58rem .7rem;
  font: inherit;
  font-weight: 800;
  cursor: pointer;
}
.apply-btn { background: var(--st-primary-color); color: white; }
.remove-btn { background: color-mix(in srgb, #e5484d 16%, transparent); color: #d33; }

.drag-ghost {
  position: fixed;
  z-index: 999999;
  pointer-events: none;
  transform: translate(-50%, -50%) rotate(2deg);
  padding: .6rem .8rem;
  border-radius: 12px;
  color: white;
  font-weight: 800;
  box-shadow: 0 10px 28px rgba(0,0,0,.24);
  opacity: .92;
  max-width: 230px;
}

.scratch-tip {
  margin-top: .65rem;
  padding: .65rem .75rem;
  border-radius: 12px;
  background: color-mix(in srgb, var(--st-primary-color) 8%, transparent);
  font-size: .82rem;
  line-height: 1.45;
}

@media (max-width: 900px) {
  .scratch-layout { grid-template-columns: 1fr; }
  .scratch-palette {
    display: grid;
    grid-template-columns: repeat(5, minmax(145px, 1fr));
    overflow-x: auto;
    overscroll-behavior-inline: contain;
  }
  .palette-block { min-width: 145px; }
  .workspace-lane { min-height: 330px; }
}

@media (max-width: 560px) {
  .scratch-toolbar { padding: .65rem; gap: .45rem; }
  .scratch-field { flex-basis: 45%; }
  .scratch-layout { gap: .65rem; }
  .scratch-panel { border-radius: 14px; }
  .workspace-lane { padding: .72rem; }
  .model-block { grid-template-columns: 30px 1fr 30px; padding: .65rem .5rem; }
  .block-title { font-size: .92rem; }
  .block-summary { font-size: .72rem; }
}
"""


JS = r"""
export default function({ parentElement, data, setStateValue }) {
  const app = parentElement.querySelector('.scratch-app');
  if (!app) return;

  const lang = data?.lang === 'fr' ? 'fr' : 'ar';
  const T = lang === 'ar' ? {
    palette: 'اللبنات', workspace: 'مساحة بناء الشبكة', inspector: 'إعدادات اللبنة',
    inputType: 'نوع المدخل', vector: 'أرقام / خصائص', image: 'صورة', features: 'الخصائص',
    channels: 'القنوات', height: 'الارتفاع', width: 'العرض', classes: 'الفئات',
    input: 'المدخلات', output: 'القرار النهائي', empty: 'اسحب لبنة إلى هنا أو اضغط عليها لإضافتها',
    tapTip: 'على الهاتف: اسحب اللبنة بإصبعك، أو اضغط عليها مرة لإضافتها مباشرة.',
    choose: 'اضغط على أي لبنة داخل الشبكة لتغيير إعداداتها.', apply: 'تطبيق', remove: 'حذف',
    neurons: 'Neurons', activation: 'Activation', filters: 'Filters', kernel: 'Kernel', stride: 'Stride',
    padding: 'Padding', pool: 'Pool size', probability: 'Dropout',
    linear: 'Fully Connected', conv2d: 'Conv2D', maxpool2d: 'MaxPool2D', flatten: 'Flatten', dropout: 'Dropout',
    linearHint: 'تجمع المعلومات وتتعلم قراراً.', convHint: 'تبحث عن أشكال صغيرة داخل الصورة.',
    poolHint: 'تصغّر الصورة وتحتفظ بالإشارات القوية.', flattenHint: 'تحول الصورة إلى لائحة أرقام.',
    dropoutHint: 'تطفئ بعض الوصلات مؤقتاً أثناء التدريب.'
  } : {
    palette: 'Blocs', workspace: 'Espace du réseau', inspector: 'Réglages du bloc',
    inputType: "Type d'entrée", vector: 'Nombres / caractéristiques', image: 'Image', features: 'Caractéristiques',
    channels: 'Canaux', height: 'Hauteur', width: 'Largeur', classes: 'Classes',
    input: 'Entrée', output: 'Décision finale', empty: 'Glisse un bloc ici ou touche-le pour l’ajouter',
    tapTip: 'Sur mobile : glisse avec le doigt, ou touche un bloc pour l’ajouter directement.',
    choose: 'Touche un bloc du réseau pour modifier ses réglages.', apply: 'Appliquer', remove: 'Supprimer',
    neurons: 'Neurones', activation: 'Activation', filters: 'Filtres', kernel: 'Kernel', stride: 'Stride',
    padding: 'Padding', pool: 'Taille du pool', probability: 'Dropout',
    linear: 'Fully Connected', conv2d: 'Conv2D', maxpool2d: 'MaxPool2D', flatten: 'Flatten', dropout: 'Dropout',
    linearHint: 'Combine les informations pour apprendre une décision.', convHint: 'Cherche de petits motifs dans une image.',
    poolHint: 'Réduit l’image en gardant les signaux importants.', flattenHint: 'Transforme l’image en liste de nombres.',
    dropoutHint: 'Désactive temporairement certaines connexions pendant l’entraînement.'
  };

  const fallback = {
    mode: 'vector', features: 2, channels: 1, height: 28, width: 28, output_size: 2, layers: []
  };
  let project = JSON.parse(JSON.stringify(data?.project ?? fallback));
  project.layers = Array.isArray(project.layers) ? project.layers : [];
  let selected = null;

  const meta = {
    linear: {name: T.linear, hint: T.linearHint, cls: 'kind-linear'},
    conv2d: {name: T.conv2d, hint: T.convHint, cls: 'kind-conv2d'},
    maxpool2d: {name: T.maxpool2d, hint: T.poolHint, cls: 'kind-maxpool2d'},
    flatten: {name: T.flatten, hint: T.flattenHint, cls: 'kind-flatten'},
    dropout: {name: T.dropout, hint: T.dropoutHint, cls: 'kind-dropout'}
  };

  const defaults = {
    linear: () => ({type: 'linear', params: {out_features: 16, activation: 'ReLU'}}),
    conv2d: () => ({type: 'conv2d', params: {out_channels: 8, kernel_size: 3, stride: 1, padding: 1, activation: 'ReLU'}}),
    maxpool2d: () => ({type: 'maxpool2d', params: {kernel_size: 2, stride: 2}}),
    flatten: () => ({type: 'flatten', params: {}}),
    dropout: () => ({type: 'dropout', params: {p: 0.2}})
  };

  function safeInt(v, fallbackValue, min=1) {
    const n = Number.parseInt(v, 10);
    return Number.isFinite(n) ? Math.max(min, n) : fallbackValue;
  }
  function safeFloat(v, fallbackValue, min=0, max=1) {
    const n = Number.parseFloat(v);
    return Number.isFinite(n) ? Math.max(min, Math.min(max, n)) : fallbackValue;
  }
  function clone(value) { return JSON.parse(JSON.stringify(value)); }
  function commit() { setStateValue('project', clone(project)); }

  function inputSummary() {
    if (project.mode === 'image') return `${project.channels} × ${project.height} × ${project.width}`;
    return `${project.features}`;
  }

  function layerSummary(layer) {
    const p = layer.params || {};
    if (layer.type === 'linear') return `${p.out_features ?? 16} ${T.neurons} · ${p.activation ?? 'ReLU'}`;
    if (layer.type === 'conv2d') return `${p.out_channels ?? 8} ${T.filters} · K${p.kernel_size ?? 3} · S${p.stride ?? 1} · ${p.activation ?? 'ReLU'}`;
    if (layer.type === 'maxpool2d') return `${T.pool}: ${p.kernel_size ?? 2} · S${p.stride ?? 2}`;
    if (layer.type === 'dropout') return `p = ${p.p ?? 0.2}`;
    return 'Tensor → Vector';
  }

  function field(label, key, value, type='number', attrs='') {
    return `<div class="scratch-field"><label>${label}</label><input data-param="${key}" type="${type}" value="${value}" ${attrs}></div>`;
  }
  function selectField(label, key, value, options) {
    return `<div class="scratch-field"><label>${label}</label><select data-param="${key}">${options.map(v => `<option value="${v}" ${v === value ? 'selected' : ''}>${v}</option>`).join('')}</select></div>`;
  }

  function inspectorHtml() {
    if (selected === null || !project.layers[selected]) {
      return `<div class="inspector-body"><div class="inspector-help">${T.choose}</div></div>`;
    }
    const layer = project.layers[selected];
    const p = layer.params || {};
    let fields = '';
    if (layer.type === 'linear') {
      fields += field(T.neurons, 'out_features', p.out_features ?? 16, 'number', 'min="1" max="4096" step="1"');
      fields += selectField(T.activation, 'activation', p.activation ?? 'ReLU', ['ReLU','GELU','Tanh','Sigmoid','None']);
    } else if (layer.type === 'conv2d') {
      fields += field(T.filters, 'out_channels', p.out_channels ?? 8, 'number', 'min="1" max="256" step="1"');
      fields += field(T.kernel, 'kernel_size', p.kernel_size ?? 3, 'number', 'min="1" max="11" step="1"');
      fields += field(T.stride, 'stride', p.stride ?? 1, 'number', 'min="1" max="8" step="1"');
      fields += field(T.padding, 'padding', p.padding ?? 1, 'number', 'min="0" max="8" step="1"');
      fields += selectField(T.activation, 'activation', p.activation ?? 'ReLU', ['ReLU','GELU','Tanh','Sigmoid','None']);
    } else if (layer.type === 'maxpool2d') {
      fields += field(T.pool, 'kernel_size', p.kernel_size ?? 2, 'number', 'min="1" max="8" step="1"');
      fields += field(T.stride, 'stride', p.stride ?? 2, 'number', 'min="1" max="8" step="1"');
    } else if (layer.type === 'dropout') {
      fields += field(T.probability, 'p', p.p ?? 0.2, 'number', 'min="0" max="0.9" step="0.05"');
    } else {
      fields += `<div class="inspector-help">${meta[layer.type].hint}</div>`;
    }
    return `<div class="inspector-body">
      <div class="inspector-help"><b>${meta[layer.type].name}</b><br>${meta[layer.type].hint}</div>
      <div class="inspector-fields">${fields}</div>
      <div class="inspector-actions"><button class="apply-btn" type="button">${T.apply}</button><button class="remove-btn" type="button">🗑 ${T.remove}</button></div>
    </div>`;
  }

  function toolbarHtml() {
    const imageFields = project.mode === 'image' ? `
      <div class="scratch-field"><label>${T.channels}</label><input id="cfg-channels" type="number" min="1" max="4" value="${project.channels ?? 1}"></div>
      <div class="scratch-field"><label>${T.height}</label><input id="cfg-height" type="number" min="4" max="256" value="${project.height ?? 28}"></div>
      <div class="scratch-field"><label>${T.width}</label><input id="cfg-width" type="number" min="4" max="256" value="${project.width ?? 28}"></div>` : `
      <div class="scratch-field"><label>${T.features}</label><input id="cfg-features" type="number" min="1" max="4096" value="${project.features ?? 2}"></div>`;
    return `<div class="scratch-toolbar">
      <div class="scratch-field"><label>${T.inputType}</label><select id="cfg-mode"><option value="vector" ${project.mode !== 'image' ? 'selected' : ''}>${T.vector}</option><option value="image" ${project.mode === 'image' ? 'selected' : ''}>${T.image}</option></select></div>
      ${imageFields}
      <div class="scratch-field"><label>${T.classes}</label><input id="cfg-output" type="number" min="2" max="100" value="${project.output_size ?? 2}"></div>
    </div>`;
  }

  function workspaceHtml() {
    let chain = `<div class="fixed-block kind-input"><div class="fixed-title">📥 ${T.input}</div><div class="fixed-sub">${inputSummary()}</div></div>`;
    if (!project.layers.length) {
      chain += `<div class="chain-arrow">↓</div><div class="empty-workspace">${T.empty}</div>`;
    } else {
      project.layers.forEach((layer, index) => {
        const m = meta[layer.type] ?? meta.linear;
        chain += `<div class="chain-arrow">↓</div><div class="model-block ${m.cls} ${selected === index ? 'selected' : ''}" data-index="${index}">
          <div class="drag-handle" title="drag">⋮⋮</div>
          <div class="block-main"><div class="block-title">${m.name}</div><div class="block-summary">${layerSummary(layer)}</div></div>
          <button class="delete-mini" type="button" aria-label="delete">×</button>
        </div>`;
      });
    }
    chain += `<div class="chain-arrow">↓</div><div class="fixed-block kind-output"><div class="fixed-title">🎯 ${T.output}</div><div class="fixed-sub">${project.output_size ?? 2} ${T.classes}</div></div>`;
    return chain;
  }

  function render() {
    app.innerHTML = `${toolbarHtml()}<div class="scratch-layout">
      <section class="scratch-panel"><div class="scratch-panel-title">🧩 ${T.palette}</div><div class="scratch-palette">
        ${['linear','conv2d','maxpool2d','flatten','dropout'].map(kind => `<div class="palette-block ${meta[kind].cls}" data-kind="${kind}">${meta[kind].name}<small>${meta[kind].hint}</small></div>`).join('')}
      </div><div class="scratch-tip">${T.tapTip}</div></section>
      <section class="scratch-panel workspace-wrap"><div class="scratch-panel-title">🧠 ${T.workspace}</div><div class="workspace-lane">${workspaceHtml()}</div></section>
      <section class="scratch-panel"><div class="scratch-panel-title">⚙️ ${T.inspector}</div><div class="inspector-slot">${inspectorHtml()}</div></section>
    </div>`;
    bindToolbar();
    bindPalette();
    bindWorkspace();
    bindInspector();
  }

  function bindToolbar() {
    const mode = app.querySelector('#cfg-mode');
    if (mode) mode.onchange = () => { project.mode = mode.value; commit(); };
    const features = app.querySelector('#cfg-features');
    if (features) features.onchange = () => { project.features = safeInt(features.value, 2, 1); commit(); };
    const channels = app.querySelector('#cfg-channels');
    if (channels) channels.onchange = () => { project.channels = safeInt(channels.value, 1, 1); commit(); };
    const height = app.querySelector('#cfg-height');
    if (height) height.onchange = () => { project.height = safeInt(height.value, 28, 4); commit(); };
    const width = app.querySelector('#cfg-width');
    if (width) width.onchange = () => { project.width = safeInt(width.value, 28, 4); commit(); };
    const output = app.querySelector('#cfg-output');
    if (output) output.onchange = () => { project.output_size = safeInt(output.value, 2, 2); commit(); };
  }

  function indexForPointer(y) {
    const blocks = [...app.querySelectorAll('.model-block')];
    for (let i = 0; i < blocks.length; i++) {
      const r = blocks[i].getBoundingClientRect();
      if (y < r.top + r.height / 2) return i;
    }
    return blocks.length;
  }

  function makeGhost(text, cls, x, y) {
    const ghost = document.createElement('div');
    ghost.className = `drag-ghost ${cls}`;
    ghost.textContent = text;
    ghost.style.left = `${x}px`;
    ghost.style.top = `${y}px`;
    document.body.appendChild(ghost);
    return ghost;
  }

  function pointInWorkspace(x, y) {
    const lane = app.querySelector('.workspace-lane');
    const r = lane.getBoundingClientRect();
    return x >= r.left && x <= r.right && y >= r.top && y <= r.bottom;
  }

  function bindPalette() {
    app.querySelectorAll('.palette-block').forEach(el => {
      const kind = el.dataset.kind;
      let startX = 0, startY = 0, moved = false, ghost = null, activeId = null;
      el.onpointerdown = e => {
        if (e.pointerType === 'mouse' && e.button !== 0) return;
        e.preventDefault();
        activeId = e.pointerId;
        startX = e.clientX; startY = e.clientY; moved = false;
        ghost = makeGhost(meta[kind].name, meta[kind].cls, e.clientX, e.clientY);
        try { el.setPointerCapture(activeId); } catch (_) {}
      };
      el.onpointermove = e => {
        if (activeId !== e.pointerId || !ghost) return;
        const dist = Math.hypot(e.clientX - startX, e.clientY - startY);
        if (dist > 7) moved = true;
        ghost.style.left = `${e.clientX}px`; ghost.style.top = `${e.clientY}px`;
        app.querySelector('.workspace-lane')?.classList.toggle('drop-active', pointInWorkspace(e.clientX, e.clientY));
      };
      const finish = e => {
        if (activeId !== e.pointerId) return;
        const inside = pointInWorkspace(e.clientX, e.clientY);
        if (inside) {
          const at = indexForPointer(e.clientY);
          project.layers.splice(at, 0, defaults[kind]());
          commit();
        } else if (!moved) {
          project.layers.push(defaults[kind]());
          commit();
        }
        ghost?.remove(); ghost = null; activeId = null;
        app.querySelector('.workspace-lane')?.classList.remove('drop-active');
      };
      el.onpointerup = finish;
      el.onpointercancel = e => { ghost?.remove(); ghost = null; activeId = null; app.querySelector('.workspace-lane')?.classList.remove('drop-active'); };
    });
  }

  function bindWorkspace() {
    app.querySelectorAll('.model-block').forEach(block => {
      const index = Number(block.dataset.index);
      block.onclick = e => {
        if (e.target.closest('.delete-mini') || e.target.closest('.drag-handle')) return;
        selected = index;
        render();
      };
      block.querySelector('.delete-mini').onclick = e => {
        e.stopPropagation();
        project.layers.splice(index, 1);
        selected = null;
        commit();
      };

      const handle = block.querySelector('.drag-handle');
      let activeId = null, ghost = null;
      handle.onpointerdown = e => {
        if (e.pointerType === 'mouse' && e.button !== 0) return;
        e.preventDefault(); e.stopPropagation();
        activeId = e.pointerId;
        const layer = project.layers[index];
        ghost = makeGhost(meta[layer.type].name, meta[layer.type].cls, e.clientX, e.clientY);
        try { handle.setPointerCapture(activeId); } catch (_) {}
      };
      handle.onpointermove = e => {
        if (activeId !== e.pointerId || !ghost) return;
        ghost.style.left = `${e.clientX}px`; ghost.style.top = `${e.clientY}px`;
        app.querySelector('.workspace-lane')?.classList.toggle('drop-active', pointInWorkspace(e.clientX, e.clientY));
      };
      handle.onpointerup = e => {
        if (activeId !== e.pointerId) return;
        if (pointInWorkspace(e.clientX, e.clientY)) {
          let target = indexForPointer(e.clientY);
          const [movedLayer] = project.layers.splice(index, 1);
          if (target > index) target -= 1;
          target = Math.max(0, Math.min(project.layers.length, target));
          project.layers.splice(target, 0, movedLayer);
          commit();
        }
        ghost?.remove(); ghost = null; activeId = null;
        app.querySelector('.workspace-lane')?.classList.remove('drop-active');
      };
      handle.onpointercancel = () => { ghost?.remove(); ghost = null; activeId = null; app.querySelector('.workspace-lane')?.classList.remove('drop-active'); };
    });
  }

  function bindInspector() {
    if (selected === null || !project.layers[selected]) return;
    const apply = app.querySelector('.apply-btn');
    const remove = app.querySelector('.remove-btn');
    if (apply) apply.onclick = () => {
      const layer = project.layers[selected];
      const p = {...(layer.params || {})};
      app.querySelectorAll('.inspector-fields [data-param]').forEach(input => {
        const key = input.dataset.param;
        if (key === 'activation') p[key] = input.value;
        else if (key === 'p') p[key] = safeFloat(input.value, 0.2, 0, 0.9);
        else if (key === 'padding') p[key] = safeInt(input.value, 0, 0);
        else p[key] = safeInt(input.value, 1, 1);
      });
      layer.params = p;
      commit();
    };
    if (remove) remove.onclick = () => {
      project.layers.splice(selected, 1);
      selected = null;
      commit();
    };
  }

  render();
}
"""


_scratch_component = st.components.v2.component(
    "ai_lab_scratch_builder",
    html=HTML,
    css=CSS,
    js=JS,
)


def render_scratch_builder(
    project: dict[str, Any],
    *,
    lang: str = "ar",
    key: str = "scratch_canvas",
    on_change: Callable[[], None] | None = None,
):
    """Render the Scratch-like editor and return its component state."""

    callback = on_change or (lambda: None)
    return _scratch_component(
        data={"project": project, "lang": lang},
        default={"project": project},
        key=key,
        on_project_change=callback,
    )
