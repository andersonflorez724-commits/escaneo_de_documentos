/**
 * Controlador de la interfaz del escaner.
 *
 * Une la captura por camara (o la carga de un archivo) con el envio al
 * backend y el pintado del resultado en pantalla.
 *
 * El documento se captura **por caras**, siempre en este orden:
 *
 * 1. **Cara frontal** — numero, apellidos, nombres y nacionalidad.
 * 2. **Reverso** — fecha de nacimiento y fecha de vencimiento.
 *
 * La camara previsualiza en vivo y, cuando `DocumentDetector` considera que el
 * documento esta bien colocado y quieto, dispara la captura sola. Al procesar
 * se envian ambas caras (`file` y `back_file`) para que el backend fusione el
 * texto antes de extraer los campos.
 */

import { CameraController, CameraError } from './camera.js';
import { DetectorStatus, DocumentDetector } from './detector.js';

/*
 * Arranque al final del modulo: los metodos de `ScannerUI.prototype` se
 * asignan a continuacion, y esas asignaciones no se elevan. Si se invocara
 * `init()` aqui arriba, el metodo todavia no existiria y no se registraria
 * ningun evento (ni la camara ni las pestanas).
 */

const SIDES = ['front', 'back'];

const SIDE_TITLES = {
  front: 'cara frontal',
  back: 'reverso',
};

const SIDE_STEPS = {
  front: 1,
  back: 2,
};

const SIDE_HINTS = {
  front:
    '<strong>Paso 1 · Cara frontal</strong>: toma primero el anverso — ' +
    'numero de documento, apellidos, nombres y nacionalidad.',
  back:
    '<strong>Paso 2 · Reverso</strong>: despues gira el documento — ' +
    'fecha de nacimiento y fecha de vencimiento (caducidad).',
};

/** Enfriamiento tras una captura automatica para no repetir fotogramas. */
const AUTO_CAPTURE_COOLDOWN_MS = 1400;

function ScannerUI(element) {
  this.root = element;
  this.scanUrl = element.dataset.scanUrl;
  this.maxUploadMb = Number(element.dataset.maxUploadMb || 8);
  this.camera = null;
  this.detector = null;
  this.activeSide = 'front';
  this.files = { front: null, back: null };
  this.previews = { front: null, back: null };
  this.meta = { front: '', back: '' };
  this.lastResult = null;
  this.busy = false;
  this.toastTimer = null;
  this.autoCapture = true;
  this.captureLockUntil = 0;
}

/* ------------------------------------------------------------------ init */
ScannerUI.prototype.init = function init() {
  this.el = {
    video: document.getElementById('camera-video'),
    shell: document.getElementById('camera-shell'),
    placeholder: document.getElementById('camera-placeholder'),
    status: document.getElementById('camera-status'),
    statusText: document.getElementById('camera-status-text'),

    btnStart: document.getElementById('btn-start'),
    btnStop: document.getElementById('btn-stop'),
    btnSwitch: document.getElementById('btn-switch'),
    btnCapture: document.getElementById('btn-capture'),
    cameraSelect: document.getElementById('camera-select'),

    sideHint: document.getElementById('side-hint'),
    sideCards: Array.from(this.root.querySelectorAll('.side-card[data-side]')),
    stepBanner: document.getElementById('step-banner'),

    fileInput: document.getElementById('file-input'),
    dropzone: document.getElementById('dropzone'),
    dropzoneLabel: document.getElementById('dropzone-label'),
    fileName: document.getElementById('file-name'),

    previewBlock: document.getElementById('preview-block'),
    previewMeta: document.getElementById('preview-meta'),
    capturesTrack: document.getElementById('captures-track'),
    capturesBlock: document.getElementById('captures-block'),
    btnProcess: document.getElementById('btn-process'),
    btnDiscard: document.getElementById('btn-discard'),

    detectOverlay: document.getElementById('detect-overlay'),
    detectText: document.getElementById('detect-text'),
    detectBar: document.getElementById('detect-bar'),
    toggleAuto: document.getElementById('toggle-auto'),

    progress: document.getElementById('progress'),
    progressBar: document.getElementById('progress-bar'),
    error: document.getElementById('capture-error'),
    btnCopy: document.getElementById('btn-copy'),
    toast: document.getElementById('toast'),

    resultsEmpty: document.getElementById('results-empty'),
    resultsLoading: document.getElementById('results-loading'),
    resultsContent: document.getElementById('results-content'),
  };

  // Elementos de cada cara: estado, miniatura y boton de quitar.
  this.el.sideStates = {};
  this.el.previewItems = {};
  this.el.previewImages = {};
  this.el.previewRemove = {};
  SIDES.forEach((side) => {
    this.el.sideStates[side] = document.getElementById(`side-state-${side}`);
    this.el.previewItems[side] = document.getElementById(`preview-item-${side}`);
    this.el.previewImages[side] = document.getElementById(`preview-${side}`);
    this.el.previewRemove[side] = this.root.querySelector(`[data-remove-side="${side}"]`);
  });

  this.setupTabs();
  this.bindSides();
  if (CameraController.isSupported()) {
    this.camera = new CameraController(this.el.video, { onChange: this.onCameraChange.bind(this) });
    this.detector = new DocumentDetector(this.el.video, {
      onUpdate: this.onDetectUpdate.bind(this),
      onReady: this.onAutoCaptureReady.bind(this),
    });
    this.bindCamera();
    this.bindAutoCapture();
  } else {
    this.el.btnStart.disabled = true;
    this.el.btnStart.textContent = 'Camara no disponible';
    this.showError('Este navegador no soporta la captura de camara. Usa la pestana "Subir archivo".');
  }

  this.bindUpload();
  this.bindActions();
  this.bindShortcuts();
  this.setActiveSide('front');
  this.refreshCaptures();

  document.addEventListener('visibilitychange', () => {
    if (document.hidden && this.camera) this.camera.stop();
  });
};

/** Atajos de teclado para agilizar la operacion en recepcion. */
ScannerUI.prototype.bindShortcuts = function bindShortcuts() {
  document.addEventListener('keydown', (event) => {
    const target = event.target;
    const typing = target instanceof HTMLElement &&
      (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable);
    if (typing || event.repeat) return;

    if (event.code === 'Space' && this.camera?.isActive) {
      event.preventDefault();
      this.captureFrame({ manual: true });
      return;
    }

    if (event.key === 'Escape') {
      if (!this.hasFiles()) return;
      event.preventDefault();
      this.discard(this.files[this.activeSide] ? this.activeSide : null);
      return;
    }

    if (event.key === 'Enter' && this.hasFiles() && !this.busy && document.activeElement !== this.el.btnProcess) {
      event.preventDefault();
      this.process();
    }
  });
};

ScannerUI.prototype.setupTabs = function setupTabs() {
  const tabs = this.root.querySelectorAll('[data-tab]');
  const panels = this.root.querySelectorAll('[data-panel]');

  tabs.forEach((tab) => {
    tab.addEventListener('click', () => {
      tabs.forEach((item) => {
        const selected = item === tab;
        item.classList.toggle('is-active', selected);
        item.setAttribute('aria-selected', String(selected));
      });

      panels.forEach((panel) => {
        panel.classList.toggle('hidden', panel.dataset.panel !== tab.dataset.tab);
      });

      if (tab.dataset.tab !== 'camera' && this.camera) this.camera.stop();
    });
  });
};

/* ----------------------------------------------------------- caras */
ScannerUI.prototype.bindSides = function bindSides() {
  this.el.sideCards.forEach((card) => {
    card.addEventListener('click', () => this.setActiveSide(card.dataset.side));
  });

  SIDES.forEach((side) => {
    const button = this.el.previewRemove[side];
    if (button) button.addEventListener('click', () => this.discard(side));
  });
};

/** Marca la cara que se esta capturando y actualiza los textos de ayuda. */
ScannerUI.prototype.setActiveSide = function setActiveSide(side) {
  if (!SIDES.includes(side)) return;

  this.activeSide = side;
  this.el.sideCards.forEach((card) => {
    const selected = card.dataset.side === side;
    card.classList.toggle('is-active', selected);
    card.setAttribute('aria-pressed', String(selected));
  });

  this.el.sideHint.innerHTML = SIDE_HINTS[side] ?? '';
  this.el.dropzoneLabel.textContent = this.files.front && this.files.back
    ? 'Ambas caras listas · puedes cambiar cualquier foto'
    : `Selecciona o arrastra la foto del ${SIDE_TITLES[side]}`;
  this.el.fileName.textContent = '';

  this.refreshStepBanner();

  // Reinicia la deteccion para la cara nueva (si la camara esta viva).
  if (this.detector && this.camera?.isActive) {
    this.detector.resetStable();
    if (this.autoCapture && !this.files[side]) this.detector.resume();
    else this.detector.pause();
  }

  if (this.el.previewBlock.classList.contains('hidden')) return;
  this.refreshPreview();
};

/**
 * Banner de progreso: indica siempre cual cara tomar primero.
 * Tras capturar la frontal el propio `setSideFile` salta al reverso.
 */
ScannerUI.prototype.refreshStepBanner = function refreshStepBanner() {
  if (!this.el.stepBanner) return;

  const doneFront = Boolean(this.files.front);
  const doneBack = Boolean(this.files.back);
  const step = SIDE_STEPS[this.activeSide] ?? 1;

  let text;
  if (doneFront && doneBack) {
    text = 'Ambas caras listas · pulsa <strong>Procesar documento</strong>.';
  } else if (!doneFront) {
    text = `<strong>Paso 1 de 2</strong> · Toma primero la <strong>cara frontal</strong> (numero, apellidos, nombres, nacionalidad).`;
  } else {
    text = `<strong>Paso 2 de 2</strong> · Ahora el <strong>reverso</strong> (fecha de nacimiento y vencimiento).`;
  }

  this.el.stepBanner.innerHTML = text;
  this.el.stepBanner.dataset.step = doneFront && doneBack ? 'done' : String(step);
};

ScannerUI.prototype.hasFiles = function hasFiles() {
  return SIDES.some((side) => Boolean(this.files[side]));
};

/**
 * Caras listas para el POST. La frontal **siempre** viaja en `file` y el
 * reverso en `back_file` (aunque solo se haya cargado el reverso no se debe
 * etiquetar como frontal: el backend trataria `file` como cara frontal).
 */
ScannerUI.prototype.uploads = function uploads() {
  const list = [];
  if (this.files.front) {
    list.push({ side: 'front', field: 'file', file: this.files.front });
  }
  if (this.files.back) {
    list.push({ side: 'back', field: 'back_file', file: this.files.back });
  }
  return list;
};

/* ---------------------------------------------------------------- camara */
ScannerUI.prototype.bindCamera = function bindCamera() {
  this.el.btnStart.addEventListener('click', () => this.startCamera());
  this.el.btnStop.addEventListener('click', () => {
    this.camera.stop();
    this.el.btnStart.hidden = false;
  });
  this.el.btnCapture.addEventListener('click', () => this.captureFrame({ manual: true }));
  this.el.btnSwitch.addEventListener('click', () => this.switchCamera());
  this.el.cameraSelect.addEventListener('change', (event) => this.startCamera(event.target.value));
};

ScannerUI.prototype.bindAutoCapture = function bindAutoCapture() {
  if (!this.el.toggleAuto) return;
  this.el.toggleAuto.addEventListener('change', (event) => {
    this.autoCapture = event.target.checked;
    if (!this.camera?.isActive || !this.detector) return;
    if (this.autoCapture && !this.files[this.activeSide] && !this.busy) {
      this.detector.resume();
    } else {
      this.detector.pause();
      this.setDetectOverlay(null);
    }
  });
};

ScannerUI.prototype.startCamera = async function startCamera(deviceId) {
  this.clearError();
  this.el.btnStart.disabled = true;
  this.el.btnStart.textContent = 'Encendiendo\u2026';

  try {
    await this.camera.start(deviceId ? { deviceId } : {});
    await this.refreshCameraList();
  } catch (error) {
    this.showError(error instanceof CameraError ? error.message : 'No se pudo iniciar la camara.');
  } finally {
    this.el.btnStart.disabled = false;
    this.el.btnStart.textContent = 'Encender camara';
  }
};

ScannerUI.prototype.switchCamera = async function switchCamera() {
  const cameras = await this.camera.listCameras();
  if (cameras.length <= 1) return;

  const currentIndex = cameras.findIndex((camera) => camera.deviceId === this.camera.deviceId);
  const next = cameras[(currentIndex + 1) % cameras.length];
  await this.startCamera(next.deviceId);
};

ScannerUI.prototype.refreshCameraList = async function refreshCameraList() {
  const cameras = await this.camera.listCameras();

  this.el.btnSwitch.hidden = cameras.length <= 1;
  this.el.cameraSelect.hidden = cameras.length <= 1;
  this.el.cameraSelect.innerHTML = '';

  cameras.forEach((camera) => {
    const option = document.createElement('option');
    option.value = camera.deviceId;
    option.textContent = camera.label;
    option.selected = camera.deviceId === this.camera.deviceId;
    this.el.cameraSelect.appendChild(option);
  });
};

ScannerUI.prototype.onCameraChange = function onCameraChange(state) {
  this.el.status.classList.toggle('is-live', state.active);
  this.el.placeholder.hidden = state.active;
  this.el.btnCapture.disabled = !state.active || this.busy;
  this.el.btnStop.hidden = !state.active;
  this.el.btnStart.hidden = state.active;

  if (state.active) {
    const resolution = state.width && state.height ? ` ${state.width}\u00d7${state.height}` : '';
    this.el.statusText.textContent = `En vivo${resolution}`;
    if (this.detector && this.autoCapture && !this.files[this.activeSide] && !this.busy) {
      this.detector.start();
    }
  } else {
    this.el.statusText.textContent = 'Apagada';
    this.detector?.stop();
    this.setDetectOverlay(null);
  }
};

/* ------------------------------------------------------- auto-deteccion */
ScannerUI.prototype.onDetectUpdate = function onDetectUpdate(report) {
  this.setDetectOverlay(report);
};

ScannerUI.prototype.setDetectOverlay = function setDetectOverlay(report) {
  if (!this.el.detectOverlay) return;

  if (!report || report.status === DetectorStatus.IDLE) {
    this.el.detectOverlay.hidden = true;
    this.el.detectOverlay.dataset.state = 'idle';
    if (this.el.detectBar) this.el.detectBar.style.width = '0%';
    return;
  }

  this.el.detectOverlay.hidden = false;
  this.el.detectOverlay.dataset.state = report.status;
  if (this.el.detectText) this.el.detectText.textContent = report.text;
  if (this.el.detectBar) {
    this.el.detectBar.style.width = `${Math.round((report.progress ?? 0) * 100)}%`;
  }
};

ScannerUI.prototype.onAutoCaptureReady = async function onAutoCaptureReady() {
  if (!this.autoCapture || this.busy) return;
  if (this.files[this.activeSide]) return;
  if (Date.now() < this.captureLockUntil) return;

  this.captureLockUntil = Date.now() + AUTO_CAPTURE_COOLDOWN_MS;
  this.showToast(`Documento detectado \u00b7 capturando ${SIDE_TITLES[this.activeSide]}\u2026`);
  await this.captureFrame({ auto: true });
};

ScannerUI.prototype.captureFrame = async function captureFrame({ manual = false, auto = false } = {}) {
  this.clearError();
  const side = this.activeSide;

  if (this.detector && (auto || manual)) this.detector.pause();

  try {
    const capture = await this.camera.capture();
    const file = new File(
      [capture.blob],
      `documento-${side}-${Date.now()}.jpg`,
      { type: 'image/jpeg' },
    );
    this.setSideFile(side, file, capture.dataUrl, `${capture.width}\u00d7${capture.height}`);

    if (auto) {
      this.showToast(`${SIDE_STEPS[side] === 1 ? 'Frontal' : 'Reverso'} capturado automaticamente.`);
    } else if (manual) {
      this.showToast(`${SIDE_TITLES[side]} capturada.`);
    }
  } catch (error) {
    this.showError(error instanceof CameraError ? error.message : 'No se pudo capturar la imagen.');
  } finally {
    this.scheduleDetectorResume();
  }
};

/** Reanuda la deteccion tras una captura (con pequeño enfriamiento). */
ScannerUI.prototype.scheduleDetectorResume = function scheduleDetectorResume() {
  if (!this.detector || !this.camera?.isActive) return;
  if (!this.autoCapture || this.busy) {
    this.detector.pause();
    return;
  }

  window.setTimeout(() => {
    if (!this.camera?.isActive || !this.autoCapture || this.busy) return;
    if (this.files[this.activeSide]) return;
    if (Date.now() < this.captureLockUntil) {
      this.scheduleDetectorResume();
      return;
    }
    this.detector.resume();
  }, 250);
};

/* --------------------------------------------------------------- archivo */
ScannerUI.prototype.bindUpload = function bindUpload() {
  this.el.dropzone.addEventListener('dragover', (event) => {
    event.preventDefault();
    this.el.dropzone.classList.add('is-dragover');
  });

  ['dragleave', 'drop'].forEach((type) => {
    this.el.dropzone.addEventListener(type, (event) => {
      event.preventDefault();
      this.el.dropzone.classList.remove('is-dragover');
    });
  });

  this.el.dropzone.addEventListener('drop', (event) => {
    this.handleFiles(event.dataTransfer?.files);
  });

  this.el.fileInput.addEventListener('change', (event) => {
    this.handleFiles(event.target.files);
    event.target.value = '';
  });
};

/**
 * Acepta una o varias imagenes. Con dos archivos (por ejemplo al arrastrar
 * `parte_de_adelante.jpeg` y `parte_de_atras.jpeg` a la vez) se reparten entre
 * frontal y reverso; el nombre del archivo ayuda a adivinar la cara.
 */
ScannerUI.prototype.handleFiles = function handleFiles(fileList) {
  this.clearError();

  const files = Array.from(fileList ?? []).filter((file) => file && file.type.startsWith('image/'));
  if (!files.length) {
    this.showError('El archivo seleccionado no es una imagen.');
    return;
  }

  const oversized = files.find((file) => file.size > this.maxUploadMb * 1024 * 1024);
  if (oversized) {
    this.showError(`La imagen supera el limite de ${this.maxUploadMb} MB.`);
    return;
  }

  const guessed = files.length === 1 ? guessSideFromFile(files[0]) : null;
  if (files.length === 1) {
    this.assignSideFile(guessed ?? this.activeSide, files[0]);
    return;
  }

  const slots = { front: null, back: null };
  const unnamed = [];

  files.forEach((file) => {
    const side = guessSideFromFile(file);
    if (side && !slots[side]) slots[side] = file;
    else unnamed.push(file);
  });

  unnamed.forEach((file) => {
    if (!slots.front) slots.front = file;
    else if (!slots.back) slots.back = file;
  });

  const assigned = SIDES.filter((side) => slots[side]);
  assigned.forEach((side) => this.assignSideFile(side, slots[side]));

  if (assigned.length === 1) {
    this.showToast('Solo se reconocio una imagen; revisa que falte la otra cara.', 'warn');
  } else {
    this.showToast('Frontal y reverso listos.', 'ok');
  }
};

/** Lee un archivo y lo guarda en la cara indicada. */
ScannerUI.prototype.assignSideFile = function assignSideFile(side, file) {
  const reader = new FileReader();
  reader.onload = () => {
    this.setSideFile(side, file, reader.result, `${Math.round(file.size / 1024)} KB`);
  };
  reader.onerror = () => this.showError('No se pudo leer el archivo seleccionado.');
  reader.readAsDataURL(file);
};

/* -------------------------------------------------------------- acciones */
ScannerUI.prototype.setSideFile = function setSideFile(side, file, dataUrl, meta) {
  this.files[side] = file;
  this.previews[side] = dataUrl;
  this.meta[side] = meta ?? '';

  this.refreshPreview();
  this.refreshCaptures();
  this.refreshStepBanner();

  // Si queda la otra cara pendiente, se cambia sola para no obligar a pulsar
  // el selector entre captura y captura. El orden siempre es frontal -> reverso.
  const missing = SIDES.find((other) => !this.files[other]);
  if (missing) this.setActiveSide(missing);
  else this.el.btnProcess.focus();

  if (this.detector && this.camera?.isActive) {
    if (missing && this.autoCapture) this.detector.resume();
    else this.detector.pause();
  }
};

/** Descarta una cara o, si no se indica ninguna, todas. */
ScannerUI.prototype.discard = function discard(side) {
  const sides = side ? [side] : SIDES;

  sides.forEach((item) => {
    this.files[item] = null;
    this.previews[item] = null;
    this.meta[item] = '';
  });

  if (!this.hasFiles()) this.setBusy(false);
  this.refreshPreview();
  this.refreshCaptures();
  this.refreshStepBanner();
  if (side) this.setActiveSide(side);
  else this.setActiveSide('front');

  if (this.detector && this.camera?.isActive && this.autoCapture && !this.busy) {
    this.detector.resume();
  }
};

/** Repinta las miniaturas, el estado de cada cara y los botones. */
ScannerUI.prototype.refreshPreview = function refreshPreview() {
  const loaded = SIDES.filter((side) => this.files[side]);

  this.el.previewBlock.classList.toggle('hidden', loaded.length === 0);
  this.el.previewMeta.textContent = loaded.length ? `${loaded.length} de 2 caras` : '';

  SIDES.forEach((side) => {
    const ready = Boolean(this.files[side] && this.previews[side]);
    const item = this.el.previewItems[side];
    const image = this.el.previewImages[side];
    const state = this.el.sideStates[side];

    if (item) item.classList.toggle('is-empty', !ready);
    if (image) {
      if (ready) {
        image.src = this.previews[side];
        image.hidden = false;
        image.alt = side === 'front'
          ? 'Captura de la cara frontal del documento'
          : 'Captura del reverso del documento';
      } else {
        image.removeAttribute('src');
        image.alt = side === 'front'
          ? 'Aun sin capturar la cara frontal'
          : 'Aun sin capturar el reverso';
        // Sin src no debe verse el icono de imagen rota del navegador.
        image.hidden = true;
      }
    }

    const remove = this.el.previewRemove[side];
    if (remove) remove.hidden = !ready;

    if (state) {
      state.dataset.state = ready ? 'ready' : 'empty';
      state.textContent = ready ? (this.meta[side] || 'Lista') : 'Sin foto';
    }
  });

  this.el.btnProcess.disabled = this.busy || loaded.length === 0;
};

/**
 * Galeria de capturas tomadas en la sesion: muestra cada foto real
 * (miniatura) con su cara y el momento en que se tomo.
 */
ScannerUI.prototype.refreshCaptures = function refreshCaptures() {
  if (!this.el.capturesTrack || !this.el.capturesBlock) return;

  const track = this.el.capturesTrack;
  track.innerHTML = '';

  const entries = SIDES.filter((side) => this.previews[side]).map((side) => ({
    side,
    src: this.previews[side],
    meta: this.meta[side] || '',
    label: SIDE_STEPS[side] === 1 ? '1 · Frontal' : '2 · Reverso',
  }));

  this.el.capturesBlock.classList.toggle('hidden', entries.length === 0);

  entries.forEach((entry) => {
    const figure = document.createElement('figure');
    figure.className = 'capture-thumb';
    figure.dataset.side = entry.side;

    const img = document.createElement('img');
    img.src = entry.src;
    img.alt = `Captura de la ${SIDE_TITLES[entry.side]}`;
    img.loading = 'lazy';

    const caption = document.createElement('figcaption');
    caption.innerHTML = `<span>${entry.label}</span><span class="capture-thumb__meta">${escapeHtml(entry.meta)}</span>`;

    figure.append(img, caption);
    track.appendChild(figure);
  });
};

ScannerUI.prototype.bindActions = function bindActions() {
  this.el.btnDiscard.addEventListener('click', () => this.discard(null));
  this.el.btnProcess.addEventListener('click', () => this.process());
  this.el.btnCopy.addEventListener('click', () => this.copyData());
};

/** Copia los datos en texto plano para pegarlos en el sistema de recepcion. */
ScannerUI.prototype.copyData = async function copyData() {
  if (!this.lastResult) {
    this.showToast('Todavia no hay datos que copiar.', 'warn');
    return;
  }

  const text = formatResultAsText(this.lastResult);

  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text);
    } else {
      copyWithFallback(text);
    }
    this.showToast('Datos copiados al portapapeles.');
  } catch {
    this.showToast('El navegador bloqueo el portapapeles. Selecciona y copia manualmente.', 'warn');
  }
};

ScannerUI.prototype.showToast = function showToast(message, tone = 'ok') {
  if (!this.el.toast) return;

  this.el.toast.textContent = message;
  this.el.toast.className = `toast toast--${tone} is-visible`;

  window.clearTimeout(this.toastTimer);
  this.toastTimer = window.setTimeout(() => {
    this.el.toast.classList.remove('is-visible');
  }, 3200);
};

ScannerUI.prototype.clearToast = function clearToast() {
  window.clearTimeout(this.toastTimer);
  if (this.el.toast) this.el.toast.classList.remove('is-visible');
};

ScannerUI.prototype.setBusy = function setBusy(busy) {
  this.busy = busy;

  this.el.btnProcess.disabled = busy || !this.hasFiles();
  this.el.btnCapture.disabled = busy || !this.camera?.isActive;
  this.el.btnProcess.textContent = busy ? 'Procesando\u2026' : 'Procesar documento';

  this.el.progress.classList.toggle('hidden', !busy);
  this.el.progressBar.style.width = busy ? '70%' : '0%';

  this.root.setAttribute('aria-busy', String(busy));

  if (this.detector && this.camera?.isActive) {
    if (busy) {
      this.detector.pause();
      this.setDetectOverlay(null);
    } else if (this.autoCapture && !this.files[this.activeSide]) {
      this.detector.resume();
    }
  }
};

ScannerUI.prototype.process = async function process() {
  const uploads = this.uploads();

  if (!uploads.length) {
    this.showError('Primero captura o selecciona la imagen de una de las caras.');
    return;
  }

  if (!this.files.front || !this.files.back) {
    const missing = this.files.front ? 'reverso' : 'cara frontal';
    this.showToast(
      `Falta la ${missing}: sin el reverso no se leen fecha de nacimiento y vencimiento.`,
      'warn',
    );
  }

  if (!this.scanUrl) {
    this.showError('La integracion con el servidor de vision aun no esta configurada.');
    return;
  }

  this.clearError();
  this.setBusy(true);
  this.showLoading();

  try {
    const formData = new FormData();
    uploads.forEach(({ field, file }) => {
      formData.append(field, file, file.name || `${field}.jpg`);
    });
    // Se pide la vista previa de cada cara ya recortada y enderezada.
    formData.append('include_preview', 'true');

    const response = await fetch(this.scanUrl, {
      method: 'POST',
      body: formData,
      headers: { 'X-CSRFToken': getCsrfToken(), Accept: 'application/json' },
      credentials: 'same-origin',
    });

    const payload = await response.json().catch(() => ({}));

    if (!response.ok) {
      this.showError(extractError(payload, response.status));
      this.hideResults();
      return;
    }

    this.renderResult(payload);
  } catch {
    this.showError('No se pudo contactar con el servidor. Revisa la conexion e intentalo de nuevo.');
    this.hideResults();
  } finally {
    this.setBusy(false);
  }
};

ScannerUI.prototype.dismissMessages = function dismissMessages() {
  this.clearError();
  this.clearToast();
};

/* -------------------------------------------------------------- pintado */
ScannerUI.prototype.showLoading = function showLoading() {
  this.el.resultsEmpty.classList.add('hidden');
  this.el.resultsContent.classList.add('hidden');
  this.el.resultsLoading.classList.remove('hidden');
};

ScannerUI.prototype.hideResults = function hideResults() {
  this.el.resultsLoading.classList.add('hidden');
  this.el.resultsContent.classList.add('hidden');
  this.el.resultsEmpty.classList.remove('hidden');
};

ScannerUI.prototype.renderResult = function renderResult(data) {
  this.lastResult = data;
  this.el.resultsLoading.classList.add('hidden');
  this.el.resultsEmpty.classList.add('hidden');
  this.el.resultsContent.classList.remove('hidden');

  const fields = data.fields ?? {};
  const surnames = [fields.first_surname, fields.second_surname].filter(Boolean).join(' ');
  setText('result-number', data.document_number, { mono: true });
  setText('result-name', data.name);
  setText('result-first-surname', surnames || fields.first_surname);
  setText('result-second-surname', fields.second_surname);
  setText('result-given-names', fields.given_names);
  setText('result-type', data.document_type);
  setText('result-birth', fields.birth_date);
  setText('result-birth-place', fields.birth_place);
  setText('result-issue-date', fields.issue_date);
  setText('result-issue-place', fields.issue_place);
  setText('result-expiry', fields.expiry_date);
  setText('result-sex', formatSex(fields.sex));
  setText('result-blood', fields.blood_type);
  setText('result-nationality', fields.nationality || fields.issuing_country);

  const status = document.getElementById('result-status');
  status.className = 'badge ' + (data.success ? 'badge--ok' : 'badge--bad');
  status.textContent = data.success ? 'Leido' : 'No legible';

  this.renderSides(data);

  setBadge('badge-photo', data.valid_photo ? 'Foto valida' : 'Foto no valida',
    data.valid_photo ? 'badge--ok' : 'badge--bad');

  const mrz = data.mrz ?? {};
  if (!mrz.detected) {
    setBadge('badge-mrz', 'Sin MRZ', 'badge--neutral');
  } else if (mrz.valid) {
    setBadge('badge-mrz', 'MRZ consistente', 'badge--ok');
  } else {
    setBadge('badge-mrz', `MRZ ${mrz.passed_checks}/${mrz.total_checks}`, 'badge--warn');
  }

  setBadge('badge-engine', String(data.engine ?? 'ocr').toUpperCase(), 'badge--info');

  const confidence = Math.round((data.confidence ?? 0) * 100);
  setBadge('badge-confidence',
    `Confianza ${confidence}%`,
    confidence >= 70 ? 'badge--ok' : confidence >= 40 ? 'badge--warn' : 'badge--bad');

  document.getElementById('result-message').textContent = data.message ?? '';
  document.getElementById('result-timing').textContent =
    `${Math.round(data.processing_ms ?? 0)} ms \u00b7 ${mrz.format ?? 'sin formato MRZ'}`;

  this.renderWarnings(data.warnings ?? []);
  this.renderMrz(mrz);
  document.getElementById('result-text').textContent = (data.text_lines ?? []).join('\n') || '(sin texto)';
};

/** Refleja en las miniaturas que caras se analizaron y su estado. */
ScannerUI.prototype.renderSides = function renderSides(data) {
  const sides = data.sides ?? [];
  const processed = data.sides_processed ?? sides.filter((side) => side.ok).length;

  setBadge('badge-sides',
    processed >= 2 ? '2 caras' : processed === 1 ? '1 cara' : 'Sin caras',
    processed >= 2 ? 'badge--ok' : processed === 1 ? 'badge--warn' : 'badge--bad');

  sides.forEach((side) => {
    if (!SIDES.includes(side.side)) return;

    if (side.preview_base64) {
      this.previews[side.side] = `data:image/jpeg;base64,${side.preview_base64}`;
    }
    this.meta[side.side] = side.ok ? 'Analizada' : (side.error || 'Error');
  });

  if (data.both_sides) {
    this.el.sideHint.innerHTML = 'Documento leido en <strong>ambas caras</strong>.';
  } else if (processed >= 1) {
    this.el.sideHint.innerHTML =
      'Solo se analizo <strong>una cara</strong>: falta el reverso ' +
      '(fecha de nacimiento y vencimiento).';
    this.showToast('Falta el reverso: fecha de nacimiento y vencimiento pueden quedar vacias.', 'warn');
  }

  this.refreshPreview();
  this.refreshCaptures();
  this.refreshStepBanner();
};

ScannerUI.prototype.renderWarnings = function renderWarnings(warnings) {
  const card = document.getElementById('result-warnings-card');
  const list = document.getElementById('result-warnings');

  list.innerHTML = '';
  warnings.forEach((warning) => {
    const item = document.createElement('li');
    item.textContent = warning;
    list.appendChild(item);
  });

  card.classList.toggle('hidden', warnings.length === 0);
};

ScannerUI.prototype.renderMrz = function renderMrz(mrz) {
  const body = document.getElementById('mrz-checks');
  const lines = document.getElementById('mrz-lines');

  body.innerHTML = '';
  const checks = mrz.checks ?? {};

  Object.keys(checks).forEach((name) => {
    const row = document.createElement('tr');
    const label = document.createElement('td');
    const value = document.createElement('td');

    label.textContent = LABELS[name] ?? name;
    value.innerHTML = checks[name]
      ? '<span class="badge badge--ok">Coincide</span>'
      : '<span class="badge badge--bad">No coincide</span>';

    row.append(label, value);
    body.appendChild(row);
  });

  if (!Object.keys(checks).length) {
    body.innerHTML = '<tr><td colspan="2" class="text-muted">No se analizo ninguna MRZ.</td></tr>';
  }

  lines.textContent = (mrz.normalized_lines ?? []).join('\n') || '(no detectada)';
};

/* --------------------------------------------------------------- errores */
ScannerUI.prototype.showError = function showError(message) {
  this.el.error.textContent = message;
  this.el.error.classList.remove('hidden');
};

ScannerUI.prototype.clearError = function clearError() {
  this.el.error.textContent = '';
  this.el.error.classList.add('hidden');
};

/* -------------------------------------------------------------- utileria */
const LABELS = {
  document_number: 'Numero de documento',
  birth_date: 'Fecha de nacimiento',
  expiry_date: 'Fecha de caducidad',
  composite: 'Digito compuesto',
  personal_number: 'Numero personal',
};

function setText(id, value, { mono = false } = {}) {
  const element = document.getElementById(id);
  if (!element) return;

  const hasValue = value !== null && value !== undefined && value !== '';
  element.textContent = hasValue ? value : '\u2014';
  element.classList.toggle('is-empty', !hasValue);
  if (mono) element.classList.add('mono');
}

function setBadge(id, text, variant) {
  const element = document.getElementById(id);
  if (!element) return;
  element.className = `badge ${variant}`;
  element.textContent = text;
}

function formatSex(value) {
  if (value === 'M') return 'M (masculino)';
  if (value === 'F') return 'F (femenino)';
  return value;
}

/** Adivina la cara a partir del nombre del archivo (`atras`, `front`, …). */
function guessSideFromFile(file) {
  const name = String(file?.name ?? '').toLowerCase();
  if (/(atras|reverso|back|reverse)/.test(name)) return 'back';
  if (/(adelante|anverso|front|frontal)/.test(name)) return 'front';
  return null;
}

function getCsrfToken() {
  const match = document.cookie.match(/(?:^|;\s*)csrftoken=([^;]+)/);
  return match ? decodeURIComponent(match[1]) : '';
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

/** Resumen en texto plano de los datos extraidos. */
function formatResultAsText(data) {
  const fields = data.fields ?? {};
  const mrz = data.mrz ?? {};
  const yesNo = (value) => (value ? 'Si' : 'No');

  const lines = [
    `Numero de documento: ${data.document_number ?? '-'}`,
    `Apellidos: ${[fields.first_surname, fields.second_surname].filter(Boolean).join(' ') || '-'}`,
    `Nombres: ${fields.given_names ?? '-'}`,
    `Nacionalidad: ${fields.nationality ?? fields.issuing_country ?? '-'}`,
    `Fecha de nacimiento: ${fields.birth_date ?? '-'}`,
    `Fecha de vencimiento (caducidad): ${fields.expiry_date ?? '-'}`,
    `Nombres y apellidos: ${data.name ?? '-'}`,
    `Tipo: ${data.document_type ?? '-'}`,
    `Lugar de nacimiento: ${fields.birth_place ?? '-'}`,
    `Fecha de expedicion: ${fields.issue_date ?? '-'}`,
    `Lugar de expedicion: ${fields.issue_place ?? '-'}`,
    `Sexo: ${fields.sex ?? '-'}`,
    `Grupo sanguineo: ${fields.blood_type ?? '-'}`,
    `Foto valida: ${yesNo(data.valid_photo)}`,
    `Caras analizadas: ${data.sides_processed ?? 0}`,
    mrz.detected
      ? `MRZ (${mrz.format}): ${mrz.valid ? 'consistente' : `inconsistente (${mrz.passed_checks}/${mrz.total_checks})`}`
      : 'MRZ: no detectada',
  ];

  return lines.join('\n');
}

function copyWithFallback(text) {
  const area = document.createElement('textarea');
  area.value = text;
  area.setAttribute('readonly', '');
  area.style.position = 'fixed';
  area.style.opacity = '0';
  document.body.appendChild(area);
  area.select();
  document.execCommand('copy');
  document.body.removeChild(area);
}

function extractError(payload, status) {
  const detail = payload?.detail;

  if (typeof detail === 'string') return detail;

  if (Array.isArray(detail)) {
    return detail.map((item) => item.msg ?? 'Dato invalido').join('. ');
  }

  return 'No se pudo procesar la imagen. Intenta con otra fotografia.';
}

export { ScannerUI, formatResultAsText, getCsrfToken };

/* ---------------------------------------------------------------- arranque */
const root = document.getElementById('scanner-root');
if (root) {
  new ScannerUI(root).init();
}
