/**
 * Controlador de la interfaz del escaner.
 *
 * Une la captura por camara (o la carga de un archivo) con el envio al
 * backend y el pintado del resultado en pantalla.
 */

import { CameraController, CameraError } from './camera.js';

const root = document.getElementById('scanner-root');
if (root) {
  new ScannerUI(root).init();
}

function ScannerUI(element) {
  this.root = element;
  this.scanUrl = element.dataset.scanUrl;
  this.maxUploadMb = Number(element.dataset.maxUploadMb || 8);
  this.camera = null;
  this.pendingFile = null;
  this.previewUrl = null;
  this.busy = false;
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

    fileInput: document.getElementById('file-input'),
    dropzone: document.getElementById('dropzone'),
    fileName: document.getElementById('file-name'),

    previewBlock: document.getElementById('preview-block'),
    previewImage: document.getElementById('preview-image'),
    previewMeta: document.getElementById('preview-meta'),
    btnProcess: document.getElementById('btn-process'),
    btnDiscard: document.getElementById('btn-discard'),

    progress: document.getElementById('progress'),
    progressBar: document.getElementById('progress-bar'),
    error: document.getElementById('capture-error'),

    resultsEmpty: document.getElementById('results-empty'),
    resultsLoading: document.getElementById('results-loading'),
    resultsContent: document.getElementById('results-content'),
  };

  this.setupTabs();
  if (CameraController.isSupported()) {
    this.camera = new CameraController(this.el.video, { onChange: this.onCameraChange.bind(this) });
    this.bindCamera();
  } else {
    this.el.btnStart.disabled = true;
    this.el.btnStart.textContent = 'Camara no disponible';
    this.showError('Este navegador no soporta la captura de camara. Usa la pestana "Subir archivo".');
  }

  this.bindUpload();
  this.bindActions();

  document.addEventListener('visibilitychange', () => {
    if (document.hidden && this.camera) this.camera.stop();
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

/* ---------------------------------------------------------------- camara */
ScannerUI.prototype.bindCamera = function bindCamera() {
  this.el.btnStart.addEventListener('click', () => this.startCamera());
  this.el.btnStop.addEventListener('click', () => {
    this.camera.stop();
    this.el.btnStart.hidden = false;
  });
  this.el.btnCapture.addEventListener('click', () => this.captureFrame());
  this.el.btnSwitch.addEventListener('click', () => this.switchCamera());
  this.el.cameraSelect.addEventListener('change', (event) => this.startCamera(event.target.value));
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
  } else {
    this.el.statusText.textContent = 'Apagada';
  }
};

ScannerUI.prototype.captureFrame = async function captureFrame() {
  this.clearError();

  try {
    const capture = await this.camera.capture();
    const file = new File([capture.blob], `documento-${Date.now()}.jpg`, { type: 'image/jpeg' });
    this.setPendingFile(file, capture.dataUrl, `${capture.width}\u00d7${capture.height}`);
  } catch (error) {
    this.showError(error instanceof CameraError ? error.message : 'No se pudo capturar la imagen.');
  }
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
    const [file] = event.dataTransfer?.files ?? [];
    if (file) this.handleFile(file);
  });

  this.el.fileInput.addEventListener('change', (event) => {
    const [file] = event.target.files ?? [];
    if (file) this.handleFile(file);
  });
};

ScannerUI.prototype.handleFile = function handleFile(file) {
  this.clearError();

  if (!file.type.startsWith('image/')) {
    this.showError('El archivo seleccionado no es una imagen.');
    return;
  }

  if (file.size > this.maxUploadMb * 1024 * 1024) {
    this.showError(`La imagen supera el limite de ${this.maxUploadMb} MB.`);
    return;
  }

  const reader = new FileReader();
  reader.onload = () => this.setPendingFile(file, reader.result, `${Math.round(file.size / 1024)} KB`);
  reader.onerror = () => this.showError('No se pudo leer el archivo seleccionado.');
  reader.readAsDataURL(file);
};

/* -------------------------------------------------------------- acciones */
ScannerUI.prototype.setPendingFile = function setPendingFile(file, dataUrl, meta) {
  this.pendingFile = file;
  this.previewUrl = dataUrl;

  this.el.previewImage.src = dataUrl;
  this.el.previewMeta.textContent = meta ?? '';
  this.el.previewBlock.classList.remove('hidden');
  this.el.fileName.textContent = file.name;
  this.el.btnProcess.focus();
};

ScannerUI.prototype.bindActions = function bindActions() {
  this.el.btnDiscard.addEventListener('click', () => this.discard());
  this.el.btnProcess.addEventListener('click', () => this.process());
};

ScannerUI.prototype.discard = function discard() {
  this.pendingFile = null;
  this.previewUrl = null;
  this.el.previewImage.removeAttribute('src');
  this.el.previewBlock.classList.add('hidden');
  this.el.fileInput.value = '';
  this.el.fileName.textContent = '';
  this.setBusy(false);
};

ScannerUI.prototype.setBusy = function setBusy(busy) {
  this.busy = busy;

  this.el.btnProcess.disabled = busy;
  this.el.btnCapture.disabled = busy || !this.camera?.isActive;
  this.el.btnProcess.textContent = busy ? 'Procesando\u2026' : 'Procesar documento';

  this.el.progress.classList.toggle('hidden', !busy);
  this.el.progressBar.style.width = busy ? '70%' : '0%';
};

ScannerUI.prototype.process = async function process() {
  if (!this.pendingFile) {
    this.showError('Primero captura o selecciona una imagen.');
    return;
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
    formData.append('file', this.pendingFile, this.pendingFile.name || 'documento.jpg');

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
  this.el.resultsLoading.classList.add('hidden');
  this.el.resultsEmpty.classList.add('hidden');
  this.el.resultsContent.classList.remove('hidden');

  const fields = data.fields ?? {};
  setText('result-number', data.document_number, { mono: true });
  setText('result-name', data.name);
  setText('result-type', data.document_type);
  setText('result-birth', fields.birth_date);
  setText('result-expiry', fields.expiry_date);
  setText('result-nationality', fields.nationality || fields.issuing_country);

  const status = document.getElementById('result-status');
  status.className = 'badge ' + (data.success ? 'badge--ok' : 'badge--bad');
  status.textContent = data.success ? 'Leido' : 'No legible';

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

function getCsrfToken() {
  const match = document.cookie.match(/(?:^|;\s*)csrftoken=([^;]+)/);
  return match ? decodeURIComponent(match[1]) : '';
}

function extractError(payload, status) {
  const detail = payload?.detail;

  if (typeof detail === 'string') return detail;

  if (Array.isArray(detail)) {
    return detail.map((item) => item.msg ?? 'Dato invalido').join('. ');
  }

  if (status === 401 || status === 403) {
    return 'Tu sesion expiro. Vuelve a iniciar sesion.';
  }

  return 'No se pudo procesar la imagen. Intenta con otra fotografia.';
}

export { ScannerUI, getCsrfToken };
