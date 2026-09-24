/**
 * Deteccion en tiempo real del documento dentro del encuadre.
 *
 * Analiza fotogramas reducidos del `<video>` en un canvas offscreen y
 * decide si el documento esta presente, bien alineado y suficientemente
 * nítido como para disparar la captura automatica.
 *
 * Sin dependencias: proyecciones de borde (izquierda/derecha/arriba/abajo),
 * nitidez (varianza del laplaciano), brillo y estabilidad entre fotogramas.
 */

const DEFAULTS = {
  sampleWidth: 160,
  sampleHeight: 120,
  intervalMs: 90,
  /** Fotogramas seguidos en estado `ready` antes de disparar onReady. */
  requiredStable: 8,
};

export const DetectorStatus = {
  IDLE: 'idle',
  SEARCHING: 'searching',
  ALIGN: 'align',
  HOLD: 'hold',
  READY: 'ready',
};

const STATUS_TEXT = {
  [DetectorStatus.IDLE]: 'Camara inactiva',
  [DetectorStatus.SEARCHING]: 'Buscando el documento…',
  [DetectorStatus.ALIGN]: 'Alinea el documento dentro del marco',
  [DetectorStatus.HOLD]: 'Mantén quieto…',
  [DetectorStatus.READY]: 'Listo, capturando…',
};

function smoothProfile(profile) {
  const out = new Float32Array(profile.length);
  for (let i = 0; i < profile.length; i += 1) {
    const left = profile[Math.max(0, i - 1)];
    const right = profile[Math.min(profile.length - 1, i + 1)];
    out[i] = (left + profile[i] + right) / 3;
  }
  return out;
}

function peakPair(profile, { minSeparationRatio = 0.35 } = {}) {
  const n = profile.length;
  if (n < 8) return null;

  const smoothed = smoothProfile(smoothProfile(profile));
  let max = 0;
  for (let i = 0; i < n; i += 1) max = Math.max(max, smoothed[i]);
  if (max <= 0) return null;

  const threshold = max * 0.35;
  const minSep = Math.floor(n * minSeparationRatio);

  let first = -1;
  let firstValue = 0;
  for (let i = 2; i < n - 2; i += 1) {
    if (smoothed[i] < threshold) continue;
    if (
      smoothed[i] >= smoothed[i - 1] &&
      smoothed[i] >= smoothed[i + 1] &&
      smoothed[i] > firstValue
    ) {
      first = i;
      firstValue = smoothed[i];
    }
  }
  if (first < 0) return null;

  let second = -1;
  let secondValue = 0;
  for (let i = 2; i < n - 2; i += 1) {
    if (Math.abs(i - first) < minSep) continue;
    if (smoothed[i] < threshold) continue;
    if (
      smoothed[i] >= smoothed[i - 1] &&
      smoothed[i] >= smoothed[i + 1] &&
      smoothed[i] > secondValue
    ) {
      second = i;
      secondValue = smoothed[i];
    }
  }
  if (second < 0) return null;

  return first < second ? [first, second] : [second, first];
}

function mean(values) {
  if (!values.length) return 0;
  let total = 0;
  for (const value of values) total += value;
  return total / values.length;
}

export class DocumentDetector {
  constructor(video, { onUpdate, onReady, ...options } = {}) {
    this.video = video;
    this.onUpdate = onUpdate ?? (() => {});
    this.onReady = onReady ?? (() => {});
    this.sampleWidth = options.sampleWidth ?? DEFAULTS.sampleWidth;
    this.sampleHeight = options.sampleHeight ?? DEFAULTS.sampleHeight;
    this.intervalMs = options.intervalMs ?? DEFAULTS.intervalMs;
    this.requiredStable = options.requiredStable ?? DEFAULTS.requiredStable;

    this.canvas = document.createElement('canvas');
    this.canvas.width = this.sampleWidth;
    this.canvas.height = this.sampleHeight;
    this.ctx = this.canvas.getContext('2d', { willReadFrequently: true });

    this.timer = null;
    this.paused = false;
    this.stableCount = 0;
    this.status = DetectorStatus.IDLE;
    this.lastReport = null;
  }

  get isRunning() {
    return this.timer !== null;
  }

  start() {
    if (this.timer) return;
    this.paused = false;
    this.stableCount = 0;
    this.#setStatus(DetectorStatus.SEARCHING);
    this.timer = window.setInterval(() => this.#tick(), this.intervalMs);
  }

  stop() {
    if (this.timer) {
      window.clearInterval(this.timer);
      this.timer = null;
    }
    this.paused = false;
    this.stableCount = 0;
    this.#setStatus(DetectorStatus.IDLE);
  }

  pause() {
    this.paused = true;
    this.stableCount = 0;
  }

  resume() {
    if (!this.timer) {
      this.start();
      return;
    }
    this.paused = false;
    this.stableCount = 0;
    this.#setStatus(DetectorStatus.SEARCHING);
  }

  resetStable() {
    this.stableCount = 0;
  }

  #setStatus(status, metrics = null) {
    this.status = status;
    const report = {
      status,
      text: STATUS_TEXT[status] ?? '',
      stable: this.stableCount,
      required: this.requiredStable,
      progress: Math.min(1, this.stableCount / this.requiredStable),
      metrics,
      ready: status === DetectorStatus.READY,
    };
    this.lastReport = report;
    this.onUpdate(report);
  }

  #tick() {
    if (this.paused || document.hidden) return;

    const video = this.video;
    if (!video || video.readyState < 2 || !video.videoWidth) {
      this.#setStatus(DetectorStatus.SEARCHING);
      return;
    }

    const metrics = this.#analyzeFrame();
    if (!metrics) {
      this.#setStatus(DetectorStatus.SEARCHING);
      return;
    }

    if (!metrics.detected) {
      this.stableCount = 0;
      this.#setStatus(DetectorStatus.SEARCHING, metrics);
      return;
    }

    const problems = [];
    if (metrics.coverage < 0.18) problems.push('muy_lejos');
    if (metrics.coverage > 0.94) problems.push('muy_cerca');
    if (metrics.ratio < 1.15 || metrics.ratio > 2.4) problems.push('encuadre');
    if (metrics.sharpness < 18) problems.push('borroso');
    if (metrics.brightness < 36 || metrics.brightness > 232) problems.push('luz');

    if (problems.length) {
      this.stableCount = 0;
      this.#setStatus(DetectorStatus.ALIGN, { ...metrics, problems });
      return;
    }

    this.stableCount += 1;
    if (this.stableCount >= this.requiredStable) {
      this.#setStatus(DetectorStatus.READY, metrics);
      this.paused = true;
      this.onReady(metrics);
      return;
    }

    this.#setStatus(DetectorStatus.HOLD, metrics);
  }

  #analyzeFrame() {
    const { sampleWidth: w, sampleHeight: h, ctx } = this;
    try {
      ctx.drawImage(this.video, 0, 0, w, h);
    } catch {
      return null;
    }

    let imageData;
    try {
      imageData = ctx.getImageData(0, 0, w, h);
    } catch {
      return null;
    }

    const { data } = imageData;
    const gray = new Float32Array(w * h);
    for (let i = 0; i < w * h; i += 1) {
      const offset = i * 4;
      gray[i] = 0.299 * data[offset] + 0.587 * data[offset + 1] + 0.114 * data[offset + 2];
    }

    // Perfiles de gradiente: las aristas del documento forman dos picos
    // fuertes en cada eje (borde izquierdo/derecho y superior/inferior).
    const colEdge = new Float32Array(w);
    const rowEdge = new Float32Array(h);
    for (let y = 0; y < h; y += 1) {
      for (let x = 1; x < w; x += 1) {
        if (Math.abs(gray[y * w + x] - gray[y * w + x - 1]) > 22) {
          colEdge[x] += 1;
          rowEdge[y] += Math.abs(gray[y * w + x] - gray[y * w + x - 1]) > 30 ? 0.35 : 0;
        }
      }
    }
    for (let y = 1; y < h; y += 1) {
      for (let x = 0; x < w; x += 1) {
        const delta = Math.abs(gray[y * w + x] - gray[(y - 1) * w + x]);
        if (delta > 22) rowEdge[y] += 1;
      }
    }

    const horizontal = peakPair(colEdge, { minSeparationRatio: 0.3 });
    const vertical = peakPair(rowEdge, { minSeparationRatio: 0.25 });
    if (!horizontal || !vertical) {
      return { detected: false, reason: 'sin_bordes' };
    }

    const [x0, x1] = horizontal;
    const [y0, y1] = vertical;
    const rectW = x1 - x0;
    const rectH = y1 - y0;
    if (rectW < 8 || rectH < 8) {
      return { detected: false, reason: 'rect_minimo' };
    }

    const coverage = (rectW * rectH) / (w * h);
    const ratio = Math.max(rectW / rectH, rectH / rectW);

    // Nitidez y brillo solo dentro del rectangulo detectado.
    let lapSum = 0;
    let lapCount = 0;
    let brightSum = 0;
    let brightCount = 0;
    const yStart = Math.max(1, y0);
    const yEnd = Math.min(h - 1, y1);
    const xStart = Math.max(1, x0);
    const xEnd = Math.min(w - 1, x1);

    for (let y = yStart; y < yEnd; y += 1) {
      for (let x = xStart; x < xEnd; x += 1) {
        const i = y * w + x;
        const lap =
          gray[i - 1] + gray[i + 1] + gray[i - w] + gray[i + w] - 4 * gray[i];
        lapSum += Math.abs(lap);
        lapCount += 1;
        brightSum += gray[i];
        brightCount += 1;
      }
    }

    const sharpness = lapCount ? lapSum / lapCount : 0;
    const brightness = brightCount ? brightSum / brightCount : 0;

    return {
      detected: true,
      coverage: Number(coverage.toFixed(3)),
      ratio: Number(ratio.toFixed(3)),
      sharpness: Number(sharpness.toFixed(2)),
      brightness: Number(brightness.toFixed(1)),
      rect: { x0, y0, x1, y1 },
    };
  }
}

export default DocumentDetector;
