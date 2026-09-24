/**
 * Control de la camara del dispositivo mediante `getUserMedia`.
 *
 * Encapsula todo lo relacionado con el stream de video: permisos, seleccion
 * de dispositivo, captura del fotograma a JPEG y liberacion de recursos.
 *
 * Nota: `getUserMedia` solo funciona en contextos seguros. `http://localhost`
 * y `http://127.0.0.1` se consideran seguros; en cualquier otro host se
 * necesita HTTPS.
 */

const DEFAULTS = {
  maxWidth: 1920,
  quality: 0.92,
  facingMode: 'environment',
};

export class CameraError extends Error {
  constructor(message, code) {
    super(message);
    this.name = 'CameraError';
    this.code = code;
  }
}

function describeError(error) {
  switch (error?.name) {
    case 'NotAllowedError':
    case 'SecurityError':
      return 'Permiso de camara denegado. Habilitalo en el navegador e intentalo de nuevo.';
    case 'NotFoundError':
    case 'DevicesNotFoundError':
      return 'No se encontro ninguna camara disponible en este dispositivo.';
    case 'NotReadableError':
    case 'TrackStartError':
      return 'La camara esta siendo usada por otra aplicacion. Cierrala e intentalo otra vez.';
    case 'OverconstrainedError':
      return 'La camara seleccionada no admite la configuracion solicitada.';
    case 'NotSupportedError':
      return 'Este navegador no soporta la captura de video.';
    default:
      return 'No se pudo iniciar la camara. Revisa los permisos del navegador.';
  }
}

export class CameraController {
  constructor(videoElement, { onChange } = {}) {
    this.video = videoElement;
    this.onChange = onChange ?? (() => {});
    this.stream = null;
    this.deviceId = null;
    this.facingMode = DEFAULTS.facingMode;
  }

  /** Indica si el navegador expone la API de captura. */
  static isSupported() {
    return Boolean(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
  }

  /**
   * `getUserMedia` exige un contexto seguro (HTTPS o localhost).
   */
  static isSecureContext() {
    return window.isSecureContext === true;
  }

  get isActive() {
    return Boolean(this.stream && this.stream.getVideoTracks().some((track) => track.readyState === 'live'));
  }

  /** Lista las camaras disponibles (requiere permiso previo para las etiquetas). */
  async listCameras() {
    if (!CameraController.isSupported()) return [];

    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      return devices
        .filter((device) => device.kind === 'videoinput')
        .map((device, index) => ({
          deviceId: device.deviceId,
          label: device.label || `Camara ${index + 1}`,
        }));
    } catch {
      return [];
    }
  }

  /**
   * Inicia el stream de video.
   *
   * @param {{ deviceId?: string, facingMode?: string }} options
   */
  async start(options = {}) {
    if (!CameraController.isSupported()) {
      throw new CameraError(
        'Este navegador no soporta la captura de camara. Usa la pestana de archivo.',
        'unsupported',
      );
    }

    if (!CameraController.isSecureContext()) {
      throw new CameraError(
        'La camara requiere un contexto seguro. Accede por HTTPS o por localhost.',
        'insecure-context',
      );
    }

    this.stop();

    if (options.deviceId) {
      this.deviceId = options.deviceId;
    }
    if (options.facingMode) {
      this.facingMode = options.facingMode;
    }

    const constraints = this.deviceId
      ? { video: { deviceId: { exact: this.deviceId }, width: { ideal: 1920 }, height: { ideal: 1440 } } }
      : {
          video: {
            facingMode: { ideal: this.facingMode },
            width: { ideal: 1920 },
            height: { ideal: 1440 },
          },
        };

    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia(constraints);
    } catch (error) {
      // Reintento conservador cuando el dispositivo elegido no esta disponible.
      if (this.deviceId) {
        this.deviceId = null;
        try {
          stream = await navigator.mediaDevices.getUserMedia({ video: true });
        } catch (retryError) {
          throw new CameraError(describeError(retryError), retryError?.name ?? 'unknown');
        }
      } else {
        throw new CameraError(describeError(error), error?.name ?? 'unknown');
      }
    }

    this.stream = stream;
    this.video.srcObject = stream;
    this.video.setAttribute('playsinline', 'true');
    this.video.muted = true;

    await this.video.play().catch(() => {
      /* algunos navegadores requieren una interaccion previa del usuario */
    });

    await this.#preferHighestResolution();
    this.onChange(this.state());

    return stream;
  }

  /** Detiene todas las pistas y libera la camara. */
  stop() {
    if (!this.stream) return;

    this.stream.getTracks().forEach((track) => track.stop());
    this.stream = null;
    this.video.srcObject = null;
    this.onChange(this.state());
  }

  /**
   * Captura el fotograma actual como JPEG.
   *
   * Se usa la resolucion nativa del stream (limitada a `maxWidth`) y **no** se
   * aplica el efecto espejo del previsualizador: el documento debe llegar tal
   * como es, sin invertir.
   *
   * @returns {Promise<{ blob: Blob, width: number, height: number, dataUrl: string }>}
   */
  async capture({ maxWidth = DEFAULTS.maxWidth, quality = DEFAULTS.quality } = {}) {
    if (!this.isActive) {
      throw new CameraError('La camara no esta activa.', 'inactive');
    }

    const track = this.stream.getVideoTracks()[0];
    const settings = track.getSettings ? track.getSettings() : {};

    const sourceWidth = this.video.videoWidth || settings.width || 0;
    const sourceHeight = this.video.videoHeight || settings.height || 0;

    if (!sourceWidth || !sourceHeight) {
      throw new CameraError('Todavia no hay imagen de la camara. Espera un momento.', 'no-frame');
    }

    const scale = Math.min(1, maxWidth / sourceWidth);
    const width = Math.round(sourceWidth * scale);
    const height = Math.round(sourceHeight * scale);

    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;

    const context = canvas.getContext('2d');
    context.drawImage(this.video, 0, 0, width, height);

    const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', quality));
    if (!blob) {
      throw new CameraError('No se pudo generar la imagen capturada.', 'encode-failed');
    }

    return { blob, width, height, dataUrl: canvas.toDataURL('image/jpeg', quality) };
  }

  /** Estado actual, util para actualizar la interfaz. */
  state() {
    const track = this.stream?.getVideoTracks?.()[0] ?? null;
    const settings = track?.getSettings?.() ?? {};

    return {
      active: this.isActive,
      label: track?.label ?? null,
      width: settings.width ?? null,
      height: settings.height ?? null,
    };
  }

  /**
   * Pide al navegador la maxima resolucion razonable para leer la MRZ.
   */
  async #preferHighestResolution() {
    const track = this.stream?.getVideoTracks?.()[0];
    const capabilities = track?.getCapabilities?.() ?? {};

    if (!capabilities.width || !capabilities.height) return;

    const targetWidth = Math.min(capabilities.width.max ?? 1920, 2560);
    if (targetWidth <= (this.video.videoWidth || 0)) return;

    try {
      await track.applyConstraints({ width: { ideal: targetWidth } });
    } catch {
      /* la camara no admite el cambio: se mantiene la resolucion actual */
    }
  }
}

export default CameraController;
