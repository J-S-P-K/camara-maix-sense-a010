"""Monitorea el almacenamiento de la maqueta con la cámara ToF y una red neuronal (TFLite).
Si la IA detecta el almacenamiento lleno durante TIEMPO_ALERTA segundos seguidos,
manda una alerta por la terminal. También muestra el stream en http://<IP>:8085
"""
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

from tof_camara import ZONA, leer_frames, preprocesar

MODELO = "modelo_almacenamiento.tflite"
UMBRAL_CONFIANZA = 0.8   # probabilidad mínima de "lleno" que da la IA para considerarlo lleno
TIEMPO_ALERTA = 20       # segundos seguidos lleno antes de alertar
SUAVIZADO = 0.3          # 0-1: cuánto pesa cada frame nuevo (evita que un frame raro corte la cuenta)
STREAM_SIZE = (400, 400)
PUERTO_WEB = 8085

latest_frame = None
frame_lock = threading.Lock()


def cargar_interprete(ruta):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            from tensorflow.lite import Interpreter
    interprete = Interpreter(model_path=ruta)
    interprete.allocate_tensors()
    return interprete


class ClasificadorAlmacenamiento:
    def __init__(self, ruta_modelo):
        self.interprete = cargar_interprete(ruta_modelo)
        self.entrada = self.interprete.get_input_details()[0]
        self.salida = self.interprete.get_output_details()[0]

    def probabilidad_lleno(self, img):
        x = preprocesar(img)[np.newaxis, ...]
        if self.entrada["dtype"] != np.float32:
            escala, cero = self.entrada["quantization"]
            x = np.round(x / escala + cero).astype(self.entrada["dtype"])
        self.interprete.set_tensor(self.entrada["index"], x)
        self.interprete.invoke()
        y = self.interprete.get_tensor(self.salida["index"])
        if self.salida["dtype"] != np.float32:
            escala, cero = self.salida["quantization"]
            y = (y.astype(np.float32) - cero) * escala
        return float(y.ravel()[0])


class MonitorLlenado:
    """Lleva la cuenta de cuánto tiempo seguido está lleno el almacenamiento."""

    def __init__(self, segundos):
        self.segundos = segundos
        self.lleno_desde = None
        self.alertado = False

    def actualizar(self, lleno, ahora):
        """Devuelve 'alerta' al cumplirse el tiempo, 'liberado' si se vacía después de alertar, o None."""
        if lleno:
            if self.lleno_desde is None:
                self.lleno_desde = ahora
            if not self.alertado and ahora - self.lleno_desde >= self.segundos:
                self.alertado = True
                return "alerta"
            return None
        evento = "liberado" if self.alertado else None
        self.lleno_desde = None
        self.alertado = False
        return evento

    def tiempo_lleno(self, ahora):
        return 0.0 if self.lleno_desde is None else ahora - self.lleno_desde


def dibujar(img, prob, lleno, segundos):
    big = cv2.resize(img, STREAM_SIZE, interpolation=cv2.INTER_NEAREST)
    color = cv2.applyColorMap(big, cv2.COLORMAP_JET)
    fx = STREAM_SIZE[0] / img.shape[1]
    fy = STREAM_SIZE[1] / img.shape[0]
    x1, y1, x2, y2 = ZONA
    cv2.rectangle(color, (int(x1 * fx), int(y1 * fy)), (int(x2 * fx) - 1, int(y2 * fy) - 1), (0, 255, 0), 2)
    texto = f"{'LLENO' if lleno else 'VACIO'} {prob * 100:.0f}%  {segundos:.0f}/{TIEMPO_ALERTA}s"
    cv2.putText(color, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv2.putText(color, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return color


def monitorear():
    global latest_frame
    clasificador = ClasificadorAlmacenamiento(MODELO)
    monitor = MonitorLlenado(TIEMPO_ALERTA)
    prob_suave = 0.0

    for img in leer_frames():
        prob = clasificador.probabilidad_lleno(img)
        prob_suave = SUAVIZADO * prob + (1 - SUAVIZADO) * prob_suave
        lleno = prob_suave >= UMBRAL_CONFIANZA
        ahora = time.monotonic()

        evento = monitor.actualizar(lleno, ahora)
        if evento == "alerta":
            print(f"\n\a🚨 ALERTA [{time.strftime('%H:%M:%S')}]: el almacenamiento está LLENO "
                  f"hace {TIEMPO_ALERTA} s (confianza IA {prob_suave * 100:.0f}%)")
        elif evento == "liberado":
            print(f"\n✅ [{time.strftime('%H:%M:%S')}] El almacenamiento se liberó.")

        segundos = monitor.tiempo_lleno(ahora)
        print(f"🤖 IA: {'LLENO' if lleno else 'vacío'} ({prob_suave * 100:5.1f}%) | "
              f"lleno hace {segundos:4.1f}/{TIEMPO_ALERTA} s   ", end="\r")

        _, jpeg = cv2.imencode('.jpg', dibujar(img, prob_suave, lleno, segundos))
        with frame_lock:
            latest_frame = jpeg.tobytes()


class StreamingHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            html = f"""
            <html>
            <head><title>Almacenamiento - IA ToF</title></head>
            <body style="background:#1e1e2e; color:#cdd6f4; font-family:sans-serif; display:flex; flex-direction:column; align-items:center; justify-content:center; height:100vh; margin:0;">
                <h2>Monitor de almacenamiento (IA + Sipeed A010)</h2>
                <div style="border: 4px solid #45475a; border-radius: 12px; overflow: hidden;">
                    <img src="/stream" width="{STREAM_SIZE[0]}" height="{STREAM_SIZE[1]}" />
                </div>
            </body>
            </html>
            """
            self.wfile.write(html.encode('utf-8'))
        elif self.path == '/stream':
            self.send_response(200)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            last_sent = None
            try:
                while True:
                    with frame_lock:
                        frame = latest_frame
                    if frame is not None and frame is not last_sent:
                        last_sent = frame
                        self.wfile.write(b'--frame\r\n')
                        self.send_header('Content-Type', 'image/jpeg')
                        self.send_header('Content-Length', str(len(frame)))
                        self.end_headers()
                        self.wfile.write(frame)
                        self.wfile.write(b'\r\n')
                    time.sleep(0.066)
            except Exception:
                pass
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        return


if __name__ == '__main__':
    threading.Thread(target=monitorear, daemon=True).start()
    server = ThreadingHTTPServer(('0.0.0.0', PUERTO_WEB), StreamingHandler)
    server.daemon_threads = True
    print(f"[🌍] Stream en http://<IP-DE-LA-RADXA>:{PUERTO_WEB}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Cerrando...")
