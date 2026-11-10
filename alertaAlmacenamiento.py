"""Monitorea el almacenamiento de la maqueta con la cámara ToF y una red neuronal (TFLite).
Si la IA detecta el almacenamiento lleno durante TIEMPO_ALERTA segundos seguidos,
manda una alerta por la terminal. También muestra el stream en http://<IP>:8085

La lectura de la cámara y el servidor web son los mismos de camara.py (el stream
que funciona); lo único que se agrega es la clasificación con la IA.
Si todavía no existe el modelo .tflite, el stream igual se muestra (sin IA).
"""
import os
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer

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


def cargar_clasificador():
    """Devuelve el clasificador, o None (con un mensaje claro) si no se puede usar la IA."""
    if not os.path.exists(MODELO):
        print(f"[⚠️] No se encontró '{MODELO}'. El stream va a funcionar SIN IA.")
        print("     Para generarlo: capturarDataset.py (vacio/lleno) -> entrenarModelo.py -> copiar el .tflite acá.")
        return None
    try:
        clasificador = ClasificadorAlmacenamiento(MODELO)
        print(f"[🤖] Modelo '{MODELO}' cargado. IA activa.")
        return clasificador
    except Exception as e:
        print(f"[⚠️] No se pudo cargar la IA ({e}). El stream va a funcionar SIN IA.")
        print("     Instalá el intérprete con: pip install ai-edge-litert  (o tflite-runtime)")
        return None


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


def dibujar(img, texto):
    big = cv2.resize(img, STREAM_SIZE, interpolation=cv2.INTER_NEAREST)
    color = cv2.applyColorMap(big, cv2.COLORMAP_JET)
    fx = STREAM_SIZE[0] / img.shape[1]
    fy = STREAM_SIZE[1] / img.shape[0]
    x1, y1, x2, y2 = ZONA
    cv2.rectangle(color, (int(x1 * fx), int(y1 * fy)), (int(x2 * fx) - 1, int(y2 * fy) - 1), (0, 255, 0), 2)
    cv2.putText(color, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv2.putText(color, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return color


def monitorear():
    global latest_frame
    clasificador = cargar_clasificador()
    monitor = MonitorLlenado(TIEMPO_ALERTA)
    prob_suave = 0.0
    frames_count = 0

    for img in leer_frames():
        try:
            frames_count += 1
            if clasificador is None:
                texto = "SIN MODELO (solo camara)"
                print(f"📷 [STREAM ACTIVO] Frame #{frames_count} | sin IA   ", end="\r")
            else:
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
                texto = f"{'LLENO' if lleno else 'VACIO'} {prob_suave * 100:.0f}%  {segundos:.0f}/{TIEMPO_ALERTA}s"
                print(f"🤖 IA: {'LLENO' if lleno else 'vacío'} ({prob_suave * 100:5.1f}%) | "
                      f"lleno hace {segundos:4.1f}/{TIEMPO_ALERTA} s   ", end="\r")

            _, jpeg = cv2.imencode('.jpg', dibujar(img, texto))
            with frame_lock:
                latest_frame = jpeg.tobytes()
        except Exception as e:
            # Un error en un frame no tiene que frenar el stream
            print(f"\n[⚠️] Error procesando un frame: {e}")
            time.sleep(0.2)


class StreamingHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global latest_frame
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
            self.send_header('Age', '0')
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            try:
                while True:
                    with frame_lock:
                        frame = latest_frame

                    if frame is not None:
                        self.wfile.write(b'--frame\r\n')
                        self.send_header('Content-Type', 'image/jpeg')
                        self.send_header('Content-Length', str(len(frame)))
                        self.end_headers()
                        self.wfile.write(frame)
                        self.wfile.write(b'\r\n')

                    time.sleep(0.125)
            except Exception:
                pass
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        return


def hilo_camara():
    try:
        monitorear()
    except Exception:
        print("\n[❌] El hilo de la cámara se detuvo por un error:")
        traceback.print_exc()


def run_server():
    server = HTTPServer(('0.0.0.0', PUERTO_WEB), StreamingHandler)
    print("\n[🌍] Servidor Web Activo!")
    print(f"     👉 Ingresá en tu navegador a: http://<IP-DE-LA-RADXA>:{PUERTO_WEB}\n")
    server.serve_forever()


if __name__ == '__main__':
    thr = threading.Thread(target=hilo_camara, daemon=True)
    thr.start()
    try:
        run_server()
    except KeyboardInterrupt:
        print("\n👋 Cerrando...")
