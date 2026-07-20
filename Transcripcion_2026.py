import os
import re
import shutil
import subprocess
import sys
import time
import warnings
from tkinter import Button, Label, StringVar, Tk, filedialog, messagebox, simpledialog

os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import imageio_ffmpeg
from faster_whisper import WhisperModel

warnings.filterwarnings("ignore")


def recurso(*partes):
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, *partes)


MODELO = recurso("modelos", "small")


def ffmpeg_exe():
    ruta = shutil.which("ffmpeg")
    if ruta:
        return ruta
    return imageio_ffmpeg.get_ffmpeg_exe()


def ventana_oculta():
    root = Tk()
    root.withdraw()
    return root


def elegir_archivo():
    root = ventana_oculta()
    ruta = filedialog.askopenfilename(
        title="Selecciona el audio o video",
        filetypes=[
            ("Audio y video", "*.m4a *.mp3 *.aac *.wav *.ogg *.flac *.wma *.mp4 *.mov *.mkv"),
            ("Todos los archivos", "*.*"),
        ],
    )
    root.destroy()
    return ruta


def pedir_terminos():
    root = ventana_oculta()
    texto = simpledialog.askstring(
        "Términos esperados",
        "Opcional: escribe nombres, siglas o términos técnicos separados por comas.\n"
        "Ejemplo: OEFA, DSIS, fiscalización ambiental, lixiviados",
        parent=root,
    )
    root.destroy()
    return texto or ""


def confirmar_inicio():
    root = ventana_oculta()
    continuar = messagebox.askokcancel(
        "Transcripción 2026",
        "El programa usará Whisper small, incluido dentro del ejecutable.\n\n"
        "No instalará Python, no descargará el modelo y no necesita internet.\n"
        "La transcripción puede tardar según la duración del audio y la velocidad de la computadora.",
        parent=root,
    )
    root.destroy()
    return continuar


def mostrar_info(titulo, texto):
    root = ventana_oculta()
    messagebox.showinfo(titulo, texto, parent=root)
    root.destroy()


def mostrar_error(texto):
    try:
        root = ventana_oculta()
        messagebox.showerror("Transcripción 2026", texto, parent=root)
        root.destroy()
    except Exception:
        pass


def mejorar_audio(entrada, salida):
    filtros = (
        "highpass=f=80,"
        "lowpass=f=7800,"
        "afftdn=nf=-25,"
        "dynaudnorm=f=150:g=15:p=0.95,"
        "volume=1.5"
    )
    comando = [
        ffmpeg_exe(), "-y", "-i", entrada, "-vn", "-ac", "1", "-ar", "16000",
        "-af", filtros, salida,
    ]
    proceso = subprocess.run(comando, capture_output=True, text=True)
    if proceso.returncode != 0:
        detalle = proceso.stderr[-1500:] if proceso.stderr else "Error desconocido de FFmpeg."
        raise RuntimeError(f"No se pudo preparar el audio.\n\n{detalle}")


def limpiar(texto):
    texto = re.sub(r"\s+", " ", texto).strip()
    texto = re.sub(r"\s+([,.;:?!])", r"\1", texto)
    texto = re.sub(r"([,.;:?!])([^\s])", r"\1 \2", texto)
    return texto


def limpiar_final(texto):
    texto = limpiar(texto)
    texto = re.sub(r"\b(\w{2,})(?:\s+\1\b){2,}", r"\1", texto, flags=re.IGNORECASE)
    texto = re.sub(
        r"\b([\wáéíóúÁÉÍÓÚñÑüÜ]+(?:\s+[\wáéíóúÁÉÍÓÚñÑüÜ]+){1,5})\s+\1\b",
        r"\1", texto, flags=re.IGNORECASE,
    )
    return texto


def tiempo(segundos):
    segundos = int(segundos)
    return f"{segundos // 3600:02d}:{(segundos % 3600) // 60:02d}:{segundos % 60:02d}"


def sospechoso(segmento, texto):
    normal = texto.strip().lower().strip("¿?¡!.,;: ")
    duracion = max(0.01, float(segmento.end - segmento.start))
    if not normal:
        return True
    if normal in {"por qué", "porque", "por que", "gracias", "muchas gracias"} and duracion > 2:
        return True
    if len(normal.split()) <= 3 and duracion > 5:
        return True
    if getattr(segmento, "no_speech_prob", 0.0) > 0.65 and getattr(segmento, "avg_logprob", 0.0) < -0.8:
        return True
    if getattr(segmento, "avg_logprob", 0.0) < -1.2:
        return True
    if getattr(segmento, "compression_ratio", 0.0) > 2.6:
        return True
    return False


def prompt_inicial(terminos):
    texto = (
        "Transcripción en español latinoamericano. Mantén nombres propios, siglas y términos técnicos. "
        "Usa puntuación natural."
    )
    if terminos.strip():
        texto += f" Términos esperados: {terminos.strip()}."
    return texto


def transcribir(audio, terminos):
    modelo_bin = os.path.join(MODELO, "model.bin")
    if not os.path.isfile(modelo_bin):
        raise RuntimeError("El modelo Whisper no está incorporado correctamente en el ejecutable.")

    modelo = WhisperModel(
        MODELO,
        device="cpu",
        compute_type="int8",
        local_files_only=True,
    )
    segmentos, _ = modelo.transcribe(
        audio,
        language="es",
        task="transcribe",
        initial_prompt=prompt_inicial(terminos),
        condition_on_previous_text=False,
        vad_filter=True,
        vad_parameters={
            "threshold": 0.55,
            "min_speech_duration_ms": 300,
            "min_silence_duration_ms": 700,
            "speech_pad_ms": 200,
        },
        beam_size=5,
        temperature=0.0,
        no_speech_threshold=0.6,
        log_prob_threshold=-1.0,
        compression_ratio_threshold=2.4,
    )

    textos = []
    lineas = []
    for numero, segmento in enumerate(segmentos, start=1):
        texto = limpiar(segmento.text)
        if sospechoso(segmento, texto):
            continue
        textos.append(texto)
        linea = f"[{tiempo(segmento.start)} - {tiempo(segmento.end)}] {texto}"
        lineas.append(linea)
        print(f"{numero:04d} {linea}", flush=True)

    return limpiar_final(" ".join(textos)), "\n".join(lineas)


def guardar(ruta, contenido):
    with open(ruta, "w", encoding="utf-8-sig") as archivo:
        archivo.write(contenido)


def autoprueba():
    ejecutable_ffmpeg = ffmpeg_exe()
    if not ejecutable_ffmpeg or not os.path.isfile(ejecutable_ffmpeg):
        raise RuntimeError("FFmpeg no quedó incorporado correctamente.")
    modelo_bin = os.path.join(MODELO, "model.bin")
    if not os.path.isfile(modelo_bin):
        raise RuntimeError("El modelo Whisper no quedó incorporado correctamente.")
    WhisperModel(MODELO, device="cpu", compute_type="int8", local_files_only=True)
    print("AUTOPRUEBA_OK")
    return 0


def main():
    if "--self-test" in sys.argv:
        return autoprueba()

    entrada = elegir_archivo()
    if not entrada:
        return 0
    if not os.path.isfile(entrada):
        raise RuntimeError("El archivo seleccionado no existe.")

    terminos = pedir_terminos()
    if not confirmar_inicio():
        return 0

    carpeta = os.path.dirname(entrada)
    nombre = os.path.splitext(os.path.basename(entrada))[0]
    audio_mejorado = os.path.join(carpeta, f"{nombre}_mejorado.wav")
    texto_limpio = os.path.join(carpeta, f"{nombre}_transcripcion_mejorada_filtrada.txt")
    texto_tiempos = os.path.join(carpeta, f"{nombre}_transcripcion_mejorada_filtrada_con_tiempos.txt")

    inicio = time.time()
    print("1) Mejorando el audio...", flush=True)
    mejorar_audio(entrada, audio_mejorado)
    print("2) Transcribiendo...", flush=True)
    limpio, con_tiempos = transcribir(audio_mejorado, terminos)
    print("3) Guardando resultados...", flush=True)
    guardar(texto_limpio, limpio)
    guardar(texto_tiempos, con_tiempos)

    transcurrido = time.time() - inicio
    mostrar_info(
        "Proceso completado",
        f"La transcripción terminó en {transcurrido / 60:.1f} minutos.\n\n"
        f"Archivos guardados junto al audio:\n\n"
        f"{os.path.basename(texto_limpio)}\n"
        f"{os.path.basename(texto_tiempos)}\n"
        f"{os.path.basename(audio_mejorado)}",
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", flush=True)
        mostrar_error(f"Se produjo un error:\n\n{error}")
        raise
