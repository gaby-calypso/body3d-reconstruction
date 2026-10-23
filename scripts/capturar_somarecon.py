"""
capturar_somarecon.py
Recorre SomaRecon 3D de forma automática con un paciente ya cargado y guarda:
  - cada pantalla completa y cada panel en PNG de alta resolución (3x)
  - las vistas frontal, lateral y posterior de la malla
  - un video MP4 de la malla girando (sin cursor, fluido)

Funciona si la interfaz de SomaRecon es web (HTML/JS: pywebview, Flask, Electron, Tauri
o un index.html local). Si usas PyQt/Tkinter, mira la nota al final.

Instalación (una vez, en la Terminal de tu Mac):
    pip install playwright
    python -m playwright install chromium
    brew install ffmpeg          # para armar el video

Uso:
    1. Arranca SomaRecon como siempre (o su servidor) y anota la dirección, p. ej. http://127.0.0.1:5000
    2. Ajusta la sección CONFIGURACIÓN de abajo.
    3. python capturar_somarecon.py
"""

import asyncio, subprocess, shutil
from pathlib import Path
from playwright.async_api import async_playwright

# ======================= CONFIGURACIÓN =======================
URL = "http://127.0.0.1:5000"            # dirección de la interfaz (o "file:///ruta/index.html")
SALIDA = Path("output/capturas_somarecon")
ESCALA = 3                                # 3 = triple resolución (≈ 4800 px de ancho a 1600 px de ventana)
VENTANA = {"width": 1600, "height": 1000}

PACIENTE = {
    "nombre": "Voluntaria 01",            # usa un código, no el nombre real
    "sexo": "Femenino",
    "edad": "30",
    "peso": "65.0",
    "talla": "1.65",
}
# Imágenes RGB y de profundidad que ya tienes guardadas para ese paciente
IMAGENES = ["data/captura_demo/frontal_rgb.png", "data/captura_demo/frontal_depth.npy"]
# Textos de los botones tal como aparecen en la interfaz (cámbialos si difieren)
BOTON = {
    "importar": "Importar imágenes",
    "continuar_mediciones": "Continuar a Mediciones",
    "detectar": "Detectar landmarks (MediaPipe)",
    "pipeline": "Ejecutar pipeline",
    "ver_resultados": "Ver resultados",
    "frontal": "Frontal", "lateral": "Lateral", "posterior": "Posterior",
    "girar": "Girar",
}
# Paneles a recortar en cada pantalla: nombre -> texto que aparece dentro del panel
PANELES = {
    "1_captura":    {"datos_paciente": "Datos del paciente", "vista": "Vista frontal"},
    "2_mediciones": {"mapa": "Landmarks y zonas de medida", "circunferencias": "Circunferencias"},
    "3_reconstruccion": {"visor": "Malla real frente a referencia", "volumen": "Volumen por zona",
                         "mayor_diferencia": "Mayor diferencia de volumen"},
}
VIDEO_SEG, VIDEO_FPS = 12, 30             # duración y cuadros por segundo del video de la malla
# =============================================================


async def clic(pg, texto, espera=800):
    await pg.get_by_text(texto, exact=False).first.click()
    await pg.wait_for_timeout(espera)


async def panel(pg, texto, archivo):
    """Recorta la tarjeta/panel más cercano que contiene el texto."""
    el = pg.get_by_text(texto, exact=False).first
    caja = el.locator("xpath=ancestor::*[self::section or self::div][1]/..")
    try:
        await caja.screenshot(path=archivo, animations="disabled")
    except Exception:
        await el.screenshot(path=archivo)


async def pantalla(pg, nombre):
    await pg.mouse.move(0, 0)  # aleja el cursor de cualquier hover
    await pg.screenshot(path=SALIDA / f"{nombre}_completa.png", full_page=True, animations="disabled")
    for clave, texto in PANELES.get(nombre, {}).items():
        await panel(pg, texto, SALIDA / f"{nombre}_{clave}.png")
    print("✓", nombre)


async def llenar_paciente(pg):
    campos = {"nombre": "Nombre del paciente"}
    await pg.get_by_placeholder(campos["nombre"]).fill(PACIENTE["nombre"])
    # Los demás campos se buscan por su etiqueta visible; ajusta si tu HTML usa otros nombres
    for etiqueta, valor in [("Edad", PACIENTE["edad"]), ("Peso", PACIENTE["peso"]), ("Talla", PACIENTE["talla"])]:
        try:
            await pg.get_by_label(etiqueta, exact=False).fill(valor)
        except Exception:
            pass


async def video_malla(pg):
    """Graba el visor 3D cuadro por cuadro mientras gira y arma un MP4."""
    cuadros = SALIDA / "_cuadros"
    cuadros.mkdir(exist_ok=True)
    visor = pg.locator("canvas").first
    await clic(pg, BOTON["girar"], espera=300)
    n = VIDEO_SEG * VIDEO_FPS
    for i in range(n):
        await visor.screenshot(path=cuadros / f"f_{i:04d}.png")
        await pg.wait_for_timeout(1000 / VIDEO_FPS)
    await clic(pg, BOTON["girar"], espera=200)  # detener giro
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", str(VIDEO_FPS),
                        "-i", str(cuadros / "f_%04d.png"),
                        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
                        "-c:v", "libx264", "-crf", "17", "-movflags", "+faststart",
                        str(SALIDA / "malla_girando.mp4")], check=True)
        shutil.rmtree(cuadros)
        print("✓ video malla_girando.mp4")
    else:
        print("Instala ffmpeg para armar el video; los cuadros quedaron en", cuadros)


async def main():
    SALIDA.mkdir(exist_ok=True)
    async with async_playwright() as p:
        nav = await p.chromium.launch(args=["--use-gl=angle", "--enable-webgl", "--ignore-gpu-blocklist"])
        pg = await nav.new_page(viewport=VENTANA, device_scale_factor=ESCALA)
        await pg.goto(URL)
        await pg.wait_for_load_state("networkidle")

        # 1 · Captura
        await llenar_paciente(pg)
        await pg.locator("input[type=file]").first.set_input_files(IMAGENES)
        await pg.wait_for_timeout(1500)
        await pantalla(pg, "1_captura")

        # 2 · Mediciones
        await clic(pg, BOTON["continuar_mediciones"])
        await clic(pg, BOTON["detectar"], espera=4000)
        await clic(pg, BOTON["pipeline"], espera=8000)   # súbelo si el pipeline tarda más
        await pantalla(pg, "2_mediciones")

        # 3 · Reconstrucción
        await clic(pg, BOTON["ver_resultados"], espera=5000)
        await pantalla(pg, "3_reconstruccion")
        visor = pg.locator("canvas").first
        for vista in ("frontal", "lateral", "posterior"):
            await clic(pg, BOTON[vista], espera=1200)
            await visor.screenshot(path=SALIDA / f"malla_{vista}.png")
        await video_malla(pg)
        await nav.close()
    print("Listo. Archivos en", SALIDA.resolve())


if __name__ == "__main__":
    asyncio.run(main())

# ---------------------------------------------------------------------------
# Si SomaRecon está hecho con PyQt / PySide (no web), agrega esta función a tu
# ventana principal y llámala desde un botón oculto o un atajo de teclado:
#
#   def exportar_pantalla(self, ruta, escala=3):
#       img = self.grab()                               # captura la ventana sin bordes de macOS
#       img = img.scaled(img.width()*escala//img.devicePixelRatio(), ...)  # opcional
#       img.save(ruta, "PNG")
#
# y para la malla (PyVista / Open3D), guarda el render directo, no la pantalla:
#   plotter.screenshot("malla_frontal.png", scale=3, transparent_background=True)
#   plotter.open_movie("malla_girando.mp4", framerate=30); plotter.orbit_on_path(write_frames=True)
# ---------------------------------------------------------------------------