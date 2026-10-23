"""
captura_qt.py - Capturas en alta resolucion para SomaRecon 3D (interfaz Qt).

Se activa desde app.py con dos lineas (ver instrucciones) y agrega atajos:
  Cmd+Shift+S  guarda la ventana completa (3x), cada panel con titulo,
               cada grafica de matplotlib (300 dpi) y el visor 3D.
  Cmd+Shift+V  con el cursor sobre el visor de la malla:
               - si es matplotlib 3D: genera solo el video girando 360 grados
                 + vistas frontal, lateral y posterior en alta resolucion.
               - si es otro visor (PyVista/VTK/OpenGL): empieza a grabar;
                 gira la malla con el mouse y vuelve a presionar para terminar.
Todo se guarda en output/capturas_somarecon/
"""
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ======================= CONFIGURACION =======================
SALIDA = Path("output/capturas_somarecon")
ESCALA = 3            # 3 = triple resolucion de pantalla
DPI_GRAFICAS = 300    # resolucion de las graficas de matplotlib
FPS = 30              # cuadros por segundo del video
SEG_GIRO = 12         # duracion del giro de 360 grados
# =============================================================


def _qt():
    nombre = next((n for n in ("PySide6", "PyQt6", "PyQt5", "PySide2") if n in sys.modules), "PyQt5")
    mods = [__import__(f"{nombre}.{m}", fromlist=["_"]) for m in ("QtCore", "QtGui", "QtWidgets")]
    return mods


def _e(obj, grupo, nombre):
    g = getattr(obj, grupo, None)
    return getattr(g, nombre) if g is not None and hasattr(g, nombre) else getattr(obj, nombre)


def _limpio(texto):
    return re.sub(r"[^\w]+", "_", texto.lower()).strip("_")[:40] or "panel"


def _carpeta_nueva(d):
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    return d


def _es_matplotlib(w):
    return hasattr(w, "figure") and any(c.__name__.startswith("FigureCanvas") for c in type(w).__mro__)


def _es_3d(w):
    if _es_matplotlib(w):
        return False
    return any(k in c.__name__ for c in type(w).__mro__ for k in ("OpenGL", "VTK", "Interactor", "GLView", "WebEngine", "Quick"))


def _ffmpeg():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg          # pip install imageio-ffmpeg (trae ffmpeg incluido)
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _armar_mp4(tmp, destino, fps=FPS):
    exe = _ffmpeg()
    if not exe:
        print("  Falta ffmpeg: ejecuta  pip3 install imageio-ffmpeg . Los cuadros quedaron en", tmp)
        return
    subprocess.run([exe, "-y", "-v", "error", "-framerate", f"{fps:.2f}",
                    "-i", str(tmp / "f_%04d.png"),
                    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2:color=white,format=yuv420p",
                    "-c:v", "libx264", "-crf", "17", "-movflags", "+faststart", str(destino)],
                   check=True)
    shutil.rmtree(tmp)
    print("  ✓ video", destino)


def instalar_capturas():
    QtCore, QtGui, QtWidgets = _qt()
    Qt, QEvent = QtCore.Qt, QtCore.QEvent
    KEYPRESS = _e(QEvent, "Type", "KeyPress")
    CTRL = _e(Qt, "KeyboardModifier", "ControlModifier")   # en Mac = Cmd
    SHIFT = _e(Qt, "KeyboardModifier", "ShiftModifier")
    KEY_S, KEY_V = _e(Qt, "Key", "Key_S"), _e(Qt, "Key", "Key_V")
    BLANCO = _e(Qt, "GlobalColor", "white")
    app = QtWidgets.QApplication.instance()

    class Capturador(QtCore.QObject):
        def __init__(s):
            super().__init__(app)
            s.n, s.ult, s.grab = 0, 0.0, None

        # ---------- atajos ----------
        def eventFilter(s, obj, ev):
            if ev.type() == KEYPRESS and not ev.isAutoRepeat():
                m = ev.modifiers()
                if (m & CTRL) == CTRL and (m & SHIFT) == SHIFT and ev.key() in (KEY_S, KEY_V):
                    if time.time() - s.ult > 0.5:
                        s.ult = time.time()
                        try:
                            s.capturar() if ev.key() == KEY_S else s.video()
                        except Exception as err:
                            print("  ✗ Error en la captura:", err)
                    return True
            return False

        # ---------- utilidades ----------
        def ventana(s):
            w = app.activeWindow()
            if w is None:
                ws = [x for x in app.topLevelWidgets() if x.isVisible()]
                w = max(ws, key=lambda x: x.width() * x.height()) if ws else None
            return w

        def render(s, w, ruta):
            sz = w.size()
            pix = QtGui.QPixmap(sz.width() * ESCALA, sz.height() * ESCALA)
            pix.setDevicePixelRatio(ESCALA)
            pix.fill(BLANCO)
            w.render(pix)
            pix.save(str(ruta), "PNG")

        def foto_3d(s, w, ruta, escala=ESCALA):
            if hasattr(w, "screenshot"):                 # PyVista
                try:
                    w.screenshot(str(ruta), scale=escala)
                    return
                except Exception:
                    pass
            if hasattr(w, "grabFramebuffer"):            # QOpenGLWidget
                w.grabFramebuffer().save(str(ruta))
                return
            top = w.window()                             # recorta de la ventana (funciona con WebEngine)
            r = QtCore.QRect(w.mapTo(top, QtCore.QPoint(0, 0)), w.size())
            top.grab(r).save(str(ruta))

        # ---------- Cmd+Shift+S ----------
        def capturar(s):
            win = s.ventana()
            if win is None:
                return
            s.n += 1
            d = SALIDA / f"pantalla_{s.n:02d}"
            d.mkdir(parents=True, exist_ok=True)
            app.processEvents()
            s.render(win, d / "00_ventana_completa.png")
            i = 0
            for w in win.findChildren(QtWidgets.QWidget):
                if not w.isVisible() or w.width() < 80 or w.height() < 80:
                    continue
                if _es_matplotlib(w):
                    i += 1
                    w.figure.savefig(d / f"{i:02d}_grafica.png", dpi=DPI_GRAFICAS,
                                     bbox_inches="tight", facecolor="white")
                elif _es_3d(w):
                    i += 1
                    s.foto_3d(w, d / f"{i:02d}_visor3d.png")
                elif isinstance(w, QtWidgets.QGroupBox) and w.title():
                    i += 1
                    s.render(w, d / f"{i:02d}_{_limpio(w.title())}.png")
                elif isinstance(w, QtWidgets.QLabel):
                    p = w.pixmap()
                    if p is not None and not p.isNull() and p.width() >= 200:
                        i += 1
                        p.save(str(d / f"{i:02d}_imagen.png"))
            print(f"  ✓ {i + 1} imagenes guardadas en {d}")

        # ---------- Cmd+Shift+V ----------
        def video(s):
            if s.grab:
                return s.parar()
            win = s.ventana()
            if win is None:
                return
            cur = QtGui.QCursor.pos()
            # todos los widgets visibles (>=300 px) que estan bajo el cursor
            cand = [w for w in win.findChildren(QtWidgets.QWidget)
                    if w.isVisible() and w.width() >= 300 and w.height() >= 300
                    and w.rect().contains(w.mapFromGlobal(cur))]
            visores = [w for w in cand if _es_matplotlib(w) or _es_3d(w)]
            grupo = visores or cand
            x = min(grupo, key=lambda w: w.width() * w.height()) if grupo else None
            if x is None:
                print("  Pon el cursor sobre el visor de la malla y presiona Cmd+Shift+V otra vez")
                return
            print(f"  Visor detectado: {type(x).__name__} ({x.width()}x{x.height()})")
            if _es_matplotlib(x) and any(getattr(a, "name", "") == "3d" for a in x.figure.axes):
                return s.giro_matplotlib(x)
            s.iniciar(x)

        def giro_matplotlib(s, canvas):
            fig = canvas.figure
            ejes = [a for a in fig.axes if getattr(a, "name", "") == "3d"]
            orig = [(a.elev, a.azim) for a in ejes]
            SALIDA.mkdir(parents=True, exist_ok=True)

            def girar(delta):
                for a, (el, az) in zip(ejes, orig):
                    a.view_init(elev=el, azim=az + delta)

            for nombre, delta in (("frontal", 0), ("lateral", 90), ("posterior", 180)):
                girar(delta)
                fig.savefig(SALIDA / f"malla_{nombre}.png", dpi=DPI_GRAFICAS,
                            bbox_inches="tight", transparent=True)
            tmp = _carpeta_nueva(SALIDA / "_cuadros")
            n = FPS * SEG_GIRO
            for k in range(n):
                girar(360 * k / n)
                fig.savefig(tmp / f"f_{k:04d}.png", dpi=200, facecolor="white")
                if k % 60 == 0:
                    print(f"  generando video... {k}/{n}")
                    app.processEvents()
            girar(0)
            canvas.draw_idle()
            print("  ✓ vistas frontal, lateral y posterior guardadas")
            _armar_mp4(tmp, SALIDA / "malla_girando.mp4")

        def iniciar(s, w):
            tmp = _carpeta_nueva(SALIDA / "_cuadros")
            t = QtCore.QTimer(s)
            s.grab = {"w": w, "k": 0, "tmp": tmp, "t": t, "t0": time.time()}
            t.timeout.connect(s.cuadro)
            t.start(int(1000 / FPS))
            print("  ● Grabando: gira la malla con el mouse y presiona Cmd+Shift+V para terminar")

        def cuadro(s):
            g = s.grab
            s.foto_3d(g["w"], g["tmp"] / f"f_{g['k']:04d}.png", escala=1)
            g["k"] += 1

        def parar(s):
            g, s.grab = s.grab, None
            g["t"].stop()
            fps_real = max(1.0, g["k"] / max(0.1, time.time() - g["t0"]))
            _armar_mp4(g["tmp"], SALIDA / f"malla_grabada_{time.strftime('%H%M%S')}.mp4", fps_real)

    cap = Capturador()
    app.installEventFilter(cap)
    globals()["_capturador"] = cap
    print("📸 Capturas activas: Cmd+Shift+S = pantallas | Cmd+Shift+V = video de la malla")
    return cap


if __name__ == "__main__":
    # Arma el video con cuadros que ya existen:  python3 scripts/captura_qt.py
    tmp = SALIDA / "_cuadros"
    if any(tmp.glob("f_*.png")):
        _armar_mp4(tmp, SALIDA / "malla_grabada.mp4", float(sys.argv[1]) if len(sys.argv) > 1 else FPS)
    else:
        print("No hay cuadros en", tmp)