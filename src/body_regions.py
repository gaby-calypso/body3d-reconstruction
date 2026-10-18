"""
Regiones del cuerpo sobre la malla (reshaper o SMPL): torso, brazos y piernas.

Se usa en tres lugares y por eso vive aparte (solo numpy):
  - el visor 3D (etiquetas ancladas al lugar correcto),
  - la comparación real vs referencia (diferencia media por región),
  - la circunferencia de referencia del brazo (el modelo de regresión no
    tiene brazo; se mide en la malla de referencia).

Convención: la malla está "alineada" (ver face_forward): y hacia arriba, el
frente mira a +Z. Con la figura de frente, el lado izquierdo de la PERSONA
queda en +X (derecha de la pantalla).
"""

import math

import numpy as np

# Bandas de altura (fracción 0 = pies, 1 = coronilla) para las estadísticas
# de superficie / volumen por zona. Siguen la anatomía de las mallas del
# reshaper (cuello ≈ 0.86, pecho ≈ 0.70, cintura ≈ 0.58, cadera ≈ 0.50,
# rodilla ≈ 0.29), a diferencia de las bandas anteriores, que caían en
# zonas equivocadas (p. ej. «cadera» sobre el muslo).
BODY_BANDS = {
    "cabeza":  (0.88, 1.00),
    "cuello":  (0.82, 0.88),
    "pecho":   (0.64, 0.82),
    "cintura": (0.54, 0.64),
    "cadera":  (0.44, 0.54),
    "muslos":  (0.27, 0.44),
    "piernas": (0.00, 0.27),
}

# Alturas típicas (fracción de la estatura) usadas cuando no se detectan
KNEE_FRAC = 0.285
ARM_MEASURE_FRAC = 0.68      # punto de medida del brazo (parte alta, bajo la axila)
CROTCH_FRAC = 0.47
_UPPER = 0.84                # por encima de esto ya es cuello / hombros


# ─────────────────────────────────────────────────────────────────────────
# Orientación
# ─────────────────────────────────────────────────────────────────────────
def face_forward(v):
    """
    Gira la malla sobre el eje vertical (Y) para que mire exactamente hacia
    +Z; así 0° es SIEMPRE la vista frontal, 90° la lateral y 180° la posterior.

    SMPL ya sale mirando a +Z, pero la malla del reshaper sale girada (mira a
    -Z y con un giro de decenas de grados). A la altura de la cadera la
    sección es más ancha (izquierda-derecha) que profunda (adelante-atrás): el
    eje menor del análisis de componentes principales es el adelante-atrás; el
    sentido se decide con los pies (los dedos quedan por delante del tobillo).
    """
    v = np.asarray(v, dtype=np.float32)
    y0, y1 = float(v[:, 1].min()), float(v[:, 1].max())
    h = y1 - y0
    if h < 1e-6:
        return v
    fr = (v[:, 1] - y0) / h
    hip = v[(fr > 0.47) & (fr < 0.60)][:, [0, 2]]
    foot = v[fr < 0.03]
    ankle = v[(fr > 0.06) & (fr < 0.10)]
    if len(hip) < 30 or len(foot) < 10 or len(ankle) < 10:
        return v
    c = hip - hip.mean(axis=0)
    w, vec = np.linalg.eigh(np.cov(c.T))
    a = vec[:, 0].astype(np.float64)
    d = (foot[:, [0, 2]].mean(axis=0) - ankle[:, [0, 2]].mean(axis=0))
    if float(d @ a) < 0:
        a = -a
    a /= max(np.linalg.norm(a), 1e-9)
    cs, sn = float(a[1]), float(-a[0])
    if abs(math.degrees(math.atan2(sn, cs))) < 2.0:
        return v
    x = v[:, 0] * cs + v[:, 2] * sn
    z = -v[:, 0] * sn + v[:, 2] * cs
    out = v.copy()
    out[:, 0], out[:, 2] = x, z
    return out


# ─────────────────────────────────────────────────────────────────────────
# Geometría de las rebanadas
# ─────────────────────────────────────────────────────────────────────────
def _hull_perimeter(pts):
    """Perímetro de la envolvente convexa de puntos 2D (cadena monótona)."""
    p = sorted(map(tuple, np.asarray(pts, dtype=np.float64)))
    if len(p) < 3:
        return 0.0

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for q in p:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], q) <= 0:
            lower.pop()
        lower.append(q)
    upper = []
    for q in reversed(p):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], q) <= 0:
            upper.pop()
        upper.append(q)
    hull = lower[:-1] + upper[:-1]
    per = 0.0
    for i in range(len(hull)):
        a, b = hull[i], hull[(i + 1) % len(hull)]
        per += float(np.hypot(a[0] - b[0], a[1] - b[1]))
    return per


def torso_edges(u, w, height, bin_frac=0.008):
    """
    Bordes (izquierdo<0, derecho>0) del TRONCO en una rebanada horizontal.

    u: coordenada horizontal de la pantalla; w: profundidad. El tronco tiene
    mucho espesor en profundidad hasta cerca de su borde; un brazo pegado al
    costado es delgado, así que el borde del tronco es donde el espesor cae
    por debajo de la mitad del espesor central. Para una elipse eso ocurre en
    ≈0.87 del semieje, por eso se divide entre 0.9.
    Devuelve (borde_neg, borde_pos) o None.
    """
    u = np.asarray(u); w = np.asarray(w)
    if len(u) < 8:
        return None
    bw = bin_frac * height
    b = np.floor(u / bw).astype(int)
    lo, hi = int(b.min()), int(b.max())
    spread = {}
    for k in range(lo, hi + 1):
        m = b == k
        spread[k] = float(w[m].max() - w[m].min()) if m.sum() >= 2 else 0.0
    s0 = float(np.median([spread.get(k, 0.0) for k in range(-3, 3)]))
    if s0 <= 0:
        return None
    res = []
    for sgn in (1, -1):
        k = 0 if sgn > 0 else -1
        last, miss = k, 0
        while lo <= k <= hi:
            if spread.get(k, 0.0) >= 0.5 * s0:
                last, miss = k, 0
            else:
                miss += 1
                if miss >= 2:
                    break
            k += sgn
        res.append(((last + 1) if sgn > 0 else last) * bw)
    return res[1] / 0.9, res[0] / 0.9          # (neg, pos)


class BodyRegions:
    """Regiones y alturas de una malla alineada (y arriba, frente a +Z)."""

    def __init__(self, verts):
        v = np.asarray(verts, dtype=np.float32)
        self.v = v
        self.y0 = float(v[:, 1].min())
        self.height = max(float(v[:, 1].max()) - self.y0, 1e-6)
        self.frac = (v[:, 1] - self.y0) / self.height
        self._edge_profile()
        self._masks()
        self.levels = self._levels()

    # ── bordes del tronco por altura ─────────────────────────────────────
    def _edge_profile(self):
        fs, en, ep = [], [], []
        for f in np.arange(0.40, _UPPER + 1e-9, 0.01):
            m = np.abs(self.frac - f) < 0.008
            e = torso_edges(self.v[m, 0], self.v[m, 2], self.height)
            if e is None:
                continue
            fs.append(f); en.append(e[0]); ep.append(e[1])
        self._fs = np.array(fs)
        if len(fs) >= 5:
            def med(a, k=2):
                a = np.asarray(a)
                return np.array([np.median(a[max(0, i - k): i + k + 1]) for i in range(len(a))])
            self._en, self._ep = med(en), med(ep)
        else:
            self._en = self._ep = None

    def edges_at(self, f):
        """(borde_neg, borde_pos) del tronco a la altura f (eje X de la malla)."""
        if self._en is None:
            return -0.15 * self.height * 0.5, 0.15 * self.height * 0.5
        return (float(np.interp(f, self._fs, self._en)),
                float(np.interp(f, self._fs, self._ep)))

    # ── máscaras ─────────────────────────────────────────────────────────
    def _masks(self):
        v, fr = self.v, self.frac
        n = len(v)
        torso = np.ones(n, dtype=bool)
        arm = np.zeros(n, dtype=bool)
        tol = 0.012 * self.height
        if self._en is not None:
            band = (fr >= self._fs[0]) & (fr <= self._fs[-1])
            en = np.interp(fr[band], self._fs, self._en)
            ep = np.interp(fr[band], self._fs, self._ep)
            x = v[band, 0]
            out = (x < en - tol) | (x > ep + tol)
            idx = np.where(band)[0]
            torso[idx[out]] = False
            arm[idx[out]] = True
        self.torso_mask = torso
        self.arm_mask = {"izq": arm & (v[:, 0] > 0), "der": arm & (v[:, 0] < 0)}
        # las manos cuelgan a la altura de la cadera pero lejos del eje: no son pierna
        leg = (fr < CROTCH_FRAC) & ~arm & (np.abs(v[:, 0]) < 0.14 * self.height)
        self.leg_mask = {"izq": leg & (v[:, 0] > 0), "der": leg & (v[:, 0] < 0)}

    # ── alturas de las zonas ─────────────────────────────────────────────
    def _slice(self, f, half=0.008):
        return self.v[np.abs(self.frac - f) < half]

    def _torso_perimeter(self, f):
        pts = self._slice(f, 0.006)
        if len(pts) < 8:
            return None
        en, ep = self.edges_at(f)
        pts = pts[(pts[:, 0] >= en) & (pts[:, 0] <= ep)]
        if len(pts) < 8:
            return None
        return _hull_perimeter(pts[:, [0, 2]])

    def _levels(self):
        v, fr = self.v, self.frac
        # cuello: corte más estrecho entre cabeza y hombros
        neck, best = None, None
        for f in np.arange(0.76, 0.925, 0.005):
            m = np.abs(fr - f) < 0.008
            if m.sum() < 6:
                continue
            wdt = float(v[m, 0].max() - v[m, 0].min())
            if best is None or wdt < best:
                neck, best = float(f), wdt
        if neck is None or best > 0.16 * self.height:
            neck = 0.86
        # pecho: punto más saliente hacia el frente en la parte alta del torso
        chest, cs = None, None
        for f in np.arange(0.78 * neck, 0.86 * neck + 1e-9, 0.005):
            near = self._slice(f)
            near = near[np.abs(near[:, 0]) < 0.14 * self.height]
            if len(near) < 6:
                continue
            sc = float(near[:, 2].max())
            if cs is None or sc > cs:
                chest, cs = float(f), sc
        chest = chest if chest is not None else 0.82 * neck
        # cadera (nivel glúteo): máxima proyección de los glúteos hacia atrás
        hip, hs = None, None
        for f in np.arange(0.42, 0.57, 0.005):
            near = self._slice(f)
            near = near[np.abs(near[:, 0]) < 0.12 * self.height]
            if len(near) < 6:
                continue
            sc = float(near[:, 2].min())
            if hs is None or sc < hs:
                hip, hs = float(f), sc
        hip = hip if hip is not None else 0.50
        # cintura: perímetro mínimo del tronco entre cadera y pecho
        waist, ws = None, None
        for f in np.arange(hip + 0.05, chest - 0.06 + 1e-9, 0.005):
            p = self._torso_perimeter(f)
            if p is None:
                continue
            if ws is None or p < ws:
                waist, ws = float(f), p
        waist = waist if waist is not None else (chest + hip) / 2.0
        return {"cuello": neck, "pecho": chest, "cintura": waist, "cadera": hip,
                "muslo": (CROTCH_FRAC + KNEE_FRAC) / 2.0, "rodilla": KNEE_FRAC,
                "brazo": ARM_MEASURE_FRAC}

    # ── anclajes (punto 3D donde se señala cada zona) ───────────────────
    def torso_anchor(self, zone, side):
        """Punto 3D sobre el contorno del tronco (lado +1 / −1 de X)."""
        f = self.levels[zone]
        if zone == "cuello":              # el cuello está solo: su borde es el de la rebanada
            pts = self._slice(f, 0.01)
            if len(pts) < 3:
                return np.array([0.0, self.y0 + f * self.height, 0.0], dtype=np.float32)
            x = 0.85 * (float(pts[:, 0].max()) if side > 0 else float(pts[:, 0].min()))
            near = pts[np.abs(pts[:, 0] - x) < 0.02 * self.height]
            z = float(near[:, 2].mean()) if len(near) else float(pts[:, 2].mean())
            return np.array([x, self.y0 + f * self.height, z], dtype=np.float32)
        en, ep = self.edges_at(f)
        x = 0.88 * (ep if side > 0 else en)
        pts = self._slice(f, 0.01)
        pts = pts[(pts[:, 0] >= min(en, 0)) & (pts[:, 0] <= max(ep, 0))]
        if len(pts) == 0:
            return np.array([x, self.y0 + f * self.height, 0.0])
        near = pts[np.abs(pts[:, 0] - x) < 0.02 * self.height]
        z = float(near[:, 2].mean()) if len(near) else float(pts[:, 2].mean())
        return np.array([x, self.y0 + f * self.height, z], dtype=np.float32)

    def limb_anchor(self, zone, side):
        """Centro de la sección del brazo / muslo / rodilla de ese lado."""
        key = "izq" if side > 0 else "der"
        if zone == "brazo":
            g = self.arm_geometry(key)
            return None if g is None else g["center"]
        f = self.levels[zone]
        pts = self._slice(f, 0.01)
        pts = pts[self.leg_mask[key][np.abs(self.frac - f) < 0.01]]
        if len(pts) < 3:
            return None
        return pts.mean(axis=0)

    # ── brazo ────────────────────────────────────────────────────────────
    def arm_geometry(self, key):
        """
        Centro y circunferencia (cm) de la parte alta del brazo ('izq'/'der').
        La sección se corta con un plano perpendicular al eje del brazo.
        """
        m = self.arm_mask[key] & (self.frac < 0.77) & (self.frac > 0.50)
        pts = self.v[m]
        if len(pts) < 30:
            return None
        c0 = pts.mean(axis=0)
        w, vec = np.linalg.eigh(np.cov((pts - c0).T))
        d = vec[:, int(np.argmax(w))]
        if d[1] > 0:
            d = -d                                     # apunta hacia abajo
        sel = self.arm_mask[key] & (np.abs(self.frac - ARM_MEASURE_FRAC) < 0.008)
        if sel.sum() < 5:
            return None
        c = self.v[sel].mean(axis=0)
        allp = self.v[self.arm_mask[key]]
        slab = allp[np.abs((allp - c) @ d) < 0.0075 * self.height]
        if len(slab) < 8:
            return None
        e1 = np.cross(d, [0.0, 0.0, 1.0])
        if np.linalg.norm(e1) < 1e-6:
            e1 = np.cross(d, [1.0, 0.0, 0.0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(d, e1)
        q = slab - c
        per = _hull_perimeter(np.c_[q @ e1, q @ e2])
        return {"center": c.astype(np.float32), "axis": d,
                "circumference_cm": float(per * 100.0), "n": int(len(slab))}

    # ── estadísticas por región (diferencia media real − referencia) ─────
    def region_stats(self, signed_dists):
        """
        {zona: {"mean_cm": …}} con la diferencia media (cm) de cada región,
        medida en el MISMO lugar donde el visor señala cada zona.
        """
        d = np.asarray(signed_dists, dtype=np.float64) * 100.0
        fr = self.frac
        out = {}

        def put(name, mask):
            if mask.sum() >= 5:
                out[name] = {"mean_cm": float(d[mask].mean())}

        for zone, half in (("cuello", 0.03), ("pecho", 0.04),
                           ("cintura", 0.035), ("cadera", 0.04)):
            put(zone, self.torso_mask & (np.abs(fr - self.levels[zone]) < half))
        for key in ("izq", "der"):
            put(f"brazo_{key}", self.arm_mask[key] & (fr > 0.55) & (fr < 0.77))
            put(f"muslo_{key}", self.leg_mask[key] & (np.abs(fr - self.levels["muslo"]) < 0.045))
            put(f"rodilla_{key}", self.leg_mask[key] & (np.abs(fr - self.levels["rodilla"]) < 0.035))
        return out


def arm_reference_cm(verts):
    """
    Circunferencia (cm) del brazo izquierdo y derecho de una malla (p. ej. la
    de referencia). Devuelve {"izq": cm|None, "der": cm|None}.
    """
    reg = BodyRegions(face_forward(verts))
    out = {}
    for key in ("izq", "der"):
        g = reg.arm_geometry(key)
        out[key] = None if g is None else g["circumference_cm"]
    return out