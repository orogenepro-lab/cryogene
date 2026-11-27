"""
terrain_features.py - Descripteurs de terrain STATIQUES pour les cuvettes froides
et les inversions (chaine CUVETTE). Calcules une fois sur le MNT IGN d'OROGENE
(mnt_puy.tif, 100 m) et mis en cache dans terrain_features.npz.

Descripteurs (tous sur la grille du MNT, ligne 0 = nord) :
  alt        altitude (m)
  slope      pente (m/m) ; aspect (rad, 0 = nord, sens horaire) ; curv (courbure, m^-1)
  tpi_500 / tpi_2k / tpi_10k   position topographique = alt - moyenne a r (m) ; <0 = creux
  depth_2k / depth_10k         profondeur sous la crete : max a r - alt (m)
  svf        sky view factor 0-1 (16 azimuts, horizon jusqu'a SVF_RADIUS)
  cold_acc   accumulation d'air froid : flux de drainage descendant (log10 du nb de
             cellules amont), sur MNT lisse ; forte dans les fonds de vallee fermes
  dist_ridge distance a la ligne de crete locale (km) (proxy ventilation)

Usage :
    python terrain_features.py           # calcule + cache + PNG de controle (TPI 2 km, SVF, cold_acc)
    import terrain_features as TF ; F = TF.load()     # dict de cartes 2D + meta
"""
import os, sys
import numpy as np
from scipy.ndimage import (uniform_filter, maximum_filter, gaussian_filter,
                           distance_transform_edt)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd())

# ============================== CONFIG ======================================= #
MNT = "mnt_puy.tif"                 # produit par puy_de_dome.fetch_mnt_wms (100 m)
TARGET_RES = 100.0
OUT = os.environ.get("CUVETTE_TERRAIN", "terrain_features.npz")   # cache des descripteurs
TPI_RADII = {"tpi_500": 500, "tpi_2k": 2000, "tpi_10k": 10000, "tpi_20k": 20000}
COLD_ACC_SMOOTH = 400.0             # lissage (m) de l'accumulation : de la ligne de talweg a la vallee
DEPTH_RADII = {"depth_2k": 2000, "depth_10k": 10000}
SVF_RADIUS = 5000.0                 # portee de recherche d'horizon (m)
SVF_NAZ = 16
DRAIN_SMOOTH = 300.0                # lissage du MNT avant drainage (m) : bruit de 100 m
# ============================================================================ #


def _nanmean_filter(a, size_px):
    valid = np.isfinite(a).astype("float64")
    num = uniform_filter(np.where(valid > 0, a, 0.0), size_px, mode="nearest")
    den = uniform_filter(valid, size_px, mode="nearest")
    return np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)


def tpi(alt, radius_m, res):
    k = max(int(round(2 * radius_m / res)) | 1, 3)
    return alt - _nanmean_filter(alt, k)


def depth_below_ridge(alt, radius_m, res):
    k = max(int(round(2 * radius_m / res)) | 1, 3)
    mx = maximum_filter(np.nan_to_num(alt, nan=-1e9), k, mode="nearest")
    return np.where(np.isfinite(alt), mx - alt, np.nan)


def slope_aspect_curv(alt, res):
    a = np.nan_to_num(alt, nan=np.nanmean(alt))
    dzdy, dzdx = np.gradient(a, -res, res)             # nord+, est+
    slope = np.hypot(dzdx, dzdy)
    aspect = np.arctan2(-dzdx, -dzdy) % (2 * np.pi)    # direction de la pente descendante
    d2x = np.gradient(dzdx, res, axis=1); d2y = np.gradient(dzdy, -res, axis=0)
    curv = d2x + d2y                                    # <0 convexe (crete), >0 concave (creux)
    return slope, aspect, curv


def sky_view_factor(alt, res, radius_m=SVF_RADIUS, naz=SVF_NAZ):
    """SVF = 1 - mean(sin(horizon_angle)) sur naz azimuts. Recherche par decalage de grille
    (pas croissant), vectorise : O(naz * n_steps) passes sur le tableau."""
    a = np.nan_to_num(alt, nan=np.nanmean(alt))
    ny, nx = a.shape
    steps = np.unique(np.round(np.geomspace(1, radius_m / res, 24)).astype(int))
    horizon = np.zeros((naz,) + a.shape)
    for i, az in enumerate(np.linspace(0, 2 * np.pi, naz, endpoint=False)):
        dx, dy = np.sin(az), -np.cos(az)                # az 0 = nord = lignes decroissantes
        best = np.zeros_like(a)
        for s in steps:
            sx, sy = int(round(dx * s)), int(round(dy * s))
            shifted = np.full_like(a, -1e9)
            ys = slice(max(0, -sy), ny - max(0, sy)); yd = slice(max(0, sy), ny - max(0, -sy))
            xs = slice(max(0, -sx), nx - max(0, sx)); xd = slice(max(0, sx), nx - max(0, -sx))
            shifted[ys, xs] = a[yd, xd]
            dist = np.hypot(sx, sy) * res
            ang = np.arctan2(shifted - a, dist)
            best = np.maximum(best, ang)
        horizon[i] = best
    return 1.0 - np.sin(horizon).mean(axis=0)


def cold_air_accumulation(alt, res, smooth_m=DRAIN_SMOOTH):
    """Accumulation de drainage D8 sur le MNT lisse : chaque cellule envoie son 'air froid'
    vers le voisin le plus bas ; on compte les cellules amont. Retourne log10(1 + n)."""
    a = gaussian_filter(np.nan_to_num(alt, nan=np.nanmean(alt)), smooth_m / res)
    ny, nx = a.shape
    pad = np.pad(a, 1, mode="edge")
    # 8 voisins : (dy, dx, distance)
    nb = [(-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1),
          (-1, -1, np.sqrt(2)), (-1, 1, np.sqrt(2)), (1, -1, np.sqrt(2)), (1, 1, np.sqrt(2))]
    best_drop = np.zeros_like(a); best_k = np.full(a.shape, -1, dtype=int)
    for k, (dy, dx, d) in enumerate(nb):
        drop = (a - pad[1 + dy:1 + dy + ny, 1 + dx:1 + dx + nx]) / d
        upd = drop > best_drop
        best_drop[upd] = drop[upd]; best_k[upd] = k
    # traitement par altitude decroissante : chaque cellule verse son cumul en aval
    acc = np.ones_like(a)
    order = np.argsort(-a, axis=None)
    rr, cc = np.unravel_index(order, a.shape)
    kk = best_k[rr, cc]
    for r, c, k in zip(rr, cc, kk):
        if k < 0:
            continue
        dy, dx, _ = nb[k]
        r2, c2 = r + dy, c + dx
        if 0 <= r2 < ny and 0 <= c2 < nx:
            acc[r2, c2] += acc[r, c]
    return np.log10(acc)


def dist_to_ridge(alt, res, radius_m=2000):
    """Distance (km) aux cellules de crete locale (TPI 2 km > 60 m)."""
    t = tpi(alt, radius_m, res)
    ridge = np.nan_to_num(t) > 60.0
    if not ridge.any():
        return np.full_like(alt, np.nan)
    return distance_transform_edt(~ridge) * res / 1000.0


def compute(mnt=MNT, target_res=TARGET_RES, verbose=True):
    import puy_de_dome as P
    if not os.path.exists(mnt):
        P.fetch_mnt_wms(P.BBOX, res=P.RES, out=mnt, fmt=P.WMS_FMT)
    h, tr, crs, res = P.load_dem(mnt, target_res=target_res)
    alt = h.filled(np.nan).astype("float64")
    F = dict(alt=alt)
    if verbose: print("[terrain] pente / exposition / courbure")
    F["slope"], F["aspect"], F["curv"] = slope_aspect_curv(alt, res)
    for k, r in TPI_RADII.items():
        if verbose: print(f"[terrain] {k}")
        F[k] = tpi(alt, r, res)
    for k, r in DEPTH_RADII.items():
        F[k] = depth_below_ridge(alt, r, res)
    if verbose: print("[terrain] sky view factor (le plus long)")
    F["svf"] = sky_view_factor(alt, res)
    if verbose: print("[terrain] accumulation d'air froid (drainage D8)")
    F["cold_acc"] = cold_air_accumulation(alt, res)
    F["cold_acc_s"] = gaussian_filter(F["cold_acc"], COLD_ACC_SMOOTH / res)
    F["dist_ridge"] = dist_to_ridge(alt, res)
    meta = dict(res=res, transform=np.array([tr.a, tr.b, tr.c, tr.d, tr.e, tr.f]),
                shape=np.array(alt.shape))
    np.savez_compressed(OUT, **F, **{f"meta_{k}": v for k, v in meta.items()})
    if verbose: print(f"[terrain] -> {OUT}  ({alt.shape[0]}x{alt.shape[1]} @ {res:.0f} m)")
    return F, meta


def load(path=None):
    """Charge le cache ; calcule et ajoute les descripteurs manquants (nouvelle version)
    sans refaire le SVF, qui est long."""
    path = path or OUT
    if not os.path.exists(path):
        return compute()
    z = np.load(path)
    F = {k: np.asarray(z[k], dtype="float64") for k in z.files if not k.startswith("meta_")}   # (cache compact float16 accepte)
    meta = {k[5:]: z[k] for k in z.files if k.startswith("meta_")}
    res = float(meta["res"]); added = []
    for k, r in TPI_RADII.items():
        if k not in F:
            F[k] = tpi(F["alt"], r, res); added.append(k)
    if "cold_acc_s" not in F:
        F["cold_acc_s"] = gaussian_filter(F["cold_acc"], COLD_ACC_SMOOTH / res); added.append("cold_acc_s")
    if added:
        np.savez_compressed(path, **F, **{f"meta_{k}": v for k, v in meta.items()})
        print(f"[terrain] ajoutes au cache : {added}")
    return F, meta


def feature_matrix(F, rows, cols, names=None):
    """Extrait un tableau (n, n_features) aux positions (rows, cols) : pour les stations."""
    names = names or FEATURE_NAMES
    return np.column_stack([F[n][rows, cols] for n in names]), names


FEATURE_NAMES = ["alt", "slope", "curv", "tpi_500", "tpi_2k", "tpi_10k", "tpi_20k",
                 "depth_2k", "depth_10k", "svf", "cold_acc", "cold_acc_s", "dist_ridge"]


def quicklook(F, meta, out="terrain_check.png"):
    """4 panneaux. Convention : BLEU = froid potentiel (creux / ferme / drainage)."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource
    res = float(meta["res"]); alt = F["alt"]
    hs = LightSource(315, 45).hillshade(np.nan_to_num(alt, nan=np.nanmin(alt)), vert_exag=3, dx=res, dy=res)
    def lim(a, p=(2, 98), sym=False):
        lo, hi = np.nanpercentile(a, p)
        if sym:
            m = max(abs(lo), abs(hi)); return -m, m
        return lo, hi
    # RdBu_r : valeurs NEGATIVES (creux) en BLEU, positives (cretes) en rouge
    panels = [("tpi_2k", "TPI 2 km : bleu = fond de vallee", "RdBu_r", lim(F["tpi_2k"], sym=True)),
              ("tpi_10k", "TPI 10 km : bleu = bassin (Limagne, Aurillac...)", "RdBu_r", lim(F["tpi_10k"], sym=True)),
              ("svf", "Sky view factor : bleu = encaisse", "viridis", lim(F["svf"])),
              ("cold_acc_s", "Accumulation d'air froid lissee : bleu = convergence", "Blues",
               lim(F["cold_acc_s"], (50, 99.5)))]
    fig, axs = plt.subplots(2, 2, figsize=(16, 13))
    for ax, (k, ttl, cm, (lo, hi)) in zip(axs.ravel(), panels):
        ax.imshow(hs, cmap="gray")
        im = ax.imshow(F[k], cmap=cm, vmin=lo, vmax=hi, alpha=0.7)
        fig.colorbar(im, ax=ax, shrink=0.7); ax.set_title(ttl); ax.axis("off")
    fig.tight_layout(); fig.savefig(out, dpi=110); print("->", out)


if __name__ == "__main__":
    if "--recompute" in sys.argv or not os.path.exists(OUT):
        F, meta = compute()
    else:
        F, meta = load()                 # reutilise le cache (SVF deja calcule)
    quicklook(F, meta)
