"""
forecast_html.py - Prevision CRYOGENE (ex-CUVETTE) J+0 -> J+7 : trous a froid, inversions, Tmin
en haute resolution, en une page HTML autonome (s'ouvre dans le navigateur).

Chaine : dernier run IFS HRES 00/12 UTC (open data, +180 h)
         -> profil + diagnostics d'inversion (profile_inv)
         -> T_base a l'altitude de chaque pixel / station
         -> + residu predit par le modele ML (train.py, modele 'ifs')
         -> cartes a 06 UTC par jour (500 m), Tmin par nuit aux 80 stations,
            serie vallee / sommet, force de l'inversion.

Usage :
    python forecast_html.py                  # dernier run disponible
    python forecast_html.py --run 2026-11-12T00
    python forecast_html.py --res 200        # carte plus fine (plus lent) ; defaut 300 m
Sortie : C:\\cuvette_archive\\prevision\\cryogene_AAAAMMJJ_HHz.html (+ cryogene_derniere.html)
"""
import os, sys, io, base64, argparse, datetime as dt, webbrowser, warnings
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd())
import ecmwf_fetch as EF
import profile_inv as P
import terrain_features as TF

warnings.filterwarnings("ignore", category=FutureWarning)
ROOT = os.environ.get("CUVETTE_ROOT", r"C:\cuvette_archive")
MODEL = os.path.join(ROOT, "models", "cuvette_hgb_ifs.joblib")
OUT_DIR = os.path.join(ROOT, "prevision")
STATIONS_CSV = os.path.join(ROOT, "stations_cuvette.csv")
STEPS = EF.STEPS_HRES                         # 0-144 h / 3 h puis 150-180 h / 6 h
PAIR = ("CLERMONT-FD", "PLATEAU CACADOGNE")
JOURS = ["lun.", "mar.", "mer.", "jeu.", "ven.", "sam.", "dim."]


# --------------------------------------------------------------------------- #
#  Donnees d'entree
# --------------------------------------------------------------------------- #
STEPS_SHORT = EF.STEPS_SHORT                   # runs 06/18 UTC : 0-90 h / 3 h
TD = lambda h: np.timedelta64(int(h), "h")


def _utc(t):
    return t.replace(tzinfo=dt.timezone.utc) if t.tzinfo is None else t


def _latest(step):
    from ecmwf.opendata import Client
    return _utc(Client(source=EF.SOURCE).latest(type="fc", stream="oper", step=step, param="2t"))


def _load_run(run, steps):
    """Run IFS sur `steps` (archive locale si complete, sinon telechargement) -> Dataset surface + altitude."""
    rdir = os.path.join(EF.ARCHIVE, run.strftime("%Y%m%d_%Hz"))
    p_sfc, p_pl = os.path.join(rdir, "hres_sfc.grib2"), os.path.join(rdir, "hres_pl.grib2")
    ok = False
    if os.path.exists(p_sfc) and os.path.exists(p_pl):
        try:
            g = EF.open_grib(p_sfc)
            ok = int(g.step.max() / np.timedelta64(1, "h")) >= max(steps) and "lcc" in g
        except Exception:
            ok = False
    if not ok:
        try:
            p_sfc, p_pl, _ = EF.download(run, ens=False, steps=steps)
        except Exception as e:
            if run.hour not in (6, 18):
                raise
            print(f"[cuvette] run {run:%H}z via stream 'oper' refuse ({str(e)[:80]}) -> essai stream 'scda'")
            p_sfc, p_pl, _ = EF.download(run, ens=False, steps=steps, stream="scda")
    sfc = EF.open_grib(p_sfc).sortby("latitude").sortby("longitude")
    pl = EF.open_grib(p_pl).sortby("latitude").sortby("longitude")
    dres = lambda d: float(np.round(abs(d.latitude.values[1] - d.latitude.values[0]), 3))
    same = (sfc.latitude.size == pl.latitude.size and sfc.longitude.size == pl.longitude.size
            and np.allclose(sfc.latitude, pl.latitude) and np.allclose(sfc.longitude, pl.longitude))
    if not same:   # ex. surface a 0.1 deg et altitude a 0.25 deg : on ramene l'altitude sur la grille surface
        print(f"[cuvette] grilles IFS differentes (surface {dres(sfc)} deg, altitude {dres(pl)} deg) "
              "-> altitude interpolee sur la grille surface")
        pl = pl.interp(latitude=sfc.latitude, longitude=sfc.longitude, kwargs={"fill_value": None})
    ds = xr.merge([sfc, pl], compat="override", join="left")
    print(f"[cuvette] run {run:%d/%m %H}z, grille IFS {dres(ds)} deg : lat {float(ds.latitude.min()):.2f}-"
          f"{float(ds.latitude.max()):.2f}, lon {float(ds.longitude.min()):.2f}-{float(ds.longitude.max()):.2f}")
    return ds


def splice(ds_s, ds_b, shift):
    """Raccord : run court (0-90 h) puis run complet plus ancien de `shift` h, recale sur le run court.
    Les cumuls de precipitation restent continus au raccord."""
    cut = int(ds_s.step.max() / np.timedelta64(1, "h"))
    bsteps = [s for s in STEPS if s - shift > cut]
    b = ds_b.sel(step=[TD(s) for s in bsteps])
    if "tp" in b and "tp" in ds_s:
        b["tp"] = ds_s["tp"].sel(step=TD(cut)) + (b["tp"] - ds_b["tp"].sel(step=TD(cut + shift)))
    b = b.assign_coords(step=[TD(s - shift) for s in bsteps])
    if not (np.array_equal(b.latitude, ds_s.latitude) and np.array_equal(b.longitude, ds_s.longitude)):
        b = b.interp(latitude=ds_s.latitude, longitude=ds_s.longitude, kwargs={"fill_value": None})
    b = b.drop_vars([c for c in ("time",) if c in b.coords])
    out = xr.concat([ds_s, b], dim="step", data_vars="all", coords="minimal", compat="override", join="override")
    return out, cut


def get_run(run=None):
    """Dernier run IFS disponible. Runs 00/12 UTC : complets jusqu'a +180 h. Runs 06/18 UTC (ou 00/12
    pas encore complets) : 0-90 h, completes au-dela par le dernier run complet (raccord indique dans la page)."""
    full = None
    if run is None:
        full = _latest(max(STEPS))
        try:
            newest = _latest(max(STEPS_SHORT))
        except Exception:
            newest = full
        run = max(full, newest)
    run = _utc(run)
    if run.hour in (0, 12) and (full is None or run == full):
        ds = _load_run(run, STEPS)
        ds.attrs["splice"] = ""
        return ds, run
    base = full if (full is not None and full < run) else None
    if base is None:                                   # dernier 00/12 UTC complet avant ce run
        base = run.replace(hour=0 if run.hour < 12 else 12)
        if base == run:
            base = run - dt.timedelta(hours=12)
    shift = int((run - base).total_seconds() // 3600)
    print(f"[cuvette] run {run:%H}z publie jusqu'a +{max(STEPS_SHORT)} h -> complete par le run {base:%d/%m %H}z")
    ds, cut = splice(_load_run(run, STEPS_SHORT), _load_run(base, STEPS), shift)
    ds.attrs["splice"] = f"{cut}|{base:%d/%m %H} UTC"
    return ds, run


INV_MODEL = os.path.join(ROOT, "models", "cuvette_inversion.joblib")
DEPTS_MAIN = ["63", "15", "19"]
DEPTS_NEAR = ["03", "23", "87", "24", "46", "12", "48", "43", "42"]
DEPTS_CACHE = os.path.join(ROOT, "departements_l93.json")
REGIMES = ["", "inversion radiative (ciel clair)", "inversion sous stratus (grisaille)", "inversion, ciel nuageux"]


def load_inversion_model():
    import joblib
    if not os.path.exists(INV_MODEL):
        print("[cuvette] pas de modele d'inversion (lance python train_inversion.py) -> indicateur absent")
        return None
    return joblib.load(INV_MODEL)


def departements():
    """Contours IGN (Admin Express, WFS Geoplateforme) en Lambert 93, simplifies a 100 m,
    mis en cache. Retourne {'main': [[x,y],...] , 'near': ...} ou None si indisponible."""
    import json
    if os.path.exists(DEPTS_CACHE):
        with open(DEPTS_CACHE, encoding="utf-8") as fh:
            return json.load(fh)
    try:
        import requests
        from shapely.geometry import shape
    except ImportError:
        print("[cuvette] shapely/requests absents -> pas de contours"); return None
    def fetch(codes):
        insee = ",".join(f"'{c}'" for c in codes)
        for tn in ("ADMINEXPRESS-COG-CARTO.LATEST:departement", "ADMINEXPRESS-COG.LATEST:departement",
                   "ADMINEXPRESS-COG-CARTO.2024:departement"):
            try:
                r = requests.get("https://data.geopf.fr/wfs/ows", timeout=40, params={
                    "service": "WFS", "version": "2.0.0", "request": "GetFeature", "typenames": tn,
                    "srsname": "EPSG:2154", "outputformat": "application/json",
                    "cql_filter": f"insee_dep IN ({insee})"})
                r.raise_for_status()
                feats = r.json().get("features", [])
                if feats:
                    out = []
                    for f in feats:
                        g = shape(f["geometry"]).simplify(100)
                        for p in (g.geoms if g.geom_type == "MultiPolygon" else [g]):
                            out.append([[round(x), round(y)] for x, y in p.exterior.coords])
                    return out
            except Exception:
                continue
        return []
    d = {"main": fetch(DEPTS_MAIN), "near": fetch(DEPTS_NEAR)}
    if not d["main"]:                        # repli : contours simplifies (france-geojson, GitHub)
        try:
            from pyproj import Transformer
            from shapely.ops import transform as shp_tr
            gj = requests.get("https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/"
                              "departements-version-simplifiee.geojson", timeout=60).json()
            tr = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True).transform
            def conv(codes):
                out = []
                for f in gj["features"]:
                    if f["properties"].get("code") in codes:
                        g = shp_tr(tr, shape(f["geometry"])).simplify(100)
                        for p in (g.geoms if g.geom_type == "MultiPolygon" else [g]):
                            out.append([[round(x), round(y)] for x, y in p.exterior.coords])
                return out
            d = {"main": conv(DEPTS_MAIN), "near": conv(DEPTS_NEAR)}
        except Exception as e:
            print(f"[cuvette] repli GitHub impossible ({str(e)[:60]})")
    if not d["main"]:
        print("[cuvette] contours des departements indisponibles (WFS IGN) -> sans contours")
        return None
    with open(DEPTS_CACHE, "w", encoding="utf-8") as fh:
        json.dump(d, fh)
    print(f"[cuvette] contours departements -> {DEPTS_CACHE}")
    return d


def dept_mask(depts, transform, shape):
    """Masque 100 m : True dans 63/15/19 (pour attenuer l'exterieur de la carte)."""
    if not depts:
        return None
    try:
        from rasterio.features import rasterize
        from rasterio.transform import Affine
        from shapely.geometry import Polygon
        polys = [Polygon(p) for p in depts["main"] if len(p) > 3]
        m = rasterize([(p, 1) for p in polys], out_shape=shape, transform=Affine(*transform[:6]), fill=0, dtype="uint8")
        return m.astype(bool)
    except Exception as e:
        print(f"[cuvette] masque departements impossible ({str(e)[:60]})"); return None


def svg_paths(polys, a, c, e, f):
    """Polylignes L93 -> attribut 'd' SVG dans le repere de la carte (pixels 100 m)."""
    out = []
    for p in polys:
        pts = [f"{(x - c) / a:.1f},{(y - f) / e:.1f}" for x, y in p]
        out.append("M" + "L".join(pts) + "Z")
    return "".join(out)


def inversion_proba(inv_bundle, dfst, st, steps):
    """P(inversion observee) par couple et par echeance + regime selon la nebulosite."""
    if inv_bundle is None:
        return []
    names = st.NOM_USUEL.to_numpy()
    res = []
    for pr in inv_bundle["pairs"]:
        ix = np.where(names == pr["low"])[0]
        if not len(ix):
            continue
        d = dfst[dfst.pt == ix[0]].set_index("step_h").reindex(steps)
        X = np.column_stack([d[f].to_numpy(float) if f in d else np.full(len(d), np.nan) for f in pr["features"]])
        p = pr["model"].predict_proba(X)[:, 1]
        lcc = d["lcc"].to_numpy(float) if "lcc" in d else np.full(len(d), np.nan)
        tcc = d["tcc"].to_numpy(float) if "tcc" in d else np.full(len(d), np.nan)
        reg = np.where(p < 0.5, 0, np.where(np.nan_to_num(lcc) >= 0.6, 2,
                       np.where(np.nan_to_num(tcc, nan=1) <= 0.4, 1, 3)))
        r2 = lambda v: None if not np.isfinite(v) else round(float(v), 2)
        res.append(dict(name=pr["name"], low=pr["low"].title(), high=pr["high"].title(),
                        P=[r2(v) for v in p], L=[r2(v) for v in lcc], C=[r2(v) for v in tcc],
                        R=[int(v) for v in reg], clim=round(pr["clim"], 2)))
    return res


def load_model():
    import joblib
    if not os.path.exists(MODEL):
        raise SystemExit(f"Modele absent : {MODEL}\n -> lance d'abord python train.py")
    return joblib.load(MODEL)


def stations():
    """Stations + descripteurs de relief (mis en cache dans stations_cuvette.csv)."""
    if os.path.exists(STATIONS_CSV):
        return pd.read_csv(STATIONS_CSV, dtype={"NUM_POSTE": str})
    import glob
    import obs_mf as OBS
    F, meta = TF.load()
    # 1er lancement : on ne lit que 6 colonnes des fichiers 'latest' (2 dernieres annees),
    # pas toute l'archive -> quelques secondes au lieu de 10+ minutes.
    files = sorted(glob.glob(os.path.join(OBS.OBS_DIR, "*latest*.csv.gz"))) or \
        sorted(glob.glob(os.path.join(OBS.OBS_DIR, "*.csv.gz")))
    cols = ["NUM_POSTE", "NOM_USUEL", "LAT", "LON", "ALTI", "T"]
    print(f"[cuvette] 1er lancement : table des stations depuis {len(files)} fichier(s) ...", flush=True)
    d = pd.concat([pd.read_csv(f, sep=";", usecols=cols, dtype={"NUM_POSTE": str}) for f in files])
    st = OBS.stations(d, terrain_meta=meta)
    X, names = TF.feature_matrix(F, st.row.values, st.col.values)
    for i, n in enumerate(names):
        st[f"ter_{n}"] = X[:, i]
    st.to_csv(STATIONS_CSV, index=False)
    return st


def grid_points(res_m):
    """Pixels de la carte : sous-echantillonnage du cache terrain (100 m) au pas res_m."""
    from pyproj import Transformer
    F, meta = TF.load()
    k = max(1, int(round(res_m / float(meta["res"]))))
    a, b, c, d, e, f = meta["transform"]
    sub = {n: F[n][::k, ::k] for n in TF.FEATURE_NAMES if n in F}
    ny, nx = sub["alt"].shape
    cols = np.arange(nx) * k + 0.5          # centre du pixel 100 m echantillonne
    rows = np.arange(ny) * k + 0.5
    XX, YY = np.meshgrid(c + a * cols, f + e * rows)
    lon, lat = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True).transform(XX, YY)
    valid = np.isfinite(sub["alt"])
    pts = pd.DataFrame({"LAT": lat[valid], "LON": lon[valid], "ALTI": sub["alt"][valid]})
    for n, v in sub.items():
        pts[f"ter_{n}"] = v[valid]
    return pts, valid, sub["alt"], (XX, YY)


# --------------------------------------------------------------------------- #
#  Prediction
# --------------------------------------------------------------------------- #
def predict_points(ds, run, pts, bundle, steps=None):
    """T predite (C) aux points pts (DataFrame LAT, LON, ALTI, ter_*) pour chaque echeance.
    Retourne DataFrame long : step_h, valid, pt, T_base, T_cuv + diagnostics."""
    if steps is not None:
        ds = ds.sel(step=[np.timedelta64(int(s), "h") for s in steps])
    la = xr.DataArray(pts.LAT.values, dims="pt")
    lo = xr.DataArray(pts.LON.values, dims="pt")
    hh = xr.DataArray(pts.ALTI.values.astype(float), dims="pt")
    dsp = ds.interp(latitude=la, longitude=lo, kwargs={"fill_value": None})
    std = P.standardize(dsp)
    d = P.diagnose(std)
    d["T_base"] = P.t_at_height(std, hh) - 273.15
    d["dz_model"] = hh - d["zs"]
    if "msl" in dsp:
        d["msl"] = dsp["msl"] / 100.0
    if "tp" in dsp:
        d["tp_cum"] = dsp["tp"] * 1000.0                   # cumul depuis le debut du run (mm)
    above = (std["Z"]["level"] < std["ps"]) & (std["Z"] > std["zs"] + 10)
    d["tmax_aloft"] = std["T"].where(above).max("level") - 273.15   # "nez chaud" en altitude
    d = d.drop_vars([c for c in d.coords if c not in ("step", "pt")])
    df = d.to_dataframe().reset_index()
    df["step_h"] = (df.step / np.timedelta64(1, "h")).astype(int)
    df["valid"] = pd.Timestamp(run) + pd.to_timedelta(df.step_h, unit="h")
    h = df.valid.dt.hour
    doy = df.valid.dt.dayofyear
    df["hour_sin"], df["hour_cos"] = np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)
    df["doy_sin"], df["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    ter_cols = [c for c in pts.columns if c.startswith("ter_")]
    ter = pts[ter_cols].to_numpy()[df.pt.to_numpy()]
    df[ter_cols] = ter
    feats = bundle["features"]
    X = np.column_stack([df[f].to_numpy(dtype=float) if f in df else np.full(len(df), np.nan) for f in feats])
    df["T_cuv"] = df.T_base + bundle["model"].predict(X)
    return df


def nights(run, n=8):
    """Nuits couvertes : (date du matin, debut 18 UTC veille, fin 09 UTC)."""
    r = pd.Timestamp(run)
    out = []
    for i in range(n):
        m = (r + pd.Timedelta(days=i)).normalize() + pd.Timedelta(days=1 if r.hour >= 9 else 0)
        out.append((m, m - pd.Timedelta(hours=6), m + pd.Timedelta(hours=9)))
    last = r + pd.Timedelta(hours=max(STEPS))
    return [x for x in out if x[0] + pd.Timedelta(hours=6) <= last]


# --------------------------------------------------------------------------- #
#  Rendu de la carte (image nette : relief ombre 100 m + couleur T)
# --------------------------------------------------------------------------- #
TMIN_C, TMAX_C = -20.0, 30.0
# Echelle "hiver" : froid detaille (violet -> bleu -> blanc bleute a 0 C), doux en vert-jaune,
# l'orange n'arrive que vers 15-20 C et le rouge vif au-dela de 25 C.
T_STOPS = [(-20, "#3b0f70"), (-15, "#3a3fa0"), (-10, "#2f6fd0"), (-5, "#6aaee6"), (-1, "#bfe3f5"),
           (0, "#e8f4f8"), (1, "#eef6e2"), (5, "#cfe8a8"), (10, "#e9eb8f"), (15, "#f6cf6e"),
           (20, "#f0a052"), (25, "#e0643c"), (30, "#b8282e")]
T_TICKS = [-20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30]
_CMAP = None


def _norm():
    from matplotlib.colors import Normalize
    return Normalize(vmin=TMIN_C, vmax=TMAX_C)


def _cmap():
    global _CMAP
    if _CMAP is None:
        from matplotlib.colors import LinearSegmentedColormap
        _CMAP = LinearSegmentedColormap.from_list(
            "cuvette_hiver", [((t - TMIN_C) / (TMAX_C - TMIN_C), c) for t, c in T_STOPS], N=512)
    return _CMAP


def hex_color(t):
    if not np.isfinite(t):
        return "#999999"
    r, g, b, _ = _cmap()(_norm()(np.clip(t, TMIN_C, TMAX_C)))
    return "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))


def legend_css():
    stops = [f"{c} {100 * (t - TMIN_C) / (TMAX_C - TMIN_C):.1f}%" for t, c in T_STOPS]
    return "linear-gradient(90deg," + ",".join(stops) + ")"


def legend_html():
    ticks = "".join(f'<span style="left:{100 * (t - TMIN_C) / (TMAX_C - TMIN_C):.1f}%">{t:+d}</span>'
                    .replace("+0", "0") for t in T_TICKS)
    return (f'<div class="tleg"><div class="bar" style="background:{legend_css()}"></div>'
            f'<div class="ticks">{ticks}</div></div>')


def hillshade(alt100, res):
    from matplotlib.colors import LightSource
    a = np.nan_to_num(alt100, nan=np.nanmin(alt100))
    return LightSource(315, 45).hillshade(a, vert_exag=2.5, dx=res, dy=res)


def upsample(Tk, k, shape):
    """Grille k (pixels 100 m d'indices i*k) -> grille 100 m complete (bilineaire)."""
    from scipy.ndimage import zoom, distance_transform_edt
    T = Tk.copy()
    bad = ~np.isfinite(T)
    if bad.any():
        idx = distance_transform_edt(bad, return_distances=False, return_indices=True)
        T = T[tuple(idx)]
    U = zoom(T, k, order=1)
    out = np.full(shape, np.nan)
    ny, nx = min(shape[0], U.shape[0]), min(shape[1], U.shape[1])
    out[:ny, :nx] = U[:ny, :nx]
    if ny < shape[0]:
        out[ny:, :] = out[ny - 1:ny, :]
    if nx < shape[1]:
        out[:, nx:] = out[:, nx - 1:nx]
    return out


def render_jpeg(T100, hs, alt100, inside=None):
    from PIL import Image
    rgb = _cmap()(_norm()(np.clip(T100, TMIN_C, TMAX_C)))[..., :3]
    shade = (0.50 + 0.55 * hs)[..., None]
    img = np.clip(rgb * shade, 0, 1)
    if inside is not None:
        img[~inside] = 0.55 * img[~inside] + 0.45 * 0.93     # hors 63/15/19 : attenue
    img[~np.isfinite(alt100)] = 0.85
    b = io.BytesIO()
    Image.fromarray((img * 255).astype(np.uint8)).save(b, format="JPEG", quality=82, optimize=True)
    return base64.b64encode(b.getvalue()).decode()


def pack_u8(a, offset, step):
    """Grille -> octets (0 = absent) en base64 : valeur = offset + (code-1)*step."""
    q = np.where(np.isfinite(a), np.clip(np.round((a - offset) / step) + 1, 1, 255), 0).astype(np.uint8)
    return base64.b64encode(q.tobytes()).decode()


# --------------------------------------------------------------------------- #
#  Precipitations : type et couche carte
# --------------------------------------------------------------------------- #
PTYPES = ["", "pluie", "pluie et neige", "neige", "pluie verglaçante ?"]
PCOLORS = ["#00000000", "#2f80ed", "#7b61ff", "#e8eef5", "#e0218a"]
RAIN_MIN = 0.2          # mm / intervalle : en dessous, sec


def ptype(rain, t_sol, t_aloft):
    """0 sec, 1 pluie, 2 pluie et neige, 3 neige, 4 risque de pluie verglacante
    (precipitation + couche chaude > 1 C en altitude + sol <= 0.5 C : typique d'inversion)."""
    rain, t_sol, t_aloft = (np.asarray(x, float) for x in (rain, t_sol, t_aloft))
    out = np.where(rain >= RAIN_MIN, 1, 0)
    warm = t_aloft > 1.0
    out = np.where((out > 0) & ~warm & (t_sol <= 1.0), 3, out)
    out = np.where((out > 0) & ~warm & (t_sol > 1.0) & (t_sol <= 2.5), 2, out)
    out = np.where((out > 0) & warm & (t_sol <= 0.5), 4, out)
    return out.astype(int)


RAIN_LEVELS = (1, 2, 5, 10, 20, 50)


def rain_svg(Rk, Pk, k):
    """Isohyetes (cumul 6 h) + contours des zones de neige / risque de pluie verglacante,
    en SVG dans le repere de la carte (pixels 100 m). Chaine vide si sec."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter
    R = np.nan_to_num(Rk)
    if R.max() < 1.0 and not ((Pk >= 3) & (R >= RAIN_MIN)).any():
        return ""
    Rs = gaussian_filter(R, 1.2)
    seg2d = lambda seg: "M" + "L".join(f"{x * k + 0.5:.0f},{y * k + 0.5:.0f}" for x, y in seg)
    fig, ax = plt.subplots()
    out = []
    for code, col, fill_op, dash in ((3, "#ffffff", .22, "16 9"), (4, "#e0218a", .18, "")):
        m = ((Pk == code) & (R >= RAIN_MIN)).astype(float)
        if m.sum() < 4:
            continue
        cs = ax.contour(gaussian_filter(m, 1.0), levels=[0.5])
        dd = "".join(seg2d(sg) + "Z" for sg in cs.allsegs[0] if len(sg) > 3)
        if dd:
            out.append(f'<path d="{dd}" fill="{col}" fill-opacity="{fill_op}" fill-rule="evenodd" stroke="{col}" '
                       f'stroke-width="3" stroke-dasharray="{dash}" stroke-opacity=".95"/>')
    levels = [l for l in RAIN_LEVELS if Rs.max() >= l]
    if levels:
        cs = ax.contour(Rs, levels=levels)
        for i, (lev, segs) in enumerate(zip(cs.levels, cs.allsegs)):
            segs = [sg for sg in segs if len(sg) > 3]
            if not segs:
                continue
            out.append(f'<path d="{"".join(seg2d(sg) for sg in segs)}" fill="none" stroke="#1f5fbf" '
                       f'stroke-width="{1.4 + 0.8 * i:.1f}" stroke-opacity=".9" stroke-linejoin="round"/>')
            lg = max(segs, key=len)
            if len(lg) > 10:
                px, py = lg[len(lg) // 2]
                out.append(f'<text class="rl" x="{px * k:.0f}" y="{py * k:.0f}">{lev:g} mm</text>')
    plt.close(fig)
    return "".join(out)


def render_rain_png(rain, ptk):
    """Couche transparente (grille de calcul) : couleur = type, opacite = cumul."""
    from PIL import Image
    h, w = rain.shape
    rgba = np.zeros((h, w, 4), np.uint8)
    wet = np.isfinite(rain) & (rain >= RAIN_MIN)
    if not wet.any():
        return None
    for code in (1, 2, 3, 4):
        c = PCOLORS[code].lstrip("#")
        m = wet & (ptk == code)
        rgba[m, 0], rgba[m, 1], rgba[m, 2] = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    alpha = np.clip(0.22 + 0.08 * np.log2(np.maximum(np.nan_to_num(rain), RAIN_MIN) / RAIN_MIN), 0.22, 0.60)
    rgba[..., 3] = np.where(wet, (alpha * 255).astype(np.uint8), 0)
    b = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(b, format="PNG", optimize=True)
    return base64.b64encode(b.getvalue()).decode()


def rain_at(ds, pts, s, ds_steps):
    """Cumul IFS (mm) sur les 6 h precedant l'echeance s, interpole aux points."""
    if "tp" not in ds or s - 6 not in ds_steps:
        return np.full(len(pts), np.nan)
    tp = ds["tp"].sel(step=np.timedelta64(int(s), "h")) - ds["tp"].sel(step=np.timedelta64(int(s - 6), "h"))
    la = xr.DataArray(pts.LAT.values, dims="pt")
    lo = xr.DataArray(pts.LON.values, dims="pt")
    return np.maximum(tp.interp(latitude=la, longitude=lo, kwargs={"fill_value": None}).values * 1000.0, 0)


# --------------------------------------------------------------------------- #
#  Page HTML interactive
# --------------------------------------------------------------------------- #
CSS = """
:root{--bg:#f4f5f7;--card:#fff;--ink:#1d2430;--mute:#667085;--line:#e3e6eb;--acc:#2b5d8a;--sel:#ff3b30}
@media (prefers-color-scheme:dark){:root{--bg:#11151b;--card:#1a2028;--ink:#e6e9ee;--mute:#9aa4b2;--line:#2b333d;--acc:#7fb2e5}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,Segoe UI,Roboto,sans-serif}
header{padding:12px 18px;border-bottom:1px solid var(--line);background:var(--card);display:flex;flex-wrap:wrap;gap:6px 18px;align-items:baseline}
h1{margin:0;font-size:19px}h2{font-size:15px;margin:14px 0 6px}.mute{color:var(--mute)}
main{max-width:1800px;margin:0 auto;padding:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px;margin-bottom:12px}
.tabs{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:6px}
.tabs button,.hours button{border:1px solid var(--line);background:var(--bg);color:var(--ink);padding:6px 11px;border-radius:8px;cursor:pointer;font:inherit}
.tabs button.on,.hours button.on{background:var(--acc);color:#fff;border-color:var(--acc)}
.hours{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin-bottom:8px}
.hours button{padding:4px 10px;font-size:13px}
.layout{display:grid;grid-template-columns:minmax(0,1fr) 410px;gap:12px;align-items:start}
@media (max-width:1100px){.layout{grid-template-columns:1fr}}
#mapbox{position:relative}
#map{width:100%;height:auto;display:block;border-radius:8px;cursor:crosshair;max-height:88vh}
#map .st{stroke:#111;stroke-width:1.6;cursor:pointer;vector-effect:non-scaling-stroke}
#map path{vector-effect:non-scaling-stroke}
#map.drag{cursor:grabbing}
.zoom{position:absolute;top:8px;right:8px;display:flex;flex-direction:column;gap:4px;z-index:2}
.zoom button{width:34px;height:34px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--ink);font:600 18px system-ui;cursor:pointer;box-shadow:0 1px 3px #0002}
.tleg{position:relative;flex:1;min-width:260px;padding-bottom:16px;margin:0 12px}.tleg .bar{height:12px;border-radius:6px}
.tleg .ticks span{position:absolute;top:14px;transform:translateX(-50%);font-size:11px;color:var(--mute)}
#map .st:hover{stroke-width:3}
#map .st.sel{stroke:var(--sel);stroke-width:3.5}
#map text{font-weight:600;font-family:system-ui,sans-serif;font-size:calc(23px * var(--zf,1));paint-order:stroke;stroke:#fff;stroke-width:calc(5px * var(--zf,1));fill:#111;pointer-events:none}
#tip{position:absolute;pointer-events:none;background:rgba(20,24,30,.88);color:#fff;padding:4px 8px;border-radius:6px;font-size:12.5px;display:none;white-space:nowrap}
.legend{display:flex;align-items:center;gap:8px;margin-top:6px;font-size:12px;flex-wrap:wrap}
.legend .bar{flex:1;min-width:160px;height:12px;border-radius:6px}
.sw{display:inline-block;width:12px;height:12px;border-radius:3px;vertical-align:-2px;border:1px solid #0002}
.k{display:grid;grid-template-columns:1fr 1fr;gap:8px}.k div{background:var(--bg);border-radius:8px;padding:7px 9px}
.k b{font-size:18px;display:block}
select{width:100%;padding:7px;border-radius:8px;border:1px solid var(--line);background:var(--bg);color:var(--ink);font:inherit}
.chips{display:flex;flex-wrap:wrap;gap:4px}.chips span{padding:3px 7px;border-radius:7px;font-size:12px;color:#111;min-width:58px;text-align:center;cursor:pointer}
.chips span.on{outline:2px solid var(--ink)}
#chartbox{position:relative;margin-top:14px;border-top:1px solid var(--line);padding-top:4px}
#chart{width:100%;height:auto;cursor:crosshair}#chart text{fill:var(--mute);font-size:13px}
#chart .tn{fill:#1f5fbf;font-weight:700}#chart .tx{fill:#c0392b;font-weight:700}#chart .dl{fill:var(--ink);font-weight:600;font-size:14px}
#ctip{position:absolute;pointer-events:none;background:rgba(20,24,30,.9);color:#fff;padding:5px 9px;border-radius:6px;font-size:12.5px;display:none;white-space:nowrap}
#map .rl{font-weight:700;font-size:calc(26px * var(--zf,1));fill:#1f5fbf;paint-order:stroke;stroke:#fff;stroke-width:calc(6px * var(--zf,1))}
table{border-collapse:collapse;width:100%;font-size:12.5px}th,td{padding:4px 6px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}th{position:sticky;top:0;background:var(--card)}
#tbl tbody tr,#hourly tbody tr{cursor:pointer}#tbl tbody tr:hover,#hourly tbody tr:hover{outline:1px solid var(--acc)}
#hourly tr.on td{font-weight:700}
.invchip{display:inline-block;padding:1px 7px;border-radius:7px;margin:2px 3px 0 0;font-size:12.5px;color:#fff}
#invs text{font-size:11px;fill:var(--ink)}#invs .lab{font-weight:600}
.tabs button .dot{display:block;height:3px;border-radius:2px;margin-top:3px}
.scroll{max-height:520px;overflow:auto}
"""

JS = r"""
const D = JSON.parse(document.getElementById('data').textContent);
let fr = D.f0, sel = D.def;
const svg = document.getElementById('map'), img = document.getElementById('img'), rimg = document.getElementById('rain'), tip = document.getElementById('tip');
const ctip = document.getElementById('ctip');
const dec = s => Uint8Array.from(atob(s), c => c.charCodeAt(0));
const grids = D.frames.map(f => dec(f.grid)), rgrids = D.frames.map(f => f.rgrid ? dec(f.rgrid) : null);
const altg = dec(D.g.alt);
const fmt = v => (v === null || v === undefined || isNaN(v)) ? '–' : (v > 0 ? '+' : '') + v.toFixed(1);
const mm = v => (v === null || v === undefined || v < 0.1) ? '–' : v.toFixed(1);
const col = t => (t === null || t === undefined) ? '#999' : D.colors[Math.max(0, Math.min(D.colors.length - 1, Math.round((t - D.tmin) / D.cstep)))];
const si = () => D.steps.indexOf(D.frames[fr].s);
const day = () => D.frames[fr].d;
const rain6 = (s, k) => { let t = 0, ok = false; const s0 = D.steps[k];
  for (let j = k; j >= 0 && D.steps[j] > s0 - 6; j--) { if (s.P[j] !== null) { t += s.P[j]; ok = true; } } return ok ? t : null; };

function pcol(p){ const C = [[242,240,247],[203,201,226],[158,154,200],[117,107,177],[84,39,143]];
  const x = Math.max(0, Math.min(.9999, p || 0)) * 4, i = Math.floor(x), t = x - i, a = C[i], b = C[i + 1];
  return `rgb(${a.map((v, j) => Math.round(v + (b[j] - v) * t)).join(',')})`; }
function invStrip(){
  if (!D.inv.length) return;
  const n = D.steps.length, LW = 215, W = 1100, rh = 26, gap = 8, top = 18, H = top + D.inv.length * (rh + gap) + 4;
  const x = k => LW + (W - LW - 6) * (D.steps[k] - D.steps[0]) / (D.steps[n - 1] - D.steps[0]);
  let o = '';
  D.days.forEach((d, j) => { const ks = D.steps.map((s, k) => k).filter(k => D.sday[k] === j); if (!ks.length) return;
    o += `<text x="${Math.min(W - 32, (x(ks[0]) + x(ks[ks.length - 1])) / 2)}" y="12" text-anchor="middle" ${j === day() ? 'font-weight="700"' : ''}>${d.lab}</text>`;
    o += `<line x1="${x(ks[0]) - 1}" x2="${x(ks[0]) - 1}" y1="14" y2="${H}" stroke="currentColor" opacity=".25"/>`; });
  D.inv.forEach((v, r) => { const y0 = top + r * (rh + gap);
    o += `<text class="lab" x="0" y="${y0 + 10}">${v.name}</text><text x="0" y="${y0 + 22}" opacity=".7" style="font-size:10px">${(v.low + ' / ' + v.high).slice(0, 36)}</text>`;
    for (let k = 0; k < n; k++) { const x0 = x(k), x1 = k < n - 1 ? x(k + 1) : x0 + (x0 - x(k - 1)), p = v.P[k], l = v.L[k];
      const fi = D.frames.findIndex(f => f.s >= D.steps[k]);
      o += `<g data-f="${fi}" style="cursor:pointer"><rect x="${x0}" y="${y0}" width="${Math.max(1, x1 - x0 - .5)}" height="${rh - 6}" fill="${p === null ? '#ddd' : pcol(p)}">` +
           `<title>${D.shour[k]} : P inversion ${p === null ? '–' : Math.round(100 * p) + ' %'} · nébulosité basse ${l === null ? '–' : Math.round(100 * l) + ' %'}${v.R[k] ? ' → ' + D.regimes[v.R[k]] : ''}</title></rect>` +
           (l ? `<rect x="${x0}" y="${y0 + rh - 5}" width="${Math.max(1, x1 - x0 - .5)}" height="4" fill="#8a8f98" opacity="${Math.min(1, l)}"/>` : '') + '</g>'; } });
  const xs = x(si());
  o += `<line x1="${xs}" x2="${xs}" y1="14" y2="${H}" stroke="var(--sel)" stroke-width="2"/>`;
  const svgI = document.getElementById('invs'); svgI.setAttribute('viewBox', `0 0 ${W} ${H}`); svgI.innerHTML = o;
  svgI.querySelectorAll('g[data-f]').forEach(g => g.onclick = () => { const f = +g.dataset.f; if (f >= 0) setFrame(f); });
}
// ---------- zoom / deplacement de la carte (viewBox) ----------
const VB0 = svg.getAttribute('viewBox').split(' ').map(Number);
let vb = VB0.slice(), dragged = false, drag = null;
const zf = () => vb[2] / VB0[2];
function setVB(){
  vb[2] = Math.min(VB0[2], vb[2]); vb[3] = vb[2] * VB0[3] / VB0[2];
  vb[0] = Math.max(0, Math.min(VB0[2] - vb[2], vb[0])); vb[1] = Math.max(0, Math.min(VB0[3] - vb[3], vb[1]));
  svg.setAttribute('viewBox', vb.map(v => v.toFixed(1)).join(' ')); svg.style.setProperty('--zf', zf());
  const m = document.getElementById('mark'); m.setAttribute('r', 14 * zf()); drawStations();
}
function zoomAt(cx, cy, f){
  const nw = Math.max(VB0[2] / 16, Math.min(VB0[2], vb[2] * f)), s = nw / vb[2];
  vb = [cx - (cx - vb[0]) * s, cy - (cy - vb[1]) * s, nw, vb[3] * s]; setVB();
}
svg.addEventListener('wheel', ev => { ev.preventDefault(); const q = toGrid(ev); zoomAt(q.x, q.y, ev.deltaY < 0 ? 0.8 : 1.25); }, {passive: false});
svg.addEventListener('mousedown', ev => { if (ev.button !== 0) return; drag = {x: ev.clientX, y: ev.clientY, vb: vb.slice()}; dragged = false; });
window.addEventListener('mousemove', ev => { if (!drag) return;
  const dx = ev.clientX - drag.x, dy = ev.clientY - drag.y;
  if (!dragged && Math.hypot(dx, dy) < 4) return;
  dragged = true; svg.classList.add('drag'); tip.style.display = 'none';
  const k = vb[2] / svg.getBoundingClientRect().width;
  vb[0] = drag.vb[0] - dx * k; vb[1] = drag.vb[1] - dy * k; setVB(); });
window.addEventListener('mouseup', () => { drag = null; svg.classList.remove('drag'); });
svg.addEventListener('dblclick', ev => { ev.preventDefault(); const q = toGrid(ev); zoomAt(q.x, q.y, 0.5); });
document.getElementById('zin').onclick = () => zoomAt(vb[0] + vb[2] / 2, vb[1] + vb[3] / 2, 0.6);
document.getElementById('zout').onclick = () => zoomAt(vb[0] + vb[2] / 2, vb[1] + vb[3] / 2, 1 / 0.6);
document.getElementById('zreset').onclick = () => { vb = VB0.slice(); setVB(); };
// pincement / glisser au doigt (tablette, telephone)
let touch = null;
svg.addEventListener('touchstart', ev => { if (ev.touches.length === 2) { const [a, b] = ev.touches;
  touch = {d: Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY)}; } else if (ev.touches.length === 1) {
  touch = {x: ev.touches[0].clientX, y: ev.touches[0].clientY, vb: vb.slice()}; } }, {passive: true});
svg.addEventListener('touchmove', ev => { if (!touch) return; ev.preventDefault();
  if (ev.touches.length === 2 && touch.d) { const [a, b] = ev.touches, d = Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY);
    const q = toGrid({clientX: (a.clientX + b.clientX) / 2, clientY: (a.clientY + b.clientY) / 2}); zoomAt(q.x, q.y, touch.d / d); touch.d = d; }
  else if (ev.touches.length === 1 && touch.vb) { const k = vb[2] / svg.getBoundingClientRect().width;
    vb[0] = touch.vb[0] - (ev.touches[0].clientX - touch.x) * k; vb[1] = touch.vb[1] - (ev.touches[0].clientY - touch.y) * k; setVB(); dragged = true; } }, {passive: false});
svg.addEventListener('touchend', () => { touch = null; });

function drawStations(){
  const g = document.getElementById('sts'); g.innerHTML = ''; const k = si(), dd = day();
  const ord = D.st.map((s,i)=>i).filter(i => D.st[i].T[k] !== null).sort((a,b)=>D.st[a].T[k]-D.st[b].T[k]);
  const inView = i => { const s = D.st[i]; return s.c >= vb[0] && s.c <= vb[0] + vb[2] && s.r >= vb[1] && s.r <= vb[1] + vb[3]; };
  const lab = new Set(ord.filter(inView).slice(0, zf() < 0.6 ? 16 : 8));
  D.st.forEach((s, i) => {
    const t = s.T[k];
    const c = document.createElementNS('http://www.w3.org/2000/svg','circle');
    c.setAttribute('cx', s.c); c.setAttribute('cy', s.r); c.setAttribute('r', 11 * zf());
    c.setAttribute('fill', col(t)); c.setAttribute('class', 'st' + (i === sel ? ' sel' : ''));
    c.addEventListener('click', ev => { ev.stopPropagation(); select(i); });
    const r6 = rain6(s, k), ty = s.Y[k];
    c.addEventListener('mousemove', ev => showTip(ev, `<b>${s.n}</b> (${s.a} m)<br>${fmt(t)} °C${r6 && r6 >= 0.2 ? ` · ${mm(r6)} mm/6 h ${D.ptypes[ty]}` : ''}`));
    c.addEventListener('mouseleave', () => tip.style.display = 'none');
    g.appendChild(c);
    if (lab.has(i) || i === sel){
      const tx = document.createElementNS('http://www.w3.org/2000/svg','text');
      const right = s.c > D.W * 0.8;
      tx.setAttribute('x', right ? s.c - 14 * zf() : s.c + 14 * zf()); tx.setAttribute('y', s.r - 10 * zf());
      if (right) tx.setAttribute('text-anchor', 'end');
      tx.textContent = `${s.n} ${Math.round(t)}°`; g.appendChild(tx);
    }
  });
}
function showTip(ev, html){
  const r = document.getElementById('mapbox').getBoundingClientRect();
  tip.innerHTML = html; tip.style.display = 'block';
  tip.style.left = Math.min(ev.clientX - r.left + 14, r.width - 220) + 'px'; tip.style.top = (ev.clientY - r.top + 10) + 'px';
}
function toGrid(ev){ const p = svg.createSVGPoint(); p.x = ev.clientX; p.y = ev.clientY; return p.matrixTransform(svg.getScreenCTM().inverse()); }
function pixel(q){
  const i = Math.floor(q.y / D.g.k), j = Math.floor(q.x / D.g.k);
  if (i < 0 || j < 0 || i >= D.g.ny || j >= D.g.nx) return null;
  const n = i * D.g.nx + j, v = grids[fr][n], a = altg[n];
  if (!v) return null;
  const rg = rgrids[fr], r = rg && rg[n] ? (rg[n] - 1) * 0.2 : 0;
  return {t: D.tmin + (v - 1) * 0.25, a: (a - 1) * 10, r};
}
svg.addEventListener('mousemove', ev => {
  if (ev.target.classList && ev.target.classList.contains('st')) return;
  const v = pixel(toGrid(ev));
  if (v) showTip(ev, `${v.a} m · ${fmt(v.t)} °C${v.r >= 0.2 ? ` · ${v.r.toFixed(1)} mm/6 h` : ''}`); else tip.style.display = 'none';
});
svg.addEventListener('mouseleave', () => tip.style.display = 'none');
svg.addEventListener('click', ev => {
  if (dragged) { dragged = false; return; }
  const q = toGrid(ev), v = pixel(q); if (!v) return;
  const m = document.getElementById('mark'); m.setAttribute('cx', q.x); m.setAttribute('cy', q.y); m.setAttribute('r', 14 * zf()); m.style.display = '';
  let best = -1, bd = 1e9;
  D.st.forEach((s, i) => { const d = Math.hypot(s.c - q.x, s.r - q.y); if (d < bd){ bd = d; best = i; } });
  document.getElementById('pt').innerHTML = `<b>Point cliqué</b> : ${v.a} m, <b>${fmt(v.t)} °C</b>${v.r >= 0.2 ? `, ${v.r.toFixed(1)} mm sur 6 h` : ', sec'} — ${D.frames[fr].lab}
    <span class="mute">· station la plus proche : ${D.st[best].n} à ${(bd * D.g.res / 1000).toFixed(1)} km</span>`;
});

function chart(s){
  const W = 1400, H = 400, L = 52, R = 64, T = 30, B = 30, PH = 90, n = D.steps.length;
  document.getElementById('ctitle').textContent = `${s.n} (${s.a} m) — température et précipitations J+0 → J+7`;
  const all = s.T.concat(s.B).filter(v => v !== null);
  let lo = Math.floor(Math.min(...all) - 1.5), hi = Math.ceil(Math.max(...all) + 1.5);
  const x = h => L + (W - L - R) * (h - D.steps[0]) / (D.steps[n - 1] - D.steps[0]);
  const yb = H - B - PH - 10;
  const y = v => T + (yb - T) * (hi - v) / (hi - lo);
  const pmax = Math.max(2, Math.ceil(Math.max(0, ...s.P.filter(v => v !== null))));
  const yp = v => H - B - PH * Math.min(1, v / pmax);
  const h0 = D.h0;   // heure UTC du run
  let o = '';
  D.nights.forEach(([a, b]) => { o += `<rect x="${x(a)}" y="${T}" width="${x(b) - x(a)}" height="${H - T - B}" fill="rgba(90,110,160,.10)"/>`; });
  const stp = (hi - lo > 18) ? 5 : 2;
  for (let v = Math.ceil(lo / stp) * stp; v <= hi; v += stp) o += `<line x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}" stroke="currentColor" opacity=".10"/><text x="${L - 6}" y="${y(v) + 4}" text-anchor="end">${v}°</text>`;
  if (lo < 0 && hi > 0) o += `<line x1="${L}" x2="${W - R}" y1="${y(0)}" y2="${y(0)}" stroke="#2b5d8a" stroke-width="1.6"/>`;
  // graduations horaires (toutes les 6 h) et separateurs de jours
  for (let hh = Math.ceil(D.steps[0] / 6) * 6; hh <= D.steps[n - 1]; hh += 6) {
    const utc = (h0 + hh) % 24, xx = x(hh);
    if (utc === 0) { o += `<line x1="${xx}" x2="${xx}" y1="${T - 6}" y2="${H - B}" stroke="currentColor" opacity=".35"/>`; }
    else o += `<line x1="${xx}" x2="${xx}" y1="${H - B}" y2="${H - B + 4}" stroke="currentColor" opacity=".4"/><text x="${xx}" y="${H - B + 16}" text-anchor="middle" style="font-size:11px">${String(utc).padStart(2, '0')}h</text>`;
  }
  let lastd = -999;
  D.days.forEach((d, i) => { const xx = Math.min(W - R - 40, Math.max(L + 30, x(d.h12))); if (xx - lastd < 95) return; lastd = xx;
    o += `<text class="dl" x="${xx}" y="${T - 12}" text-anchor="middle" ${i === day() ? 'style="text-decoration:underline"' : ''}>${d.lab}</text>`; });
  // pluie
  o += `<line x1="${L}" x2="${W - R}" y1="${H - B}" y2="${H - B}" stroke="currentColor" opacity=".3"/>`;
  o += `<text x="${W - R + 6}" y="${H - B - PH + 10}">${pmax} mm</text><text x="${W - R + 6}" y="${H - B}">0</text>`;
  s.P.forEach((p, i) => { if (p && p >= 0.1 && i > 0) { const x0 = x(D.steps[i - 1]), x1 = x(D.steps[i]);
    o += `<rect x="${x0 + 1}" y="${yp(p)}" width="${Math.max(1, x1 - x0 - 2)}" height="${H - B - yp(p)}" fill="${D.pcolors[s.Y[i]] || '#2f80ed'}" stroke="#0003"/>`;
    if (x1 - x0 > 18 && p >= 0.5) o += `<text x="${(x0 + x1) / 2}" y="${yp(p) - 3}" text-anchor="middle" style="font-size:10px">${p.toFixed(1)}</text>`; } });
  // courbes
  const path = arr => { let d = '', pen = false; arr.forEach((v, i) => { if (v === null) { pen = false; return; }
    d += `${pen ? 'L' : 'M'}${x(D.steps[i]).toFixed(1)},${y(v).toFixed(1)}`; pen = true; }); return d; };
  o += `<path d="${path(s.B)}" fill="none" stroke="#888" stroke-width="1.8" stroke-dasharray="6 4"/>`;
  o += `<path d="${path(s.T)}" fill="none" stroke="#d64541" stroke-width="3"/>`;
  s.T.forEach((v, i) => { if (v !== null) o += `<circle cx="${x(D.steps[i])}" cy="${y(v)}" r="2.6" fill="#d64541"/>`; });
  // Tmin de nuit / Tmax de jour
  D.nights.forEach(([a, b]) => { let km = -1; s.T.forEach((v, i) => { const h = D.steps[i];
      if (v !== null && h >= a && h <= b + 3 && (km < 0 || v < s.T[km])) km = i; });
    if (km >= 0) o += `<text class="tn" x="${x(D.steps[km])}" y="${y(s.T[km]) + 18}" text-anchor="middle">${fmt(s.T[km])}°</text>`; });
  D.days.forEach(d => { let kx = -1; s.T.forEach((v, i) => { const h = D.steps[i];
      if (v !== null && h >= d.h12 - 6 && h <= d.h12 + 6 && (kx < 0 || v > s.T[kx])) kx = i; });
    if (kx >= 0) o += `<text class="tx" x="${x(D.steps[kx])}" y="${y(s.T[kx]) - 9}" text-anchor="middle">${fmt(s.T[kx])}°</text>`; });
  if (D.splice) { const xr = x(D.splice.cut);
    o += `<line x1="${xr}" x2="${xr}" y1="${T}" y2="${H - B}" stroke="#b26a00" stroke-width="1.5" stroke-dasharray="2 4"/>` +
         `<text x="${xr + 4}" y="${T + 12}" style="font-size:11px;fill:#b26a00">run du ${D.splice.base} →</text>`; }
  const xs = x(D.frames[fr].s);
  o += `<line x1="${xs}" x2="${xs}" y1="${T}" y2="${H - B}" stroke="var(--acc)" stroke-width="2"/>`;
  o += `<line id="cguide" x1="0" x2="0" y1="${T}" y2="${H - B}" stroke="currentColor" opacity=".5" stroke-dasharray="3 3" style="display:none"/>`;
  o += `<rect id="chit" x="${L}" y="${T}" width="${W - L - R}" height="${H - T - B}" fill="transparent"/>`;
  const C = document.getElementById('chart'); C.innerHTML = o;
  const near = ev => { const p = C.createSVGPoint(); p.x = ev.clientX; p.y = ev.clientY;
    const q = p.matrixTransform(C.getScreenCTM().inverse()); let best = 0, bd = 1e9;
    D.steps.forEach((h, i) => { const d = Math.abs(x(h) - q.x); if (d < bd) { bd = d; best = i; } }); return best; };
  const hit = document.getElementById('chit'), g = document.getElementById('cguide');
  hit.onmousemove = ev => { const k = near(ev), xx = x(D.steps[k]);
    g.setAttribute('x1', xx); g.setAttribute('x2', xx); g.style.display = '';
    const r = document.getElementById('chartbox').getBoundingClientRect();
    ctip.innerHTML = `<b>${D.shour[k]} UTC</b><br>CRYOGÈNE ${fmt(s.T[k])} °C · IFS seul ${fmt(s.B[k])} °C<br>` +
      `effet relief ${fmt(s.T[k] - s.B[k])} °C · inversion IFS ${fmt(s.I[k])} K` +
      (s.P[k] >= 0.1 ? `<br>${mm(s.P[k])} mm ${s.P[k] >= 0.2 ? D.ptypes[s.Y[k]] : ''}` : '');
    ctip.style.display = 'block'; ctip.style.left = Math.min(ev.clientX - r.left + 14, r.width - 260) + 'px'; ctip.style.top = (ev.clientY - r.top + 12) + 'px'; };
  hit.onmouseleave = () => { ctip.style.display = 'none'; g.style.display = 'none'; };
  hit.onclick = ev => { const k = near(ev); let bf = 0, bd = 1e9;
    D.frames.forEach((f, j) => { const d = Math.abs(f.s - D.steps[k]); if (d < bd) { bd = d; bf = j; } }); setFrame(bf); };
}
function hourly(s){
  const dd = day(); let h = '';
  D.steps.forEach((st, k) => { if (D.sday[k] !== dd) return;
    const on = st === D.frames[fr].s ? ' class="on"' : '', fi = D.frames.findIndex(f => f.s === st);
    h += `<tr${on} data-f="${fi}"><td>${D.shour[k]}</td><td style="background:${col(s.T[k])};color:#111">${fmt(s.T[k])}</td><td>${fmt(s.B[k])}</td><td>${mm(s.P[k])}</td><td>${s.P[k] >= 0.2 ? D.ptypes[s.Y[k]] : ''}</td></tr>`; });
  const tb = document.querySelector('#hourly tbody'); tb.innerHTML = h;
  tb.querySelectorAll('tr').forEach(tr => { if (+tr.dataset.f >= 0) tr.onclick = () => setFrame(+tr.dataset.f); });
}
function select(i){
  sel = i; const s = D.st[i], k = si(), dd = day();
  document.getElementById('pick').value = i;
  document.getElementById('sname').textContent = s.n;
  document.getElementById('salt').textContent = `${s.a} m · poste ${s.id}`;
  const r6 = rain6(s, k);
  document.getElementById('kpi').innerHTML = `
    <div><span class="mute">T ${D.frames[fr].hl}</span><b>${fmt(s.T[k])} °C</b></div>
    <div><span class="mute">Tmin de la nuit</span><b>${fmt(s.tmin[dd])} °C</b></div>
    <div><span class="mute">Effet relief local</span><b>${fmt(s.T[k] - s.B[k])} °C</b></div>
    <div><span class="mute">Inversion IFS au point</span><b>${fmt(s.I[k])} K</b></div>
    <div style="grid-column:1/3"><span class="mute">Précipitations 6 h (IFS)</span><b>${r6 && r6 >= 0.2 ? mm(r6) + ' mm · ' + D.ptypes[s.Y[k]] : 'sec'}</b></div>`;
  document.getElementById('chips').innerHTML = D.days.map((d, j) => s.tmin[j] === null ? '' :
    `<span data-d="${j}" class="${j === dd ? 'on' : ''}" style="background:${col(s.tmin[j])}">${d.sl}<br><b>${fmt(s.tmin[j])}°</b></span>`).join('');
  document.querySelectorAll('#chips span').forEach(c => c.onclick = () => setDay(+c.dataset.d));
  chart(s); hourly(s); drawStations();
}
function setFrame(f){
  fr = f; const F = D.frames[f];
  img.setAttribute('href', 'data:image/jpeg;base64,' + F.img);
  rimg.innerHTML = F.rain || '';
  rimg.style.display = document.getElementById('rtog').checked ? '' : 'none';
  document.querySelectorAll('.tabs button').forEach((b, i) => b.classList.toggle('on', i === F.d));
  const hb = document.getElementById('hours'); hb.innerHTML = '<span class="mute">Heure :</span>';
  D.frames.forEach((g, j) => { if (g.d !== F.d) return; const b = document.createElement('button');
    b.textContent = g.hl; b.className = j === f ? 'on' : ''; b.onclick = () => setFrame(j); hb.appendChild(b); });
  const d = D.days[F.d], k = si(), c = D.st[D.cl], z = D.st[D.cc];
  document.getElementById('dinfo').innerHTML = `<b>${F.lab}</b> · fiabilité <b style="color:${d.fc}">${d.fiab}</b>` +
    (c && z ? ` · écart Clermont − Cacadogne ${fmt(c.T[k] - z.T[k])} °C` : '') +
    (D.inv.length ? '<br>' + D.inv.map(v => { const p = v.P[k]; if (p === null) return '';
      return `<span class="invchip" style="background:${pcol(p)};color:${p > .45 ? '#fff' : '#222'}" title="${v.low} / ${v.high}">${v.name} ${Math.round(100 * p)} %${v.R[k] ? ' · ' + D.regimes[v.R[k]] : ''}</span>`; }).join('') : '');
  invStrip();
  document.getElementById('pt').innerHTML = '<span class="mute">Clique sur la carte pour la valeur d\'un point, sur un rond pour une station.</span>';
  document.getElementById('mark').style.display = 'none';
  document.querySelectorAll('#tbl tbody tr').forEach(tr => tr.querySelectorAll('td').forEach((td, j) => td.style.fontWeight = (j === F.d + 2 ? '700' : '')));
  select(sel);
}
function setDay(d){
  const fs = D.frames.map((f, j) => [f, j]).filter(([f]) => f.d === d);
  if (!fs.length) return;
  const cur = D.frames[fr].h, same = fs.find(([f]) => f.h === cur), six = fs.find(([f]) => f.h === 6);
  setFrame((same || six || fs[0])[1]);
}
document.querySelectorAll('.tabs button').forEach((b, i) => b.onclick = () => setDay(i));
document.getElementById('rtog').onchange = () => setFrame(fr);
const pick = document.getElementById('pick');
D.st.map((s, i) => [s.n, i]).sort().forEach(([n, i]) => { const o = document.createElement('option'); o.value = i; o.textContent = `${n} (${D.st[i].a} m)`; pick.appendChild(o); });
pick.onchange = () => select(+pick.value);
document.querySelectorAll('#tbl tbody tr').forEach(tr => tr.onclick = () => { select(+tr.dataset.i); window.scrollTo({top: 0, behavior: 'smooth'}); });
document.addEventListener('keydown', e => { if (e.target.tagName === 'SELECT') return;
  if (e.key === 'ArrowRight' && fr < D.frames.length - 1) setFrame(fr + 1);
  if (e.key === 'ArrowLeft' && fr > 0) setFrame(fr - 1); });
if (D.dmain) { document.getElementById('dmain').setAttribute('d', D.dmain); document.getElementById('dnear').setAttribute('d', D.dnear); }
if (!D.inv.length) document.getElementById('invcard').style.display = 'none';
document.querySelectorAll('.tabs button').forEach((b, i) => {
  if (!D.inv.length) return;
  let mx = 0; D.steps.forEach((s, k) => { if (D.sday[k] === i) D.inv.forEach(v => { if (v.P[k] !== null) mx = Math.max(mx, v.P[k]); }); });
  b.insertAdjacentHTML('beforeend', `<span class="dot" style="background:${pcol(mx)}" title="P inversion max ${Math.round(100 * mx)} %"></span>`); });
setFrame(fr);
"""


def fiab(lead_days):
    return ("élevée", "#2e7d32") if lead_days <= 2 else (("moyenne", "#b26a00") if lead_days <= 4 else ("tendance", "#c0392b"))


def tcell(v):
    if not np.isfinite(v):
        return "<td>–</td>"
    return f'<td style="background:{hex_color(v)};color:{"#fff" if v < -9 else "#111"}">{v:.1f}</td>'


def build_html(run, bundle, data, tmin_tab, days, ny100, nx100):
    import json
    gen = dt.datetime.now().strftime("%d/%m/%Y %H:%M")
    sp = data.get("splice")
    splice_txt = (f" (jusqu'à +{sp['cut']} h, puis run du {sp['base']} au-delà)" if sp else "")
    tabs = "".join(f'<button>{d["lab"]}</button>' for d in days)
    nd = [i for i, d in enumerate(days) if d["has_night"]]
    head = "".join(f"<th>{days[i]['lab']}</th>" for i in nd)
    body = ""
    for i, r in tmin_tab.sort_values("ALTI").iterrows():
        cells = "".join(tcell(r[f"n{j}"]) for j in nd)
        body += f'<tr data-i="{i}"><td>{r.NOM_USUEL.title()}</td><td>{int(r.ALTI)}</td>{cells}</tr>'
    js_data = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    plg = ('<span style="color:#1f5fbf;font-weight:700">━</span> isohyètes 1 · 2 · 5 · 10 · 20 mm '
           '<span class="sw" style="background:#fff;border:2px dashed #999"></span> neige '
           '<span class="sw" style="background:#e0218a55;border:2px solid #e0218a"></span> risque de pluie verglaçante')
    return f"""<!doctype html><html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>CRYOGÈNE prévision</title>
<style>{CSS}</style></head><body>
<header><h1>CRYOGÈNE <span style="font-weight:400">— le froid au creux des vallées, prévu jusqu'à 7 jours</span></h1>
<a href="#apropos" style="color:var(--acc)">Qu'est-ce que c'est ?</a>
<span class="mute">Run IFS HRES {pd.Timestamp(run):%d/%m/%Y %H} UTC{splice_txt} · généré le {gen} ·
modèle ML ERA5 × {bundle.get('n_stations', '?')} stations ({bundle.get('trained_on', '?')})</span></header>
<main>
<div class="card">
  <div class="tabs">{tabs}</div>
  <div class="hours" id="hours"></div>
  <div id="dinfo" style="margin-bottom:8px"></div>
  <div class="layout">
    <div>
      <div id="mapbox">
        <svg id="map" viewBox="0 0 {nx100} {ny100}" preserveAspectRatio="xMidYMid meet">
          <image id="img" x="0" y="0" width="{nx100}" height="{ny100}" preserveAspectRatio="none"/>
          <g id="rain"></g>
          <path id="dnear" fill="none" stroke="#333" stroke-opacity=".45" stroke-width="1.2" stroke-dasharray="6 4"/>
          <path id="dmain" fill="none" stroke="#111" stroke-opacity=".85" stroke-width="2.4"/>
          <g id="sts"></g>
          <circle id="mark" r="14" fill="none" stroke="#ff3b30" stroke-width="3" style="display:none;vector-effect:non-scaling-stroke"/>
        </svg>
        <div id="tip"></div>
        <div class="zoom"><button id="zin" title="Zoomer">+</button><button id="zout" title="Dézoomer">−</button><button id="zreset" title="Vue entière">⟲</button></div>
      </div>
      <div class="legend"><span>°C</span>{legend_html()}</div>
      <div class="legend"><label><input type="checkbox" id="rtog" checked> Précipitations IFS (6 h précédentes)</label> {plg}
        <span class="mute">· ← → pour changer d'heure · molette / double-clic = zoom, glisser = déplacer</span></div>
      <div id="pt" style="margin-top:6px"></div>
    </div>
    <div>
      <select id="pick"></select>
      <h2 id="sname" style="font-size:19px;margin:10px 0 0"></h2><div id="salt" class="mute"></div>
      <div id="kpi" class="k" style="margin-top:8px"></div>
      <h2>Tmin par nuit (18 → 09 UTC)</h2><div id="chips" class="chips"></div>
      <h2>Détail du jour</h2>
      <table id="hourly"><thead><tr><th>Heure</th><th>T °C</th><th>IFS</th><th>mm</th><th>Type</th></tr></thead><tbody></tbody></table>
    </div>
  </div>
  <div id="chartbox">
    <h2 id="ctitle"></h2>
    <svg id="chart" viewBox="0 0 1400 400"></svg>
    <div id="ctip"></div>
    <div class="mute" style="font-size:12px"><span style="color:#d64541;font-weight:700">━</span> CRYOGÈNE (relief local) ·
      <span style="color:#888">╌</span> profil IFS seul · barres = pluie par pas de temps (couleur = type) · bandes = nuits ·
      étiquettes = Tmin de la nuit / Tmax du jour · survol = valeurs, clic = carte à cette heure</div>
  </div>
</div>
<div class="card" id="invcard"><h2 style="margin-top:0">Inversions : probabilité par secteur</h2>
<div class="mute" style="margin-bottom:6px">Probabilité que la station de vallée soit plus froide que la station d'altitude
(modèle calé sur 6 hivers d'observations). Barre grise = nébulosité basse IFS : haute + inversion = stratus / grisaille en plaine,
soleil en altitude ; basse + inversion = refroidissement radiatif, gelées dans les creux. Clic = carte à cette heure.</div>
<svg id="invs" style="width:100%;height:auto"></svg>
<div class="legend"><span>0 %</span><div class="bar" style="max-width:260px;background:linear-gradient(90deg,#f2f0f7,#cbc9e2,#9e9ac8,#756bb1,#54278f)"></div><span>100 %</span>
<span class="sw" style="background:#8a8f98"></span> nébulosité basse</div></div>
<div class="card"><h2 style="margin-top:0">Tmin prévue par nuit, toutes stations (clic = détail)</h2>
<div class="scroll"><table id="tbl"><thead><tr><th>Station</th><th>Alt.</th>{head}</tr></thead><tbody>{body}</tbody></table></div>
<p class="mute">Heures en UTC (heure légale d'hiver = UTC + 1). Au-delà de +144 h l'IFS open data passe au pas de 6 h.</p></div>
<div class="card" id="apropos"><h2 style="margin-top:0">CRYOGÈNE — le froid au creux des vallées, prévu jusqu'à 7 jours</h2>
<p>En hiver, par temps calme et ciel dégagé, l'air froid, plus lourd, descend la nuit le long des pentes et s'accumule
au fond des vallées et des cuvettes. On peut y relever 5 à 10 °C de moins que sur les hauteurs voisines : c'est
l'<b>inversion de température</b>, quand il fait plus froid en bas qu'en haut. Elle explique les gelées dans les creux, et la
grisaille tenace sur la Limagne alors que le soleil brille sur les puys. Elle favorise aussi l'accumulation de la pollution
dans les villes de vallée.</p>
<p>Les modèles météo habituels voient le relief à plusieurs kilomètres près et passent à côté de ces contrastes. CRYOGÈNE part
de la prévision du Centre européen (ECMWF) et l'affine pour chaque point du Puy-de-Dôme, du Cantal et de la Corrèze. Il
s'appuie sur la forme du relief (creux, pentes, ciel plus ou moins dégagé) et sur un apprentissage automatique entraîné sur
six hivers de mesures de 80 stations Météo-France.</p>
<p><b>Comment lire la page</b> : choisissez un jour et une heure. La carte montre la température prévue ; survolez-la ou
cliquez dessus pour une valeur précise, zoomez avec la molette. Cliquez sur une station pour sa courbe sur 7 jours. La frise
violette indique la probabilité d'inversion par secteur, et la barre grise la présence de nuages bas (grisaille). Les lignes
bleues sur la carte sont les cumuls de pluie sur 6 h.</p>
<p><b>Fiabilité</b> : de l'ordre de 1 à 2 °C aux stations pour les deux premiers jours. Au-delà de 4 à 5 jours, c'est une
tendance. Outil expérimental : il ne remplace pas les prévisions et vigilances de
<a href="https://vigilance.meteofrance.fr" style="color:var(--acc)">Météo-France</a>.</p>
<p class="mute" style="font-size:12px">Données : prévisions ECMWF open data (CC BY 4.0), réanalyse ERA5 (Copernicus),
observations Météo-France (data.gouv.fr), relief et limites administratives IGN.</p>
<details class="mute"><summary style="cursor:pointer">Détails techniques</summary>
<p>Température = profil IFS ramené à l'altitude de chaque point + correction apprise
(relief local : TPI, sky view factor, drainage d'air froid ; situation : inversion, vent, nébulosité).
Précision mesurée sur 6 hivers : ~1,2 °C aux stations (1,6 °C les matins d'inversion) ; entre les stations ~1,5 °C
(2,6 °C par inversion). S'y ajoute l'erreur de la prévision IFS, qui croît avec l'échéance : J+1–J+2 utilisables,
J+3–J+4 ordres de grandeur, J+5–J+7 tendance de régime.</p>
<p>Inversion : probabilité que la station de vallée soit plus froide que la station d'altitude (classifieur calé sur
6 hivers d'observations ; régime déduit de la nébulosité basse et totale IFS).</p>
<p>Précipitations = IFS brut, non affinées par le relief. Type déduit de la température CRYOGÈNE au sol et du profil IFS :
« pluie verglaçante ? » = précipitation avec couche &gt; 1 °C en altitude et sol ≤ 0,5 °C (situation d'inversion) —
indicateur de risque, pas une prévision officielle.</p></details></div>
</main>
<script id="data" type="application/json">{js_data}</script>
<script>{JS}</script>
</body></html>"""


# --------------------------------------------------------------------------- #
def main(run=None, res=300, open_browser=True, every=6):
    bundle = load_model()
    ds, run = get_run(run)
    run_ts = pd.Timestamp(run)
    ds_steps = [int(s / np.timedelta64(1, "h")) for s in ds.step.values]
    print(f"[cuvette] run {run_ts:%Y-%m-%d %H}z, {len(ds_steps)} échéances")

    F, meta = TF.load()
    res100 = float(meta["res"])
    alt100 = F["alt"]
    ny100, nx100 = alt100.shape
    a_, b_, c_, d_, e_, f_ = meta["transform"]
    hs = hillshade(alt100, res100)
    depts = departements()
    inside = dept_mask(depts, meta["transform"], alt100.shape)
    inv_bundle = load_inversion_model()

    # ---------------- stations : toutes les echeances ----------------
    st = stations()
    from pyproj import Transformer
    st["X"], st["Y"] = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True).transform(
        st.LON.values, st.LAT.values)
    print(f"[cuvette] prévision aux {len(st)} stations ...", flush=True)
    dfst = predict_points(ds, run, st, bundle)
    steps = sorted(int(x) for x in dfst.step_h.unique())
    piv = lambda col: dfst.pivot(index="pt", columns="step_h", values=col).reindex(index=range(len(st)), columns=steps)
    pT, pB, pI, pA = piv("T_cuv"), piv("T_base"), piv("inv_strength"), piv("tmax_aloft")
    if "tp_cum" in dfst:
        cum = piv("tp_cum").to_numpy()
        pP = np.maximum(np.diff(np.concatenate([np.zeros((len(st), 1)), cum], axis=1), axis=1), 0)
        pP[:, 0] = 0
    else:
        pP = np.zeros((len(st), len(steps)))
    pY = ptype(pP, pT.to_numpy(), pA.to_numpy())

    # ---------------- jours / nuits ----------------
    valid = [run_ts + pd.Timedelta(hours=s) for s in steps]
    last = valid[-1]
    dates = sorted({v.normalize() for v in valid if v.hour % every == 0 and v > run_ts})
    days, tmin_tab = [], st[["NOM_USUEL", "ALTI"]].copy()
    vser = pd.Series(valid)
    for j, dte in enumerate(dates):
        lead = (dte - run_ts.normalize()).days
        f, fc = fiab(lead)
        lab = f"{JOURS[dte.weekday()]} {dte:%d/%m}"
        t0, t1 = dte - pd.Timedelta(hours=6), dte + pd.Timedelta(hours=9)
        has = (t0 >= valid[0]) and (dte + pd.Timedelta(hours=6) <= last)
        if has:
            cols = [k for k, v in enumerate(valid) if t0 <= v <= t1]
            tmin_tab[f"n{j}"] = pT.to_numpy()[:, cols].min(axis=1)
        else:
            tmin_tab[f"n{j}"] = np.nan
        noon = dte + pd.Timedelta(hours=12)
        days.append(dict(lab=lab, sl=f"{JOURS[dte.weekday()][:-1]} {dte.day}", fiab=f, fc=fc, has_night=bool(has),
                         h12=float(np.clip((noon - run_ts) / pd.Timedelta(hours=1), steps[0], steps[-1]))))

    # ---------------- cartes toutes les `every` heures ----------------
    pts, valid_mask, altk, _ = grid_points(res)
    k = max(1, int(round(res / res100)))
    k2 = 2                                                   # grille de clic : 1 point sur 2
    frames = []
    fsteps = [s for s, v in zip(steps, valid) if v.hour % every == 0 and v > run_ts]
    for n, s in enumerate(fsteps, 1):
        v = run_ts + pd.Timedelta(hours=s)
        dj = dates.index(v.normalize())
        print(f"[cuvette] carte {n}/{len(fsteps)} : {v:%d/%m %H} UTC (+{s} h) ...", flush=True)
        dfp = predict_points(ds, run, pts, bundle, steps=[s]).sort_values("pt")
        Tk = np.full(altk.shape, np.nan); Tk[valid_mask] = dfp.T_cuv.to_numpy()
        Ak = np.full(altk.shape, np.nan); Ak[valid_mask] = dfp.tmax_aloft.to_numpy()
        Rk = np.full(altk.shape, np.nan); Rk[valid_mask] = rain_at(ds, pts, s, ds_steps)
        Pk = ptype(np.nan_to_num(Rk), Tk, Ak)
        hl = f"{v:%H} h UTC"
        frames.append(dict(
            s=int(s), d=dj, h=int(v.hour), hl=hl, lab=f"{JOURS[v.weekday()]} {v:%d/%m} {v:%H} UTC (+{s} h)",
            img=render_jpeg(upsample(Tk, k, alt100.shape), hs, alt100, inside),
            rain=rain_svg(Rk, Pk, k),
            grid=pack_u8(Tk[::k2, ::k2], TMIN_C - 12, 0.25),
            rgrid=pack_u8(np.where(Rk >= RAIN_MIN, Rk, np.nan)[::k2, ::k2], 0, 0.2) if np.nanmax(np.nan_to_num(Rk)) >= RAIN_MIN else None))

    # ---------------- donnees pour la page ----------------
    r1 = lambda v: None if v is None or not np.isfinite(v) else round(float(v), 1)
    st_json = []
    for i, s in st.iterrows():
        st_json.append(dict(
            id=s.NUM_POSTE, n=s.NOM_USUEL.title(), a=int(s.ALTI),
            c=round((s.X - c_) / a_, 1), r=round((s.Y - f_) / e_, 1),
            T=[r1(x) for x in pT.loc[i]], B=[r1(x) for x in pB.loc[i]], I=[r1(x) for x in pI.loc[i]],
            P=[r1(x) for x in pP[i]], Y=[int(x) for x in pY[i]],
            tmin=[r1(tmin_tab.loc[i, f"n{j}"]) for j in range(len(days))]))
    nights_h = []
    for dte in dates:
        a0 = max((dte - pd.Timedelta(hours=6) - run_ts) / pd.Timedelta(hours=1), steps[0])
        a1 = min((dte + pd.Timedelta(hours=6) - run_ts) / pd.Timedelta(hours=1), steps[-1])
        if a1 > a0:
            nights_h.append([a0, a1])
    names = st.NOM_USUEL.to_numpy()
    idx = lambda nm: int(np.where(names == nm)[0][0]) if (names == nm).any() else None
    cvals = np.arange(TMIN_C - 12, TMAX_C + 12.01, 0.5)
    f0 = next((j for j, f in enumerate(frames) if f["h"] == 6), 0)
    data = dict(
        steps=steps, sday=[dates.index(v.normalize()) if v.normalize() in dates else -1 for v in valid],
        shour=[f"{v:%d/%m %H} h" for v in valid], nights=nights_h, days=days, frames=frames, f0=f0,
        g=dict(k=k * k2, ny=int(altk[::k2, ::k2].shape[0]), nx=int(altk[::k2, ::k2].shape[1]), res=res100,
               alt=pack_u8(altk[::k2, ::k2], 0, 10)),
        st=st_json, colors=[hex_color(x) for x in cvals], tmin=TMIN_C - 12, cstep=0.5,
        ptypes=PTYPES, pcolors=PCOLORS, cl=idx(PAIR[0]), cc=idx(PAIR[1]), W=int(nx100), h0=int(run_ts.hour))
    data["def"] = data["cl"] if data["cl"] is not None else 0
    spl = ds.attrs.get("splice", "")
    data["splice"] = dict(cut=int(spl.split("|")[0]), base=spl.split("|")[1]) if spl else None
    data["inv"] = inversion_proba(inv_bundle, dfst, st, steps)
    data["regimes"] = REGIMES
    data["dmain"] = svg_paths(depts["main"], a_, c_, e_, f_) if depts else ""
    data["dnear"] = svg_paths(depts["near"], a_, c_, e_, f_) if depts else ""

    os.makedirs(OUT_DIR, exist_ok=True)
    html = build_html(run, bundle, data, tmin_tab, days, ny100, nx100)
    out = os.path.join(OUT_DIR, f"cryogene_{run_ts:%Y%m%d_%H}z.html")
    for p in (out, os.path.join(OUT_DIR, "cryogene_derniere.html")):
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(html)
    print(f"[cuvette] -> {out}  ({len(html) / 1e6:.1f} Mo)")
    if open_browser:
        webbrowser.open("file:///" + out.replace("\\", "/"))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="ex 2026-11-12T00")
    ap.add_argument("--res", type=int, default=300, help="resolution du calcul des cartes (m)")
    ap.add_argument("--every", type=int, default=6, choices=[3, 6], help="une carte toutes les 3 ou 6 h")
    ap.add_argument("--no-open", action="store_true")
    a = ap.parse_args()
    r = dt.datetime.fromisoformat(a.run).replace(tzinfo=dt.timezone.utc) if a.run else None
    main(r, a.res, not a.no_open, a.every)
