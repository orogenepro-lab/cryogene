"""
ecmwf_fetch.py - Acces a l'open data ECMWF (IFS HRES/ENS, 0.25 deg) pour la chaine
"froid / inversions" (CUVETTE). Telecharge, decode (cfgrib) et regride sur la grille
Lambert 93 du domaine 63/15/19. Archive chaque run dans ECMWF_ARCHIVE/<run>/.

    pip install ecmwf-opendata cfgrib xarray pyproj
    (eccodes est deja en place via l'env 'icon')

Usage rapide :
    python ecmwf_fetch.py                  # dernier run HRES, J+0..J+7, sauve en archive
    python ecmwf_fetch.py --ens            # idem + ensemble (50 membres) sur le domaine
Depuis un script :
    import ecmwf_fetch as EF
    ds = EF.fetch_run(ens=False)           # xarray.Dataset lat/lon decoupe
    f  = EF.to_l93(ds, "t2m", step=24)     # champ 2D sur la grille L93 (res 1 km)
"""
import os, sys, datetime as dt, argparse
import numpy as np

# ============================== CONFIG ======================================= #
BBOX_L93 = (565000, 6400000, 730000, 6545000)   # meme emprise qu'OROGENE (63+15+19)
PAD_DEG = 0.75                                    # marge lat/lon autour du domaine
RES_M = 1000.0                                    # grille L93 de sortie (m)
SOURCE = os.environ.get("CUVETTE_ECMWF_SOURCE", "ecmwf")   # "ecmwf", "azure", "aws" ou "google" (miroirs)
ARCHIVE = os.environ.get("CUVETTE_ECMWF_ARCHIVE", r"C:\cuvette_archive\ecmwf")
# Open data, runs 00/12 UTC : 0-144 h par pas de 3 h, puis 150-360 h par pas de 6 h.
STEPS_HRES = list(range(0, 145, 3)) + list(range(150, 181, 6))   # J+0 .. J+7.5
STEPS_ENS = list(range(0, 145, 6)) + list(range(150, 181, 6))
# parametres surface et niveaux pression (noms open data)
SFC_PARAMS = ["2t", "10u", "10v", "msl", "sp", "skt", "tcc", "lcc", "tp"]   # lcc : stratus d inversion
PL_PARAMS = ["t", "q", "u", "v", "gh"]
PL_LEVELS = [1000, 925, 850, 700, 500]
# ============================================================================ #


def _bbox_latlon():
    """Emprise lat/lon (N, W, S, E) du domaine L93 + marge."""
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)
    xs = [BBOX_L93[0], BBOX_L93[2], BBOX_L93[0], BBOX_L93[2]]
    ys = [BBOX_L93[1], BBOX_L93[1], BBOX_L93[3], BBOX_L93[3]]
    lons, lats = tr.transform(xs, ys)
    return (max(lats) + PAD_DEG, min(lons) - PAD_DEG, min(lats) - PAD_DEG, max(lons) + PAD_DEG)


def latest_run(ens=False):
    """Date/heure du dernier run publie (sans telecharger)."""
    from ecmwf.opendata import Client
    c = Client(source=SOURCE)
    return c.latest(type="pf" if ens else "fc", stream="enfo" if ens else "oper",
                    step=24, param="2t")


def download(run_dt=None, ens=False, steps=None, out_dir=None):
    """Telecharge sfc + pl pour un run dans out_dir. Retourne (path_sfc, path_pl, run_dt).
    run_dt=None -> dernier run. ens=True -> 50 membres perturbes (stream enfo)."""
    from ecmwf.opendata import Client
    c = Client(source=SOURCE)
    steps = steps or (STEPS_ENS if ens else STEPS_HRES)
    kind = "ens" if ens else "hres"
    common = dict(stream="enfo" if ens else "oper", type="pf" if ens else "fc", step=steps)
    if ens:
        common["number"] = list(range(1, 51))
    if run_dt is not None:
        common["date"] = run_dt.strftime("%Y%m%d"); common["time"] = run_dt.hour
    out_dir = out_dir or os.path.join(ARCHIVE, "_tmp")
    os.makedirs(out_dir, exist_ok=True)
    p_sfc = os.path.join(out_dir, f"{kind}_sfc.grib2")
    p_pl = os.path.join(out_dir, f"{kind}_pl.grib2")
    print(f"[ecmwf] telechargement {kind} sfc ({len(steps)} ech.)...")
    r = c.retrieve(levtype="sfc", param=SFC_PARAMS, target=p_sfc, **common)
    print(f"[ecmwf] telechargement {kind} pl ({len(PL_LEVELS)} niveaux)...")
    c.retrieve(levtype="pl", param=PL_PARAMS, levelist=PL_LEVELS, target=p_pl, **common)
    run = r.datetime.replace(tzinfo=dt.timezone.utc)
    # renomme le dossier avec la date de run reelle
    final = os.path.join(ARCHIVE, run.strftime("%Y%m%d_%Hz"))
    os.makedirs(final, exist_ok=True)
    for p in (p_sfc, p_pl):
        dst = os.path.join(final, os.path.basename(p))
        if os.path.exists(dst):
            os.remove(dst)
        os.replace(p, dst)
    print(f"[ecmwf] run {run:%Y-%m-%d %Hz} -> {final}")
    return os.path.join(final, os.path.basename(p_sfc)), os.path.join(final, os.path.basename(p_pl)), run


def open_grib(path, crop=True):
    """Ouvre un GRIB open data en xarray (cfgrib), concatene les 'hypercubes'
    heterogenes (10u/10v/2t ont des typeOfLevel differents) et decoupe au domaine."""
    import xarray as xr
    import cfgrib
    dss = cfgrib.open_datasets(path, backend_kwargs={"indexpath": ""})
    merged = {}
    for d in dss:
        for v in d.data_vars:
            da = d[v]
            for coord in ("heightAboveGround", "surface", "meanSea", "entireAtmosphere"):
                if coord in da.coords:
                    da = da.drop_vars(coord)
            merged[v] = da
    ds = xr.Dataset(merged)
    if crop:
        n, w, s, e = _bbox_latlon()
        lon = ds.longitude
        if float(lon.max()) > 180:                   # 0..360 -> -180..180
            ds = ds.assign_coords(longitude=((lon + 180) % 360) - 180).sortby("longitude")
        ds = ds.sel(latitude=slice(n, s), longitude=slice(w, e))
    return ds


def fetch_run(run_dt=None, ens=False):
    """Telecharge (si absent) puis ouvre le run : retourne un Dataset decoupe
    avec sfc + pl fusionnes (dims : [number], step, [isobaricInhPa], latitude, longitude)."""
    import xarray as xr
    p_sfc, p_pl, run = download(run_dt, ens)
    ds = xr.merge([open_grib(p_sfc), open_grib(p_pl)], compat="override")
    ds.attrs["run"] = run.isoformat()
    return ds


def l93_grid(res_m=RES_M):
    """Grille L93 reguliere du domaine : (gx, gy, XX, YY, lon2d, lat2d)."""
    from pyproj import Transformer
    xmin, ymin, xmax, ymax = BBOX_L93
    gx = np.arange(xmin + res_m / 2, xmax, res_m)
    gy = np.arange(ymax - res_m / 2, ymin, -res_m)      # ligne 0 = nord
    XX, YY = np.meshgrid(gx, gy)
    tr = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)
    lon, lat = tr.transform(XX.ravel(), YY.ravel())
    return gx, gy, XX, YY, lon.reshape(XX.shape), lat.reshape(YY.shape)


_GRID = None


def to_l93(ds, var, step=None, level=None, number=None, res_m=RES_M):
    """Interpole bilineairement un champ lat/lon du Dataset sur la grille L93.
    step en heures, level en hPa, number = membre ENS. Retourne un array 2D (ny, nx)."""
    global _GRID
    from scipy.interpolate import RegularGridInterpolator
    if _GRID is None:
        _GRID = l93_grid(res_m)
    gx, gy, XX, YY, lon2d, lat2d = _GRID
    da = ds[var]
    sel = {}
    if step is not None and "step" in da.dims:
        sel["step"] = np.timedelta64(int(step), "h")
    if level is not None and "isobaricInhPa" in da.dims:
        sel["isobaricInhPa"] = level
    if number is not None and "number" in da.dims:
        sel["number"] = number
    da = da.sel(**sel).squeeze()
    lats = np.asarray(da.latitude.values); lons = np.asarray(da.longitude.values)
    vals = np.asarray(da.values, dtype="float64")
    if lats[0] > lats[-1]:
        lats = lats[::-1]; vals = vals[::-1, :]
    f = RegularGridInterpolator((lats, lons), vals, bounds_error=False, fill_value=np.nan)
    return f(np.column_stack([lat2d.ravel(), lon2d.ravel()])).reshape(XX.shape)


def list_archive():
    """Runs disponibles dans l'archive (tries)."""
    if not os.path.isdir(ARCHIVE):
        return []
    return sorted(d for d in os.listdir(ARCHIVE) if d.endswith("z"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ens", action="store_true", help="telecharge aussi l'ensemble")
    ap.add_argument("--run", default=None, help="run force, ex 2026-11-11T00")
    a = ap.parse_args()
    run_dt = dt.datetime.fromisoformat(a.run).replace(tzinfo=dt.timezone.utc) if a.run else None
    ds = fetch_run(run_dt, ens=False)
    print(ds)
    t2 = to_l93(ds, "t2m", step=24) - 273.15
    t850 = to_l93(ds, "t", step=24, level=850) - 273.15
    print(f"[check] T2m +24h : min {np.nanmin(t2):.1f}  max {np.nanmax(t2):.1f} C | "
          f"T850 moy {np.nanmean(t850):.1f} C | delta T850-T2m moy {np.nanmean(t850 - t2):+.1f} C")
    if a.ens:
        fetch_run(run_dt, ens=True)
    print(f"[archive] runs en cache : {list_archive()[-5:]}")
