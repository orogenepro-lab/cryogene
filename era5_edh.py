"""
era5_edh.py - ERA5 via l'Earth Data Hub (DestinE) : MEME DONNEE que le CDS, mais en
Zarr lisible directement, SANS FILE D'ATTENTE. Remplace era5_fetch.py (CDS sature).
Produit des fichiers que era5_fetch.open_all() et profile_inv.py lisent tels quels.

Prerequis (une fois) :
    pip install zarr fsspec aiohttp requests dask
    Compte DestinE : https://earthdatahub.destine.eu  (Sign in -> creer un compte)
    Jeton : https://earthdatahub.destine.eu/account-settings -> "My personal access tokens"
    Colle le jeton seul dans le fichier  C:\\Users\\nikle\\.edh_token   (une ligne, rien d'autre)
    (ou variable d'environnement EDH_TOKEN)

Usage :
    python era5_edh.py --test            # verifie le jeton + affiche la structure (1 min)
    python era5_edh.py                   # hivers 2020-21 -> 2025-26, horaire
    python era5_edh.py --from 2010       # plus d'hivers (reprise : les hivers deja la sont sautes)

Sortie (dans ERA5_DIR, comme era5_fetch) :
    era5_pl_2020-21.nc   t, u, v, z sur 1000/925/850/700 hPa, oct -> mars, horaire
    era5_sfc_2020-21.nc  t2m d2m u10 v10 msl sp tcc lcc strd sd blh
    era5_orog.nc         zs (m), lsm : relief ERA5 exact
Les Zarr sont decoupes en blocs de 45-60 jours x 15 x 15 deg : on telecharge un hiver
entier d'un coup pour ne lire chaque bloc qu'une fois (~1-2 Go par hiver, quelques minutes).
"""
import os, sys, argparse, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd())

# ============================== CONFIG ======================================= #
ERA5_DIR = os.environ.get("CUVETTE_ERA5_DIR", r"C:\cuvette_archive\era5")
BASE = "https://edh:{tok}@data.earthdatahub.destine.eu/era5/"
ZARR_PL = "era5-pressure-levels-v0.zarr"
ZARR_SFC = "era5-single-levels-atmosphere-v0.zarr"
PL_VARS = ["t", "u", "v", "z"]
LEVELS = [1000, 925, 850, 700]          # = profile_inv.COMMON_LEVELS (et ceux de l'IFS open data)
SFC_VARS = ["t2m", "d2m", "u10", "v10", "msl", "sp", "tcc", "lcc", "strd", "sd", "blh"]
# ============================================================================ #


def _area():
    """[N, W, S, E] : emprise d'ecmwf_fetch arrondie au 0.25 deg exterieur."""
    import ecmwf_fetch as EF
    n, w, s, e = EF._bbox_latlon()
    return [np.ceil(n * 4) / 4, np.floor(w * 4) / 4, np.floor(s * 4) / 4, np.ceil(e * 4) / 4]


def _token():
    tok = os.environ.get("EDH_TOKEN")
    p = os.path.join(os.path.expanduser("~"), ".edh_token")
    if not tok and os.path.exists(p):
        tok = open(p, encoding="utf-8").read().strip()
    if not tok:
        raise SystemExit(f"[edh] jeton absent : colle ton jeton dans {p} "
                         "(earthdatahub.destine.eu/account-settings -> My personal access tokens)")
    return tok


def _open(name):
    import xarray as xr
    url = BASE.format(tok=_token()) + name
    return xr.open_dataset(url, engine="zarr", chunks={})


def _lev_dim(ds):
    for k in ("level", "pressure_level", "isobaricInhPa", "plev"):
        if k in ds.dims:
            return k
    raise KeyError(f"dimension de niveau introuvable : {list(ds.dims)}")


def _time_dim(ds):
    for k in ("valid_time", "time"):
        if k in ds.dims:
            return k
    raise KeyError(f"dimension temps introuvable : {list(ds.dims)}")


def _crop(ds):
    """Decoupe a l'emprise de era5_fetch (N, W, S, E), quel que soit le sens des lat / lon."""
    n, w, s, e = _area()
    lat = ds["latitude"].values
    ds = ds.sel(latitude=slice(n, s) if lat[0] > lat[-1] else slice(s, n))
    lon = ds["longitude"].values
    if lon.max() > 180 and w < 0:                     # domaine a cheval sur Greenwich, grille 0..360
        ds = ds.assign_coords(longitude=((ds.longitude + 180) % 360) - 180).sortby("longitude")
    return ds.sel(longitude=slice(w, e))


def _to_cds_names(ds):
    """Memes noms que les fichiers CDS : valid_time, pressure_level."""
    td = _time_dim(ds)
    if td != "valid_time":
        ds = ds.rename({td: "valid_time"})
    for k in ("level", "isobaricInhPa", "plev"):
        if k in ds.dims:
            ds = ds.rename({k: "pressure_level"})
    return ds


def _load(ds, label):
    from dask.diagnostics import ProgressBar
    t0 = time.time()
    print(f"[edh] {label} : telechargement...", flush=True)
    with ProgressBar(dt=5):
        ds = ds.load()
    print(f"[edh] {label} : OK en {(time.time() - t0) / 60:.1f} min "
          f"({ds.nbytes / 1e6:.0f} Mo en memoire)", flush=True)
    return ds


def winter(y, pl, sfc, overwrite=False):
    """Hiver oct y -> mars y+1 : 2 fichiers."""
    os.makedirs(ERA5_DIR, exist_ok=True)
    tag = f"{y}-{str(y + 1)[2:]}"
    t0, t1 = f"{y}-10-01", f"{y + 1}-03-31T23:00"
    for kind, src, vars_ in (("pl", pl, PL_VARS), ("sfc", sfc, SFC_VARS)):
        out = os.path.join(ERA5_DIR, f"era5_{kind}_{tag}.nc")
        if os.path.exists(out) and not overwrite:
            print(f"[edh] {os.path.basename(out)} deja present"); continue
        td = _time_dim(src)
        tmax = src[td].values[-1]
        if np.datetime64(t0) > tmax:
            print(f"[edh] hiver {tag} pas encore dans l'archive EDH (fin {str(tmax)[:10]})"); continue
        ds = src[vars_].sel({td: slice(t0, t1)})
        if kind == "pl":
            ds = ds.sel({_lev_dim(ds): LEVELS})
        ds = _to_cds_names(_load(_crop(ds), f"{kind} {tag}"))
        tmp = out + ".part"
        ds.to_netcdf(tmp)
        os.replace(tmp, out)
        print(f"[edh] -> {out}")


def orog(sfc):
    out = os.path.join(ERA5_DIR, "era5_orog.nc")
    if os.path.exists(out):
        return
    import xarray as xr
    td = _time_dim(sfc)
    d = _crop(sfc[["z", "lsm"]].isel({td: 0})).load()
    d = d.drop_vars([c for c in d.coords if c not in ("latitude", "longitude")])
    os.makedirs(ERA5_DIR, exist_ok=True)
    xr.Dataset({"zs": d["z"] / 9.80665, "lsm": d["lsm"]}).to_netcdf(out)
    print(f"[edh] relief ERA5 -> {out}")


def open_all(kind=None):
    """Ouvre tous les hivers telecharges : Dataset (time, level, latitude, longitude)
    avec pl + sfc + relief (zs, lsm)."""
    import xarray as xr, glob
    def _clean(d):
        return d.drop_vars([c for c in ("expver", "number") if c in d.variables])
    def _one(k):
        fs = sorted(glob.glob(os.path.join(ERA5_DIR, f"era5_{k}_*.nc")))
        if not fs:
            raise SystemExit(f"Aucun fichier era5_{k}_*.nc dans {ERA5_DIR} : lance python era5_edh.py")
        d = xr.open_mfdataset(fs, combine="by_coords", preprocess=_clean)
        return d.rename({k2: v for k2, v in (("valid_time", "time"), ("pressure_level", "level")) if k2 in d.dims})
    if kind:
        return _one(kind)
    ds = xr.merge([_one("pl"), _one("sfc")], compat="override")
    orog = os.path.join(ERA5_DIR, "era5_orog.nc")
    if os.path.exists(orog):
        ds = ds.merge(xr.open_dataset(orog), compat="override")
    return ds


def test():
    pl, sfc = _open(ZARR_PL), _open(ZARR_SFC)
    print("[edh] jeton OK, catalogues ouverts")
    print("  pl  dims :", dict(pl.sizes), "| niveaux :", list(pl[_lev_dim(pl)].values)[:12])
    print("  sfc dims :", dict(sfc.sizes))
    td = _time_dim(sfc)
    print("  couverture sfc :", str(sfc[td].values[0])[:10], "->", str(sfc[td].values[-1])[:10])
    lat, lon = 45.78, 3.08
    lon_q = lon % 360 if float(sfc.longitude.max()) > 180 else lon
    t = sfc["t2m"].sel({td: "2025-01-15T06:00"}).sel(latitude=lat, longitude=lon_q, method="nearest")
    print(f"[edh] TEST OK : T2m Clermont 15/01/2025 06 UTC = {float(t.load()) - 273.15:.1f} C")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="y0", type=int, default=2020, help="1er hiver (oct de cette annee)")
    ap.add_argument("--to", dest="y1", type=int, default=2026, help="annee de fin du dernier hiver")
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    if a.test:
        test(); sys.exit(0)
    pl, sfc = _open(ZARR_PL), _open(ZARR_SFC)
    orog(sfc)
    for y in range(a.y0, a.y1):
        winter(y, pl, sfc)
    print("[edh] termine. Verification : python profile_inv.py --era5")
