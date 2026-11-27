"""
profile_inv.py - Diagnostic d'inversion thermique a partir des profils verticaux
d'un modele global (ERA5 pour l'entrainement, IFS open data pour la prevision).
Le meme code sert aux deux : c'est ce qui garantit la coherence entrainement / prevision.

Principe : a chaque point de grille (~25 km) et chaque echeance on reconstruit le
profil T(z) : un noeud a 2 m au-dessus du relief du modele + les niveaux pression
situes au-dessus du sol (1000 -> 700 hPa). On en tire :

  inv_strength  (K)  max(T_niveau - T2m) dans les 1500 m au-dessus du sol ; > 0 = inversion
  inv_top       (m)  hauteur sol de ce maximum (sommet approximatif de l'inversion)
  dtheta_850    (K)  theta(850) - theta(2 m) : stabilite de la couche basse (> 10 K = tres stable)
  lapse_free    (K/km) gradient libre 850-700 hPa (6.5 = standard)
  ws925, ws850, ws10 (m/s)  vent : brasse ou non la couche froide
  tcc, lcc      (0-1) nebulosite : ciel clair = refroidissement radiatif
  zs            (m)  altitude du relief du modele (ERA5 lisse : ~700 m sur la Limagne !)

et la fonction cle pour la descente d'echelle :
  t_at_height(std, h)   temperature du profil modele a l'altitude h (m), extrapolee
                        sous le relief du modele avec le gradient de la couche basse.
  -> "T de base" pour une station ou un pixel du MNT ; le ML apprendra le residu.

Usage :
    python profile_inv.py --ifs        # dernier run IFS archive : serie J+0..J+7 Clermont / Aurillac
    python profile_inv.py --era5       # diagnostic 06 UTC sur toute l'archive ERA5 -> era5_diag_06.nc
"""
import os, sys, argparse
import numpy as np
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd())

G = 9.80665
KAPPA = 0.2857
INV_DEPTH = 1500.0          # couche sondee pour l'inversion (m au-dessus du sol modele)
SITES = {                   # points de controle (lat, lon, altitude reelle)
    "Clermont-Aulnat": (45.786, 3.149, 331),
    "Puy de Dome":     (45.772, 2.964, 1465),
    "Aurillac":        (44.897, 2.419, 639),
    "Ussel":           (45.534, 2.424, 700),
}


# --------------------------------------------------------------------------- #
#  Normalisation ERA5 / IFS -> noms communs
# --------------------------------------------------------------------------- #
COMMON_LEVELS = [1000, 925, 850, 700]   # niveaux presents a la fois dans ERA5 ET l'IFS open data


def standardize(ds, levels=COMMON_LEVELS):
    """Retourne un Dataset avec : T, Z (m), U, V sur 'level' (hPa, decroissant) ;
    t2m, ps (hPa), zs (m), u10, v10, tcc, lcc (si dispo). Les autres dims
    (time / step / number) sont conservees telles quelles.
    levels : par defaut on ne garde QUE les niveaux communs ERA5 / IFS open data, pour que
    les diagnostics d'entrainement (ERA5) et de prevision (IFS) soient calcules pareil.
    levels=None garde tout (ERA5 complet, pour etudier la structure fine des inversions)."""
    ren = {k: "level" for k in ("isobaricInhPa", "pressure_level") if k in ds.dims}
    ds = ds.rename(ren)
    if "valid_time" in ds.dims:
        ds = ds.rename(valid_time="time")
    if levels is not None:
        ds = ds.sel(level=[l for l in levels if l in ds["level"].values])
    ds = ds.sortby("level", ascending=False)                    # 1000 -> 700 : Z croissant
    out = xr.Dataset()
    out["T"] = ds["t"]
    out["Z"] = ds["gh"] if "gh" in ds else ds["z"] / G           # IFS : gh (m) ; ERA5 : z (m2/s2)
    out["U"], out["V"] = ds["u"], ds["v"]
    out["t2m"] = ds["t2m"]
    out["ps"] = ds["sp"] / 100.0
    out["u10"], out["v10"] = ds["u10"], ds["v10"]
    for k in ("tcc", "lcc"):
        if k in ds:
            v = ds[k]
            out[k] = v / 100.0 if float(v.max()) > 1.5 else v     # IFS open data en %
    out["zs"] = ds["zs"] if "zs" in ds else surface_height(out["Z"], out["ps"])
    return out


def surface_height(Z, ps):
    """Altitude du sol modele par interpolation de Z en ln(p) a p = ps (si pas d'orographie)."""
    lev = Z["level"].values
    lnp = np.log(lev)
    zs = xr.full_like(ps, np.nan, dtype="float64")
    lnps = np.log(ps)
    for k in range(len(lev) - 1):                               # couche [p_k, p_k+1]
        w = (lnps - lnp[k + 1]) / (lnp[k] - lnp[k + 1])
        zk, zk1 = Z.isel(level=k, drop=True), Z.isel(level=k + 1, drop=True)
        cand = zk1 + w * (zk - zk1)
        inside = (ps <= lev[k]) & (ps > lev[k + 1])
        if k == 0:
            inside = inside | (ps > lev[0])                      # extrapolation sous 1000 hPa
        zs = xr.where(inside & zs.isnull(), cand, zs)
    return zs


# --------------------------------------------------------------------------- #
#  Diagnostic
# --------------------------------------------------------------------------- #
def diagnose(std):
    T, Z, t2, zs, ps = std["T"], std["Z"], std["t2m"], std["zs"], std["ps"]
    agl = Z - zs
    above = (Z["level"] < ps) & (agl > 10)
    d = xr.Dataset()
    dT = (T - t2).where(above & (agl < INV_DEPTH))
    d["inv_strength"] = dT.max("level")
    d["inv_top"] = agl.where(dT == d["inv_strength"]).max("level")   # compatible dask (pas d'isel)
    theta = T * (1000.0 / T["level"]) ** KAPPA
    theta2 = t2 * (1000.0 / ps) ** KAPPA
    d["dtheta_850"] = (theta.sel(level=850) - theta2).where(above.sel(level=850))
    d["lapse_free"] = -1000 * (T.sel(level=700) - T.sel(level=850)) / (Z.sel(level=700) - Z.sel(level=850))
    ws = np.hypot(std["U"], std["V"])
    d["ws925"] = ws.sel(level=925).where(above.sel(level=925))
    d["ws850"] = ws.sel(level=850)
    d["ws10"] = np.hypot(std["u10"], std["v10"])
    for k in ("tcc", "lcc"):
        if k in std:
            d[k] = std[k]
    d["t2m"] = t2 - 273.15
    d["t850"] = T.sel(level=850) - 273.15
    d["zs"] = zs
    d["inv_flag"] = (d["inv_strength"] > 1.0).astype("int8")
    return d


def t_at_height(std, h, g_min=-10.0, g_max=20.0):
    """T (K) du profil modele a l'altitude h (m, scalaire ou DataArray diffusable).
    Au-dessus du sol modele : interpolation lineaire entre noeuds (2 m, niveaux).
    En dessous : extrapolation avec le gradient de la couche basse, borne a
    [g_min, g_max] K/km (en inversion le gradient est positif -> plus froid en bas)."""
    T, Z, t2, zs, ps = std["T"], std["Z"], std["t2m"], std["zs"], std["ps"]
    z0 = zs + 2.0
    above = (Z["level"] < ps) & (Z > z0 + 1)
    Zn = xr.where(above, Z, z0)                                # niveaux sous le sol -> noeud 2 m
    Tn = xr.where(above, T, t2)
    nlev = Z.sizes["level"]
    res = t2 * np.nan + 0 * h                                  # forme diffusee (t2 x h), rempli de NaN
    # couche 2 m -> premier niveau au-dessus du sol
    Tfirst = t2 * np.nan; Zfirst = t2 * np.nan                 # premier niveau au-dessus du sol
    for k in reversed(range(nlev)):                            # du haut vers le bas : le dernier ecrit = le plus bas
        ab = above.isel(level=k, drop=True)
        Tfirst = xr.where(ab, Tn.isel(level=k, drop=True), Tfirst)
        Zfirst = xr.where(ab, Zn.isel(level=k, drop=True), Zfirst)
    g0 = ((Tfirst - t2) / (Zfirst - z0) * 1000).clip(g_min, g_max)
    below = h < z0
    res = xr.where(below, t2 + g0 / 1000 * (h - z0), res)
    za, ta = z0, t2
    for k in range(nlev):
        zb, tb = Zn.isel(level=k, drop=True), Tn.isel(level=k, drop=True)
        seg = (h >= za) & (h < zb) & (zb > za)
        res = xr.where(seg & res.isnull(), ta + (h - za) / (zb - za).where(zb > za) * (tb - ta), res)
        za, ta = xr.where(zb > za, zb, za), xr.where(zb > za, tb, ta)
    # au-dessus du dernier niveau : gradient libre
    top = (h >= za) & res.isnull()
    glast = (Tn.isel(level=-1, drop=True) - Tn.isel(level=-2, drop=True)) / \
            (Zn.isel(level=-1, drop=True) - Zn.isel(level=-2, drop=True))
    res = xr.where(top, ta + glast * (h - za), res)
    return res


def at_site(da, lat, lon):
    """Valeur au point de grille le plus proche (pour les series de controle)."""
    return da.sel(latitude=lat, longitude=lon, method="nearest")


# --------------------------------------------------------------------------- #
#  Lancements
# --------------------------------------------------------------------------- #
def run_ifs():
    import ecmwf_fetch as EF
    runs = EF.list_archive()
    if not runs:
        raise SystemExit("Aucun run IFS archive : lance d'abord python ecmwf_fetch.py")
    rdir = os.path.join(EF.ARCHIVE, runs[-1])
    ds = xr.merge([EF.open_grib(os.path.join(rdir, "hres_sfc.grib2")),
                   EF.open_grib(os.path.join(rdir, "hres_pl.grib2"))], compat="override")
    std = standardize(ds)
    d = diagnose(std)
    print(f"\n=== IFS run {runs[-1]} ===")
    lat, lon, h = SITES["Clermont-Aulnat"]
    lat2, lon2, h2 = SITES["Puy de Dome"]
    Tv = at_site(t_at_height(std, h), lat, lon) - 273.15
    Ts = at_site(t_at_height(std, h2), lat2, lon2) - 273.15
    dc = at_site(d, lat, lon)
    vt = ds["valid_time"].values if "valid_time" in ds else ds["step"].values
    print(f"Point Clermont : relief modele {float(dc.zs):.0f} m (reel {h} m)")
    print(" echeance           inv(K) top(m) dth850 ws850  tcc | T Clermont  T PdD  ecart vallee-sommet")
    for i in range(0, dc.sizes["step"], 2):
        r = dc.isel(step=i)
        tcc = float(r["tcc"]) if "tcc" in r else np.nan
        print(f" {str(vt[i])[:16]}  {float(r.inv_strength):6.1f} {float(r.inv_top):6.0f} "
              f"{float(r.dtheta_850):6.1f} {float(r.ws850):5.1f} {tcc:4.2f} | "
              f"{float(Tv.isel(step=i)):8.1f} {float(Ts.isel(step=i)):6.1f} {float(Tv.isel(step=i) - Ts.isel(step=i)):+8.1f}")
    print("ecart vallee-sommet < 0 = inversion Clermont / Puy de Dome (le sommet est plus chaud)")


def run_era5(hour=6):
    import era5_edh as E
    ds = E.open_all()
    ds = ds.sel(time=ds.time.dt.hour == hour).load()        # petit : on charge en memoire
    std = standardize(ds)
    d = diagnose(std)
    out = os.path.join(E.ERA5_DIR, f"era5_diag_{hour:02d}.nc")
    d.to_netcdf(out)
    print(f"-> {out}")
    lat, lon, _ = SITES["Clermont-Aulnat"]
    s = at_site(d, lat, lon).to_dataframe()
    s["winter"] = np.where(s.index.month >= 10, s.index.year, s.index.year - 1)
    tab = s.groupby("winter").agg(jours=("inv_flag", "size"), inversions=("inv_flag", "sum"),
                                  inv_moy=("inv_strength", "mean"), inv_max=("inv_strength", "max"))
    print(f"\nInversions a {hour:02d} UTC au point ERA5 de Clermont, par hiver (oct-mars) :")
    print(tab.round(1).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ifs", action="store_true")
    ap.add_argument("--era5", action="store_true")
    a = ap.parse_args()
    if a.era5:
        run_era5()
    else:
        run_ifs()
