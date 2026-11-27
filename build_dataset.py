"""
build_dataset.py - Table d'apprentissage CUVETTE : une ligne = (station, heure).

Pour chaque station Meteo-France et chaque echeance ERA5 (pas de 3 h, hiver oct -> mars) :
  - le profil ERA5 interpole (bilineaire) a la position de la station -> diagnostics
    d'inversion (profile_inv.diagnose) + variables de surface (LW descendant, neige, BLH...)
  - T_base : temperature du profil modele a l'altitude REELLE de la station
    (profile_inv.t_at_height) ; c'est ce que "dirait" ERA5 sans connaitre le relief fin
  - les descripteurs de terrain a la station (terrain_features : TPI multi-echelle, SVF...)
  - la cible : resid = T_obs - T_base  (C). Le ML apprendra ce residu : c'est le froid
    (ou le chaud) que le relief local ajoute a ce que voit le modele.

Usage :
    python build_dataset.py                 # -> C:\\cuvette_archive\\dataset_hiver.parquet
    python build_dataset.py --hours all     # toutes les heures au lieu du pas de 3 h
Sortie : un resume (nb lignes, stations, biais / ecart-type de T_base par classe d'altitude)
pour verifier que tout est coherent avant l'entrainement.
"""
import os, sys, argparse
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd())
import profile_inv as P
import terrain_features as TF
import obs_mf as OBS
import era5_edh as E

OUT = os.environ.get("CUVETTE_DATASET", r"C:\cuvette_archive\dataset_hiver.parquet")
WINTER = (10, 11, 12, 1, 2, 3)
SFC_EXTRA = {"strd": "lw_down", "sd": "snow_we", "blh": "blh", "d2m": "td2m", "msl": "msl"}


def era5_at_stations(st, hours="3h"):
    """Dataset ERA5 interpole aux stations : dims (time, station)."""
    ds = E.open_all()
    ds = ds.sel(time=ds.time.dt.month.isin(WINTER))
    if hours == "3h":
        ds = ds.sel(time=ds.time.dt.hour % 3 == 0)
    ds = ds.sortby("latitude").sortby("longitude")
    lat = xr.DataArray(st.LAT.values, dims="station", coords={"station": st.NUM_POSTE.values})
    lon = xr.DataArray(st.LON.values, dims="station", coords={"station": st.NUM_POSTE.values})
    print(f"[ds] interpolation ERA5 -> {len(st)} stations, {ds.sizes['time']} echeances ...", flush=True)
    return ds.interp(latitude=lat, longitude=lon).load()


def features(dsi, st):
    """Diagnostics + T_base + variables de surface, aux stations."""
    std = P.standardize(dsi)
    d = P.diagnose(std)
    alt = xr.DataArray(st.ALTI.values.astype(float), dims="station", coords={"station": st.NUM_POSTE.values})
    d["T_base"] = P.t_at_height(std, alt) - 273.15
    d["dz_model"] = alt - d["zs"]                      # station sous (<0) / au-dessus du relief ERA5
    for k, new in SFC_EXTRA.items():
        if k in dsi:
            v = dsi[k]
            if k == "strd":
                v = v / 3600.0                             # J/m2 sur 1 h -> W/m2
            if k in ("d2m",):
                v = v - 273.15
            if k == "msl":
                v = v / 100.0
            d[new] = v
    d = d.drop_vars([c for c in d.coords if c not in ("time", "station")])
    df = d.to_dataframe().reset_index().rename(columns={"station": "NUM_POSTE"})
    return df


def terrain_at_stations(st):
    F, meta = TF.load()
    rows, cols = st.row.values, st.col.values
    X, names = TF.feature_matrix(F, rows, cols)
    t = pd.DataFrame(X, columns=[f"ter_{n}" for n in names])
    t.insert(0, "NUM_POSTE", st.NUM_POSTE.values)
    t["ter_dem_minus_alti"] = t["ter_alt"] - st.ALTI.values   # controle position (doit etre ~0)
    return t, meta


def build(hours="3h"):
    obs = OBS.load()
    obs = obs[obs.time.dt.month.isin(WINTER)]
    F, meta = TF.load()
    st = OBS.stations(obs, terrain_meta=meta)
    print(f"[ds] {len(st)} stations dans l'emprise du MNT")
    ter, _ = terrain_at_stations(st)
    bad = ter[ter.ter_dem_minus_alti.abs() > 150]
    if len(bad):
        print(f"[ds] attention : {len(bad)} stations dont l'altitude MNT differe de > 150 m "
              f"(coordonnees douteuses ?) :\n{bad[['NUM_POSTE', 'ter_dem_minus_alti']].to_string(index=False)}")
    dsi = era5_at_stations(st, hours)
    feat = features(dsi, st)
    feat["time"] = pd.to_datetime(feat["time"]).dt.tz_localize("UTC")
    o = obs[["NUM_POSTE", "time", "T", "FF", "N"]].rename(columns={"T": "T_obs", "FF": "ff_obs", "N": "nebul_obs"})
    df = feat.merge(o, on=["NUM_POSTE", "time"], how="inner").merge(ter, on="NUM_POSTE", how="left")
    df = df.merge(st[["NUM_POSTE", "NOM_USUEL", "ALTI"]], on="NUM_POSTE", how="left")
    df = df.dropna(subset=["T_obs", "T_base"])
    df["resid"] = df["T_obs"] - df["T_base"]
    # temps : heure UTC (cyclique), saison (cyclique)
    h = df.time.dt.hour + df.time.dt.minute / 60
    doy = df.time.dt.dayofyear
    df["hour_sin"], df["hour_cos"] = np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)
    df["doy_sin"], df["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    df["winter"] = np.where(df.time.dt.month >= 10, df.time.dt.year, df.time.dt.year - 1)
    return df


def summary(df):
    print(f"\n[ds] {len(df):,} lignes | {df.NUM_POSTE.nunique()} stations | "
          f"hivers {sorted(df.winter.unique())} | {df.time.min():%Y-%m-%d} -> {df.time.max():%Y-%m-%d}")
    bins = [0, 400, 700, 1000, 2000]
    df = df.assign(classe=pd.cut(df.ALTI, bins, labels=["<400 m", "400-700", "700-1000", ">1000 m"]))
    g = df.groupby("classe", observed=True)["resid"].agg(["count", "mean", "std"]).round(2)
    print("\nResidu T_obs - T_base (C) par classe d'altitude (mean < 0 = le site est plus froid qu'ERA5) :")
    print(g.to_string())
    inv = df.inv_strength > 3
    print("\nResidu moyen selon la situation ERA5 (06 UTC) :")
    m = df[df.time.dt.hour == 6]
    for lab, sel in (("inversion ERA5 > 3 K", m.inv_strength > 3), ("pas d'inversion", m.inv_strength <= 0)):
        s = m[sel]
        if len(s):
            print(f"  {lab:22s} n={len(s):6d}  fonds (TPI10k<-50 m) {s[s.ter_tpi_10k < -50].resid.mean():+.2f}"
                  f"  | reliefs (TPI10k>+50 m) {s[s.ter_tpi_10k > 50].resid.mean():+.2f}")
    print("\nStations les plus froides par rapport a ERA5 (residu moyen a 06 UTC par inversion > 3 K) :")
    w = m[m.inv_strength > 3].groupby(["NOM_USUEL", "ALTI"]).resid.mean().sort_values().head(10).round(2)
    print(w.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", default="3h", choices=["3h", "all"])
    a = ap.parse_args()
    df = build(a.hours)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    try:
        df.to_parquet(OUT, index=False)
        print(f"[ds] -> {OUT}")
    except Exception as e:                                  # pas de pyarrow
        out = OUT.replace(".parquet", ".csv.gz")
        df.to_csv(out, index=False)
        print(f"[ds] parquet indisponible ({e}) -> {out}")
    summary(df)
