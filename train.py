"""
train.py - Apprentissage du residu T_obs - T_base (CUVETTE) avec scikit-learn
(HistGradientBoostingRegressor) et validation HONNETE :

  1. GroupKFold par STATION (5 plis) : le modele predit des stations qu'il n'a jamais vues
     -> c'est la vraie mesure de sa capacite a generaliser au relief (carte 100 m).
  2. Hiver le plus recent tenu a l'ecart (entrainement sur les hivers precedents)
     -> mesure la robustesse d'une annee sur l'autre.

Comparaisons affichees (RMSE / biais en C, par classe d'altitude et par situation) :
  ERA5 brut   : T2m ERA5 interpolee, sans correction d'altitude
  T_base      : profil ERA5 ramene a l'altitude de la station (profile_inv)
  CUVETTE     : T_base + residu predit par le ML

Usage :
    python train.py                    # CV + modele final -> C:\\cuvette_archive\\models\\cuvette_hgb.joblib
    python train.py --quick            # 1 hiver sur 3 lignes pour tester vite
Necessite : pip install scikit-learn joblib
"""
import os, sys, argparse, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd())

DATASET = os.environ.get("CUVETTE_DATASET", r"C:\cuvette_archive\dataset_hiver.parquet")
MODEL_DIR = os.environ.get("CUVETTE_MODELS", r"C:\cuvette_archive\models")

# Predicteurs = UNIQUEMENT ce qui sera disponible en prevision (IFS + relief + calendrier).
# Exclus : T_obs, ff_obs, nebul_obs (observations !), resid, identifiants, winter.
FEATURES = [
    # profil modele (profile_inv)
    "T_base", "t2m", "t850", "inv_strength", "inv_top", "dtheta_850", "lapse_free",
    "ws925", "ws850", "ws10", "tcc", "lcc", "zs", "dz_model",
    # surface ERA5 (absents de l'IFS open data pour certains : voir FEATURES_IFS)
    "lw_down", "snow_we", "blh", "td2m", "msl",
    # relief
    "ter_alt", "ter_slope", "ter_curv", "ter_tpi_500", "ter_tpi_2k", "ter_tpi_10k", "ter_tpi_20k",
    "ter_depth_2k", "ter_depth_10k", "ter_svf", "ter_cold_acc", "ter_cold_acc_s", "ter_dist_ridge",
    # calendrier
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",
]
# Variables presentes dans l'IFS open data (ecmwf_fetch) : on entraine AUSSI un modele
# restreint a celles-ci, c'est lui qui servira en prevision tant qu'on n'a pas les autres.
NOT_IN_IFS_OPENDATA = {"lw_down", "snow_we", "blh", "td2m", "lcc"}
HGB = dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=200,
           l2_regularization=1.0, early_stopping=False, random_state=0)


def rmse(e):
    e = np.asarray(e, float)
    return float(np.sqrt(np.nanmean(e ** 2)))


def scores(df, pred_col):
    """Tableau RMSE / biais par classe d'altitude pour 3 estimateurs."""
    rows = []
    groups = [("TOUT", df)] + [(k, g) for k, g in df.groupby("classe", observed=True)]
    m6 = df[df.time.dt.hour == 6]
    groups += [("06 UTC inversion > 3 K", m6[m6.inv_strength > 3]),
               ("06 UTC sans inversion", m6[m6.inv_strength <= 0]),
               ("06 UTC inv. + fonds TPI2k<-30", m6[(m6.inv_strength > 3) & (m6.ter_tpi_2k < -30)])]
    for name, g in groups:
        if not len(g):
            continue
        rows.append(dict(groupe=name, n=len(g),
                         rmse_era5=rmse(g.t2m - g.T_obs), rmse_base=rmse(g.T_base - g.T_obs),
                         rmse_cuvette=rmse(g[pred_col] - g.T_obs),
                         biais_base=float((g.T_base - g.T_obs).mean()),
                         biais_cuvette=float((g[pred_col] - g.T_obs).mean())))
    return pd.DataFrame(rows).round(2)


def cv_station(df, feats, k=5):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold
    pred = np.full(len(df), np.nan)
    X, y, g = df[feats].values, df.resid.values, df.NUM_POSTE.values
    for i, (tr, te) in enumerate(GroupKFold(k).split(X, y, g), 1):
        t0 = time.time()
        m = HistGradientBoostingRegressor(**HGB).fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
        print(f"[cv] pli {i}/{k} : {len(set(g[te]))} stations tenues a l'ecart, "
              f"RMSE residu {rmse(pred[te] - y[te]):.2f} C ({time.time() - t0:.0f} s)", flush=True)
    return pred


def cv_winter(df, feats):
    from sklearn.ensemble import HistGradientBoostingRegressor
    last = df.winter.max()
    tr, te = df.winter < last, df.winter == last
    m = HistGradientBoostingRegressor(**HGB).fit(df.loc[tr, feats].values, df.loc[tr, "resid"].values)
    p = m.predict(df.loc[te, feats].values)
    d = df.loc[te].copy()
    d["T_cuv"] = d.T_base + p
    return last, d


def importance(df, feats, n=40000):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.inspection import permutation_importance
    s = df.sample(min(n * 3, len(df)), random_state=1)
    tr, te = s.iloc[: len(s) * 2 // 3], s.iloc[len(s) * 2 // 3:]
    m = HistGradientBoostingRegressor(**HGB).fit(tr[feats].values, tr.resid.values)
    r = permutation_importance(m, te[feats].values, te.resid.values, n_repeats=3,
                               random_state=0, scoring="neg_root_mean_squared_error")
    return pd.Series(r.importances_mean, index=feats).sort_values(ascending=False)


def fit_final(df, feats, name):
    import joblib
    from sklearn.ensemble import HistGradientBoostingRegressor
    m = HistGradientBoostingRegressor(**HGB).fit(df[feats].values, df.resid.values)
    os.makedirs(MODEL_DIR, exist_ok=True)
    out = os.path.join(MODEL_DIR, f"{name}.joblib")
    joblib.dump({"model": m, "features": feats, "hgb": HGB,
                 "trained_on": f"{df.time.min():%Y-%m-%d} -> {df.time.max():%Y-%m-%d}",
                 "n_stations": int(df.NUM_POSTE.nunique())}, out)
    print(f"[train] modele final -> {out}")


def main(quick=False):
    df = pd.read_parquet(DATASET) if DATASET.endswith(".parquet") else pd.read_csv(DATASET, parse_dates=["time"])
    df = df.dropna(subset=["resid", "T_base", "t2m"])
    if quick:
        df = df[df.winter == df.winter.max()].iloc[::3]
    df = df.reset_index(drop=True)
    df["classe"] = pd.cut(df.ALTI, [0, 400, 700, 1000, 2000], labels=["<400 m", "400-700", "700-1000", ">1000 m"])
    feats = [f for f in FEATURES if f in df.columns]
    feats_ifs = [f for f in feats if f not in NOT_IN_IFS_OPENDATA]
    print(f"[train] {len(df):,} lignes, {df.NUM_POSTE.nunique()} stations, {len(feats)} predicteurs")

    for label, fs in (("COMPLET (ERA5)", feats), ("IFS OPEN DATA", feats_ifs)):
        print(f"\n================ Modele {label} : {len(fs)} predicteurs ================")
        df["T_cuv"] = df.T_base + cv_station(df, fs)
        print("\nValidation par STATION (stations jamais vues) - RMSE / biais en C :")
        print(scores(df, "T_cuv").to_string(index=False))
        if label.startswith("COMPLET"):
            per = df.groupby(["NOM_USUEL", "ALTI"]).apply(
                lambda g: pd.Series({"rmse_base": rmse(g.T_base - g.T_obs), "rmse_cuvette": rmse(g.T_cuv - g.T_obs)}),
                include_groups=False).round(2)
            per["gain"] = (per.rmse_base - per.rmse_cuvette).round(2)
            print("\nStations les plus ameliorees / degradees :")
            print(pd.concat([per.sort_values("gain").tail(8), per.sort_values("gain").head(5)]).to_string())
            df.to_parquet(os.path.join(os.path.dirname(DATASET), "cv_predictions.parquet"), index=False)

    if df.winter.nunique() > 1:
        last, d = cv_winter(df, feats)
        print(f"\nValidation TEMPORELLE (entraine sans l'hiver {last}-{last + 1 - 2000}, teste dessus) :")
        print(scores(d, "T_cuv").to_string(index=False))

    print("\nImportance des predicteurs (permutation, gain de RMSE en C) :")
    print(importance(df, feats).head(15).round(3).to_string())

    if not quick:
        fit_final(df, feats, "cuvette_hgb_complet")
        fit_final(df, feats_ifs, "cuvette_hgb_ifs")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
