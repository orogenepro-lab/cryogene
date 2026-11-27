"""
train_inversion.py - Probabilite d'INVERSION OBSERVEE (CUVETTE), calee sur 6 hivers.

Pour chaque couple vallee / altitude, la cible est simple et verifiable :
    inversion = T_obs(vallee) < T_obs(altitude)   (la vallee est plus froide que le sommet)
Les predicteurs sont ceux du profil modele au point de la station de vallee (ERA5 pour
l'entrainement, IFS en prevision) : force d'inversion du profil, stabilite, vent, nebulosite
totale ET basse, pression, temperature a 850 hPa, heure, saison.
Modele : HistGradientBoostingClassifier, evalue en laissant chaque hiver de cote tour a tour.

Usage :
    python train_inversion.py        # -> C:\\cuvette_archive\\models\\cuvette_inversion.joblib
Affiche par couple : frequence climatologique, AUC, score de Brier (vs climatologie),
fiabilite (probabilite annoncee vs frequence observee).
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd())

DATASET = os.environ.get("CUVETTE_DATASET", r"C:\cuvette_archive\dataset_hiver.parquet")
MODEL_DIR = os.environ.get("CUVETTE_MODELS", r"C:\cuvette_archive\models")
PAIRS = [  # (nom affiche, station de vallee, station d'altitude)
    ("Limagne", "CLERMONT-FD", "PLATEAU CACADOGNE"),
    ("Bassin d'Aurillac", "AURILLAC", "PRAT DE BOUC"),
    ("Val d'Allier (Issoire)", "ISSOIRE", "ANZAT-LE-LUGUET_SAPC"),
]
FEATURES = ["inv_strength", "dtheta_850", "lapse_free", "ws925", "ws850", "ws10", "tcc", "lcc",
            "msl", "t850", "t2m", "hour_sin", "hour_cos", "doy_sin", "doy_cos"]
HGB = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=100,
           l2_regularization=1.0, random_state=0)


def pair_table(df, low, high):
    lo = df[df.NOM_USUEL == low]
    hi = df[df.NOM_USUEL == high][["time", "T_obs"]].rename(columns={"T_obs": "T_high"})
    if lo.empty or hi.empty:
        return None
    t = lo.merge(hi, on="time", how="inner")
    t["dT"] = t.T_obs - t.T_high
    t["y"] = (t.dT < 0).astype(int)
    return t


def reliability(p, y, bins=(0, .1, .3, .5, .7, .9, 1.0001)):
    c = pd.cut(p, bins, right=False)
    g = pd.DataFrame({"p": p, "y": y, "c": c}).groupby("c", observed=True)
    return g.agg(n=("y", "size"), annonce=("p", "mean"), observe=("y", "mean")).round(2)


def main():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score, brier_score_loss
    import joblib
    df = pd.read_parquet(DATASET)
    out = {"pairs": [], "features": FEATURES, "hgb": HGB}
    for name, low, high in PAIRS:
        t = pair_table(df, low, high)
        if t is None or len(t) < 2000:
            print(f"\n[inv] {name} : stations introuvables ou trop peu de donnees -> ignore"); continue
        feats = [f for f in FEATURES if f in t.columns]
        X, y = t[feats].to_numpy(float), t.y.to_numpy()
        p = np.full(len(t), np.nan)
        for w in sorted(t.winter.unique()):
            tr, te = (t.winter != w).to_numpy(), (t.winter == w).to_numpy()
            m = HistGradientBoostingClassifier(**HGB).fit(X[tr], y[tr])
            p[te] = m.predict_proba(X[te])[:, 1]
        clim = y.mean()
        bs, bs_clim = brier_score_loss(y, p), brier_score_loss(y, np.full(len(y), clim))
        print(f"\n================ {name} : {low.title()} / {high.title()} ================")
        print(f"[inv] {len(t):,} echeances, inversion observee {100 * clim:.0f} % du temps")
        print(f"[inv] AUC {roc_auc_score(y, p):.3f} | Brier {bs:.3f} (climatologie {bs_clim:.3f}, "
              f"gain {100 * (1 - bs / bs_clim):.0f} %)")
        rule = (t.inv_strength > 0).astype(int)
        print(f"[inv] comparaison regle simple 'inversion dans le profil modele' : "
              f"bonne reponse {100 * (rule == y).mean():.0f} % ; classifieur (p>0.5) {100 * ((p > .5) == y).mean():.0f} %")
        print("Fiabilite (probabilite annoncee vs frequence observee) :")
        print(reliability(p, y).to_string())
        h6 = t.time.dt.hour == 6
        print(f"[inv] a 06 UTC : inversion {100 * y[h6].mean():.0f} % du temps, AUC {roc_auc_score(y[h6], p[h6]):.3f}")
        m = HistGradientBoostingClassifier(**HGB).fit(X, y)
        out["pairs"].append(dict(name=name, low=low, high=high, model=m, features=feats, clim=float(clim)))
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, "cuvette_inversion.joblib")
    joblib.dump(out, path)
    print(f"\n[inv] -> {path} ({len(out['pairs'])} couples)")


if __name__ == "__main__":
    main()
