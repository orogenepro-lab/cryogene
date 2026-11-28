"""
export_for_github.py - Prepare les donnees dont la prevision a besoin pour tourner sur GitHub :
    modeles entraines, table des stations, contours des departements, relief (version compacte).
A lancer sur le PC, dans le dossier des scripts (la ou se trouve terrain_features.npz) :

    python export_for_github.py                    # -> dossier cryogene_data\\
    python export_for_github.py --res 300                 # plus leger si besoin

Le relief est sous-echantillonne a ~200 m (--res) et enregistre en float16 pour rester sous les
limites GitHub (25 Mo par l'interface web, 100 Mo par git). La carte est calculee a 200-400 m et
les stations gardent leurs descripteurs pleine resolution (stations_cuvette.csv) : rien n'est perdu
pour la prevision.
"""
import os, sys, shutil
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import terrain_features as TF

ROOT = os.environ.get("CUVETTE_ROOT", r"C:\cuvette_archive")
MODELS = ["cuvette_hgb_ifs.joblib", "cuvette_inversion.joblib"]


def main(dest, res_target=200.0):
    os.makedirs(os.path.join(dest, "models"), exist_ok=True)
    for m in MODELS:
        src = os.path.join(ROOT, "models", m)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dest, "models", m))
            print(f"[export] modele      {m:32s} {os.path.getsize(src) / 1e6:6.1f} Mo")
        else:
            print(f"[export] ATTENTION : {src} introuvable (lance train.py / train_inversion.py)")
    for f in ("stations_cuvette.csv", "departements_l93.json"):
        src = os.path.join(ROOT, f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dest, f))
            print(f"[export] donnees     {f:32s} {os.path.getsize(src) / 1e6:6.2f} Mo")
        else:
            print(f"[export] (absent, sera recree sur GitHub) {f}")
    F, meta = TF.load()
    res = float(meta["res"])
    k = max(1, int(round(res_target / res)))
    keep = ["alt"] + [n for n in TF.FEATURE_NAMES if n != "alt"]
    a, b, c, d, e, f = [float(x) for x in meta["transform"][:6]]
    sub = {n: F[n][::k, ::k].astype(np.float16) for n in keep}
    meta2 = dict(res=np.array(res * k), transform=np.array([a * k, b, c, d, e * k, f]),
                 shape=np.array(sub["alt"].shape))
    out = os.path.join(dest, "terrain_features.npz")
    np.savez_compressed(out, **sub, **{f"meta_{kk}": v for kk, v in meta2.items()})
    print(f"[export] relief : {F['alt'].shape[1]}x{F['alt'].shape[0]} px a {res:.0f} m -> "
          f"{sub['alt'].shape[1]}x{sub['alt'].shape[0]} px a {res * k:.0f} m")
    size = os.path.getsize(out) / 1e6
    print(f"[export] relief      terrain_features.npz             {size:6.1f} Mo")
    if size > 95:
        print("[export] ATTENTION : > 95 Mo, GitHub refusera le fichier : relance avec --res 300.")
    elif size > 25:
        print("[export] > 25 Mo : l'interface web de GitHub le refusera, passe par git (ou --res 300).")
    print(f"\n[export] termine -> {os.path.abspath(dest)}")
    print("Copie ce contenu dans data/ du depot GitHub (ou televerse-le sur github.com).")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("dest", nargs="?", default="cryogene_data")
    ap.add_argument("--res", type=float, default=200.0, help="resolution du relief exporte (m)")
    a = ap.parse_args()
    main(a.dest, a.res)
