"""
export_for_github.py - Prepare les donnees dont la prevision a besoin pour tourner sur GitHub :
    modeles entraines, table des stations, contours des departements, relief (version compacte).
A lancer sur le PC, dans le dossier des scripts (la ou se trouve terrain_features.npz) :

    python export_for_github.py                    # -> dossier cryogene_data\\
    python export_for_github.py D:\\git\\cryogene\\data

Le relief est enregistre en float16 (precision ~1 m sur l'altitude, largement suffisante)
pour rester sous la limite GitHub de 100 Mo par fichier.
"""
import os, sys, shutil
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import terrain_features as TF

ROOT = os.environ.get("CUVETTE_ROOT", r"C:\cuvette_archive")
MODELS = ["cuvette_hgb_ifs.joblib", "cuvette_inversion.joblib"]


def main(dest):
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
    keep = ["alt"] + [n for n in TF.FEATURE_NAMES if n != "alt"]
    out = os.path.join(dest, "terrain_features.npz")
    np.savez_compressed(out, **{k: F[k].astype(np.float16) for k in keep},
                        **{f"meta_{k}": v for k, v in meta.items()})
    size = os.path.getsize(out) / 1e6
    print(f"[export] relief      terrain_features.npz             {size:6.1f} Mo")
    if size > 95:
        print("[export] ATTENTION : > 95 Mo, GitHub refusera le fichier. Dis-le a Claude.")
    print(f"\n[export] termine -> {os.path.abspath(dest)}")
    print("Copie ce contenu dans data/ du depot GitHub (ou televerse-le sur github.com).")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "cryogene_data")
