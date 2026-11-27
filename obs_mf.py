"""
obs_mf.py - Observations horaires Meteo-France (open data data.gouv.fr) pour la
verite terrain de CUVETTE : T, TD, FF, DD, neige au sol, par station des
departements 63 / 15 / 19. Telecharge les CSV compresses (par departement, par
periode), les concatene, et produit un tableau station x heure.

    pip install pandas requests pyproj

Usage :
    python obs_mf.py                 # telecharge (ou met a jour) les fichiers 'latest' des 3 dpts
    import obs_mf as OBS
    st  = OBS.stations()             # DataFrame : NUM_POSTE, NOM, LAT, LON, ALTI, X, Y (L93), row, col
    obs = OBS.load(years=(2022, 2026))   # DataFrame long : NUM_POSTE, time(UTC), T, TD, FF, DD, HNEIGEF...
Jeux data.gouv.fr :
    principal      6569b4473bedf2e7abad3b72  (stations synoptiques + RADOME)
    complementaire 6791055a6d553b689a5e2ba4  (stations complementaires)
Fichiers nommes H_<dep>_previous-....csv.gz (archive) et H_<dep>_latest-....csv.gz
(2 dernieres annees, mis a jour chaque jour).
"""
import os, re, io, gzip
import numpy as np
import pandas as pd
import requests

# ============================== CONFIG ======================================= #
DEPTS = ["63", "15", "19"]
DATASETS = {"principal": "6569b4473bedf2e7abad3b72",
            "complementaire": "6791055a6d553b689a5e2ba4"}
OBS_DIR = os.environ.get("CUVETTE_OBS_DIR", r"C:\cuvette_archive\obs_mf")
VARS = ["T", "TD", "U", "FF", "DD", "FXY", "HNEIGEF", "NEIGETOT", "N", "GLO", "RR1"]
MIN_HOURS = 24 * 30                 # stations gardees si >= 30 j de donnees T
# ============================================================================ #


# Noms possibles : H_63_2010-2019.csv.gz (decennie), H_63_previous-2020-2024.csv.gz,
# H_63_latest-2025-2026.csv.gz (ou latest-AAAAMMJJ-AAAAMMJJ). On lit le nom dans le
# titre OU dans l'URL du fichier, car les deux ne sont pas toujours renseignes pareil.
FILE_RE = re.compile(r"H_(\d{2,3})_(?:(previous|latest)-)?(\d{4})\d*-(\d{4})\d*")
API = "https://www.data.gouv.fr/api"


def _list_resources(dataset_id):
    """Liste TOUTES les ressources d'un jeu (sans filtre serveur, pagination API v2,
    repli sur l'API v1 si besoin)."""
    out = []
    try:
        page = 1
        while True:
            r = requests.get(f"{API}/2/datasets/{dataset_id}/resources/",
                             params={"page": page, "page_size": 100}, timeout=60)
            r.raise_for_status()
            j = r.json()
            data = j.get("data", [])
            out += data
            if not data or not j.get("next_page"):
                break
            page += 1
    except Exception as e:
        print(f"[obs] API v2 en echec ({e}), repli API v1")
    if not out:
        r = requests.get(f"{API}/1/datasets/{dataset_id}/", timeout=120)
        r.raise_for_status()
        out = r.json().get("resources", [])
    return out


def _match(res):
    """Retourne (dep, kind, y0, y1, url) si la ressource est un fichier horaire H_, sinon None."""
    for txt in (res.get("title") or "", os.path.basename(res.get("url") or "")):
        m = FILE_RE.search(txt)
        if m:
            dep, kind, y0, y1 = m.group(1), m.group(2) or "decade", int(m.group(3)), int(m.group(4))
            url = res.get("latest") or res.get("url")      # 'latest' = URL stable data.gouv
            return dep, kind, y0, y1, url
    return None


def _get(url, out):
    with requests.get(url, stream=True, timeout=900, allow_redirects=True) as r:
        r.raise_for_status()
        n = 0
        with open(out + ".part", "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk); n += len(chunk)
    os.replace(out + ".part", out)
    return n


def list_files(depts=DEPTS, datasets=("principal",)):
    """Diagnostic : affiche TOUTES les ressources dont le titre ou l'URL evoque un
    departement demande, reconnues ou non par FILE_RE."""
    for dsn in datasets:
        for r in _list_resources(DATASETS[dsn]):
            t, u = r.get("title") or "", os.path.basename(r.get("url") or "")
            if any(f"_{d}_" in t or f"_{d}_" in u or f"_{d}." in u for d in depts):
                print(f"  {'OK ' if _match(r) else '?? '} {_match(r)[1:4] if _match(r) else ''} "
                      f"title={t!r} | url={u}")


def download(depts=DEPTS, years=(2010, 2026), datasets=("principal",), force=False):
    """Telecharge les CSV.gz horaires des departements dont la periode recoupe `years`.
    Les fichiers 'latest' (mis a jour chaque jour) sont toujours re-telecharges ;
    les autres seulement s'ils manquent (ou force=True)."""
    os.makedirs(OBS_DIR, exist_ok=True)
    got = []
    for dsn in datasets:
        res_all = _list_resources(DATASETS[dsn])
        print(f"[obs] jeu '{dsn}' : {len(res_all)} ressources listees par l'API")
        matched = [(_match(r), r) for r in res_all]
        matched = [(m, r) for m, r in matched if m]
        for (dep, kind, y0, y1, url), r in sorted(matched, key=lambda x: (x[0][0], x[0][2])):
            if dep not in depts or y1 < years[0] or y0 > years[1]:
                continue
            name = f"H_{dep}_latest.csv.gz" if kind == "latest" else f"H_{dep}_{kind}_{y0}-{y1}.csv.gz"
            out = os.path.join(OBS_DIR, f"{dsn}_{name}")
            if os.path.exists(out) and not force and kind != "latest":
                got.append(out); continue
            print(f"[obs] {name} ({y0}-{y1}) ...", end=" ", flush=True)
            n = _get(url, out)
            print(f"{n/1e6:.1f} Mo"); got.append(out)
        if not got:
            print(f"[obs] AUCUN fichier retenu pour {depts}. {len(matched)} fichiers H_ reconnus. "
                  "Exemples de ressources vues :")
            for r in res_all[:15]:
                print("    title=", repr(r.get("title")), "| url=", os.path.basename(r.get("url") or ""))
    return got


def _read_csv(path):
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        df = pd.read_csv(f, sep=";", dtype={"NUM_POSTE": str}, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    keep = ["NUM_POSTE", "NOM_USUEL", "LAT", "LON", "ALTI", "AAAAMMJJHH"] + \
           [v for v in VARS if v in df.columns]
    df = df[keep]
    df["time"] = pd.to_datetime(df["AAAAMMJJHH"].astype(str), format="%Y%m%d%H", utc=True)
    return df.drop(columns="AAAAMMJJHH")


def load(years=None, depts=DEPTS):
    """Concatene tous les fichiers presents dans OBS_DIR (dedoublonne latest/previous).
    years=(y0, y1) filtre l'intervalle."""
    files = [os.path.join(OBS_DIR, f) for f in sorted(os.listdir(OBS_DIR))
             if f.endswith(".csv.gz") and re.search(r"H_(\d{2,3})_", f)
             and re.search(r"H_(\d{2,3})_", f).group(1) in depts]
    if not files:
        raise SystemExit("Aucun fichier obs : lance obs_mf.download() d'abord.")
    parts = []
    for p in files:
        d = _read_csv(p)
        if years:
            d = d[(d.time.dt.year >= years[0]) & (d.time.dt.year <= years[1])]
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates(["NUM_POSTE", "time"], keep="last").sort_values(["NUM_POSTE", "time"])
    return df.reset_index(drop=True)


def stations(df=None, terrain_meta=None):
    """Table des stations avec coordonnees L93 et, si terrain_meta fourni
    (meta de terrain_features.load()), les indices (row, col) sur la grille MNT."""
    from pyproj import Transformer
    df = df if df is not None else load()
    cnt = df.groupby("NUM_POSTE")["T"].count()
    st = (df.groupby("NUM_POSTE").first()[["NOM_USUEL", "LAT", "LON", "ALTI"]]
            .reset_index())
    st = st[st.NUM_POSTE.map(cnt) >= MIN_HOURS].copy()
    tr = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)
    st["X"], st["Y"] = tr.transform(st.LON.values, st.LAT.values)
    if terrain_meta is not None:
        a, b, c, d, e, f = terrain_meta["transform"]
        st["col"] = ((st.X - c) / a).astype(int)
        st["row"] = ((st.Y - f) / e).astype(int)
        ny, nx = terrain_meta["shape"]
        st = st[(st.row >= 0) & (st.row < ny) & (st.col >= 0) & (st.col < nx)]
    return st.reset_index(drop=True)


def winter_nights(df, months=(10, 11, 12, 1, 2, 3)):
    """Sous-ensemble hivernal + Tmin nocturne par station et par nuit (18 UTC -> 09 UTC).
    Retourne DataFrame : NUM_POSTE, night (date du matin), Tmin, T06 (T a 06 UTC),
    ff_night (vent moyen nocturne)."""
    d = df[df.time.dt.month.isin(months)].copy()
    d["night"] = (d.time + pd.Timedelta(hours=6)).dt.date      # 18h J -> nuit datee J+1
    d = d[(d.time.dt.hour >= 18) | (d.time.dt.hour <= 9)]
    g = d.groupby(["NUM_POSTE", "night"])
    out = g.agg(Tmin=("T", "min"), ff_night=("FF", "mean"), n=("T", "count")).reset_index()
    t06 = d[d.time.dt.hour == 6].set_index(["NUM_POSTE", "night"])["T"].rename("T06")
    out = out.join(t06, on=["NUM_POSTE", "night"])
    return out[out.n >= 10]


def inversion_pairs(df, st, low_name="CLERMONT-FD", high_name="CACADOGNE"):
    """Serie de l'inversion vallee/sommet (T_bas - T_haut, C) a chaque heure, pour
    une paire de stations (recherche par nom). Utile pour valider le diagnostic modele.
    Le sommet du Puy de Dome (OPGC) n'est pas dans l'open data MF : par defaut on prend
    le Plateau de Cacadogne (Sancy, 1664 m) face a Clermont-Fd (331 m)."""
    def pick(name):
        s = st[st.NOM_USUEL.str.contains(name, case=False, na=False)]
        if s.empty:
            raise ValueError(f"station '{name}' introuvable ; stations : {sorted(st.NOM_USUEL)}")
        return s.iloc[0].NUM_POSTE
    lo, hi = pick(low_name), pick(high_name)
    a = df[df.NUM_POSTE == lo].set_index("time")["T"]
    b = df[df.NUM_POSTE == hi].set_index("time")["T"]
    j = pd.concat([a.rename("T_low"), b.rename("T_high")], axis=1).dropna()
    j["dT"] = j.T_low - j.T_high            # > 0 avec un gradient normal ; < 0 = inversion
    return j


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="y0", type=int, default=2020,
                    help="annee de debut (2010 pour une vraie base d'entrainement)")
    ap.add_argument("--to", dest="y1", type=int, default=2026)
    ap.add_argument("--list", action="store_true", help="liste les fichiers disponibles, sans telecharger")
    a = ap.parse_args()
    if a.list:
        list_files(); raise SystemExit
    files = download(years=(a.y0, a.y1))
    if not files:
        raise SystemExit("[obs] rien telecharge : copie-moi les lignes 'title= ... | url= ...' ci-dessus.")
    df = load()
    st = stations(df)
    print(f"[obs] {len(df)} lignes, {len(st)} stations avec T :")
    print(st[["NUM_POSTE", "NOM_USUEL", "ALTI"]].sort_values("ALTI").to_string(index=False))
    try:
        j = inversion_pairs(df, st)
        inv = j[j.dT < 0]
        print(f"[obs] Clermont-Fd vs Cacadogne : {len(j)} h en commun, "
              f"{100*len(inv)/max(len(j),1):.0f}% d'heures en inversion, "
              f"inversion max {j.dT.min():.1f} C le {j.dT.idxmin():%Y-%m-%d %Hh}")
    except ValueError as e:
        print("[obs]", e)
