"""
MLB Player Similarity Engine — Data Builder (Baseball-Reference + Savant version)
"""

import argparse
import re
import sys

import pandas as pd

try:
    from pybaseball import (
        batting_stats_bref,
        statcast_batter_exitvelo_barrels,
        statcast_sprint_speed,
        chadwick_register,
        cache,
    )
except ImportError:
    sys.exit("pybaseball is not installed.\nRun:  pip install pybaseball pandas")

cache.enable()


def fetch_bref_batting(start_year: int, end_year: int, min_pa: int) -> pd.DataFrame:
    frames = []
    for year in range(start_year, end_year + 1):
        print(f"Fetching Baseball-Reference batting stats for {year} ...")
        try:
            df = batting_stats_bref(year)
        except Exception as e:
            print(f"  Warning: failed for {year}: {e}")
            continue
        df["season"] = year
        frames.append(df)

    if not frames:
        sys.exit("Couldn't fetch any Baseball-Reference data — check your internet connection.")

    df = pd.concat(frames, ignore_index=True)

    rename = {
        "Name": "name", "Tm": "team", "Age": "age", "PA": "pa", "AB": "ab",
        "BB": "bb", "SO": "so", "HBP": "hbp", "SF": "sf", "IBB": "ibb",
        "2B": "b2", "3B": "b3", "HR": "hr", "BA": "avg", "SLG": "slg",
        "OPS+": "ops_plus", "SB": "sb", "mlbID": "key_mlbam", "bbrefID": "key_bbref",
    }
    keep = [c for c in rename if c in df.columns]
    df = df[keep + ["season"]].rename(columns=rename)

    df = df[pd.to_numeric(df["pa"], errors="coerce") >= min_pa].copy()

    for col in ["pa", "ab", "bb", "so", "hbp", "sf", "ibb", "b2", "b3", "hr"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["k_pct"] = df["so"] / df["pa"]
    df["bb_pct"] = df["bb"] / df["pa"]
    df["iso"] = pd.to_numeric(df["slg"], errors="coerce") - pd.to_numeric(df["avg"], errors="coerce")

    ubb = df["bb"] - df.get("ibb", 0).fillna(0)
    if "H" in df.columns:
        hits = pd.to_numeric(df["H"], errors="coerce")
    else:
        hits = pd.to_numeric(df["avg"], errors="coerce") * df["ab"]
    singles = hits - df["hr"] - df.get("b2", 0).fillna(0) - df.get("b3", 0).fillna(0)

    denom = df["ab"] + df["bb"] - df.get("ibb", 0).fillna(0) + df.get("sf", 0).fillna(0) + df.get("hbp", 0).fillna(0)
    df["woba"] = (
        0.69 * ubb
        + 0.72 * df.get("hbp", 0).fillna(0)
        + 0.89 * singles
        + 1.27 * df.get("b2", 0).fillna(0)
        + 1.62 * df.get("b3", 0).fillna(0)
        + 2.10 * df["hr"]
    ) / denom

    df["name_key"] = df["name"].astype(str).apply(normalize_name)
    return df


def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    name = re.sub(r"[^a-zA-Z\s]", "", name)
    return re.sub(r"\s+", " ", name).strip().lower()


def fetch_savant(start_year: int, end_year: int) -> pd.DataFrame:
    frames = []
    for year in range(start_year, end_year + 1):
        print(f"Fetching Savant exit velo / barrels for {year} ...")
        try:
            ev = statcast_batter_exitvelo_barrels(year, minBBE=50)
            spd = statcast_sprint_speed(year, min_opp=10)
        except Exception as e:
            print(f"  Warning: Savant fetch failed for {year}: {e}")
            continue

        ev = ev.rename(columns={
            "player_id": "key_mlbam", "avg_hit_speed": "exit_velocity",
            "brl_percent": "barrel_pct", "ev95percent": "hard_hit_pct",
        })
        spd = spd.rename(columns={"player_id": "key_mlbam", "sprint_speed": "sprint_speed"})

        keep_ev = [c for c in ["key_mlbam", "exit_velocity", "barrel_pct", "hard_hit_pct"] if c in ev.columns]
        keep_spd = [c for c in ["key_mlbam", "sprint_speed"] if c in spd.columns]

        merged = ev[keep_ev].merge(spd[keep_spd], on="key_mlbam", how="outer")
        merged["season"] = year
        frames.append(merged)

    if not frames:
        print("  Warning: no Savant data fetched.")
        return pd.DataFrame(columns=["key_mlbam", "season"])

    return pd.concat(frames, ignore_index=True)


def load_savant_extra(path: str) -> pd.DataFrame:
    print(f"Loading optional Savant custom leaderboard: {path}")
    df = pd.read_csv(path)

    col_map = {
        "name": ["last_name, first_name", "player_name", "Name"],
        "gb_pct": ["gb_percent", "GB%"],
        "fb_pct": ["fb_percent", "FB%"],
        "pull_pct": ["pull_percent", "Pull%"],
        "chase_pct": ["chase_percent", "Chase%"],
        "whiff_pct": ["whiff_percent", "Whiff%"],
        "zone_contact_pct": ["z_contact_percent", "Z-Contact%"],
    }
    out = pd.DataFrame()
    for new_col, candidates in col_map.items():
        norm = {re.sub(r"[^a-z0-9]", "", c.lower()): c for c in df.columns}
        found = next((norm[re.sub(r'[^a-z0-9]', '', c.lower())] for c in candidates
                      if re.sub(r'[^a-z0-9]', '', c.lower()) in norm), None)
        if found:
            out[new_col] = df[found]

    if "name" in out.columns:
        out["name_key"] = out["name"].apply(normalize_name)
    return out


def fetch_id_crosswalk() -> pd.DataFrame:
    print("Fetching player ID crosswalk (Chadwick register) ...")
    reg = chadwick_register()
    reg = reg[["key_mlbam", "key_bbref"]].dropna()
    reg["key_mlbam"] = reg["key_mlbam"].astype(int)
    return reg


def build_dataset(start_year: int, end_year: int, min_pa: int, savant_extra_path):
    bref = fetch_bref_batting(start_year, end_year, min_pa)
    savant = fetch_savant(start_year, end_year)

    if "key_mlbam" not in bref.columns:
        crosswalk = fetch_id_crosswalk()
        bref = bref.merge(crosswalk, on="key_bbref", how="left")

    bref["key_mlbam"] = pd.to_numeric(bref["key_mlbam"], errors="coerce")
    savant["key_mlbam"] = pd.to_numeric(savant["key_mlbam"], errors="coerce")

    merged = bref.merge(savant, on=["key_mlbam", "season"], how="left")

    if savant_extra_path:
        extra = load_savant_extra(savant_extra_path)
        if "name_key" in extra.columns:
            merged = merged.merge(extra.drop(columns=["name"], errors="ignore"), on="name_key", how="left")

    return merged


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    core = [c for c in ["k_pct", "bb_pct", "iso", "woba"] if c in df.columns]
    df = df.dropna(subset=["name"] + core)
    print(f"Cleaning: kept {len(df)}/{before} players after dropping incomplete rows.")
    return df.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Build the MLB similarity engine dataset (Bref + Savant)")
    parser.add_argument("--start", type=int, default=2024)
    parser.add_argument("--end", type=int, default=2025)
    parser.add_argument("--min-pa", type=int, default=200)
    parser.add_argument("--savant-extra", type=str, default=None)
    parser.add_argument("--out", type=str, default="player_profiles.csv")
    args, _unknown = parser.parse_known_args()

    df = build_dataset(args.start, args.end, args.min_pa, args.savant_extra)
    df = clean_dataset(df)

    df.to_csv(args.out, index=False)
    print(f"\nDone. Wrote {len(df)} player rows x {len(df.columns)} columns to {args.out}")
    print("Columns:", ", ".join(df.columns))


if __name__ == "__main__":
    main()
