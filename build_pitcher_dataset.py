"""Build pitcher-season profiles from Baseball-Reference data via pybaseball."""

import argparse
import re
import sys
import unicodedata

import pandas as pd


def _innings_to_float(value):
    if pd.isna(value):
        return float("nan")
    text = str(value).strip()
    match = re.fullmatch(r"(\d+)\.([12])", text)
    if match:
        return int(match.group(1)) + int(match.group(2)) / 3
    return pd.to_numeric(value, errors="coerce")


def fetch_bref_pitching(start_year: int, end_year: int, min_ip: float) -> pd.DataFrame:
    try:
        from pybaseball import cache, pitching_stats_bref
    except ImportError:
        sys.exit("pybaseball is not installed.\nRun:  pip install pybaseball pandas")

    cache.enable()
    frames = []
    for year in range(start_year, end_year + 1):
        print(f"Fetching Baseball-Reference pitching stats for {year} ...")
        try:
            frame = pitching_stats_bref(year)
        except Exception as error:
            print(f"  Warning: failed for {year}: {error}")
            continue
        frame["season"] = year
        frames.append(frame)

    if not frames:
        sys.exit("Couldn't fetch any Baseball-Reference pitching data — check your internet connection.")

    rename = {
        "Name": "name",
        "Tm": "team",
        "Age": "age",
        "IP": "ip",
        "G": "g",
        "GS": "gs",
        "H": "h",
        "ER": "er",
        "HR": "hr",
        "BB": "bb",
        "SO": "so",
        "BF": "bf",
        "ERA": "era",
        "FIP": "fip",
        "WHIP": "whip",
        "ERA+": "era_plus",
        "mlbID": "key_mlbam",
        "bbrefID": "key_bbref",
    }
    data = pd.concat(frames, ignore_index=True)
    keep = [column for column in rename if column in data.columns]
    data = data[keep + ["season"]].rename(columns=rename)
    data["name"] = data["name"].map(_decode_escaped_text)
    if "ip" not in data.columns:
        sys.exit("Baseball-Reference pitching data did not include innings pitched (IP).")

    data["ip"] = data["ip"].map(_innings_to_float)
    data = data[data["ip"] >= min_ip].copy()

    numeric_columns = [
        "age", "g", "gs", "h", "er", "hr", "bb", "so", "bf",
        "era", "fip", "whip", "era_plus",
    ]
    for column in numeric_columns:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    if {"er", "ip"} <= set(data.columns):
        data["era"] = 9 * data["er"] / data["ip"]
    if {"h", "bb", "ip"} <= set(data.columns):
        data["whip"] = (data["h"] + data["bb"]) / data["ip"]
    if {"so", "ip"} <= set(data.columns):
        data["k_per_9"] = 9 * data["so"] / data["ip"]
    if {"bb", "ip"} <= set(data.columns):
        data["bb_per_9"] = 9 * data["bb"] / data["ip"]
    if {"hr", "ip"} <= set(data.columns):
        data["hr_per_9"] = 9 * data["hr"] / data["ip"]
    if {"so", "bf"} <= set(data.columns):
        data["k_pct"] = data["so"] / data["bf"]
    if {"bb", "bf"} <= set(data.columns):
        data["bb_pct"] = data["bb"] / data["bf"]
    if {"gs", "g"} <= set(data.columns):
        data["gs_pct"] = data["gs"] / data["g"]

    data["name_key"] = data["name"].astype(str).apply(_name_key)
    return data.reset_index(drop=True)


def _name_key(name: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(name))
    without_accents = "".join(
        character for character in normalized if not unicodedata.combining(character)
    )
    parts = re.sub(r"[^a-zA-Z\s]", " ", without_accents).casefold().split()
    while parts and parts[-1] in {"jr", "sr", "ii", "iii", "iv", "v"}:
        parts.pop()
    return " ".join(parts)


def _decode_escaped_text(value: str) -> str:
    if not isinstance(value, str):
        return value

    def decode_bytes(match: re.Match[str]) -> str:
        values = re.findall(r"\\x([0-9a-fA-F]{2})", match.group())
        escaped = bytes(int(value, 16) for value in values)
        try:
            return escaped.decode("utf-8")
        except UnicodeDecodeError:
            return escaped.decode("latin-1")

    return re.sub(r"(?:\\x[0-9a-fA-F]{2})+", decode_bytes, value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=2015)
    parser.add_argument("--end", type=int, default=2025)
    parser.add_argument("--min-ip", type=float, default=30)
    parser.add_argument("--out", default="pitcher_profiles.csv")
    args = parser.parse_args()

    profiles = fetch_bref_pitching(args.start, args.end, args.min_ip)
    profiles = profiles.dropna(subset=["name", "era", "whip"])
    profiles.to_csv(args.out, index=False)
    print(f"\nDone. Wrote {len(profiles)} pitcher rows x {len(profiles.columns)} columns to {args.out}")
    print("Columns:", ", ".join(profiles.columns))


if __name__ == "__main__":
    main()