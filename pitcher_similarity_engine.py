"""Find MLB pitcher-seasons with similar pitching profiles."""

import argparse
import unicodedata

import numpy as np
import pandas as pd


PITCHER_FEATURES = [
    "age",
    "ip",
    "era",
    "fip",
    "whip",
    "k_per_9",
    "bb_per_9",
    "hr_per_9",
    "k_pct",
    "bb_pct",
    "gs_pct",
    "era_plus",
]


def _name_key(name: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(name))
    without_accents = "".join(
        character for character in normalized if not unicodedata.combining(character)
    )
    parts = without_accents.casefold().split()
    while parts and parts[-1].rstrip(".") in {"jr", "sr", "ii", "iii", "iv", "v"}:
        parts.pop()
    return " ".join(parts)


def find_similar_pitcher_seasons(
    profiles: pd.DataFrame,
    name: str,
    season: int,
    n_neighbors: int = 10,
) -> pd.DataFrame:
    """Return pitcher-seasons nearest to the named pitcher's season."""
    required = {"name", "season"}
    missing = required - set(profiles.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    data = profiles.copy().reset_index(drop=True)
    data["season"] = pd.to_numeric(data["season"], errors="coerce")
    name_keys = data["name"].map(_name_key)
    query_name_key = _name_key(name)
    matches = data[name_keys.eq(query_name_key) & data["season"].eq(season)]
    if matches.empty:
        raise ValueError(f"No row found for {name} in {season}.")
    if len(matches) > 1:
        raise ValueError(f"More than one row found for {name} in {season}.")

    feature_columns = [column for column in PITCHER_FEATURES if column in data.columns]
    if len(feature_columns) < 4:
        raise ValueError("At least four pitching features are required.")

    values = data[feature_columns].apply(pd.to_numeric, errors="coerce")
    values = values.dropna(axis=1, how="all")
    if values.shape[1] < 4:
        raise ValueError("At least four pitching features must contain data.")
    values = values.fillna(values.median())
    scales = values.std(ddof=0).replace(0, 1)
    transformed = ((values - values.mean()) / scales).to_numpy(dtype=float)

    query_index = matches.index[0]
    if "key_mlbam" in data.columns and pd.notna(data.at[query_index, "key_mlbam"]):
        same_player = data["key_mlbam"].eq(data.at[query_index, "key_mlbam"])
    else:
        same_player = name_keys.eq(query_name_key)
    candidate_positions = (~same_player).to_numpy().nonzero()[0]
    query_result = data.loc[[query_index]].copy()
    query_result.insert(0, "distance", 0.0)
    if not len(candidate_positions):
        return query_result.reset_index(drop=True)

    differences = transformed[candidate_positions] - transformed[query_index]
    distances = np.sqrt(np.square(differences).sum(axis=1))
    neighbor_count = min(len(candidate_positions), max(0, n_neighbors))
    nearest = np.argsort(distances, kind="stable")[:neighbor_count]
    result = data.iloc[candidate_positions[nearest]].copy()
    result.insert(0, "distance", distances[nearest])
    return pd.concat([query_result, result.head(n_neighbors)]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", help='Pitcher name, for example "Gerrit Cole"')
    parser.add_argument("season", type=int)
    parser.add_argument("--n", type=int, default=10, dest="n_neighbors")
    parser.add_argument("--data", default="pitcher_profiles.csv")
    args = parser.parse_args()

    profiles = pd.read_csv(args.data)
    results = find_similar_pitcher_seasons(
        profiles, args.name, args.season, n_neighbors=args.n_neighbors
    )
    columns = [
        column
        for column in ["name", "season", "team", "age", "ip", "era", "fip", "whip", "k_per_9", "bb_per_9", "hr_per_9", "distance"]
        if column in results
    ]
    print(results[columns].to_string(index=False))


if __name__ == "__main__":
    main()