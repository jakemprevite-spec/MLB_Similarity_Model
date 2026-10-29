"""Find MLB player-seasons with similar batting profiles."""

import argparse
import unicodedata

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = [
    "age",
    "pa",
    "k_pct",
    "bb_pct",
    "avg",
    "iso",
    "woba",
    "ops_plus",
    "hr",
    "sb",
    "exit_velocity",
    "barrel_pct",
    "hard_hit_pct",
    "sprint_speed",
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


def find_similar_seasons(
    profiles: pd.DataFrame,
    name: str,
    season: int,
    n_neighbors: int = 10,
) -> pd.DataFrame:
    """Return the nearest player-seasons for a named player-season."""
    required = {"name", "season"}
    missing = required - set(profiles.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    data = profiles.copy().reset_index(drop=True)
    data["season"] = pd.to_numeric(data["season"], errors="coerce")
    query_name_key = _name_key(name)
    name_keys = data["name"].map(_name_key)
    matches = data[
        name_keys.eq(query_name_key)
        & data["season"].eq(season)
    ]
    if matches.empty:
        raise ValueError(f"No row found for {name} in {season}.")
    if len(matches) > 1:
        raise ValueError(f"More than one row found for {name} in {season}.")

    feature_columns = [column for column in FEATURES if column in data.columns]
    if len(feature_columns) < 4:
        raise ValueError("At least four similarity features are required.")

    values = data[feature_columns].apply(pd.to_numeric, errors="coerce")
    preprocessor = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
    )
    transformed = preprocessor.fit_transform(values)

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

    neighbors = NearestNeighbors(metric="euclidean")
    neighbors.fit(transformed[candidate_positions])
    distances, indices = neighbors.kneighbors(
        preprocessor.transform(values.loc[[query_index]]),
        n_neighbors=min(len(candidate_positions), max(0, n_neighbors)),
    )

    result = data.iloc[candidate_positions[indices[0]]].copy()
    result.insert(0, "distance", distances[0])
    return pd.concat([query_result, result.head(n_neighbors)]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", help='Exact player name, for example "Mike Trout"')
    parser.add_argument("season", type=int)
    parser.add_argument("--n", type=int, default=10, dest="n_neighbors")
    parser.add_argument("--data", default="player_profiles.csv")
    args = parser.parse_args()

    profiles = pd.read_csv(args.data)
    results = find_similar_seasons(
        profiles, args.name, args.season, n_neighbors=args.n_neighbors
    )
    columns = [
        column
        for column in ["name", "season", "team", "age", "pa", "avg", "iso", "woba", "ops_plus", "hr", "sb", "distance"]
        if column in results
    ]
    print(results[columns].to_string(index=False))


if __name__ == "__main__":
    main()