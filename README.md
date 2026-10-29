# MLB Player Similarity Engine

Find the MLB player-seasons most statistically similar to any hitter or pitcher season from 2015 onward.

The project pulls season stats from Baseball-Reference and Statcast (via [pybaseball](https://github.com/jldbc/pybaseball)), merges the sources on MLB player IDs, and uses a k-nearest-neighbors model on standardized features to return the closest comparisons for any player-season.

## Example

```
$ python similarity_engine.py "Aaron Judge" 2024 --n 5

          name  season         team  age  pa   avg   iso     woba  hr  sb  distance
   Aaron Judge    2024     New York   32 768 0.311 0.367 0.469247  61  12  0.000000
 Shohei Ohtani    2025  Los Angeles   30 811 0.280 0.349 0.425355  63  21  2.619908
 Shohei Ohtani    2023  Los Angeles   28 599 0.304 0.350 0.443086  44  20  3.619653
Kyle Schwarber    2025 Philadelphia   32 742 0.239 0.326 0.397174  58  10  3.810905
     Juan Soto    2024     New York   25 777 0.291 0.283 0.428136  45   7  4.031014
    Matt Olson    2023      Atlanta   29 737 0.282 0.313 0.417300  54   1  4.115208
```

```
$ python pitcher_similarity_engine.py "Tarik Skubal" 2024 --n 5

             name  season          team  age         ip      era     whip   k_per_9  bb_per_9  hr_per_9  distance
     Tarik Skubal    2024       Detroit   27 211.000000 2.388626 0.909953 10.578199  1.578199  0.682464  0.000000
     Jacob deGrom    2015      New York   27 191.000000 2.544503 0.979058  9.659686  1.790576  0.753927  0.901452
  Clayton Kershaw    2015   Los Angeles   27 232.666667 2.127507 0.881089 11.643266  1.624642  0.580229  1.007921
 Brandon Woodruff    2021     Milwaukee   28 186.666667 2.603571 0.964286 10.558929  2.121429  0.916071  1.075685
  Garrett Crochet    2025        Boston   26 213.000000 2.535211 1.009390 11.239437  1.943662  1.056338  1.115133
```

The player's own other seasons are excluded, so results are always other players.

## How it works

1. **Build the data.** `build_hitter_dataset.py` pulls Baseball-Reference batting stats and Statcast exit velocity, barrel rate, hard-hit rate, and sprint speed, then joins them on MLB player IDs (using the Chadwick register as a crosswalk). `build_pitcher_dataset.py` does the same for Baseball-Reference pitching stats.
2. **Engineer features.** Rate stats such as K%, BB%, ISO, wOBA, K/9, BB/9, and HR/9 are calculated for each player-season.
3. **Find neighbors.** Missing values are filled with the median, every feature is standardized, and seasons are ranked by Euclidean distance from the chosen season.

| Model | Seasons | Features |
|---|---|---|
| Hitters | 2015–2026, 200+ PA | age, PA, K%, BB%, AVG, ISO, wOBA, HR, SB, exit velocity, barrel %, hard-hit %, sprint speed |
| Pitchers | 2015–2025, 30+ IP | age, IP, ERA, WHIP, K/9, BB/9, HR/9, K%, BB%, games-started % |

## Project structure

```
build_hitter_dataset.py         Builds player_profiles.csv (hitters)
build_pitcher_dataset.py        Builds pitcher_profiles.csv
similarity_engine.py            Hitter similarity search
pitcher_similarity_engine.py    Pitcher similarity search
player_profiles.csv             Hitter-season data
pitcher_profiles.csv            Pitcher-season data
MLB_Similarity_Engine.ipynb     Data building, cleaning, and exploration
similarity_engine_search.ipynb  Interactive search for hitters and pitchers
```

## Getting started

```
git clone https://github.com/<your-username>/MLB_Similarity.git
cd MLB_Similarity
python -m venv .venv
.venv\Scripts\activate          # Windows (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

The CSVs are included, so you can search right away:

```
python similarity_engine.py "Mike Trout" 2019
python pitcher_similarity_engine.py "Gerrit Cole" 2019 --n 5
```

To rebuild the data from scratch:

```
python build_hitter_dataset.py --start 2015 --end 2026
python build_pitcher_dataset.py --start 2015 --end 2025
```

## Data sources

- [Baseball-Reference](https://www.baseball-reference.com/) season batting and pitching stats
- [Baseball Savant](https://baseballsavant.mlb.com/) (Statcast) exit velocity, barrel, hard-hit, and sprint speed leaderboards
- [Chadwick Baseball Bureau](https://github.com/chadwickbureau/register) player ID register

All data is accessed through pybaseball.
