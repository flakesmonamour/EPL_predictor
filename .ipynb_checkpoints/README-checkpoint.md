# EPL Match Outcome Predictor

Predicts English Premier League match results (Home win / Draw / Away win)
using Elo ratings and recent form, built from raw match data.

## Data

`data/raw/E0.csv` — 2025/26 EPL season, 380 matches, from
[football-data.co.uk](https://www.football-data.co.uk/data.php).

## Notebook

`notebooks/epl_analysis.ipynb` walks through the whole project end to end —
loading data, building the Elo/form features, training both models, and the
results — with explanations alongside each step. Start there.

`src/build_features.py` and `src/train_models.py` hold the same logic as
plain scripts, if you'd rather run it from the command line or import it
elsewhere.

## Approach

**Features** (`src/build_features.py`), all computed using only information
available *before* each match kicked off, so there's no data leakage:

- **Elo rating** per team — a running strength score updated after every
  match, with a home-advantage bonus built in.
- **Rolling form** — points won in each team's last 5 matches.
- **Rolling attack/defense** — average goals scored/conceded over the last 5.

**Models** (`src/train_models.py`), compared on a **time-based split**
(train on the first 80% of the season chronologically, test on the last
20% — never a random shuffle, since that would let future information leak
into training):

| Model | Accuracy | Log-loss |
|---|---|---|
| Always predict Home win (baseline) | 45.9% | — |
| Logistic Regression (scaled) | 55.4% | 1.014 |
| XGBoost | 48.6% | 1.136 |

## What this taught me

- **Logistic Regression beat XGBoost here.** With only ~296 training
  matches, XGBoost has enough capacity to overfit noise, while a simpler
  linear model generalizes better on small data. More power isn't
  automatically better — it needs enough data to earn it.
- **Draws are hard.** Every model's recall on draws was near-random
  (~20-25%). This is a known, well-documented pattern in football
  prediction — draws don't have a clean statistical signature the way
  home/away wins do.
- **Elo and form both carry real signal** — `elo_diff` and `home_form5`
  were the top two features by importance in XGBoost, matching the
  intuition that "who's the better team" and "who's in form right now"
  are what actually drives results.

## Next steps

- Time-series cross-validation (multiple rolling splits) instead of one
  train/test split, since the current test set is only 74 matches and
  one particular split could be lucky or unlucky.
- Pull additional seasons for more training data — should help XGBoost
  more than it helps Logistic Regression.
- Head-to-head features once multiple seasons are available.
- Streamlit dashboard for picking two teams and getting a live prediction.

## Running it

```bash
pip install -r requirements.txt
jupyter notebook notebooks/epl_analysis.ipynb   # walk through the full analysis
# — or, from the command line —
python src/build_features.py   # raw CSV -> data/processed/epl_features.csv
python src/train_models.py     # trains + evaluates both models
streamlit run app.py           # interactive dashboard: pick a fixture, see live probabilities
```
EOF
