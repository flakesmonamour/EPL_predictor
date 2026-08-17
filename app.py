"""
app.py — Streamlit dashboard for the EPL match predictor.

Trains the model fresh on load (small dataset, so this is fast) and lets
you pick a home/away matchup to see live Win/Draw/Loss probabilities.

Run: streamlit run app.py
"""
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

FEATURES_PATH = '/home/flakesmonamour/Documents/predictor/data/raw/English_premier_leagues_processed.csv'
FEATURE_COLS = [
    'home_elo', 'away_elo', 'elo_diff',
    'home_form5', 'away_form5',
    'home_gf5', 'away_gf5',
    'home_ga5', 'away_ga5',
]


@st.cache_data
def load_data():
    return pd.read_csv(FEATURES_PATH).sort_values('date').reset_index(drop=True)


@st.cache_resource
def train_model(df):
    X = df[FEATURE_COLS]
    y = df['ftr']
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    model = LogisticRegression(max_iter=2000).fit(X_s, y_enc)
    return model, scaler, le


def latest_team_snapshot(df):
    """Most recent Elo/form/goal stats seen for each team, whichever side
    (home or away) they last played on."""
    snap = {}
    for _, row in df.iterrows():
        snap[row['home']] = {'elo': row['home_elo'], 'form5': row['home_form5'],
                              'gf5': row['home_gf5'], 'ga5': row['home_ga5']}
        snap[row['away']] = {'elo': row['away_elo'], 'form5': row['away_form5'],
                              'gf5': row['away_gf5'], 'ga5': row['away_ga5']}
    return snap


st.set_page_config(page_title='EPL Match Predictor', layout='centered')
st.title('EPL Match Predictor')
st.caption('Predictions from a Logistic Regression model trained on Elo ratings and recent form.')

df = load_data()
model, scaler, le = train_model(df)
snapshot = latest_team_snapshot(df)
teams = sorted(snapshot.keys())

col1, col2 = st.columns(2)
with col1:
    home = st.selectbox('Home team', teams, index=teams.index('Arsenal') if 'Arsenal' in teams else 0)
with col2:
    away_options = [t for t in teams if t != home]
    away = st.selectbox('Away team', away_options, index=0)

h, a = snapshot[home], snapshot[away]
x = [[h['elo'], a['elo'], h['elo'] - a['elo'], h['form5'], a['form5'],
      h['gf5'], a['gf5'], h['ga5'], a['ga5']]]
x_scaled = scaler.transform(x)
probs = model.predict_proba(x_scaled)[0]
prob_map = dict(zip(le.classes_, probs))

st.subheader('Prediction')
st.write(f"**{home} win:** {prob_map['H']*100:.0f}%")
st.progress(float(prob_map['H']))
st.write(f"**Draw:** {prob_map['D']*100:.0f}%")
st.progress(float(prob_map['D']))
st.write(f"**{away} win:** {prob_map['A']*100:.0f}%")
st.progress(float(prob_map['A']))

st.subheader('Team snapshots (latest matchday)')
snap_df = pd.DataFrame([
    {'Team': home, 'Elo': round(h['elo']), 'Form (5)': h['form5'], 'GF (5)': h['gf5'], 'GA (5)': h['ga5']},
    {'Team': away, 'Elo': round(a['elo']), 'Form (5)': a['form5'], 'GF (5)': a['gf5'], 'GA (5)': a['ga5']},
])
st.table(snap_df.set_index('Team'))
