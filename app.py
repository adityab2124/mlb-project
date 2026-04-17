import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss, accuracy_score
from sklearn.calibration import calibration_curve
import xgboost as xgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MLB Outcome Predictor",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

.stApp {
    background-color: #0d0d0d;
    color: #e8e8e8;
}

h1, h2, h3 {
    font-family: 'Bebas Neue', cursive !important;
    letter-spacing: 2px;
    color: #f5f5f5;
}

.metric-card {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-left: 3px solid #c8f542;
    padding: 16px 20px;
    border-radius: 4px;
    margin-bottom: 8px;
}

.metric-card .label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 1.5px;
}

.metric-card .value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 26px;
    font-weight: 600;
    color: #c8f542;
    margin-top: 4px;
}

.metric-card .sub {
    font-size: 11px;
    color: #666;
    margin-top: 2px;
}

.section-header {
    font-family: 'Bebas Neue', cursive;
    font-size: 13px;
    letter-spacing: 3px;
    color: #555;
    text-transform: uppercase;
    border-bottom: 1px solid #222;
    padding-bottom: 6px;
    margin-bottom: 16px;
    margin-top: 8px;
}

[data-testid="stSidebar"] {
    background-color: #111111;
    border-right: 1px solid #222;
}

[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMultiSelect label,
[data-testid="stSidebar"] .stSlider label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    letter-spacing: 1px;
    color: #666;
}

.stTabs [aria-selected="true"] {
    color: #c8f542 !important;
    border-bottom-color: #c8f542 !important;
}

div[data-testid="stDataFrame"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
}

.hero-title {
    font-family: 'Bebas Neue', cursive;
    font-size: 52px;
    letter-spacing: 4px;
    line-height: 1;
    color: #f5f5f5;
}

.hero-sub {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    color: #555;
    letter-spacing: 2px;
    margin-top: 4px;
}

.badge {
    display: inline-block;
    background: #1e2a0e;
    color: #c8f542;
    border: 1px solid #c8f542;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    padding: 2px 8px;
    border-radius: 2px;
    letter-spacing: 1px;
    margin-right: 6px;
}
</style>
""", unsafe_allow_html=True)

# ── Data pipeline ─────────────────────────────────────────────────────────────
@st.cache_data
def load_and_train(csv_path):
    raw = pd.read_csv(csv_path)

    def ml_to_prob(ml):
        ml = pd.to_numeric(ml, errors='coerce')
        return np.where(ml < 0, -ml / (-ml + 100), 100 / (ml + 100))

    park_home = (raw.groupby(['parkName', 'team']).size()
                 .reset_index(name='n')
                 .sort_values('n', ascending=False)
                 .drop_duplicates('parkName'))
    home_map = dict(zip(park_home['parkName'], park_home['team']))
    raw['home_team'] = raw['parkName'].map(home_map)

    games = raw[raw['team'] == raw['home_team']].copy()
    games = games.rename(columns={
        'team': 'home_team_name', 'opponent': 'away_team',
        'projectedRuns': 'home_proj_runs', 'moneyLine': 'home_ml'
    })
    away = raw[raw['team'] != raw['home_team']][['date', 'parkName', 'projectedRuns', 'moneyLine']].copy()
    away = away.rename(columns={'projectedRuns': 'away_proj_runs', 'moneyLine': 'away_ml'})
    games = games.merge(away, on=['date', 'parkName'], how='inner')

    games['home_win'] = (games['runs'] > games['oppRuns']).astype(int)
    games['implied_prob'] = ml_to_prob(games['home_ml'])
    vig = games['implied_prob'] + ml_to_prob(games['away_ml'])
    games['implied_prob'] = games['implied_prob'] / vig
    games['diff_proj_runs'] = games['home_proj_runs'] - games['away_proj_runs']
    games = games.sort_values('date').reset_index(drop=True)
    games['season'] = pd.to_numeric(games['season'], errors='coerce')

    home_hist, away_hist = {}, {}
    home_rolling, away_rolling = [], []
    for _, row in games.iterrows():
        ht, at = row['home_team_name'], row['away_team']
        hh = home_hist.get(ht, [])
        ah = away_hist.get(at, [])
        home_rolling.append(np.mean(hh[-15:]) if len(hh) >= 5 else np.nan)
        away_rolling.append(np.mean(ah[-15:]) if len(ah) >= 5 else np.nan)
        hh.append(row['home_win']); home_hist[ht] = hh
        ah.append(1 - row['home_win']); away_hist[at] = ah

    games['home_rolling_wr'] = home_rolling
    games['away_rolling_wr'] = away_rolling
    games['diff_rolling_wr'] = games['home_rolling_wr'] - games['away_rolling_wr']

    FEATURE_COLS = ['home_proj_runs', 'away_proj_runs', 'diff_proj_runs', 'total',
                    'home_rolling_wr', 'away_rolling_wr', 'diff_rolling_wr']
    model_df = games.dropna(subset=FEATURE_COLS + ['implied_prob', 'home_win']).copy()

    train = model_df[model_df['season'] <= 2019]
    test = model_df[model_df['season'] >= 2020].copy()
    X_train = train[FEATURE_COLS].values
    y_train = train['home_win'].values
    X_test = test[FEATURE_COLS].values
    y_test = test['home_win'].values

    # Train models
    lr = LogisticRegression(max_iter=1000, random_state=42, penalty='l2', C=0.1, solver='liblinear')
    lr.fit(X_train, y_train)

    rf = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=20, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    xgb_model = xgb.XGBClassifier(objective='binary:logistic', n_estimators=300, learning_rate=0.05,
                                    max_depth=4, subsample=0.8, colsample_bytree=0.8,
                                    min_child_weight=10, random_state=42, eval_metric='logloss', verbosity=0)
    xgb_model.fit(X_train, y_train)

    cat_model = CatBoostClassifier(iterations=300, learning_rate=0.05, depth=5,
                                    l2_leaf_reg=10, random_seed=42, verbose=0)
    cat_model.fit(X_train, y_train)

    test = test.copy()
    test['lr_prob'] = lr.predict_proba(X_test)[:, 1]
    test['rf_prob'] = rf.predict_proba(X_test)[:, 1]
    test['xgb_prob'] = xgb_model.predict_proba(X_test)[:, 1]
    test['cat_prob'] = cat_model.predict_proba(X_test)[:, 1]

    def ece(y_true, y_prob, n_bins=10):
        bins = np.linspace(0, 1, n_bins + 1)
        total, err = 0, 0
        for i in range(n_bins):
            mask = (y_prob >= bins[i]) & (y_prob < bins[i+1])
            if mask.sum() > 0:
                err += mask.sum() * abs(y_true[mask].mean() - y_prob[mask].mean())
                total += mask.sum()
        return err / total if total > 0 else 0

    metrics = {}
    for name, col in [('Logistic Regression', 'lr_prob'), ('Random Forest', 'rf_prob'),
                       ('XGBoost', 'xgb_prob'), ('CatBoost', 'cat_prob')]:
        p = test[col].values
        metrics[name] = {
            'accuracy': accuracy_score(y_test, (p > 0.5).astype(int)),
            'auc': roc_auc_score(y_test, p),
            'brier': brier_score_loss(y_test, p),
            'ece': ece(y_test, p)
        }
    sb_p = test['implied_prob'].values
    metrics['Sportsbook'] = {
        'accuracy': accuracy_score(y_test, (sb_p > 0.5).astype(int)),
        'auc': roc_auc_score(y_test, sb_p),
        'brier': brier_score_loss(y_test, sb_p),
        'ece': ece(y_test, sb_p)
    }

    test['gap_lr'] = test['lr_prob'] - test['implied_prob']
    test['gap_rf'] = test['rf_prob'] - test['implied_prob']
    test['gap_xgb'] = test['xgb_prob'] - test['implied_prob']
    test['gap_cat'] = test['cat_prob'] - test['implied_prob']

    return test, metrics, model_df

# ── Load data ─────────────────────────────────────────────────────────────────
DATA_PATH = "oddsDataMLB.csv"

try:
    test, metrics, model_df = load_and_train(DATA_PATH)
    data_loaded = True
except FileNotFoundError:
    data_loaded = False

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="hero-title">⚾</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-family:\'Bebas Neue\',cursive;font-size:22px;letter-spacing:3px;color:#f5f5f5">MLB PREDICTOR</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">TEAM 139 · CSE 6242</div>', unsafe_allow_html=True)
    st.markdown("---")

    if data_loaded:
        all_teams = sorted(set(test['home_team_name'].unique()) | set(test['away_team'].unique()))
        selected_teams = st.multiselect("Filter by Team", all_teams, default=[])
        selected_seasons = st.multiselect("Season", sorted(test['season'].unique()), default=sorted(test['season'].unique()))
        selected_model = st.selectbox("Primary Model", ['Logistic Regression', 'Random Forest', 'XGBoost', 'CatBoost'])
        gap_threshold = st.slider("Bet Gap Threshold", 0.00, 0.10, 0.03, 0.005, format="%.3f",
                                   help="Min probability gap (model - sportsbook) to place a simulated bet")
        st.markdown("---")
        st.markdown('<div style="font-family:\'IBM Plex Mono\',monospace;font-size:10px;color:#444;letter-spacing:1px">DATA: KAGGLE ODDS 2012–2021<br>TEST SET: 2020–2021<br>N GAMES: {:,}</div>'.format(len(test)), unsafe_allow_html=True)

# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">Predicting MLB Game Outcomes</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">HOW GOOD ARE SPORTSBOOK MODELS? &nbsp;·&nbsp; TEAM 139 &nbsp;·&nbsp; GEORGIA TECH CSE 6242</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

if not data_loaded:
    st.error("Place `oddsDataMLB.csv` in the same directory and restart.")
    st.stop()

# Filter test data
filtered = test.copy()
if selected_teams:
    filtered = filtered[filtered['home_team_name'].isin(selected_teams) | filtered['away_team'].isin(selected_teams)]
if selected_seasons:
    filtered = filtered[filtered['season'].isin(selected_seasons)]

model_col_map = {
    'Logistic Regression': 'lr_prob',
    'Random Forest': 'rf_prob',
    'XGBoost': 'xgb_prob',
    'CatBoost': 'cat_prob'
}
gap_col_map = {
    'Logistic Regression': 'gap_lr',
    'Random Forest': 'gap_rf',
    'XGBoost': 'gap_xgb',
    'CatBoost': 'gap_cat'
}
sel_col = model_col_map[selected_model]
sel_gap = gap_col_map[selected_model]

# ── Top metrics row ───────────────────────────────────────────────────────────
m = metrics[selected_model]
sb = metrics['Sportsbook']
col1, col2, col3, col4, col5 = st.columns(5)

def metric_card(label, value, sub=""):
    return f"""<div class="metric-card">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        <div class="sub">{sub}</div>
    </div>"""

with col1:
    st.markdown(metric_card("Accuracy", f"{m['accuracy']:.1%}", f"Sportsbook: {sb['accuracy']:.1%}"), unsafe_allow_html=True)
with col2:
    st.markdown(metric_card("ROC-AUC", f"{m['auc']:.4f}", f"Sportsbook: {sb['auc']:.4f}"), unsafe_allow_html=True)
with col3:
    st.markdown(metric_card("Brier Score", f"{m['brier']:.4f}", f"Sportsbook: {sb['brier']:.4f}"), unsafe_allow_html=True)
with col4:
    st.markdown(metric_card("ECE", f"{m['ece']:.4f}", f"Sportsbook: {sb['ece']:.4f}"), unsafe_allow_html=True)
with col5:
    bets = filtered[filtered[sel_gap].abs() >= gap_threshold]
    if len(bets) > 0:
        wins = bets[((bets[sel_gap] > 0) & (bets['home_win'] == 1)) | ((bets[sel_gap] < 0) & (bets['home_win'] == 0))]
        roi = (len(wins) / len(bets) - 0.5) / 0.5 * 100
        st.markdown(metric_card("Sim ROI", f"{roi:+.1f}%", f"{len(bets)} bets @ ≥{gap_threshold:.3f} gap"), unsafe_allow_html=True)
    else:
        st.markdown(metric_card("Sim ROI", "N/A", "No bets at threshold"), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 GAME EXPLORER", "📈 CALIBRATION", "💰 BACKTEST", "🏆 MODEL COMPARISON", "🔍 GAP ANALYSIS"])

PLOT_LAYOUT = dict(
    paper_bgcolor='#0d0d0d',
    plot_bgcolor='#111111',
    font=dict(family='IBM Plex Mono', color='#aaa', size=11),
    xaxis=dict(gridcolor='#1e1e1e', linecolor='#333'),
    yaxis=dict(gridcolor='#1e1e1e', linecolor='#333'),
    margin=dict(l=40, r=20, t=40, b=40)
)

# TAB 1: Game Explorer
with tab1:
    st.markdown('<div class="section-header">Test Set Games (2020–2021)</div>', unsafe_allow_html=True)

    display = filtered[['date','season','home_team_name','away_team','implied_prob',
                          sel_col,'home_win']].copy()
    display.columns = ['Date','Season','Home Team','Away Team','Sportsbook Prob',
                        f'{selected_model} Prob','Home Win']
    display['Gap'] = display[f'{selected_model} Prob'] - display['Sportsbook Prob']
    display['Gap'] = display['Gap'].round(4)
    display['Sportsbook Prob'] = display['Sportsbook Prob'].round(3)
    display[f'{selected_model} Prob'] = display[f'{selected_model} Prob'].round(3)
    display['Home Win'] = display['Home Win'].map({1: '✓', 0: '✗'})

    st.dataframe(
        display.sort_values('Date', ascending=False).reset_index(drop=True),
        use_container_width=True,
        height=400
    )
    st.caption(f"Showing {len(display):,} games · Gap = Model Prob − Sportsbook Prob")

# TAB 2: Calibration Curves
with tab2:
    st.markdown('<div class="section-header">Calibration Curves: Model vs. Sportsbook</div>', unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines',
                              line=dict(dash='dash', color='#444', width=1),
                              name='Perfect Calibration', showlegend=True))

    colors = {'Logistic Regression':'#c8f542','Random Forest':'#42b8f5',
               'XGBoost':'#f5a742','CatBoost':'#f542a7','Sportsbook':'#ffffff'}

    for name, col in [*model_col_map.items(), ('Sportsbook', 'implied_prob')]:
        frac_pos, mean_pred = calibration_curve(test['home_win'], test[col], n_bins=10)
        lw = 2.5 if name == 'Sportsbook' else 1.5
        fig.add_trace(go.Scatter(x=mean_pred, y=frac_pos, mode='lines+markers',
                                  name=f"{name} (ECE={metrics[name]['ece']:.4f})",
                                  line=dict(color=colors[name], width=lw),
                                  marker=dict(size=6)))

    fig.update_layout(title='Calibration Curves', xaxis_title='Mean Predicted Probability',
                       yaxis_title='Actual Win Rate', **PLOT_LAYOUT, height=450)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Well-calibrated models track the dashed line. Lower ECE = better calibration. Sportsbook ECE: 0.0142.")

# TAB 3: Backtest
with tab3:
    st.markdown('<div class="section-header">Simulated Betting Backtest</div>', unsafe_allow_html=True)

    col_l, col_r = st.columns([1.2, 1])

    with col_l:
        thresholds = np.arange(0.00, 0.11, 0.005)
        roi_by_model = {}
        for mname, gcol in gap_col_map.items():
            rois, counts = [], []
            for t in thresholds:
                b = test[test[gcol].abs() >= t]
                if len(b) > 0:
                    w = b[((b[gcol] > 0) & (b['home_win'] == 1)) | ((b[gcol] < 0) & (b['home_win'] == 0))]
                    rois.append((len(w)/len(b) - 0.5)/0.5*100)
                    counts.append(len(b))
                else:
                    rois.append(0); counts.append(0)
            roi_by_model[mname] = rois

        fig2 = go.Figure()
        for mname, rois in roi_by_model.items():
            fig2.add_trace(go.Scatter(
                x=thresholds, y=rois, mode='lines',
                name=mname, line=dict(color=colors[mname], width=2 if mname==selected_model else 1.5)
            ))
        fig2.add_hline(y=0, line_dash='dash', line_color='#555')
        fig2.add_vline(x=gap_threshold, line_dash='dot', line_color='#c8f542',
                        annotation_text=f"  {gap_threshold:.3f}", annotation_font_color='#c8f542')
        fig2.update_layout(title='ROI by Gap Threshold', xaxis_title='Gap Threshold',
                            yaxis_title='Simulated ROI (%)', **PLOT_LAYOUT, height=380)
        st.plotly_chart(fig2, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-header">Bet Summary</div>', unsafe_allow_html=True)
        rows = []
        for mname, gcol in gap_col_map.items():
            b = test[test[gcol].abs() >= gap_threshold]
            if len(b) > 0:
                w = b[((b[gcol] > 0) & (b['home_win'] == 1)) | ((b[gcol] < 0) & (b['home_win'] == 0))]
                r = (len(w)/len(b) - 0.5)/0.5*100
            else:
                r = 0
            rows.append({'Model': mname, 'Bets': len(b),
                          'Win Rate': f"{len(w)/len(b):.1%}" if len(b) > 0 else "—",
                          'ROI': f"{r:+.1f}%"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(f"Gap threshold: {gap_threshold:.3f} · Adjust slider in sidebar")

# TAB 4: Model Comparison
with tab4:
    st.markdown('<div class="section-header">Model Performance vs. Sportsbook Benchmark</div>', unsafe_allow_html=True)

    rows = []
    for name in ['Logistic Regression', 'Random Forest', 'XGBoost', 'CatBoost', 'Sportsbook']:
        m2 = metrics[name]
        rows.append({
            'Model': name,
            'Accuracy': f"{m2['accuracy']:.1%}" if name != 'Sportsbook' else '—',
            'AUC': f"{m2['auc']:.4f}",
            'Brier Score': f"{m2['brier']:.4f}",
            'ECE': f"{m2['ece']:.4f}"
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=230)

    col_a, col_b = st.columns(2)
    with col_a:
        names = list(metrics.keys())
        aucs = [metrics[n]['auc'] for n in names]
        colors_bar = ['#c8f542' if n != 'Sportsbook' else '#ffffff' for n in names]
        fig3 = go.Figure(go.Bar(x=names, y=aucs, marker_color=colors_bar,
                                 text=[f"{v:.4f}" for v in aucs], textposition='outside',
                                 textfont=dict(family='IBM Plex Mono', size=10)))
        fig3.add_hline(y=metrics['Sportsbook']['auc'], line_dash='dash', line_color='#555')
        layout3 = {**PLOT_LAYOUT, 'height': 320}
        layout3['yaxis'] = {**layout3.get('yaxis', {}), 'range': [0.55, 0.65]}
        fig3.update_layout(title='AUC Comparison', **layout3)
        st.plotly_chart(fig3, use_container_width=True)

    with col_b:
        eces = [metrics[n]['ece'] for n in names]
        fig4 = go.Figure(go.Bar(x=names, y=eces, marker_color=colors_bar,
                                 text=[f"{v:.4f}" for v in eces], textposition='outside',
                                 textfont=dict(family='IBM Plex Mono', size=10)))
        fig4.update_layout(title='ECE Comparison (lower = better calibrated)',
                            **PLOT_LAYOUT, height=320)
        st.plotly_chart(fig4, use_container_width=True)

# TAB 5: Gap Analysis
with tab5:
    st.markdown('<div class="section-header">Model vs. Sportsbook Probability Gap</div>', unsafe_allow_html=True)

    col_l2, col_r2 = st.columns(2)

    with col_l2:
        fig5 = go.Figure()
        fig5.add_trace(go.Histogram(x=filtered[sel_gap], nbinsx=40,
                                     marker_color='#c8f542', opacity=0.8,
                                     name=f'{selected_model} Gap'))
        fig5.add_vline(x=0, line_color='#555', line_dash='dash')
        fig5.update_layout(title=f'{selected_model} Gap Distribution',
                            xaxis_title='Model Prob − Sportsbook Prob',
                            yaxis_title='Games', **PLOT_LAYOUT, height=350)
        st.plotly_chart(fig5, use_container_width=True)

    with col_r2:
        filtered_sorted = filtered.copy()
        filtered_sorted['gap_quintile'] = pd.qcut(filtered_sorted[sel_gap], 5,
                                                    labels=['Q1\nModel↓Away', 'Q2', 'Q3\nNeutral', 'Q4', 'Q5\nModel↑Home'])
        win_by_q = filtered_sorted.groupby('gap_quintile')['home_win'].mean().reset_index()
        fig6 = go.Figure(go.Bar(
            x=win_by_q['gap_quintile'].astype(str),
            y=win_by_q['home_win'],
            marker_color=['#f54242','#f59542','#888','#85c442','#c8f542'],
            text=[f"{v:.1%}" for v in win_by_q['home_win']],
            textposition='outside',
            textfont=dict(family='IBM Plex Mono', size=11)
        ))
        fig6.add_hline(y=0.5, line_dash='dash', line_color='#555')
        fig6.update_layout(title='Actual Win Rate by Gap Quintile',
                            yaxis=dict(range=[0, 0.7]),
                            xaxis_title='Gap Quintile', yaxis_title='Home Win Rate',
                            **PLOT_LAYOUT, height=350)
        st.plotly_chart(fig6, use_container_width=True)

    st.caption("Gap = Model Probability − Sportsbook Implied Probability. Q5 = games where model strongly favors home team relative to sportsbook.")
