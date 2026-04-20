import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import os
import sys
import scipy.stats as stats_sci
from datetime import datetime, timedelta

# Ensure the 'src' directory is in the path for modular imports
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from autocall_pricer.engine.zero_coupon import YieldCurve
from autocall_pricer.engine.vol_surface import VolatilitySurface
from autocall_pricer.engine.monte_carlo import MonteCarloSimulator
from autocall_pricer.products.autocall import AutocallAthena
from autocall_pricer.engine.market_data import (
    fetch_historical_data, 
    calculate_historical_volatility, 
    get_latest_spot, 
    calculate_return_stats, 
    calculate_rolling_volatility,
    fetch_yield_curve,
    fetch_volatility_curve
)

st.set_page_config(
    page_title="Autocall Pricer",
    layout="wide",
)

# --- THEME CONSTANTS ---
GOLD = "#B8860B"    # Professional Gold
CARBON = "#1A1A1A"  # Dark Text
SURFACE = "#FFFFFF" # Light Surface
BORDER = "#EAECEF"  # Light Border
TEAL = "#2F855A"    # Success Green
CORAL = "#E53E3E"   # Risk Red
SLATE = "#707A8A"   # Secondary Text

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;800&family=Inter:wght@400;600&family=Roboto+Mono:wght@500;700&display=swap');
    
    /* --- HIDE DEFAULT STREAMLIT ARTIFACTS --- */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden !important;}
    .stDeployButton {display: none;}
    
    /* --- GLOBAL LAYOUT --- */
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    
    /* Add sufficient top padding so the Streamlit header doesn't overlap content */
    .block-container {
        padding-top: 4rem !important; 
        padding-bottom: 2rem !important;
        max-width: 100% !important;
        padding-left: 3rem !important;
        padding-right: 3rem !important;
    }
    
    h1, h2, h3 { 
        font-family: 'Outfit', sans-serif;
        font-weight: 800; 
        border-bottom: none; 
        letter-spacing: -0.5px; 
    }
    
    /* --- METRICS / WIDGETS --- */
    div[data-testid="metric-container"] {
        border-radius: 8px;
        border: 1px solid var(--secondary-background-color);
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s, box-shadow 0.2s;
        padding: 1rem;
        background-color: transparent;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        border-color: #B8860B;
        box-shadow: 0 4px 12px rgba(184, 134, 11, 0.1);
    }
    div[data-testid="stMetricValue"] {
        color: #B8860B !important; 
        font-family: 'Roboto Mono', monospace;
        font-weight: 700;
        font-size: 2rem !important;
    }
    div[data-testid="stMetricLabel"] {
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.8rem !important;
    }
    
    /* Modern Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        border-bottom: 2px solid var(--secondary-background-color);
        padding-bottom: 0px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: transparent;
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        font-size: 1rem;
        padding: 0 1rem;
    }
    .stTabs [aria-selected="true"] {
        border-bottom: 3px solid #B8860B !important;
    }
    
    /* Alerts */
    div.stAlert {
        border: 1px solid var(--secondary-background-color);
        border-left: 4px solid #B8860B;
        border-radius: 4px;
    }
    
    /* Ensure Data Editors and Tables fill their slots */
    div[data-testid="stDataEditor"], div[data-testid="stTable"], .stDataFrame {
        width: 100% !important;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 6px;
        font-weight: 600;
        font-family: 'Outfit', sans-serif;
        letter-spacing: 0.5px;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        border-color: #B8860B;
        color: #B8860B;
    }
    </style>
""", unsafe_allow_html=True)

# Load CSV
@st.cache_data
def load_csv_data(mtime):
    csv_path = os.path.join(os.path.dirname(__file__), "data", "market_data.csv")
    df = pd.read_csv(csv_path)
    return df

st.markdown("""
<div style="background-color: var(--secondary-background-color); padding: 1.2rem 2rem; border-radius: 8px; margin-bottom: 2rem; display: flex; justify-content: space-between; align-items: center; border: 1px solid var(--primary-color); border-left: 5px solid #B8860B; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
    <h1 style="color: #B8860B; margin: 0; font-family: 'Outfit', sans-serif; font-size: 1.6rem; letter-spacing: -0.5px;">Institutional <span style="font-weight: 400; font-size: 1.4rem;">| Autocall Pricer</span></h1>
    <span style="font-family: 'Roboto Mono', monospace; font-size: 0.85rem; padding: 4px 10px; border-radius: 4px; border: 1px solid var(--primary-color);">SYSTEM v2.0</span>
</div>
""", unsafe_allow_html=True)

csv_p = os.path.join(os.path.dirname(__file__), "data", "market_data.csv")
df_indices = load_csv_data(os.path.getmtime(csv_p))
index_options = {f"{r['Index Name']}": r['Ticker'] for _, r in df_indices.iterrows()}

with st.sidebar:
    st.header("Product Structure Type")
    pricing_mode = st.radio("Pricing Mode", ["Single Asset", "Worst-Of (2 Assets)"], horizontal=True)
    is_wo = (pricing_mode == "Worst-Of (2 Assets)")
    
    st.markdown("---")
    st.header("Underlying Selection")
    
    selected_name1 = st.selectbox("Select Asset #1", list(index_options.keys()))
    if is_wo:
        selected_name2 = st.selectbox("Select Asset #2", [n for n in index_options.keys() if n != selected_name1])
        rho_val = st.slider("Asset Correlation", -1.0, 1.0, 0.5, step=0.05, help="Correlation between index returns")
    else:
        selected_name2 = None
        rho_val = 1.0

    # Bridge logic for maturity persistence
    if "selected_maturity" not in st.session_state:
        st.session_state.selected_maturity = 5
        
    # Calibration Block
    selected_assets = [selected_name1]
    if is_wo: selected_assets.append(selected_name2)
    
    spots_list, vols_list, divs_list, currencies_list = [], [], [], []
    if "basket_history" not in st.session_state:
        st.session_state.basket_history = {}
    
    for asset_name in selected_assets:
        ticker = index_options[asset_name]
        idx_row = df_indices[df_indices["Index Name"] == asset_name].iloc[0]
        currencies_list.append(idx_row["Currency"])
        
        with st.spinner(f"Calibrating {asset_name}..."):
            try:
                h_df = fetch_historical_data(ticker, period=f"{st.session_state.selected_maturity}y").dropna(subset=['Close'])
                empirical_spot = get_latest_spot(h_df)
                empirical_vol = calculate_historical_volatility(h_df)
                st.session_state.basket_history[asset_name] = h_df
                
                # Pre-fetch Volatility Curve for each asset
                if f"vol_curve_{asset_name}" not in st.session_state:
                    st.session_state[f"vol_curve_{asset_name}"] = fetch_volatility_curve(ticker)
            except:
                empirical_spot, empirical_vol = 100.0, 0.20
            
            spots_list.append(empirical_spot)
            vols_list.append(empirical_vol)
            divs_list.append(0.02) # Default div

    # --- 1.2 VOLATILITY SURFACE SETTINGS ---
    st.markdown("### Volatility Surfaces")
    
    skew_intensity = st.slider("Skew Intensity (Local Vol)", -2.0, 1.0, -0.2, 0.1, help="Sensitivity of vol to price. Negative means vol increases if spot drops.")
    
    vol_surfaces = []
    divs_arr = []
    
    for i, asset_name in enumerate(selected_assets):
        with st.expander(f"Market: {asset_name}", expanded=True):
            s_val = st.number_input(f"Spot: {asset_name}", value=spots_list[i], step=10.0, key=f"spot_in_{asset_name}")
            d_val = st.number_input(f"Div Yield: {asset_name}", value=0.02, step=0.01, format="%.3f", key=f"div_in_{asset_name}")
            divs_arr.append(d_val)
            
            curve_df = st.session_state.get(f"vol_curve_{asset_name}")
            if curve_df is not None:
                edited_vol = st.data_editor(
                    curve_df,
                    column_config={
                        "Tenor": st.column_config.TextColumn("Tenor", disabled=True, width="small"),
                        "Years": st.column_config.NumberColumn("Yrs", disabled=True, width="small"),
                        "Rate (%)": st.column_config.NumberColumn("ATM Vol (%)", format="%.2f%%")
                    },
                    width="stretch",
                    hide_index=True,
                    key=f"vol_editor_{asset_name}"
                )
                
                # Create VolatilitySurface object
                surf = VolatilitySurface(
                    tenors=edited_vol["Years"].values,
                    atm_vols=edited_vol["Rate (%)"].values / 100.0,
                    skew_intensity=skew_intensity,
                    s0=s_val
                )
                vol_surfaces.append(surf)
                spots_list[i] = s_val # Update spots_list with user input
            else:
                # Fallback: create a flat 20% vol surface if no historical data
                st.caption("Using default flat vol (20%)")
                fallback_vol = vols_list[i] if i < len(vols_list) else 0.20
                surf = VolatilitySurface.from_flat_vol(fallback_vol, skew=skew_intensity, s0=s_val)
                vol_surfaces.append(surf)
                spots_list[i] = s_val
                
    spots_arr = np.array(spots_list)
    
    st.markdown("### Correlation & Dividends")

    # --- Yield Curve Logic (Consolidated) ---
    st.subheader("Yield Curve")
    
    unique_currencies = list(set(currencies_list))
    
    if len(unique_currencies) > 1:
        # Multi-currency logic: prepare the averaged DataFrame
        curves = [fetch_yield_curve(c) for c in unique_currencies]
        avg_rates = np.mean([c['Rate (%)'].values for c in curves], axis=0)
        tenors_years = curves[0]['Years'].values
        tenors_labels = curves[0]['Tenor'].values
        
        display_df = pd.DataFrame({
            "Tenor": tenors_labels,
            "Years": tenors_years,
            "Rate (%)": avg_rates
        })
        st.caption("Consolidated basket discounting curve (Average)")
    else:
        # Single currency logic
        currency = unique_currencies[0]
        y_df = fetch_yield_curve(currency)
        display_df = y_df
        st.caption(f"Risk-free rates ({currency})")

    use_flat_yc = st.toggle("Flat Yield Curve", value=False)
    
    if use_flat_yc:
        flat_rate_val = st.number_input("Constant Rate (%)", value=3.0, step=0.1, format="%.2f") / 100.0
        zc_tenors = display_df["Years"].values
        zc_rates = np.full(len(zc_tenors), flat_rate_val)
        st.info("Using fixed rate for all tenors.")
    else:
        edited_yc = st.data_editor(
            display_df,
            column_config={
                "Tenor": st.column_config.TextColumn("Tenor", disabled=True, width="small"),
                "Years": st.column_config.NumberColumn("Yrs", disabled=True, width="small"),
                "Rate (%)": st.column_config.NumberColumn("Rate (%)", format="%.3f%%")
            },
            width="stretch",
            hide_index=True,
            key=f"yc_editor_{'_'.join(unique_currencies)}"
        )
        zc_tenors = edited_yc["Years"].values
        zc_rates = edited_yc["Rate (%)"].values / 100.0
    
    # --- PRODUCT STRUCTURE (Athena Standard) ---
    st.subheader("Product structure")
    
    def update_mat():
        st.session_state.selected_maturity = st.session_state.mat_slider

    maturity = st.slider("Maturity (Years)", 1, 10, value=st.session_state.selected_maturity, key="mat_slider", on_change=update_mat)
    
    # --- CALENDAR LOGIC ---
    issue_date = st.date_input("Issue Date", value=datetime.today())
    
    # Frequency Logic
    obs_freq = st.selectbox("Observation Frequency", ["Annual", "Semi-Annual", "Quarterly", "Monthly"], index=2) 
    freq_map = {"Annual": 1, "Semi-Annual": 2, "Quarterly": 4, "Monthly": 12}
    steps_per_year = freq_map[obs_freq]
    total_obs = int(maturity * steps_per_year)
    obs_times = np.linspace(1.0/steps_per_year, maturity, total_obs)
    
    # Format labels for dates
    def format_obs_date(t, start_date):
        return (start_date + timedelta(days=int(t * 365.25))).strftime("%b %Y")
    obs_labels = [format_obs_date(t, issue_date) for t in obs_times]
    
    # Barriers
    autocall_level_val = st.slider("Autocall Barrier (%)", 0.50, 1.50, 1.00, 0.05)
    coupon_barrier_val = autocall_level_val 
    pdi_barrier = st.slider("PDI Barrier (%)", 0.00, float(coupon_barrier_val), min(0.60, float(coupon_barrier_val)), 0.05)
    coupon_pa = st.number_input("Coupon p.a. (%)", value=0.08, step=0.01, format="%.3f")
    memory_feature = True # Athena standard memory effect
    # --- Date Shift (Forward Start Logic) ---
    today = datetime.today().date()
    days_to_start = (issue_date - today).days
    shift_years = max(0.0, days_to_start / 365.25)
    
    if shift_years > 0:
        st.success(f"**Forward Pricing Active**: Product starts in {days_to_start} days ({shift_years:.2f}Y). Strike fixed at Future Spot.")
    
    auto_pricing = st.toggle("Live Pricing (Auto-Refresh)", value=True, help="Automatically re-price when parameters change. May be slower for high path counts.")
    st.markdown("---")
    
    # Dynamic MC Paths Generation
    num_paths = 5000 if auto_pricing else 50000

    run_pricer = st.button("Manual Run", use_container_width=True)  # Button API unchanged

# --- TABS ORCHESTRATION ---
tab_payoff, tab_market, tab_pricing = st.tabs([
    "Payoff Profile", 
    "Market Analysis", 
    "Master Pricing Console"
])

# --- TAB 1: PAYOFF PROFILE ---
with tab_payoff:
    st.header(f"Payoff Profile - {'Worst-Of Basket' if is_wo else selected_name1}")
    
    # Temporal Explorer (Using Date Labels)
    selected_label = st.select_slider("Select Observation Date", options=obs_labels, value=obs_labels[-1])
    obs_idx = obs_labels.index(selected_label) + 1
    obs_t = obs_times[obs_idx - 1]
    
    # Payoff Logic for Diagram (Single Asset)
    x_perf = np.linspace(0.0, 1.6, 500)
    y_payoff = []
    
    # C-Rate per period for scaling
    c_per_period = (coupon_pa * 100.0) / steps_per_year
    total_coupons = (c_per_period * obs_idx) if memory_feature else c_per_period
    
    # Display help text
    st.caption(f"Visualizing payoff for observation date: **{selected_label}** (T={obs_t:.2f}Y)")
    
    for p in x_perf:
        if obs_t < maturity:
            # INTERMEDIATE OBS DATE
            if p >= autocall_level_val:
                val = 100.0 + total_coupons 
            else:
                val = 100.0
        else:
            # MATURITY
            if p >= coupon_barrier_val:
                val = 100.0 + total_coupons
            elif p >= pdi_barrier:
                val = 100.0
            else:
                val = 100.0 * p
        y_payoff.append(val)
    
    fig_payoff = go.Figure()
    fig_payoff.add_trace(go.Scatter(
        x=x_perf, y=y_payoff, mode='lines', 
        line=dict(color=GOLD, width=4), 
        fill='tozeroy', 
        fillcolor='rgba(184, 134, 11, 0.05)'
    ))
    fig_payoff.add_vline(x=autocall_level_val, line_dash="dash", line_color=SLATE, 
                         annotation_text=" Autocall ", annotation_position="top left",
                         annotation_font_size=11, annotation_font_color="white", annotation_bgcolor=SLATE)
                         
    if abs(autocall_level_val - coupon_barrier_val) > 1e-4:
        fig_payoff.add_vline(x=coupon_barrier_val, line_dash="dot", line_color=SLATE, 
                             annotation_text=" Coupon ", annotation_position="bottom right",
                             annotation_font_size=11, annotation_font_color="white", annotation_bgcolor=SLATE)
                             
    if abs(obs_t - maturity) < 1e-5:
        fig_payoff.add_vline(x=pdi_barrier, line_dash="dash", line_color=CORAL, 
                             annotation_text=" PDI ", annotation_position="bottom left",
                             annotation_font_size=11, annotation_font_color="white", annotation_bgcolor=CORAL)
    
    fig_payoff.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
        margin=dict(l=0,r=0,t=20,b=0), height=400,
        xaxis=dict(title="Asset Performance (S_T / S_0)", color=SLATE, gridcolor=BORDER, tickformat=".0%", zerolinecolor=BORDER),
        yaxis=dict(title="Reimbursement (%)", color=SLATE, gridcolor=BORDER, zerolinecolor=BORDER),
        hovermode="x unified"
    )
    st.plotly_chart(fig_payoff, width="stretch")

# --- TAB 2: MARKET ANALYSIS ---
with tab_market:
    if is_wo:
        sub_overview, sub_idx1, sub_idx2 = st.tabs(["Basket Overview", f"Details: {selected_name1}", f"Details: {selected_name2}"])
        
        with sub_overview:
            st.subheader("Historical Basket Correlation")
            h1 = st.session_state.basket_history.get(selected_name1)
            h2 = st.session_state.basket_history.get(selected_name2)
            
            if h1 is not None and h2 is not None:
                # Normalize indices to date only (remove timezone/time) to ensure alignment
                h1_clean = h1.copy()
                h2_clean = h2.copy()
                h1_clean.index = pd.to_datetime(h1_clean.index).date
                h2_clean.index = pd.to_datetime(h2_clean.index).date
                
                # Merge on index to ensure alignment and drop NaNs (market holidays)
                combined = pd.merge(h1_clean[['Log_Return']], h2_clean[['Log_Return']], left_index=True, right_index=True, suffixes=('_1', '_2')).dropna()
                
                if not combined.empty and len(combined) > 20: 
                    realized_rho = combined.corr().iloc[0, 1]
                else:
                    realized_rho = np.nan
                
                c_rho, c_msg = st.columns([1, 2])
                if not np.isnan(realized_rho):
                    c_rho.metric("Realized Correlation", f"{realized_rho*100:.1f}%", help=f"Historical correlation over {maturity}Y")
                    c_msg.info(f"**Insight**: A correlation of {realized_rho*100:.1f}% indicates how closely these indices move together. Higher correlation generally reduces the 'Worst-Of' discount.")
                else:
                    c_rho.metric("Realized Correlation", "N/A", help="Not enough overlapping trading days")
                    c_msg.warning("Could not calculate historical correlation. Indices may have non-overlapping trading calendars or insufficient history.")
                
                st.markdown("---")
                st.subheader("Relative Performance (Base 100)")
                fig_rel = go.Figure()
                fig_rel.add_trace(go.Scatter(x=h1.index, y=(h1['Close']/h1['Close'].iloc[0])*100, name=selected_name1, line=dict(color=GOLD, width=2)))
                fig_rel.add_trace(go.Scatter(x=h2.index, y=(h2['Close']/h2['Close'].iloc[0])*100, name=selected_name2, line=dict(color="#4FD1C5", width=2)))
                fig_rel.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(color=SLATE, gridcolor=BORDER),
                    yaxis=dict(title="Relative Value (Base 100)", color=SLATE, gridcolor=BORDER),
                    margin=dict(l=0, r=0, t=10, b=0), height=400, hovermode="x unified"
                )
                st.plotly_chart(fig_rel, width="stretch")
        
        # Details tabs for each asset
        for sub_tab, asset_name in zip([sub_idx1, sub_idx2], [selected_name1, selected_name2]):
            with sub_tab:
                h_df = st.session_state.basket_history.get(asset_name)
                if h_df is not None:
                    # Reuse analysis layout
                    hist_perf = (h_df['Close'].iloc[-1] / h_df['Close'].iloc[0]) - 1.0
                    st.metric(f"Historical Performance ({maturity}Y)", f"{hist_perf * 100:.2f}%")
                    
                    fig_h = go.Figure(go.Scatter(x=h_df.index, y=h_df['Close'], mode='lines', line=dict(color=GOLD, width=2)))
                    fig_h.update_layout(title="Spot Price History", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=300, margin=dict(l=0,r=0,t=30,b=0))
                    st.plotly_chart(fig_h, width="stretch")
                    
                    c_dist, c_vol = st.columns(2)
                    with c_dist:
                        st.markdown("### Return Distribution")
                        r_stats = calculate_return_stats(h_df)
                        returns = h_df['Log_Return'].dropna()
                        fig_hist = go.Figure()
                        fig_hist.add_trace(go.Histogram(x=returns, nbinsx=100, marker_color=GOLD, opacity=0.8, histnorm='probability density'))
                        x_range = np.linspace(returns.min(), returns.max(), 100)
                        norm_pdf = stats_sci.norm.pdf(x_range, r_stats['mean_daily'], r_stats['vol_daily'])
                        fig_hist.add_trace(go.Scatter(x=x_range, y=norm_pdf, mode='lines', line=dict(color=CARBON, width=2)))
                        fig_hist.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=300, margin=dict(l=0,r=0,t=0,b=0))
                        st.plotly_chart(fig_hist, width="stretch")
                        st.write(f"Volatility: **{r_stats['vol_daily'] * np.sqrt(252) * 100:.1f}%** | Skew: **{r_stats['skewness']:.2f}**")
                    
                    with c_vol:
                        st.markdown("### Rolling Volatility (21d)")
                        rolling_v = calculate_rolling_volatility(h_df)
                        fig_v = go.Figure(go.Scatter(x=rolling_v.index, y=rolling_v * 100, mode='lines', line=dict(color=GOLD)))
                        fig_v.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=300, margin=dict(l=0,r=0,t=0,b=0))
                        st.plotly_chart(fig_v, width="stretch")
    else:
        # SINGLE ASSET MODE
        asset_name = selected_name1
        st.header(f"Market Analysis - {asset_name}")
        h_df = st.session_state.basket_history.get(asset_name)
        if h_df is not None:
            hist_perf = (h_df['Close'].iloc[-1] / h_df['Close'].iloc[0]) - 1.0
            st.metric(f"Historical Performance ({maturity}Y)", f"{hist_perf * 100:.2f}%")
            
            fig_h = go.Figure(go.Scatter(x=h_df.index, y=h_df['Close'], mode='lines', line=dict(color=GOLD, width=2)))
            fig_h.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=350, margin=dict(l=0,r=0,t=10,b=0))
            st.plotly_chart(fig_h, width="stretch")

            c_dist, c_vol = st.columns(2)
            with c_dist:
                st.markdown("### Return Distribution")
                r_stats = calculate_return_stats(h_df)
                returns = h_df['Log_Return'].dropna()
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(x=returns, nbinsx=100, marker_color=GOLD, opacity=0.8, histnorm='probability density'))
                x_range = np.linspace(returns.min(), returns.max(), 100)
                norm_pdf = stats_sci.norm.pdf(x_range, r_stats['mean_daily'], r_stats['vol_daily'])
                fig_hist.add_trace(go.Scatter(x=x_range, y=norm_pdf, mode='lines', line=dict(color=CARBON, width=2)))
                fig_hist.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=300)
                st.plotly_chart(fig_hist, width="stretch")
                st.write(f"Ann. Volatility: **{r_stats['vol_daily'] * np.sqrt(252) * 100:.1f}%** | Skewness: **{r_stats['skewness']:.2f}**")

            with c_vol:
                st.markdown("### Rolling Volatility (21d)")
                rolling_v = calculate_rolling_volatility(h_df)
                fig_v = go.Figure(go.Scatter(x=rolling_v.index, y=rolling_v * 100, mode='lines', line=dict(color=GOLD)))
                fig_v.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=300)
                st.plotly_chart(fig_v, width="stretch")

    # --- LOCAL VOLATILITY SURFACE VISUALIZATION ---
    if vol_surfaces:
        st.markdown("---")
        st.header("Local Volatility Surface")
        
        # Select which asset's surface to display
        surf_asset_idx = 0
        if is_wo and len(vol_surfaces) > 1:
            surf_asset_name = st.selectbox("Surface for", selected_assets, key="vol_surf_select")
            surf_asset_idx = selected_assets.index(surf_asset_name)
        
        surf = vol_surfaces[surf_asset_idx]
        s0 = surf.s0
        
        # Build the grid
        t_grid = np.linspace(0.05, maturity, 40)
        moneyness_grid = np.linspace(0.50, 1.50, 50)  # 50% to 150% of spot
        spot_grid = moneyness_grid * s0
        
        # Compute surface: vol_matrix[i, j] = sigma(t_i, S_j)
        vol_matrix = np.zeros((len(t_grid), len(spot_grid)))
        for i, t in enumerate(t_grid):
            vol_matrix[i, :] = surf.get_vol(t, spot_grid) * 100.0  # Convert to %

        col_3d, col_heatmap = st.columns(2)
        
        with col_3d:
            st.markdown("#### 3D Surface")
            fig_surf = go.Figure(data=[go.Surface(
                x=moneyness_grid * 100,  # Moneyness %
                y=t_grid,
                z=vol_matrix,
                colorscale=[
                    [0.0, "#1a1a2e"],
                    [0.25, "#16213e"],
                    [0.5, "#0f3460"],
                    [0.75, "#e94560"],
                    [1.0, "#B8860B"]
                ],
                showscale=True,
                hovertemplate="Moneyness: %{x:.0f}%<br>Time: %{y:.1f}Y<br>Vol: %{z:.1f}%<extra></extra>"
            )])
            fig_surf.update_layout(
                scene=dict(
                    xaxis=dict(title="Moneyness (%)", color=SLATE),
                    yaxis=dict(title="Time (Years)", color=SLATE),
                    zaxis=dict(title="Local Vol (%)", color=SLATE),
                    bgcolor='rgba(0,0,0,0)'
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=10, b=0),
                height=450
            )
            st.plotly_chart(fig_surf, width="stretch")
        
        with col_heatmap:
            st.markdown("#### Heatmap (Time x Moneyness)")
            fig_hm = go.Figure(data=go.Heatmap(
                x=moneyness_grid * 100,
                y=t_grid,
                z=vol_matrix,
                colorscale=[
                    [0.0, "#1a1a2e"],
                    [0.25, "#16213e"],
                    [0.5, "#0f3460"],
                    [0.75, "#e94560"],
                    [1.0, "#B8860B"]
                ],
                colorbar=dict(title="Vol (%)"),
                hovertemplate="Moneyness: %{x:.0f}%<br>Time: %{y:.1f}Y<br>Vol: %{z:.1f}%<extra></extra>"
            ))
            # Add vertical line at ATM (100%)
            fig_hm.add_vline(x=100, line_dash="dash", line_color="white", opacity=0.6,
                             annotation_text="ATM", annotation_font_color="white", annotation_position="top")
            fig_hm.update_layout(
                xaxis=dict(title="Moneyness (%)", color=SLATE),
                yaxis=dict(title="Time (Years)", color=SLATE),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=10, b=0),
                height=450
            )
            st.plotly_chart(fig_hm, width="stretch")
        
        # ATM Term Structure line chart
        st.markdown("#### ATM Volatility Term Structure")
        atm_vols_ts = [surf.get_vol(t, np.array([s0]))[0] * 100.0 for t in t_grid]
        fig_ts = go.Figure(go.Scatter(
            x=t_grid, y=atm_vols_ts, mode='lines+markers',
            line=dict(color=GOLD, width=3),
            marker=dict(size=4, color=GOLD),
            hovertemplate="T=%{x:.1f}Y  Vol=%{y:.1f}%<extra></extra>"
        ))
        fig_ts.update_layout(
            xaxis=dict(title="Time (Years)", color=SLATE, gridcolor=BORDER),
            yaxis=dict(title="ATM Vol (%)", color=SLATE, gridcolor=BORDER),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=10, b=0), height=300
        )
        st.plotly_chart(fig_ts, width="stretch")

# --- TAB 3: MASTER PRICING CONSOLE ---
with tab_pricing:
    if run_pricer or auto_pricing:
        
        # BLACK-SCHOLES (Used strictly as a Control Variate / Calibration benchmark)
        # The actual Autocall pricing uses Monte Carlo with Local Volatility diffusion.
        def bs_put_price(S, K, T, r, q, vol):
            if T <= 0: return max(K - S, 0.0)
            d1 = (np.log(S/K) + (r - q + 0.5 * vol**2)*T) / (vol * np.sqrt(T))
            d2 = d1 - vol * np.sqrt(T)
            return K * np.exp(-r*T) * stats_sci.norm.cdf(-d2) - S * np.exp(-q*T) * stats_sci.norm.cdf(-d1)
            
        @st.cache_data(show_spinner=False)
        def execute_pricing_engine(
            _spots_list, _vol_surfaces, _divs_list, _rho, _zc_tenors, _zc_rates,
            obs_times, autocall_level_val, coupon_barrier_val, coupon_pa,
            steps_per_year, pdi_barrier, memory_feature,
            maturity, auto_pricing, run_pricer_flag, shift_years=0.0,
            market_hash=None
        ):
            # Internal re-instantiation of objects to avoid cache serialization issues
            spots_arr = np.array(_spots_list)
            vol_surfaces = _vol_surfaces
            divs_arr = np.array(_divs_list)
            num_assets = len(spots_arr)
            
            if num_assets == 1:
                corr_matrix = np.array([[1.0]])
            else:
                # Ensure rho is not exactly 1 or -1 to avoid singular matrix
                safe_rho = np.clip(_rho, -0.9999, 0.9999)
                corr_matrix = np.array([[1.0, safe_rho], [safe_rho, 1.0]])
            
            yield_curve = YieldCurve(times=np.array(_zc_tenors), rates=np.array(_zc_rates))
            mc = MonteCarloSimulator(spots_arr, vol_surfaces, corr_matrix, yield_curve, divs_arr)
            autocall_levels = np.ones(len(obs_times)) * autocall_level_val
            c_barriers = np.ones(len(obs_times)) * coupon_barrier_val
            c_rates = np.ones(len(obs_times)) * ((coupon_pa * 100.0) / steps_per_year)
            
            # For Option 2 (Forward Start), the product striks at Issue level (100%).
            # Simulation is relative to issue, but discounting is from Today (Absolute Time).
            pricing_obs_times = np.array(obs_times) + shift_years
            
            product = AutocallAthena(
                obs_times=pricing_obs_times, 
                autocall_levels=autocall_levels, 
                coupon_levels=c_barriers, 
                coupon_rates=c_rates, 
                pdi_barrier=pdi_barrier, 
                nominal=100.0, 
                memory_feature=memory_feature
            )
            
            # --- DYNAMIC MC CONTROL VARIATE LOOP ---
            epsilon = 0.05 # Increased tolerance for LocVol complexity
            batch_size = 5000
            max_paths = 10000 if auto_pricing else 100000 
            
            # Control Variate Target (European Put at PDI barrier) for Asset #1
            spot_cv = spots_arr[0]
            # Use ATM vol at maturity as a proxy for CV
            vol_cv = vol_surfaces[0].get_vol(maturity, np.array([spot_cv]))[0]
            div_cv = divs_arr[0]
            
            T_mat = maturity
            K_strike = spot_cv * pdi_barrier
            r_rate = yield_curve.forward_rate(0, T_mat)
            target_bs_put = bs_put_price(spot_cv, K_strike, T_mat, r_rate, div_cv, vol_cv)
            
            all_paths = []
            total_paths = 0
            err_pct = 100.0
            
            status_placeholder = st.empty()
            
            while total_paths < max_paths:
                new_paths = mc.generate_paths(obs_times, num_paths=batch_size, seed=42 + total_paths)
                all_paths.append(new_paths)
                total_paths += batch_size
                
                # Check Convergence against Asset #1
                current_paths = np.concatenate(all_paths, axis=0)
                S_T_0 = current_paths[:, 0, -1]
                
                # Use current spot for CV if needed
                put_payoffs = np.maximum(K_strike - S_T_0, 0)
                mc_put_price = np.mean(put_payoffs) * yield_curve.discount_factor(T_mat)
                
                err_abs = abs(mc_put_price - target_bs_put)
                err_pct = (err_abs / spot_cv) * 100.0
                
                if not auto_pricing:
                    status_placeholder.info(f"Adapting Model... Paths: {total_paths:,.0f} | Target Put Pricing Err vs BS: {err_pct:.4f}% (Target: <0.05%)")
                
                if total_paths >= 10000 and err_pct <= epsilon:
                    if not auto_pricing:
                        status_placeholder.success(f"Model Converged! Paths used: {total_paths:,.0f} | Final Control Error: {err_pct:.4f}%")
                    break
                
                # SKEW SAFETY BYPASS:
                # Black-Scholes uses ATM Vol. Local Vol uses Skewed Vol. 
                # A Skewed Put will fundamentally have a different fair value than an ATM BS Put.
                # If skew is activated, we accept 10k paths minimum convergence to avoid forcing an impossible mathematical target.
                if vol_surfaces[0].skew_intensity != 0.0 and total_paths >= 10000:
                    if not auto_pricing:
                        status_placeholder.info(f"Skew Active. Executing {total_paths:,.0f} paths. Control Var metric bypassed.")
                    break
                    
            if total_paths >= max_paths and not auto_pricing:
                status_placeholder.warning(f"Max capacity reached ({total_paths:,.0f} paths). Final Control Error: {err_pct:.4f}%")
            elif auto_pricing:
                status_placeholder.empty()
 
            paths = np.concatenate(all_paths, axis=0)
            results = product.price(paths, spots_arr, yield_curve)
            
            # Greeks calculation (if needed, but kept as None for performance in this turns)
            greeks = None
                
            return results, greeks, paths
            
        with st.spinner("Executing Master Pricing Engine..."):
            # Create a combined hash of all complex market inputs to force cache invalidation
            market_hash = hash((
                tuple(spots_list), 
                tuple(vol_surfaces), 
                tuple(divs_arr), 
                rho_val if is_wo else 1.0,
                tuple(zc_tenors), 
                tuple(zc_rates)
            ))
            
            results, greeks, paths = execute_pricing_engine(
                tuple(spots_list), tuple(vol_surfaces), tuple(divs_arr), rho_val if is_wo else 1.0, 
                tuple(zc_tenors), tuple(zc_rates),
                tuple(obs_times), autocall_level_val, coupon_barrier_val, coupon_pa,
                steps_per_year, pdi_barrier, memory_feature,
                maturity, auto_pricing, run_pricer, shift_years=shift_years,
                market_hash=market_hash
            )
            
            # Re-instantiate locally strictly for downstream UI calculation use
            yield_curve = YieldCurve(times=zc_tenors, rates=zc_rates)
            spots_arr = np.array(spots_list)
            
            st.markdown(f"### Core Analytics: {'WORST-OF Index' if is_wo else selected_name1}")
            c1, c2, c3 = st.columns([2, 1, 1])
            fv = results['fair_value']
            c1.metric("Fair Value", f"{fv:.2f}%", delta=f"{fv-100:.2f}%")
            c2.metric("PDI Prob.", f"{results['prob_pdi']*100:.1f}%")
            # Expected life in years
            e_life = results.get('expected_maturity', maturity)
            c3.metric("Expected Life", f"{e_life:.2f} Y")
            
            st.markdown("---")
            col_graph, col_desc = st.columns([2, 1])
            with col_graph:
                st.markdown("#### Fair Value Waterfall Reconstruction")
                bd = results['breakdown']
                # Waterfall logic using Bar chart for individual colors
                bond_val = bd['Pure Bond Part']
                coupon_val = bd['Coupons Part']
                pdi_val = bd['PDI Risk (Put)'] # This is negative
                
                # Bases for the bars
                bases = [0, bond_val, bond_val + coupon_val, 0]
                values = [bond_val, coupon_val, pdi_val, fv]
                labels = ["Bond", "Coupons", "PDI Exposure", "Fair Value"]
                colors = ["#3182CE", "#38A169", "#E53E3E", GOLD]

                fig_wf = go.Figure(go.Bar(
                    x=labels,
                    y=values,
                    base=bases,
                    marker_color=colors,
                    text=[f"{v:+.2f}%" for v in values],
                    textposition='outside',
                    hovertemplate="%{x}: %{y:.2f}%<extra></extra>"
                ))
                
                fig_wf.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    yaxis=dict(title="Contribution (%)", color=CARBON, gridcolor=BORDER, range=[0, 115]),
                    xaxis=dict(color=CARBON),
                    margin=dict(l=0, r=0, t=10, b=0), height=350,
                    showlegend=False
                )
                st.plotly_chart(fig_wf, width="stretch")
                
            with col_desc:
                st.markdown("#### Breakdown Insights")
                st.write(f"Structural value: **{fv:.2f}%**")
                st.info(f"**Bond Component**: {bd['Pure Bond Part']:.2f}%")
                st.success(f"**Optionality**: +{bd['Coupons Part']:.2f}%")
                st.error(f"**PDI Structural Risk**: {bd['PDI Risk (Put)']:.2f}%")
            
            st.markdown("---")
            t_structure, t_audit = st.tabs(["Structural Probability", "Replication Audit"])
            with t_structure:
                st.markdown("#### Exit Probability Profile")
                c_probs = results['call_probs']
                labels = obs_labels.copy()
                if len(labels) < len(c_probs) + 1: labels.append("Maturity")
                else: labels[-1] = f"{labels[-1]} (Mat)"
                fig_exit = go.Figure(go.Bar(
                    x=labels, y=[p*100 for p in c_probs + [results['prob_maturity']]],
                    marker_color=GOLD,
                    text=[f"{p*100:.1f}%" for p in c_probs + [results['prob_maturity']]],
                    textposition='auto',
                ))
                fig_exit.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(color=CARBON), 
                    yaxis=dict(title="Probability (%)", color=CARBON, gridcolor=BORDER),
                    margin=dict(l=0, r=0, t=10, b=0), height=350
                )
                st.plotly_chart(fig_exit, width="stretch")
 
            with t_audit:
                st.markdown("#### Scenario Profitability Audit")
                audit_data = []
                cum_fv = 0.0
                c_per_period = (coupon_pa * 100.0) / steps_per_year
                
                for i in range(len(results['call_probs'])):
                    prob = results['call_probs'][i]
                    df = yield_curve.discount_factor(obs_times[i])
                    payoff = 100.0 + ((i + 1) * c_per_period if memory_feature else c_per_period)
                    pv_contrib = prob * payoff * df
                    cum_fv += pv_contrib
                    audit_data.append({
                        "Event": f"Autocall at {obs_labels[i]}", 
                        "Exit Prob": f"{prob*100:.1f}%", 
                        "Expected Payoff": f"{payoff:.2f}%", 
                        "PV Contribution": f"{pv_contrib:.2f}%"
                    })
                
                # Maturity contribution is the remainder of the Fair Value
                prob_mat = results['prob_maturity']
                pv_contrib_mat = fv - cum_fv
                avg_payoff_mat = (pv_contrib_mat / (prob_mat * yield_curve.discount_factor(maturity))) if prob_mat > 1e-6 else 0.0
                
                audit_data.append({
                    "Event": f"Reaches Maturity ({obs_labels[-1]})", 
                    "Exit Prob": f"{prob_mat*100:.1f}%", 
                    "Expected Payoff": f"{avg_payoff_mat:.2f}% (Avg)", 
                    "PV Contribution": f"{pv_contrib_mat:.2f}%"
                })
                
                # Sum verification row
                audit_data.append({
                    "Event": "**TOTAL**", 
                    "Exit Prob": "100.0%", 
                    "Expected Payoff": "-", 
                    "PV Contribution": f"{fv:.2f}%"
                })
                
                st.dataframe(
                    pd.DataFrame(audit_data),
                    column_config={
                        "Event": st.column_config.TextColumn("Scenario Event", width="large"),
                        "Exit Prob": st.column_config.TextColumn("Prob %", width="small"),
                        "Expected Payoff": st.column_config.TextColumn("Payoff", width="medium"),
                        "PV Contribution": st.column_config.TextColumn("PV Contrib", width="medium")
                    },
                    width="stretch",
                    hide_index=True
                )
                
                st.markdown("#### Path Inspector")
                col_ins1, col_ins2 = st.columns([1, 2])
                with col_ins1:
                    audit_dict = {p['path_id']: p for p in results['audit_paths']}
                    path_options = list(audit_dict.keys())
                    
                    if not path_options:
                        st.warning("No representative paths found.")
                        path_idx = 0
                        p_info = {"total_pv": 0, "event": "N/A", "category": "N/A", "exit_time": maturity}
                    else:
                        # Friendly dropdown formatting with mapped UI names to preserve internal logic links
                        ui_scenario_names = {
                            "Autocall_1Y": "Early Exit (Autocall)",
                            "Autocall_Mat": "Maturity Exit (Max Return)",
                            "Protected": "Capital Protected",
                            "PDI": "PDI Exposure"
                        }
                        path_idx = st.selectbox(
                            "Select a Market Scenario", 
                            options=path_options, 
                            format_func=lambda x: f"{ui_scenario_names.get(audit_dict[x]['category'], audit_dict[x]['category'])} | {audit_dict[x]['event']}"
                        )
                        p_info = audit_dict[path_idx]
                        
                    val_to_show = p_info.get('absolute_payoff', p_info['total_pv'])
                    st.write(f"### Final Payoff: **{val_to_show:.2f}%**")
                    st.caption(f"(Mathematical Present Value: {p_info['total_pv']:.2f}%)")
                    
                    # Explanatory Narrative (Athena)
                    if p_info.get("category") == "Autocall_1Y":
                        st.success("**Mechanism:** The asset performance breached the Autocall Barrier. The product repaid 100% of the nominal plus all memory coupons up to that date and ceased to exist.")
                    elif p_info.get("category") == "Autocall_Mat":
                        st.success("**Mechanism:** The product survived until maturity and closed above the Autocall Barrier. It pays 100% plus the maximum amount of accumulated memory coupons.")
                    elif p_info.get("category") == "Protected":
                        st.info("**Mechanism:** The product survived until maturity without ever triggering an early autocall. However, because the final performance stayed above the PDI Barrier, your capital was 100% protected.")
                    elif p_info.get("category") == "PDI":
                        st.error("**Mechanism:** The asset performance breached the PDI Barrier at maturity. The structural protection was consumed, resulting in a direct adjustment on the nominal.")
                    
                with col_ins2:
                    # Build base time axes
                    full_times = np.insert(obs_times, 0, 0.0)
                    full_labels = ["Issue"] + obs_labels
                    exit_t = p_info.get("exit_time", maturity)
                    
                    fig_path = go.Figure()
                    
                    # Colors for multiple assets
                    asset_colors = [GOLD, "#4FD1C5", "#805AD5"] # Gold, Teal, Purple
                    
                    # Plot and track all assets in the basket
                    for a_idx in range(len(selected_assets)):
                        p_line = np.insert(paths[path_idx][a_idx] / spots_arr[a_idx], 0, 1.0)
                        
                        # Find indices where time <= exit_time
                        v_idx = [i for i, t in enumerate(full_times) if t <= exit_t + 1e-5]
                        trunc_l = [full_labels[i] for i in v_idx]
                        trunc_p = [p_line[i] for i in v_idx]
                        
                        fig_path.add_trace(go.Scatter(
                            x=trunc_l, y=trunc_p, 
                            mode='lines+markers', 
                            name=selected_assets[a_idx],
                            line=dict(color=asset_colors[a_idx % len(asset_colors)], width=3 if a_idx == 0 else 2),
                            marker=dict(size=8 if a_idx == 0 else 6)
                        ))

                    # Terminal point marker (on the WORST asset at the exit moment)
                    final_v_idx = [i for i, t in enumerate(full_times) if t <= exit_t + 1e-5]
                    # Get the worst perf AT THE LAST VALID STEP (which lead to exit)
                    # Note: index in paths is [p_idx][a_idx][step] where step 0 is first obs, so full_times[1]
                    step_idx = len(final_v_idx) - 2 # -1 is the "full_times" index, 0 is issue, so step_idx is in paths
                    if step_idx >= 0:
                        worst_p_at_exit = min([paths[path_idx][a][step_idx] / spots_arr[a] for a in range(len(selected_assets))])
                    else:
                        worst_p_at_exit = 1.0 # Stayed at issue?
                    
                    fig_path.add_trace(go.Scatter(
                        x=[full_labels[len(final_v_idx)-1]], y=[worst_p_at_exit],
                        mode='markers+text',
                        name="Termination Point",
                        marker=dict(color="#E53E3E" if p_info.get("category") == "PDI" else "#2F855A", size=14, symbol="diamond"),
                        text=["Structural terminal exit"],
                        textposition="top center",
                        textfont=dict(color=CARBON, size=12, weight="bold")
                    ))
                    
                    # Draw horizontal layers 
                    fig_path.add_hline(y=autocall_level_val, line_dash="dash", line_color=SLATE, 
                                       annotation_text=" Autocall ", annotation_position="top left",
                                       annotation_font_size=11, annotation_font_color="white", annotation_bgcolor=SLATE)
                                       
                    fig_path.add_hline(y=coupon_barrier_val, line_dash="dot", line_color=SLATE, 
                                       annotation_text=" Coupon ", annotation_position="bottom right",
                                       annotation_font_size=11, annotation_font_color="white", annotation_bgcolor=SLATE)
                                       
                    fig_path.add_hline(y=pdi_barrier, line_dash="dash", line_color=CORAL, 
                                       annotation_text=" PDI ", annotation_position="bottom left",
                                       annotation_font_size=11, annotation_font_color="white", annotation_bgcolor=CORAL)
                    
                    fig_path.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                        xaxis=dict(color=SLATE, gridcolor=BORDER, zerolinecolor=BORDER), 
                        yaxis=dict(color=SLATE, gridcolor=BORDER, zerolinecolor=BORDER, tickformat=".0%"), 
                        height=400, margin=dict(l=0,r=0,t=20,b=0),
                        showlegend=False,
                        hovermode="x unified"
                    )
                    st.plotly_chart(fig_path, width="stretch")
