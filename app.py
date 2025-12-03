import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import plotly.express as px
import plotly.graph_objects as go
from pypfopt import expected_returns, risk_models, HRPOpt, DiscreteAllocation, get_latest_prices, EfficientFrontier , objective_functions
# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="QuantAlloc: AI Portfolio Optimizer",
    page_icon="🧠",
    layout="wide"
)

# ==========================================
# 1.1 BLOOMBERG THEME STYLING
# ==========================================
st.markdown("""
    <style>
    /* Main Background - Pitch Black */
    .stApp {
        background-color: #000000;
        color: #ff9900; /* Bloomberg Orange */
    }

    /* Sidebar Background - Dark Grey */
    section[data-testid="stSidebar"] {
        background-color: #121212;
    }

    /* FORCE SIDEBAR TEXT COLOR (Labels, Text, Paragraphs) */
    section[data-testid="stSidebar"] label, 
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] div[data-testid="stMarkdown"] {
        color: #ff9900 !important;
        font-weight: bold;
    }

    /* Headers - Bloomberg Orange */
    h1, h2, h3, h4, h5, h6 {
        color: #ff9900 !important;
        font-family: 'Courier New', Courier, monospace; /* Terminal font feel */
    }

    /* Metric Values - Terminal Green */
    div[data-testid="stMetricValue"] {
        color: #00ff00 !important;
    }
    
    /* Metric Labels - NUCLEAR OPTION to Fix Faint Gray */
    [data-testid="stMetricLabel"] {
        color: #ff9900 !important;
    }
    [data-testid="stMetricLabel"] * {
        /* This targets every single element inside the label container */
        color: #ff9900 !important;
        font-weight: bold;
        opacity: 1 !important; /* Force full visibility */
    }

    /* Buttons - Orange Border */
    div.stButton > button {
        background-color: #000000;
        color: #ff9900;
        border: 1px solid #ff9900;
    }
    div.stButton > button:hover {
        background-color: #ff9000;
        color: #000000;
    }

    /* Dataframes/Tables - Dark Mode */
    div[data-testid="stDataFrame"] {
        background-color: #121212;
    }
    /* CUSTOM STRATEGY BOXES */
    .bloomberg-box-blue {
        background-color: #001a33; /* Dark Navy */
        border: 1px solid #0099cc;
        color: #33ccff;
        padding: 15px;
        border-radius: 5px;
        font-family: 'Courier New', Courier, monospace;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .bloomberg-box-orange {
        background-color: #331a00; /* Dark Burnt Orange */
        border: 1px solid #ff9900;
        color: #ffcc00;
        padding: 15px;
        border-radius: 5px;
        font-family: 'Courier New', Courier, monospace;
        font-weight: bold;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. HELPER FUNCTIONS (Cached)
# ==========================================

@st.cache_data(ttl=86400)  # Cache for 24 hours
def get_sp500_tickers():
    """Scrapes the S&P 500 tickers from Wikipedia"""
    # This URL is a community-maintained list of S&P 500 companies
    url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
    fallback_tickers = [
        "AAPL", "MSFT", "GOOGL"
    ]

    try:
        df = pd.read_csv(url)
        tickers = df['Symbol'].tolist()
        tickers = [t.replace('.', '-') for t in tickers]
        return sorted(tickers)
    except Exception as e:
        return sorted(fallback_tickers)


@st.cache_data(ttl=3600)  # Cache price data for 1 hour
def fetch_data(tickers, start, end):
    """
    Fetches Adjusted Close prices.
    Handles the complexities of yfinance MultiIndex returns.
    """
    # Always include SPY for benchmarking
    req_tickers = tickers.copy()
    if "SPY" not in req_tickers:
        req_tickers.append("SPY")

    try:
        # Download data
        data = yf.download(req_tickers, start=start, end=end, progress=False)

        # 1. Handle 'Adj Close' vs 'Close' preference
        if 'Adj Close' in data.columns:
            prices = data['Adj Close']
        elif 'Close' in data.columns:
            prices = data['Close']
        else:
            # Fallback if structure is weird (e.g. single ticker flattened)
            prices = data

        # 2. Ensure it's a DataFrame, not a Series (happens if only 1 ticker downloaded)
        if isinstance(prices, pd.Series):
            prices = prices.to_frame()
            prices.columns = req_tickers  # Rename column to ticker name

        # 3. Clean Data
        # Forward fill missing values (hold previous price), then drop remaining NaNs
        prices = prices.ffill().dropna()

        return prices
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()


def run_optimization(prices):
    """Runs BOTH HRP and Mean-Variance Optimization (Max Sharpe)"""

    prices_cleaned = prices.ffill()
    returns = prices_cleaned.pct_change().dropna()

    valid_assets = returns.columns[returns.std() > 1e-6]
    if len(valid_assets) < 2:
        raise ValueError("Not enough valid assets (need > 0 variance).")

    returns = returns[valid_assets]
    prices_cleaned = prices_cleaned[valid_assets]

    # Strategy 1: HRP (Machine Learning)
    hrp = HRPOpt(returns=returns)
    hrp_weights = hrp.optimize()
    hrp_perf = hrp.portfolio_performance(verbose=False)

    # Strategy 2: MVO (Max Sharpe)
    mu = expected_returns.mean_historical_return(prices_cleaned)
    S = risk_models.sample_cov(prices_cleaned)
    ef = EfficientFrontier(mu, S)
    # Add L2 regularization to prevent weights going to 100% on one asset (overfitting)
    # FIX: Use built-in objective function to avoid numpy/cvxpy version conflicts
    ef.add_objective(objective_functions.L2_reg, gamma=0.1)

    try:
        mvo_weights = ef.max_sharpe()
    except:
        # Fallback if Max Sharpe fails (e.g., negative returns) -> Min Volatility
        mvo_weights = ef.min_volatility()

    mvo_perf = ef.portfolio_performance(verbose=False)
    return hrp_weights, hrp_perf, mvo_weights, mvo_perf, returns

# ==========================================
# 3. SIDEBAR INPUTS
# ==========================================
st.sidebar.header("⚙️ Strategy Settings")

# --- Ticker Selector ---
with st.sidebar:
    with st.spinner("Loading Tickers..."):
        available_tickers = get_sp500_tickers()

desired_defaults = ["AAPL", "MSFT", "AMZN", "GOOGL"]
valid_defaults = [t for t in desired_defaults if t in available_tickers]
if not valid_defaults:
    valid_defaults = available_tickers[:5]

selected_tickers = st.sidebar.multiselect(
    "Select Assets (S&P 500)",
    options=available_tickers,
    default=valid_defaults
)

custom_ticker_input = st.sidebar.text_input("➕ Add Custom Ticker (e.g. BTC-USD)", value="")
if custom_ticker_input:
    custom_tickers = [t.strip().upper() for t in custom_ticker_input.split(",") if t.strip()]
    selected_tickers.extend(custom_tickers)
    selected_tickers = list(set(selected_tickers))
st.sidebar.caption(f"Selected: {len(selected_tickers)} assets")

# --- NEW: Strategy Selector ---
strategy_mode = st.sidebar.selectbox(
    "Select Optimization Strategy",
    options=["Compare Both (Recommended)", "Hierarchical Risk Parity (HRP)", "Max Sharpe (MVO)"],
    index=0
)

investment_amount = st.sidebar.number_input("Investment Capital ($)", min_value=1000, value=10000, step=500)

col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Start Date", value=dt.date(2018, 1, 1))
with col2:
    end_date = st.date_input("End Date", value=dt.date.today())

run_btn = st.sidebar.button("🚀 Build Portfolio", type="primary")

# ==========================================
# 4. MAIN DASHBOARD LOGIC
# ==========================================
st.title("🧠 QuantAlloc: ML-Driven Portfolio Optimization")

if run_btn:
    if len(selected_tickers) < 2:
        st.error("⚠️ Please select at least 2 assets to optimize.")
    else:
        with st.spinner("Fetching market data & running simulations..."):
            try:
                # FETCH DATA
                raw_data = fetch_data(selected_tickers, start_date, end_date)

                if raw_data.empty:
                    st.error("No data found.")
                    st.stop()

                if "SPY" in raw_data.columns:
                    spy_data = raw_data["SPY"]
                    user_tickers = [t for t in selected_tickers if t in raw_data.columns and t != "SPY"]
                    if len(user_tickers) < 2:
                        st.error("Not enough valid data returned.")
                        st.stop()
                    portfolio_data = raw_data[user_tickers]
                else:
                    spy_data = None
                    portfolio_data = raw_data

                # OPTIMIZATION
                hrp_weights, hrp_perf, mvo_weights, mvo_perf, returns = run_optimization(portfolio_data)

                clean_hrp = hrp_weights
                clean_mvo = dict(mvo_weights)
                latest_prices = get_latest_prices(portfolio_data)

            except Exception as e:
                st.error(f"Optimization Failed: {e}")
                st.stop()

        # ==========================================
        # 5. RESULTS DISPLAY (CONDITIONAL)
        # ==========================================

        show_hrp = "HRP" in strategy_mode or "Compare" in strategy_mode
        show_mvo = "MVO" in strategy_mode or "Compare" in strategy_mode

        # A. Metrics Section
        st.subheader("📊 Performance Metrics")
        col_metrics = st.columns(2) if (show_hrp and show_mvo) else st.columns(1)

        if show_hrp:
            with col_metrics[0]:
                st.markdown("""
                                <div class="bloomberg-box-blue">
                                    🤖 Hierarchical Risk Parity strategy (HRP)
                                </div>
                                """, unsafe_allow_html=True)
                c1, c2, c3 = st.columns(3)
                c1.metric("Return", f"{hrp_perf[0]:.1%}")
                c2.metric("Volatility", f"{hrp_perf[1]:.1%}")
                c3.metric("Sharpe", f"{hrp_perf[2]:.2f}")

        if show_mvo:
            idx = 1 if (show_hrp and show_mvo) else 0
            with col_metrics[idx]:
                st.markdown("""
                                <div class="bloomberg-box-orange">
                                    ⚡ Max Sharpe Strategy (MVO)
                                </div>
                                """, unsafe_allow_html=True)
                c1, c2, c3 = st.columns(3)
                c1.metric("Return", f"{mvo_perf[0]:.1%}")
                c2.metric("Volatility", f"{mvo_perf[1]:.1%}")
                c3.metric("Sharpe", f"{mvo_perf[2]:.2f}")

        st.divider()

        # B. Allocation Visuals
        st.subheader("🎨 Asset Allocation")
        col_charts = st.columns(2) if (show_hrp and show_mvo) else st.columns(1)

        if show_hrp:
            with col_charts[0]:
                st.markdown("**HRP Weights**")
                df_hrp = pd.DataFrame(list(clean_hrp.items()), columns=['Asset', 'Weight'])
                df_hrp = df_hrp[df_hrp['Weight'] > 0.001]
                fig_hrp = px.pie(df_hrp, values='Weight', names='Asset', hole=0.4,
                                 color_discrete_sequence=px.colors.qualitative.Pastel)
                fig_hrp.update_layout(template="plotly_dark",
                                      paper_bgcolor="rgba(0,0,0,0)",
                                      legend=dict(font=dict(color="#ff9900")))
                st.plotly_chart(fig_hrp, use_container_width=True)

                # HRP Buying Guide
                with st.expander("🛒 View HRP Buying Guide"):
                    da = DiscreteAllocation(clean_hrp, latest_prices, total_portfolio_value=investment_amount)
                    allocation, leftover = da.greedy_portfolio()
                    buy_data = [{"Ticker": t, "Shares": s, "Price": f"${latest_prices[t]:.2f}",
                                 "Cost": f"${s * latest_prices[t]:.2f}"} for t, s in allocation.items()]
                    buy_df=pd.DataFrame(buy_data)
                    # REPLACEMENT: Plotly Table for Full Color Control
                    # Dynamic Height Calculation to remove gap
                    row_height = 30
                    header_height = 40
                    num_rows = len(buy_df)
                    table_height = header_height + (num_rows * row_height) + 10  # small buffer

                    fig_table = go.Figure(data=[go.Table(
                        header=dict(values=list(buy_df.columns),
                                    fill_color='#ff9900',  # Bloomberg Orange Header
                                    font=dict(color='black', size=14, family="Courier New"),
                                    align='center'),
                        cells=dict(values=[buy_df[k].tolist() for k in buy_df.columns],
                                   fill_color='#121212',  # Dark Grey Cells
                                   font=dict(color='#ff9900', size=12, family="Courier New"),  # Orange text
                                   align='center',
                                   height=row_height))
                    ])

                    fig_table.update_layout(
                        margin=dict(l=0, r=0, t=0, b=0),
                        height=table_height,  # Dynamic Height
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)"
                    )
                    st.plotly_chart(fig_table, use_container_width=True)
                    st.write(f"Cash Remaining: ${leftover:.2f}")

        if show_mvo:
            idx = 1 if (show_hrp and show_mvo) else 0
            with col_charts[idx]:
                st.markdown("**MVO Weights**")
                df_mvo = pd.DataFrame(list(clean_mvo.items()), columns=['Asset', 'Weight'])
                df_mvo = df_mvo[df_mvo['Weight'] > 0.001]
                fig_mvo = px.pie(df_mvo, values='Weight', names='Asset', hole=0.4,
                                 color_discrete_sequence=px.colors.qualitative.Vivid)
                fig_mvo.update_layout(template="plotly_dark",
                                      paper_bgcolor="rgba(0,0,0,0)",
                                      legend=dict(font=dict(color="#ff9900")))
                st.plotly_chart(fig_mvo, use_container_width=True)

                # MVO Buying Guide
                with st.expander("🛒 View MVO Buying Guide"):
                    da = DiscreteAllocation(clean_mvo, latest_prices, total_portfolio_value=investment_amount)
                    allocation, leftover = da.greedy_portfolio()
                    buy_data = [{"Ticker": t, "Shares": s, "Price": f"${latest_prices[t]:.2f}",
                                 "Cost": f"${s * latest_prices[t]:.2f}"} for t, s in allocation.items()]
                    buy_df=pd.DataFrame(buy_data)
                    # REPLACEMENT: Plotly Table for Full Color Control
                    # Dynamic Height Calculation to remove gap
                    row_height = 30
                    header_height = 40
                    num_rows = len(buy_df)
                    table_height = header_height + (num_rows * row_height) + 10  # small buffer

                    fig_table = go.Figure(data=[go.Table(
                        header=dict(values=list(buy_df.columns),
                                    fill_color='#ff9900',  # Bloomberg Orange Header
                                    font=dict(color='black', size=14, family="Courier New"),
                                    align='center'),
                        cells=dict(values=[buy_df[k].tolist() for k in buy_df.columns],
                                   fill_color='#121212',  # Dark Grey Cells
                                   font=dict(color='#ff9900', size=12, family="Courier New"),  # Orange text
                                   align='center',
                                   height=row_height))
                    ])

                    fig_table.update_layout(
                        margin=dict(l=0, r=0, t=0, b=0),
                        height=table_height,  # Dynamic Height
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)"
                    )
                    st.plotly_chart(fig_table, use_container_width=True)
                    st.write(f"Cash Remaining: ${leftover:.2f}")

        # C. Efficient Frontier
        st.divider()
        st.subheader("📉 Efficient Frontier Analysis")

        # ... (Monte Carlo simulation logic same as before) ...
        with st.spinner("Running Monte Carlo Simulation..."):
            n_samples = 1000
            w = np.random.dirichlet(np.ones(len(portfolio_data.columns)), n_samples)
            rets = returns.mean()
            cov = returns.cov()
            port_rets = np.sum(w * rets.values, axis=1) * 252
            port_vol = np.sqrt(np.einsum('ij,jk,ik->i', w, cov.values * 252, w))
            port_sharpe = port_rets / port_vol
            sim_df = pd.DataFrame({'Volatility': port_vol, 'Return': port_rets, 'Sharpe': port_sharpe})

            fig = px.scatter(sim_df, x='Volatility', y='Return', color='Sharpe', color_continuous_scale='Plasma',
                             title="Monte Carlo Simulation")

            # Add Stars based on selection
            if show_hrp:
                fig.add_trace(go.Scatter(x=[hrp_perf[1]], y=[hrp_perf[0]], mode='markers+text',
                                         marker=dict(symbol='star', size=20, color='blue'), name='HRP', text=['HRP'],
                                         textposition="top center"))
            if show_mvo:
                fig.add_trace(go.Scatter(x=[mvo_perf[1]], y=[mvo_perf[0]], mode='markers+text',
                                         marker=dict(symbol='star', size=20, color='red'), name='MVO', text=['MVO'],
                                         textposition="bottom center"))
            fig.update_layout(
                template="plotly_dark",
                title_font_color="#ff9900",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Courier New, monospace", color="#ff9900"),
                legend_font_color="#ff9900"
            )
            st.plotly_chart(fig, use_container_width=True)

        # D. Backtest (Only show if comparing or specific selected)
        # ... (Backtest logic mostly same, just wrap add_trace in if show_hrp / if show_mvo) ...
        st.divider()
        st.subheader("📉 Historical Backtest")

        if spy_data is not None:
            weight_series_hrp = pd.Series(clean_hrp).reindex(returns.columns).fillna(0)
            hrp_cum_ret = (1 + (returns * weight_series_hrp).sum(axis=1)).cumprod()

            weight_series_mvo = pd.Series(clean_mvo).reindex(returns.columns).fillna(0)
            mvo_cum_ret = (1 + (returns * weight_series_mvo).sum(axis=1)).cumprod()

            spy_cum_ret = (1 + spy_data.pct_change().dropna()).cumprod()
            common_idx = hrp_cum_ret.index.intersection(spy_cum_ret.index).intersection(mvo_cum_ret.index)

            hrp_value = hrp_cum_ret.loc[common_idx] * investment_amount
            mvo_value = mvo_cum_ret.loc[common_idx] * investment_amount
            spy_value = spy_cum_ret.loc[common_idx] * investment_amount
            fig_perf = go.Figure()
            if show_hrp:
                fig_perf.add_trace(
                    go.Scatter(x=hrp_cum_ret.loc[common_idx].index, y=hrp_cum_ret.loc[common_idx] * investment_amount,
                               mode='lines', name='HRP (AI)', line=dict(color='#00CC96')))
            if show_mvo:
                fig_perf.add_trace(
                    go.Scatter(x=mvo_cum_ret.loc[common_idx].index, y=mvo_cum_ret.loc[common_idx] * investment_amount,
                               mode='lines', name='MVO (Classic)', line=dict(color='#EF553B')))
            fig_perf.add_trace(
                go.Scatter(x=spy_cum_ret.loc[common_idx].index, y=spy_cum_ret.loc[common_idx] * investment_amount,
                           mode='lines', name='S&P 500', line=dict(color='gray', dash='dash')))

            fig_perf.update_layout(
                title="Portfolio Value Over Time ($)",
                xaxis_title="Date",
                yaxis_title="Value ($)",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                title_font_color="#ff9900",
                font=dict(family="Courier New, monospace", color="#ff9900"),
                legend_font_color="#ff9900"
            )
            st.plotly_chart(fig_perf, use_container_width=True)
            # 7. Final Winner Declaration
            final_hrp = hrp_value.iloc[-1]
            final_mvo = mvo_value.iloc[-1]
            final_spy = spy_value.iloc[-1]


            # Sort to find the winner
            results = {"Hierarchical Risk Parity (HRP)": final_hrp, "Max Sharpe (MVO)": final_mvo, "S&P 500": final_spy}
            if (show_hrp and show_mvo):
                winner = max(results, key=results.get)
                winnings = results[winner] - investment_amount

                st.success(
                    f"🏆 **Winner:** {winner} with a final value of **\${results[winner]:.2f}** (Profit: \${winnings:.2f})")

                # Add context if HRP lost to MVO but had lower volatility
                if winner == "Max Sharpe (MVO)" and hrp_perf[1] < mvo_perf[1]:
                    st.info(
                        f"ℹ️ **Note:** MVO had higher returns, but HRP had lower volatility ({hrp_perf[1]:.1%} vs {mvo_perf[1]:.1%}). This makes HRP a safer ride.")
            elif show_hrp:
                if final_hrp>final_spy:
                    profit_diff=final_hrp-final_spy
                    st.success(f"**Winner:** The HRP strategy beat the market by **\${profit_diff:,.2f}**!")
                else:
                    st.warning(f"**Result:** The S&P 500 beat HRP strategy by **\${abs(profit_diff):,.2f}**.")
            else:
                if final_mvo > final_spy:
                    profit_diff_m = final_mvo - final_spy
                    st.success(f"**Winner:** The MVO strategy beat the market by **\${profit_diff_m:,.2f}**!")
                else:
                    st.warning(f"**Result:** The S&P 500 beat MVO strategy by **\${abs(profit_diff_m):,.2f}**.")
        else:
            st.warning("SPY data not available for benchmark comparison.")
