import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dotenv import load_dotenv
from supabase import create_client, Client
import os
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import enhanced modules
from enhanced_strategy_engine import EnhancedStrategyEngine
from ml_engine import PatternRecognitionEngine
from risk_models import AdvancedRiskManager, analyze_symbol_risk
from data_collector import fetch_data_from_supabase
from config import TOTAL_DAYS, TRAINING_DAYS, MARKET_ENVIRONMENTS

st.set_page_config(
    page_title="🧠 Enhanced Cognitive Trade Agent", 
    page_icon="🤖", 
    layout="wide",
    initial_sidebar_state="expanded"
)

load_dotenv()
supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# Initialize enhanced components
@st.cache_resource
def initialize_enhanced_components():
    """Enhanced bileşenleri başlat."""
    enhanced_engine = EnhancedStrategyEngine()
    ml_engine = PatternRecognitionEngine()
    risk_manager = AdvancedRiskManager()
    
    return enhanced_engine, ml_engine, risk_manager

enhanced_engine, ml_engine, risk_manager = initialize_enhanced_components()

@st.cache_data(ttl=600)
def load_enhanced_results():
    """Enhanced analiz sonuçlarını yükle."""
    try:
        response = supabase.table('enhanced_backtest_results').select('*').execute()
        if response.data:
            return pd.DataFrame(response.data)
        else:
            return pd.DataFrame()
    except:
        return pd.DataFrame()

@st.cache_data(ttl=300)
def load_market_data_enhanced(symbol: str, timeframe: str):
    """Market verilerini yükle ve ML özelliklerini ekle."""
    df = fetch_data_from_supabase(symbol, timeframe, TOTAL_DAYS)
    
    if not df.empty:
        # ML engine'den teknik özellikleri ekle
        try:
            df_enhanced = ml_engine.extract_technical_features(df)
            return df_enhanced
        except Exception as e:
            st.warning(f"Teknik özellik çıkarımında hata: {e}")
            return df
    
    return df

def create_ml_prediction_chart(df: pd.DataFrame, symbol: str):
    """ML tahminleri grafiği."""
    try:
        # ML modelini yükle ve tahmin yap
        if ml_engine.load_model('random_forest'):
            predictions = []
            confidences = []
            
            # Son 50 veri noktası için tahmin
            for i in range(len(df) - 50, len(df)):
                current_data = df.iloc[:i+1]
                if len(current_data) > 100:
                    pred_result = ml_engine.predict_market_direction(current_data)
                    predictions.append(pred_result['prediction'])
                    confidences.append(pred_result['confidence'])
                else:
                    predictions.append('HOLD')
                    confidences.append(0.0)
            
            # Chart oluştur
            fig = go.Figure()
            
            # Fiyat
            recent_df = df.tail(50).copy()
            fig.add_trace(go.Scatter(
                x=recent_df.index,
                y=recent_df['close'],
                mode='lines',
                name='Price',
                line=dict(color='blue', width=2)
            ))
            
            # ML sinyalleri
            for i, (pred, conf) in enumerate(zip(predictions, confidences)):
                if conf > 0.6:  # Yüksek confidence
                    color = 'green' if pred == 'BUY' else 'red' if pred == 'SELL' else 'gray'
                    fig.add_trace(go.Scatter(
                        x=[recent_df.index[i]],
                        y=[recent_df.iloc[i]['close']],
                        mode='markers',
                        name=f'ML {pred}',
                        marker=dict(color=color, size=8, symbol='triangle-up' if pred == 'BUY' else 'triangle-down'),
                        showlegend=False
                    ))
            
            fig.update_layout(
                title=f'🤖 {symbol} ML Predictions (Last 50 Bars)',
                xaxis_title='Time',
                yaxis_title='Price',
                height=400
            )
            
            return fig
            
    except Exception as e:
        st.error(f"ML prediction chart error: {e}")
        return None

def create_risk_dashboard(df: pd.DataFrame, symbol: str, portfolio_value: float = 10000):
    """Risk dashboard oluştur."""
    if df.empty:
        return None
    
    # Risk analizini çalıştır
    risk_analysis = analyze_symbol_risk(df['close'], portfolio_value)
    
    if 'error' in risk_analysis:
        st.error(f"Risk analysis error: {risk_analysis['error']}")
        return None
    
    # Risk metrikleri
    col1, col2, col3, col4 = st.columns(4)
    
    var_hist = risk_analysis.get('var_historical', {})
    cvar = risk_analysis.get('cvar', {})
    drawdown = risk_analysis.get('drawdown', {})
    vol_metrics = risk_analysis.get('volatility', {})
    
    with col1:
        st.metric(
            "📊 VaR (95%)",
            f"${var_hist.get('var_amount', 0):,.0f}",
            f"{var_hist.get('var_percent', 0):.2f}%"
        )
    
    with col2:
        st.metric(
            "⚠️ CVaR (95%)",
            f"${cvar.get('cvar_amount', 0):,.0f}",
            f"{cvar.get('cvar_percent', 0):.2f}%"
        )
    
    with col3:
        st.metric(
            "📉 Max Drawdown",
            f"{drawdown.get('max_drawdown_percent', 0):.1f}%",
            f"Current: {drawdown.get('current_drawdown_percent', 0):.1f}%"
        )
    
    with col4:
        st.metric(
            "📈 Annual Volatility",
            f"{vol_metrics.get('volatility_annual', 0)*100:.1f}%",
            f"Sharpe: {risk_analysis.get('performance_ratios', {}).get('sharpe_ratio', 0):.2f}"
        )
    
    # Risk level indicator
    risk_assessment = risk_analysis.get('risk_assessment', {})
    risk_level = risk_assessment.get('risk_level', 'UNKNOWN')
    risk_color = {
        'VERY LOW': 'green',
        'LOW': 'lightgreen',
        'MEDIUM': 'yellow',
        'HIGH': 'orange',
        'VERY HIGH': 'red'
    }.get(risk_level, 'gray')
    
    st.markdown(f"""
    <div style="padding: 10px; background-color: {risk_color}; border-radius: 5px; text-align: center; margin: 10px 0;">
        <h3 style="color: white; margin: 0;">Risk Level: {risk_level}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Risk faktörleri
    risk_factors = risk_assessment.get('risk_factors', [])
    if risk_factors:
        st.subheader("🚨 Risk Factors")
        for factor in risk_factors:
            st.warning(factor)
    
    # Öneriler
    recommendation = risk_assessment.get('recommendation', '')
    if recommendation:
        st.subheader("💡 Risk Management Recommendation")
        st.info(recommendation)
    
    # VaR vs CVaR karşılaştırması
    fig_risk = go.Figure()
    
    fig_risk.add_trace(go.Bar(
        name='VaR',
        x=['Daily Risk'],
        y=[var_hist.get('var_percent', 0)],
        marker_color='blue'
    ))
    
    fig_risk.add_trace(go.Bar(
        name='CVaR',
        x=['Daily Risk'],
        y=[cvar.get('cvar_percent', 0)],
        marker_color='red'
    ))
    
    fig_risk.update_layout(
        title='📊 VaR vs CVaR Comparison',
        yaxis_title='Risk (%)',
        height=300
    )
    
    return fig_risk

def create_enhanced_performance_comparison():
    """Enhanced vs Traditional performans karşılaştırması."""
    enhanced_df = load_enhanced_results()
    
    if enhanced_df.empty:
        st.warning("Enhanced analysis results not found. Please run enhanced_orchestrator.py first.")
        return
    
    # Performance comparison
    if 'enhanced_total_return' in enhanced_df.columns and 'traditional_total_return' in enhanced_df.columns:
        fig = go.Figure()
        
        symbols = enhanced_df['symbol'] + '-' + enhanced_df['timeframe']
        
        fig.add_trace(go.Bar(
            name='Traditional Strategy',
            x=symbols,
            y=enhanced_df['traditional_total_return'],
            marker_color='lightblue'
        ))
        
        fig.add_trace(go.Bar(
            name='Enhanced (ML + Risk)',
            x=symbols,
            y=enhanced_df['enhanced_total_return'],
            marker_color='darkgreen'
        ))
        
        fig.update_layout(
            title='🚀 Enhanced vs Traditional Strategy Performance',
            xaxis_title='Symbol-Timeframe',
            yaxis_title='Total Return (%)',
            barmode='group',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Improvement summary
        avg_improvement = enhanced_df['improvement_percent'].mean()
        total_symbols = len(enhanced_df)
        improved_symbols = len(enhanced_df[enhanced_df['improvement_percent'] > 0])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Improvement", f"{avg_improvement:.2f}%")
        with col2:
            st.metric("Symbols Analyzed", total_symbols)
        with col3:
            st.metric("Improved Symbols", f"{improved_symbols}/{total_symbols}")

# Main Dashboard
st.title("🧠 Enhanced Cognitive Trade Agent Dashboard")
st.markdown("**Machine Learning + Advanced Risk Management Integration**")

# Sidebar
st.sidebar.header("🎛️ Dashboard Controls")

# Tab structure
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Enhanced Performance", 
    "🤖 ML Analysis", 
    "⚖️ Risk Management", 
    "🎯 Live Signals",
    "📈 Portfolio Analysis"
])

with tab1:
    st.header("🚀 Enhanced Strategy Performance Analysis")
    
    # Performance comparison
    create_enhanced_performance_comparison()
    
    # Enhanced results table
    enhanced_df = load_enhanced_results()
    if not enhanced_df.empty:
        st.subheader("📋 Detailed Enhanced Results")
        
        display_cols = [
            'symbol', 'timeframe', 'enhanced_total_return', 'traditional_total_return',
            'improvement_percent', 'ml_contribution_percent', 'risk_level'
        ]
        
        available_cols = [col for col in display_cols if col in enhanced_df.columns]
        st.dataframe(enhanced_df[available_cols].style.format({
            'enhanced_total_return': '{:.2f}%',
            'traditional_total_return': '{:.2f}%',
            'improvement_percent': '{:.2f}%',
            'ml_contribution_percent': '{:.1f}%'
        }))

with tab2:
    st.header("🤖 Machine Learning Analysis")
    
    # Symbol selection for ML analysis
    available_symbols = [env['symbol'] for env in MARKET_ENVIRONMENTS]
    selected_symbol = st.selectbox("Select Symbol for ML Analysis:", available_symbols)
    
    if selected_symbol:
        # Find timeframe for selected symbol
        selected_timeframe = next(env['timeframe'] for env in MARKET_ENVIRONMENTS if env['symbol'] == selected_symbol)
        
        # Load data
        df_ml = load_market_data_enhanced(selected_symbol, selected_timeframe)
        
        if not df_ml.empty:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # ML prediction chart
                ml_chart = create_ml_prediction_chart(df_ml, selected_symbol)
                if ml_chart:
                    st.plotly_chart(ml_chart, use_container_width=True)
            
            with col2:
                # Current ML prediction
                try:
                    if ml_engine.load_model('random_forest'):
                        current_pred = ml_engine.predict_market_direction(df_ml)
                        
                        st.subheader(f"🎯 Current ML Signal")
                        
                        # Signal color
                        signal_color = {
                            'BUY': 'green',
                            'SELL': 'red',
                            'HOLD': 'gray'
                        }.get(current_pred['prediction'], 'gray')
                        
                        st.markdown(f"""
                        <div style="padding: 20px; background-color: {signal_color}; border-radius: 10px; text-align: center;">
                            <h2 style="color: white; margin: 0;">{current_pred['prediction']}</h2>
                            <p style="color: white; margin: 5px 0;">Confidence: {current_pred['confidence']:.1%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Probability distribution
                        st.subheader("📊 Prediction Probabilities")
                        prob_data = pd.DataFrame({
                            'Outcome': ['DOWN', 'SIDEWAYS', 'UP'],
                            'Probability': current_pred['probabilities']
                        })
                        
                        fig_prob = px.bar(prob_data, x='Outcome', y='Probability', 
                                        title='ML Prediction Probabilities')
                        st.plotly_chart(fig_prob, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"ML prediction error: {e}")
            
            # Feature importance (if available)
            st.subheader("🔍 Technical Features Analysis")
            
            # Show some key technical indicators
            recent_data = df_ml.tail(10)
            
            if 'rsi_14' in recent_data.columns:
                fig_tech = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=['RSI', 'MACD', 'Volatility', 'Returns'],
                    specs=[[{"secondary_y": False}, {"secondary_y": False}],
                           [{"secondary_y": False}, {"secondary_y": False}]]
                )
                
                # RSI
                if 'rsi_14' in recent_data.columns:
                    fig_tech.add_trace(
                        go.Scatter(y=recent_data['rsi_14'], mode='lines', name='RSI'),
                        row=1, col=1
                    )
                
                # MACD
                if 'macd' in recent_data.columns:
                    fig_tech.add_trace(
                        go.Scatter(y=recent_data['macd'], mode='lines', name='MACD'),
                        row=1, col=2
                    )
                
                # Volatility
                if 'volatility_20' in recent_data.columns:
                    fig_tech.add_trace(
                        go.Scatter(y=recent_data['volatility_20'], mode='lines', name='Volatility'),
                        row=2, col=1
                    )
                
                # Returns
                if 'returns' in recent_data.columns:
                    fig_tech.add_trace(
                        go.Scatter(y=recent_data['returns'], mode='lines', name='Returns'),
                        row=2, col=2
                    )
                
                fig_tech.update_layout(height=600, showlegend=False)
                st.plotly_chart(fig_tech, use_container_width=True)

with tab3:
    st.header("⚖️ Advanced Risk Management")
    
    # Symbol selection for risk analysis
    risk_symbol = st.selectbox("Select Symbol for Risk Analysis:", available_symbols, key="risk_symbol")
    portfolio_value = st.number_input("Portfolio Value ($):", min_value=1000, value=10000, step=1000)
    
    if risk_symbol:
        risk_timeframe = next(env['timeframe'] for env in MARKET_ENVIRONMENTS if env['symbol'] == risk_symbol)
        df_risk = load_market_data_enhanced(risk_symbol, risk_timeframe)
        
        if not df_risk.empty:
            # Risk dashboard
            risk_chart = create_risk_dashboard(df_risk, risk_symbol, portfolio_value)
            
            if risk_chart:
                st.plotly_chart(risk_chart, use_container_width=True)
            
            # Position sizing recommendation
            st.subheader("📏 Position Sizing Recommendation")
            
            try:
                from risk_models import get_position_sizing_recommendation
                pos_sizing = get_position_sizing_recommendation(
                    df_risk['close'], portfolio_value, max_risk=0.02
                )
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Suggested Position Size", f"{pos_sizing['suggested_position_size']:.4f}")
                
                with col2:
                    st.metric("Max Position Value", f"${pos_sizing['max_position_value']:,.0f}")
                
                with col3:
                    st.metric("Expected Risk", f"${pos_sizing['expected_risk_amount']:,.0f}")
                
                # Risk-reward details
                rr_metrics = pos_sizing.get('risk_reward_metrics', {})
                st.info(f"""
                **Position Sizing Details:**
                - Current Price: ${rr_metrics.get('current_price', 0):.2f}
                - VaR-based Stop Loss: ${rr_metrics.get('var_based_stop_loss', 0):.2f}
                - Max Risk per Trade: {rr_metrics.get('max_risk_per_trade_percent', 0):.1f}%
                """)
                
            except Exception as e:
                st.error(f"Position sizing error: {e}")

with tab4:
    st.header("🎯 Live Enhanced Signals")
    
    st.info("🔄 Live signals are generated by combining traditional strategies, ML predictions, and risk analysis.")
    
    # Real-time signals for all symbols
    signal_results = []
    
    for env in MARKET_ENVIRONMENTS:
        symbol, timeframe = env['symbol'], env['timeframe']
        
        try:
            df_signal = load_market_data_enhanced(symbol, timeframe)
            
            if not df_signal.empty:
                # Simulate enhanced signal (would need strategy info in real implementation)
                mock_strategy = {
                    'name': 'Mock Strategy',
                    'type': 'CONFLUENCE',
                    'params': {
                        'buy_rules': [{'indicator': 'RSI', 'condition': 'cross_above', 'value': 30}],
                        'sell_rules': [{'indicator': 'RSI', 'condition': 'cross_below', 'value': 70}],
                        'buy_quorum': 1,
                        'sell_quorum': 1
                    }
                }
                
                # Get enhanced signal
                enhanced_signal = enhanced_engine.get_enhanced_signal(
                    df_signal, mock_strategy, symbol, 10000
                )
                
                signal_results.append({
                    'Symbol': f"{symbol}-{timeframe}",
                    'Signal': enhanced_signal['signal'],
                    'Confidence': enhanced_signal['confidence'],
                    'Traditional': enhanced_signal['components']['traditional'],
                    'ML': enhanced_signal['components']['ml'],
                    'Risk Level': enhanced_signal['components']['risk_level'],
                    'Market Regime': enhanced_signal.get('market_regime', 'Unknown')
                })
                
        except Exception as e:
            st.error(f"Signal generation error for {symbol}: {e}")
    
    if signal_results:
        signals_df = pd.DataFrame(signal_results)
        
        # Signal summary
        buy_signals = len(signals_df[signals_df['Signal'] == 'BUY'])
        sell_signals = len(signals_df[signals_df['Signal'] == 'SELL'])
        hold_signals = len(signals_df[signals_df['Signal'] == 'HOLD'])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🟢 BUY Signals", buy_signals)
        with col2:
            st.metric("🔴 SELL Signals", sell_signals)
        with col3:
            st.metric("🟡 HOLD Signals", hold_signals)
        
        # Signals table
        st.dataframe(signals_df.style.format({
            'Confidence': '{:.1%}'
        }))

with tab5:
    st.header("📈 Portfolio-Level Analysis")
    
    st.subheader("🎯 Portfolio Risk Overview")
    
    # Mock portfolio for demonstration
    mock_positions = {}
    mock_market_data = {}
    
    for env in MARKET_ENVIRONMENTS:
        symbol = env['symbol']
        df = load_market_data_enhanced(symbol, env['timeframe'])
        
        if not df.empty:
            mock_positions[symbol] = {
                'quantity': 1.0,  # Normalized
                'entry_price': df['close'].iloc[-50]  # Entry 50 bars ago
            }
            mock_market_data[symbol] = df['close']
    
    if mock_positions:
        try:
            portfolio_risk = enhanced_engine.generate_portfolio_risk_report(
                mock_positions, mock_market_data
            )
            
            # Portfolio metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Portfolio Value", f"${portfolio_risk['portfolio_value']:,.0f}")
            
            with col2:
                st.metric("Portfolio VaR", f"{portfolio_risk['portfolio_var_percent']:.2f}%")
            
            with col3:
                st.metric("Risk Level", portfolio_risk['risk_level'])
            
            with col4:
                st.metric("Diversification Score", f"{portfolio_risk['diversification_score']:.0f}/100")
            
            # Position risks
            st.subheader("📊 Position-Level Risk Breakdown")
            
            if portfolio_risk['position_risks']:
                pos_risk_df = pd.DataFrame(portfolio_risk['position_risks'])
                
                fig_positions = px.pie(
                    pos_risk_df, 
                    values='risk_contribution', 
                    names='symbol',
                    title='Risk Contribution by Position'
                )
                st.plotly_chart(fig_positions, use_container_width=True)
                
                # Detailed position risks
                st.dataframe(pos_risk_df.style.format({
                    'position_value': '${:,.0f}',
                    'var_amount': '${:,.0f}',
                    'cvar_amount': '${:,.0f}',
                    'risk_contribution': '{:.1f}%',
                    'max_drawdown': '{:.1f}%'
                }))
            
            # Recommendations
            st.subheader("💡 Portfolio Recommendations")
            for rec in portfolio_risk.get('recommendations', []):
                st.info(rec)
            
            # ML signals for positions
            ml_signals = portfolio_risk.get('ml_signals', {})
            if ml_signals:
                st.subheader("🤖 ML Signals for Current Positions")
                
                ml_df = pd.DataFrame([
                    {
                        'Symbol': symbol,
                        'ML Signal': data['prediction'],
                        'Confidence': data['confidence'],
                        'Probabilities': str([f"{p:.2f}" for p in data['probabilities']])
                    }
                    for symbol, data in ml_signals.items()
                ])
                
                st.dataframe(ml_df)
                
        except Exception as e:
            st.error(f"Portfolio analysis error: {e}")

# Footer
st.markdown("---")
st.markdown("**🧠 Enhanced Cognitive Trade Agent** - Powered by Machine Learning & Advanced Risk Management")
st.markdown("*Dashboard last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "*")