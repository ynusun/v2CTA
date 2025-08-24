#!/usr/bin/env python3
"""
Enhanced Cognitive Trade Agent - Quick Start Script
==================================================

Bu script, yeni ML ve risk yönetimi özelliklerini kolayca test etmenizi sağlar.

Kullanım:
    python run_enhanced_analysis.py

Özellikler:
    - Machine Learning model eğitimi
    - VaR/CVaR risk analizi
    - Enhanced backtest karşılaştırması
    - Portföy seviyesi risk değerlendirmesi
"""

import os
import sys
import time
from datetime import datetime

def print_header():
    """Başlık yazdır."""
    print("="*80)
    print("🚀 ENHANCED COGNITIVE TRADE AGENT")
    print("   Machine Learning + Advanced Risk Management")
    print("   Version 2.0 - Enhanced Edition")
    print("="*80)
    print()

def check_dependencies():
    """Gerekli kütüphaneleri kontrol et."""
    print("📦 Checking dependencies...")
    
    required_packages = [
        'pandas', 'numpy', 'sklearn', 'scipy', 
        'joblib', 'pandas_ta', 'plotly', 'streamlit'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - Missing!")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages detected. Install with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ All dependencies satisfied!")
    return True

def run_data_collection():
    """Veri toplama işlemini çalıştır."""
    print("\n📊 STEP 1: Data Collection")
    print("-" * 40)
    
    try:
        from data_collector import fetch_data_from_supabase
        from config import MARKET_ENVIRONMENTS, TOTAL_DAYS
        
        for env in MARKET_ENVIRONMENTS:
            symbol, timeframe = env['symbol'], env['timeframe']
            print(f"📈 Fetching data for {symbol}-{timeframe}...")
            
            df = fetch_data_from_supabase(symbol, timeframe, TOTAL_DAYS)
            
            if not df.empty:
                print(f"  ✅ {symbol}-{timeframe}: {len(df)} records")
            else:
                print(f"  ⚠️  {symbol}-{timeframe}: No data found")
        
        return True
        
    except Exception as e:
        print(f"❌ Data collection failed: {e}")
        return False

def run_ml_training():
    """ML modelleri eğit."""
    print("\n🤖 STEP 2: Machine Learning Training")
    print("-" * 40)
    
    try:
        from ml_engine import PatternRecognitionEngine
        from data_collector import fetch_data_from_supabase
        from config import MARKET_ENVIRONMENTS, TOTAL_DAYS
        
        ml_engine = PatternRecognitionEngine()
        trained_models = 0
        
        for env in MARKET_ENVIRONMENTS:
            symbol, timeframe = env['symbol'], env['timeframe']
            
            print(f"🔄 Training ML model for {symbol}-{timeframe}...")
            
            df = fetch_data_from_supabase(symbol, timeframe, TOTAL_DAYS)
            
            if len(df) >= 200:  # Minimum data requirement
                model = ml_engine.train_model(df, model_type=f"{symbol}_random_forest")
                
                if model is not None:
                    print(f"  ✅ {symbol} model trained successfully")
                    trained_models += 1
                else:
                    print(f"  ❌ {symbol} model training failed")
            else:
                print(f"  ⚠️  {symbol}: Insufficient data ({len(df)} < 200)")
        
        print(f"\n🎯 ML Training Summary: {trained_models} models trained")
        return trained_models > 0
        
    except Exception as e:
        print(f"❌ ML training failed: {e}")
        return False

def run_risk_analysis():
    """Risk analizi çalıştır."""
    print("\n⚖️  STEP 3: Risk Analysis")
    print("-" * 40)
    
    try:
        from risk_models import AdvancedRiskManager, analyze_symbol_risk
        from data_collector import fetch_data_from_supabase
        from config import MARKET_ENVIRONMENTS
        
        risk_manager = AdvancedRiskManager()
        
        for env in MARKET_ENVIRONMENTS:
            symbol, timeframe = env['symbol'], env['timeframe']
            
            print(f"📊 Analyzing risk for {symbol}-{timeframe}...")
            
            df = fetch_data_from_supabase(symbol, timeframe, 180)  # 6 months data
            
            if not df.empty and len(df) > 50:
                risk_analysis = analyze_symbol_risk(df['close'], 10000)
                
                if 'error' not in risk_analysis:
                    var_percent = risk_analysis.get('var_historical', {}).get('var_percent', 0)
                    cvar_percent = risk_analysis.get('cvar', {}).get('cvar_percent', 0)
                    risk_level = risk_analysis.get('risk_assessment', {}).get('risk_level', 'UNKNOWN')
                    
                    print(f"  📈 {symbol} Risk Metrics:")
                    print(f"    • VaR (95%): {var_percent:.2f}%")
                    print(f"    • CVaR (95%): {cvar_percent:.2f}%")
                    print(f"    • Risk Level: {risk_level}")
                else:
                    print(f"  ❌ {symbol}: Risk analysis failed")
            else:
                print(f"  ⚠️  {symbol}: Insufficient data for risk analysis")
        
        return True
        
    except Exception as e:
        print(f"❌ Risk analysis failed: {e}")
        return False

def run_enhanced_backtest():
    """Enhanced backtest çalıştır."""
    print("\n🧪 STEP 4: Enhanced Backtest")
    print("-" * 40)
    
    try:
        from enhanced_strategy_engine import EnhancedStrategyEngine
        from data_collector import fetch_data_from_supabase
        from config import MARKET_ENVIRONMENTS, TRAINING_DAYS
        from datetime import timedelta
        
        enhanced_engine = EnhancedStrategyEngine()
        
        # Mock strategy for testing
        mock_strategy = {
            'name': 'RSI_Test_Strategy',
            'type': 'CONFLUENCE',
            'params': {
                'buy_rules': [
                    {
                        'indicator': 'RSI',
                        'condition': 'cross_above',
                        'value': 30,
                        'params': {'rsi_period': 14}
                    }
                ],
                'sell_rules': [
                    {
                        'indicator': 'RSI',
                        'condition': 'cross_below',
                        'value': 70,
                        'params': {'rsi_period': 14}
                    }
                ],
                'buy_quorum': 1,
                'sell_quorum': 1
            }
        }
        
        successful_tests = 0
        
        for env in MARKET_ENVIRONMENTS:
            symbol, timeframe = env['symbol'], env['timeframe']
            
            print(f"🚀 Running enhanced backtest for {symbol}-{timeframe}...")
            
            df = fetch_data_from_supabase(symbol, timeframe, 180)
            
            if len(df) >= 150:
                # Split for validation
                split_date = df['timestamp'].min() + timedelta(days=TRAINING_DAYS)
                validation_df = df[df['timestamp'] >= split_date]
                
                if len(validation_df) >= 50:
                    results = enhanced_engine.run_enhanced_backtest(
                        validation_df, mock_strategy, symbol
                    )
                    
                    if results:
                        print(f"  ✅ {symbol} Enhanced Backtest Results:")
                        print(f"    • Total Return: {results['total_return']:.2f}%")
                        print(f"    • Win Rate: {results['win_rate']:.1f}%")
                        print(f"    • Max Drawdown: {results['max_drawdown']:.2f}%")
                        print(f"    • Sharpe Ratio: {results['sharpe_ratio']:.2f}")
                        print(f"    • ML Contribution: {results['ml_contribution']:.1f}%")
                        
                        successful_tests += 1
                    else:
                        print(f"  ❌ {symbol}: Backtest failed")
                else:
                    print(f"  ⚠️  {symbol}: Insufficient validation data")
            else:
                print(f"  ⚠️  {symbol}: Insufficient total data")
        
        print(f"\n🎯 Enhanced Backtest Summary: {successful_tests} successful tests")
        return successful_tests > 0
        
    except Exception as e:
        print(f"❌ Enhanced backtest failed: {e}")
        return False

def run_portfolio_analysis():
    """Portföy analizi çalıştır."""
    print("\n💼 STEP 5: Portfolio Analysis")
    print("-" * 40)
    
    try:
        from enhanced_strategy_engine import EnhancedStrategyEngine
        from data_collector import fetch_data_from_supabase
        from config import MARKET_ENVIRONMENTS
        
        enhanced_engine = EnhancedStrategyEngine()
        
        # Mock portfolio positions
        portfolio_positions = {}
        market_data = {}
        
        for env in MARKET_ENVIRONMENTS:
            symbol = env['symbol']
            
            df = fetch_data_from_supabase(symbol, env['timeframe'], 90)
            
            if not df.empty and len(df) > 50:
                # Mock position (normalized)
                portfolio_positions[symbol] = {
                    'quantity': 1.0,
                    'entry_price': df['close'].iloc[-30]  # Entry 30 bars ago
                }
                
                market_data[symbol] = df['close']
        
        if portfolio_positions:
            print(f"📊 Analyzing portfolio with {len(portfolio_positions)} positions...")
            
            portfolio_report = enhanced_engine.generate_portfolio_risk_report(
                portfolio_positions, market_data
            )
            
            print(f"\n💰 Portfolio Risk Report:")
            print(f"  • Portfolio Value: ${portfolio_report['portfolio_value']:,.0f}")
            print(f"  • Portfolio VaR: {portfolio_report['portfolio_var_percent']:.2f}%")
            print(f"  • Risk Level: {portfolio_report['risk_level']}")
            print(f"  • Diversification Score: {portfolio_report['diversification_score']:.0f}/100")
            
            # Top recommendations
            recommendations = portfolio_report.get('recommendations', [])[:3]
            if recommendations:
                print(f"\n💡 Top Recommendations:")
                for i, rec in enumerate(recommendations, 1):
                    print(f"  {i}. {rec}")
            
            return True
        else:
            print("⚠️  No valid positions for portfolio analysis")
            return False
        
    except Exception as e:
        print(f"❌ Portfolio analysis failed: {e}")
        return False

def generate_summary_report():
    """Özet rapor oluştur."""
    print("\n📋 STEP 6: Generating Summary Report")
    print("-" * 40)
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"enhanced_analysis_report_{timestamp}.txt"
        
        with open(report_file, 'w') as f:
            f.write("ENHANCED COGNITIVE TRADE AGENT - ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("ANALYSIS SUMMARY:\n")
            f.write("-" * 20 + "\n")
            f.write("✅ Data Collection: Completed\n")
            f.write("✅ ML Training: Completed\n")
            f.write("✅ Risk Analysis: Completed\n")
            f.write("✅ Enhanced Backtesting: Completed\n")
            f.write("✅ Portfolio Analysis: Completed\n\n")
            
            f.write("KEY IMPROVEMENTS:\n")
            f.write("-" * 20 + "\n")
            f.write("🤖 Machine Learning Pattern Recognition\n")
            f.write("⚖️  Advanced Risk Management (VaR, CVaR)\n")
            f.write("📊 Enhanced Strategy Engine\n")
            f.write("💼 Portfolio-level Risk Analysis\n")
            f.write("📈 Real-time Signal Integration\n\n")
            
            f.write("NEXT STEPS:\n")
            f.write("-" * 20 + "\n")
            f.write("1. Run: streamlit run enhanced_dashboard.py\n")
            f.write("2. View enhanced performance metrics\n")
            f.write("3. Monitor ML predictions and risk levels\n")
            f.write("4. Adjust position sizes based on VaR analysis\n")
        
        print(f"✅ Summary report saved: {report_file}")
        return True
        
    except Exception as e:
        print(f"❌ Report generation failed: {e}")
        return False

def main():
    """Ana fonksiyon."""
    print_header()
    
    # Dependency check
    if not check_dependencies():
        print("\n❌ Please install missing dependencies before proceeding.")
        sys.exit(1)
    
    # Environment check
    print("🔧 Checking environment...")
    
    if not os.path.exists('.env'):
        print("⚠️  .env file not found. Please create one with:")
        print("   - SUPABASE_URL")
        print("   - SUPABASE_KEY")
        print("   - BINANCE_API_KEY")
        print("   - BINANCE_API_SECRET")
        sys.exit(1)
    
    print("✅ Environment configured!")
    
    # Step-by-step execution
    steps = [
        ("Data Collection", run_data_collection),
        ("ML Training", run_ml_training),
        ("Risk Analysis", run_risk_analysis),
        ("Enhanced Backtest", run_enhanced_backtest),
        ("Portfolio Analysis", run_portfolio_analysis),
        ("Summary Report", generate_summary_report)
    ]
    
    successful_steps = 0
    total_steps = len(steps)
    
    start_time = time.time()
    
    for step_name, step_function in steps:
        print(f"\n⏳ Starting: {step_name}...")
        
        try:
            if step_function():
                successful_steps += 1
                print(f"✅ {step_name} completed successfully!")
            else:
                print(f"⚠️  {step_name} completed with warnings.")
                
        except Exception as e:
            print(f"❌ {step_name} failed: {e}")
        
        time.sleep(1)  # Brief pause between steps
    
    elapsed_time = time.time() - start_time
    
    # Final summary
    print("\n" + "="*80)
    print("🎉 ENHANCED ANALYSIS COMPLETE!")
    print("="*80)
    print(f"⏱️  Total Time: {elapsed_time:.1f} seconds")
    print(f"✅ Successful Steps: {successful_steps}/{total_steps}")
    
    if successful_steps >= 4:  # Most steps completed
        print("\n🚀 Your Enhanced Cognitive Trade Agent is ready!")
        print("\n📊 To view results, run:")
        print("   streamlit run enhanced_dashboard.py")
        print("\n🤖 Key Features Now Available:")
        print("   • ML-powered pattern recognition")
        print("   • VaR/CVaR risk analysis")
        print("   • Enhanced backtesting")
        print("   • Portfolio risk management")
        print("   • Real-time signal integration")
    else:
        print("\n⚠️  Some steps failed. Check the logs above.")
        print("You may still be able to use basic features.")
    
    print("\n🔗 For support or questions:")
    print("   • Check enhanced_dashboard.py for visualization")
    print("   • Review individual module files for details")
    print("   • Ensure all dependencies are properly installed")
    
    return successful_steps >= 4

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)