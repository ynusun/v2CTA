import os
import json
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
from supabase import create_client, Client
import httpx

# Original imports
from config import MARKET_ENVIRONMENTS, TOTAL_DAYS, TRAINING_DAYS
from data_collector import fetch_data_from_supabase
from orchestrator import generate_strategies, get_or_create_strategy_in_db

# New enhanced modules
from enhanced_strategy_engine import EnhancedStrategyEngine
from ml_engine import PatternRecognitionEngine
from risk_models import AdvancedRiskManager

load_dotenv()
supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

class EnhancedOrchestrator:
    """
    ML ve gelişmiş risk yönetimi entegre edilmiş orkestratör.
    Daha akıllı ve güvenilir trading stratejileri geliştirir.
    """
    
    def __init__(self):
        self.enhanced_engine = EnhancedStrategyEngine()
        self.ml_engine = PatternRecognitionEngine()
        self.risk_manager = AdvancedRiskManager()
        
        # Results storage
        self.enhanced_results = {}
        self.ml_models_status = {}
        
        # Performance tracking
        self.performance_comparison = {
            'traditional': {},
            'enhanced': {},
            'ml_only': {}
        }
    
    def run_enhanced_analysis(self, force_retrain_ml: bool = False):
        """
        Gelişmiş ML ve risk entegre analizi çalıştırır.
        """
        print("\n" + "="*80)
        print("🚀 ENHANCED COGNITIVE TRADE AGENT - GELIŞMIŞ ANALİZ BAŞLATILIYOR")
        print("   Machine Learning + Advanced Risk Management")
        print("="*80)
        
        # Veri toplama
        training_data = {}
        validation_data = {}
        full_datasets = {}
        
        print("\n📊 ADIM 1: VERİ TOPLAMA VE HAZIRLIK")
        print("-" * 50)
        
        for env in MARKET_ENVIRONMENTS:
            symbol, timeframe = env['symbol'], env['timeframe']
            key = f"{symbol}-{timeframe}"
            
            print(f"📈 {key} verisi çekiliyor...")
            full_df = fetch_data_from_supabase(symbol, timeframe, TOTAL_DAYS)
            
            if full_df.empty:
                print(f"❌ {key} için veri bulunamadı, atlanıyor...")
                continue
            
            # Train/validation split
            split_date = full_df['timestamp'].min() + timedelta(days=TRAINING_DAYS)
            train_df = full_df[full_df['timestamp'] < split_date]
            val_df = full_df[full_df['timestamp'] >= split_date]
            
            if len(train_df) < 100 or len(val_df) < 50:
                print(f"⚠️ {key} için yetersiz veri, atlanıyor...")
                continue
            
            full_datasets[key] = full_df
            training_data[key] = train_df
            validation_data[key] = val_df
            
            print(f"✅ {key}: Train={len(train_df)}, Val={len(val_df)} kayıt")
        
        if not training_data:
            print("❌ Analiz için yeterli veri bulunamadı!")
            return
        
        # ML modelleri eğitimi
        print("\n🤖 ADIM 2: MACHINE LEARNING MODELLERİ EĞİTİMİ")
        print("-" * 50)
        
        self.enhanced_engine.initialize_ml_models(training_data, force_retrain_ml)
        
        # Geleneksel stratejiler
        print("\n📈 ADIM 3: GELENEKSEL STRATEJİLER ANALİZİ")
        print("-" * 50)
        
        traditional_strategies = {}
        for key in training_data.keys():
            symbol = key.split('-')[0]
            
            # Market regime detection
            regime = self.enhanced_engine.ml_engine.detect_market_regime(training_data[key])
            print(f"🔍 {key} piyasa rejimi: {regime}")
            
            # Generate strategies for this regime
            strategies_to_test = generate_strategies(regime)
            
            if not strategies_to_test:
                print(f"⚠️ {key} için uygun strateji bulunamadı")
                continue
            
            # Find best traditional strategy
            best_strategy = None
            best_performance = -999
            
            for strategy_info in strategies_to_test[:10]:  # Test top 10 strategies
                try:
                    # Traditional backtest
                    from strategy_engine import run_backtest
                    results = run_backtest(training_data[key].copy(), strategy_info)
                    
                    if results and results['net_profit_percent'] > best_performance:
                        best_performance = results['net_profit_percent']
                        best_strategy = strategy_info
                        
                except Exception as e:
                    print(f"⚠️ Strateji test hatası: {e}")
                    continue
            
            if best_strategy:
                traditional_strategies[key] = {
                    'strategy': best_strategy,
                    'performance': best_performance
                }
                print(f"🏆 {key} en iyi geleneksel strateji: {best_strategy['name']} ({best_performance:.2f}%)")
        
        # Enhanced (ML + Risk) analizi
        print("\n🧠 ADIM 4: ENHANCED (ML + RISK) STRATEJİ ANALİZİ")
        print("-" * 50)
        
        enhanced_results = {}
        
        for key, traditional_info in traditional_strategies.items():
            symbol = key.split('-')[0]
            strategy_info = traditional_info['strategy']
            
            print(f"\n🔬 {key} enhanced analiz...")
            
            try:
                # Enhanced backtest (ML + Risk integrated)
                enhanced_result = self.enhanced_engine.run_enhanced_backtest(
                    validation_data[key], strategy_info, symbol
                )
                
                # Traditional backtest on same validation data (for comparison)
                from strategy_engine import run_backtest
                traditional_result = run_backtest(validation_data[key].copy(), strategy_info)
                
                # Risk analysis
                risk_metrics = self.risk_manager.calculate_risk_metrics(
                    validation_data[key]['close'], 10000
                )
                
                enhanced_results[key] = {
                    'enhanced': enhanced_result,
                    'traditional': traditional_result,
                    'risk_metrics': risk_metrics,
                    'strategy_info': strategy_info,
                    'symbol': symbol
                }
                
                # Performance comparison
                enh_return = enhanced_result['total_return']
                trad_return = traditional_result['net_profit_percent'] if traditional_result else 0
                improvement = enh_return - trad_return
                
                print(f"📊 {key} Sonuçlar:")
                print(f"   Geleneksel: {trad_return:.2f}%")
                print(f"   Enhanced:   {enh_return:.2f}%")
                print(f"   İyileşme:   {improvement:+.2f}%")
                print(f"   ML Katkısı: {enhanced_result['ml_contribution']:.1f}%")
                print(f"   Risk Seviyesi: {risk_metrics.get('risk_assessment', {}).get('risk_level', 'N/A')}")
                
            except Exception as e:
                print(f"❌ {key} enhanced analiz hatası: {e}")
                continue
        
        self.enhanced_results = enhanced_results
        
        # Portföy seviyesi analiz
        print("\n💼 ADIM 5: PORTFÖY SEVİYESİ RİSK ANALİZİ")
        print("-" * 50)
        
        portfolio_analysis = self._analyze_portfolio_level_performance(enhanced_results, full_datasets)
        
        # Sonuçları kaydet
        self._save_enhanced_results(enhanced_results, portfolio_analysis)
        
        # Final rapor
        print("\n" + "="*80)
        print("🎯 ENHANCED ANALİZ TAMAMLANDI - ÖZET RAPOR")
        print("="*80)
        
        self._print_summary_report(enhanced_results, portfolio_analysis)
        
        return enhanced_results, portfolio_analysis
    
    def _analyze_portfolio_level_performance(self, enhanced_results: dict, full_datasets: dict) -> dict:
        """
        Portföy seviyesinde performans ve risk analizi.
        """
        print("🔍 Portföy seviyesi analiz...")
        
        # Simulate portfolio with enhanced strategies
        portfolio_positions = {}
        market_data_for_risk = {}
        
        # Create mock positions for risk analysis
        for key, results in enhanced_results.items():
            symbol = results['symbol']
            
            # Mock position (assuming we held from start of validation period)
            portfolio_positions[symbol] = {
                'quantity': 1.0,  # Normalized position
                'entry_price': full_datasets[key].iloc[len(full_datasets[key])//2]['close']
            }
            
            market_data_for_risk[symbol] = full_datasets[key]['close']
        
        if not portfolio_positions:
            return {'error': 'No valid positions for portfolio analysis'}
        
        # Portfolio risk analysis
        portfolio_risk = self.enhanced_engine.generate_portfolio_risk_report(
            portfolio_positions, market_data_for_risk
        )
        
        # Performance aggregation
        total_enhanced_return = sum([r['enhanced']['total_return'] for r in enhanced_results.values()])
        total_traditional_return = sum([r['traditional']['net_profit_percent'] for r in enhanced_results.values() if r['traditional']])
        
        avg_enhanced_return = total_enhanced_return / len(enhanced_results)
        avg_traditional_return = total_traditional_return / len(enhanced_results)
        
        # ML contribution analysis
        ml_contributions = [r['enhanced']['ml_contribution'] for r in enhanced_results.values()]
        avg_ml_contribution = sum(ml_contributions) / len(ml_contributions) if ml_contributions else 0
        
        # Risk-adjusted returns
        portfolio_var = portfolio_risk.get('portfolio_var_percent', 5)
        risk_adjusted_return = avg_enhanced_return / max(portfolio_var, 1)
        
        return {
            'portfolio_performance': {
                'avg_enhanced_return': avg_enhanced_return,
                'avg_traditional_return': avg_traditional_return,
                'avg_improvement': avg_enhanced_return - avg_traditional_return,
                'avg_ml_contribution': avg_ml_contribution,
                'risk_adjusted_return': risk_adjusted_return
            },
            'portfolio_risk': portfolio_risk,
            'individual_results': enhanced_results,
            'summary': {
                'total_symbols_analyzed': len(enhanced_results),
                'ml_models_successful': sum([1 for r in enhanced_results.values() if r['enhanced']['ml_contribution'] > 0]),
                'overall_risk_level': portfolio_risk.get('risk_level', 'UNKNOWN')
            }
        }
    
    def _save_enhanced_results(self, enhanced_results: dict, portfolio_analysis: dict):
        """
        Gelişmiş analiz sonuçlarını kaydet.
        """
        print("💾 Sonuçlar kaydediliyor...")
        
        # Enhanced results'ı JSON'a kaydet
        results_to_save = {}
        for key, result in enhanced_results.items():
            # Serialize complex objects
            serializable_result = {
                'enhanced_performance': {
                    'total_return': result['enhanced']['total_return'],
                    'win_rate': result['enhanced']['win_rate'],
                    'max_drawdown': result['enhanced']['max_drawdown'],
                    'sharpe_ratio': result['enhanced']['sharpe_ratio'],
                    'ml_contribution': result['enhanced']['ml_contribution']
                },
                'traditional_performance': {
                    'total_return': result['traditional']['net_profit_percent'] if result['traditional'] else 0,
                    'win_rate': result['traditional']['win_rate'] if result['traditional'] else 0,
                    'max_drawdown': result['traditional']['max_drawdown'] if result['traditional'] else 0
                },
                'risk_level': result['risk_metrics'].get('risk_assessment', {}).get('risk_level', 'UNKNOWN'),
                'strategy_name': result['strategy_info']['name']
            }
            results_to_save[key] = serializable_result
        
        # Save to files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # Enhanced results
            with open(f"enhanced_results_{timestamp}.json", 'w') as f:
                json.dump(results_to_save, f, indent=2, default=str)
            
            # Portfolio analysis
            with open(f"portfolio_analysis_{timestamp}.json", 'w') as f:
                json.dump(portfolio_analysis, f, indent=2, default=str)
                
            print(f"✅ Sonuçlar kaydedildi: enhanced_results_{timestamp}.json")
            
        except Exception as e:
            print(f"⚠️ Sonuç kaydetme hatası: {e}")
        
        # Database'e enhanced results kaydet (opsiyonel)
        try:
            self._save_to_database(enhanced_results)
        except Exception as e:
            print(f"⚠️ Veritabanı kayıt hatası: {e}")
    
    def _save_to_database(self, enhanced_results: dict):
        """
        Enhanced sonuçları veritabanına kaydet.
        """
        for key, result in enhanced_results.items():
            symbol, timeframe = key.split('-')
            
            # Strategy'yi database'e kaydet
            strategy_id = get_or_create_strategy_in_db(result['strategy_info'])
            
            # Enhanced results kaydet
            enhanced_record = {
                'symbol': symbol,
                'timeframe': timeframe,
                'strategy_id': strategy_id,
                'analysis_type': 'ENHANCED_ML_RISK',
                'enhanced_total_return': result['enhanced']['total_return'],
                'enhanced_win_rate': result['enhanced']['win_rate'],
                'enhanced_max_drawdown': result['enhanced']['max_drawdown'],
                'enhanced_sharpe_ratio': result['enhanced']['sharpe_ratio'],
                'ml_contribution_percent': result['enhanced']['ml_contribution'],
                'traditional_total_return': result['traditional']['net_profit_percent'] if result['traditional'] else 0,
                'improvement_percent': result['enhanced']['total_return'] - (result['traditional']['net_profit_percent'] if result['traditional'] else 0),
                'risk_level': result['risk_metrics'].get('risk_assessment', {}).get('risk_level', 'UNKNOWN'),
                'var_percent': result['risk_metrics'].get('var_historical', {}).get('var_percent', 0),
                'created_at': datetime.now().isoformat()
            }
            
            # Create table if not exists (simplified)
            supabase.table('enhanced_backtest_results').upsert(enhanced_record).execute()
    
    def _print_summary_report(self, enhanced_results: dict, portfolio_analysis: dict):
        """
        Özet raporu yazdır.
        """
        perf = portfolio_analysis['portfolio_performance']
        summary = portfolio_analysis['summary']
        
        print(f"📊 Analiz Edilen Sembol Sayısı: {summary['total_symbols_analyzed']}")
        print(f"🤖 ML Başarılı Model Sayısı: {summary['ml_models_successful']}")
        print(f"💰 Ortalama Enhanced Return: {perf['avg_enhanced_return']:.2f}%")
        print(f"📈 Ortalama Geleneksel Return: {perf['avg_traditional_return']:.2f}%")
        print(f"🚀 Ortalama İyileşme: {perf['avg_improvement']:+.2f}%")
        print(f"🧠 Ortalama ML Katkısı: {perf['avg_ml_contribution']:.1f}%")
        print(f"⚖️ Risk Ayarlı Return: {perf['risk_adjusted_return']:.2f}")
        print(f"🎯 Genel Risk Seviyesi: {summary['overall_risk_level']}")
        
        print("\n🏆 EN İYI PERFORMANS GÖSTEREN SEMBOLLER:")
        sorted_results = sorted(enhanced_results.items(), 
                              key=lambda x: x[1]['enhanced']['total_return'], reverse=True)
        
        for i, (key, result) in enumerate(sorted_results[:5]):
            print(f"  {i+1}. {key}: {result['enhanced']['total_return']:.2f}% " +
                  f"(ML: {result['enhanced']['ml_contribution']:.1f}%)")


def main():
    """
    Enhanced orchestrator ana fonksiyon.
    """
    orchestrator = EnhancedOrchestrator()
    
    try:
        enhanced_results, portfolio_analysis = orchestrator.run_enhanced_analysis(force_retrain_ml=False)
        
        print("\n✨ Enhanced analiz başarıyla tamamlandı!")
        print("Detaylı sonuçlar JSON dosyalarına kaydedildi.")
        
    except Exception as e:
        print(f"\n❌ Enhanced analiz sırasında hata: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()