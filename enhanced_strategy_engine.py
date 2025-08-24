import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
from typing import Dict, Any, Optional, Tuple

# Import our new modules
from ml_engine import PatternRecognitionEngine
from risk_models import AdvancedRiskManager
from strategy_engine import (
    detect_market_regime, calculate_strategy_indicators, 
    _evaluate_rule, USE_STOP_LOSS, ATR_PERIOD, ATR_MULTIPLIER
)

class EnhancedStrategyEngine:
    """
    ML ve gelişmiş risk yönetimi entegre edilmiş strateji motoru.
    """
    
    def __init__(self, initial_capital: float = 10000, ml_confidence_threshold: float = 0.65):
        self.initial_capital = initial_capital
        self.ml_confidence_threshold = ml_confidence_threshold
        
        # ML ve Risk yönetimi motorları
        self.ml_engine = PatternRecognitionEngine()
        self.risk_manager = AdvancedRiskManager(confidence_level=0.95)
        
        # Model durumu
        self.ml_models_trained = {}
        
        # Performance tracking
        self.performance_history = []
    
    def initialize_ml_models(self, training_data: Dict[str, pd.DataFrame], force_retrain: bool = False):
        """
        Her sembol için ML modellerini başlatır/eğitir.
        """
        print("🤖 ML Modelleri başlatılıyor...")
        
        for symbol, df in training_data.items():
            model_file = f"ml_models/pattern_model_{symbol}_random_forest.pkl"
            
            # Eğer model varsa ve yeniden eğitim zorunlu değilse, yükle
            if os.path.exists(model_file) and not force_retrain:
                success = self.ml_engine.load_model(f"{symbol}_random_forest")
                if success:
                    self.ml_models_trained[symbol] = True
                    print(f"✅ {symbol} için ML modeli yüklendi")
                    continue
            
            # Yeni model eğit
            if len(df) >= 200:  # Minimum veri gereksinimi
                print(f"🔄 {symbol} için ML modeli eğitiliyor...")
                model = self.ml_engine.train_model(df, model_type=f"{symbol}_random_forest")
                
                if model is not None:
                    self.ml_models_trained[symbol] = True
                    print(f"✅ {symbol} ML modeli eğitildi ve kaydedildi")
                else:
                    print(f"❌ {symbol} ML modeli eğitilemedi")
                    self.ml_models_trained[symbol] = False
            else:
                print(f"⚠️ {symbol} için yetersiz veri (minimum 200 gerekli)")
                self.ml_models_trained[symbol] = False
    
    def get_enhanced_signal(self, df: pd.DataFrame, strategy_info: dict, 
                           symbol: str, portfolio_value: float) -> dict:
        """
        Geleneksel strateji + ML + Risk yönetimi birleşik sinyal.
        """
        if len(df) < 100:
            return {
                'signal': 'HOLD',
                'confidence': 0.0,
                'components': {
                    'traditional': 'HOLD',
                    'ml': 'HOLD',
                    'risk': 'NEUTRAL'
                },
                'risk_metrics': {}
            }
        
        # 1. Geleneksel strateji sinyali
        traditional_signal = self._get_traditional_signal(df, strategy_info)
        
        # 2. ML sinyali (eğer model mevcutsa)
        ml_signal = 'HOLD'
        ml_confidence = 0.0
        ml_strength = 0.0
        
        if self.ml_models_trained.get(symbol, False):
            try:
                # Model yükle (cache mekanizması olabilir)
                if self.ml_engine.load_model(f"{symbol}_random_forest"):
                    ml_result = self.ml_engine.predict_market_direction(df)
                    ml_signal = ml_result['prediction']
                    ml_confidence = ml_result['confidence']
                    ml_strength = self.ml_engine.get_ml_signal_strength(df, self.ml_confidence_threshold)
            except Exception as e:
                print(f"⚠️ ML tahmin hatası ({symbol}): {e}")
        
        # 3. Risk analizi
        risk_analysis = self.risk_manager.calculate_risk_metrics(df['close'], portfolio_value)
        risk_level = risk_analysis.get('risk_assessment', {}).get('risk_level', 'MEDIUM')
        
        # Risk bazlı pozisyon büyüklüğü önerisi
        position_sizing = self.risk_manager.calculate_position_sizing(
            df['close'], portfolio_value, max_risk_per_trade=0.02
        )
        
        # 4. Sinyal birleştirme
        final_signal, final_confidence = self._combine_signals(
            traditional_signal, ml_signal, ml_confidence, ml_strength, risk_level
        )
        
        return {
            'signal': final_signal,
            'confidence': final_confidence,
            'components': {
                'traditional': traditional_signal,
                'ml': ml_signal,
                'ml_confidence': ml_confidence,
                'ml_strength': ml_strength,
                'risk_level': risk_level
            },
            'risk_metrics': {
                'var_percent': risk_analysis.get('var_historical', {}).get('var_percent', 0),
                'cvar_percent': risk_analysis.get('cvar', {}).get('cvar_percent', 0),
                'max_drawdown': risk_analysis.get('drawdown', {}).get('max_drawdown_percent', 0),
                'volatility_annual': risk_analysis.get('volatility', {}).get('volatility_annual', 0),
                'sharpe_ratio': risk_analysis.get('performance_ratios', {}).get('sharpe_ratio', 0)
            },
            'position_sizing': position_sizing,
            'market_regime': detect_market_regime(df)
        }
    
    def _get_traditional_signal(self, df: pd.DataFrame, strategy_info: dict) -> str:
        """
        Geleneksel (mevcut) strateji mantığını kullanarak sinyal üretir.
        """
        df_with_indicators = calculate_strategy_indicators(df, strategy_info)
        params = strategy_info.get('params', {})
        
        if len(df_with_indicators) < 2:
            return 'HOLD'
        
        prev_row = df_with_indicators.iloc[-2]
        curr_row = df_with_indicators.iloc[-1]
        
        # Buy signal check
        buy_rules = params.get('buy_rules', [])
        buy_quorum = params.get('buy_quorum', len(buy_rules))
        buy_votes = sum(_evaluate_rule(rule, prev_row, curr_row) for rule in buy_rules)
        
        if buy_votes >= buy_quorum:
            return 'BUY'
        
        # Sell signal check
        sell_rules = params.get('sell_rules', [])
        sell_quorum = params.get('sell_quorum', len(sell_rules))
        sell_votes = sum(_evaluate_rule(rule, prev_row, curr_row) for rule in sell_rules)
        
        if sell_votes >= sell_quorum:
            return 'SELL'
        
        return 'HOLD'
    
    def _combine_signals(self, traditional: str, ml: str, ml_conf: float, 
                        ml_strength: float, risk_level: str) -> Tuple[str, float]:
        """
        Farklı sinyalleri birleştirerek final karar verir.
        """
        # Signal scoring
        signal_scores = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        total_weight = 0
        
        # Traditional strategy weight
        traditional_weight = 0.4
        signal_scores[traditional] += traditional_weight
        total_weight += traditional_weight
        
        # ML signal weight (confidence'e dayalı)
        if ml_conf >= self.ml_confidence_threshold:
            ml_weight = min(0.6, ml_conf)  # Max 60% weight for ML
            signal_scores[ml] += ml_weight
            total_weight += ml_weight
        
        # Risk adjustment
        risk_weights = {
            'VERY LOW': 1.2,    # Risk düşükse daha agresif
            'LOW': 1.1,
            'MEDIUM': 1.0,      # Baseline
            'HIGH': 0.8,        # Risk yüksekse daha konservatif
            'VERY HIGH': 0.5
        }
        
        risk_multiplier = risk_weights.get(risk_level, 1.0)
        
        # Final scoring with risk adjustment
        for signal in signal_scores:
            if signal in ['BUY', 'SELL']:  # Apply risk adjustment only to action signals
                signal_scores[signal] *= risk_multiplier
        
        # Final decision
        final_signal = max(signal_scores, key=signal_scores.get)
        max_score = signal_scores[final_signal]
        
        # Confidence calculation
        if total_weight > 0:
            confidence = min(0.95, max_score / total_weight)
        else:
            confidence = 0.0
        
        # Minimum confidence threshold
        if confidence < 0.3 or final_signal == 'HOLD':
            return 'HOLD', confidence
        
        return final_signal, confidence
    
    def run_enhanced_backtest(self, df: pd.DataFrame, strategy_info: dict, 
                             symbol: str) -> dict:
        """
        ML ve risk entegre edilmiş gelişmiş backtest.
        """
        print(f"🧪 Enhanced backtest başlatılıyor: {symbol}")
        
        # ML modelinin hazır olduğundan emin ol
        if symbol not in self.ml_models_trained:
            if len(df) >= 200:
                print(f"🔄 {symbol} için ML modeli eğitiliyor...")
                self.ml_engine.train_model(df.iloc[:150], model_type=f"{symbol}_random_forest")
                self.ml_models_trained[symbol] = True
        
        # Backtest variables
        cash = self.initial_capital
        position = 0.0
        entry_price = 0.0
        in_position = False
        stop_loss_price = 0.0
        
        trades = []
        equity_curve = []
        risk_history = []
        
        # Start backtest from a reasonable point
        start_idx = max(100, len(df) // 4)  # Skip first 25% or minimum 100 bars
        
        for i in range(start_idx, len(df)):
            current_data = df.iloc[:i+1]
            current_price = current_data.iloc[-1]['close']
            current_time = current_data.iloc[-1]['timestamp']
            
            # Current portfolio value
            if in_position:
                portfolio_value = position * current_price
            else:
                portfolio_value = cash
            
            equity_curve.append({
                'timestamp': current_time,
                'portfolio_value': portfolio_value,
                'price': current_price,
                'in_position': in_position
            })
            
            # Get enhanced signal
            if i % 20 == 0:  # Calculate risk metrics every 20 bars (for performance)
                try:
                    signal_result = self.get_enhanced_signal(
                        current_data, strategy_info, symbol, portfolio_value
                    )
                    
                    risk_history.append({
                        'timestamp': current_time,
                        'var_percent': signal_result['risk_metrics'].get('var_percent', 0),
                        'risk_level': signal_result['components'].get('risk_level', 'MEDIUM'),
                        'ml_confidence': signal_result['components'].get('ml_confidence', 0)
                    })
                except Exception as e:
                    print(f"⚠️ Signal hesaplama hatası: {e}")
                    continue
            else:
                # Quick signal for other bars
                signal_result = {'signal': 'HOLD', 'confidence': 0.0}
                try:
                    signal_result['signal'] = self._get_traditional_signal(current_data, strategy_info)
                except:
                    pass
            
            signal = signal_result['signal']
            confidence = signal_result.get('confidence', 0.0)
            
            # Stop loss check
            if in_position and USE_STOP_LOSS and stop_loss_price > 0:
                if current_data.iloc[-1]['low'] <= stop_loss_price:
                    # Stop loss triggered
                    cash = position * stop_loss_price
                    pnl_percent = (stop_loss_price / entry_price - 1) * 100
                    
                    trades.append({
                        'type': 'STOP_LOSS',
                        'entry_price': entry_price,
                        'exit_price': stop_loss_price,
                        'pnl_percent': pnl_percent,
                        'timestamp': current_time,
                        'confidence': confidence
                    })
                    
                    position = 0.0
                    in_position = False
                    stop_loss_price = 0.0
                    continue
            
            # Trading logic
            if not in_position and signal == 'BUY' and confidence > 0.4:
                # Open position
                position = cash / current_price
                entry_price = current_price
                cash = 0.0
                in_position = True
                
                # Set stop loss
                if USE_STOP_LOSS:
                    try:
                        df_with_atr = calculate_strategy_indicators(current_data, strategy_info)
                        atr_col = f"ATRr_{ATR_PERIOD}"
                        if atr_col in df_with_atr.columns and not pd.isna(df_with_atr.iloc[-1][atr_col]):
                            atr_value = df_with_atr.iloc[-1][atr_col]
                            stop_loss_price = current_price - (atr_value * ATR_MULTIPLIER)
                    except:
                        stop_loss_price = current_price * 0.95  # 5% stop loss fallback
                
                trades.append({
                    'type': 'BUY',
                    'entry_price': entry_price,
                    'timestamp': current_time,
                    'confidence': confidence
                })
                
            elif in_position and signal == 'SELL' and confidence > 0.4:
                # Close position
                exit_price = current_price
                cash = position * exit_price
                pnl_percent = (exit_price / entry_price - 1) * 100
                
                trades.append({
                    'type': 'SELL',
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl_percent': pnl_percent,
                    'timestamp': current_time,
                    'confidence': confidence
                })
                
                position = 0.0
                in_position = False
                stop_loss_price = 0.0
        
        # Final calculations
        final_value = cash if not in_position else position * df.iloc[-1]['close']
        total_return = (final_value / self.initial_capital - 1) * 100
        
        # Trade analysis
        completed_trades = [t for t in trades if 'pnl_percent' in t]
        total_trades = len(completed_trades)
        
        if total_trades > 0:
            winning_trades = len([t for t in completed_trades if t['pnl_percent'] > 0])
            win_rate = (winning_trades / total_trades) * 100
            avg_win = np.mean([t['pnl_percent'] for t in completed_trades if t['pnl_percent'] > 0]) if winning_trades > 0 else 0
            avg_loss = np.mean([t['pnl_percent'] for t in completed_trades if t['pnl_percent'] < 0]) if (total_trades - winning_trades) > 0 else 0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
        
        # Drawdown calculation
        if equity_curve:
            equity_df = pd.DataFrame(equity_curve)
            peak = equity_df['portfolio_value'].expanding().max()
            drawdown = (equity_df['portfolio_value'] - peak) / peak
            max_drawdown = abs(drawdown.min()) * 100
        else:
            max_drawdown = 0
        
        # Advanced metrics
        if len(equity_curve) > 1:
            equity_df = pd.DataFrame(equity_curve)
            daily_returns = equity_df['portfolio_value'].pct_change().dropna()
            
            if len(daily_returns) > 0 and daily_returns.std() > 0:
                sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
                
                # Sortino ratio
                downside_returns = daily_returns[daily_returns < 0]
                sortino_ratio = daily_returns.mean() / downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
                
                # Calmar ratio
                annual_return = daily_returns.mean() * 252
                calmar_ratio = annual_return / max(max_drawdown / 100, 0.001) if max_drawdown > 0 else 0
            else:
                sharpe_ratio = 0
                sortino_ratio = 0
                calmar_ratio = 0
        else:
            sharpe_ratio = 0
            sortino_ratio = 0
            calmar_ratio = 0
        
        # ML contribution analysis
        ml_trades = [t for t in completed_trades if t.get('confidence', 0) > 0.6]
        ml_contribution = 0
        if len(ml_trades) > 0:
            ml_pnl = sum([t['pnl_percent'] for t in ml_trades])
            total_pnl = sum([t['pnl_percent'] for t in completed_trades])
            ml_contribution = (ml_pnl / max(abs(total_pnl), 1)) * 100 if total_pnl != 0 else 0
        
        results = {
            'total_return': total_return,
            'final_value': final_value,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'ml_contribution': ml_contribution,
            'ml_trades_count': len(ml_trades),
            'trades': trades,
            'equity_curve': equity_curve,
            'risk_history': risk_history,
            'strategy_info': strategy_info,
            'symbol': symbol
        }
        
        print(f"✅ Enhanced backtest tamamlandı: {symbol}")
        print(f"   Total Return: {total_return:.2f}%")
        print(f"   Win Rate: {win_rate:.1f}%")
        print(f"   Max Drawdown: {max_drawdown:.2f}%")
        print(f"   Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"   ML Contribution: {ml_contribution:.1f}%")
        
        return results
    
    def generate_portfolio_risk_report(self, portfolio_positions: dict, 
                                     market_data: dict) -> dict:
        """
        Portföy seviyesinde kapsamlı risk raporu.
        """
        print("📊 Portföy risk raporu oluşturuluyor...")
        
        portfolio_risk = self.risk_manager.calculate_portfolio_risk(
            portfolio_positions, market_data
        )
        
        # Enhanced risk metrics
        total_value = portfolio_risk.get('total_portfolio_value', 0)
        portfolio_var = portfolio_risk.get('portfolio_var_percent', 0)
        
        # Position-level ML predictions
        position_ml_signals = {}
        for symbol in portfolio_positions.keys():
            if symbol in market_data and self.ml_models_trained.get(symbol, False):
                try:
                    df = pd.DataFrame({
                        'close': market_data[symbol],
                        'timestamp': pd.date_range(end='today', periods=len(market_data[symbol]), freq='H')
                    })
                    ml_result = self.ml_engine.predict_market_direction(df)
                    position_ml_signals[symbol] = ml_result
                except Exception as e:
                    print(f"⚠️ ML tahmin hatası ({symbol}): {e}")
        
        # Risk-adjusted recommendations
        risk_recommendations = portfolio_risk.get('recommendations', [])
        
        # ML-based recommendations
        ml_recommendations = []
        for symbol, ml_result in position_ml_signals.items():
            if ml_result['confidence'] > 0.7:
                if ml_result['prediction'] == 'SELL':
                    ml_recommendations.append(f"Consider reducing {symbol} position (ML confidence: {ml_result['confidence']:.2f})")
                elif ml_result['prediction'] == 'BUY':
                    ml_recommendations.append(f"Consider increasing {symbol} position (ML confidence: {ml_result['confidence']:.2f})")
        
        # Combine all recommendations
        all_recommendations = risk_recommendations + ml_recommendations
        
        return {
            'portfolio_value': total_value,
            'portfolio_var_percent': portfolio_var,
            'risk_level': self._assess_portfolio_risk_level(portfolio_var),
            'position_risks': portfolio_risk.get('position_risks', []),
            'diversification_score': portfolio_risk.get('diversification_score', 0),
            'ml_signals': position_ml_signals,
            'recommendations': all_recommendations[:10],  # Limit to top 10
            'detailed_risk_metrics': portfolio_risk,
            'report_timestamp': datetime.now().isoformat()
        }
    
    def _assess_portfolio_risk_level(self, var_percent: float) -> str:
        """Portföy VaR'ına göre risk seviyesi."""
        if var_percent > 15:
            return "VERY HIGH"
        elif var_percent > 10:
            return "HIGH"
        elif var_percent > 5:
            return "MEDIUM"
        elif var_percent > 2:
            return "LOW"
        else:
            return "VERY LOW"
    
    def optimize_strategy_parameters(self, df: pd.DataFrame, strategy_info: dict, 
                                   symbol: str, optimization_metric: str = 'sharpe_ratio') -> dict:
        """
        ML destekli strateji parametre optimizasyonu.
        """
        print(f"🔧 Strateji parametreleri optimize ediliyor: {symbol}")
        
        base_results = self.run_enhanced_backtest(df, strategy_info, symbol)
        best_params = strategy_info.copy()
        best_score = base_results.get(optimization_metric, 0)
        
        # Parameter ranges to test (simplified for demo)
        param_variations = self._generate_parameter_variations(strategy_info)
        
        optimization_results = []
        
        for i, variant_params in enumerate(param_variations[:10]):  # Limit to 10 variants for performance
            print(f"  Variant {i+1}/10 test ediliyor...")
            
            try:
                variant_results = self.run_enhanced_backtest(df, variant_params, symbol)
                variant_score = variant_results.get(optimization_metric, 0)
                
                optimization_results.append({
                    'parameters': variant_params,
                    'score': variant_score,
                    'results': variant_results
                })
                
                if variant_score > best_score:
                    best_score = variant_score
                    best_params = variant_params
                    
            except Exception as e:
                print(f"    ⚠️ Variant test hatası: {e}")
                continue
        
        return {
            'original_strategy': strategy_info,
            'optimized_strategy': best_params,
            'original_score': base_results.get(optimization_metric, 0),
            'optimized_score': best_score,
            'improvement': best_score - base_results.get(optimization_metric, 0),
            'optimization_metric': optimization_metric,
            'all_results': optimization_results
        }
    
    def _generate_parameter_variations(self, strategy_info: dict) -> list:
        """
        Strateji parametrelerinin varyasyonlarını üretir.
        """
        variations = []
        params = strategy_info.get('params', {})
        
        # Buy/sell quorum variations
        for buy_quorum in [1, 2]:
            for sell_quorum in [1, 2]:
                if 'buy_rules' in params and 'sell_rules' in params:
                    variant = strategy_info.copy()
                    variant['params'] = params.copy()
                    variant['params']['buy_quorum'] = min(buy_quorum, len(params.get('buy_rules', [])))
                    variant['params']['sell_quorum'] = min(sell_quorum, len(params.get('sell_rules', [])))
                    variations.append(variant)
        
        # Rule parameter variations (simplified)
        if 'buy_rules' in params:
            for rule in params['buy_rules']:
                if rule.get('indicator') == 'RSI':
                    for rsi_period in [10, 14, 21]:
                        variant = strategy_info.copy()
                        variant['params'] = params.copy()
                        variant['params']['buy_rules'] = params['buy_rules'].copy()
                        # Update RSI period (simplified)
                        new_rule = rule.copy()
                        new_rule['params'] = rule['params'].copy()
                        new_rule['params']['rsi_period'] = rsi_period
                        
                        variant_rules = []
                        for r in variant['params']['buy_rules']:
                            if r.get('indicator') == 'RSI':
                                variant_rules.append(new_rule)
                            else:
                                variant_rules.append(r)
                        variant['params']['buy_rules'] = variant_rules
                        variations.append(variant)
        
        return variations[:20]  # Return max 20 variations
    
    def save_performance_history(self, results: dict, filepath: str = "performance_history.json"):
        """
        Performans geçmişini kaydeder.
        """
        self.performance_history.append({
            'timestamp': datetime.now().isoformat(),
            'results': results
        })
        
        try:
            with open(filepath, 'w') as f:
                json.dump(self.performance_history, f, indent=2, default=str)
            print(f"✅ Performans geçmişi kaydedildi: {filepath}")
        except Exception as e:
            print(f"⚠️ Performans geçmişi kaydetme hatası: {e}")


# Convenience functions
def create_enhanced_engine(initial_capital: float = 10000, ml_confidence: float = 0.65):
    """Enhanced strategy engine oluşturur."""
    return EnhancedStrategyEngine(initial_capital, ml_confidence)

def run_ml_backtest(df: pd.DataFrame, strategy_info: dict, symbol: str, 
                    initial_capital: float = 10000) -> dict:
    """ML destekli backtest çalıştırır."""
    engine = EnhancedStrategyEngine(initial_capital)
    return engine.run_enhanced_backtest(df, strategy_info, symbol)