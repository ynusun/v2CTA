import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class AdvancedRiskManager:
    """
    Gelişmiş risk yönetimi modeli.
    VaR (Value at Risk), CVaR (Conditional VaR) ve diğer risk metriklerini hesaplar.
    """
    
    def __init__(self, confidence_level=0.95, holding_period=1):
        """
        Args:
            confidence_level: VaR confidence level (default: 95%)
            holding_period: Holding period in days (default: 1 day)
        """
        self.confidence_level = confidence_level
        self.holding_period = holding_period
        self.alpha = 1 - confidence_level  # Risk level
        
    def calculate_returns(self, prices: pd.Series) -> pd.Series:
        """Logaritmik getirileri hesaplar."""
        return np.log(prices / prices.shift(1)).dropna()
    
    def calculate_var_historical(self, returns: pd.Series, portfolio_value: float) -> dict:
        """
        Historical VaR hesaplama (Tarihsel Simülasyon Metodu)
        """
        if len(returns) < 30:
            return {
                'var_amount': 0,
                'var_percent': 0,
                'method': 'historical',
                'observations': len(returns),
                'warning': 'Insufficient data for reliable VaR calculation'
            }
        
        # Getiri dağılımından percentile hesapla
        var_percentile = np.percentile(returns, self.alpha * 100)
        
        # VaR değerlerini hesapla
        var_amount = abs(var_percentile * portfolio_value * np.sqrt(self.holding_period))
        var_percent = abs(var_percentile * 100)
        
        return {
            'var_amount': var_amount,
            'var_percent': var_percent,
            'var_return': var_percentile,
            'method': 'historical',
            'observations': len(returns),
            'confidence_level': self.confidence_level
        }
    
    def calculate_var_parametric(self, returns: pd.Series, portfolio_value: float) -> dict:
        """
        Parametric VaR hesaplama (Normal dağılım varsayımı)
        """
        if len(returns) < 10:
            return {
                'var_amount': 0,
                'var_percent': 0,
                'method': 'parametric',
                'warning': 'Insufficient data for parametric VaR'
            }
        
        # İstatistikleri hesapla
        mean_return = returns.mean()
        std_return = returns.std()
        
        # Z-score for confidence level
        z_score = stats.norm.ppf(self.alpha)
        
        # VaR hesapla (holding period için adjust et)
        var_return = mean_return + z_score * std_return * np.sqrt(self.holding_period)
        var_amount = abs(var_return * portfolio_value)
        var_percent = abs(var_return * 100)
        
        # Normallik testi
        _, normality_p_value = stats.shapiro(returns[-min(5000, len(returns)):])
        is_normal = normality_p_value > 0.05
        
        return {
            'var_amount': var_amount,
            'var_percent': var_percent,
            'var_return': var_return,
            'method': 'parametric',
            'mean_return': mean_return,
            'volatility': std_return,
            'is_normal_dist': is_normal,
            'normality_p_value': normality_p_value,
            'observations': len(returns),
            'confidence_level': self.confidence_level
        }
    
    def calculate_cvar(self, returns: pd.Series, portfolio_value: float) -> dict:
        """
        Conditional VaR (Expected Shortfall) hesaplama.
        VaR'ı aşan kayıpların beklenen değeri.
        """
        if len(returns) < 30:
            return {
                'cvar_amount': 0,
                'cvar_percent': 0,
                'warning': 'Insufficient data for CVaR calculation'
            }
        
        # İlk önce VaR'ı hesapla
        var_result = self.calculate_var_historical(returns, portfolio_value)
        var_return = var_result['var_return']
        
        # VaR'ı aşan kayıpları bul
        extreme_losses = returns[returns <= var_return]
        
        if len(extreme_losses) == 0:
            # Eğer VaR'ı aşan kayıp yoksa, en kötü kaybı al
            cvar_return = returns.min()
        else:
            # CVaR = VaR'ı aşan kayıpların ortalaması
            cvar_return = extreme_losses.mean()
        
        cvar_amount = abs(cvar_return * portfolio_value * np.sqrt(self.holding_period))
        cvar_percent = abs(cvar_return * 100)
        
        return {
            'cvar_amount': cvar_amount,
            'cvar_percent': cvar_percent,
            'cvar_return': cvar_return,
            'var_exceedances': len(extreme_losses),
            'confidence_level': self.confidence_level,
            'cvar_var_ratio': cvar_amount / max(var_result['var_amount'], 1)
        }
    
    def calculate_maximum_drawdown(self, prices: pd.Series) -> dict:
        """
        Maksimum Drawdown hesaplama.
        """
        if len(prices) < 2:
            return {'max_drawdown': 0, 'max_drawdown_percent': 0}
        
        # Kümülatif maksimum hesapla
        cumulative_max = prices.expanding().max()
        drawdown = (prices - cumulative_max) / cumulative_max
        
        max_drawdown = drawdown.min()
        max_drawdown_percent = max_drawdown * 100
        
        # Drawdown periyodlarını bul
        max_drawdown_idx = drawdown.idxmin()
        
        return {
            'max_drawdown': abs(max_drawdown),
            'max_drawdown_percent': abs(max_drawdown_percent),
            'max_drawdown_date': max_drawdown_idx,
            'current_drawdown': abs(drawdown.iloc[-1]),
            'current_drawdown_percent': abs(drawdown.iloc[-1] * 100)
        }
    
    def calculate_volatility_metrics(self, returns: pd.Series) -> dict:
        """
        Çeşitli volatilite metriklerini hesaplar.
        """
        if len(returns) < 10:
            return {'volatility_daily': 0, 'volatility_annual': 0}
        
        # Temel volatilite
        daily_vol = returns.std()
        annual_vol = daily_vol * np.sqrt(252)  # 252 trading days
        
        # EWMA volatilite (Exponentially Weighted Moving Average)
        ewma_vol = returns.ewm(alpha=0.06).std().iloc[-1]  # RiskMetrics style lambda=0.94
        
        # GARCH-like volatility (simple approach)
        returns_squared = returns ** 2
        garch_vol = returns_squared.ewm(alpha=0.1).mean().iloc[-1] ** 0.5
        
        # Volatilite persistence
        vol_window = 20
        if len(returns) >= vol_window * 2:
            recent_vol = returns.iloc[-vol_window:].std()
            past_vol = returns.iloc[-vol_window*2:-vol_window].std()
            vol_persistence = recent_vol / max(past_vol, 0.001)
        else:
            vol_persistence = 1.0
        
        return {
            'volatility_daily': daily_vol,
            'volatility_annual': annual_vol,
            'volatility_ewma': ewma_vol,
            'volatility_garch_like': garch_vol,
            'volatility_persistence': vol_persistence,
            'volatility_percentile_95': np.percentile(returns.abs(), 95)
        }
    
    def calculate_risk_metrics(self, prices: pd.Series, portfolio_value: float = 10000) -> dict:
        """
        Tüm risk metriklerini hesaplar ve kapsamlı risk raporu oluşturur.
        """
        returns = self.calculate_returns(prices)
        
        if len(returns) < 10:
            return {
                'error': 'Insufficient data for risk calculation',
                'data_points': len(returns)
            }
        
        # Temel istatistikler
        basic_stats = {
            'mean_return': returns.mean(),
            'median_return': returns.median(),
            'std_return': returns.std(),
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'min_return': returns.min(),
            'max_return': returns.max(),
            'observations': len(returns)
        }
        
        # VaR hesaplamaları
        var_historical = self.calculate_var_historical(returns, portfolio_value)
        var_parametric = self.calculate_var_parametric(returns, portfolio_value)
        
        # CVaR hesaplaması
        cvar = self.calculate_cvar(returns, portfolio_value)
        
        # Drawdown analizi
        drawdown = self.calculate_maximum_drawdown(prices)
        
        # Volatilite metrikleri
        volatility = self.calculate_volatility_metrics(returns)
        
        # Sharpe ratio (risk-free rate = 0 varsayımı)
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        sortino_ratio = returns.mean() / downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 and downside_returns.std() > 0 else 0
        
        # Calmar ratio (Annual return / Max Drawdown)
        annual_return = returns.mean() * 252
        calmar_ratio = annual_return / max(drawdown['max_drawdown'], 0.001) if drawdown['max_drawdown'] > 0 else 0
        
        # Risk consolidation
        risk_metrics = {
            'basic_statistics': basic_stats,
            'var_historical': var_historical,
            'var_parametric': var_parametric,
            'cvar': cvar,
            'drawdown': drawdown,
            'volatility': volatility,
            'performance_ratios': {
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio
            },
            'risk_assessment': self._assess_risk_level(var_historical, cvar, drawdown, volatility),
            'portfolio_value': portfolio_value,
            'confidence_level': self.confidence_level,
            'calculation_date': datetime.now().isoformat()
        }
        
        return risk_metrics
    
    def _assess_risk_level(self, var_result: dict, cvar_result: dict, 
                          drawdown_result: dict, volatility_result: dict) -> dict:
        """
        Risk seviyesini kategorilere ayırır ve genel risk skorunu hesaplar.
        """
        risk_score = 0
        risk_factors = []
        
        # VaR bazlı risk skorlaması
        var_percent = var_result.get('var_percent', 0)
        if var_percent > 5:
            risk_score += 3
            risk_factors.append("High daily VaR (>5%)")
        elif var_percent > 3:
            risk_score += 2
            risk_factors.append("Medium daily VaR (3-5%)")
        elif var_percent > 1:
            risk_score += 1
            risk_factors.append("Low daily VaR (1-3%)")
        
        # CVaR bazlı risk skorlaması
        cvar_var_ratio = cvar_result.get('cvar_var_ratio', 1)
        if cvar_var_ratio > 1.5:
            risk_score += 2
            risk_factors.append("High tail risk (CVaR/VaR > 1.5)")
        elif cvar_var_ratio > 1.2:
            risk_score += 1
            risk_factors.append("Medium tail risk")
        
        # Drawdown bazlı risk skorlaması
        max_dd = drawdown_result.get('max_drawdown_percent', 0)
        if max_dd > 20:
            risk_score += 3
            risk_factors.append("High maximum drawdown (>20%)")
        elif max_dd > 10:
            risk_score += 2
            risk_factors.append("Medium maximum drawdown (10-20%)")
        elif max_dd > 5:
            risk_score += 1
            risk_factors.append("Low maximum drawdown (5-10%)")
        
        # Volatilite bazlı risk skorlaması
        annual_vol = volatility_result.get('volatility_annual', 0)
        if annual_vol > 0.5:  # >50% annual volatility
            risk_score += 2
            risk_factors.append("High volatility (>50% annual)")
        elif annual_vol > 0.3:  # 30-50% annual volatility
            risk_score += 1
            risk_factors.append("Medium volatility (30-50% annual)")
        
        # Risk seviyesi belirleme
        if risk_score >= 7:
            risk_level = "VERY HIGH"
        elif risk_score >= 5:
            risk_level = "HIGH"
        elif risk_score >= 3:
            risk_level = "MEDIUM"
        elif risk_score >= 1:
            risk_level = "LOW"
        else:
            risk_level = "VERY LOW"
        
        return {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'risk_factors': risk_factors,
            'recommendation': self._get_risk_recommendation(risk_level)
        }
    
    def _get_risk_recommendation(self, risk_level: str) -> str:
        """Risk seviyesine göre öneriler."""
        recommendations = {
            "VERY HIGH": "Pozisyon boyutunu azalt, stop-loss seviyelerini sıkılaştır, çeşitlendirmeyi artır",
            "HIGH": "Risk yönetimini güçlendir, pozisyon boyutunu kontrol et",
            "MEDIUM": "Mevcut risk parametreleri kabul edilebilir, düzenli izleme yap",
            "LOW": "Risk seviyeleri düşük, opportunistic pozisyonlar değerlendirilebilir",
            "VERY LOW": "Risk çok düşük, pozisyon boyutu artırılabilir"
        }
        return recommendations.get(risk_level, "Risk seviyesi değerlendirme dışı")
    
    def calculate_position_sizing(self, prices: pd.Series, total_capital: float, 
                                 max_risk_per_trade: float = 0.02) -> dict:
        """
        VaR bazlı pozisyon büyüklüğü hesaplama.
        
        Args:
            prices: Fiyat serisi
            total_capital: Toplam sermaye
            max_risk_per_trade: Trade başına maksimum risk (% olarak)
        """
        returns = self.calculate_returns(prices)
        
        if len(returns) < 10:
            return {
                'suggested_position_size': 0,
                'max_position_value': 0,
                'risk_per_unit': 0,
                'warning': 'Insufficient data for position sizing'
            }
        
        # 1-day VaR hesapla
        var_result = self.calculate_var_historical(returns, 1.0)  # Per unit risk
        risk_per_unit = var_result.get('var_percent', 5) / 100  # Convert to decimal
        
        # Maksimum risk tutarı
        max_risk_amount = total_capital * max_risk_per_trade
        
        # Pozisyon büyüklüğü hesapla
        current_price = prices.iloc[-1]
        max_units = max_risk_amount / (risk_per_unit * current_price) if risk_per_unit > 0 else 0
        max_position_value = max_units * current_price
        
        # Position sizing constraints
        max_position_percent = min(max_position_value / total_capital, 0.25)  # Max %25 of capital
        adjusted_position_value = total_capital * max_position_percent
        adjusted_units = adjusted_position_value / current_price
        
        return {
            'suggested_position_size': adjusted_units,
            'max_position_value': adjusted_position_value,
            'position_percent_of_capital': max_position_percent * 100,
            'risk_per_unit_percent': risk_per_unit * 100,
            'expected_risk_amount': adjusted_units * risk_per_unit * current_price,
            'risk_reward_metrics': {
                'current_price': current_price,
                'var_based_stop_loss': current_price * (1 - risk_per_unit * 2),  # 2x VaR for stop
                'max_risk_per_trade_percent': max_risk_per_trade * 100
            }
        }
    
    def calculate_portfolio_risk(self, portfolio_positions: dict, 
                                market_data: dict) -> dict:
        """
        Portföy seviyesinde risk hesaplama.
        
        Args:
            portfolio_positions: {'SYMBOL': {'quantity': float, 'entry_price': float}}
            market_data: {'SYMBOL': pd.Series} (prices)
        """
        if not portfolio_positions:
            return {'total_portfolio_risk': 0, 'positions': []}
        
        position_risks = []
        total_var = 0
        total_cvar = 0
        total_value = 0
        
        for symbol, position in portfolio_positions.items():
            if symbol not in market_data:
                continue
                
            prices = market_data[symbol]
            quantity = position['quantity']
            current_price = prices.iloc[-1]
            position_value = quantity * current_price
            total_value += position_value
            
            # Individual position risk
            risk_metrics = self.calculate_risk_metrics(prices, position_value)
            
            if 'var_historical' in risk_metrics:
                pos_var = risk_metrics['var_historical'].get('var_amount', 0)
                pos_cvar = risk_metrics['cvar'].get('cvar_amount', 0)
                
                total_var += pos_var ** 2  # For portfolio VaR calculation (assuming independence)
                total_cvar += pos_cvar
                
                position_risks.append({
                    'symbol': symbol,
                    'position_value': position_value,
                    'var_amount': pos_var,
                    'cvar_amount': pos_cvar,
                    'risk_contribution': pos_var / max(total_value, 1) * 100,
                    'max_drawdown': risk_metrics['drawdown'].get('max_drawdown_percent', 0)
                })
        
        # Portfolio-level VaR (simplified - assumes independence)
        portfolio_var = np.sqrt(total_var)
        portfolio_cvar = total_cvar
        
        # Risk concentration
        if position_risks:
            max_risk_contribution = max([p['risk_contribution'] for p in position_risks])
            risk_concentration = "HIGH" if max_risk_contribution > 50 else "MEDIUM" if max_risk_contribution > 30 else "LOW"
        else:
            risk_concentration = "N/A"
        
        return {
            'total_portfolio_value': total_value,
            'portfolio_var': portfolio_var,
            'portfolio_cvar': portfolio_cvar,
            'portfolio_var_percent': (portfolio_var / max(total_value, 1)) * 100,
            'number_of_positions': len(position_risks),
            'risk_concentration': risk_concentration,
            'position_risks': sorted(position_risks, key=lambda x: x['var_amount'], reverse=True),
            'diversification_score': self._calculate_diversification_score(position_risks),
            'recommendations': self._get_portfolio_recommendations(position_risks, total_value)
        }
    
    def _calculate_diversification_score(self, position_risks: list) -> float:
        """
        Portföy çeşitlendirme skorunu hesaplar (0-100 arası).
        """
        if len(position_risks) <= 1:
            return 0.0
        
        # Risk contribution'larının eşit dağılımdan farkı
        equal_weight = 100 / len(position_risks)
        variance = sum([(p['risk_contribution'] - equal_weight) ** 2 for p in position_risks])
        max_variance = len(position_risks) * (100 - equal_weight) ** 2
        
        # Score: 100 = perfect diversification, 0 = no diversification
        diversification_score = max(0, 100 - (variance / max_variance * 100))
        
        return diversification_score
    
    def _get_portfolio_recommendations(self, position_risks: list, total_value: float) -> list:
        """Portföy risk yönetimi önerileri."""
        recommendations = []
        
        if not position_risks:
            return ["No positions to evaluate"]
        
        # High risk positions
        high_risk_positions = [p for p in position_risks if p['risk_contribution'] > 40]
        if high_risk_positions:
            recommendations.append(f"Consider reducing exposure to high-risk positions: {[p['symbol'] for p in high_risk_positions]}")
        
        # Concentration risk
        if len(position_risks) < 3:
            recommendations.append("Consider adding more positions for better diversification")
        
        # Overall portfolio VaR
        if position_risks:
            avg_var_percent = sum([p['var_amount'] for p in position_risks]) / total_value * 100
            if avg_var_percent > 10:
                recommendations.append("Overall portfolio risk is high - consider reducing position sizes")
            elif avg_var_percent < 2:
                recommendations.append("Portfolio risk is low - consider increasing position sizes if appropriate")
        
        return recommendations if recommendations else ["Portfolio risk profile appears balanced"]


# Convenience functions
def analyze_symbol_risk(prices: pd.Series, portfolio_value: float = 10000, confidence_level: float = 0.95):
    """
    Tek bir sembol için hızlı risk analizi.
    """
    risk_manager = AdvancedRiskManager(confidence_level=confidence_level)
    return risk_manager.calculate_risk_metrics(prices, portfolio_value)

def get_position_sizing_recommendation(prices: pd.Series, total_capital: float, max_risk: float = 0.02):
    """
    VaR bazlı pozisyon büyüklüğü önerisi.
    """
    risk_manager = AdvancedRiskManager()
    return risk_manager.calculate_position_sizing(prices, total_capital, max_risk)