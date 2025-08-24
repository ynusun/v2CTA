import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class PatternRecognitionEngine:
    """
    Gelişmiş pattern recognition için ML motoru.
    Teknik göstergeleri kullanarak fiyat hareketlerini tahmin eder.
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.models_dir = "ml_models"
        self.lookback_period = 20  # Feature engineering için geriye bakış
        self.prediction_horizon = 5  # Kaç mum sonrasını tahmin ediyoruz
        
        # Model dosyalarını kaydetmek için dizin oluştur
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
    
    def extract_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Gelişmiş teknik analiz özelliklerini çıkarır.
        """
        df_copy = df.copy()
        
        # Temel fiyat özellikleri
        df_copy['returns'] = df_copy['close'].pct_change()
        df_copy['log_returns'] = np.log(df_copy['close'] / df_copy['close'].shift(1))
        
        # Volatilite özellikleri
        df_copy['volatility_5'] = df_copy['returns'].rolling(5).std()
        df_copy['volatility_20'] = df_copy['returns'].rolling(20).std()
        
        # Momentum göstergeleri
        for period in [5, 10, 20]:
            df_copy[f'rsi_{period}'] = self._calculate_rsi(df_copy['close'], period)
            df_copy[f'momentum_{period}'] = df_copy['close'] / df_copy['close'].shift(period) - 1
        
        # Moving Average özellikleri
        for ma_period in [5, 10, 20, 50]:
            df_copy[f'sma_{ma_period}'] = df_copy['close'].rolling(ma_period).mean()
            df_copy[f'sma_ratio_{ma_period}'] = df_copy['close'] / df_copy[f'sma_{ma_period}']
            df_copy[f'sma_slope_{ma_period}'] = (df_copy[f'sma_{ma_period}'] - df_copy[f'sma_{ma_period}'].shift(5)) / 5
        
        # MACD özellikleri
        exp1 = df_copy['close'].ewm(span=12).mean()
        exp2 = df_copy['close'].ewm(span=26).mean()
        df_copy['macd'] = exp1 - exp2
        df_copy['macd_signal'] = df_copy['macd'].ewm(span=9).mean()
        df_copy['macd_histogram'] = df_copy['macd'] - df_copy['macd_signal']
        
        # Bollinger Bands
        df_copy['bb_middle'] = df_copy['close'].rolling(20).mean()
        bb_std = df_copy['close'].rolling(20).std()
        df_copy['bb_upper'] = df_copy['bb_middle'] + (bb_std * 2)
        df_copy['bb_lower'] = df_copy['bb_middle'] - (bb_std * 2)
        df_copy['bb_position'] = (df_copy['close'] - df_copy['bb_lower']) / (df_copy['bb_upper'] - df_copy['bb_lower'])
        
        # Volume özellikleri (varsa)
        if 'volume' in df_copy.columns:
            df_copy['volume_sma_10'] = df_copy['volume'].rolling(10).mean()
            df_copy['volume_ratio'] = df_copy['volume'] / df_copy['volume_sma_10']
            
            # OBV (On Balance Volume)
            df_copy['obv'] = (np.sign(df_copy['returns']) * df_copy['volume']).cumsum()
            df_copy['obv_sma'] = df_copy['obv'].rolling(10).mean()
        
        # Price patterns (candlestick patterns simplification)
        df_copy['body_size'] = abs(df_copy['close'] - df_copy['open']) / df_copy['open']
        df_copy['upper_shadow'] = (df_copy['high'] - np.maximum(df_copy['open'], df_copy['close'])) / df_copy['open']
        df_copy['lower_shadow'] = (np.minimum(df_copy['open'], df_copy['close']) - df_copy['low']) / df_copy['open']
        
        # Support/Resistance levels approximation
        df_copy['high_20_max'] = df_copy['high'].rolling(20).max()
        df_copy['low_20_min'] = df_copy['low'].rolling(20).min()
        df_copy['resistance_distance'] = (df_copy['high_20_max'] - df_copy['close']) / df_copy['close']
        df_copy['support_distance'] = (df_copy['close'] - df_copy['low_20_min']) / df_copy['close']
        
        # Fractal-like features
        df_copy['local_max'] = (df_copy['high'] == df_copy['high'].rolling(5, center=True).max()).astype(int)
        df_copy['local_min'] = (df_copy['low'] == df_copy['low'].rolling(5, center=True).min()).astype(int)
        
        return df_copy
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """RSI hesaplama fonksiyonu"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def create_target_variable(self, df: pd.DataFrame) -> pd.Series:
        """
        Tahmin hedefini oluşturur.
        Gelecekteki fiyat hareketini sınıflandırır: 0=DOWN, 1=SIDEWAYS, 2=UP
        """
        future_returns = df['close'].shift(-self.prediction_horizon) / df['close'] - 1
        
        # Threshold'ları dinamik olarak belirle (volatiliteye dayalı)
        volatility = df['returns'].rolling(20).std()
        up_threshold = volatility * 1.5    # %1.5 volatilite üzeri = UP
        down_threshold = -volatility * 1.5  # -%1.5 volatilite altı = DOWN
        
        target = pd.Series(index=df.index, dtype=int)
        target[future_returns > up_threshold] = 2    # UP
        target[future_returns < down_threshold] = 0  # DOWN
        target[(future_returns >= down_threshold) & (future_returns <= up_threshold)] = 1  # SIDEWAYS
        
        return target
    
    def prepare_training_data(self, df: pd.DataFrame):
        """
        ML modeli için eğitim verisini hazırlar.
        """
        # Teknik özellikleri çıkar
        df_features = self.extract_technical_features(df)
        
        # Target variable oluştur
        target = self.create_target_variable(df_features)
        
        # Feature columns'ları seç (sayısal olanları)
        feature_cols = [col for col in df_features.columns if 
                       col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume'] and
                       df_features[col].dtype in ['float64', 'int64']]
        
        self.feature_columns = feature_cols
        
        # NaN değerleri temizle
        df_clean = df_features[feature_cols].copy()
        df_clean['target'] = target
        df_clean = df_clean.dropna()
        
        X = df_clean[feature_cols]
        y = df_clean['target']
        
        return X, y
    
    def train_model(self, df: pd.DataFrame, model_type='random_forest'):
        """
        ML modelini eğitir ve kaydeder.
        """
        print(f"🤖 ML Pattern Recognition modeli eğitiliyor...")
        
        X, y = self.prepare_training_data(df)
        
        if len(X) == 0:
            print("❌ Eğitim için yeterli temiz veri bulunamadı!")
            return None
        
        print(f"📊 Eğitim veri seti: {len(X)} örnek, {len(X.columns)} özellik")
        print(f"🎯 Target dağılımı: DOWN={sum(y==0)}, SIDEWAYS={sum(y==1)}, UP={sum(y==2)}")
        
        # Veriyi ölçeklendir
        X_scaled = self.scaler.fit_transform(X)
        
        # Model seçimi
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
        
        # Time series cross validation
        tscv = TimeSeriesSplit(n_splits=3)
        cv_scores = []
        
        for train_idx, val_idx in tscv.split(X_scaled):
            X_train_cv, X_val_cv = X_scaled[train_idx], X_scaled[val_idx]
            y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
            
            temp_model = self.model.__class__(**self.model.get_params())
            temp_model.fit(X_train_cv, y_train_cv)
            
            val_pred = temp_model.predict(X_val_cv)
            cv_score = accuracy_score(y_val_cv, val_pred)
            cv_scores.append(cv_score)
        
        print(f"📈 Cross-validation accuracy: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
        
        # Final model eğitimi
        self.model.fit(X_scaled, y)
        
        # Model ve scaler'ı kaydet
        model_path = os.path.join(self.models_dir, f"pattern_model_{model_type}.pkl")
        scaler_path = os.path.join(self.models_dir, f"pattern_scaler_{model_type}.pkl")
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        # Feature importance'ı yazdır
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\n🔍 En Önemli Özellikler:")
            for _, row in feature_importance.head(10).iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")
        
        return self.model
    
    def load_model(self, model_type='random_forest'):
        """
        Kaydedilmiş modeli yükler.
        """
        model_path = os.path.join(self.models_dir, f"pattern_model_{model_type}.pkl")
        scaler_path = os.path.join(self.models_dir, f"pattern_scaler_{model_type}.pkl")
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            print(f"✅ ML modeli yüklendi: {model_type}")
            return True
        else:
            print(f"❌ ML modeli bulunamadı: {model_type}")
            return False
    
    def predict_market_direction(self, df: pd.DataFrame) -> dict:
        """
        Mevcut piyasa verisi için tahmin yapar.
        """
        if self.model is None:
            return {'prediction': 'HOLD', 'confidence': 0.0, 'probabilities': [0.33, 0.33, 0.34]}
        
        # Son veriyi hazırla
        df_features = self.extract_technical_features(df)
        
        if len(df_features) == 0:
            return {'prediction': 'HOLD', 'confidence': 0.0, 'probabilities': [0.33, 0.33, 0.34]}
        
        # Son satırı al ve özellikler için hazırla
        last_row = df_features[self.feature_columns].iloc[-1:].fillna(method='ffill').fillna(0)
        
        try:
            # Ölçeklendir ve tahmin et
            X_scaled = self.scaler.transform(last_row)
            
            # Sınıf tahminleri ve olasılıkları
            prediction = self.model.predict(X_scaled)[0]
            probabilities = self.model.predict_proba(X_scaled)[0]
            
            # En yüksek olasılık = confidence
            confidence = np.max(probabilities)
            
            # Tahmin sonucunu string'e çevir
            prediction_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
            prediction_str = prediction_map.get(prediction, 'HOLD')
            
            return {
                'prediction': prediction_str,
                'confidence': float(confidence),
                'probabilities': probabilities.tolist(),
                'raw_prediction': int(prediction)
            }
            
        except Exception as e:
            print(f"❌ ML tahmin hatası: {e}")
            return {'prediction': 'HOLD', 'confidence': 0.0, 'probabilities': [0.33, 0.33, 0.34]}
    
    def get_ml_signal_strength(self, df: pd.DataFrame, min_confidence=0.6) -> float:
        """
        ML modelinin sinyal gücünü döndürür (-1 ile +1 arası).
        -1: Güçlü satış, 0: Belirsiz, +1: Güçlü alış
        """
        prediction_result = self.predict_market_direction(df)
        
        prediction = prediction_result['prediction']
        confidence = prediction_result['confidence']
        
        # Minimum confidence altındaysa sinyal verme
        if confidence < min_confidence:
            return 0.0
        
        # Sinyal gücünü hesapla
        if prediction == 'BUY':
            return confidence  # Pozitif değer (alış)
        elif prediction == 'SELL':
            return -confidence  # Negatif değer (satış)
        else:
            return 0.0  # Nötr
    
    def backtest_ml_signals(self, df: pd.DataFrame, initial_cash=10000):
        """
        ML sinyallerinin geçmiş performansını test eder.
        """
        if self.model is None:
            print("❌ Model henüz eğitilmedi!")
            return None
        
        print("🧪 ML sinyalleri backtest ediliyor...")
        
        cash = initial_cash
        position = 0
        trades = []
        
        df_test = df.iloc[50:].copy()  # İlk 50 veriyi skip et (feature hazırlama için)
        
        for i in range(len(df_test)):
            current_data = df.iloc[:50+i+1]  # Şimdiye kadarki tüm veri
            current_price = df_test.iloc[i]['close']
            
            # ML sinyalini al
            ml_signal = self.get_ml_signal_strength(current_data, min_confidence=0.65)
            
            # Pozisyon yönetimi
            if position == 0 and ml_signal > 0.5:  # BUY signal
                position = cash / current_price
                cash = 0
                trades.append({'type': 'BUY', 'price': current_price, 'ml_confidence': ml_signal})
            
            elif position > 0 and ml_signal < -0.5:  # SELL signal
                cash = position * current_price
                trades.append({'type': 'SELL', 'price': current_price, 'ml_confidence': ml_signal})
                position = 0
        
        # Final değer hesabı
        final_value = cash if position == 0 else position * df_test.iloc[-1]['close']
        total_return = (final_value / initial_cash - 1) * 100
        
        # Trade analizi
        buy_trades = [t for t in trades if t['type'] == 'BUY']
        sell_trades = [t for t in trades if t['type'] == 'SELL']
        
        total_trades = min(len(buy_trades), len(sell_trades))
        winning_trades = 0
        
        for i in range(total_trades):
            if sell_trades[i]['price'] > buy_trades[i]['price']:
                winning_trades += 1
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        results = {
            'total_return': total_return,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'final_value': final_value
        }
        
        print(f"📊 ML Backtest Sonuçları:")
        print(f"   Total Return: {total_return:.2f}%")
        print(f"   Total Trades: {total_trades}")
        print(f"   Win Rate: {win_rate:.2f}%")
        print(f"   Final Value: ${final_value:,.2f}")
        
        return results


# Convenience function
def train_pattern_recognition_model(df: pd.DataFrame, model_type='random_forest'):
    """
    Kolay kullanım için wrapper function
    """
    ml_engine = PatternRecognitionEngine()
    return ml_engine.train_model(df, model_type)