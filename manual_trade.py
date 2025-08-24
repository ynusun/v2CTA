from binance.client import Client as BinanceClient
from trade_executor import place_order

def manual_trade_test():
    """
    Testnet'e manuel olarak bir alım emri gönderir.
    """
    print("Manuel Test Emri Gönderici Başlatıldı.")
    print("Binance Testnet'e deneme amaçlı bir alım emri gönderilecek...")

    # --- EMİR PARAMETRELERİ ---
    # Dilerseniz bu değerleri değiştirebilirsiniz.
    symbol = "BTCUSDT"
    side = BinanceClient.SIDE_BUY  # Alım için SIDE_BUY, Satım için SIDE_SELL
    quantity = 0.001  # Almak istediğiniz BTC miktarı

    # place_order fonksiyonunu çağırarak emri gönder
    result = place_order(symbol=symbol, side=side, quantity=quantity)

    if result:
        print("\nManuel test emri başarıyla işlendi.")
    else:
        print("\nManuel test emri gönderilirken bir sorun oluştu.")


if __name__ == "__main__":
    manual_trade_test()