import requests

BASE_URL = "https://api.toobit.com"

symbols = [
    "BTC-SWAP-USDT",
    "ETH-SWAP-USDT",
    "SOL-SWAP-USDT",
    "XRP-SWAP-USDT",
    "BNB-SWAP-USDT",
    "DOGE-SWAP-USDT",
    "AVAX-SWAP-USDT",
    "LINK-SWAP-USDT",
]

for symbol in symbols:
    try:
        response = requests.get(
            f"{BASE_URL}/quote/v1/klines",
            params={
                "symbol": symbol,
                "interval": "5m",
                "limit": 5
            },
            timeout=10
        )

        response.raise_for_status()
        data = response.json()

        if data:
            print(symbol, "OK -", data[-1][4])
        else:
            print(symbol, "NO DATA")

    except Exception as e:
        print(symbol, "ERROR -", e)