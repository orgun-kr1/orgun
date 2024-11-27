# upbit_api.py
import pyupbit

# 업비트 API 객체 생성 (본인의 키를 입력)
access_key = "your_access_key"
secret_key = "your_secret_key"
upbit = pyupbit.Upbit(access_key, secret_key)

# 마켓 데이터 조회
def get_market_data():
    markets = pyupbit.get_markets()
    market_data = []
    for market in markets:
        # 마켓 정보 필터링 (BTC 마켓만)
        if "BTC" in market['market']:
            market_data.append(market)
    return market_data

# 비트코인 현재 가격 조회
def get_current_price():
    return pyupbit.get_current_price("KRW-BTC")

# 주문 처리 함수 (매수, 매도)
def place_order(side, market, price, volume):
    """주문을 처리하는 함수"""
    if side == "buy":
        # 매수 주문 실행
        return upbit.buy_limit_order(market, price, volume)
    elif side == "sell":
        # 매도 주문 실행
        return upbit.sell_limit_order(market, price, volume)
    else:
        raise ValueError("Invalid action. Use 'buy' or 'sell'.")
