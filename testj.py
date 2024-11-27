import requests
import jwt
import uuid
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

access_key = os.getenv("UPBIT_ACCESS_KEY")
secret_key = os.getenv("UPBIT_SECRET_KEY")

# JWT 생성
payload = {
    'access_key': access_key,
    'nonce': str(uuid.uuid4()),  # 각 요청마다 고유한 값
}
jwt_token = jwt.encode(payload, secret_key, algorithm='HS256')

# API 요청
url = "https://api.upbit.com/v1/accounts"
headers = {
    'Authorization': f'Bearer {jwt_token}',  # Bearer 토큰 포함
}

response = requests.get(url, headers=headers)

# 응답 출력
print(response.status_code)
print(response.json())
