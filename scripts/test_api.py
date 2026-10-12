import requests
import json

url = "https://llmapi.evolventapi.com/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer sk-PkAxUjysS49_JKGA-zTP9g"
}
data = {
    "model": "gpt-6-astra",
    "messages": [{"role": "user", "content": "hi"}],
    "max_tokens": 10
}

try:
    response = requests.post(url, headers=headers, json=data, timeout=30)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text[:500]}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
