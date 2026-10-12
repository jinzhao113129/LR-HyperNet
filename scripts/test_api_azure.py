import requests
import json

url = "https://llmapi.evolventapi.com/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer sk-PkAxUjysS49_JKGA-zTP9g"
}

# Test with azure-gpt-6-astra
print("=== Test: azure-gpt-6-astra ===")
data = {
    "model": "azure-gpt-6-astra",
    "messages": [{"role": "user", "content": "hi"}],
    "max_tokens": 10
}
try:
    response = requests.post(url, headers=headers, json=data, timeout=60)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text[:500]}")
except Exception as e:
    print(f"Error: {e}")
