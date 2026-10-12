import requests
import json

url = "https://llmapi.evolventapi.com/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer sk-PkAxUjysS49_JKGA-zTP9g"
}

# Test 1: Simple prompt
print("=== Test 1: Simple prompt ===")
data1 = {
    "model": "gpt-6-astra",
    "messages": [{"role": "user", "content": "Hello, how are you?"}],
    "max_tokens": 50
}
try:
    response = requests.post(url, headers=headers, json=data1, timeout=60)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text[:300]}")
except Exception as e:
    print(f"Error: {e}")

# Test 2: More complex prompt (like scout phase)
print("\n=== Test 2: Complex prompt (scout-like) ===")
data2 = {
    "model": "gpt-6-astra",
    "messages": [{
        "role": "user",
        "content": "Find one real public creator's stated project goal for a compact crafted interior scene. Search for architectural design commissions or artist studio projects. Return the source URL and a short description."
    }],
    "max_tokens": 200
}
try:
    response = requests.post(url, headers=headers, json=data2, timeout=60)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text[:500]}")
except Exception as e:
    print(f"Error: {e}")
