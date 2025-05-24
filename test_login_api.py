import requests
import json

# Test the login API directly
url = "http://localhost:8000/user/login"
data = {
    "username": "testuser",
    "password": "testpassword"
}

print(f"Testing POST to {url}")
print(f"Data: {data}")

try:
    response = requests.post(url, data=data)
    print(f"Status code: {response.status_code}")
    print(f"Response headers: {response.headers}")
    print(f"Response text: {response.text}")
except Exception as e:
    print(f"Error: {e}")
