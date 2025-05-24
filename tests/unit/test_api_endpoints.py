"""
Unit tests for API endpoints
"""
import pytest
import requests
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class TestAPIEndpoints:
    """Test API endpoint functionality"""
    
    BASE_URL = "http://localhost:8000"
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for each test"""
        self.session = requests.Session()
        self.auth_token = None
    
    def test_server_status(self):
        """Test server status endpoint"""
        try:
            response = self.session.get(f"{self.BASE_URL}/server/status", timeout=5)
            assert response.status_code == 200
            data = response.json()
            assert "status" in data
        except requests.exceptions.RequestException:
            pytest.skip("API server not running")
    
    def test_user_login(self):
        """Test user login endpoint"""
        try:
            login_data = {
                "username": "testuser",
                "password": "testpassword"
            }
            response = self.session.post(f"{self.BASE_URL}/user/login", data=login_data, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                assert "access_token" in data
                assert "token_type" in data
                self.auth_token = data["access_token"]
            else:
                pytest.fail(f"Login failed with status {response.status_code}: {response.text}")
        except requests.exceptions.RequestException:
            pytest.skip("API server not running")
    
    def test_image_generation_with_auth(self):
        """Test image generation endpoint with authentication"""
        try:
            # First login
            login_data = {
                "username": "testuser", 
                "password": "testpassword"
            }
            login_response = self.session.post(f"{self.BASE_URL}/user/login", data=login_data, timeout=10)
            
            if login_response.status_code != 200:
                pytest.skip("Cannot authenticate for image generation test")
            
            token = login_response.json().get("access_token")
            if not token:
                pytest.skip("No access token received")
            
            # Test image generation
            headers = {"Authorization": f"Bearer {token}"}
            image_request = {
                "prompt": "A simple test image",
                "width": 512,
                "height": 512,
                "num_inference_steps": 10
            }
            
            response = self.session.post(
                f"{self.BASE_URL}/image/generate", 
                json=image_request, 
                headers=headers, 
                timeout=60
            )
            
            # Check response
            assert response.status_code in [200, 202]  # Accept both OK and Accepted
            
        except requests.exceptions.RequestException:
            pytest.skip("API server not running")


if __name__ == "__main__":
    pytest.main([__file__])
