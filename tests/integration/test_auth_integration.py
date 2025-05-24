"""
Integration tests for the full authentication and API workflow
"""
import pytest
import requests
import sys
from pathlib import Path
from hashlib import sha256

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from services.database.users import user_exists, get_user_id
from services.database.passwords import verify_credentials


class TestAuthenticationIntegration:
    """Integration tests for authentication workflow"""
    
    BASE_URL = "http://localhost:8000"
    
    def test_full_auth_workflow(self):
        """Test complete authentication workflow from database to API"""
        username = "testuser"
        password = "testpassword"
        
        # 1. Test database level authentication
        assert user_exists(username), "User should exist in database"
        
        user_id = get_user_id(username)
        assert user_id is not None, "Should get valid user ID"
        
        password_hash = sha256(password.encode()).hexdigest()
        assert verify_credentials(username, password_hash), "Database credentials should verify"
        
        # 2. Test API level authentication
        try:
            login_data = {"username": username, "password": password}
            response = requests.post(f"{self.BASE_URL}/user/login", data=login_data, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                assert "access_token" in data, "Should receive access token"
                assert "token_type" in data, "Should receive token type"
                
                # 3. Test using the token for authenticated requests
                token = data["access_token"]
                headers = {"Authorization": f"Bearer {token}"}
                
                # Test accessing protected endpoint
                status_response = requests.get(f"{self.BASE_URL}/server/status", headers=headers, timeout=5)
                assert status_response.status_code == 200, "Should access status with valid token"
                
            else:
                pytest.fail(f"API login failed: {response.status_code} - {response.text}")
                
        except requests.exceptions.RequestException:
            pytest.skip("API server not running")
    
    def test_invalid_auth_workflow(self):
        """Test authentication workflow with invalid credentials"""
        username = "testuser"
        wrong_password = "wrongpassword"
        
        # 1. Test database level with wrong password
        wrong_hash = sha256(wrong_password.encode()).hexdigest()
        assert not verify_credentials(username, wrong_hash), "Wrong password should fail"
        
        # 2. Test API level with wrong password
        try:
            login_data = {"username": username, "password": wrong_password}
            response = requests.post(f"{self.BASE_URL}/user/login", data=login_data, timeout=10)
            assert response.status_code == 401, "Should return 401 for wrong password"
            
        except requests.exceptions.RequestException:
            pytest.skip("API server not running")


if __name__ == "__main__":
    pytest.main([__file__])
