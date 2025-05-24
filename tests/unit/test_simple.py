"""
Simple test to verify pytest is working
"""
import pytest


class TestSimple:
    """Simple test class"""
    
    def test_basic_assertion(self):
        """Test basic assertion"""
        assert 1 + 1 == 2
    
    def test_string_operations(self):
        """Test string operations"""
        test_string = "hello world"
        assert "hello" in test_string
        assert test_string.upper() == "HELLO WORLD"
    
    def test_list_operations(self):
        """Test list operations"""
        test_list = [1, 2, 3, 4, 5]
        assert len(test_list) == 5
        assert 3 in test_list
        assert test_list[0] == 1


def test_simple_function():
    """Simple function test"""
    assert True is True
    assert False is not True
