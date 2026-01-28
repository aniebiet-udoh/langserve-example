"""
Integration tests for the endpoints with mocked dependencies.
"""

import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from tests.conftest import MockRunnable


@pytest.fixture
def client(mock_all_factories):
    """Create a test client for the FastAPI app."""
    from app.main import app
    return TestClient(app)


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


class TestEndpointIntegration:
    """Integration tests for all endpoints."""

    def test_full_app_health_check(self, client):
        """Test that the app responds to health checks."""
        response = client.get("/")
        assert response.status_code == 200
        assert "status" in response.json()

    def test_openapi_documentation(self, client):
        """Test that OpenAPI documentation is available."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        schema = response.json()
        assert "paths" in schema
        assert "info" in schema
        
        # Verify that routes are documented
        paths = schema["paths"]
        assert "/" in paths
        
    def test_all_paths_have_documentation(self, client):
        """Test that all endpoints have proper documentation in OpenAPI schema."""
        response = client.get("/openapi.json")
        schema = response.json()
        
        required_paths = ["/"]
        for path in required_paths:
            assert path in schema["paths"], f"Path {path} not found in OpenAPI schema"
            
            # Each path should have methods defined
            methods = schema["paths"][path]
            assert isinstance(methods, dict)
            assert len(methods) > 0

    @patch("app.chains.get_rag_chain")
    def test_agent_and_rag_routes_registered(self, mock_rag, client):
        """Test that both agent and RAG routes are properly registered."""
        response = client.get("/openapi.json")
        schema = response.json()
        paths = schema.get("paths", {})
        
        # Check for agent routes
        agent_paths = [p for p in paths if "/agent" in p]
        assert len(agent_paths) > 0, "Agent routes not found in OpenAPI schema"
        
        # Check for rag routes
        rag_paths = [p for p in paths if "/rag" in p]
        assert len(rag_paths) > 0, "RAG routes not found in OpenAPI schema"

    def test_status_endpoint_response_format(self, client):
        """Test that the status endpoint returns properly formatted JSON."""
        response = client.get("/")
        
        # Check response structure
        assert response.headers["content-type"] == "application/json"
        
        data = response.json()
        assert isinstance(data, dict)
        assert "status" in data
        assert data["status"] == "AI backend running"


class TestErrorHandling:
    """Test error handling for the endpoints."""

    def test_404_for_nonexistent_route(self, client):
        """Test that nonexistent routes return 404."""
        response = client.get("/nonexistent")
        assert response.status_code == 404

    def test_405_for_unsupported_methods(self, client):
        """Test that unsupported HTTP methods return 405."""
        response = client.post("/")
        assert response.status_code == 405

    def test_error_response_contains_detail(self, client):
        """Test that error responses contain detail information."""
        response = client.get("/nonexistent")
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data


class TestResponseTypes:
    """Test response types and content negotiation."""

    def test_root_response_is_json(self, client):
        """Test that root endpoint returns JSON."""
        response = client.get("/")
        assert response.headers["content-type"] == "application/json"
        # Should be valid JSON
        response.json()

    def test_openapi_response_is_json(self, client):
        """Test that OpenAPI schema response is JSON."""
        response = client.get("/openapi.json")
        assert response.headers["content-type"] == "application/json"
        # Should be valid JSON
        response.json()

    def test_html_responses_for_ui_endpoints(self, client):
        """Test that UI endpoints return HTML."""
        response = client.get("/docs")
        # 307 is a redirect, 200 is direct response
        assert response.status_code in [200, 307]


class TestEndpointBehavior:
    """Test specific endpoint behaviors."""

    def test_root_endpoint_is_idempotent(self, client):
        """Test that calling root endpoint multiple times returns same result."""
        response1 = client.get("/")
        response2 = client.get("/")
        response3 = client.get("/")
        
        assert response1.json() == response2.json() == response3.json()

    def test_root_endpoint_no_request_body_needed(self, client):
        """Test that root endpoint works without request body."""
        response = client.get("/")
        assert response.status_code == 200

    def test_app_maintains_state_between_requests(self, client):
        """Test that app state persists across requests."""
        response1 = client.get("/")
        status1 = response1.json()["status"]
        
        response2 = client.get("/")
        status2 = response2.json()["status"]
        
        # Status should be consistent
        assert status1 == status2 == "AI backend running"
