"""
Tests for the main FastAPI application endpoints.
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


class TestRootEndpoint:
    """Tests for the root endpoint."""

    def test_root_endpoint_returns_200(self, client):
        """Test that the root endpoint returns a 200 status code."""
        response = client.get("/")
        assert response.status_code == 200

    def test_root_endpoint_returns_correct_response(self, client):
        """Test that the root endpoint returns the expected response."""
        response = client.get("/")
        assert response.json() == {"status": "AI backend running"}

    def test_root_endpoint_response_structure(self, client):
        """Test that the root endpoint response has the correct structure."""
        response = client.get("/")
        data = response.json()
        assert "status" in data
        assert isinstance(data["status"], str)


class TestAgentEndpoints:
    """Tests for the agent endpoints."""

    def test_agent_invoke_endpoint_exists(self, client):
        """Test that the agent invoke endpoint is available."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        openapi_schema = response.json()
        assert "/agent/invoke" in openapi_schema.get("paths", {})

    def test_agent_endpoints_are_registered(self, client):
        """Test that agent endpoints are properly registered via add_routes."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        openapi_schema = response.json()
        
        # Check that agent routes exist
        paths = openapi_schema.get("paths", {})
        agent_paths = [p for p in paths.keys() if "/agent" in p]
        assert len(agent_paths) > 0, "No agent endpoints found in OpenAPI schema"


class TestRAGEndpoints:
    """Tests for the RAG chain endpoints."""

    def test_rag_endpoints_are_registered(self, client):
        """Test that RAG endpoints are properly registered via add_routes."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        openapi_schema = response.json()
        
        # Check that rag routes exist
        paths = openapi_schema.get("paths", {})
        rag_paths = [p for p in paths.keys() if "/rag" in p]
        assert len(rag_paths) > 0, "No RAG endpoints found in OpenAPI schema"


class TestAppMetadata:
    """Tests for the FastAPI app metadata."""

    def test_app_title(self, client):
        """Test that the app has the correct title."""
        from app.main import app
        assert app.title == "Agentic AI Backend"

    def test_app_version(self, client):
        """Test that the app has the correct version."""
        from app.main import app
        assert app.version == "1.0"

    def test_openapi_schema_accessible(self, client):
        """Test that the OpenAPI schema is accessible."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "info" in schema
        assert schema["info"]["title"] == "Agentic AI Backend"
        assert schema["info"]["version"] == "1.0"

    def test_docs_endpoint_accessible(self, client):
        """Test that the Swagger UI docs endpoint is accessible."""
        response = client.get("/docs")
        assert response.status_code == 200
        assert "Swagger UI" in response.text or "swagger-ui" in response.text


class TestHTTPMethods:
    """Tests for HTTP method handling."""

    def test_root_endpoint_only_accepts_get(self, client):
        """Test that the root endpoint only accepts GET requests."""
        # POST should not be allowed
        response = client.post("/")
        assert response.status_code == 405  # Method Not Allowed

    def test_root_endpoint_does_not_accept_put(self, client):
        """Test that the root endpoint does not accept PUT requests."""
        response = client.put("/")
        assert response.status_code == 405

    def test_root_endpoint_does_not_accept_delete(self, client):
        """Test that the root endpoint does not accept DELETE requests."""
        response = client.delete("/")
        assert response.status_code == 405


class TestApplicationStartup:
    """Tests for application startup and initialization."""

    def test_app_initialization(self):
        """Test that the app is properly initialized."""
        assert app is not None
        assert hasattr(app, "routes")
        assert len(app.routes) > 0

    def test_app_has_openapi_endpoints(self, client):
        """Test that the app has OpenAPI-related endpoints."""
        # These are standard FastAPI endpoints
        endpoints_to_test = ["/openapi.json", "/docs", "/redoc"]
        for endpoint in endpoints_to_test:
            response = client.get(endpoint)
            # These should either return 200 or 307 (redirect)
            assert response.status_code in [200, 307]

    def test_langserve_routes_added(self):
        """Test that langserve routes were added to the app."""
        route_paths = {route.path for route in app.routes}
        
        # Check that agent and rag routes exist
        agent_routes = {p for p in route_paths if "/agent" in p}
        rag_routes = {p for p in route_paths if "/rag" in p}
        
        assert len(agent_routes) > 0, "No agent routes found"
        assert len(rag_routes) > 0, "No rag routes found"
