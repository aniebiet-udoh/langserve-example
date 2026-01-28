# Test Suite Documentation

This directory contains comprehensive test coverage for the LangServe example application endpoints.

## Test Files

### `test_main.py`
Core tests for the main FastAPI application endpoints and metadata:

- **TestRootEndpoint**: Tests for the `GET /` endpoint
  - Status code validation
  - Response content and structure
  
- **TestAgentEndpoints**: Tests for the agent routes
  - Endpoint registration verification
  - OpenAPI schema validation
  
- **TestRAGEndpoints**: Tests for the RAG chain routes
  - Endpoint registration verification
  - OpenAPI schema validation
  
- **TestAppMetadata**: Tests for application configuration
  - Title and version validation
  - OpenAPI schema accessibility
  - API documentation endpoints
  
- **TestHTTPMethods**: Tests for HTTP method restrictions
  - GET-only validation for root endpoint
  - 405 Method Not Allowed responses
  
- **TestApplicationStartup**: Tests for app initialization
  - Routes verification
  - LangServe route registration
  - Standard FastAPI endpoints

### `test_integration.py`
Integration tests for endpoint behavior and error handling:

- **TestEndpointIntegration**: End-to-end endpoint tests
  - Health checks
  - API documentation
  - Route registration
  
- **TestErrorHandling**: Error response tests
  - 404 responses for invalid routes
  - 405 responses for unsupported methods
  - Error detail messages
  
- **TestResponseTypes**: Content type and response format tests
  - JSON response validation
  - HTML UI endpoint responses
  
- **TestEndpointBehavior**: Endpoint behavior tests
  - Idempotency
  - State persistence
  - Request handling

## Running Tests

### Run all tests
```bash
pytest
```

### Run specific test file
```bash
pytest tests/test_main.py
```

### Run specific test class
```bash
pytest tests/test_main.py::TestRootEndpoint
```

### Run specific test
```bash
pytest tests/test_main.py::TestRootEndpoint::test_root_endpoint_returns_200
```

### Run with verbose output
```bash
pytest -v
```

### Run with coverage report
```bash
pytest --cov=app --cov-report=html
```

### Run with specific markers
```bash
pytest -m integration
```

## Test Coverage

The test suite covers:

- ✅ Root endpoint functionality (`GET /`)
- ✅ Agent routes registration (via LangServe)
- ✅ RAG chain routes registration (via LangServe)
- ✅ API documentation and OpenAPI schema
- ✅ HTTP method restrictions
- ✅ Error handling (404, 405)
- ✅ Response types and content negotiation
- ✅ Application metadata
- ✅ Application startup and initialization

## Dependencies

The test suite requires:
- `pytest` - Test framework
- `pytest-asyncio` - Async test support
- `httpx` - HTTP client for testing

These are included in the `requirements.txt` file.

## Mocking

The tests use `unittest.mock` to mock:
- `app.agent.get_agent()` - Agent factory
- `app.chains.get_rag_chain()` - RAG chain factory

This allows testing endpoint registration without requiring actual LLM calls or retriever setup.

## Notes

- Tests use FastAPI's `TestClient` for synchronous endpoint testing
- All tests are isolated with proper fixtures
- Mocking prevents actual LLM/retriever calls during testing
- The test suite validates both the endpoints and the app's metadata
