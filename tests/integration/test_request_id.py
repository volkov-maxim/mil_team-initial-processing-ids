"""Integration tests for request ID middleware propagation behavior."""

from uuid import UUID

from fastapi.testclient import TestClient

from app.main import app


def test_request_id_is_generated_when_header_missing() -> None:
    """Generate and return request ID when request header is absent."""
    client = TestClient(app)

    response = client.get("/health")

    request_id = response.headers.get("X-Request-ID")
    assert response.status_code == 200
    assert request_id is not None
    assert UUID(request_id).version == 4


def test_request_id_is_propagated_from_incoming_header() -> None:
    """Return the same request ID when provided by the client header."""
    client = TestClient(app)
    expected_request_id = "req-fixed-001"

    response = client.get(
        "/health",
        headers={"X-Request-ID": expected_request_id},
    )

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == expected_request_id