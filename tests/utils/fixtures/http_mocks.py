"""HTTP mocking fixtures and response templates for integration tests.

This module provides reusable fixtures and response templates for testing
external service integrations with comprehensive mock data that covers:
- Realistic API response structures
- Error scenarios and edge cases
- Performance testing scenarios
- Authentication and rate limiting responses
- Large data set responses
- Concurrent testing support
"""

import json
from datetime import UTC, datetime
from typing import Any, ClassVar
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from pytest import fixture


class ResponseTemplates:
    """Pre-built response templates for different API scenarios."""

    # Exa API Response Templates
    EXA_SEARCH_SUCCESS: ClassVar[dict] = {
        "requestId": "exa-req-{timestamp}",
        "resolvedSearchType": "neural",
        "searchType": "auto",
        "results": [
            {
                "title": "Comprehensive Guide to Python Async Programming",
                "url": "https://realpython.com/async-io-python/",
                "publishedDate": "2024-01-15T10:30:00Z",
                "author": "Real Python Team",
                "score": 0.96,
                "id": "exa-result-1",
                "text": (
                    "Python's asyncio library is a powerful tool for writing "
                    "asynchronous code. This comprehensive guide covers everything "
                    "from basic concepts to advanced patterns for building "
                    "high-performance applications."
                ),
                "highlights": [
                    "asyncio library is a powerful tool",
                    "asynchronous code",
                    "high-performance applications",
                ],
                "highlightScores": [0.94, 0.91, 0.89],
                "summary": (
                    "Complete guide to Python asyncio programming with "
                    "practical examples"
                ),
            },
            {
                "title": "Advanced Async Patterns in Python",
                "url": "https://docs.python.org/3/library/asyncio.html",
                "publishedDate": "2024-02-20T14:45:00Z",
                "author": "Python Documentation Team",
                "score": 0.93,
                "id": "exa-result-2",
                "text": (
                    "Advanced patterns for using asyncio including context "
                    "managers, queues, and synchronization primitives for "
                    "building robust concurrent applications."
                ),
                "highlights": [
                    "Advanced patterns for using asyncio",
                    "context managers, queues",
                    "concurrent applications",
                ],
                "highlightScores": [0.92, 0.88, 0.86],
                "summary": "Official documentation covering advanced asyncio patterns",
            },
        ],
        "costDollars": {"total": 0.003, "search": 0.002, "contents": 0.001},
    }

    EXA_SEARCH_EMPTY: ClassVar[dict] = {
        "requestId": "exa-req-empty-{timestamp}",
        "resolvedSearchType": "neural",
        "searchType": "auto",
        "results": [],
        "costDollars": {"total": 0.001},
    }

    EXA_CONTENTS_SUCCESS: ClassVar[dict] = {
        "results": [
            {
                "id": "exa-content-1",
                "title": "Full Article Content",
                "url": "https://example.com/full-article",
                "text": "This is the complete text content of the article, providing \
comprehensive information about the topic with detailed \
explanations, code examples, and best practices. The content \
includes multiple sections covering fundamentals, advanced \
concepts, and real-world applications.",
                "highlights": [
                    "comprehensive information about the topic",
                    "detailed explanations, code examples",
                    "best practices",
                    "real-world applications",
                ],
                "summary": (
                    "Comprehensive article covering fundamentals through "
                    "advanced applications"
                ),
            }
        ]
    }

    EXA_RESEARCH_TASK_CREATED: ClassVar[dict] = {
        "id": "research-task-{timestamp}",
        "status": "pending",
    }

    EXA_RESEARCH_TASK_COMPLETED: ClassVar[dict] = {
        "status": "completed",
        "result": {
            "summary": "Comprehensive research findings on the requested topic",
            "key_findings": [
                "Primary insight from research",
                "Secondary important discovery",
                "Additional contextual information",
            ],
            "sources": [
                {"url": "https://source1.com", "title": "Primary Source"},
                {"url": "https://source2.com", "title": "Secondary Source"},
            ],
            "confidence_score": 0.87,
            "methodology": "Multi-source analysis with cross-validation",
        },
    }

    # Firecrawl API Response Templates
    FIRECRAWL_SCRAPE_SUCCESS: ClassVar[dict] = {
        "success": True,
        "data": {
            "markdown": """# Example Domain

This domain is for use in illustrative examples in documents. \
You may use this domain in literature without prior coordination \
or asking for permission.

## More Information

For more information, please visit our [documentation](https://example.com/docs).

### Features

- Simple and clean design
- Fast loading times
- Mobile responsive
- SEO optimized

### Contact

Email: info@example.com
Phone: +1-555-0123""",
            "html": """<html>
<head>
    <title>Example Domain</title>
    <meta name="description" content="This domain is for use in illustrative examples">
</head>
<body>
    <h1>Example Domain</h1>
    <p>This domain is for use in illustrative examples in documents.</p>
    <h2>More Information</h2>
    <p>For more information, please visit our <a href="https://example.com/docs">documentation</a>.</p>
    <h3>Features</h3>
    <ul>
        <li>Simple and clean design</li>
        <li>Fast loading times</li>
        <li>Mobile responsive</li>
        <li>SEO optimized</li>
    </ul>
</body>
</html>""",
            "rawHtml": """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Example Domain</title>
    <meta name="description" content="This domain is for use in illustrative examples">
    <link rel="stylesheet" href="/styles.css">
</head>
<body>
    <header>
        <nav>
            <ul>
                <li><a href="/">Home</a></li>
                <li><a href="/about">About</a></li>
                <li><a href="/contact">Contact</a></li>
            </ul>
        </nav>
    </header>
    <main>
        <h1>Example Domain</h1>
        <p>This domain is for use in illustrative examples in documents.</p>
    </main>
    <footer>
        <p>&copy; 2024 Example Domain. All rights reserved.</p>
    </footer>
</body>
</html>""",
            "links": [
                "https://example.com",
                "https://example.com/about",
                "https://example.com/contact",
                "https://example.com/docs",
                "https://example.com/privacy",
            ],
            "screenshot": (
                "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
                "AAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
            ),
            "json": {
                "title": "Example Domain",
                "description": "This domain is for use in illustrative examples",
                "features": [
                    "Simple design",
                    "Fast loading",
                    "Mobile responsive",
                    "SEO optimized",
                ],
                "contact": {"email": "info@example.com", "phone": "+1-555-0123"},
            },
            "metadata": {
                "title": "Example Domain",
                "description": (
                    "This domain is for use in illustrative examples in documents"
                ),
                "language": "en",
                "keywords": "example, domain, documentation, testing",
                "robots": "index, follow",
                "ogTitle": "Example Domain - Illustrative Examples",
                "ogDescription": "Perfect domain for use in documentation and examples",
                "ogUrl": "https://example.com",
                "ogImage": "https://example.com/images/og-image.png",
                "ogSiteName": "Example Domain",
                "sourceURL": "https://example.com",
                "statusCode": 200,
            },
        },
    }

    FIRECRAWL_SCRAPE_ERROR: ClassVar[dict] = {
        "success": False,
        "error": "URL not accessible: Connection timeout after 30 seconds",
    }

    FIRECRAWL_CRAWL_STARTED: ClassVar[dict] = {
        "success": True,
        "id": "crawl-{timestamp}",
        "url": "https://api.firecrawl.dev/v1/crawl/crawl-{timestamp}",
    }

    FIRECRAWL_CRAWL_COMPLETED: ClassVar[dict] = {
        "success": True,
        "status": "completed",
        "total": 25,
        "completed": 25,
        "creditsUsed": 50,
        "expiresAt": "2024-04-15T10:30:00Z",
        "next": None,
        "data": [
            {
                "markdown": (
                    "# Home Page\n\nWelcome to our website. This is the main "
                    "landing page with information about our services and company."
                ),
                "html": "<h1>Home Page</h1><p>Welcome to our website.</p>",
                "metadata": {
                    "title": "Home - Company Website",
                    "sourceURL": "https://example.com",
                    "statusCode": 200,
                },
            },
            {
                "markdown": (
                    "# About Us\n\nLearn more about our company, our mission, "
                    "and our team of experts."
                ),
                "html": "<h1>About Us</h1><p>Learn more about our company.</p>",
                "metadata": {
                    "title": "About Us - Company Website",
                    "sourceURL": "https://example.com/about",
                    "statusCode": 200,
                },
            },
            {
                "markdown": (
                    "# Services\n\nWe offer a comprehensive range of "
                    "professional services to meet your business needs."
                ),
                "html": "<h1>Services</h1><p>We offer professional services.</p>",
                "metadata": {
                    "title": "Services - Company Website",
                    "sourceURL": "https://example.com/services",
                    "statusCode": 200,
                },
            },
        ],
    }

    FIRECRAWL_MAP_SUCCESS: ClassVar[dict[str, Any]] = {
        "success": True,
        "links": [
            "https://example.com",
            "https://example.com/about",
            "https://example.com/services",
            "https://example.com/products",
            "https://example.com/blog",
            "https://example.com/contact",
            "https://example.com/privacy",
            "https://example.com/terms",
            "https://example.com/support",
            "https://example.com/careers",
        ],
    }

    # Error Response Templates
    ERROR_RATE_LIMIT: ClassVar[dict[str, Any]] = {
        "error": {
            "message": "Rate limit exceeded. Please try again later.",
            "type": "rate_limit",
            "retry_after": 60,
        }
    }

    ERROR_UNAUTHORIZED: ClassVar[dict[str, Any]] = {
        "error": {"message": "Invalid API key provided", "type": "unauthorized"}
    }

    ERROR_QUOTA_EXCEEDED: ClassVar[dict[str, Any]] = {
        "error": {
            "message": "Monthly quota exceeded. Please upgrade your plan.",
            "type": "quota_exceeded",
        }
    }

    ERROR_SERVICE_UNAVAILABLE: ClassVar[dict[str, Any]] = {
        "error": {
            "message": "Service temporarily unavailable due to maintenance",
            "type": "service_unavailable",
            "estimated_recovery": "2024-04-15T12:00:00Z",
        }
    }

    @classmethod
    def personalize_template(cls, template: dict[str, Any], **kwargs) -> dict[str, Any]:
        """Personalize a template with dynamic values."""
        template_str = json.dumps(template)

        # Default personalizations
        defaults = {
            "timestamp": datetime.now(UTC).strftime("%Y%m%d%H%M%S"),
        }
        defaults.update(kwargs)

        # Replace placeholders
        for key, value in defaults.items():
            template_str = template_str.replace(f"{{{key}}}", str(value))

        return json.loads(template_str)


class MockResponseBuilder:
    """Builder for creating mock HTTP responses."""

    def __init__(self):
        self.status_code = 200
        self.headers = {}
        self.json_data = None
        self.text_data = None
        self.should_raise_for_status = False

    def with_status(self, status_code: int) -> "MockResponseBuilder":
        """Set response status code."""
        self.status_code = status_code
        return self

    def with_headers(self, headers: dict[str, str]) -> "MockResponseBuilder":
        """Set response headers."""
        self.headers.update(headers)
        return self

    def with_json(self, data: dict[str, Any]) -> "MockResponseBuilder":
        """Set JSON response data."""
        self.json_data = data
        return self

    def with_text(self, text: str) -> "MockResponseBuilder":
        """Set text response data."""
        self.text_data = text
        return self

    def with_error(self, should_raise: bool = True) -> "MockResponseBuilder":
        """Set whether response should raise on raise_for_status()."""
        self.should_raise_for_status = should_raise
        return self

    def build(self) -> AsyncMock:
        """Build the mock response."""
        mock_response = AsyncMock()
        mock_response.status_code = self.status_code
        mock_response.headers = self.headers

        if self.json_data is not None:
            mock_response.json.return_value = self.json_data
        else:
            mock_response.json.side_effect = json.JSONDecodeError("No JSON", "", 0)

        if self.text_data is not None:
            mock_response.text = self.text_data

        if self.should_raise_for_status and self.status_code >= 400:
            error = httpx.HTTPStatusError(
                f"HTTP {self.status_code}",
                request=MagicMock(),
                response=MagicMock(status_code=self.status_code),
            )
            mock_response.raise_for_status.side_effect = error
        else:
            mock_response.raise_for_status.return_value = None

        return mock_response


# Pytest Fixtures


@fixture
def mock_response_builder():
    """Provide MockResponseBuilder instance."""
    return MockResponseBuilder()


@fixture
def response_templates():
    """Provide ResponseTemplates class."""
    return ResponseTemplates()


@fixture
def exa_success_response(response_templates):
    """Pre-built successful Exa search response."""
    return response_templates.personalize_template(
        response_templates.EXA_SEARCH_SUCCESS
    )


@fixture
def firecrawl_success_response(response_templates):
    """Pre-built successful Firecrawl scrape response."""
    return response_templates.personalize_template(
        response_templates.FIRECRAWL_SCRAPE_SUCCESS
    )


@fixture
def rate_limit_response(response_templates):
    """Pre-built rate limit error response."""
    return response_templates.ERROR_RATE_LIMIT


@fixture
def unauthorized_response(response_templates):
    """Pre-built unauthorized error response."""
    return response_templates.ERROR_UNAUTHORIZED


@fixture
async def mock_exa_client():
    """Mock Exa client with common response patterns."""
    # Already defined patch import from unittest.mock
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client

        # Default successful response
        success_response = (
            MockResponseBuilder()
            .with_json(
                ResponseTemplates.personalize_template(
                    ResponseTemplates.EXA_SEARCH_SUCCESS
                )
            )
            .build()
        )

        mock_client.request.return_value = success_response
        yield mock_client


@fixture
async def mock_firecrawl_client():
    """Mock Firecrawl client with common response patterns."""
    # Already defined patch import from unittest.mock
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client

        # Default successful response
        success_response = (
            MockResponseBuilder()
            .with_json(
                ResponseTemplates.personalize_template(
                    ResponseTemplates.FIRECRAWL_SCRAPE_SUCCESS
                )
            )
            .build()
        )

        mock_client.request.return_value = success_response
        yield mock_client


@fixture
def mock_http_errors():
    """Factory for creating various HTTP error scenarios."""

    def create_error(error_type: str, **kwargs):
        """Create specific error types."""
        if error_type == "connection":
            return httpx.ConnectError(kwargs.get("message", "Connection failed"))
        elif error_type == "timeout":
            return httpx.TimeoutException(kwargs.get("message", "Request timeout"))
        elif error_type == "read_timeout":
            return httpx.ReadTimeout(kwargs.get("message", "Read timeout"))
        elif error_type == "write_timeout":
            return httpx.WriteTimeout(kwargs.get("message", "Write timeout"))
        elif error_type == "pool_timeout":
            return httpx.PoolTimeout(kwargs.get("message", "Pool timeout"))
        elif error_type == "http_status":
            status_code = kwargs.get("status_code", 500)
            mock_response = MagicMock()
            mock_response.status_code = status_code
            mock_response.json.return_value = kwargs.get("response_data", {})
            return httpx.HTTPStatusError(
                f"HTTP {status_code}", request=MagicMock(), response=mock_response
            )
        else:
            raise ValueError(f"Unknown error type: {error_type}")

    return create_error


@fixture
def large_response_data():
    """Generate large response data for performance testing."""

    def generate_large_data(size_mb: int = 1):
        """Generate response data of specified size."""
        # Generate approximately size_mb MB of text data
        text_size = size_mb * 1024 * 1024
        large_text = "x" * text_size

        return {
            "requestId": "large-response-test",
            "results": [
                {
                    "title": f"Large Document {i}",
                    "url": f"https://large-doc-{i}.com",
                    "id": f"large-{i}",
                    "text": large_text if i == 0 else "Normal sized text",
                    "score": 0.9 - (i * 0.1),
                }
                for i in range(10)
            ],
        }

    return generate_large_data


@fixture
def concurrent_response_factory():
    """Factory for creating responses for concurrent testing."""

    def create_responses(count: int, success_rate: float = 1.0):
        """Create multiple responses with specified success rate."""
        responses = []
        success_count = int(count * success_rate)

        for i in range(count):
            if i < success_count:
                # Success response
                response = (
                    MockResponseBuilder()
                    .with_json(
                        {
                            "success": True,
                            "requestId": f"concurrent-{i}",
                            "data": f"Response {i}",
                        }
                    )
                    .build()
                )
            else:
                # Error response
                response = (
                    MockResponseBuilder()
                    .with_status(500)
                    .with_json({"error": f"Error in request {i}"})
                    .with_error(True)
                    .build()
                )

            responses.append(response)

        return responses

    return create_responses


@fixture
def progressive_failure_responses():
    """Responses that simulate progressive system degradation."""

    def create_degradation_sequence(total_requests: int):
        """Create sequence showing progressive degradation."""
        responses = []

        for i in range(total_requests):
            if i < total_requests * 0.5:
                # First half - all success
                response = (
                    MockResponseBuilder()
                    .with_json(
                        {
                            "success": True,
                            "response_time_ms": 100 + (i * 10),  # Gradually slower
                        }
                    )
                    .build()
                )
            elif i < total_requests * 0.8:
                # Next 30% - intermittent failures
                if i % 3 == 0:  # Every third request fails
                    response = (
                        MockResponseBuilder()
                        .with_status(500)
                        .with_json({"error": "Intermittent server error"})
                        .with_error(True)
                        .build()
                    )
                else:
                    response = (
                        MockResponseBuilder()
                        .with_json(
                            {
                                "success": True,
                                "response_time_ms": 200 + (i * 20),  # Much slower
                            }
                        )
                        .build()
                    )
            else:
                # Final 20% - all failures
                response = (
                    MockResponseBuilder()
                    .with_status(503)
                    .with_json({"error": "Service unavailable"})
                    .with_error(True)
                    .build()
                )

            responses.append(response)

        return responses

    return create_degradation_sequence


@fixture
def authentication_flow_responses():
    """Responses for testing authentication flows."""
    return {
        "invalid_key": MockResponseBuilder()
        .with_status(401)
        .with_json(ResponseTemplates.ERROR_UNAUTHORIZED)
        .with_error(True)
        .build(),
        "expired_key": MockResponseBuilder()
        .with_status(403)
        .with_json(
            {
                "error": {
                    "message": "API key has expired",
                    "type": "expired_key",
                    "expires_at": "2024-01-01T00:00:00Z",
                }
            }
        )
        .with_error(True)
        .build(),
        "insufficient_permissions": MockResponseBuilder()
        .with_status(403)
        .with_json(
            {
                "error": {
                    "message": "Insufficient permissions for this operation",
                    "type": "insufficient_permissions",
                    "required_scopes": ["read", "write"],
                }
            }
        )
        .with_error(True)
        .build(),
        "valid_auth": MockResponseBuilder()
        .with_json(
            {
                "success": True,
                "user": {
                    "id": "user-123",
                    "plan": "professional",
                    "quota_remaining": 950,
                },
            }
        )
        .build(),
    }


@fixture
def rate_limiting_flow_responses():
    """Responses for testing rate limiting scenarios."""
    return {
        "rate_limited": MockResponseBuilder()
        .with_status(429)
        .with_headers(
            {
                "Retry-After": "60",
                "X-RateLimit-Limit": "100",
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": "1640995200",
            }
        )
        .with_json(ResponseTemplates.ERROR_RATE_LIMIT)
        .with_error(True)
        .build(),
        "quota_exceeded": MockResponseBuilder()
        .with_status(429)
        .with_json(ResponseTemplates.ERROR_QUOTA_EXCEEDED)
        .with_error(True)
        .build(),
        "rate_limit_recovered": MockResponseBuilder()
        .with_headers(
            {
                "X-RateLimit-Limit": "100",
                "X-RateLimit-Remaining": "99",
                "X-RateLimit-Reset": "1640998800",
            }
        )
        .with_json({"success": True, "message": "Request processed successfully"})
        .build(),
    }


# Utility Functions


def create_mock_sequence(*responses) -> list[AsyncMock]:
    """Create a sequence of mock responses for side_effect."""
    return list(responses)


def create_alternating_responses(success_response, error_response, count: int = 10):
    """Create alternating success/error responses."""
    responses = []
    for i in range(count):
        if i % 2 == 0:
            responses.append(success_response)
        else:
            responses.append(error_response)
    return responses


def create_timeout_then_success_sequence(timeout_count: int = 2):
    """Create sequence of timeouts followed by success."""
    sequence = []

    # Add timeout errors
    for _ in range(timeout_count):
        sequence.append(httpx.TimeoutException("Request timeout"))

    # Add success response
    success_response = (
        MockResponseBuilder().with_json({"success": True, "recovered": True}).build()
    )
    sequence.append(success_response)

    return sequence
