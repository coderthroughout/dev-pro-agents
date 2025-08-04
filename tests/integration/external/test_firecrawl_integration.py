"""Comprehensive test suite for Firecrawl API client integration.

This module provides extensive testing for the FirecrawlClient including:
- All API endpoints (scrape, crawl, map, batch operations)
- Error handling and retry logic
- Timeout and rate limiting scenarios
- Response validation and data transformation
- Authentication and configuration management
- Advanced features (Actions API, FIRE-1 agent, JSON extraction)
- 90%+ code coverage target
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import pytest_asyncio

from src.config import FirecrawlSettings
from src.integrations.firecrawl_client import (
    FirecrawlAction,
    FirecrawlAgent,
    FirecrawlClient,
    FirecrawlCrawlResponse,
    FirecrawlDocument,
    FirecrawlLocation,
    FirecrawlMapResponse,
    FirecrawlMetadata,
    FirecrawlScrapeResponse,
)


class TestFirecrawlClientInitialization:
    """Test FirecrawlClient initialization and configuration."""

    def test_init_with_direct_api_key(self):
        """Test initialization with directly provided API key."""
        client = FirecrawlClient(api_key="fc-test-key")
        assert client.api_key == "fc-test-key"
        assert client.base_url == "https://api.firecrawl.dev/v1"
        assert client.timeout == 120
        assert client.max_retries == 3
        assert client.base_retry_delay == 1.0

    def test_init_with_config_object(self):
        """Test initialization with FirecrawlSettings config object."""
        config = FirecrawlSettings(
            api_key="fc-config-key",
            base_url="https://custom.firecrawl.dev/v1",
            timeout_seconds=180,
            max_retries=5,
            default_formats=["markdown", "html"],
            only_main_content=False,
            crawl_limit=200,
        )
        client = FirecrawlClient(config=config)
        assert client.api_key == "fc-config-key"
        assert client.base_url == "https://custom.firecrawl.dev/v1"
        assert client.timeout == 180
        assert client.max_retries == 5
        assert client.default_formats == ["markdown", "html"]
        assert client.default_only_main_content is False
        assert client.default_crawl_limit == 200

    def test_init_with_parameter_overrides(self):
        """Test initialization with parameter overrides."""
        config = FirecrawlSettings(
            api_key="config-key",
            timeout_seconds=120,
            max_retries=3,
        )
        client = FirecrawlClient(
            config=config,
            api_key="override-key",
            timeout=240,
            max_retries=7,
        )
        assert client.api_key == "override-key"
        assert client.timeout == 240
        assert client.max_retries == 7

    def test_init_without_api_key_raises_error(self):
        """Test initialization without API key raises ValueError."""
        with (
            patch.dict("os.environ", {}, clear=True),
            patch(
                "src.integrations.firecrawl_client.get_firecrawl_config",
                return_value={},
            ),
            pytest.raises(ValueError, match="FIRECRAWL_API_KEY must be provided"),
        ):
            FirecrawlClient()

    def test_init_with_environment_variable(self):
        """Test initialization retrieves API key from environment."""
        mock_config = {
            "api_key": "fc-env-key",
            "base_url": "https://api.firecrawl.dev/v1",
            "timeout_seconds": 120,
            "max_retries": 3,
            "base_retry_delay": 1.0,
            "default_formats": ["markdown"],
            "only_main_content": True,
            "crawl_limit": 100,
            "max_wait_time": 600,
            "poll_interval": 10,
        }
        with patch(
            "src.integrations.firecrawl_client.get_firecrawl_config",
            return_value=mock_config,
        ):
            client = FirecrawlClient()
            assert client.api_key == "fc-env-key"


class TestFirecrawlClientHTTPOperations:
    """Test HTTP client management and request handling."""

    @pytest_asyncio.fixture
    async def mock_client_setup(self):
        """Setup mock HTTP client for testing."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Setup default success response
            mock_response = AsyncMock()
            mock_response.json.return_value = {"success": True}
            mock_response.raise_for_status.return_value = None
            mock_client.request.return_value = mock_response

            yield mock_client_class, mock_client, mock_response

    @pytest.mark.asyncio
    async def test_get_client_creates_async_client(self, mock_client_setup):
        """Test _get_client creates httpx.AsyncClient with correct headers."""
        mock_client_class, mock_client, _ = mock_client_setup

        client = FirecrawlClient(api_key="fc-test-key")
        http_client = await client._get_client()

        assert http_client is mock_client
        mock_client_class.assert_called_once_with(
            headers={
                "Authorization": "Bearer fc-test-key",
                "Content-Type": "application/json",
            },
            timeout=120,
        )

    @pytest.mark.asyncio
    async def test_get_client_reuses_existing_client(self, mock_client_setup):
        """Test _get_client reuses existing client instance."""
        mock_client_class, mock_client, _ = mock_client_setup

        client = FirecrawlClient(api_key="fc-test-key")

        # First call
        http_client1 = await client._get_client()
        # Second call
        http_client2 = await client._get_client()

        assert http_client1 is http_client2
        # Should only create client once
        assert mock_client_class.call_count == 1

    @pytest.mark.asyncio
    async def test_make_request_success(self, mock_client_setup):
        """Test successful HTTP request."""
        _, mock_client, mock_response = mock_client_setup

        expected_response = {"success": True, "data": {"markdown": "# Test"}}
        mock_response.json.return_value = expected_response

        client = FirecrawlClient(api_key="fc-test-key")
        response = await client._make_request(
            "POST", "/scrape", data={"url": "https://test.com"}
        )

        assert response == expected_response
        mock_client.request.assert_called_once_with(
            method="POST",
            url="https://api.firecrawl.dev/v1/scrape",
            json={"url": "https://test.com"},
            params=None,
        )

    @pytest.mark.asyncio
    async def test_make_request_with_params(self, mock_client_setup):
        """Test HTTP request with query parameters."""
        _, mock_client, mock_response = mock_client_setup

        client = FirecrawlClient(api_key="fc-test-key")
        await client._make_request("GET", "/crawl/status", params={"id": "crawl-123"})

        mock_client.request.assert_called_once_with(
            method="GET",
            url="https://api.firecrawl.dev/v1/crawl/status",
            json=None,
            params={"id": "crawl-123"},
        )


class TestFirecrawlClientRetryLogic:
    """Test retry logic and error handling."""

    @pytest.mark.asyncio
    async def test_retry_on_rate_limit(self):
        """Test retry logic for 429 rate limit errors."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Mock rate limit error on first two attempts, success on third
            mock_response_error = MagicMock()
            mock_response_error.status_code = 429
            mock_response_error.json.return_value = {"error": "Rate limit exceeded"}

            mock_response_success = AsyncMock()
            mock_response_success.json.return_value = {"success": True}
            mock_response_success.raise_for_status.return_value = None

            error = httpx.HTTPStatusError(
                "Too Many Requests", request=MagicMock(), response=mock_response_error
            )

            mock_client.request.side_effect = [error, error, mock_response_success]

            client = FirecrawlClient(api_key="fc-test-key", max_retries=3)

            with patch("asyncio.sleep") as mock_sleep:
                response = await client._make_request("POST", "/scrape")

                assert response == {"success": True}
                # Should have slept twice (for the two retries)
                assert mock_sleep.call_count == 2
                # Check exponential backoff: 1.0 * (2^0) = 1.0, 1.0 * (2^1) = 2.0
                mock_sleep.assert_any_call(1.0)
                mock_sleep.assert_any_call(2.0)

    @pytest.mark.asyncio
    async def test_retry_on_timeout(self):
        """Test retry logic for timeout errors."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Mock timeout on first attempt, success on second
            mock_response_success = AsyncMock()
            mock_response_success.json.return_value = {"success": True}
            mock_response_success.raise_for_status.return_value = None

            mock_client.request.side_effect = [
                httpx.TimeoutException("Request timed out"),
                mock_response_success,
            ]

            client = FirecrawlClient(api_key="fc-test-key", max_retries=2)

            with patch("asyncio.sleep") as mock_sleep:
                response = await client._make_request("POST", "/scrape")

                assert response == {"success": True}
                mock_sleep.assert_called_once_with(1.0)

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self):
        """Test exception when max retries exceeded."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Always fail with rate limit
            mock_response_error = MagicMock()
            mock_response_error.status_code = 429
            mock_response_error.json.return_value = {"error": "Rate limit exceeded"}

            error = httpx.HTTPStatusError(
                "Too Many Requests", request=MagicMock(), response=mock_response_error
            )
            mock_client.request.side_effect = error

            client = FirecrawlClient(api_key="fc-test-key", max_retries=2)

            with (
                patch("asyncio.sleep"),
                pytest.raises(
                    Exception, match="Firecrawl API error: Rate limit exceeded"
                ),
            ):
                await client._make_request("POST", "/scrape")

    @pytest.mark.asyncio
    async def test_non_retryable_error(self):
        """Test immediate failure for non-retryable errors."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Mock 400 bad request (not retryable)
            mock_response_error = MagicMock()
            mock_response_error.status_code = 400
            mock_response_error.json.return_value = {"error": "Invalid URL"}

            error = httpx.HTTPStatusError(
                "Bad Request", request=MagicMock(), response=mock_response_error
            )
            mock_client.request.side_effect = error

            client = FirecrawlClient(api_key="fc-test-key", max_retries=3)

            with pytest.raises(Exception, match="Firecrawl API error: Invalid URL"):
                await client._make_request("POST", "/scrape")


class TestFirecrawlClientScrapeAPI:
    """Test Firecrawl scrape API functionality."""

    @pytest_asyncio.fixture
    async def setup_scrape_mock(self):
        """Setup mock for scrape API responses."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            mock_response = AsyncMock()
            mock_response.raise_for_status.return_value = None
            mock_client.request.return_value = mock_response

            yield mock_client, mock_response

    @pytest.mark.asyncio
    async def test_scrape_basic_url(self, setup_scrape_mock):
        """Test basic scrape functionality."""
        mock_client, mock_response = setup_scrape_mock

        mock_api_response = {
            "success": True,
            "data": {
                "markdown": "# Example Domain\n\nThis domain is for use in illustrative examples...",
                "html": "<h1>Example Domain</h1><p>This domain is for use in illustrative examples...</p>",
                "rawHtml": "<!DOCTYPE html><html><head><title>Example</title></head>...",
                "links": ["https://example.com", "https://example.com/about"],
                "screenshot": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
                "metadata": {
                    "title": "Example Domain",
                    "description": "This domain is for use in illustrative examples",
                    "language": "en",
                    "sourceURL": "https://example.com",
                    "statusCode": 200,
                    "ogTitle": "Example Domain",
                    "ogDescription": "Example website",
                    "ogImage": "https://example.com/image.png",
                },
            },
        }
        mock_response.json.return_value = mock_api_response

        client = FirecrawlClient(api_key="fc-test-key")
        result = await client.scrape("https://example.com")

        # Verify response structure
        assert isinstance(result, FirecrawlScrapeResponse)
        assert result.success is True
        assert result.error is None

        # Verify document data
        assert result.data is not None
        assert isinstance(result.data, FirecrawlDocument)
        assert result.data.markdown.startswith("# Example Domain")
        assert result.data.html.startswith("<h1>Example Domain</h1>")
        assert result.data.raw_html.startswith("<!DOCTYPE html>")
        assert len(result.data.links) == 2
        assert result.data.screenshot.startswith("data:image/png;base64")

        # Verify metadata
        assert result.data.metadata is not None
        assert isinstance(result.data.metadata, FirecrawlMetadata)
        assert result.data.metadata.title == "Example Domain"
        assert result.data.metadata.source_url == "https://example.com"
        assert result.data.metadata.status_code == 200
        assert result.data.metadata.og_title == "Example Domain"

        # Verify request parameters
        call_args = mock_client.request.call_args
        assert call_args[1]["method"] == "POST"
        assert call_args[1]["url"] == "https://api.firecrawl.dev/v1/scrape"
        payload = call_args[1]["json"]
        assert payload["url"] == "https://example.com"
        assert payload["formats"] == ["markdown"]  # default format
        assert payload["onlyMainContent"] is True  # default setting

    @pytest.mark.asyncio
    async def test_scrape_with_all_options(self, setup_scrape_mock):
        """Test scrape with all available options."""
        mock_client, mock_response = setup_scrape_mock

        mock_response.json.return_value = {
            "success": True,
            "data": {
                "markdown": "# Test Content",
                "html": "<h1>Test Content</h1>",
                "json": {"title": "Test", "content": "Content"},
                "metadata": {"title": "Test", "sourceURL": "https://test.com"},
            },
        }

        # Create action and agent objects
        actions = [
            FirecrawlAction(type="wait", milliseconds=2000),
            FirecrawlAction(type="click", selector="button.load-more"),
            FirecrawlAction(
                type="write", selector="input[name='search']", text="test query"
            ),
            FirecrawlAction(type="screenshot", full_page=True),
        ]

        agent = FirecrawlAgent(
            model="FIRE-1",
            prompt="Navigate to the main content and extract all product information",
        )

        location = FirecrawlLocation(country="US", languages=["en", "es"])

        json_schema = {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "content": {"type": "string"},
                "price": {"type": "number"},
            },
            "required": ["title", "content"],
        }

        client = FirecrawlClient(api_key="fc-test-key")

        await client.scrape(
            url="https://test.com",
            formats=["markdown", "html", "json", "screenshot"],
            include_tags=["article", "main", "section"],
            exclude_tags=["nav", "footer", "aside"],
            only_main_content=False,
            actions=actions,
            agent=agent,
            location=location,
            json_schema=json_schema,
            json_prompt="Extract product information",
            wait_for=5000,
            timeout=30000,
        )

        # Verify comprehensive request payload
        call_args = mock_client.request.call_args
        payload = call_args[1]["json"]

        assert payload["url"] == "https://test.com"
        assert payload["formats"] == ["markdown", "html", "json", "screenshot"]
        assert payload["includeTags"] == ["article", "main", "section"]
        assert payload["excludeTags"] == ["nav", "footer", "aside"]
        assert payload["onlyMainContent"] is False
        assert payload["waitFor"] == 5000
        assert payload["timeout"] == 30000

        # Verify actions
        assert "actions" in payload
        assert len(payload["actions"]) == 4
        assert payload["actions"][0]["type"] == "wait"
        assert payload["actions"][0]["milliseconds"] == 2000
        assert payload["actions"][1]["type"] == "click"
        assert payload["actions"][1]["selector"] == "button.load-more"
        assert payload["actions"][2]["type"] == "write"
        assert payload["actions"][2]["text"] == "test query"
        assert payload["actions"][3]["type"] == "screenshot"
        assert payload["actions"][3]["fullPage"] is True

        # Verify agent configuration
        assert "agent" in payload
        assert payload["agent"]["model"] == "FIRE-1"
        assert (
            payload["agent"]["prompt"]
            == "Navigate to the main content and extract all product information"
        )

        # Verify location settings
        assert "location" in payload
        assert payload["location"]["country"] == "US"
        assert payload["location"]["languages"] == ["en", "es"]

        # Verify JSON extraction
        assert "jsonOptions" in payload
        assert payload["jsonOptions"]["schema"] == json_schema
        assert payload["jsonOptions"]["prompt"] == "Extract product information"

    @pytest.mark.asyncio
    async def test_scrape_json_format_auto_added(self, setup_scrape_mock):
        """Test that json format is automatically added when using JSON options."""
        mock_client, mock_response = setup_scrape_mock

        mock_response.json.return_value = {
            "success": True,
            "data": {"markdown": "# Test", "json": {"title": "Test"}},
        }

        client = FirecrawlClient(api_key="fc-test-key")
        await client.scrape(
            url="https://test.com",
            formats=["markdown"],  # Only markdown initially
            json_schema={"type": "object", "properties": {"title": {"type": "string"}}},
        )

        call_args = mock_client.request.call_args
        payload = call_args[1]["json"]
        # Should automatically include "json" format
        assert "json" in payload["formats"]
        assert "markdown" in payload["formats"]

    @pytest.mark.asyncio
    async def test_scrape_error_response(self, setup_scrape_mock):
        """Test handling of scrape error responses."""
        mock_client, mock_response = setup_scrape_mock

        mock_response.json.return_value = {
            "success": False,
            "error": "URL not accessible: timeout after 30 seconds",
        }

        client = FirecrawlClient(api_key="fc-test-key")
        result = await client.scrape("https://unreachable.com")

        assert isinstance(result, FirecrawlScrapeResponse)
        assert result.success is False
        assert result.data is None
        assert result.error == "URL not accessible: timeout after 30 seconds"

    @pytest.mark.asyncio
    async def test_scrape_missing_metadata(self, setup_scrape_mock):
        """Test scrape handling when metadata is missing."""
        mock_client, mock_response = setup_scrape_mock

        mock_response.json.return_value = {
            "success": True,
            "data": {
                "markdown": "# Test Page",
                "html": "<h1>Test Page</h1>",
                # No metadata field
            },
        }

        client = FirecrawlClient(api_key="fc-test-key")
        result = await client.scrape("https://test.com")

        assert result.success is True
        assert result.data is not None
        assert result.data.metadata is None  # Should handle missing metadata gracefully


class TestFirecrawlClientCrawlAPI:
    """Test Firecrawl crawl API functionality."""

    @pytest.mark.asyncio
    async def test_crawl_basic(self):
        """Test basic crawl functionality."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            mock_response = AsyncMock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {
                "success": True,
                "id": "crawl-abc123",
                "url": "https://api.firecrawl.dev/v1/crawl/crawl-abc123",
            }
            mock_client.request.return_value = mock_response

            client = FirecrawlClient(api_key="fc-test-key")
            result = await client.crawl("https://example.com")

            assert isinstance(result, FirecrawlCrawlResponse)
            assert result.success is True
            assert result.id == "crawl-abc123"
            assert result.url == "https://api.firecrawl.dev/v1/crawl/crawl-abc123"
            assert result.error is None

            # Verify request parameters
            call_args = mock_client.request.call_args
            assert call_args[1]["method"] == "POST"
            assert call_args[1]["url"] == "https://api.firecrawl.dev/v1/crawl"
            payload = call_args[1]["json"]
            assert payload["url"] == "https://example.com"
            assert payload["limit"] == 100  # default crawl limit
            assert payload["allowSubdomains"] is False
            assert payload["crawlEntireDomain"] is False

    @pytest.mark.asyncio
    async def test_crawl_with_all_options(self):
        """Test crawl with all available options."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            mock_response = AsyncMock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {
                "success": True,
                "id": "crawl-comprehensive",
                "url": "https://api.firecrawl.dev/v1/crawl/crawl-comprehensive",
            }
            mock_client.request.return_value = mock_response

            client = FirecrawlClient(api_key="fc-test-key")

            await client.crawl(
                url="https://docs.example.com",
                limit=500,
                max_depth=3,
                formats=["markdown", "html"],
                include_paths=["/docs/*", "/api/*", "/guides/*"],
                exclude_paths=["/admin/*", "/internal/*"],
                allow_subdomains=True,
                crawl_entire_domain=False,
                only_main_content=False,
                max_age=3600000,  # 1 hour cache
                webhook_url="https://my-app.com/webhooks/crawl-progress",
            )

            # Verify comprehensive request payload
            call_args = mock_client.request.call_args
            payload = call_args[1]["json"]

            assert payload["url"] == "https://docs.example.com"
            assert payload["limit"] == 500
            assert payload["maxDepth"] == 3
            assert payload["includePaths"] == ["/docs/*", "/api/*", "/guides/*"]
            assert payload["excludePaths"] == ["/admin/*", "/internal/*"]
            assert payload["allowSubdomains"] is True
            assert payload["crawlEntireDomain"] is False

            # Verify webhook configuration
            assert "webhook" in payload
            assert (
                payload["webhook"]["url"]
                == "https://my-app.com/webhooks/crawl-progress"
            )

            # Verify scrape options
            assert "scrapeOptions" in payload
            scrape_options = payload["scrapeOptions"]
            assert scrape_options["formats"] == ["markdown", "html"]
            assert scrape_options["onlyMainContent"] is False
            assert scrape_options["maxAge"] == 3600000

    @pytest.mark.asyncio
    async def test_check_crawl_status(self):
        """Test checking crawl job status."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            mock_response = AsyncMock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {
                "success": True,
                "status": "completed",
                "total": 45,
                "completed": 45,
                "creditsUsed": 90,
                "expiresAt": "2024-04-15T10:30:00Z",
                "next": None,
                "data": [
                    {
                        "markdown": "# Home Page\nWelcome to our website...",
                        "html": "<h1>Home Page</h1><p>Welcome to our website...</p>",
                        "metadata": {
                            "title": "Home Page",
                            "sourceURL": "https://example.com",
                            "statusCode": 200,
                        },
                    },
                    {
                        "markdown": "# About Us\nLearn more about our company...",
                        "html": "<h1>About Us</h1><p>Learn more about our company...</p>",
                        "metadata": {
                            "title": "About Us",
                            "sourceURL": "https://example.com/about",
                            "statusCode": 200,
                        },
                    },
                ],
            }
            mock_client.request.return_value = mock_response

            client = FirecrawlClient(api_key="fc-test-key")
            result = await client.check_crawl_status("crawl-status-test")

            assert isinstance(result, FirecrawlCrawlResponse)
            assert result.success is True
            assert result.id == "crawl-status-test"
            assert result.status == "completed"
            assert result.total == 45
            assert result.completed == 45
            assert result.credits_used == 90
            assert result.expires_at == "2024-04-15T10:30:00Z"
            assert result.next_url is None

            # Verify parsed documents
            assert result.data is not None
            assert len(result.data) == 2

            doc1 = result.data[0]
            assert isinstance(doc1, FirecrawlDocument)
            assert doc1.markdown.startswith("# Home Page")
            assert doc1.metadata.title == "Home Page"
            assert doc1.metadata.source_url == "https://example.com"

            doc2 = result.data[1]
            assert doc2.markdown.startswith("# About Us")
            assert doc2.metadata.source_url == "https://example.com/about"

            # Verify request
            call_args = mock_client.request.call_args
            assert call_args[1]["method"] == "GET"
            assert (
                call_args[1]["url"]
                == "https://api.firecrawl.dev/v1/crawl/crawl-status-test"
            )

    @pytest.mark.asyncio
    async def test_wait_for_crawl_completion(self):
        """Test waiting for crawl completion."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Mock progression: running -> running -> completed
            responses = [
                {"success": True, "status": "running", "completed": 10, "total": 25},
                {"success": True, "status": "running", "completed": 20, "total": 25},
                {
                    "success": True,
                    "status": "completed",
                    "completed": 25,
                    "total": 25,
                    "data": [
                        {"markdown": "# Final Page", "metadata": {"title": "Final"}}
                    ],
                },
            ]

            mock_response = AsyncMock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.side_effect = responses
            mock_client.request.return_value = mock_response

            client = FirecrawlClient(api_key="fc-test-key")

            with patch("asyncio.sleep") as mock_sleep:
                result = await client.wait_for_crawl_completion(
                    crawl_id="wait-test-crawl",
                    max_wait_time=120,
                    poll_interval=5,
                )

                assert result.status == "completed"
                assert result.completed == 25
                assert result.total == 25
                assert len(result.data) == 1

                # Should have slept twice (for the two "running" statuses)
                assert mock_sleep.call_count == 2
                mock_sleep.assert_called_with(5)  # poll_interval

    @pytest.mark.asyncio
    async def test_wait_for_crawl_timeout(self):
        """Test crawl wait timeout."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Always return "running" status
            mock_response = AsyncMock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {"success": True, "status": "running"}
            mock_client.request.return_value = mock_response

            client = FirecrawlClient(api_key="fc-test-key")

            with (
                patch("asyncio.sleep"),
                patch("asyncio.get_event_loop") as mock_loop,
                pytest.raises(
                    Exception, match="Crawl timeout-crawl timed out after 10 seconds"
                ),
            ):
                # Mock elapsed time progression
                mock_loop.return_value.time.side_effect = [
                    0,
                    2,
                    4,
                    6,
                    8,
                    12,
                ]  # Exceeds 10 second limit

                await client.wait_for_crawl_completion(
                    crawl_id="timeout-crawl",
                    max_wait_time=10,
                    poll_interval=2,
                )

    @pytest.mark.asyncio
    async def test_crawl_and_wait(self):
        """Test complete crawl and wait workflow."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Mock crawl start response
            start_response = AsyncMock()
            start_response.raise_for_status.return_value = None
            start_response.json.return_value = {
                "success": True,
                "id": "end-to-end-crawl",
            }

            # Mock crawl completion response
            complete_response = AsyncMock()
            complete_response.raise_for_status.return_value = None
            complete_response.json.return_value = {
                "success": True,
                "status": "completed",
                "total": 15,
                "completed": 15,
                "data": [
                    {"markdown": "# Page 1", "metadata": {"title": "Page 1"}},
                    {"markdown": "# Page 2", "metadata": {"title": "Page 2"}},
                ],
            }

            # Set up responses in order
            mock_client.request.side_effect = [start_response, complete_response]

            client = FirecrawlClient(api_key="fc-test-key")

            with patch("asyncio.sleep"):
                result = await client.crawl_and_wait(
                    url="https://test-site.com",
                    limit=50,
                    formats=["markdown"],
                    max_wait_time=60,
                    poll_interval=3,
                )

                assert result.status == "completed"
                assert result.total == 15
                assert len(result.data) == 2

    @pytest.mark.asyncio
    async def test_crawl_and_wait_start_failure(self):
        """Test crawl_and_wait when initial crawl start fails."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            mock_response = AsyncMock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {
                "success": False,
                "error": "Invalid domain: not allowed",
            }
            mock_client.request.return_value = mock_response

            client = FirecrawlClient(api_key="fc-test-key")
            result = await client.crawl_and_wait(
                url="https://restricted-site.com", max_wait_time=30
            )

            # Should return the failed start response without waiting
            assert result.success is False
            assert result.error == "Invalid domain: not allowed"


class TestFirecrawlClientMapAPI:
    """Test Firecrawl map API functionality."""

    @pytest.mark.asyncio
    async def test_map_website_basic(self):
        """Test basic website mapping."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            mock_response = AsyncMock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {
                "success": True,
                "links": [
                    "https://example.com",
                    "https://example.com/about",
                    "https://example.com/products",
                    "https://example.com/contact",
                    "https://example.com/blog",
                    "https://example.com/support",
                ],
            }
            mock_client.request.return_value = mock_response

            client = FirecrawlClient(api_key="fc-test-key")
            result = await client.map_website("https://example.com")

            assert isinstance(result, FirecrawlMapResponse)
            assert result.success is True
            assert result.error is None
            assert result.links is not None
            assert len(result.links) == 6
            assert "https://example.com" in result.links
            assert "https://example.com/about" in result.links
            assert "https://example.com/blog" in result.links

            # Verify request parameters
            call_args = mock_client.request.call_args
            assert call_args[1]["method"] == "POST"
            assert call_args[1]["url"] == "https://api.firecrawl.dev/v1/map"
            payload = call_args[1]["json"]
            assert payload["url"] == "https://example.com"
            assert payload["ignoreSitemap"] is False
            assert payload["includeSubdomains"] is False

    @pytest.mark.asyncio
    async def test_map_website_with_options(self):
        """Test website mapping with all options."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            mock_response = AsyncMock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {
                "success": True,
                "links": [
                    "https://example.com",
                    "https://docs.example.com/guide1",
                    "https://api.example.com/reference",
                ],
            }
            mock_client.request.return_value = mock_response

            client = FirecrawlClient(api_key="fc-test-key")

            await client.map_website(
                url="https://example.com",
                ignore_sitemap=True,
                include_subdomains=True,
                limit=1000,
            )

            # Verify request payload
            call_args = mock_client.request.call_args
            payload = call_args[1]["json"]
            assert payload["url"] == "https://example.com"
            assert payload["ignoreSitemap"] is True
            assert payload["includeSubdomains"] is True
            assert payload["limit"] == 1000

    @pytest.mark.asyncio
    async def test_map_website_error_response(self):
        """Test map website error handling."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            mock_response = AsyncMock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {
                "success": False,
                "error": "Unable to access website: connection timeout",
            }
            mock_client.request.return_value = mock_response

            client = FirecrawlClient(api_key="fc-test-key")
            result = await client.map_website("https://unreachable.com")

            assert result.success is False
            assert result.links is None
            assert result.error == "Unable to access website: connection timeout"


class TestFirecrawlClientBatchOperations:
    """Test Firecrawl batch operations."""

    @pytest.mark.asyncio
    async def test_batch_scrape(self):
        """Test batch scrape functionality."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            mock_response = AsyncMock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {
                "success": True,
                "id": "batch-scrape-456",
                "url": "https://api.firecrawl.dev/v1/batch/scrape/batch-scrape-456",
            }
            mock_client.request.return_value = mock_response

            client = FirecrawlClient(api_key="fc-test-key")

            urls = [
                "https://example.com",
                "https://example.com/about",
                "https://example.com/products",
            ]

            result = await client.batch_scrape(
                urls=urls,
                formats=["markdown", "html"],
                only_main_content=False,
                webhook_url="https://my-app.com/batch-webhook",
            )

            assert isinstance(result, FirecrawlCrawlResponse)
            assert result.success is True
            assert result.id == "batch-scrape-456"
            assert (
                result.url
                == "https://api.firecrawl.dev/v1/batch/scrape/batch-scrape-456"
            )

            # Verify request payload
            call_args = mock_client.request.call_args
            assert call_args[1]["method"] == "POST"
            assert call_args[1]["url"] == "https://api.firecrawl.dev/v1/batch/scrape"
            payload = call_args[1]["json"]
            assert payload["urls"] == urls
            assert "webhook" in payload
            assert payload["webhook"]["url"] == "https://my-app.com/batch-webhook"
            assert "scrapeOptions" in payload
            assert payload["scrapeOptions"]["formats"] == ["markdown", "html"]
            assert payload["scrapeOptions"]["onlyMainContent"] is False

    @pytest.mark.asyncio
    async def test_batch_scrape_and_wait(self):
        """Test batch scrape with waiting for completion."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Mock batch start response
            start_response = AsyncMock()
            start_response.raise_for_status.return_value = None
            start_response.json.return_value = {
                "success": True,
                "id": "batch-wait-test",
            }

            # Mock batch completion response
            complete_response = AsyncMock()
            complete_response.raise_for_status.return_value = None
            complete_response.json.return_value = {
                "success": True,
                "status": "completed",
                "total": 3,
                "completed": 3,
                "data": [
                    {
                        "markdown": "# Page 1",
                        "metadata": {"sourceURL": "https://example.com"},
                    },
                    {
                        "markdown": "# Page 2",
                        "metadata": {"sourceURL": "https://example.com/about"},
                    },
                    {
                        "markdown": "# Page 3",
                        "metadata": {"sourceURL": "https://example.com/products"},
                    },
                ],
            }

            # Set up responses in order
            mock_client.request.side_effect = [start_response, complete_response]

            client = FirecrawlClient(api_key="fc-test-key")

            urls = [
                "https://example.com",
                "https://example.com/about",
                "https://example.com/products",
            ]

            with patch("asyncio.sleep"):
                result = await client.batch_scrape_and_wait(
                    urls=urls, formats=["markdown"], max_wait_time=60, poll_interval=2
                )

                assert result.status == "completed"
                assert result.total == 3
                assert len(result.data) == 3
                assert result.data[0].metadata.source_url == "https://example.com"


class TestFirecrawlClientUtilityOperations:
    """Test utility operations like cancel_crawl."""

    @pytest.mark.asyncio
    async def test_cancel_crawl(self):
        """Test canceling a running crawl job."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            mock_response = AsyncMock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {
                "success": True,
                "message": "Crawl job canceled successfully",
            }
            mock_client.request.return_value = mock_response

            client = FirecrawlClient(api_key="fc-test-key")
            result = await client.cancel_crawl("crawl-to-cancel")

            assert result["success"] is True
            assert result["message"] == "Crawl job canceled successfully"

            # Verify request
            call_args = mock_client.request.call_args
            assert call_args[1]["method"] == "DELETE"
            assert (
                call_args[1]["url"]
                == "https://api.firecrawl.dev/v1/crawl/crawl-to-cancel"
            )


class TestFirecrawlClientContextManagement:
    """Test FirecrawlClient context manager functionality."""

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """Test FirecrawlClient as async context manager."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            async with FirecrawlClient(api_key="fc-test-key") as client:
                assert isinstance(client, FirecrawlClient)
                assert client.api_key == "fc-test-key"

            # Should have called aclose on the HTTP client
            mock_client.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_manual_close(self):
        """Test manual client closure."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            client = FirecrawlClient(api_key="fc-test-key")
            # Create the HTTP client
            await client._get_client()

            # Close manually
            await client.close()

            mock_client.aclose.assert_called_once()
            assert client._client is None

    @pytest.mark.asyncio
    async def test_close_without_client(self):
        """Test close when no HTTP client exists."""
        client = FirecrawlClient(api_key="fc-test-key")
        # Should not raise exception
        await client.close()
        assert client._client is None


class TestFirecrawlClientErrorScenarios:
    """Test various error scenarios and edge cases."""

    @pytest.mark.asyncio
    async def test_malformed_json_response(self):
        """Test handling of malformed JSON responses."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            mock_response = AsyncMock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
            mock_client.request.return_value = mock_response

            client = FirecrawlClient(api_key="fc-test-key")

            with pytest.raises(json.JSONDecodeError):
                await client.scrape("https://test.com")

    @pytest.mark.asyncio
    async def test_network_error_handling(self):
        """Test handling of network errors."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Simulate network error
            mock_client.request.side_effect = httpx.ConnectError("Network unreachable")

            client = FirecrawlClient(api_key="fc-test-key", max_retries=2)

            with pytest.raises(
                Exception, match="Firecrawl API request failed: Network unreachable"
            ):
                await client.scrape("https://test.com")

    @pytest.mark.asyncio
    async def test_missing_document_fields(self):
        """Test handling of API responses with missing document fields."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Response with minimal fields
            mock_response = AsyncMock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {
                "success": True,
                "data": {
                    # Only minimal fields, missing many optional ones
                    "markdown": "# Basic Content"
                    # Missing: html, rawHtml, links, screenshot, json, metadata
                },
            }
            mock_client.request.return_value = mock_response

            client = FirecrawlClient(api_key="fc-test-key")
            result = await client.scrape("https://test.com")

            # Should handle missing fields gracefully
            assert result.success is True
            document = result.data
            assert document.markdown == "# Basic Content"
            assert document.html is None
            assert document.raw_html is None
            assert document.links is None
            assert document.screenshot is None
            assert document.json_data is None
            assert document.metadata is None

    @pytest.mark.asyncio
    async def test_empty_data_response(self):
        """Test handling of responses with no data."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            mock_response = AsyncMock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {
                "success": True,
                # No data field
            }
            mock_client.request.return_value = mock_response

            client = FirecrawlClient(api_key="fc-test-key")
            result = await client.scrape("https://empty.com")

            assert result.success is True
            assert result.data is None

    @pytest.mark.asyncio
    async def test_http_error_without_json_body(self):
        """Test HTTP error handling when response has no JSON body."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Mock response that fails JSON parsing
            mock_response = MagicMock()
            mock_response.status_code = 502
            mock_response.json.side_effect = json.JSONDecodeError("No JSON", "", 0)

            error = httpx.HTTPStatusError(
                "Bad Gateway", request=MagicMock(), response=mock_response
            )
            mock_client.request.side_effect = error

            client = FirecrawlClient(api_key="fc-test-key")

            with pytest.raises(Exception, match="Firecrawl API error: 502 Bad Gateway"):
                await client.scrape("https://test.com")


class TestFirecrawlModelClasses:
    """Test Firecrawl model classes and their validation."""

    def test_firecrawl_action_creation(self):
        """Test creating FirecrawlAction instances."""
        # Test wait action
        wait_action = FirecrawlAction(type="wait", milliseconds=3000)
        assert wait_action.type == "wait"
        assert wait_action.milliseconds == 3000
        assert wait_action.selector is None

        # Test click action
        click_action = FirecrawlAction(type="click", selector="button.submit")
        assert click_action.type == "click"
        assert click_action.selector == "button.submit"
        assert click_action.milliseconds is None

        # Test write action
        write_action = FirecrawlAction(
            type="write", selector="input[name='search']", text="test query"
        )
        assert write_action.type == "write"
        assert write_action.selector == "input[name='search']"
        assert write_action.text == "test query"

        # Test screenshot action
        screenshot_action = FirecrawlAction(type="screenshot", full_page=True)
        assert screenshot_action.type == "screenshot"
        assert screenshot_action.full_page is True

        # Test executeJavascript action
        js_action = FirecrawlAction(
            type="executeJavascript",
            script="window.scrollTo(0, document.body.scrollHeight);",
        )
        assert js_action.type == "executeJavascript"
        assert js_action.script == "window.scrollTo(0, document.body.scrollHeight);"

    def test_firecrawl_agent_creation(self):
        """Test creating FirecrawlAgent instances."""
        agent = FirecrawlAgent(
            model="FIRE-1",
            prompt="Navigate to the product section and extract all product details",
        )
        assert agent.model == "FIRE-1"
        assert (
            agent.prompt
            == "Navigate to the product section and extract all product details"
        )

    def test_firecrawl_location_creation(self):
        """Test creating FirecrawlLocation instances."""
        # Test with defaults
        location1 = FirecrawlLocation()
        assert location1.country == "US"
        assert location1.languages is None

        # Test with custom values
        location2 = FirecrawlLocation(country="CA", languages=["en", "fr"])
        assert location2.country == "CA"
        assert location2.languages == ["en", "fr"]

    def test_firecrawl_metadata_field_aliases(self):
        """Test FirecrawlMetadata field aliases work correctly."""
        # Test with alias names (as they come from API)
        metadata_data = {
            "title": "Test Page",
            "ogTitle": "OG Test Page",
            "ogDescription": "OG description",
            "ogUrl": "https://og.example.com",
            "ogImage": "https://og.example.com/image.png",
            "ogSiteName": "OG Site",
            "sourceURL": "https://source.example.com",
            "statusCode": 200,
        }

        metadata = FirecrawlMetadata(**metadata_data)
        assert metadata.title == "Test Page"
        assert metadata.og_title == "OG Test Page"
        assert metadata.og_description == "OG description"
        assert metadata.og_url == "https://og.example.com"
        assert metadata.og_image == "https://og.example.com/image.png"
        assert metadata.og_site_name == "OG Site"
        assert metadata.source_url == "https://source.example.com"
        assert metadata.status_code == 200

    def test_firecrawl_document_field_aliases(self):
        """Test FirecrawlDocument field aliases work correctly."""
        document_data = {
            "markdown": "# Test Document",
            "html": "<h1>Test Document</h1>",
            "rawHtml": "<!DOCTYPE html><html>...",
            "links": ["https://test.com", "https://test.com/page2"],
            "screenshot": "data:image/png;base64,iVBORw0KGgo...",
            "json": {"title": "Test", "content": "Document content"},
        }

        document = FirecrawlDocument(**document_data)
        assert document.markdown == "# Test Document"
        assert document.html == "<h1>Test Document</h1>"
        assert document.raw_html == "<!DOCTYPE html><html>..."
        assert document.links == ["https://test.com", "https://test.com/page2"]
        assert document.screenshot == "data:image/png;base64,iVBORw0KGgo..."
        assert document.json_data == {"title": "Test", "content": "Document content"}


# Performance and integration test markers
@pytest.mark.integration
@pytest.mark.skipif(
    "FIRECRAWL_API_KEY" not in locals() and "FIRECRAWL_API_KEY" not in globals(),
    reason="FIRECRAWL_API_KEY not available for integration tests",
)
class TestFirecrawlClientLiveIntegration:
    """Live integration tests requiring actual API key."""

    @pytest.mark.asyncio
    async def test_live_scrape_request(self):
        """Test live scrape request with real API."""
        import os

        api_key = os.getenv("FIRECRAWL_API_KEY")
        if not api_key:
            pytest.skip("FIRECRAWL_API_KEY environment variable not set")

        async with FirecrawlClient(api_key=api_key) as client:
            result = await client.scrape(
                url="https://example.com", formats=["markdown"], only_main_content=True
            )

            assert isinstance(result, FirecrawlScrapeResponse)
            assert result.success is True
            assert result.data is not None
            assert result.data.markdown is not None
            assert len(result.data.markdown) > 0

    @pytest.mark.asyncio
    async def test_live_map_website(self):
        """Test live website mapping with real API."""
        import os

        api_key = os.getenv("FIRECRAWL_API_KEY")
        if not api_key:
            pytest.skip("FIRECRAWL_API_KEY environment variable not set")

        async with FirecrawlClient(api_key=api_key) as client:
            result = await client.map_website(url="https://example.com", limit=10)

            assert isinstance(result, FirecrawlMapResponse)
            assert result.success is True
            assert result.links is not None
            assert len(result.links) > 0
            assert all(link.startswith("http") for link in result.links)
