"""Tests for Exa and Firecrawl API integration clients.

These tests validate the API wrapper functionality and error handling.
Note: Some tests require valid API keys and will be skipped if not available.
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.integrations.exa_client import (
    ExaClient,
    ExaResearchTask,
    ExaSearchResponse,
)
from src.integrations.firecrawl_client import (
    FirecrawlClient,
    FirecrawlCrawlResponse,
    FirecrawlDocument,
    FirecrawlMetadata,
    FirecrawlScrapeResponse,
)


class TestExaClient:
    """Test cases for ExaClient."""

    def test_init_with_api_key(self):
        """Test initialization with API key."""
        client = ExaClient(api_key="test-key")
        assert client.api_key == "test-key"
        assert client.base_url == "https://api.exa.ai"
        assert client.timeout == 60
        assert client.max_retries == 3

    def test_init_without_api_key_raises_error(self):
        """Test initialization without API key raises error."""
        with (
            patch.dict(os.environ, {}, clear=True),
            pytest.raises(ValueError, match="EXA_API_KEY must be provided"),
        ):
            ExaClient()

    def test_init_with_env_var(self):
        """Test initialization with environment variable."""
        with patch.dict(os.environ, {"EXA_API_KEY": "env-key"}):
            client = ExaClient()
            assert client.api_key == "env-key"

    @pytest.mark.asyncio
    async def test_search_success(self):
        """Test successful search operation."""
        mock_response = {
            "requestId": "test-123",
            "resolvedSearchType": "neural",
            "searchType": "auto",
            "results": [
                {
                    "title": "Test Article",
                    "url": "https://example.com/test",
                    "publishedDate": "2024-01-01T00:00:00Z",
                    "author": "Test Author",
                    "score": 0.95,
                    "id": "test-id",
                    "text": "Test content",
                    "highlights": ["test highlight"],
                    "summary": "Test summary",
                }
            ],
            "costDollars": {"total": 0.005},
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            mock_response_obj = AsyncMock()
            mock_response_obj.json.return_value = mock_response
            mock_response_obj.raise_for_status.return_value = None
            mock_client.request.return_value = mock_response_obj

            client = ExaClient(api_key="test-key")
            result = await client.search("test query")

            assert isinstance(result, ExaSearchResponse)
            assert result.request_id == "test-123"
            assert result.resolved_search_type == "neural"
            assert len(result.results) == 1
            assert result.results[0].title == "Test Article"
            assert result.results[0].url == "https://example.com/test"

            await client.close()

    @pytest.mark.asyncio
    async def test_search_with_options(self):
        """Test search with various options."""
        mock_response = {
            "requestId": "test-123",
            "resolvedSearchType": "neural",
            "searchType": "neural",
            "results": [],
            "costDollars": {"total": 0.005},
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            mock_response_obj = AsyncMock()
            mock_response_obj.json.return_value = mock_response
            mock_response_obj.raise_for_status.return_value = None
            mock_client.request.return_value = mock_response_obj

            client = ExaClient(api_key="test-key")

            await client.search(
                query="test query",
                search_type="neural",
                num_results=5,
                include_text=True,
                include_highlights=True,
                include_summary=True,
                category="research paper",
                include_domains=["arxiv.org"],
                exclude_domains=["example.com"],
            )

            # Verify the request was made with correct parameters
            call_args = mock_client.request.call_args
            assert call_args[1]["json"]["query"] == "test query"
            assert call_args[1]["json"]["type"] == "neural"
            assert call_args[1]["json"]["numResults"] == 5
            assert call_args[1]["json"]["category"] == "research paper"
            assert call_args[1]["json"]["includeDomains"] == ["arxiv.org"]
            assert call_args[1]["json"]["excludeDomains"] == ["example.com"]
            assert "contents" in call_args[1]["json"]

            await client.close()

    @pytest.mark.asyncio
    async def test_research_task_creation(self):
        """Test research task creation."""
        mock_response = {"id": "task-123"}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            mock_response_obj = AsyncMock()
            mock_response_obj.json.return_value = mock_response
            mock_response_obj.raise_for_status.return_value = None
            mock_client.request.return_value = mock_response_obj

            client = ExaClient(api_key="test-key")

            task = await client.create_research_task(
                instructions="Research AI trends",
                model="exa-research",
                output_schema={
                    "type": "object",
                    "properties": {"summary": {"type": "string"}},
                },
            )

            assert isinstance(task, ExaResearchTask)
            assert task.id == "task-123"
            assert task.status == "pending"

            await client.close()

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling for API failures."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Mock HTTP error
            from httpx import HTTPStatusError, Request, Response

            mock_response = MagicMock(spec=Response)
            mock_response.status_code = 400
            mock_response.json.return_value = {"error": {"message": "Bad request"}}

            mock_request = MagicMock(spec=Request)
            error = HTTPStatusError(
                "Bad request", request=mock_request, response=mock_response
            )
            mock_client.request.side_effect = error

            client = ExaClient(api_key="test-key")

            with pytest.raises(Exception, match="Exa API error: Bad request"):
                await client.search("test query")

            await client.close()


class TestFirecrawlClient:
    """Test cases for FirecrawlClient."""

    def test_init_with_api_key(self):
        """Test initialization with API key."""
        client = FirecrawlClient(api_key="test-key")
        assert client.api_key == "test-key"
        assert client.base_url == "https://api.firecrawl.dev/v1"
        assert client.timeout == 120
        assert client.max_retries == 3

    def test_init_without_api_key_raises_error(self):
        """Test initialization without API key raises error."""
        with (
            patch.dict(os.environ, {}, clear=True),
            pytest.raises(ValueError, match="FIRECRAWL_API_KEY must be provided"),
        ):
            FirecrawlClient()

    def test_init_with_env_var(self):
        """Test initialization with environment variable."""
        with patch.dict(os.environ, {"FIRECRAWL_API_KEY": "env-key"}):
            client = FirecrawlClient()
            assert client.api_key == "env-key"

    @pytest.mark.asyncio
    async def test_scrape_success(self):
        """Test successful scrape operation."""
        mock_response = {
            "success": True,
            "data": {
                "markdown": "# Test Page\nThis is test content.",
                "html": "<h1>Test Page</h1><p>This is test content.</p>",
                "metadata": {
                    "title": "Test Page",
                    "description": "A test page",
                    "sourceURL": "https://example.com/test",
                    "statusCode": 200,
                },
            },
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            mock_response_obj = AsyncMock()
            mock_response_obj.json.return_value = mock_response
            mock_response_obj.raise_for_status.return_value = None
            mock_client.request.return_value = mock_response_obj

            client = FirecrawlClient(api_key="test-key")
            result = await client.scrape("https://example.com/test")

            assert isinstance(result, FirecrawlScrapeResponse)
            assert result.success is True
            assert result.data is not None
            assert isinstance(result.data, FirecrawlDocument)
            assert result.data.markdown == "# Test Page\nThis is test content."
            assert result.data.html == "<h1>Test Page</h1><p>This is test content.</p>"
            assert result.data.metadata is not None
            assert isinstance(result.data.metadata, FirecrawlMetadata)
            assert result.data.metadata.title == "Test Page"
            assert result.data.metadata.source_url == "https://example.com/test"

            await client.close()

    @pytest.mark.asyncio
    async def test_scrape_with_json_extraction(self):
        """Test scrape with JSON extraction."""
        mock_response = {
            "success": True,
            "data": {
                "markdown": "# Company Page\nWe offer AI services.",
                "json": {
                    "company_mission": "AI-powered solutions",
                    "services": ["ML", "NLP"],
                },
                "metadata": {
                    "title": "Company",
                    "sourceURL": "https://company.com",
                    "statusCode": 200,
                },
            },
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            mock_response_obj = AsyncMock()
            mock_response_obj.json.return_value = mock_response
            mock_response_obj.raise_for_status.return_value = None
            mock_client.request.return_value = mock_response_obj

            client = FirecrawlClient(api_key="test-key")

            result = await client.scrape(
                url="https://company.com",
                formats=["markdown", "json"],
                json_schema={
                    "type": "object",
                    "properties": {
                        "company_mission": {"type": "string"},
                        "services": {"type": "array", "items": {"type": "string"}},
                    },
                },
            )

            assert result.success is True
            assert result.data.json_data is not None
            assert result.data.json_data["company_mission"] == "AI-powered solutions"
            assert result.data.json_data["services"] == ["ML", "NLP"]

            # Verify request included json format and options
            call_args = mock_client.request.call_args
            assert "json" in call_args[1]["json"]["formats"]
            assert "jsonOptions" in call_args[1]["json"]

            await client.close()

    @pytest.mark.asyncio
    async def test_crawl_start(self):
        """Test starting a crawl job."""
        mock_response = {
            "success": True,
            "id": "crawl-123",
            "url": "https://api.firecrawl.dev/v1/crawl/crawl-123",
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            mock_response_obj = AsyncMock()
            mock_response_obj.json.return_value = mock_response
            mock_response_obj.raise_for_status.return_value = None
            mock_client.request.return_value = mock_response_obj

            client = FirecrawlClient(api_key="test-key")

            result = await client.crawl(
                url="https://example.com",
                limit=50,
                max_depth=2,
                formats=["markdown"],
                include_paths=["/blog/*"],
                exclude_paths=["/admin/*"],
            )

            assert isinstance(result, FirecrawlCrawlResponse)
            assert result.success is True
            assert result.id == "crawl-123"
            assert result.url == "https://api.firecrawl.dev/v1/crawl/crawl-123"

            # Verify request parameters
            call_args = mock_client.request.call_args
            payload = call_args[1]["json"]
            assert payload["url"] == "https://example.com"
            assert payload["limit"] == 50
            assert payload["maxDepth"] == 2
            assert payload["includePaths"] == ["/blog/*"]
            assert payload["excludePaths"] == ["/admin/*"]

            await client.close()

    @pytest.mark.asyncio
    async def test_map_website(self):
        """Test website mapping."""
        mock_response = {
            "success": True,
            "links": [
                "https://example.com",
                "https://example.com/about",
                "https://example.com/contact",
                "https://example.com/blog",
            ],
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            mock_response_obj = AsyncMock()
            mock_response_obj.json.return_value = mock_response
            mock_response_obj.raise_for_status.return_value = None
            mock_client.request.return_value = mock_response_obj

            client = FirecrawlClient(api_key="test-key")

            result = await client.map_website(
                url="https://example.com",
                include_subdomains=False,
                limit=100,
            )

            assert result.success is True
            assert result.links is not None
            assert len(result.links) == 4
            assert "https://example.com/about" in result.links

            await client.close()

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling for API failures."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Mock HTTP error
            from httpx import HTTPStatusError, Request, Response

            mock_response = MagicMock(spec=Response)
            mock_response.status_code = 401
            mock_response.json.return_value = {"error": "Unauthorized"}

            mock_request = MagicMock(spec=Request)
            error = HTTPStatusError(
                "Unauthorized", request=mock_request, response=mock_response
            )
            mock_client.request.side_effect = error

            client = FirecrawlClient(api_key="test-key")

            with pytest.raises(Exception, match="Firecrawl API error: Unauthorized"):
                await client.scrape("https://example.com")

            await client.close()


# Integration tests (require API keys)
@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("EXA_API_KEY"), reason="EXA_API_KEY not set")
class TestExaClientIntegration:
    """Integration tests for ExaClient that require a real API key."""

    @pytest.mark.asyncio
    async def test_real_search(self):
        """Test real search with Exa API."""
        client = ExaClient()

        try:
            result = await client.search(
                query="LangGraph framework documentation",
                search_type="neural",
                num_results=3,
                include_text=True,
            )

            assert isinstance(result, ExaSearchResponse)
            assert len(result.results) > 0
            assert result.results[0].title is not None
            assert result.results[0].url is not None

        finally:
            await client.close()


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("FIRECRAWL_API_KEY"), reason="FIRECRAWL_API_KEY not set"
)
class TestFirecrawlClientIntegration:
    """Integration tests for FirecrawlClient that require a real API key."""

    @pytest.mark.asyncio
    async def test_real_scrape(self):
        """Test real scrape with Firecrawl API."""
        client = FirecrawlClient()

        try:
            result = await client.scrape(
                url="https://example.com",
                formats=["markdown"],
                only_main_content=True,
            )

            assert isinstance(result, FirecrawlScrapeResponse)
            assert result.success is True
            assert result.data is not None
            assert result.data.markdown is not None

        finally:
            await client.close()
