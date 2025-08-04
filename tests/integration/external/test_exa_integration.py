"""Comprehensive test suite for Exa API client integration.

This module provides extensive testing for the ExaClient including:
- All API endpoints (search, contents, find_similar, answer, research)
- Error handling and retry logic
- Timeout and rate limiting scenarios
- Response validation and data transformation
- Authentication and configuration management
- 90%+ code coverage target
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import pytest_asyncio

from src.config import ExaSettings
from src.integrations.exa_client import (
    ExaClient,
    ExaResearchTask,
    ExaSearchResponse,
    ExaSearchResult,
)


class TestExaClientInitialization:
    """Test ExaClient initialization and configuration."""

    def test_init_with_direct_api_key(self):
        """Test initialization with directly provided API key."""
        client = ExaClient(api_key="test-direct-key")
        assert client.api_key == "test-direct-key"
        assert client.base_url == "https://api.exa.ai/"
        assert client.timeout == 60
        assert client.max_retries == 3
        assert client.base_retry_delay == 1.0

    def test_init_with_config_object(self):
        """Test initialization with ExaSettings config object."""
        config = ExaSettings(
            api_key="test-config-key",
            base_url="https://custom.exa.ai",
            timeout_seconds=90,
            max_retries=5,
            search_type="neural",
            num_results=20,
        )
        client = ExaClient(config=config)
        assert client.api_key == "test-config-key"
        assert client.base_url == "https://custom.exa.ai/"
        assert client.timeout == 90
        assert client.max_retries == 5
        assert client.default_search_type == "neural"
        assert client.default_num_results == 20

    def test_init_with_parameter_overrides(self):
        """Test initialization with parameter overrides."""
        config = ExaSettings(
            api_key="config-key",
            timeout_seconds=60,
            max_retries=3,
        )
        client = ExaClient(
            config=config,
            api_key="override-key",
            timeout=120,
            max_retries=7,
        )
        assert client.api_key == "override-key"
        assert client.timeout == 120
        assert client.max_retries == 7

    def test_init_without_api_key_raises_error(self):
        """Test initialization without API key raises ValueError."""
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("src.integrations.exa_client.get_exa_config", return_value={}),
            pytest.raises(Exception, match="Field required"),
        ):
            ExaClient()

    def test_init_with_environment_variable(self):
        """Test initialization retrieves API key from environment."""
        mock_config = {
            "api_key": "env-api-key",
            "base_url": "https://api.exa.ai/",
            "timeout_seconds": 60,
            "max_retries": 3,
            "base_retry_delay": 1.0,
            "search_type": "auto",
            "num_results": 10,
            "include_text": False,
            "include_highlights": True,
            "include_summary": False,
        }
        with patch(
            "src.integrations.exa_client.get_exa_config", return_value=mock_config
        ):
            client = ExaClient()
            assert client.api_key == "env-api-key"


class TestExaClientHTTPOperations:
    """Test HTTP client management and request handling."""

    @pytest_asyncio.fixture
    async def mock_client_setup(self):
        """Setup mock HTTP client for testing."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Setup default success response
            mock_response = AsyncMock()
            mock_response.json.return_value = {"test": "response"}
            mock_response.raise_for_status.return_value = None
            mock_client.request.return_value = mock_response

            yield mock_client_class, mock_client, mock_response

    @pytest.mark.asyncio
    async def test_get_client_creates_async_client(self, mock_client_setup):
        """Test _get_client creates httpx.AsyncClient with correct headers."""
        mock_client_class, mock_client, _ = mock_client_setup

        client = ExaClient(api_key="test-key")
        http_client = await client._get_client()

        assert http_client is mock_client
        mock_client_class.assert_called_once_with(
            headers={
                "x-api-key": "test-key",
                "Content-Type": "application/json",
            },
            timeout=60,
        )

    @pytest.mark.asyncio
    async def test_get_client_reuses_existing_client(self, mock_client_setup):
        """Test _get_client reuses existing client instance."""
        mock_client_class, mock_client, _ = mock_client_setup

        client = ExaClient(api_key="test-key")

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

        expected_response = {"success": True, "data": "test"}
        mock_response.json.return_value = expected_response

        client = ExaClient(api_key="test-key")
        response = await client._make_request("POST", "/test", data={"query": "test"})

        assert response == expected_response
        mock_client.request.assert_called_once_with(
            method="POST",
            url="https://api.exa.ai/test",
            json={"query": "test"},
            params=None,
        )

    @pytest.mark.asyncio
    async def test_make_request_with_params(self, mock_client_setup):
        """Test HTTP request with query parameters."""
        _, mock_client, mock_response = mock_client_setup

        client = ExaClient(api_key="test-key")
        await client._make_request("GET", "/test", params={"limit": 10, "offset": 20})

        mock_client.request.assert_called_once_with(
            method="GET",
            url="https://api.exa.ai/test",
            json=None,
            params={"limit": 10, "offset": 20},
        )


class TestExaClientRetryLogic:
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
            mock_response_error.json.return_value = {
                "error": {"message": "Rate limit exceeded"}
            }

            mock_response_success = AsyncMock()
            mock_response_success.json.return_value = {"success": True}
            mock_response_success.raise_for_status.return_value = None

            error = httpx.HTTPStatusError(
                "Too Many Requests", request=MagicMock(), response=mock_response_error
            )

            mock_client.request.side_effect = [error, error, mock_response_success]

            client = ExaClient(api_key="test-key", max_retries=3)

            with patch("asyncio.sleep") as mock_sleep:
                response = await client._make_request("POST", "/test")

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

            client = ExaClient(api_key="test-key", max_retries=2)

            with patch("asyncio.sleep") as mock_sleep:
                response = await client._make_request("POST", "/test")

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
            mock_response_error.json.return_value = {
                "error": {"message": "Rate limit exceeded"}
            }

            error = httpx.HTTPStatusError(
                "Too Many Requests", request=MagicMock(), response=mock_response_error
            )
            mock_client.request.side_effect = error

            client = ExaClient(api_key="test-key", max_retries=2)

            with (
                patch("asyncio.sleep"),
                pytest.raises(Exception, match="Exa API error: Rate limit exceeded"),
            ):
                await client._make_request("POST", "/test")

    @pytest.mark.asyncio
    async def test_non_retryable_error(self):
        """Test immediate failure for non-retryable errors."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Mock 400 bad request (not retryable)
            mock_response_error = MagicMock()
            mock_response_error.status_code = 400
            mock_response_error.json.return_value = {
                "error": {"message": "Invalid query"}
            }

            error = httpx.HTTPStatusError(
                "Bad Request", request=MagicMock(), response=mock_response_error
            )
            mock_client.request.side_effect = error

            client = ExaClient(api_key="test-key", max_retries=3)

            with pytest.raises(Exception, match="Exa API error: Invalid query"):
                await client._make_request("POST", "/test")


class TestExaClientSearchAPI:
    """Test Exa search API functionality."""

    @pytest_asyncio.fixture
    async def setup_search_mock(self):
        """Setup mock for search API responses."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            mock_response = AsyncMock()
            mock_response.raise_for_status.return_value = None
            mock_client.request.return_value = mock_response

            yield mock_client, mock_response

    @pytest.mark.asyncio
    async def test_search_basic_query(self, setup_search_mock):
        """Test basic search functionality."""
        mock_client, mock_response = setup_search_mock

        mock_api_response = {
            "requestId": "req-123",
            "resolvedSearchType": "neural",
            "searchType": "auto",
            "results": [
                {
                    "title": "LangGraph Documentation",
                    "url": "https://langchain-ai.github.io/langgraph/",
                    "publishedDate": "2024-01-15T10:30:00Z",
                    "author": "LangChain Team",
                    "score": 0.98,
                    "id": "result-1",
                    "text": "LangGraph is a library for building stateful applications...",
                    "highlights": ["LangGraph", "stateful applications"],
                    "highlightScores": [0.95, 0.87],
                    "summary": "Comprehensive guide to LangGraph framework",
                }
            ],
            "costDollars": {"total": 0.001, "search": 0.001},
        }
        mock_response.json.return_value = mock_api_response

        client = ExaClient(api_key="test-key")
        result = await client.search("LangGraph tutorial")

        # Verify response structure
        assert isinstance(result, ExaSearchResponse)
        assert result.request_id == "req-123"
        assert result.resolved_search_type == "neural"
        assert result.search_type == "auto"
        assert len(result.results) == 1

        # Verify result details
        search_result = result.results[0]
        assert isinstance(search_result, ExaSearchResult)
        assert search_result.title == "LangGraph Documentation"
        assert search_result.url == "https://langchain-ai.github.io/langgraph/"
        assert search_result.score == 0.98
        assert (
            search_result.text
            == "LangGraph is a library for building stateful applications..."
        )
        assert search_result.highlights == ["LangGraph", "stateful applications"]

        # Verify request parameters
        call_args = mock_client.request.call_args
        assert call_args[1]["method"] == "POST"
        assert call_args[1]["url"] == "https://api.exa.ai/search"
        payload = call_args[1]["json"]
        assert payload["query"] == "LangGraph tutorial"
        assert payload["type"] == "auto"  # default search type
        assert payload["numResults"] == 10  # default num_results

    @pytest.mark.asyncio
    async def test_search_with_all_options(self, setup_search_mock):
        """Test search with all available options."""
        mock_client, mock_response = setup_search_mock

        mock_response.json.return_value = {
            "requestId": "req-456",
            "resolvedSearchType": "keyword",
            "searchType": "keyword",
            "results": [],
            "costDollars": {"total": 0.002},
        }

        client = ExaClient(api_key="test-key")

        await client.search(
            query="machine learning research",
            search_type="keyword",
            num_results=25,
            include_text=True,
            include_highlights=True,
            include_summary=True,
            category="research paper",
            include_domains=["arxiv.org", "scholar.google.com"],
            exclude_domains=["wikipedia.org"],
            start_crawl_date="2024-01-01",
            end_crawl_date="2024-12-31",
            start_published_date="2023-01-01",
            end_published_date="2024-12-31",
        )

        # Verify comprehensive request payload
        call_args = mock_client.request.call_args
        payload = call_args[1]["json"]

        assert payload["query"] == "machine learning research"
        assert payload["type"] == "keyword"
        assert payload["numResults"] == 25
        assert payload["category"] == "research paper"
        assert payload["includeDomains"] == ["arxiv.org", "scholar.google.com"]
        assert payload["excludeDomains"] == ["wikipedia.org"]
        assert payload["startCrawlDate"] == "2024-01-01"
        assert payload["endCrawlDate"] == "2024-12-31"
        assert payload["startPublishedDate"] == "2023-01-01"
        assert payload["endPublishedDate"] == "2024-12-31"

        # Verify contents configuration
        assert "contents" in payload
        contents = payload["contents"]
        assert contents["text"] is True
        assert "highlights" in contents
        assert "summary" in contents

    @pytest.mark.asyncio
    async def test_search_respects_max_results_limit(self, setup_search_mock):
        """Test search enforces maximum results limit."""
        mock_client, mock_response = setup_search_mock

        mock_response.json.return_value = {
            "requestId": "req-789",
            "resolvedSearchType": "neural",
            "searchType": "neural",
            "results": [],
        }

        client = ExaClient(api_key="test-key")
        await client.search("test query", num_results=150)  # Exceeds max of 100

        call_args = mock_client.request.call_args
        payload = call_args[1]["json"]
        assert payload["numResults"] == 100  # Should be capped at 100

    @pytest.mark.asyncio
    async def test_search_date_parsing(self, setup_search_mock):
        """Test search properly handles date fields."""
        mock_client, mock_response = setup_search_mock

        mock_api_response = {
            "requestId": "req-date",
            "results": [
                {
                    "title": "Test Article",
                    "url": "https://test.com",
                    "publishedDate": "2024-03-15T14:30:00Z",
                    "id": "date-test",
                }
            ],
        }
        mock_response.json.return_value = mock_api_response

        client = ExaClient(api_key="test-key")
        result = await client.search("test")

        # Verify date parsing
        search_result = result.results[0]
        assert search_result.published_date == "2024-03-15T14:30:00Z"


class TestExaClientContentsAPI:
    """Test Exa contents API functionality."""

    @pytest.mark.asyncio
    async def test_get_contents_basic(self):
        """Test basic get_contents functionality."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            mock_response = AsyncMock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {
                "results": [
                    {
                        "id": "content-1",
                        "title": "Full Content Article",
                        "url": "https://example.com/article",
                        "text": "This is the full article content with detailed information...",
                        "highlights": ["detailed information", "comprehensive"],
                        "summary": "Article provides comprehensive information",
                    }
                ]
            }
            mock_client.request.return_value = mock_response

            client = ExaClient(api_key="test-key")
            results = await client.get_contents(
                ids=["content-1", "content-2"],
                include_text=True,
                include_highlights=True,
                include_summary=True,
            )

            assert len(results) == 1
            result = results[0]
            assert isinstance(result, ExaSearchResult)
            assert result.id == "content-1"
            assert result.title == "Full Content Article"
            assert "detailed information" in result.text
            assert result.highlights == ["detailed information", "comprehensive"]
            assert result.summary == "Article provides comprehensive information"

            # Verify request payload
            call_args = mock_client.request.call_args
            assert call_args[1]["method"] == "POST"
            assert call_args[1]["url"] == "https://api.exa.ai/contents"
            payload = call_args[1]["json"]
            assert payload["ids"] == ["content-1", "content-2"]
            assert "contents" in payload
            assert payload["contents"]["text"] is True

    @pytest.mark.asyncio
    async def test_get_contents_minimal(self):
        """Test get_contents with minimal options."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            mock_response = AsyncMock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {"results": []}
            mock_client.request.return_value = mock_response

            client = ExaClient(api_key="test-key")
            results = await client.get_contents(ids=["test-id"])

            assert results == []

            # Verify minimal request payload
            call_args = mock_client.request.call_args
            payload = call_args[1]["json"]
            assert payload["ids"] == ["test-id"]
            assert "contents" in payload
            assert payload["contents"]["text"] is True  # Default include_text=True


class TestExaClientSimilarAPI:
    """Test Exa find_similar API functionality."""

    @pytest.mark.asyncio
    async def test_find_similar_basic(self):
        """Test basic find_similar functionality."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            mock_response = AsyncMock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {
                "results": [
                    {
                        "id": "similar-1",
                        "title": "Similar Article 1",
                        "url": "https://similar1.com",
                        "score": 0.89,
                    },
                    {
                        "id": "similar-2",
                        "title": "Similar Article 2",
                        "url": "https://similar2.com",
                        "score": 0.76,
                    },
                ]
            }
            mock_client.request.return_value = mock_response

            client = ExaClient(api_key="test-key")
            results = await client.find_similar(
                url="https://reference.com/article",
                num_results=15,
                include_text=True,
            )

            assert len(results) == 2

            # Verify first result
            result1 = results[0]
            assert result1.id == "similar-1"
            assert result1.title == "Similar Article 1"
            assert result1.score == 0.89

            # Verify second result
            result2 = results[1]
            assert result2.id == "similar-2"
            assert result2.score == 0.76

            # Verify request payload
            call_args = mock_client.request.call_args
            assert call_args[1]["method"] == "POST"
            assert call_args[1]["url"] == "https://api.exa.ai/findSimilar"
            payload = call_args[1]["json"]
            assert payload["url"] == "https://reference.com/article"
            assert payload["numResults"] == 15
            assert "contents" in payload
            assert payload["contents"]["text"] is True

    @pytest.mark.asyncio
    async def test_find_similar_no_content_options(self):
        """Test find_similar without content extraction options."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            mock_response = AsyncMock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {"results": []}
            mock_client.request.return_value = mock_response

            client = ExaClient(api_key="test-key")
            await client.find_similar(
                url="https://test.com",
                include_text=False,
                include_highlights=False,
                include_summary=False,
            )

            # Verify no contents field when all content options are False
            call_args = mock_client.request.call_args
            payload = call_args[1]["json"]
            assert "contents" not in payload


class TestExaClientAnswerAPI:
    """Test Exa answer API functionality."""

    @pytest.mark.asyncio
    async def test_answer_basic(self):
        """Test basic answer functionality."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            mock_response = AsyncMock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {
                "answer": "LangGraph is a framework for building stateful applications with LLMs.",
                "sources": [
                    {"url": "https://docs.langchain.com", "title": "LangChain Docs"},
                    {
                        "url": "https://github.com/langchain-ai",
                        "title": "LangChain GitHub",
                    },
                ],
                "confidence": 0.92,
            }
            mock_client.request.return_value = mock_response

            client = ExaClient(api_key="test-key")
            result = await client.answer("What is LangGraph?")

            # Verify response structure
            assert "answer" in result
            assert "sources" in result
            assert (
                result["answer"]
                == "LangGraph is a framework for building stateful applications with LLMs."
            )
            assert len(result["sources"]) == 2
            assert result["confidence"] == 0.92

            # Verify request
            call_args = mock_client.request.call_args
            assert call_args[1]["method"] == "POST"
            assert call_args[1]["url"] == "https://api.exa.ai/answer"
            payload = call_args[1]["json"]
            assert payload["query"] == "What is LangGraph?"

    @pytest.mark.asyncio
    async def test_answer_with_options(self):
        """Test answer with domain filtering and category."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            mock_response = AsyncMock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {"answer": "Test answer"}
            mock_client.request.return_value = mock_response

            client = ExaClient(api_key="test-key")
            await client.answer(
                query="Python best practices",
                include_domains=["python.org", "realpython.com"],
                exclude_domains=["stackoverflow.com"],
                category="programming",
            )

            # Verify request payload
            call_args = mock_client.request.call_args
            payload = call_args[1]["json"]
            assert payload["query"] == "Python best practices"
            assert payload["includeDomains"] == ["python.org", "realpython.com"]
            assert payload["excludeDomains"] == ["stackoverflow.com"]
            assert payload["category"] == "programming"


class TestExaClientResearchAPI:
    """Test Exa research API functionality."""

    @pytest.mark.asyncio
    async def test_create_research_task_basic(self):
        """Test basic research task creation."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            mock_response = AsyncMock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {
                "id": "research-task-123",
                "status": "pending",
            }
            mock_client.request.return_value = mock_response

            client = ExaClient(api_key="test-key")
            task = await client.create_research_task(
                instructions="Research the latest trends in AI and machine learning",
                model="exa-research-pro",
            )

            assert isinstance(task, ExaResearchTask)
            assert task.id == "research-task-123"
            assert task.status == "pending"

            # Verify request
            call_args = mock_client.request.call_args
            assert call_args[1]["method"] == "POST"
            assert call_args[1]["url"] == "https://api.exa.ai/research/v0/tasks"
            payload = call_args[1]["json"]
            assert (
                payload["instructions"]
                == "Research the latest trends in AI and machine learning"
            )
            assert payload["model"] == "exa-research-pro"

    @pytest.mark.asyncio
    async def test_create_research_task_with_schema(self):
        """Test research task creation with output schema."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            mock_response = AsyncMock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {"id": "research-schema-456"}
            mock_client.request.return_value = mock_response

            client = ExaClient(api_key="test-key")

            schema = {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "key_findings": {"type": "array", "items": {"type": "string"}},
                    "confidence_score": {"type": "number"},
                },
                "required": ["summary", "key_findings"],
            }

            await client.create_research_task(
                instructions="Analyze AI market trends",
                output_schema=schema,
                infer_schema=True,
            )

            # Verify request payload includes output configuration
            call_args = mock_client.request.call_args
            payload = call_args[1]["json"]
            assert "output" in payload
            assert payload["output"]["schema"] == schema
            assert payload["output"]["inferSchema"] is True

    @pytest.mark.asyncio
    async def test_get_research_task(self):
        """Test getting research task status and results."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            mock_response = AsyncMock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {
                "status": "completed",
                "result": {
                    "summary": "AI is rapidly evolving with significant advances in LLMs",
                    "key_findings": ["GPT models improving", "More efficient training"],
                    "confidence_score": 0.87,
                },
            }
            mock_client.request.return_value = mock_response

            client = ExaClient(api_key="test-key")
            task = await client.get_research_task("research-task-789")

            assert isinstance(task, ExaResearchTask)
            assert task.id == "research-task-789"
            assert task.status == "completed"
            assert task.result is not None
            assert (
                task.result["summary"]
                == "AI is rapidly evolving with significant advances in LLMs"
            )
            assert task.result["confidence_score"] == 0.87

            # Verify request
            call_args = mock_client.request.call_args
            assert call_args[1]["method"] == "GET"
            assert (
                call_args[1]["url"]
                == "https://api.exa.ai/research/v0/tasks/research-task-789"
            )

    @pytest.mark.asyncio
    async def test_wait_for_research_task_completion(self):
        """Test waiting for research task completion."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Mock progression: pending -> running -> completed
            responses = [
                {"status": "pending"},
                {"status": "running"},
                {"status": "completed", "result": {"summary": "Research complete"}},
            ]

            mock_response = AsyncMock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.side_effect = responses
            mock_client.request.return_value = mock_response

            client = ExaClient(api_key="test-key")

            with patch("asyncio.sleep") as mock_sleep:
                task = await client.wait_for_research_task(
                    task_id="wait-task-123",
                    max_wait_time=60,
                    poll_interval=2,
                )

                assert task.status == "completed"
                assert task.result["summary"] == "Research complete"

                # Should have slept twice (for pending and running statuses)
                assert mock_sleep.call_count == 2
                mock_sleep.assert_called_with(2)  # poll_interval

    @pytest.mark.asyncio
    async def test_wait_for_research_task_timeout(self):
        """Test research task wait timeout."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Always return "running" status
            mock_response = AsyncMock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {"status": "running"}
            mock_client.request.return_value = mock_response

            client = ExaClient(api_key="test-key")

            with (
                patch("asyncio.sleep"),
                patch("asyncio.get_event_loop") as mock_loop,
                pytest.raises(
                    Exception,
                    match="Research task timeout-task timed out after 5 seconds",
                ),
            ):
                # Mock elapsed time progression
                mock_loop.return_value.time.side_effect = [
                    0,
                    1,
                    2,
                    3,
                    4,
                    6,
                ]  # Exceeds 5 second limit

                await client.wait_for_research_task(
                    task_id="timeout-task",
                    max_wait_time=5,
                    poll_interval=1,
                )

    @pytest.mark.asyncio
    async def test_research_end_to_end(self):
        """Test complete research workflow."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Mock create task response
            create_response = AsyncMock()
            create_response.raise_for_status.return_value = None
            create_response.json.return_value = {"id": "end-to-end-task"}

            # Mock task completion response
            complete_response = AsyncMock()
            complete_response.raise_for_status.return_value = None
            complete_response.json.return_value = {
                "status": "completed",
                "result": {"findings": "Comprehensive research results"},
            }

            # Set up responses in order
            mock_client.request.side_effect = [create_response, complete_response]

            client = ExaClient(api_key="test-key")

            with patch("asyncio.sleep"):
                result = await client.research(
                    instructions="Research AI development trends",
                    model="exa-research",
                    wait_for_completion=True,
                    max_wait_time=30,
                )

                # Should return the result directly when wait_for_completion=True
                assert result == {"findings": "Comprehensive research results"}

    @pytest.mark.asyncio
    async def test_research_no_wait(self):
        """Test research without waiting for completion."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            mock_response = AsyncMock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {"id": "no-wait-task"}
            mock_client.request.return_value = mock_response

            client = ExaClient(api_key="test-key")
            result = await client.research(
                instructions="Quick research task",
                wait_for_completion=False,
            )

            # Should return the task object when wait_for_completion=False
            assert isinstance(result, ExaResearchTask)
            assert result.id == "no-wait-task"


class TestExaClientContextManagement:
    """Test ExaClient context manager functionality."""

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """Test ExaClient as async context manager."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            async with ExaClient(api_key="test-key") as client:
                assert isinstance(client, ExaClient)
                assert client.api_key == "test-key"

            # Should have called aclose on the HTTP client
            mock_client.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_manual_close(self):
        """Test manual client closure."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            client = ExaClient(api_key="test-key")
            # Create the HTTP client
            await client._get_client()

            # Close manually
            await client.close()

            mock_client.aclose.assert_called_once()
            assert client._client is None

    @pytest.mark.asyncio
    async def test_close_without_client(self):
        """Test close when no HTTP client exists."""
        client = ExaClient(api_key="test-key")
        # Should not raise exception
        await client.close()
        assert client._client is None


class TestExaClientErrorScenarios:
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

            client = ExaClient(api_key="test-key")

            with pytest.raises(json.JSONDecodeError):
                await client.search("test query")

    @pytest.mark.asyncio
    async def test_network_error_handling(self):
        """Test handling of network errors."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Simulate network error
            mock_client.request.side_effect = httpx.ConnectError("Network unreachable")

            client = ExaClient(api_key="test-key", max_retries=2)

            with pytest.raises(
                Exception, match="Exa API request failed: Network unreachable"
            ):
                await client.search("test query")

    @pytest.mark.asyncio
    async def test_missing_result_fields(self):
        """Test handling of API responses with missing fields."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Response with minimal fields
            mock_response = AsyncMock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {
                "results": [
                    {
                        # Only required fields
                        "title": "",
                        "url": "",
                        "id": "",
                        # All optional fields missing
                    }
                ]
            }
            mock_client.request.return_value = mock_response

            client = ExaClient(api_key="test-key")
            result = await client.search("test")

            # Should handle missing fields gracefully
            search_result = result.results[0]
            assert search_result.title == ""
            assert search_result.url == ""
            assert search_result.id == ""
            assert search_result.published_date is None
            assert search_result.author is None
            assert search_result.score is None
            assert search_result.text is None
            assert search_result.highlights is None

    @pytest.mark.asyncio
    async def test_empty_results_response(self):
        """Test handling of empty results."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            mock_response = AsyncMock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {
                "requestId": "empty-123",
                "results": [],  # Empty results
            }
            mock_client.request.return_value = mock_response

            client = ExaClient(api_key="test-key")
            result = await client.search("nonexistent query")

            assert isinstance(result, ExaSearchResponse)
            assert len(result.results) == 0
            assert result.request_id == "empty-123"

    @pytest.mark.asyncio
    async def test_http_error_without_json_body(self):
        """Test HTTP error handling when response has no JSON body."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Mock response that fails JSON parsing
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_response.json.side_effect = json.JSONDecodeError("No JSON", "", 0)

            error = httpx.HTTPStatusError(
                "Internal Server Error", request=MagicMock(), response=mock_response
            )
            mock_client.request.side_effect = error

            client = ExaClient(api_key="test-key")

            with pytest.raises(
                Exception, match="Exa API error: 500 Internal Server Error"
            ):
                await client.search("test query")


# Performance and integration test markers
@pytest.mark.integration
@pytest.mark.skipif(
    "EXA_API_KEY" not in locals() and "EXA_API_KEY" not in globals(),
    reason="EXA_API_KEY not available for integration tests",
)
class TestExaClientLiveIntegration:
    """Live integration tests requiring actual API key."""

    @pytest.mark.asyncio
    async def test_live_search_request(self):
        """Test live search request with real API."""
        import os

        api_key = os.getenv("EXA_API_KEY")
        if not api_key:
            pytest.skip("EXA_API_KEY environment variable not set")

        async with ExaClient(api_key=api_key) as client:
            result = await client.search(
                query="Python async programming tutorial",
                search_type="neural",
                num_results=3,
                include_highlights=True,
            )

            assert isinstance(result, ExaSearchResponse)
            assert len(result.results) > 0
            assert all(r.url.startswith("http") for r in result.results)
            assert all(r.title for r in result.results)

    @pytest.mark.asyncio
    async def test_live_research_task(self):
        """Test live research task creation and polling."""
        import os

        api_key = os.getenv("EXA_API_KEY")
        if not api_key:
            pytest.skip("EXA_API_KEY environment variable not set")

        async with ExaClient(api_key=api_key) as client:
            # Create a simple research task
            task = await client.create_research_task(
                instructions="Briefly summarize the current state of Python async programming",
                model="exa-research",
            )

            assert isinstance(task, ExaResearchTask)
            assert task.id
            assert task.status in ["pending", "running"]

            # Check task status (don't wait for completion in tests)
            status_task = await client.get_research_task(task.id)
            assert status_task.id == task.id
            assert status_task.status in ["pending", "running", "completed", "failed"]
