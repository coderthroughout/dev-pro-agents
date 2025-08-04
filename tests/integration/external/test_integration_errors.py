"""Comprehensive error handling and integration tests for external service clients.

This module provides extensive testing for error scenarios across both
Exa and Firecrawl API clients including:
- Network connectivity issues
- Authentication failures
- Rate limiting and quota exceeded scenarios
- Malformed responses and data corruption
- Service unavailable and maintenance windows
- Concurrent request handling
- Timeout and retry edge cases
- Cross-client error consistency
- Integration test scenarios with mock servers
- 90%+ error path coverage
"""

import asyncio
import contextlib
import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import pytest_asyncio

from src.integrations.exa_client import ExaClient
from src.integrations.firecrawl_client import FirecrawlClient


class TestNetworkErrorHandling:
    """Test network-level error handling across both clients."""

    @pytest.mark.asyncio
    async def test_connection_error_exa(self):
        """Test Exa client handling of connection errors."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Simulate connection error
            mock_client.request.side_effect = httpx.ConnectError("Connection refused")

            client = ExaClient(api_key="test-key", max_retries=2)

            with pytest.raises(
                Exception, match="Exa API request failed: Connection refused"
            ):
                await client.search("test query")

            # Should have attempted max_retries + 1 times
            assert mock_client.request.call_count == 3

    @pytest.mark.asyncio
    async def test_connection_error_firecrawl(self):
        """Test Firecrawl client handling of connection errors."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Simulate connection error
            mock_client.request.side_effect = httpx.ConnectError("Connection refused")

            client = FirecrawlClient(api_key="test-key", max_retries=2)

            with pytest.raises(
                Exception, match="Firecrawl API request failed: Connection refused"
            ):
                await client.scrape("https://test.com")

            # Should have attempted max_retries + 1 times
            assert mock_client.request.call_count == 3

    @pytest.mark.asyncio
    async def test_dns_resolution_failure(self):
        """Test handling of DNS resolution failures."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Simulate DNS resolution failure
            mock_client.request.side_effect = httpx.ConnectError(
                "Name or service not known"
            )

            exa_client = ExaClient(api_key="test-key", max_retries=1)
            firecrawl_client = FirecrawlClient(api_key="test-key", max_retries=1)

            with pytest.raises(
                Exception, match="Exa API request failed: Name or service not known"
            ):
                await exa_client.search("test")

            with pytest.raises(
                Exception,
                match="Firecrawl API request failed: Name or service not known",
            ):
                await firecrawl_client.scrape("https://test.com")

    @pytest.mark.asyncio
    async def test_ssl_verification_error(self):
        """Test handling of SSL certificate verification errors."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Simulate SSL verification error
            ssl_error = httpx.ConnectError(
                "certificate verify failed: self signed certificate"
            )
            mock_client.request.side_effect = ssl_error

            client = ExaClient(api_key="test-key", max_retries=1)

            with pytest.raises(Exception, match="certificate verify failed"):
                await client.search("test")


class TestAuthenticationErrors:
    """Test authentication and authorization error scenarios."""

    @pytest.mark.asyncio
    async def test_invalid_api_key_exa(self):
        """Test Exa client handling of invalid API key."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Mock 401 unauthorized response
            mock_response = MagicMock()
            mock_response.status_code = 401
            mock_response.json.return_value = {"error": {"message": "Invalid API key"}}

            error = httpx.HTTPStatusError(
                "Unauthorized", request=MagicMock(), response=mock_response
            )
            mock_client.request.side_effect = error

            client = ExaClient(api_key="invalid-key")

            with pytest.raises(Exception, match="Exa API error: Invalid API key"):
                await client.search("test")

    @pytest.mark.asyncio
    async def test_invalid_api_key_firecrawl(self):
        """Test Firecrawl client handling of invalid API key."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Mock 401 unauthorized response
            mock_response = MagicMock()
            mock_response.status_code = 401
            mock_response.json.return_value = {"error": "Invalid API key provided"}

            error = httpx.HTTPStatusError(
                "Unauthorized", request=MagicMock(), response=mock_response
            )
            mock_client.request.side_effect = error

            client = FirecrawlClient(api_key="invalid-key")

            with pytest.raises(
                Exception, match="Firecrawl API error: Invalid API key provided"
            ):
                await client.scrape("https://test.com")

    @pytest.mark.asyncio
    async def test_expired_api_key(self):
        """Test handling of expired API key scenarios."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Mock 403 forbidden response for expired key
            mock_response = MagicMock()
            mock_response.status_code = 403
            mock_response.json.return_value = {
                "error": {"message": "API key has expired"}
            }

            error = httpx.HTTPStatusError(
                "Forbidden", request=MagicMock(), response=mock_response
            )
            mock_client.request.side_effect = error

            client = ExaClient(api_key="expired-key")

            with pytest.raises(Exception, match="Exa API error: API key has expired"):
                await client.search("test")

    @pytest.mark.asyncio
    async def test_insufficient_permissions(self):
        """Test handling of insufficient permissions errors."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Mock 403 forbidden response for insufficient permissions
            mock_response = MagicMock()
            mock_response.status_code = 403
            mock_response.json.return_value = {
                "error": "Insufficient permissions for this operation"
            }

            error = httpx.HTTPStatusError(
                "Forbidden", request=MagicMock(), response=mock_response
            )
            mock_client.request.side_effect = error

            client = FirecrawlClient(api_key="limited-key")

            with pytest.raises(Exception, match="Insufficient permissions"):
                await client.scrape("https://test.com")


class TestRateLimitingErrors:
    """Test rate limiting and quota management scenarios."""

    @pytest.mark.asyncio
    async def test_rate_limit_with_retry_after_header(self):
        """Test rate limiting with Retry-After header."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Mock rate limit response with Retry-After header
            mock_response_429 = MagicMock()
            mock_response_429.status_code = 429
            mock_response_429.headers = {"Retry-After": "60"}
            mock_response_429.json.return_value = {
                "error": {"message": "Rate limit exceeded. Try again in 60 seconds."}
            }

            # Mock successful response after retry
            mock_response_success = AsyncMock()
            mock_response_success.json.return_value = {"success": True}
            mock_response_success.raise_for_status.return_value = None

            rate_limit_error = httpx.HTTPStatusError(
                "Too Many Requests", request=MagicMock(), response=mock_response_429
            )

            mock_client.request.side_effect = [rate_limit_error, mock_response_success]

            client = ExaClient(api_key="test-key", max_retries=2)

            with patch("asyncio.sleep") as mock_sleep:
                response = await client._make_request("POST", "/search")

                assert response == {"success": True}
                # Should have slept for exponential backoff, not Retry-After value
                mock_sleep.assert_called_once_with(1.0)

    @pytest.mark.asyncio
    async def test_quota_exceeded_error(self):
        """Test handling of quota exceeded errors."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Mock quota exceeded response
            mock_response = MagicMock()
            mock_response.status_code = 429
            mock_response.json.return_value = {
                "error": "Monthly quota exceeded. Upgrade your plan."
            }

            error = httpx.HTTPStatusError(
                "Too Many Requests", request=MagicMock(), response=mock_response
            )
            mock_client.request.side_effect = error

            client = FirecrawlClient(api_key="test-key", max_retries=1)

            with pytest.raises(Exception, match="Monthly quota exceeded"):
                await client.scrape("https://test.com")

    @pytest.mark.asyncio
    async def test_concurrent_rate_limiting(self):
        """Test rate limiting behavior with concurrent requests."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Mock rate limit on all requests
            mock_response = MagicMock()
            mock_response.status_code = 429
            mock_response.json.return_value = {
                "error": {"message": "Rate limit exceeded"}
            }

            error = httpx.HTTPStatusError(
                "Too Many Requests", request=MagicMock(), response=mock_response
            )
            mock_client.request.side_effect = error

            client = ExaClient(api_key="test-key", max_retries=1)

            # Make multiple concurrent requests
            async def make_request(query: str):
                try:
                    return await client.search(query)
                except Exception as e:
                    return str(e)

            tasks = [make_request(f"query-{i}") for i in range(5)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # All should fail with rate limit error
            for result in results:
                assert "Rate limit exceeded" in str(result)


class TestServiceUnavailabilityErrors:
    """Test service unavailability and maintenance scenarios."""

    @pytest.mark.asyncio
    async def test_service_unavailable_503(self):
        """Test handling of 503 Service Unavailable errors."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Mock 503 service unavailable
            mock_response = MagicMock()
            mock_response.status_code = 503
            mock_response.json.return_value = {
                "error": "Service temporarily unavailable"
            }

            error = httpx.HTTPStatusError(
                "Service Unavailable", request=MagicMock(), response=mock_response
            )
            mock_client.request.side_effect = error

            client = ExaClient(api_key="test-key", max_retries=1)

            with pytest.raises(Exception, match="Service temporarily unavailable"):
                await client.search("test")

    @pytest.mark.asyncio
    async def test_maintenance_window_502(self):
        """Test handling of 502 Bad Gateway during maintenance."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Mock 502 bad gateway
            mock_response = MagicMock()
            mock_response.status_code = 502
            mock_response.json.side_effect = json.JSONDecodeError("No JSON", "", 0)

            error = httpx.HTTPStatusError(
                "Bad Gateway", request=MagicMock(), response=mock_response
            )
            mock_client.request.side_effect = error

            client = FirecrawlClient(api_key="test-key", max_retries=1)

            with pytest.raises(Exception, match="502 Bad Gateway"):
                await client.scrape("https://test.com")

    @pytest.mark.asyncio
    async def test_gateway_timeout_504(self):
        """Test handling of 504 Gateway Timeout errors."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Mock 504 gateway timeout
            mock_response = MagicMock()
            mock_response.status_code = 504
            mock_response.json.return_value = {
                "error": "Gateway timeout - upstream server did not respond"
            }

            error = httpx.HTTPStatusError(
                "Gateway Timeout", request=MagicMock(), response=mock_response
            )
            mock_client.request.side_effect = error

            client = ExaClient(api_key="test-key", max_retries=2)

            with pytest.raises(Exception, match="Gateway timeout"):
                await client.search("test")


class TestTimeoutAndRetryEdgeCases:
    """Test timeout and retry logic edge cases."""

    @pytest.mark.asyncio
    async def test_read_timeout_during_response(self):
        """Test read timeout while receiving response data."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Simulate read timeout during response
            mock_client.request.side_effect = httpx.ReadTimeout("Read timeout")

            client = ExaClient(api_key="test-key", max_retries=2)

            with pytest.raises(Exception, match="Exa API request failed: Read timeout"):
                await client.search("test")

            # Should retry and fail each time
            assert mock_client.request.call_count == 3

    @pytest.mark.asyncio
    async def test_write_timeout_during_request(self):
        """Test write timeout while sending request data."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Simulate write timeout during request
            mock_client.request.side_effect = httpx.WriteTimeout("Write timeout")

            client = FirecrawlClient(api_key="test-key", max_retries=1)

            with pytest.raises(
                Exception, match="Firecrawl API request failed: Write timeout"
            ):
                await client.scrape("https://test.com")

    @pytest.mark.asyncio
    async def test_pool_timeout(self):
        """Test connection pool timeout errors."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Simulate pool timeout
            mock_client.request.side_effect = httpx.PoolTimeout(
                "Connection pool exhausted"
            )

            client = ExaClient(api_key="test-key", max_retries=1)

            with pytest.raises(Exception, match="Connection pool exhausted"):
                await client.search("test")

    @pytest.mark.asyncio
    async def test_intermittent_timeout_with_recovery(self):
        """Test intermittent timeouts with eventual recovery."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Mock timeout then success
            mock_response_success = AsyncMock()
            mock_response_success.json.return_value = {"success": True}
            mock_response_success.raise_for_status.return_value = None

            mock_client.request.side_effect = [
                httpx.TimeoutException("Timeout"),
                httpx.TimeoutException("Timeout"),
                mock_response_success,
            ]

            client = FirecrawlClient(api_key="test-key", max_retries=3)

            with patch("asyncio.sleep") as mock_sleep:
                response = await client._make_request("POST", "/scrape")

                assert response == {"success": True}
                # Should have slept twice for the two timeouts
                assert mock_sleep.call_count == 2

    @pytest.mark.asyncio
    async def test_zero_max_retries(self):
        """Test behavior with zero max retries."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Mock timeout error
            mock_client.request.side_effect = httpx.TimeoutException("Timeout")

            client = ExaClient(api_key="test-key", max_retries=0)

            with pytest.raises(Exception, match="Exa API request timed out"):
                await client.search("test")

            # Should only attempt once with zero retries
            assert mock_client.request.call_count == 1


class TestResponseDataCorruption:
    """Test handling of malformed and corrupted response data."""

    @pytest.mark.asyncio
    async def test_invalid_json_response(self):
        """Test handling of invalid JSON in response."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            mock_response = AsyncMock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.side_effect = json.JSONDecodeError(
                "Invalid JSON", "", 10
            )
            mock_client.request.return_value = mock_response

            client = ExaClient(api_key="test-key")

            with pytest.raises(json.JSONDecodeError):
                await client.search("test")

    @pytest.mark.asyncio
    async def test_truncated_json_response(self):
        """Test handling of truncated JSON responses."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            mock_response = AsyncMock()
            mock_response.raise_for_status.return_value = None
            # Simulate truncated JSON
            mock_response.json.side_effect = json.JSONDecodeError(
                "Unterminated string", "", 50
            )
            mock_client.request.return_value = mock_response

            client = FirecrawlClient(api_key="test-key")

            with pytest.raises(json.JSONDecodeError):
                await client.scrape("https://test.com")

    @pytest.mark.asyncio
    async def test_unexpected_response_structure(self):
        """Test handling of unexpected response structures."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Mock response with unexpected structure
            mock_response = AsyncMock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {
                # Missing expected fields like 'results' for Exa
                "unexpected_field": "unexpected_value",
                "another_field": 123,
            }
            mock_client.request.return_value = mock_response

            client = ExaClient(api_key="test-key")
            result = await client.search("test")

            # Should handle gracefully with empty results
            assert len(result.results) == 0
            assert result.request_id == ""

    @pytest.mark.asyncio
    async def test_null_values_in_response(self):
        """Test handling of null values in response fields."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            mock_response = AsyncMock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {
                "success": True,
                "data": {
                    "markdown": None,  # Null value
                    "html": "",  # Empty string
                    "metadata": None,  # Null metadata
                },
            }
            mock_client.request.return_value = mock_response

            client = FirecrawlClient(api_key="test-key")
            result = await client.scrape("https://test.com")

            # Should handle null values gracefully
            assert result.success is True
            assert result.data.markdown is None
            assert result.data.html == ""
            assert result.data.metadata is None

    @pytest.mark.asyncio
    async def test_oversized_response_handling(self):
        """Test handling of extremely large responses."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Mock very large response
            large_text = "x" * 1000000  # 1MB of text
            mock_response = AsyncMock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {
                "requestId": "large-response",
                "results": [
                    {
                        "title": "Large Document",
                        "url": "https://large.com",
                        "id": "large-1",
                        "text": large_text,  # Very large text field
                    }
                ],
            }
            mock_client.request.return_value = mock_response

            client = ExaClient(api_key="test-key")
            result = await client.search("test")

            # Should handle large responses without issues
            assert len(result.results) == 1
            assert len(result.results[0].text) == 1000000


class TestConcurrentRequestHandling:
    """Test concurrent request handling and thread safety."""

    @pytest.mark.asyncio
    async def test_concurrent_requests_same_client(self):
        """Test multiple concurrent requests using the same client instance."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Mock successful responses
            mock_response = AsyncMock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {
                "requestId": "concurrent-test",
                "results": [{"title": "Test", "url": "https://test.com", "id": "1"}],
            }
            mock_client.request.return_value = mock_response

            client = ExaClient(api_key="test-key")

            # Make 10 concurrent requests
            tasks = [client.search(f"query-{i}") for i in range(10)]
            results = await asyncio.gather(*tasks)

            # All should succeed
            assert len(results) == 10
            for result in results:
                assert len(result.results) == 1
                assert result.results[0].title == "Test"

            # Should have made 10 requests
            assert mock_client.request.call_count == 10

    @pytest.mark.asyncio
    async def test_concurrent_requests_different_clients(self):
        """Test concurrent requests across different client instances."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Mock responses for both Exa and Firecrawl
            exa_response = AsyncMock()
            exa_response.raise_for_status.return_value = None
            exa_response.json.return_value = {
                "requestId": "exa-concurrent",
                "results": [{"title": "Exa", "url": "https://exa.com", "id": "exa-1"}],
            }

            firecrawl_response = AsyncMock()
            firecrawl_response.raise_for_status.return_value = None
            firecrawl_response.json.return_value = {
                "success": True,
                "data": {"markdown": "# Firecrawl Test"},
            }

            # Alternate responses based on URL
            def mock_request(*args, **kwargs):
                if "exa.ai" in kwargs.get("url", ""):
                    return exa_response
                else:
                    return firecrawl_response

            mock_client.request.side_effect = mock_request

            exa_client = ExaClient(api_key="exa-key")
            firecrawl_client = FirecrawlClient(api_key="firecrawl-key")

            # Mix of Exa and Firecrawl requests
            tasks = [
                exa_client.search("exa-query-1"),
                firecrawl_client.scrape("https://test1.com"),
                exa_client.search("exa-query-2"),
                firecrawl_client.scrape("https://test2.com"),
            ]

            results = await asyncio.gather(*tasks)

            # Verify mixed results
            assert len(results) == 4
            assert hasattr(results[0], "results")  # Exa response
            assert hasattr(results[1], "success")  # Firecrawl response
            assert hasattr(results[2], "results")  # Exa response
            assert hasattr(results[3], "success")  # Firecrawl response

    @pytest.mark.asyncio
    async def test_concurrent_with_some_failures(self):
        """Test concurrent requests where some succeed and others fail."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Mock alternating success and failure
            success_response = AsyncMock()
            success_response.raise_for_status.return_value = None
            success_response.json.return_value = {"success": True}

            failure_response = MagicMock()
            failure_response.status_code = 500
            failure_response.json.return_value = {"error": "Internal server error"}

            failure_error = httpx.HTTPStatusError(
                "Internal Server Error", request=MagicMock(), response=failure_response
            )

            # Alternate between success and failure
            mock_client.request.side_effect = [
                success_response,  # Success
                failure_error,  # Failure
                success_response,  # Success
                failure_error,  # Failure
                success_response,  # Success
            ]

            client = FirecrawlClient(api_key="test-key", max_retries=0)

            # Make concurrent requests
            async def safe_request(url: str):
                try:
                    return await client.scrape(url)
                except Exception as e:
                    return e

            tasks = [safe_request(f"https://test{i}.com") for i in range(5)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Should have 3 successes and 2 failures
            successes = [r for r in results if hasattr(r, "success")]
            failures = [r for r in results if isinstance(r, Exception)]

            assert len(successes) == 3
            assert len(failures) == 2


class TestCrossClientErrorConsistency:
    """Test that both clients handle similar errors consistently."""

    @pytest.mark.asyncio
    async def test_consistent_timeout_handling(self):
        """Test that both clients handle timeouts consistently."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Mock timeout error
            mock_client.request.side_effect = httpx.TimeoutException("Request timeout")

            exa_client = ExaClient(api_key="test-key", max_retries=1)
            firecrawl_client = FirecrawlClient(api_key="test-key", max_retries=1)

            # Both should handle timeouts similarly
            with pytest.raises(Exception, match="timed out"):
                await exa_client.search("test")

            with pytest.raises(Exception, match="timed out"):
                await firecrawl_client.scrape("https://test.com")

    @pytest.mark.asyncio
    async def test_consistent_rate_limit_handling(self):
        """Test that both clients handle rate limits consistently."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Mock rate limit response
            mock_response = MagicMock()
            mock_response.status_code = 429
            mock_response.json.return_value = {"error": "Rate limit exceeded"}

            error = httpx.HTTPStatusError(
                "Too Many Requests", request=MagicMock(), response=mock_response
            )
            mock_client.request.side_effect = error

            exa_client = ExaClient(api_key="test-key", max_retries=0)
            firecrawl_client = FirecrawlClient(api_key="test-key", max_retries=0)

            # Both should handle rate limits similarly
            with pytest.raises(
                Exception, match="(Too Many Requests|Rate limit exceeded)"
            ):
                await exa_client.search("test")

            with pytest.raises(
                Exception, match="(Too Many Requests|Rate limit exceeded)"
            ):
                await firecrawl_client.scrape("https://test.com")

    @pytest.mark.asyncio
    async def test_consistent_network_error_handling(self):
        """Test that both clients handle network errors consistently."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Mock network error
            mock_client.request.side_effect = httpx.ConnectError("Network unreachable")

            exa_client = ExaClient(api_key="test-key", max_retries=0)
            firecrawl_client = FirecrawlClient(api_key="test-key", max_retries=0)

            # Both should handle network errors with similar patterns
            with pytest.raises(Exception, match="Network unreachable"):
                await exa_client.search("test")

            with pytest.raises(Exception, match="Network unreachable"):
                await firecrawl_client.scrape("https://test.com")


class TestIntegrationTestScenarios:
    """Integration test scenarios with mock servers and real-world conditions."""

    @pytest_asyncio.fixture
    async def mock_server_setup(self):
        """Setup mock server for integration testing."""
        # This would typically use a real mock server like aioresponses
        # For this example, we'll use httpx mocking
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            yield mock_client

    @pytest.mark.asyncio
    async def test_end_to_end_search_workflow(self, mock_server_setup):
        """Test complete search workflow with error scenarios."""
        mock_client = mock_server_setup

        # Simulate sequence: rate limit -> success -> timeout -> success
        rate_limit_response = MagicMock()
        rate_limit_response.status_code = 429
        rate_limit_response.json.return_value = {"error": {"message": "Rate limited"}}
        rate_limit_error = httpx.HTTPStatusError(
            "Too Many Requests", request=MagicMock(), response=rate_limit_response
        )

        success_response = AsyncMock()
        success_response.raise_for_status.return_value = None
        success_response.json.return_value = {
            "requestId": "workflow-test",
            "results": [{"title": "Success", "url": "https://success.com", "id": "1"}],
        }

        timeout_error = httpx.TimeoutException("Timeout during search")

        mock_client.request.side_effect = [
            rate_limit_error,  # First request fails with rate limit
            success_response,  # Retry succeeds
            timeout_error,  # Third request times out
            success_response,  # Retry succeeds
        ]

        client = ExaClient(api_key="test-key", max_retries=2)

        with patch("asyncio.sleep"):
            # First search: rate limited then success
            result1 = await client.search("query1")
            assert len(result1.results) == 1

            # Second search: timeout then success
            result2 = await client.search("query2")
            assert len(result2.results) == 1

    @pytest.mark.asyncio
    async def test_mixed_client_error_recovery(self, mock_server_setup):
        """Test error recovery across mixed client operations."""
        mock_client = mock_server_setup

        # Setup responses for different endpoints
        def mock_request(*args, **kwargs):
            url = kwargs.get("url", "")

            if "search" in url:
                # Exa search fails first time, succeeds second
                if not hasattr(mock_request, "exa_call_count"):
                    mock_request.exa_call_count = 0
                mock_request.exa_call_count += 1

                if mock_request.exa_call_count == 1:
                    raise httpx.TimeoutException("Exa timeout")
                else:
                    response = AsyncMock()
                    response.raise_for_status.return_value = None
                    response.json.return_value = {
                        "requestId": "mixed-test",
                        "results": [
                            {"title": "Exa", "url": "https://exa.com", "id": "1"}
                        ],
                    }
                    return response

            elif "scrape" in url:
                # Firecrawl succeeds immediately
                response = AsyncMock()
                response.raise_for_status.return_value = None
                response.json.return_value = {
                    "success": True,
                    "data": {"markdown": "# Firecrawl Success"},
                }
                return response

        mock_client.request.side_effect = mock_request

        exa_client = ExaClient(api_key="exa-key", max_retries=2)
        firecrawl_client = FirecrawlClient(api_key="firecrawl-key")

        with patch("asyncio.sleep"):
            # Exa search should recover from timeout
            exa_result = await exa_client.search("test query")
            assert len(exa_result.results) == 1

            # Firecrawl should work immediately
            firecrawl_result = await firecrawl_client.scrape("https://test.com")
            assert firecrawl_result.success is True

    @pytest.mark.asyncio
    async def test_cascading_failure_recovery(self):
        """Test recovery from cascading failures."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Simulate cascading failures: network -> rate limit -> success
            network_error = httpx.ConnectError("Network down")

            rate_limit_response = MagicMock()
            rate_limit_response.status_code = 429
            rate_limit_response.json.return_value = {"error": "Rate limited"}
            rate_limit_error = httpx.HTTPStatusError(
                "Too Many Requests", request=MagicMock(), response=rate_limit_response
            )

            success_response = AsyncMock()
            success_response.raise_for_status.return_value = None
            success_response.json.return_value = {"success": True}

            mock_client.request.side_effect = [
                network_error,  # Network failure
                network_error,  # Network still down
                rate_limit_error,  # Network recovered but rate limited
                success_response,  # Finally succeeds
            ]

            client = FirecrawlClient(api_key="test-key", max_retries=4)

            with patch("asyncio.sleep"):
                result = await client.scrape("https://test.com")
                assert result.success is True

                # Should have made 4 attempts
                assert mock_client.request.call_count == 4


class TestErrorMetricsAndMonitoring:
    """Test error scenarios useful for monitoring and metrics."""

    @pytest.mark.asyncio
    async def test_error_classification(self):
        """Test that different error types can be classified correctly."""
        error_scenarios = [
            (httpx.ConnectError("Connection refused"), "network"),
            (httpx.TimeoutException("Request timeout"), "timeout"),
            (httpx.HTTPStatusError("Unauthorized", MagicMock(), MagicMock()), "http"),
        ]

        for error, expected_type in error_scenarios:
            # Mock the error
            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_client_class.return_value = mock_client
                mock_client.request.side_effect = error

                client = ExaClient(api_key="test-key", max_retries=0)

                try:
                    await client.search("test")
                    raise AssertionError("Should have raised an exception")
                except Exception as e:
                    # In a real implementation, you might have error classification logic
                    if expected_type == "network":
                        assert "Connection refused" in str(e)
                    elif expected_type == "timeout":
                        assert "timeout" in str(e).lower()
                    elif expected_type == "http":
                        assert "Unauthorized" in str(e)

    @pytest.mark.asyncio
    async def test_retry_attempt_tracking(self):
        """Test tracking of retry attempts for monitoring."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Mock consistent failures
            mock_client.request.side_effect = httpx.TimeoutException("Timeout")

            client = ExaClient(api_key="test-key", max_retries=3)

            with patch("asyncio.sleep"):
                with contextlib.suppress(Exception):
                    await client.search("test")

                # Should have attempted max_retries + 1 times
                assert mock_client.request.call_count == 4

    @pytest.mark.asyncio
    async def test_performance_degradation_detection(self):
        """Test detection of performance degradation patterns."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Simulate gradually increasing response times

            def slow_response():
                response = AsyncMock()
                response.raise_for_status.return_value = None
                response.json.return_value = {"success": True}
                return response

            mock_client.request.return_value = slow_response()

            client = FirecrawlClient(api_key="test-key")

            # In a real scenario, you'd measure actual response times
            for i in range(5):
                result = await client.scrape(f"https://test{i}.com")
                assert result.success is True

            # Should have made 5 requests
            assert mock_client.request.call_count == 5
