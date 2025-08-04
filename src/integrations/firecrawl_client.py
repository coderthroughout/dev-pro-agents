"""Direct Firecrawl API client for LangGraph agent integration.

This module provides a clean, async-compatible wrapper around the Firecrawl API
without dependencies on MCP servers or additional frameworks.
"""

import asyncio
from typing import Any

import httpx
from pydantic import BaseModel, Field

from ..config import FirecrawlSettings, get_firecrawl_config


class FirecrawlMetadata(BaseModel):
    """Metadata for scraped content."""

    title: str | None = None
    description: str | None = None
    language: str | None = None
    keywords: str | None = None
    robots: str | None = None
    og_title: str | None = Field(None, alias="ogTitle")
    og_description: str | None = Field(None, alias="ogDescription")
    og_url: str | None = Field(None, alias="ogUrl")
    og_image: str | None = Field(None, alias="ogImage")
    og_site_name: str | None = Field(None, alias="ogSiteName")
    source_url: str | None = Field(None, alias="sourceURL")
    status_code: int | None = Field(None, alias="statusCode")


class FirecrawlDocument(BaseModel):
    """Single document from Firecrawl API."""

    markdown: str | None = None
    html: str | None = None
    raw_html: str | None = Field(None, alias="rawHtml")
    links: list[str] | None = None
    screenshot: str | None = None
    json_data: dict[str, Any] | None = Field(None, alias="json")
    metadata: FirecrawlMetadata | None = None


class FirecrawlScrapeResponse(BaseModel):
    """Response from Firecrawl scrape API."""

    success: bool
    data: FirecrawlDocument | None = None
    error: str | None = None


class FirecrawlCrawlResponse(BaseModel):
    """Response from Firecrawl crawl API."""

    success: bool
    id: str | None = None
    url: str | None = None
    status: str | None = None
    total: int | None = None
    completed: int | None = None
    credits_used: int | None = Field(None, alias="creditsUsed")
    expires_at: str | None = Field(None, alias="expiresAt")
    next_url: str | None = Field(None, alias="next")
    data: list[FirecrawlDocument] | None = None
    error: str | None = None


class FirecrawlMapResponse(BaseModel):
    """Response from Firecrawl map API."""

    success: bool
    links: list[str] | None = None
    error: str | None = None


class FirecrawlAction(BaseModel):
    """Action for page interaction using Firecrawl's Actions API."""

    type: (
        str  # wait, click, write, press, scroll, screenshot, scrape, executeJavascript
    )
    selector: str | None = None
    text: str | None = None
    key: str | None = None
    milliseconds: int | None = None
    script: str | None = None  # For executeJavascript actions
    direction: str | None = None  # For scroll actions: up, down
    full_page: bool | None = Field(None, alias="fullPage")  # For screenshot actions


class FirecrawlLocation(BaseModel):
    """Location settings for scraping."""

    country: str = "US"
    languages: list[str] | None = None


class FirecrawlAgent(BaseModel):
    """FIRE-1 agent configuration for intelligent web interaction."""

    model: str = "FIRE-1"  # Currently only FIRE-1 is supported
    prompt: str  # Required: Instructions for the agent on how to interact with the page


class FirecrawlClient:
    """Direct Firecrawl API client optimized for LangGraph agent integration.

    Features:
    - Full async/await support
    - Comprehensive error handling with retries
    - Agent-friendly response models
    - Batch operations
    - WebSocket support for real-time crawling
    - Authentication management
    """

    def __init__(
        self,
        config: FirecrawlSettings | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: int | None = None,
        max_retries: int | None = None,
    ):
        """Initialize Firecrawl client with centralized configuration.

        Args:
            config: FirecrawlSettings instance. If None, uses global settings.
            api_key: Firecrawl API key override. If None, uses config or env variable.
            base_url: Base URL override. If None, uses config default.
            timeout: Request timeout override. If None, uses config default.
            max_retries: Max retries override. If None, uses config default.

        """
        # Use provided config or get from global settings
        if config is None:
            config_dict = get_firecrawl_config()
            self.config = FirecrawlSettings(**config_dict)
        else:
            self.config = config

        # Apply overrides if provided
        self.api_key = api_key or self.config.api_key
        if not self.api_key:
            raise ValueError(
                "FIRECRAWL_API_KEY must be provided via config, parameter, "
                "or environment variable"
            )

        self.base_url = base_url or str(self.config.base_url)
        self.timeout = timeout or self.config.timeout_seconds
        self.max_retries = max_retries or self.config.max_retries
        self.base_retry_delay = self.config.base_retry_delay

        # Store config defaults for method usage
        self.default_formats = self.config.default_formats.copy()
        self.default_only_main_content = self.config.only_main_content
        self.default_crawl_limit = self.config.crawl_limit
        self.default_max_wait_time = self.config.max_wait_time
        self.default_poll_interval = self.config.poll_interval

        # Initialize async HTTP client
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=self.timeout,
            )
        return self._client

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make HTTP request with retry logic and error handling."""
        client = await self._get_client()
        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        for attempt in range(self.max_retries + 1):
            try:
                response = await client.request(
                    method=method,
                    url=url,
                    json=data,
                    params=params,
                )
                response.raise_for_status()
                return response.json()

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429 and attempt < self.max_retries:
                    wait_time = self.base_retry_delay * (2**attempt)
                    await asyncio.sleep(wait_time)
                    continue

                error_detail = "Unknown error"
                try:
                    error_data = e.response.json()
                    error_detail = error_data.get("error", str(e))
                except Exception:
                    error_detail = str(e)

                raise Exception(f"Firecrawl API error: {error_detail}") from e

            except httpx.TimeoutException:
                if attempt < self.max_retries:
                    wait_time = self.base_retry_delay * (2**attempt)
                    await asyncio.sleep(wait_time)
                    continue
                raise Exception("Firecrawl API request timed out") from None

            except Exception as e:
                if attempt < self.max_retries:
                    wait_time = self.base_retry_delay * (2**attempt)
                    await asyncio.sleep(wait_time)
                    continue
                raise Exception(f"Firecrawl API request failed: {e!s}") from e

        raise Exception("Maximum retry attempts exceeded")

    async def scrape(
        self,
        url: str,
        formats: list[str] | None = None,
        include_tags: list[str] | None = None,
        exclude_tags: list[str] | None = None,
        only_main_content: bool | None = None,
        actions: list[FirecrawlAction] | None = None,
        agent: FirecrawlAgent | None = None,
        location: FirecrawlLocation | None = None,
        json_schema: dict[str, Any] | None = None,
        json_prompt: str | None = None,
        wait_for: int | None = None,
        timeout: int | None = None,
    ) -> FirecrawlScrapeResponse:
        """Scrape a single URL with Firecrawl.

        Args:
            url: URL to scrape.
            formats: Output formats
                ("markdown", "html", "rawHtml", "screenshot", "links", "json").
            include_tags: HTML tags to include.
            exclude_tags: HTML tags to exclude.
            only_main_content: Extract only main content.
            actions: List of actions to perform before scraping.
            agent: FIRE-1 agent configuration for intelligent interaction.
            location: Location settings for geolocation.
            json_schema: JSON schema for structured extraction.
            json_prompt: Prompt for schema-less extraction.
            wait_for: Time to wait for dynamic content (ms).
            timeout: Request timeout override (ms).

        Returns:
            FirecrawlScrapeResponse with scraped content.

        """
        # Apply configuration defaults
        formats = formats or self.default_formats
        only_main_content = (
            only_main_content
            if only_main_content is not None
            else self.default_only_main_content
        )

        # Build request payload
        payload = {
            "url": url,
            "formats": formats,
            "onlyMainContent": only_main_content,
        }

        # Add optional parameters
        if include_tags:
            payload["includeTags"] = include_tags
        if exclude_tags:
            payload["excludeTags"] = exclude_tags
        if wait_for:
            payload["waitFor"] = wait_for
        if timeout:
            payload["timeout"] = timeout

        # Add actions
        if actions:
            payload["actions"] = [action.model_dump() for action in actions]

        # Add FIRE-1 agent configuration
        if agent:
            payload["agent"] = agent.model_dump()

        # Add location settings
        if location:
            payload["location"] = location.model_dump()

        # Add JSON extraction options
        if json_schema or json_prompt:
            json_options = {}
            if json_schema:
                json_options["schema"] = json_schema
            if json_prompt:
                json_options["prompt"] = json_prompt
            payload["jsonOptions"] = json_options

            # Ensure json format is included
            if "json" not in payload["formats"]:
                payload["formats"].append("json")

        # Make API request
        response_data = await self._make_request("POST", "/scrape", data=payload)

        # Parse response
        document = None
        if response_data.get("success") and response_data.get("data"):
            data = response_data["data"]
            document = FirecrawlDocument(
                markdown=data.get("markdown"),
                html=data.get("html"),
                raw_html=data.get("rawHtml"),
                links=data.get("links"),
                screenshot=data.get("screenshot"),
                json_data=data.get("json"),
                metadata=FirecrawlMetadata(**data.get("metadata", {}))
                if data.get("metadata")
                else None,
            )

        return FirecrawlScrapeResponse(
            success=response_data.get("success", False),
            data=document,
            error=response_data.get("error"),
        )

    async def crawl(
        self,
        url: str,
        limit: int | None = None,
        max_depth: int | None = None,
        formats: list[str] | None = None,
        include_paths: list[str] | None = None,
        exclude_paths: list[str] | None = None,
        allow_subdomains: bool = False,
        crawl_entire_domain: bool = False,
        only_main_content: bool | None = None,
        max_age: int | None = None,
        webhook_url: str | None = None,
    ) -> FirecrawlCrawlResponse:
        """Start a crawl job for a website.

        Args:
            url: Starting URL for crawling.
            limit: Maximum number of pages to crawl.
            max_depth: Maximum crawl depth.
            formats: Output formats for each page.
            include_paths: Path patterns to include.
            exclude_paths: Path patterns to exclude.
            allow_subdomains: Whether to crawl subdomains.
            crawl_entire_domain: Whether to crawl entire domain.
            only_main_content: Extract only main content.
            max_age: Use cached data if available and younger than this (ms).
            webhook_url: URL to receive crawl progress webhooks.

        Returns:
            FirecrawlCrawlResponse with crawl job details.

        """
        # Apply configuration defaults
        limit = limit or self.default_crawl_limit
        formats = formats or self.default_formats
        only_main_content = (
            only_main_content
            if only_main_content is not None
            else self.default_only_main_content
        )

        # Build request payload
        payload = {
            "url": url,
            "limit": limit,
            "allowSubdomains": allow_subdomains,
            "crawlEntireDomain": crawl_entire_domain,
        }

        # Add optional parameters
        if max_depth is not None:
            payload["maxDepth"] = max_depth
        if include_paths:
            payload["includePaths"] = include_paths
        if exclude_paths:
            payload["excludePaths"] = exclude_paths
        if webhook_url:
            payload["webhook"] = {"url": webhook_url}

        # Configure scrape options
        scrape_options = {
            "formats": formats,
            "onlyMainContent": only_main_content,
        }
        if max_age is not None:
            scrape_options["maxAge"] = max_age

        payload["scrapeOptions"] = scrape_options

        # Make API request
        response_data = await self._make_request("POST", "/crawl", data=payload)

        return FirecrawlCrawlResponse(
            success=response_data.get("success", False),
            id=response_data.get("id"),
            url=response_data.get("url"),
            error=response_data.get("error"),
        )

    async def check_crawl_status(self, crawl_id: str) -> FirecrawlCrawlResponse:
        """Check the status of a crawl job.

        Args:
            crawl_id: Crawl job ID.

        Returns:
            FirecrawlCrawlResponse with current status and data.

        """
        response_data = await self._make_request("GET", f"/crawl/{crawl_id}")

        # Parse documents if available
        documents = None
        if response_data.get("data"):
            documents = []
            for item in response_data["data"]:
                document = FirecrawlDocument(
                    markdown=item.get("markdown"),
                    html=item.get("html"),
                    raw_html=item.get("rawHtml"),
                    links=item.get("links"),
                    screenshot=item.get("screenshot"),
                    json_data=item.get("json"),
                    metadata=FirecrawlMetadata(**item.get("metadata", {}))
                    if item.get("metadata")
                    else None,
                )
                documents.append(document)

        return FirecrawlCrawlResponse(
            success=response_data.get("success", False),
            id=crawl_id,
            status=response_data.get("status"),
            total=response_data.get("total"),
            completed=response_data.get("completed"),
            credits_used=response_data.get("creditsUsed"),
            expires_at=response_data.get("expiresAt"),
            next_url=response_data.get("next"),
            data=documents,
            error=response_data.get("error"),
        )

    async def wait_for_crawl_completion(
        self,
        crawl_id: str,
        max_wait_time: int | None = None,
        poll_interval: int | None = None,
    ) -> FirecrawlCrawlResponse:
        """Wait for a crawl job to complete.

        Args:
            crawl_id: Crawl job ID.
            max_wait_time: Maximum time to wait in seconds.
            poll_interval: Time between status checks in seconds.

        Returns:
            Completed FirecrawlCrawlResponse.

        """
        # Apply configuration defaults
        max_wait_time = max_wait_time or self.default_max_wait_time
        poll_interval = poll_interval or self.default_poll_interval

        start_time = asyncio.get_event_loop().time()

        while True:
            crawl_status = await self.check_crawl_status(crawl_id)

            if crawl_status.status in ["completed", "failed"]:
                return crawl_status

            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed >= max_wait_time:
                raise Exception(
                    f"Crawl {crawl_id} timed out after {max_wait_time} seconds"
                )

            await asyncio.sleep(poll_interval)

    async def crawl_and_wait(
        self,
        url: str,
        limit: int = 100,
        max_depth: int | None = None,
        formats: list[str] | None = None,
        include_paths: list[str] | None = None,
        exclude_paths: list[str] | None = None,
        allow_subdomains: bool = False,
        crawl_entire_domain: bool = False,
        only_main_content: bool = True,
        max_age: int | None = None,
        max_wait_time: int = 600,
        poll_interval: int = 10,
    ) -> FirecrawlCrawlResponse:
        """Start a crawl and wait for completion.

        Args:
            url: Starting URL for crawling.
            limit: Maximum number of pages to crawl.
            max_depth: Maximum crawl depth.
            formats: Output formats for each page.
            include_paths: Path patterns to include.
            exclude_paths: Path patterns to exclude.
            allow_subdomains: Whether to crawl subdomains.
            crawl_entire_domain: Whether to crawl entire domain.
            only_main_content: Extract only main content.
            max_age: Use cached data if available and younger than this (ms).
            max_wait_time: Maximum time to wait in seconds.
            poll_interval: Time between status checks in seconds.

        Returns:
            Completed FirecrawlCrawlResponse with all data.

        """
        # Start crawl
        crawl_response = await self.crawl(
            url=url,
            limit=limit,
            max_depth=max_depth,
            formats=formats,
            include_paths=include_paths,
            exclude_paths=exclude_paths,
            allow_subdomains=allow_subdomains,
            crawl_entire_domain=crawl_entire_domain,
            only_main_content=only_main_content,
            max_age=max_age,
        )

        if not crawl_response.success or not crawl_response.id:
            return crawl_response

        # Wait for completion
        return await self.wait_for_crawl_completion(
            crawl_response.id,
            max_wait_time=max_wait_time,
            poll_interval=poll_interval,
        )

    async def map_website(
        self,
        url: str,
        ignore_sitemap: bool = False,
        include_subdomains: bool = False,
        limit: int | None = None,
    ) -> FirecrawlMapResponse:
        """Map a website to discover URLs without scraping content.

        Args:
            url: Starting URL for mapping.
            ignore_sitemap: Whether to ignore sitemap.xml.
            include_subdomains: Whether to include subdomains.
            limit: Maximum number of URLs to discover.

        Returns:
            FirecrawlMapResponse with discovered URLs.

        """
        # Build request payload
        payload = {
            "url": url,
            "ignoreSitemap": ignore_sitemap,
            "includeSubdomains": include_subdomains,
        }

        if limit is not None:
            payload["limit"] = limit

        # Make API request
        response_data = await self._make_request("POST", "/map", data=payload)

        return FirecrawlMapResponse(
            success=response_data.get("success", False),
            links=response_data.get("links"),
            error=response_data.get("error"),
        )

    async def batch_scrape(
        self,
        urls: list[str],
        formats: list[str] | None = None,
        only_main_content: bool = True,
        webhook_url: str | None = None,
    ) -> FirecrawlCrawlResponse:
        """Start a batch scrape job for multiple URLs.

        Args:
            urls: List of URLs to scrape.
            formats: Output formats for each page.
            only_main_content: Extract only main content.
            webhook_url: URL to receive scrape progress webhooks.

        Returns:
            FirecrawlCrawlResponse with batch job details.

        """
        # Build request payload
        payload = {
            "urls": urls,
        }

        if webhook_url:
            payload["webhook"] = {"url": webhook_url}

        # Configure scrape options
        scrape_options = {
            "formats": formats or ["markdown"],
            "onlyMainContent": only_main_content,
        }
        payload["scrapeOptions"] = scrape_options

        # Make API request
        response_data = await self._make_request("POST", "/batch/scrape", data=payload)

        return FirecrawlCrawlResponse(
            success=response_data.get("success", False),
            id=response_data.get("id"),
            url=response_data.get("url"),
            error=response_data.get("error"),
        )

    async def batch_scrape_and_wait(
        self,
        urls: list[str],
        formats: list[str] | None = None,
        only_main_content: bool = True,
        max_wait_time: int = 600,
        poll_interval: int = 10,
    ) -> FirecrawlCrawlResponse:
        """Start batch scrape and wait for completion.

        Args:
            urls: List of URLs to scrape.
            formats: Output formats for each page.
            only_main_content: Extract only main content.
            max_wait_time: Maximum time to wait in seconds.
            poll_interval: Time between status checks in seconds.

        Returns:
            Completed FirecrawlCrawlResponse with all scraped data.

        """
        # Start batch scrape
        batch_response = await self.batch_scrape(
            urls=urls,
            formats=formats,
            only_main_content=only_main_content,
        )

        if not batch_response.success or not batch_response.id:
            return batch_response

        # Wait for completion
        return await self.wait_for_crawl_completion(
            batch_response.id,
            max_wait_time=max_wait_time,
            poll_interval=poll_interval,
        )

    async def cancel_crawl(self, crawl_id: str) -> dict[str, Any]:
        """Cancel a running crawl job.

        Args:
            crawl_id: Crawl job ID to cancel.

        Returns:
            Cancellation response.

        """
        return await self._make_request("DELETE", f"/crawl/{crawl_id}")

    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
