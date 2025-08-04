# Exa and Firecrawl API Integration for LangGraph Agents

This module provides clean, async-compatible Python API wrappers for Exa and Firecrawl APIs, specifically designed for integration with LangGraph agents and multi-agent workflows.

## Features

- **Direct API Access**: No MCP server dependencies

- **Full Async/Await Support**: Compatible with LangGraph async workflows

- **Comprehensive Error Handling**: Robust retry logic and error management

- **Agent-Friendly Interfaces**: Pydantic models for structured responses

- **Cost Tracking**: Built-in cost monitoring for Exa API

- **Authentication Management**: Environment variable support

## Quick Start

### Installation

The required dependencies are already included in the project's `pyproject.toml`:

```bash

# Install project dependencies
uv install
```

### Environment Setup

Create a `.env` file in your project root:

```bash

# Required API keys
EXA_API_KEY=your_exa_api_key_here
FIRECRAWL_API_KEY=your_firecrawl_api_key_here
```

## API Clients Overview

### Exa Client Features

- **Neural Search**: Semantic search using embeddings

- **Keyword Search**: Traditional Google-style search

- **Research API**: Automated research tasks with structured output

- **Content Extraction**: Get full content from URLs

- **Similar Search**: Find pages similar to a given URL

- **Answer API**: Direct question answering with citations

### Firecrawl Client Features

- **Single URL Scraping**: Extract content from individual pages

- **Website Crawling**: Recursive crawling with depth control

- **Batch Operations**: Scrape multiple URLs simultaneously

- **Website Mapping**: Discover URL structure without content extraction

- **Structured Extraction**: JSON extraction with Pydantic schemas

- **Dynamic Content**: Handle JavaScript-rendered pages

- **Location Support**: Geographic targeting for content

## Basic Usage Examples

### Exa Client

```python
import asyncio
from src.integrations import ExaClient

async def basic_exa_example():
    async with ExaClient() as client:
        # Semantic search
        results = await client.search(
            query="LangGraph framework documentation",
            search_type="neural",
            num_results=5,
            include_text=True,
            include_highlights=True,
        )
        
        print(f"Found {len(results.results)} results")
        for result in results.results:
            print(f"- {result.title}: {result.url}")
        
        # Research task
        research = await client.research(
            instructions="Research current trends in LangGraph and agent frameworks",
            model="exa-research",
            wait_for_completion=True,
        )
        
        print(f"Research complete: {research}")

# Run example
asyncio.run(basic_exa_example())
```

### Firecrawl Client

```python
import asyncio
from src.integrations import FirecrawlClient

async def basic_firecrawl_example():
    async with FirecrawlClient() as client:
        # Single page scraping
        result = await client.scrape(
            url="https://docs.langchain.com/docs/langgraph",
            formats=["markdown", "json"],
            json_prompt="Extract key features and benefits",
        )
        
        if result.success:
            print(f"Title: {result.data.metadata.title}")
            print(f"Content length: {len(result.data.markdown or '')}")
            print(f"Extracted data: {result.data.json_data}")
        
        # Website crawling
        crawl_result = await client.crawl_and_wait(
            url="https://docs.langchain.com",
            limit=10,
            formats=["markdown"],
            include_paths=["/docs/*"],
            max_wait_time=300,
        )
        
        print(f"Crawled {len(crawl_result.data or [])} pages")

# Run example
asyncio.run(basic_firecrawl_example())
```

## LangGraph Integration

### Simple Research Agent

```python
from typing import TypedDict
from langgraph.graph import StateGraph, END, START
from src.integrations import ExaClient, FirecrawlClient

class ResearchState(TypedDict):
    query: str
    search_results: list
    scraped_content: list
    final_report: str

class ResearchAgent:
    def __init__(self):
        self.exa_client = ExaClient()
        self.firecrawl_client = FirecrawlClient()
    
    async def search_node(self, state: ResearchState) -> ResearchState:
        # Use Exa for discovery
        results = await self.exa_client.search(
            query=state["query"],
            search_type="neural",
            num_results=5,
            include_text=True,
        )
        
        return {
            **state,
            "search_results": [
                {"title": r.title, "url": r.url, "text": r.text}
                for r in results.results
            ]
        }
    
    async def extract_node(self, state: ResearchState) -> ResearchState:
        # Use Firecrawl for comprehensive extraction
        urls = [r["url"] for r in state["search_results"]]
        
        scrape_result = await self.firecrawl_client.batch_scrape_and_wait(
            urls=urls,
            formats=["markdown"],
            max_wait_time=300,
        )
        
        scraped_content = []
        if scrape_result.success and scrape_result.data:
            for doc in scrape_result.data:
                if doc.metadata:
                    scraped_content.append({
                        "url": doc.metadata.source_url,
                        "title": doc.metadata.title,
                        "content": doc.markdown,
                    })
        
        return {
            **state,
            "scraped_content": scraped_content,
        }
    
    async def synthesize_node(self, state: ResearchState) -> ResearchState:
        # Create final report
        report = f"# Research Report: {state['query']}\n\n"
        report += f"Found {len(state['search_results'])} sources\n"
        report += f"Successfully extracted {len(state['scraped_content'])} documents\n\n"
        
        for content in state["scraped_content"]:
            report += f"## {content['title']}\n"
            report += f"URL: {content['url']}\n"
            report += f"Content preview: {content['content'][:200]}...\n\n"
        
        return {
            **state,
            "final_report": report,
        }
    
    def create_workflow(self):
        workflow = StateGraph(ResearchState)
        
        workflow.add_node("search", self.search_node)
        workflow.add_node("extract", self.extract_node)
        workflow.add_node("synthesize", self.synthesize_node)
        
        workflow.add_edge(START, "search")
        workflow.add_edge("search", "extract")
        workflow.add_edge("extract", "synthesize")
        workflow.add_edge("synthesize", END)
        
        return workflow.compile()
```

## Advanced Features

### Cost-Aware Exa Usage

```python
async def cost_aware_search():
    async with ExaClient() as client:
        # Track costs across multiple searches
        total_cost = 0
        
        # Use cheaper keyword search first
        quick_results = await client.search(
            query="Python AI frameworks",
            search_type="keyword",  # Cheaper than neural
            num_results=10,
        )
        
        if quick_results.cost_dollars:
            total_cost += quick_results.cost_dollars.get("total", 0)
        
        # Use neural search for refined results
        detailed_results = await client.search(
            query="LangGraph vs LangChain comparison",
            search_type="neural",  # More expensive but better quality
            num_results=5,
            include_text=True,
        )
        
        if detailed_results.cost_dollars:
            total_cost += detailed_results.cost_dollars.get("total", 0)
        
        print(f"Total API cost: ${total_cost}")
```

### Firecrawl with Actions

```python
async def interactive_scraping():
    async with FirecrawlClient() as client:
        from src.integrations.firecrawl_client import FirecrawlAction
        
        # Scrape with page interactions
        result = await client.scrape(
            url="https://example-spa.com",
            formats=["markdown", "json"],
            actions=[
                FirecrawlAction(type="wait", milliseconds=2000),
                FirecrawlAction(type="click", selector="button.load-more"),
                FirecrawlAction(type="wait", milliseconds=3000),
            ],
            json_prompt="Extract all product listings with prices",
        )
        
        print(f"Extracted products: {result.data.json_data}")
```

## Error Handling

Both clients include comprehensive error handling:

```python
async def robust_usage():
    try:
        async with ExaClient() as exa_client, FirecrawlClient() as firecrawl_client:
            # Exa search with retry logic
            results = await exa_client.search("test query")
            
            # Firecrawl scraping with timeout handling
            scrape_result = await firecrawl_client.scrape(
                url="https://example.com",
                timeout=30000,  # 30 seconds
            )
            
    except Exception as e:
        print(f"API error: {e}")
        # Handle rate limits, timeouts, authentication errors, etc.
```

## Testing

Run the test suite:

```bash

# Run unit tests (no API keys required)
pytest tests/test_integrations.py -v

# Run integration tests (requires API keys)
pytest tests/test_integrations.py -m integration -v

# Run all tests
pytest tests/test_integrations.py --integration -v
```

## Best Practices

### For Production Use

1. **API Key Management**: Always use environment variables
2. **Cost Monitoring**: Track API usage and costs
3. **Rate Limiting**: Implement proper backoff strategies
4. **Error Handling**: Handle network failures gracefully
5. **Resource Cleanup**: Always close clients when done

### For LangGraph Integration

1. **State Management**: Use TypedDict for clear state definitions
2. **Async Operations**: Leverage async/await for concurrent operations
3. **Error Propagation**: Include error handling in your state
4. **Resource Sharing**: Reuse clients across workflow nodes
5. **Checkpointing**: Use LangGraph checkpoints for long-running operations

## Performance Optimization

### Exa Optimization

- Use keyword search for initial discovery (cheaper)

- Use neural search for refined results (better quality)

- Batch content requests when possible

- Monitor cost with `cost_dollars` field

### Firecrawl Optimization

- Use `max_age` parameter for caching (500% faster)

- Batch scrape multiple URLs together

- Use website mapping to discover URLs first

- Configure appropriate timeouts for your use case

## API Limits and Pricing

### Exa API

- Neural search: $5/1k requests (1-25 results), $25/1k (26-100 results)

- Keyword search: $2.5/1k requests

- Content retrieval: $1/1k pages

- Research API: $5-10/1k requests

### Firecrawl API

- Pricing varies by plan (check <https://firecrawl.dev/pricing>)

- Offers caching for faster repeated requests

- Supports webhooks for real-time updates

## Contributing

To contribute to this integration module:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Support

For issues with the integration:

- Check the test files for usage examples

- Review the API documentation for Exa and Firecrawl

- Open an issue in the project repository

For API-specific support:

- Exa: <https://docs.exa.ai>

- Firecrawl: <https://docs.firecrawl.dev>
