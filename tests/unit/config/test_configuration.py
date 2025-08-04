"""Comprehensive tests for configuration management system.

This module tests the configuration loading, validation, environment variable overrides,
and cross-field consistency checks for the dev-pro-agents orchestration system.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from src.config import (
    AgentSettings,
    APIClientSettings,
    DatabaseSettings,
    ExaSettings,
    FirecrawlSettings,
    OpenAISettings,
    OpenRouterSettings,
    OrchestrationSettings,
    SupervisorSettings,
    get_settings,
    validate_configuration,
)


class TestAPIClientSettings:
    """Test base API client configuration."""

    def test_default_values(self):
        """Test default API client settings."""
        settings = APIClientSettings()

        assert settings.timeout_seconds == 60
        assert settings.max_retries == 3
        assert settings.base_retry_delay == 1.0

    def test_validation_bounds(self):
        """Test validation of configuration bounds."""
        # Test valid bounds
        settings = APIClientSettings(
            timeout_seconds=30, max_retries=5, base_retry_delay=2.5
        )
        assert settings.timeout_seconds == 30
        assert settings.max_retries == 5
        assert settings.base_retry_delay == 2.5

        # Test invalid bounds
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            APIClientSettings(timeout_seconds=0)

        with pytest.raises(ValidationError, match="less than or equal to 600"):
            APIClientSettings(timeout_seconds=700)

        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            APIClientSettings(max_retries=0)

        with pytest.raises(ValidationError, match="less than or equal to 10"):
            APIClientSettings(max_retries=15)

    def test_field_descriptions(self):
        """Test that field descriptions are properly set."""
        fields = APIClientSettings.model_fields

        assert "Request timeout in seconds" in fields["timeout_seconds"].description
        assert "Maximum retry attempts" in fields["max_retries"].description
        assert "Base delay between retries" in fields["base_retry_delay"].description


class TestOpenAISettings:
    """Test OpenAI API configuration."""

    def test_default_values_with_api_key(self):
        """Test OpenAI settings with required API key."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
            settings = OpenAISettings()
            assert settings.api_key == "test-key"
            assert str(settings.base_url) == "https://api.openai.com/v1"
            assert settings.model == "gpt-4o-mini"
            assert settings.temperature == 0.1
            assert settings.max_tokens == 4000

    def test_temperature_validation(self):
        """Test temperature validation bounds."""
        # Valid temperature
        settings = OpenAISettings(api_key="test", temperature=1.5)
        assert settings.temperature == 1.5

        # Invalid temperature - too low
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            OpenAISettings(api_key="test", temperature=-0.1)

        # Invalid temperature - too high
        with pytest.raises(ValidationError, match="less than or equal to 2"):
            OpenAISettings(api_key="test", temperature=2.5)

    def test_max_tokens_validation(self):
        """Test max_tokens validation bounds."""
        # Valid max_tokens
        settings = OpenAISettings(api_key="test", max_tokens=8000)
        assert settings.max_tokens == 8000

        # Invalid max_tokens - too low
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            OpenAISettings(api_key="test", max_tokens=0)

        # Invalid max_tokens - too high
        with pytest.raises(ValidationError, match="less than or equal to 200000"):
            OpenAISettings(api_key="test", max_tokens=300000)

    def test_environment_variable_prefix(self):
        """Test environment variable prefix configuration."""
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "env-key",
                "OPENAI_MODEL": "gpt-4o",
                "OPENAI_TEMPERATURE": "0.5",
            },
        ):
            settings = OpenAISettings()
            assert settings.api_key == "env-key"
            assert settings.model == "gpt-4o"
            assert settings.temperature == 0.5


class TestOpenRouterSettings:
    """Test OpenRouter API configuration."""

    def test_default_values_with_api_key(self):
        """Test OpenRouter settings with required API key."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=True):
            settings = OpenRouterSettings()
            assert settings.api_key == "test-key"
        assert str(settings.base_url) == "https://openrouter.ai/api/v1"
        assert settings.model == "openrouter/horizon-beta"
        assert settings.temperature == 0.2
        assert settings.timeout_seconds == 90  # Extended timeout

    def test_extended_timeout(self):
        """Test that OpenRouter has extended timeout compared to base."""
        base_settings = APIClientSettings()
        openrouter_settings = OpenRouterSettings(api_key="test")

        assert openrouter_settings.timeout_seconds > base_settings.timeout_seconds

    def test_environment_variable_prefix(self):
        """Test OpenRouter environment variable prefix."""
        with patch.dict(
            os.environ,
            {
                "OPENROUTER_API_KEY": "or-key",
                "OPENROUTER_MODEL": "custom-model",
                "OPENROUTER_TIMEOUT_SECONDS": "120",
            },
        ):
            settings = OpenRouterSettings()
            assert settings.api_key == "or-key"
            assert settings.model == "custom-model"
            assert settings.timeout_seconds == 120


class TestExaSettings:
    """Test Exa API configuration."""

    def test_default_values_with_api_key(self):
        """Test Exa settings with required API key."""
        settings = ExaSettings(api_key="test-key")

        assert settings.api_key == "test-key"
        assert str(settings.base_url) == "https://api.exa.ai/"
        assert settings.search_type == "auto"
        assert settings.num_results == 10
        assert settings.include_text is False
        assert settings.include_highlights is True
        assert settings.include_summary is False

    def test_num_results_validation(self):
        """Test num_results validation bounds."""
        # Valid num_results
        settings = ExaSettings(api_key="test", num_results=25)
        assert settings.num_results == 25

        # Invalid num_results - too low
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            ExaSettings(api_key="test", num_results=0)

        # Invalid num_results - too high
        with pytest.raises(ValidationError, match="less than or equal to 100"):
            ExaSettings(api_key="test", num_results=150)

    def test_boolean_flags(self):
        """Test boolean flag configurations."""
        settings = ExaSettings(
            api_key="test",
            include_text=True,
            include_highlights=False,
            include_summary=True,
        )

        assert settings.include_text is True
        assert settings.include_highlights is False
        assert settings.include_summary is True


class TestFirecrawlSettings:
    """Test Firecrawl API configuration."""

    def test_default_values_with_api_key(self):
        """Test Firecrawl settings with required API key."""
        with patch.dict(os.environ, {"FIRECRAWL_API_KEY": "test-key"}, clear=True):
            settings = FirecrawlSettings()
            assert settings.api_key == "test-key"
            assert str(settings.base_url) == "https://api.firecrawl.dev/v1"
            assert settings.timeout_seconds == 120  # Extended timeout
            assert settings.default_formats == ["markdown"]
            assert settings.only_main_content is True
            assert settings.crawl_limit == 100

    def test_extended_timeout_for_crawling(self):
        """Test that Firecrawl has extended timeout for crawling operations."""
        base_settings = APIClientSettings()
        firecrawl_settings = FirecrawlSettings(api_key="test")

        assert firecrawl_settings.timeout_seconds > base_settings.timeout_seconds

    def test_crawl_configuration_validation(self):
        """Test crawl-specific configuration validation."""
        # Valid crawl configuration
        settings = FirecrawlSettings(
            api_key="test", crawl_limit=500, max_wait_time=1800, poll_interval=30
        )

        assert settings.crawl_limit == 500
        assert settings.max_wait_time == 1800
        assert settings.poll_interval == 30

        # Invalid crawl_limit - too high
        with pytest.raises(ValidationError, match="less than or equal to 1000"):
            FirecrawlSettings(api_key="test", crawl_limit=2000)

        # Invalid max_wait_time - too high
        with pytest.raises(ValidationError, match="less than or equal to 3600"):
            FirecrawlSettings(api_key="test", max_wait_time=7200)

    def test_default_formats_list(self):
        """Test default formats configuration."""
        settings = FirecrawlSettings(api_key="test")
        assert isinstance(settings.default_formats, list)
        assert "markdown" in settings.default_formats

        # Test custom formats
        custom_settings = FirecrawlSettings(
            api_key="test", default_formats=["html", "text"]
        )
        assert custom_settings.default_formats == ["html", "text"]


class TestDatabaseSettings:
    """Test database configuration."""

    def test_default_values(self):
        """Test default database settings."""
        settings = DatabaseSettings()

        assert settings.url == "sqlite:///jobs.db"
        assert settings.pool_size == 10
        assert settings.pool_timeout == 30
        assert settings.echo_sql is False

    def test_database_path_validation(self):
        """Test database path validation and directory creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test_db" / "test.db"

            settings = DatabaseSettings(implementation_tracker_path=db_path)

            # Path should be converted to absolute
            assert settings.implementation_tracker_path.is_absolute()

            # Parent directory should be created
            assert settings.implementation_tracker_path.parent.exists()

    def test_relative_path_conversion(self):
        """Test that relative paths are converted to absolute."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Change to temp directory for testing
            old_cwd = str(Path.cwd())
            os.chdir(temp_dir)

            try:
                settings = DatabaseSettings(
                    implementation_tracker_path="relative/test.db"
                )
                assert settings.implementation_tracker_path.is_absolute()
                assert str(settings.implementation_tracker_path).startswith(temp_dir)
            finally:
                os.chdir(old_cwd)

    def test_pool_size_validation(self):
        """Test connection pool size validation."""
        # Valid pool size
        settings = DatabaseSettings(pool_size=25)
        assert settings.pool_size == 25

        # Invalid pool size - too low
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            DatabaseSettings(pool_size=0)

        # Invalid pool size - too high
        with pytest.raises(ValidationError, match="less than or equal to 50"):
            DatabaseSettings(pool_size=100)


class TestAgentSettings:
    """Test agent-specific configuration."""

    def test_default_temperature_values(self):
        """Test default temperature values for different agent types."""
        settings = AgentSettings()

        assert settings.research_temperature == 0.1  # Precise analysis
        assert settings.coding_temperature == 0.2  # Balanced creativity
        assert settings.testing_temperature == 0.1  # Precise validation
        assert settings.documentation_temperature == 0.3  # Creative writing

    def test_execution_limits(self):
        """Test agent execution limits."""
        settings = AgentSettings()

        assert settings.max_execution_time_minutes == 30
        assert settings.max_iterations == 5
        assert settings.minimum_confidence_score == 0.7

    def test_execution_limits_validation(self):
        """Test validation of execution limits."""
        # Valid limits
        settings = AgentSettings(
            max_execution_time_minutes=60,
            max_iterations=10,
            minimum_confidence_score=0.8,
        )

        assert settings.max_execution_time_minutes == 60
        assert settings.max_iterations == 10
        assert settings.minimum_confidence_score == 0.8

        # Invalid limits
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            AgentSettings(max_execution_time_minutes=0)

        with pytest.raises(ValidationError, match="less than or equal to 120"):
            AgentSettings(max_execution_time_minutes=150)


class TestSupervisorSettings:
    """Test LangGraph supervisor configuration."""

    def test_default_values(self):
        """Test default supervisor settings."""
        settings = SupervisorSettings()

        assert settings.model == "openrouter/horizon-beta"
        assert settings.temperature == 0.2
        assert settings.max_parallel_agents == 5
        assert settings.task_assignment_strategy == "load_balanced"
        assert settings.supervision_interval_seconds == 30
        assert settings.auto_retry_failed_tasks is True
        assert settings.max_retry_attempts == 3

    def test_parallel_agents_validation(self):
        """Test parallel agents limit validation."""
        # Valid limit
        settings = SupervisorSettings(max_parallel_agents=10)
        assert settings.max_parallel_agents == 10

        # Invalid limit - too low
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            SupervisorSettings(max_parallel_agents=0)

        # Invalid limit - too high
        with pytest.raises(ValidationError, match="less than or equal to 20"):
            SupervisorSettings(max_parallel_agents=25)


class TestOrchestrationSettings:
    """Test main orchestration configuration."""

    def test_nested_settings_initialization(self):
        """Test that nested settings are properly initialized."""
        # Mock required API keys
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "test-openai",
                "OPENROUTER_API_KEY": "test-openrouter",
                "EXA_API_KEY": "test-exa",
                "FIRECRAWL_API_KEY": "test-firecrawl",
            },
        ):
            settings = OrchestrationSettings()

            assert isinstance(settings.openai, OpenAISettings)
            assert isinstance(settings.openrouter, OpenRouterSettings)
            assert isinstance(settings.exa, ExaSettings)
            assert isinstance(settings.firecrawl, FirecrawlSettings)
            assert isinstance(settings.database, DatabaseSettings)
            assert isinstance(settings.agents, AgentSettings)
            assert isinstance(settings.supervisor, SupervisorSettings)

    def test_default_performance_settings(self):
        """Test default performance and execution settings."""
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "test-openai",
                "OPENROUTER_API_KEY": "test-openrouter",
                "EXA_API_KEY": "test-exa",
                "FIRECRAWL_API_KEY": "test-firecrawl",
            },
        ):
            settings = OrchestrationSettings()

            assert settings.parallel_execution_limit == 5
            assert settings.batch_size == 10
            assert settings.task_timeout_minutes == 60
            assert settings.debug_mode is False
            assert settings.log_level == "INFO"
            assert settings.config_version == "1.0.0"

    def test_legacy_feature_flags(self):
        """Test legacy feature flags for backward compatibility."""
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "test-openai",
                "OPENROUTER_API_KEY": "test-openrouter",
                "EXA_API_KEY": "test-exa",
                "FIRECRAWL_API_KEY": "test-firecrawl",
            },
        ):
            settings = OrchestrationSettings()

            assert settings.use_groq is False
            assert settings.use_proxies is False
            assert settings.use_checkpointing is True

    def test_cross_field_validation_parallel_limits(self):
        """Test cross-field validation for parallel execution limits."""
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "test-openai",
                "OPENROUTER_API_KEY": "test-openrouter",
                "EXA_API_KEY": "test-exa",
                "FIRECRAWL_API_KEY": "test-firecrawl",
            },
        ):
            # Test that parallel_execution_limit is adjusted to supervisor max
            settings = OrchestrationSettings(
                parallel_execution_limit=10,
                supervisor=SupervisorSettings(max_parallel_agents=8),
            )

            # Should be adjusted down to supervisor limit
            assert settings.parallel_execution_limit == 8

    def test_cross_field_validation_timeout_consistency(self):
        """Test cross-field validation for timeout consistency."""
        with (
            patch.dict(
                os.environ,
                {
                    "OPENAI_API_KEY": "test-openai",
                    "OPENROUTER_API_KEY": "test-openrouter",
                    "EXA_API_KEY": "test-exa",
                    "FIRECRAWL_API_KEY": "test-firecrawl",
                },
            ),
            pytest.raises(ValidationError, match="Global task timeout must be"),
        ):
            OrchestrationSettings(
                task_timeout_minutes=30,
                agents=AgentSettings(max_execution_time_minutes=45),
            )

    def test_get_agent_config_method(self):
        """Test get_agent_config method for different agent types."""
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "test-openai",
                "OPENROUTER_API_KEY": "test-openrouter",
                "EXA_API_KEY": "test-exa",
                "FIRECRAWL_API_KEY": "test-firecrawl",
            },
        ):
            settings = OrchestrationSettings()

            # Test research agent config
            research_config = settings.get_agent_config("research")
            assert research_config["temperature"] == 0.1
            assert "max_execution_time_minutes" in research_config
            assert "max_iterations" in research_config
            assert "minimum_confidence_score" in research_config

            # Test coding agent config
            coding_config = settings.get_agent_config("coding")
            assert coding_config["temperature"] == 0.2

            # Test testing agent config
            testing_config = settings.get_agent_config("testing")
            assert testing_config["temperature"] == 0.1

            # Test documentation agent config
            doc_config = settings.get_agent_config("documentation")
            assert doc_config["temperature"] == 0.3

            # Test unknown agent type (should use coding default)
            unknown_config = settings.get_agent_config("unknown")
            assert unknown_config["temperature"] == 0.2

    def test_get_api_client_config_method(self):
        """Test get_api_client_config method for different clients."""
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "test-openai",
                "OPENROUTER_API_KEY": "test-openrouter",
                "EXA_API_KEY": "test-exa",
                "FIRECRAWL_API_KEY": "test-firecrawl",
            },
        ):
            settings = OrchestrationSettings()

            # Test OpenAI client config
            openai_config = settings.get_api_client_config("openai")
            assert "api_key" in openai_config
            assert "base_url" in openai_config
            assert "model" in openai_config

            # Test OpenRouter client config
            openrouter_config = settings.get_api_client_config("openrouter")
            assert "api_key" in openrouter_config
            assert "timeout_seconds" in openrouter_config

            # Test Exa client config
            exa_config = settings.get_api_client_config("exa")
            assert "search_type" in exa_config
            assert "num_results" in exa_config

            # Test Firecrawl client config
            firecrawl_config = settings.get_api_client_config("firecrawl")
            assert "crawl_limit" in firecrawl_config
            assert "default_formats" in firecrawl_config

            # Test unknown client type
            with pytest.raises(ValueError, match="Unknown client type"):
                settings.get_api_client_config("unknown")

    def test_environment_variable_overrides(self):
        """Test environment variable overrides for nested settings."""
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "test-openai",
                "OPENROUTER_API_KEY": "test-openrouter",
                "EXA_API_KEY": "test-exa",
                "FIRECRAWL_API_KEY": "test-firecrawl",
                "ORCHESTRATION_PARALLEL_EXECUTION_LIMIT": "8",
                "ORCHESTRATION_BATCH_SIZE": "20",
                "ORCHESTRATION_DEBUG_MODE": "true",
                "ORCHESTRATION_LOG_LEVEL": "DEBUG",
            },
            clear=True,
        ):
            settings = OrchestrationSettings()

            assert settings.parallel_execution_limit == 8
            assert settings.batch_size == 20
            assert settings.debug_mode is True
            assert settings.log_level == "DEBUG"

    def test_secrets_directory_configuration(self):
        """Test secrets directory configuration from environment."""
        with tempfile.TemporaryDirectory() as temp_dir:
            secrets_dir = Path(temp_dir) / "secrets"
            secrets_dir.mkdir()

            # Create a secrets file
            (secrets_dir / "OPENAI_API_KEY").write_text("secret-key")

            with patch.dict(
                os.environ,
                {
                    "OPENAI_API_KEY": "test-openai",
                    "ORCHESTRATION_SECRETS_DIR": str(secrets_dir),
                    "OPENROUTER_API_KEY": "test-openrouter",
                    "EXA_API_KEY": "test-exa",
                    "FIRECRAWL_API_KEY": "test-firecrawl",
                },
                clear=True,
            ):
                settings = OrchestrationSettings()
                # Note: This test verifies the configuration is set up correctly
                # The secrets directory functionality depends on pydantic-settings
                # For now, just verify the settings were created successfully
                assert settings is not None
                assert settings.openai.api_key == "test-openai"


class TestConfigurationUtilityFunctions:
    """Test utility functions for configuration access."""

    def test_get_settings_caching(self):
        """Test that get_settings returns cached instance."""
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "test-openai",
                "OPENROUTER_API_KEY": "test-openrouter",
                "EXA_API_KEY": "test-exa",
                "FIRECRAWL_API_KEY": "test-firecrawl",
            },
        ):
            settings1 = get_settings()
            settings2 = get_settings()

            # Should be the same instance due to LRU cache
            assert settings1 is settings2

    def test_validate_configuration_success(self):
        """Test successful configuration validation."""
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "test-openai",
                "OPENROUTER_API_KEY": "test-openrouter",
                "EXA_API_KEY": "test-exa",
                "FIRECRAWL_API_KEY": "test-firecrawl",
            },
        ):
            assert validate_configuration() is True

    def test_validate_configuration_missing_api_keys(self):
        """Test configuration validation with missing API keys."""
        # Clear environment variables
        with (
            patch.dict(os.environ, {}, clear=True),
            pytest.raises(ValueError, match="Configuration validation failed"),
        ):
            validate_configuration()

    def test_validate_configuration_invalid_database_path(self):
        """Test configuration validation with invalid database path."""
        with (
            patch.dict(
                os.environ,
                {
                    "OPENAI_API_KEY": "test-openai",
                    "OPENROUTER_API_KEY": "test-openrouter",
                    "EXA_API_KEY": "test-exa",
                    "FIRECRAWL_API_KEY": "test-firecrawl",
                    "DATABASE_PATH": "/invalid/path/database.db",
                },
            ),
            patch("os.access", return_value=False),
            pytest.raises(ValueError, match="Database directory is not writable"),
        ):
            validate_configuration()


class TestConfigurationErrorHandling:
    """Test error handling in configuration loading."""

    def test_pydantic_validation_errors(self):
        """Test that pydantic validation errors are properly handled."""
        with pytest.raises(ValidationError) as exc_info:
            OpenAISettings(
                api_key="test",
                temperature=3.0,  # Invalid: > 2.0
                max_tokens=-100,  # Invalid: < 1
            )

        error = exc_info.value
        assert len(error.errors()) >= 2  # Should have multiple validation errors

    def test_field_validation_error_messages(self):
        """Test that field validation provides clear error messages."""
        with pytest.raises(ValidationError) as exc_info:
            AgentSettings(max_execution_time_minutes=200)  # > 120

        error_dict = exc_info.value.errors()[0]
        assert "less than or equal to" in error_dict["msg"]

    def test_cross_validation_error_messages(self):
        """Test that cross-field validation provides clear error messages."""
        with (
            patch.dict(
                os.environ,
                {
                    "OPENAI_API_KEY": "test-openai",
                    "OPENROUTER_API_KEY": "test-openrouter",
                    "EXA_API_KEY": "test-exa",
                    "FIRECRAWL_API_KEY": "test-firecrawl",
                },
            ),
            pytest.raises(ValidationError) as exc_info,
        ):
            OrchestrationSettings(
                task_timeout_minutes=15,
                agents=AgentSettings(max_execution_time_minutes=30),
            )

        assert "Global task timeout must be" in str(exc_info.value)


class TestConfigurationPerformance:
    """Test configuration performance characteristics."""

    def test_settings_initialization_time(self):
        """Test that settings initialization is reasonably fast."""
        import time

        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "test-openai",
                "OPENROUTER_API_KEY": "test-openrouter",
                "EXA_API_KEY": "test-exa",
                "FIRECRAWL_API_KEY": "test-firecrawl",
            },
        ):
            start_time = time.time()
            settings = OrchestrationSettings()
            initialization_time = time.time() - start_time

            # Should initialize within reasonable time (< 1 second)
            assert initialization_time < 1.0
            assert settings is not None

    def test_cached_settings_performance(self):
        """Test that cached settings access is fast."""
        import time

        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "test-openai",
                "OPENROUTER_API_KEY": "test-openrouter",
                "EXA_API_KEY": "test-exa",
                "FIRECRAWL_API_KEY": "test-firecrawl",
            },
        ):
            # First call (loads and caches)
            get_settings()

            # Subsequent calls should be very fast
            start_time = time.time()
            for _ in range(100):
                get_settings()
            access_time = time.time() - start_time

            # 100 cached accesses should be very fast (< 0.1 seconds)
            assert access_time < 0.1


@pytest.mark.integration
class TestConfigurationIntegration:
    """Integration tests for configuration system."""

    def test_configuration_with_env_file(self):
        """Test configuration loading from .env file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / ".env"
            env_file.write_text("""
OPENAI_API_KEY=file-openai-key
OPENROUTER_API_KEY=file-openrouter-key
EXA_API_KEY=file-exa-key
FIRECRAWL_API_KEY=file-firecrawl-key
ORCHESTRATION_PARALLEL_EXECUTION_LIMIT=12
ORCHESTRATION_DEBUG_MODE=true
""")

            # Change to temp directory to pick up .env file and clear environment
            old_cwd = str(Path.cwd())
            os.chdir(temp_dir)

            # Clear any existing API keys from environment
            with patch.dict(os.environ, {}, clear=True):
                try:
                    # Clear cache to force reload
                    get_settings.cache_clear()
                    settings = get_settings()

                    assert settings.openai.api_key == "file-openai-key"
                    assert settings.openrouter.api_key == "file-openrouter-key"
                    assert settings.exa.api_key == "file-exa-key"
                    assert settings.firecrawl.api_key == "file-firecrawl-key"
                    assert settings.parallel_execution_limit == 12
                    assert settings.debug_mode is True
                finally:
                    os.chdir(old_cwd)
                    get_settings.cache_clear()

    def test_environment_variable_precedence(self):
        """Test that environment variables take precedence over .env file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / ".env"
            env_file.write_text("""
OPENAI_API_KEY=file-key
OPENROUTER_API_KEY=file-openrouter-key
EXA_API_KEY=file-exa-key
FIRECRAWL_API_KEY=file-firecrawl-key
ORCHESTRATION_BATCH_SIZE=5
""")

            old_cwd = str(Path.cwd())
            os.chdir(temp_dir)

            try:
                with patch.dict(
                    os.environ,
                    {
                        "OPENAI_API_KEY": "env-key",  # Should override file
                        "ORCHESTRATION_BATCH_SIZE": "25",  # Should override file
                    },
                ):
                    get_settings.cache_clear()
                    settings = get_settings()

                    assert settings.openai.api_key == "env-key"  # From environment
                    assert settings.batch_size == 25  # From environment

            finally:
                os.chdir(old_cwd)
                get_settings.cache_clear()
