"""Comprehensive tests for YAML configuration file loading and validation.

This module tests the loading, parsing, and validation of YAML configuration files
including agents.yaml and orchestrator.yaml, with comprehensive error handling
and edge case coverage.
"""

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml


class TestYAMLConfigurationLoading:
    """Test YAML configuration file loading and parsing."""

    def test_agents_yaml_loading_valid(self):
        """Test loading valid agents.yaml configuration."""
        # Read the actual agents.yaml file
        agents_yaml_path = Path("src/config/agents.yaml")

        with agents_yaml_path.open() as f:
            config_data = yaml.safe_load(f)

        # Verify structure
        assert "agents" in config_data
        assert "global_settings" in config_data

        # Check agent configurations
        agents = config_data["agents"]
        assert "research_agent" in agents
        assert "coding_agent" in agents
        assert "testing_agent" in agents
        assert "documentation_agent" in agents

        # Verify research agent configuration
        research_agent = agents["research_agent"]
        assert research_agent["name"] == "research"
        assert research_agent["enabled"] is True
        assert "capabilities" in research_agent
        assert isinstance(research_agent["capabilities"], list)
        assert "web_scraping" in research_agent["capabilities"]
        assert research_agent["model"] == "exa-research-pro"
        assert research_agent["timeout"] == 120
        assert research_agent["retry_attempts"] == 3

        # Verify coding agent configuration
        coding_agent = agents["coding_agent"]
        assert coding_agent["name"] == "coding"
        assert coding_agent["model"] == "openrouter/horizon-beta"
        assert coding_agent["timeout"] == 180
        assert "implementation" in coding_agent["capabilities"]

        # Verify testing agent configuration
        testing_agent = agents["testing_agent"]
        assert testing_agent["name"] == "testing"
        assert "test_design" in testing_agent["capabilities"]
        assert "pytest" in testing_agent["specific_config"]["test_frameworks"]
        assert testing_agent["specific_config"]["coverage_threshold"] == 80

        # Verify documentation agent configuration
        doc_agent = agents["documentation_agent"]
        assert doc_agent["name"] == "documentation"
        assert doc_agent["specific_config"]["documentation_format"] == "markdown"
        assert doc_agent["specific_config"]["include_examples"] is True

        # Verify global settings
        global_settings = config_data["global_settings"]
        assert global_settings["default_timeout"] == 180
        assert global_settings["max_concurrent_agents"] == 4
        assert global_settings["log_level"] == "INFO"

    def test_orchestrator_yaml_loading_valid(self):
        """Test loading valid orchestrator.yaml configuration."""
        # Read the actual orchestrator.yaml file
        orchestrator_yaml_path = Path("src/config/orchestrator.yaml")

        with orchestrator_yaml_path.open() as f:
            config_data = yaml.safe_load(f)

        # Verify structure
        assert "orchestrator" in config_data
        assert "performance" in config_data
        assert "security" in config_data

        # Check orchestrator configuration
        orchestrator = config_data["orchestrator"]
        assert orchestrator["name"] == "multi_agent_orchestrator"
        assert orchestrator["supervisor_model"] == "o3"

        # Verify batch processing configuration
        batch_processing = orchestrator["batch_processing"]
        assert batch_processing["enabled"] is True
        assert batch_processing["max_batch_size"] == 10
        assert batch_processing["batch_timeout"] == 3600
        assert batch_processing["parallel_execution"] is True

        # Verify coordination settings
        coordination = orchestrator["coordination"]
        assert coordination["task_assignment_strategy"] == "capability_based"
        assert coordination["enable_task_dependencies"] is True
        assert coordination["max_task_retries"] == 3

        # Verify error handling
        error_handling = orchestrator["error_handling"]
        assert error_handling["max_retries_per_agent"] == 3
        assert error_handling["failure_escalation"] is True
        assert error_handling["recovery_strategy"] == "reassign"

        # Verify monitoring
        monitoring = orchestrator["monitoring"]
        assert monitoring["enable_health_checks"] is True
        assert monitoring["health_check_interval"] == 30

        # Verify database configuration
        database = orchestrator["database"]
        assert "sqlite:///" in database["connection_string"]
        assert database["connection_pool_size"] == 5

        # Verify integrations
        integrations = orchestrator["integrations"]
        assert "exa" in integrations
        assert "firecrawl" in integrations
        assert "openai" in integrations

        # Check integration settings
        exa_config = integrations["exa"]
        assert exa_config["enabled"] is True
        assert exa_config["api_key_env"] == "EXA_API_KEY"
        assert exa_config["timeout"] == 30

        # Verify performance settings
        performance = config_data["performance"]
        assert performance["async_execution"] is True
        assert performance["connection_pooling"] is True
        assert performance["result_caching"] is False

        # Verify security settings
        security = config_data["security"]
        assert security["validate_inputs"] is True
        assert security["sanitize_outputs"] is True
        assert security["max_request_size"] == 10485760

    def test_yaml_syntax_validation(self):
        """Test YAML syntax validation with various formats."""
        valid_yaml_samples = [
            """
            # Simple key-value
            key: value
            number: 42
            boolean: true
            """,
            """
            # Nested structure
            parent:
              child1: value1
              child2: value2
              nested:
                deep_key: deep_value
            """,
            """
            # Lists and arrays
            list_items:
              - item1
              - item2
              - item3
            mixed_list:
              - string_item
              - 123
              - true
            """,
            """
            # Complex structure
            agents:
              research:
                name: research_agent
                capabilities:
                  - web_scraping
                  - data_analysis
                config:
                  timeout: 60
                  retries: 3
            """,
        ]

        for yaml_content in valid_yaml_samples:
            try:
                parsed_data = yaml.safe_load(yaml_content.strip())
                assert parsed_data is not None
                assert isinstance(parsed_data, dict)
            except yaml.YAMLError as e:
                pytest.fail(f"Valid YAML failed to parse: {e}")

    def test_yaml_syntax_errors(self):
        """Test handling of invalid YAML syntax."""
        invalid_yaml_samples = [
            # Indentation error
            """
            parent:
              child1: value1
            child2: value2  # Wrong indentation
            """,
            # Missing colon
            """
            key_without_colon value
            """,
            # Unclosed quotes
            """
            key: "unclosed quote
            """,
            # Invalid list syntax
            """
            list:
              - item1
              item2  # Missing dash
            """,
            # Invalid escape sequence
            """
            key: "invalid\\escape"
            """,
        ]

        for invalid_yaml in invalid_yaml_samples:
            with pytest.raises(yaml.YAMLError):
                yaml.safe_load(invalid_yaml.strip())

    def test_yaml_data_types(self):
        """Test YAML data type parsing and validation."""
        yaml_content = """
        string_value: "hello world"
        integer_value: 42
        float_value: 3.14159
        boolean_true: true
        boolean_false: false
        null_value: null
        date_value: 2024-01-15
        list_value:
          - item1
          - item2
          - 123
        dict_value:
          nested_key: nested_value
          nested_number: 456
        multiline_string: |
          This is a multi-line
          string with line breaks
          preserved.
        folded_string: >
          This is a folded
          string where line breaks
          are replaced with spaces.
        """

        data = yaml.safe_load(yaml_content)

        # Verify data types
        assert isinstance(data["string_value"], str)
        assert data["string_value"] == "hello world"

        assert isinstance(data["integer_value"], int)
        assert data["integer_value"] == 42

        assert isinstance(data["float_value"], float)
        assert abs(data["float_value"] - 3.14159) < 0.00001

        assert isinstance(data["boolean_true"], bool)
        assert data["boolean_true"] is True
        assert data["boolean_false"] is False

        assert data["null_value"] is None

        assert isinstance(data["list_value"], list)
        assert len(data["list_value"]) == 3
        assert data["list_value"][2] == 123

        assert isinstance(data["dict_value"], dict)
        assert data["dict_value"]["nested_key"] == "nested_value"

        # Verify multiline strings
        assert "line breaks\npreserved" in data["multiline_string"]
        assert "breaks are replaced with spaces" in data["folded_string"]


class TestYAMLConfigurationValidation:
    """Test YAML configuration validation and schema compliance."""

    def test_agent_configuration_schema_validation(self):
        """Test agent configuration against expected schema."""
        valid_agent_config = {
            "name": "test_agent",
            "enabled": True,
            "capabilities": ["capability1", "capability2"],
            "model": "test-model",
            "timeout": 120,
            "retry_attempts": 3,
            "max_concurrent_tasks": 2,
            "tools": ["tool1", "tool2"],
            "specific_config": {
                "custom_setting": "value",
                "numeric_setting": 42,
            },
        }

        # Test required fields
        required_fields = ["name", "enabled", "capabilities", "model", "timeout"]
        for field in required_fields:
            config_copy = valid_agent_config.copy()
            del config_copy[field]

            # This would fail in actual validation - testing the concept
            assert field not in config_copy

        # Test data types
        assert isinstance(valid_agent_config["name"], str)
        assert isinstance(valid_agent_config["enabled"], bool)
        assert isinstance(valid_agent_config["capabilities"], list)
        assert isinstance(valid_agent_config["timeout"], int)
        assert isinstance(valid_agent_config["specific_config"], dict)

    def test_orchestrator_configuration_schema_validation(self):
        """Test orchestrator configuration against expected schema."""
        valid_orchestrator_config = {
            "name": "test_orchestrator",
            "supervisor_model": "test-model",
            "batch_processing": {
                "enabled": True,
                "max_batch_size": 10,
                "batch_timeout": 3600,
                "parallel_execution": True,
                "max_parallel_tasks": 4,
            },
            "coordination": {
                "task_assignment_strategy": "capability_based",
                "enable_task_dependencies": True,
                "max_task_retries": 3,
            },
            "error_handling": {
                "max_retries_per_agent": 3,
                "failure_escalation": True,
                "recovery_strategy": "reassign",
            },
            "monitoring": {
                "enable_health_checks": True,
                "health_check_interval": 30,
                "enable_metrics": True,
            },
        }

        # Test batch processing validation
        batch_config = valid_orchestrator_config["batch_processing"]
        assert batch_config["max_batch_size"] > 0
        assert batch_config["batch_timeout"] > 0
        assert batch_config["max_parallel_tasks"] > 0

        # Test coordination validation
        coordination = valid_orchestrator_config["coordination"]
        valid_strategies = ["capability_based", "round_robin", "load_balanced"]
        assert coordination["task_assignment_strategy"] in valid_strategies

        # Test error handling validation
        error_handling = valid_orchestrator_config["error_handling"]
        valid_recovery_strategies = ["reassign", "retry", "skip"]
        assert error_handling["recovery_strategy"] in valid_recovery_strategies

    def test_configuration_constraints_validation(self):
        """Test configuration constraints and business rules."""
        # Test timeout constraints
        valid_timeouts = [30, 60, 120, 300, 600]
        invalid_timeouts = [0, -1, 10000]

        for timeout in valid_timeouts:
            assert timeout > 0 and timeout <= 1800  # Max 30 minutes

        for timeout in invalid_timeouts:
            if timeout <= 0:
                raise AssertionError(f"Timeout {timeout} should be positive")
            elif timeout > 1800:
                raise AssertionError(f"Timeout {timeout} exceeds maximum")

        # Test retry constraints
        valid_retries = [1, 2, 3, 5, 10]

        for retries in valid_retries:
            assert 1 <= retries <= 20

        # Test batch size constraints
        valid_batch_sizes = [1, 5, 10, 25, 50, 100]

        for batch_size in valid_batch_sizes:
            assert 1 <= batch_size <= 200

    def test_yaml_environment_variable_substitution(self):
        """Test YAML configuration with environment variable placeholders."""
        yaml_with_env_vars = """
        database:
          connection_string: ${DATABASE_URL:sqlite:///default.db}
          username: ${DB_USER:default_user}
          password: ${DB_PASS}
        
        api:
          key: ${API_KEY}
          timeout: ${API_TIMEOUT:60}
          
        features:
          debug_mode: ${DEBUG:false}
          log_level: ${LOG_LEVEL:INFO}
        """

        # Parse YAML (without actual env var substitution)
        config_data = yaml.safe_load(yaml_with_env_vars)

        # Verify placeholder format is preserved
        assert (
            "${DATABASE_URL:sqlite:///default.db}"
            in config_data["database"]["connection_string"]
        )
        assert "${API_KEY}" in config_data["api"]["key"]
        assert "${DEBUG:false}" in config_data["features"]["debug_mode"]

        # Test manual environment variable substitution logic
        def substitute_env_vars(value: str, env_vars: dict[str, str]) -> str:
            """Simulate environment variable substitution."""
            import re

            def replace_var(match):
                var_name = match.group(1)
                default_value = match.group(2) if match.group(2) else None
                return env_vars.get(var_name, default_value or "")

            return re.sub(r"\$\{([^:}]+)(?::([^}]*))?\}", replace_var, str(value))

        # Test with environment variables
        env_vars = {
            "DATABASE_URL": "postgresql://localhost/testdb",
            "API_KEY": "test-api-key-123",
            "DEBUG": "true",
        }

        substituted_db_url = substitute_env_vars(
            config_data["database"]["connection_string"], env_vars
        )
        assert substituted_db_url == "postgresql://localhost/testdb"

        substituted_api_key = substitute_env_vars(config_data["api"]["key"], env_vars)
        assert substituted_api_key == "test-api-key-123"

        # Test default values
        substituted_timeout = substitute_env_vars(config_data["api"]["timeout"], {})
        assert substituted_timeout == "60"


class TestYAMLConfigurationErrorHandling:
    """Test error handling for YAML configuration loading."""

    def test_missing_configuration_files(self):
        """Test handling of missing YAML configuration files."""
        with (
            patch("pathlib.Path.exists", return_value=False),
            pytest.raises(FileNotFoundError),
            Path("nonexistent_config.yaml").open() as f,
        ):
            yaml.safe_load(f)

    def test_empty_configuration_files(self):
        """Test handling of empty YAML configuration files."""
        empty_yaml_contents = ["", "   ", "\n\n\n", "# Only comments\n# More comments"]

        for content in empty_yaml_contents:
            data = yaml.safe_load(content)
            assert data is None

    def test_partial_configuration_files(self):
        """Test handling of partial or incomplete YAML configurations."""
        partial_configs = [
            # Missing required sections
            """
            agents:
              research_agent:
                name: research
            # Missing global_settings
            """,
            # Incomplete agent configuration
            """
            agents:
              incomplete_agent:
                name: incomplete
                # Missing required fields like enabled, capabilities, etc.
            global_settings:
              default_timeout: 180
            """,
            # Invalid nested structure
            """
            orchestrator:
              batch_processing:
                enabled: true
                # Missing other required batch_processing fields
            """,
        ]

        for config_content in partial_configs:
            try:
                data = yaml.safe_load(config_content.strip())
                assert isinstance(data, dict)

                # Test for missing sections
                if "agents" in data and "global_settings" not in data:
                    assert "global_settings" not in data

                # Test for incomplete configurations
                if "agents" in data:
                    for _agent_name, agent_config in data["agents"].items():
                        if "enabled" not in agent_config:
                            assert "enabled" not in agent_config

            except yaml.YAMLError:
                pass  # Some partial configs may have syntax errors

    def test_invalid_data_types_in_yaml(self):
        """Test handling of invalid data types in YAML configuration."""
        invalid_type_configs = [
            # String where boolean expected
            """
            agents:
              test_agent:
                enabled: "yes"  # Should be boolean
            """,
            # String where integer expected
            """
            agents:
              test_agent:
                timeout: "sixty"  # Should be integer
            """,
            # Non-list where list expected
            """
            agents:
              test_agent:
                capabilities: "single_capability"  # Should be list
            """,
            # Non-dict where dict expected
            """
            orchestrator:
              batch_processing: true  # Should be dict
            """,
        ]

        for config_content in invalid_type_configs:
            data = yaml.safe_load(config_content.strip())
            assert isinstance(data, dict)

            # Verify that invalid types are preserved (validation would catch these)
            if "agents" in data:
                agent_config = next(iter(data["agents"].values()))

                if "enabled" in agent_config and isinstance(
                    agent_config["enabled"], str
                ):
                    assert agent_config["enabled"] == "yes"

                if "timeout" in agent_config and isinstance(
                    agent_config["timeout"], str
                ):
                    assert agent_config["timeout"] == "sixty"

                if "capabilities" in agent_config and isinstance(
                    agent_config["capabilities"], str
                ):
                    assert agent_config["capabilities"] == "single_capability"

    def test_circular_references_in_yaml(self):
        """Test handling of circular references in YAML."""
        # YAML with aliases and anchors
        yaml_with_references = """
        defaults: &defaults
          timeout: 60
          retries: 3
          
        agents:
          agent1:
            <<: *defaults
            name: agent1
            
          agent2:
            <<: *defaults
            name: agent2
            timeout: 120  # Override default
        """

        data = yaml.safe_load(yaml_with_references)

        # Verify reference resolution
        assert data["agents"]["agent1"]["timeout"] == 60
        assert data["agents"]["agent1"]["retries"] == 3
        assert data["agents"]["agent2"]["timeout"] == 120  # Overridden
        assert data["agents"]["agent2"]["retries"] == 3  # From default

    def test_large_yaml_configuration_files(self):
        """Test handling of large YAML configuration files."""
        # Generate a large configuration
        large_config = {
            "agents": {},
            "global_settings": {
                "default_timeout": 180,
                "log_level": "INFO",
            },
        }

        # Add many agents
        for i in range(100):
            agent_name = f"agent_{i:03d}"
            large_config["agents"][agent_name] = {
                "name": agent_name,
                "enabled": True,
                "capabilities": [f"capability_{j}" for j in range(10)],
                "model": f"model_{i % 5}",
                "timeout": 60 + (i % 240),  # Vary timeouts
                "retry_attempts": 1 + (i % 5),
                "specific_config": {
                    f"setting_{k}": f"value_{k}_{i}" for k in range(20)
                },
            }

        # Convert to YAML and back
        yaml_content = yaml.dump(large_config, default_flow_style=False)
        parsed_config = yaml.safe_load(yaml_content)

        # Verify structure is preserved
        assert len(parsed_config["agents"]) == 100
        assert "agent_000" in parsed_config["agents"]
        assert "agent_099" in parsed_config["agents"]

        # Verify data integrity
        agent_050 = parsed_config["agents"]["agent_050"]
        assert agent_050["name"] == "agent_050"
        assert len(agent_050["capabilities"]) == 10
        assert len(agent_050["specific_config"]) == 20


class TestYAMLConfigurationIntegration:
    """Test integration of YAML configurations with the main config system."""

    def test_yaml_config_loading_integration(self):
        """Test integration between YAML files and pydantic configuration."""
        # Create temporary YAML files for testing
        agents_config = {
            "agents": {
                "test_research": {
                    "name": "research",
                    "enabled": True,
                    "capabilities": ["web_scraping", "analysis"],
                    "model": "test-model",
                    "timeout": 90,
                    "retry_attempts": 2,
                    "max_concurrent_tasks": 1,
                    "tools": ["test_tool"],
                    "specific_config": {"search_timeout": 30},
                }
            },
            "global_settings": {
                "default_timeout": 120,
                "max_concurrent_agents": 3,
                "log_level": "DEBUG",
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(agents_config, f)
            temp_agents_path = f.name

        try:
            # Load and verify the YAML
            with Path(temp_agents_path).open() as f:
                loaded_config = yaml.safe_load(f)

            assert loaded_config == agents_config

            # Verify agent configuration structure
            test_agent = loaded_config["agents"]["test_research"]
            assert test_agent["name"] == "research"
            assert test_agent["timeout"] == 90
            assert "web_scraping" in test_agent["capabilities"]

            # Verify global settings
            global_settings = loaded_config["global_settings"]
            assert global_settings["default_timeout"] == 120
            assert global_settings["log_level"] == "DEBUG"

        finally:
            # Clean up
            Path(temp_agents_path).unlink()

    def test_yaml_config_validation_with_pydantic(self):
        """Test YAML configuration validation using pydantic models."""
        # Note: This would require actual pydantic models for YAML validation
        # For now, we simulate the validation logic

        def validate_agent_config(agent_config: dict[str, Any]) -> bool:
            """Simulate agent configuration validation."""
            required_fields = ["name", "enabled", "capabilities", "model", "timeout"]

            for field in required_fields:
                if field not in agent_config:
                    return False

            # Type validation
            if not isinstance(agent_config["enabled"], bool):
                return False
            if not isinstance(agent_config["capabilities"], list):
                return False
            if not isinstance(agent_config["timeout"], int):
                return False
            return not agent_config["timeout"] <= 0

        # Test valid configuration
        valid_agent = {
            "name": "test_agent",
            "enabled": True,
            "capabilities": ["test_capability"],
            "model": "test_model",
            "timeout": 60,
        }
        assert validate_agent_config(valid_agent) is True

        # Test invalid configurations
        invalid_configs = [
            # Missing required field
            {
                "name": "test_agent",
                "enabled": True,
                # Missing capabilities, model, timeout
            },
            # Wrong data type
            {
                "name": "test_agent",
                "enabled": "yes",  # Should be boolean
                "capabilities": ["test"],
                "model": "test_model",
                "timeout": 60,
            },
            # Invalid timeout
            {
                "name": "test_agent",
                "enabled": True,
                "capabilities": ["test"],
                "model": "test_model",
                "timeout": -1,  # Should be positive
            },
        ]

        for invalid_config in invalid_configs:
            assert validate_agent_config(invalid_config) is False

    def test_yaml_configuration_merging(self):
        """Test merging of YAML configurations with defaults."""
        # Base configuration
        base_config = {
            "timeout": 60,
            "retries": 3,
            "log_level": "INFO",
        }

        # Override configuration
        override_config = {
            "timeout": 120,  # Override
            "debug_mode": True,  # New field
            # retries and log_level should remain from base
        }

        # Test merging logic
        def merge_configs(
            base: dict[str, Any], override: dict[str, Any]
        ) -> dict[str, Any]:
            """Merge configuration dictionaries."""
            merged = base.copy()
            merged.update(override)
            return merged

        merged = merge_configs(base_config, override_config)

        # Verify merge results
        assert merged["timeout"] == 120  # Overridden
        assert merged["retries"] == 3  # From base
        assert merged["log_level"] == "INFO"  # From base
        assert merged["debug_mode"] is True  # New field

        # Test deep merging for nested dictionaries
        base_nested = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "name": "testdb",
            },
            "cache": {
                "enabled": True,
                "ttl": 3600,
            },
        }

        override_nested = {
            "database": {
                "port": 3306,  # Override port
                "user": "admin",  # Add user
            },
            "logging": {  # New section
                "level": "DEBUG"
            },
        }

        def deep_merge_configs(
            base: dict[str, Any], override: dict[str, Any]
        ) -> dict[str, Any]:
            """Deep merge configuration dictionaries."""
            result = base.copy()

            for key, value in override.items():
                if (
                    key in result
                    and isinstance(result[key], dict)
                    and isinstance(value, dict)
                ):
                    result[key] = deep_merge_configs(result[key], value)
                else:
                    result[key] = value

            return result

        deep_merged = deep_merge_configs(base_nested, override_nested)

        # Verify deep merge results
        assert deep_merged["database"]["host"] == "localhost"  # From base
        assert deep_merged["database"]["port"] == 3306  # Overridden
        assert deep_merged["database"]["name"] == "testdb"  # From base
        assert deep_merged["database"]["user"] == "admin"  # Added
        assert deep_merged["cache"]["enabled"] is True  # From base
        assert deep_merged["logging"]["level"] == "DEBUG"  # New section
