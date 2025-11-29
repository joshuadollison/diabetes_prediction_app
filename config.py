"""
Configuration Module for Race Day Prediction Application

This module centralizes all configuration settings for the horse racing
application, mirroring the structure of the previous diabetes project while
switching the domain to racing. All infrastructure choices (Flask, Databricks
model serving, environment-driven config) remain the same so deployment and
operations stay consistent.

Educational Note:
-----------------
Configuration management is a best practice in software development because it:
1. Separates concerns - keeps config separate from business logic
2. Makes the app more maintainable - all settings in one place
3. Enhances security - sensitive data stored in environment variables
4. Enables multiple environments - easy to switch between dev/prod settings
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
# This allows us to store sensitive configuration in a file that won't be committed to git
load_dotenv()


class Config:
    """
    Base configuration class containing all application settings.

    This class uses environment variables with sensible defaults for flexibility.
    Students can modify these settings either by:
    1. Creating a .env file (recommended for sensitive data)
    2. Setting environment variables in their shell
    3. Modifying the default values here (for non-sensitive settings)
    """

    # ============================================================================
    # Flask Application Settings
    # ============================================================================

    # Secret key for Flask session management
    # In production, this should be a long, random string stored as an environment variable
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')

    # Debug mode - should be False in production
    # Debug mode provides detailed error messages and auto-reloading
    DEBUG = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'

    # Server host - 0.0.0.0 allows external connections, 127.0.0.1 is local only
    HOST = os.getenv('FLASK_HOST', '0.0.0.0')

    # Server port - the port on which the Flask application will run
    PORT = int(os.getenv('FLASK_PORT', 4000))

    # ============================================================================
    # MLflow / Databricks Configuration
    # ============================================================================

    # Databricks authentication token
    # IMPORTANT: Never commit this token to version control!
    # Set this in your .env file or as an environment variable
    DATABRICKS_TOKEN = os.getenv('DATABRICKS_TOKEN')

    # MLflow model serving endpoint URL
    # This is the URL where your deployed model is hosted
    # Students should update this to match their own Databricks workspace and endpoint
    MLFLOW_ENDPOINT_URL = os.getenv('MLFLOW_ENDPOINT_URL')

    # Request timeout in seconds for API calls to the MLflow endpoint
    # Prevents the application from hanging indefinitely on slow responses
    REQUEST_TIMEOUT = int(os.getenv('REQUEST_TIMEOUT', 30))

    # ============================================================================
    # Model Configuration
    # ============================================================================

    # Expected feature names that the racing model accepts.
    # These are mock placeholders to be replaced with real features later.
    MODEL_FEATURES = [
        'pace_early',       # Early speed indicator
        'pace_late',        # Closing kick indicator
        'stamina_score',    # Stamina for route distances
        'surface_fit',      # How well the horse handles the surface
        'post_position',    # Gate position
        'class_rating'      # Competition class rating
    ]

    # ============================================================================
    # Application Metadata
    # ============================================================================

    # Application name and version for display purposes
    APP_NAME = os.getenv('APP_NAME', 'Race Day Predictor')
    APP_VERSION = os.getenv('APP_VERSION', '1.0.0')

    # Path to the text configuration file controlling race days, tracks, and races
    RACE_CONFIG_FILE = os.getenv('RACE_CONFIG_FILE', 'race_config.json')

    # Optional flag to force the app to use mocked model outputs
    USE_MOCK_MODEL = os.getenv('USE_MOCK_MODEL', 'True').lower() == 'true'

    # ============================================================================
    # Validation Methods
    # ============================================================================

    @classmethod
    def validate_config(cls):
        """
        Validates that all required configuration variables are set.

        This method should be called on application startup to ensure
        that the application has all necessary configuration to run properly.

        Returns:
            tuple: (is_valid: bool, error_messages: list)

        Educational Note:
        ----------------
        Configuration validation is important because it:
        1. Fails fast - catches configuration errors before they cause runtime issues
        2. Provides clear error messages - helps users understand what's missing
        3. Ensures application integrity - prevents running with incomplete config
        """
        errors = []

        # Check for required Databricks token unless we are in mock mode
        if not cls.DATABRICKS_TOKEN and not cls.USE_MOCK_MODEL:
            errors.append(
                "DATABRICKS_TOKEN is not set. Please set it in your .env file or as an environment variable."
            )

        # Check for required MLflow endpoint URL unless we are in mock mode
        if not cls.MLFLOW_ENDPOINT_URL and not cls.USE_MOCK_MODEL:
            errors.append(
                "MLFLOW_ENDPOINT_URL is not set. Please configure your MLflow endpoint URL."
            )

        # Validate that endpoint URL is a valid HTTPS URL
        if cls.MLFLOW_ENDPOINT_URL and not cls.MLFLOW_ENDPOINT_URL.startswith('https://'):
            errors.append(
                "MLFLOW_ENDPOINT_URL should use HTTPS for security. Current URL: " + cls.MLFLOW_ENDPOINT_URL
            )

        # Validate that the race configuration file exists
        if not os.path.exists(cls.RACE_CONFIG_FILE):
            errors.append(
                f"Race configuration file not found at {cls.RACE_CONFIG_FILE}. "
                "Create the file or update RACE_CONFIG_FILE."
            )

        return len(errors) == 0, errors

    @classmethod
    def print_config_status(cls):
        """
        Prints the current configuration status to the console.

        This is useful for debugging and verifying that configuration
        is loaded correctly. Sensitive values are masked for security.

        Educational Note:
        ----------------
        This method demonstrates the principle of "secure logging" by:
        1. Masking sensitive information (tokens) when displaying config
        2. Providing useful debugging information without compromising security
        3. Making it easy for developers to verify their configuration
        """
        print("\n" + "="*70)
        print("DIABETES PREDICTION APP - CONFIGURATION STATUS")
        print("="*70)
        print(f"App Name: {cls.APP_NAME}")
        print(f"Version: {cls.APP_VERSION}")
        print(f"Debug Mode: {cls.DEBUG}")
        print(f"Host: {cls.HOST}")
        print(f"Port: {cls.PORT}")
        print("-"*70)
        print("MLflow Configuration:")
        print(f"Endpoint URL: {cls.MLFLOW_ENDPOINT_URL}")
        print(f"Token Set: {'Yes (***hidden***)' if cls.DATABRICKS_TOKEN else 'No (NOT SET!)'}")
        print(f"Request Timeout: {cls.REQUEST_TIMEOUT}s")
        print("-"*70)
        print(f"Model Features: {', '.join(cls.MODEL_FEATURES)}")
        print(f"Race Config File: {cls.RACE_CONFIG_FILE}")
        print(f"Using Mock Model: {cls.USE_MOCK_MODEL}")
        print("="*70 + "\n")


class DevelopmentConfig(Config):
    """
    Development-specific configuration.

    Inherits from Config and overrides settings specific to development.
    This allows developers to have different settings while coding and testing.
    """
    DEBUG = True


class ProductionConfig(Config):
    """
    Production-specific configuration.

    Inherits from Config and overrides settings specific to production deployment.
    Production config should prioritize security and performance over developer convenience.

    Educational Note:
    ----------------
    In production, you should:
    1. Always set DEBUG = False
    2. Use strong, random SECRET_KEY
    3. Use environment variables for all sensitive data
    4. Enable HTTPS and other security measures
    5. Implement proper logging and monitoring
    """
    DEBUG = False

    # In production, we require certain environment variables to be set
    # This prevents accidentally running in production with default/insecure values
    @classmethod
    def validate_config(cls):
        is_valid, errors = super().validate_config()

        # Additional production-specific validations
        if cls.SECRET_KEY == 'dev-secret-key-change-in-production':
            errors.append(
                "SECRET_KEY is using the default development value. "
                "Please set a secure random value in production."
            )

        return len(errors) == 0, errors


# Configuration dictionary for easy access to different config classes
# This allows the application to easily switch between configurations
config_by_name = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}


def get_config(config_name='default'):
    """
    Returns the appropriate configuration class based on the environment.

    Args:
        config_name (str): The name of the configuration ('development', 'production', or 'default')

    Returns:
        Config: The configuration class for the specified environment

    Example:
        config = get_config('production')
        app.config.from_object(config)
    """
    return config_by_name.get(config_name, DevelopmentConfig)
