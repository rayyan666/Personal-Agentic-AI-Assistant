"""
Configuration settings for the Agentic AI Personal Agent System
"""
import os
from typing import Dict, Any, Optional
from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class AgentConfig(BaseSettings):
    """Base configuration for all agents"""
    
    # API Keys and Authentication
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    huggingface_api_key: Optional[str] = Field(default=None, env="HUGGINGFACE_API_KEY")
    
    # LLM Configuration
    default_llm_provider: str = Field(default="huggingface", env="DEFAULT_LLM_PROVIDER")
    default_model: str = Field(default="microsoft/DialoGPT-medium", env="DEFAULT_MODEL")
    max_tokens: int = Field(default=2048, env="MAX_TOKENS")
    temperature: float = Field(default=0.7, env="TEMPERATURE")
    
    # Database Configuration
    database_url: str = Field(default="sqlite:///./agents.db", env="DATABASE_URL")
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    
    # Email Configuration
    smtp_server: Optional[str] = Field(default=None, env="SMTP_SERVER")
    smtp_port: int = Field(default=587, env="SMTP_PORT")
    email_username: Optional[str] = Field(default=None, env="EMAIL_USERNAME")
    email_password: Optional[str] = Field(default=None, env="EMAIL_PASSWORD")
    
    # Web Scraping Configuration
    selenium_driver_path: Optional[str] = Field(default=None, env="SELENIUM_DRIVER_PATH")
    scraping_delay: float = Field(default=1.0, env="SCRAPING_DELAY")
    max_concurrent_requests: int = Field(default=5, env="MAX_CONCURRENT_REQUESTS")
    
    # Task Management Configuration
    task_db_path: str = Field(default="./tasks.db", env="TASK_DB_PATH")
    reminder_check_interval: int = Field(default=300, env="REMINDER_CHECK_INTERVAL")  # 5 minutes
    
    # Agent-specific configurations
    personal_assistant_enabled: bool = Field(default=True, env="PERSONAL_ASSISTANT_ENABLED")
    code_generation_enabled: bool = Field(default=True, env="CODE_GENERATION_ENABLED")
    research_agent_enabled: bool = Field(default=True, env="RESEARCH_AGENT_ENABLED")
    resource_finder_enabled: bool = Field(default=True, env="RESOURCE_FINDER_ENABLED")
    email_agent_enabled: bool = Field(default=True, env="EMAIL_AGENT_ENABLED")
    web_scraper_enabled: bool = Field(default=True, env="WEB_SCRAPER_ENABLED")
    task_manager_enabled: bool = Field(default=True, env="TASK_MANAGER_ENABLED")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="./agents.log", env="LOG_FILE")
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Global configuration instance
config = AgentConfig()

# Agent-specific configurations
AGENT_CONFIGS = {
    "personal_assistant": {
        "name": "Personal Assistant",
        "description": "Handles day-to-day tasks, scheduling, and general assistance",
        "model": config.default_model,
        "max_tokens": 1024,
        "temperature": 0.6,
        "system_prompt": """You are a helpful personal assistant. You can help with scheduling, 
        reminders, general questions, and day-to-day tasks. Be friendly, efficient, and proactive."""
    },
    "code_generation": {
        "name": "Code Generation Agent",
        "description": "Specialized in generating, reviewing, and explaining code",
        "model": "microsoft/DialoGPT-medium",  # Fast model for testing
        "max_tokens": 2048,
        "temperature": 0.2,  # Lower temperature for more deterministic code
        "system_prompt": """You are an expert software developer. You can generate high-quality code,
        review existing code, explain complex programming concepts, and help with debugging."""
    },
    "research_agent": {
        "name": "R&D Agent",
        "description": "Conducts research and development tasks",
        "model": config.default_model,
        "max_tokens": 2048,
        "temperature": 0.5,
        "system_prompt": """You are a research specialist. You can conduct literature reviews,
        analyze data, summarize research papers, and provide insights on various topics."""
    },
    "resource_finder": {
        "name": "Resource Finder Agent",
        "description": "Finds the best resources and materials for any topic",
        "model": config.default_model,
        "max_tokens": 1536,
        "temperature": 0.4,
        "system_prompt": """You are an expert at finding high-quality resources, tutorials,
        courses, and materials for any topic. You provide curated recommendations with explanations."""
    },
    "email_agent": {
        "name": "Email Generation Agent",
        "description": "Handles automated email composition and management",
        "model": config.default_model,
        "max_tokens": 1024,
        "temperature": 0.5,
        "system_prompt": """You are an email composition specialist. You can write professional,
        personal, and marketing emails with appropriate tone and structure."""
    },
    "web_scraper": {
        "name": "Web Scraper Agent",
        "description": "Performs web scraping and data extraction",
        "model": config.default_model,
        "max_tokens": 1536,
        "temperature": 0.3,
        "system_prompt": """You are a web scraping specialist. You can extract data from websites,
        analyze web content, and provide structured information from various sources."""
    },
    "task_manager": {
        "name": "Task Manager Agent",
        "description": "Manages daily tasks, scheduling, and productivity",
        "model": config.default_model,
        "max_tokens": 1024,
        "temperature": 0.4,
        "system_prompt": """You are a productivity and task management expert. You can help organize
        tasks, create schedules, set reminders, and optimize daily workflows."""
    }
}

def get_agent_config(agent_name: str) -> Dict[str, Any]:
    """Get configuration for a specific agent"""
    return AGENT_CONFIGS.get(agent_name, {})

def is_agent_enabled(agent_name: str) -> bool:
    """Check if an agent is enabled"""
    enabled_map = {
        "personal_assistant": config.personal_assistant_enabled,
        "code_generation": config.code_generation_enabled,
        "research_agent": config.research_agent_enabled,
        "resource_finder": config.resource_finder_enabled,
        "email_agent": config.email_agent_enabled,
        "web_scraper": config.web_scraper_enabled,
        "task_manager": config.task_manager_enabled,
    }
    return enabled_map.get(agent_name, False)
