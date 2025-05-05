from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Configuration settings for FastAPI. Values are overridden by .env file."""

    title: str = "My FastAPI Backend"  # Default value
    description: str = "A structured FastAPI backend."  # Default value
    version: str = "1.0"        # Default value
    debug: bool = False         # Default debug mode
    docs_url: str = "/docs"     # Default Swagger UI path // look inside .env file. new path is /swagger

    class Config:
        env_file = ".env" 
        case_sensitive = False 

settings = Settings() 
