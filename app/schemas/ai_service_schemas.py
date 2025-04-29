from pydantic import BaseModel


class AiServicePrompt(BaseModel):
    """Schema for AI service prompts."""
    system_prompt: str
    user_prompt: str 