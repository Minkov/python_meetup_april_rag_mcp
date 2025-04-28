import json
from typing import Type
from openai import OpenAI
from pydantic import BaseModel

from app.schemas.base import AiResponseBaseModel


class AiServicePrompt(BaseModel):
    system_prompt: str
    user_prompt: str


class AiService:
    def __init__(self, openai_api_key):
        self.client = OpenAI(api_key=openai_api_key)

    def generate_response(
        self,
        response_model: Type[AiResponseBaseModel],
        prompt: AiServicePrompt,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.5,
        max_tokens: int = 1000,
        top_p: float = 1.0,
    ):
        json_schema = {
            'name': response_model.__name__,
            'schema': response_model.model_json_schema(),
        }

        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt.system_prompt},
                {"role": "user", "content": prompt.user_prompt}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": json_schema,
            },
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        try:
            return response_model(**json.loads(response.choices[0].message.content))
        except Exception as e:
            raise ValueError(f"Failed to validate response: {e}")
