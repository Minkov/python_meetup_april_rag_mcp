from abc import ABC, abstractmethod
from enum import Enum
import json
import os
from typing import Type, TypeVar
from openai import OpenAI
import google.generativeai as genai
from google.generativeai.types import GenerationConfig

from app.configs import FAST_GEMINI_MODEL, FAST_OPENAI_MODEL, THINKING_OPENAI_MODEL
from app.schemas.base import AiResponseBaseModel
from app.schemas.ai_service_schemas import AiServicePrompt


class AiServiceCapability(Enum):
    FAST = "fast"
    THINKING = "thinking"


TGenericModel = TypeVar(
    'TGenericModel', bound=AiResponseBaseModel, covariant=True)


class AiServiceBase(ABC):
    strict_schema_additions_template = (
        '\n\nCRITICAL: You MUST strictly follow the provided JSON schema '
        'structure for your response. Ensure your entire output is a single, '
        'valid JSON object matching the schema, with no extra text before or after it.'
        '\nHere is the JSON schema you MUST adhere to:\n```json\n{json_schema}\n```'
    )

    @abstractmethod
    def generate_response(self, response_model: Type[TGenericModel], prompt: AiServicePrompt) -> TGenericModel:
        pass


class OpenAIAiService(AiServiceBase):
    def __init__(self, openai_api_key, model: str = FAST_OPENAI_MODEL):
        self.client = OpenAI(api_key=openai_api_key)
        self.model = model

    def generate_response(
        self,
        response_model: Type[TGenericModel],
        prompt: AiServicePrompt,
        temperature: float = 0.5,
        max_tokens: int = 1000,
        top_p: float = 1.0,
    ) -> TGenericModel:
        schema_prompt = None
        response_format = None
        if self.model == THINKING_OPENAI_MODEL:
            json_schema = {
                'name': response_model.__name__,
                'schema': response_model.model_json_schema(),
            }
            response_format = {
                'type': 'json_schema',
                'json_schema': json_schema
            }
            system_prompt = prompt.system_prompt
        else:
            schema_dict = response_model.model_json_schema()
            schema_json_string = json.dumps(schema_dict, indent=2)
            schema_prompt = self.strict_schema_additions_template.format(
                json_schema=schema_json_string)
            system_prompt = f"{prompt.system_prompt}\n\n{schema_prompt}"

            response_format = {
                'type': 'json_object'
            }

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt.user_prompt}
            ],
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        try:
            return response_model(**json.loads(response.choices[0].message.content))
        except Exception as e:
            raise ValueError(f"Failed to validate response: {e}")


genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


class GenimiAiService(AiServiceBase):
    default_config = GenerationConfig(
        temperature=0.2,
        max_output_tokens=8192,
        top_p=1.0,
        top_k=1,
    )

    def __init__(
        self, model: str = FAST_GEMINI_MODEL,
    ):
        self.client = genai.GenerativeModel(model)

    def generate_response(self, response_model: Type[TGenericModel], prompt: AiServicePrompt) -> TGenericModel:
        """
        Makes a call to the Gemini API. Optionally performs a preliminary AI step
        to generate a web search query, executes the search, adds results to context,
        and then makes the main call expecting JSON output.

        Args:
            model: The Pydantic model class for the expected *final* response.
            prompt_name: The name of the *main* prompt template (e.g., 'research/main').
            context: Dictionary with values to fill prompt placeholders.
            configs: Optional dictionary to override default generation parameters for the *main* call.

        Returns:
            An instance of the Pydantic model 'model' populated with the AI response.
        """
        schema_dict = response_model.model_json_schema()
        schema_json_string = json.dumps(
            schema_dict, indent=2)
        schema_instruction = self.strict_schema_additions_template.format(
            json_schema=schema_json_string)

        full_prompt = f"""{prompt.system_prompt}

{prompt.user_prompt}

{schema_instruction}
"""

        generation_config = self.default_config

        try:
            response = self.client.generate_content(
                contents=full_prompt,
                generation_config=generation_config,
            )

            # --- Step 6: Process Final Response ---
            if not response.candidates:
                raise ValueError("Gemini main response was empty or blocked.")

            finish_reason = response.candidates[0].finish_reason.name
            if finish_reason not in ("STOP", "MAX_TOKENS"):
                print(
                    f"Warning: Gemini main response finished unexpectedly: {finish_reason}")
                print(
                    f"Safety Ratings: {response.candidates[0].safety_ratings}")

            raw_content = response.text

            if raw_content.strip().startswith("```json"):
                raw_content = raw_content.strip()[7:]
                if raw_content.strip().endswith("```"):
                    raw_content = raw_content.strip()[:-3]

            if not raw_content.strip():
                print(
                    "Error: Received empty content after stripping markdown for main response.")
                print(f"Original Response Text: {response.text}")
                raise ValueError(
                    "Failed to get valid content from main AI response.")

            try:
                response_json = json.loads(raw_content.strip())
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from main Gemini response: {e}")
                print(
                    f"Raw content attempted to parse:\n>>>\n{raw_content.strip()}\n<<<")
                print(f"Finish Reason: {finish_reason}")
                print(
                    f"Safety Ratings: {response.candidates[0].safety_ratings if response.candidates else 'N/A'}")
                raise ValueError(
                    f"Failed to decode JSON from main AI response. Content: {raw_content.strip()}") from e

            validated_model = response_model(**response_json)
            return validated_model

        except Exception as e:
            print(
                f"An error occurred during Gemini main API call or processing: {e}")
            raise


def get_ai_service(capability: AiServiceCapability) -> AiServiceBase:
    """Get the appropriate AI service based on the capability."""

    # if capability == AiServiceCapability.FAST:
    #     return GenimiAiService(FAST_GEMINI_MODEL)
    # elif capability == AiServiceCapability.THINKING:
    #     return GenimiAiService(THINKING_GEMINI_MODEL)
    if capability == AiServiceCapability.FAST:
        return OpenAIAiService(os.getenv("OPENAI_API_KEY"), FAST_OPENAI_MODEL)
    elif capability == AiServiceCapability.THINKING:
        return OpenAIAiService(os.getenv("OPENAI_API_KEY"), THINKING_OPENAI_MODEL)
