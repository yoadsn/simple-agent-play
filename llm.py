import os
from enum import Enum

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.chat_models.base import BaseChatModel
from langsmith.utils import get_env_var

get_env_var.cache_clear()
load_dotenv(".env", override=True)


class ModelName(str, Enum):
    OR_GEMINI_2_0_FLASH_MODEL_NAME = "google/gemini-2.0-flash-001"
    OR_GEMINI_2_5_FLASH_MODEL_NAME = "google/gemini-2.5-flash"
    OR_GEMINI_2_5_PRO_MODEL_NAME = "google/gemini-2.5-pro-preview-05-06"
    OR_CLAUDE_SONNET_4_MODEL_NAME = "anthropic/claude-sonnet-4"


def get_open_router_chat_model(model_name: str, **kwargs) -> BaseChatModel:
    return init_chat_model(
        model_name,
        model_provider="openai",
        api_key=os.environ["OPEN_ROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
        extra_body={
            "provider": {
                # "only": ["google-vertex"],
                "require_parameters": True
            }
        },
        **kwargs,
    )
