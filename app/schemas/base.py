from typing import Dict, Any
from pydantic import BaseModel

class AiResponseBaseModel(BaseModel):
    class Config:
        extra = "forbid"