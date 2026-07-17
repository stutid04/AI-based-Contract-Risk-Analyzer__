# schemas.py

from pydantic import BaseModel, Field

class ContractRequest(BaseModel):
 contract_text: str = Field(..., description="Raw contract text")

class RiskRequest(BaseModel):
 contract_text: str
 risk_level: str = "HIGH"
 top_n: int = 5
