from typing import Optional
from pydantic import BaseModel


class UploadStatus(BaseModel):
    loaded_files: list[str]
    missing_files: list[str]
    ready: bool
    rows_total: int = 0
    message: str = ""


class SiteTrendPoint(BaseModel):
    month: int
    value: Optional[float]


class SiteTrendResponse(BaseModel):
    site_id: str
    found: bool
    provider: Optional[str] = None
    company: Optional[str] = None
    site_type: Optional[str] = None
    metric: Optional[str] = None
    series: list[SiteTrendPoint] = []