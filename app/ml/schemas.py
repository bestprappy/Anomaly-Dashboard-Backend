"""Pydantic request bodies for /api/ml/* — kept separate from app/schemas.py
so the ML slice stays self-contained and easy to find.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class DropOptionsIn(BaseModel):
    duplicate_site: bool = False
    common_site: bool = False
    shutdown_site: bool = False
    maintenance_site: bool = False


class PreviewRequest(BaseModel):
    drop_options: DropOptionsIn = DropOptionsIn()
    start_month: int = Field(..., description="YYYYMM, inclusive")
    end_month: int = Field(..., description="YYYYMM, inclusive")


class BuildRequest(BaseModel):
    drop_options: DropOptionsIn = DropOptionsIn()
    train_start: int = Field(..., description="YYYYMM, inclusive")
    train_end: int = Field(..., description="YYYYMM, inclusive")
    test_start: int = Field(..., description="YYYYMM, inclusive")
    test_end: int = Field(..., description="YYYYMM, inclusive")
    q_low: float = 0.05
    q_mid: float = 0.50
    q_high: float = 0.95


class ClassifyRequest(BaseModel):
    up: float = 1.5
    down: float = 1 / 1.5
    sustain: float = 1.3