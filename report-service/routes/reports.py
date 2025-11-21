from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Dict
from pydantic import BaseModel, Field
from database import get_db
from services.report_generator import ReportGenerator
from models.report_model import Report
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/reports", tags=["reports"])

class LLMResponse(BaseModel):
    model: str = Field(..., description="Model used for the query")
    keyword: str = Field(..., description="Keyword that was queried")
    response: str = Field(..., description="Response from the LLM")

class CreateReportRequest(BaseModel):
    brand_name: str = Field(..., description="Brand name to analyze")
    keywords: List[str] = Field(..., description="Keywords to search for in responses")
    llm_responses: List[LLMResponse] = Field(..., description="LLM responses to analyze")

class KeywordMatch(BaseModel):
    keyword: str
    count: int
    found: bool

class ProviderAnalysis(BaseModel):
    model: str
    response: str
    keyword_matches: List[KeywordMatch]
    visibility_score: float

class ReportResponse(BaseModel):
    id: int
    brand_name: str
    keywords: List[str]
    overall_score: float
    provider_analyses: List[ProviderAnalysis]
    created_at: str

    class Config:
        from_attributes = True

@router.post("", status_code=201)
async def create_report(request: CreateReportRequest, db: Session = Depends(get_db)):
    try:
        logger.info(f"Creating report for brand: {request.brand_name}")
        logger.info(f"Keywords: {request.keywords}")
        logger.info(f"LLM responses count: {len(request.llm_responses)}")
        
        # Convert Pydantic models to dicts using model_dump()
        llm_responses = [resp.model_dump() for resp in request.llm_responses]
        
        generator = ReportGenerator(db)
        report = generator.generate_report(
            brand_name=request.brand_name,
            keywords=request.keywords,
            llm_responses=llm_responses
        )
        
        logger.info(f"Successfully created report {report.id}")
        
        # Convert to dict to ensure datetime is serialized
        return report.to_dict()
        
    except Exception as e:
        logger.error(f"Error creating report: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error creating report: {str(e)}")

@router.get("")
async def list_reports(db: Session = Depends(get_db)):
    reports = db.query(Report).all()
    # Convert all reports to dict
    return [report.to_dict() for report in reports]

@router.get("/{report_id}")
async def get_report(report_id: int, db: Session = Depends(get_db)):
    report = db.query(Report).filter(Report.id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    # Convert to dict
    return report.to_dict()

@router.delete("/{report_id}", status_code=204)
async def delete_report(report_id: int, db: Session = Depends(get_db)):
    report = db.query(Report).filter(Report.id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    db.delete(report)
    db.commit()
    return None