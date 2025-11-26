from fastapi import APIRouter, Depends, HTTPException, Header
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel, Field
from database import get_db
from services.report_generator import ReportGenerator
from models.report_model import Report, LLMResponse
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/reports", tags=["reports"])

class LLMResponseData(BaseModel):
    model: str = Field(..., description="Model used for the query")
    keyword: str = Field(..., description="Keyword that was queried")
    language_code: str = Field(..., description="Language code used")
    region: str = Field(..., description="Region context used")
    prompt_text: str = Field(..., description="Full prompt text sent to the model")
    response: str = Field(..., description="Response from the LLM")

class CreateReportRequest(BaseModel):
    user_id: Optional[int] = Field(default=None, description="Optional user ID")
    brand_name: str = Field(..., description="Brand name to analyze")
    llm_responses: List[LLMResponseData] = Field(..., description="LLM responses to store")

@router.post("", status_code=201)
async def create_report(
    request: CreateReportRequest,
    db: Session = Depends(get_db),
    x_user_id: int = Header(...)
):
    try:
        logger.info(f"Creating report for brand: {request.brand_name} by user {x_user_id}")
        llm_responses = [resp.model_dump() for resp in request.llm_responses]
        generator = ReportGenerator(db)
        report = generator.generate_report(
            user_id=x_user_id,
            brand_name=request.brand_name,
            llm_responses=llm_responses
        )
        logger.info(f"Successfully created report {report.id}")
        return report.to_dict()
    except Exception as e:
        logger.error(f"Error creating report: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error creating report: {str(e)}")

@router.get("")
async def list_reports(
    db: Session = Depends(get_db),
    x_user_id: int = Header(...)
):
    reports = db.query(Report).filter(Report.user_id == x_user_id).all()
    return [report.to_dict() for report in reports]

@router.get("/{report_id}")
async def get_report(
    report_id: int,
    db: Session = Depends(get_db),
    x_user_id: int = Header(...)
):
    report = db.query(Report).filter(Report.id == report_id, Report.user_id == x_user_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found or access denied")
    return report.to_dict()

@router.delete("/{report_id}", status_code=204)
async def delete_report(
    report_id: int,
    db: Session = Depends(get_db),
    x_user_id: int = Header(...)
):
    report = db.query(Report).filter(Report.id == report_id, Report.user_id == x_user_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found or access denied")
    db.delete(report)
    db.commit()
    return None

@router.get("/{report_id}/llm-responses")
async def get_report_llm_responses(
    report_id: int,
    db: Session = Depends(get_db),
    x_user_id: int = Header(...)
):
    report = db.query(Report).filter(Report.id == report_id, Report.user_id == x_user_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found or access denied")
    llm_responses = db.query(LLMResponse).filter(
        LLMResponse.report_id == report_id
    ).all()
    return [resp.to_dict() for resp in llm_responses]