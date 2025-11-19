from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from pydantic import BaseModel
from database import get_db
from services.report_generator import ReportGenerator
from models.report_model import Report

router = APIRouter(prefix="/reports", tags=["reports"])

class CreateReportRequest(BaseModel):
    content: str
    llm_responses: List[dict]

class ReportResponse(BaseModel):
    id: int
    content: str
    score: float
    llm_responses: List[dict]
    created_at: str

    class Config:
        from_attributes = True

@router.post("/", response_model=ReportResponse, status_code=201)
async def create_report(request: CreateReportRequest, db: Session = Depends(get_db)):
    generator = ReportGenerator(db)
    report = generator.generate_report(request.content, request.llm_responses)
    return report

@router.get("/", response_model=List[ReportResponse])
async def list_reports(db: Session = Depends(get_db)):
    reports = db.query(Report).all()
    return reports

@router.get("/{report_id}", response_model=ReportResponse)
async def get_report(report_id: int, db: Session = Depends(get_db)):
    report = db.query(Report).filter(Report.id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    return report

@router.delete("/{report_id}", status_code=204)
async def delete_report(report_id: int, db: Session = Depends(get_db)):
    report = db.query(Report).filter(Report.id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    db.delete(report)
    db.commit()
    return None

@router.get("/{report_id}/score")
async def get_report_score(report_id: int, db: Session = Depends(get_db)):
    report = db.query(Report).filter(Report.id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    return {"report_id": report_id, "score": report.score}

@router.get("/{report_id}/llm-responses")
async def get_report_llm_responses(report_id: int, db: Session = Depends(get_db)):
    report = db.query(Report).filter(Report.id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    return {"report_id": report_id, "llm_responses": report.llm_responses}