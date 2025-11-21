from models.report_model import Report
from typing import List, Dict
from sqlalchemy.orm import Session
import logging
import re
from datetime import datetime, timezone
import json

logger = logging.getLogger(__name__)

class ReportGenerator:
    def __init__(self, db_session: Session):
        self.db_session = db_session

    def count_brand_mentions(self, text: str, brand_name: str) -> int:
        """Count case-insensitive mentions of brand name in text"""
        if not text or not brand_name:
            return 0
        pattern = re.compile(re.escape(brand_name), re.IGNORECASE)
        return len(pattern.findall(text))

    def analyze_keyword_response(self, keyword: str, responses: list, brand_name: str) -> dict:
        """Analyze all model responses for a specific keyword"""
        model_results = []
        total_mentions = 0
        
        for resp in responses:
            if resp.get("keyword") == keyword:
                mentions = self.count_brand_mentions(resp.get("response", ""), brand_name)
                total_mentions += mentions
                
                model_results.append({
                    "model": resp.get("model"),
                    "response": resp.get("response", ""),
                    "brand_mentions": mentions
                })
        
        return {
            "keyword": keyword,
            "model_results": model_results,
            "total_mentions": total_mentions,
            "visibility_score": min(total_mentions * 25, 100)  # 1 mention = 25 points, cap at 100
        }

    def generate_report(self, brand_name: str, keywords: list, llm_responses: list):
        """Generate GEO report by analyzing brand mentions per keyword"""
        try:
            logger.info(f"Generating report for brand: {brand_name}")
            
            # Analyze each keyword
            keyword_analyses = []
            for keyword in keywords:
                analysis = self.analyze_keyword_response(keyword, llm_responses, brand_name)
                keyword_analyses.append(analysis)
            
            # Calculate overall score (average of keyword scores)
            overall_score = sum(k["visibility_score"] for k in keyword_analyses) / len(keyword_analyses) if keyword_analyses else 0
            
            logger.info(f"Overall score: {overall_score}")
            
            # Create report
            report = Report(
                brand_name=brand_name,
                keywords=json.dumps(keywords),
                overall_score=overall_score,
                provider_analyses=json.dumps(keyword_analyses),
                created_at=datetime.now(timezone.utc)
            )
            
            self.db_session.add(report)
            self.db_session.commit()
            self.db_session.refresh(report)
            
            logger.info(f"Report saved with ID: {report.id}")
            return report
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}", exc_info=True)
            self.db_session.rollback()
            raise