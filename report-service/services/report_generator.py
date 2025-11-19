from models.report_model import Report
from typing import List, Dict
from sqlalchemy.orm import Session
import logging
import re

logger = logging.getLogger(__name__)

class ReportGenerator:
    def __init__(self, db_session: Session):
        self.db_session = db_session

    def generate_report(self, brand_name: str, keywords: List[str], llm_responses: List[Dict]) -> Report:
        try:
            # Validate input
            if not brand_name:
                raise ValueError("Brand name cannot be empty")
            if not keywords:
                raise ValueError("At least one keyword is required")
            if not llm_responses:
                raise ValueError("At least one LLM response is required")
            
            # Analyze each provider's response
            provider_analyses = []
            for llm_response in llm_responses:
                # Validate required fields
                provider = llm_response.get("provider")
                if not provider:
                    raise ValueError("Each LLM response must include a 'provider' field")
                
                response_text = llm_response.get("response", "")
                model = llm_response.get("model", "unknown")
                
                analysis = self._analyze_response(
                    response_text,
                    keywords,
                    provider,
                    model
                )
                provider_analyses.append(analysis)
            
            # Compute overall GEO score
            overall_score = self._compute_overall_score(provider_analyses)
            
            # Create and save report
            report = Report(
                brand_name=brand_name,
                keywords=keywords,
                overall_score=overall_score,
                provider_analyses=provider_analyses
            )
            self.db_session.add(report)
            self.db_session.commit()
            self.db_session.refresh(report)
            
            logger.info(f"Generated report {report.id} for brand '{brand_name}' with score {overall_score}")
            return report
            
        except Exception as e:
            self.db_session.rollback()
            logger.error(f"Error generating report: {str(e)}")
            raise

    def _analyze_response(self, response: str, keywords: List[str], provider: str, model: str) -> Dict:
        """
        Analyze a single LLM response for keyword matches
        """
        response_lower = response.lower()
        keyword_matches = []
        
        for keyword in keywords:
            # Count occurrences of keyword (case-insensitive)
            count = len(re.findall(r'\b' + re.escape(keyword.lower()) + r'\b', response_lower))
            keyword_matches.append({
                "keyword": keyword,
                "count": count,
                "found": count > 0
            })
        
        # Calculate visibility score for this provider
        # Score = (keywords found / total keywords) * 100
        keywords_found = sum(1 for km in keyword_matches if km["found"])
        visibility_score = (keywords_found / len(keywords)) * 100 if keywords else 0
        
        return {
            "provider": provider,
            "model": model,
            "response": response,
            "keyword_matches": keyword_matches,
            "visibility_score": visibility_score
        }

    def _compute_overall_score(self, provider_analyses: List[Dict]) -> float:
        """
        Compute overall GEO score as average of all provider visibility scores
        """
        if not provider_analyses:
            return 0.0
        
        total_score = sum(analysis["visibility_score"] for analysis in provider_analyses)
        return round(total_score / len(provider_analyses), 2)