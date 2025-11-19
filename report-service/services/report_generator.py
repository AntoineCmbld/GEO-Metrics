from models.report_model import Report
from typing import List, Dict

class ReportGenerator:
    def __init__(self, db_session):
        self.db_session = db_session

    def generate_report(self, content: str, llm_responses: List[Dict]) -> Report:
        # Placeholder: Compute GEO score based on responses
        score = self.compute_geo_score(llm_responses)
        report = Report(content=content, score=score, llm_responses=llm_responses)
        self.db_session.add(report)
        self.db_session.commit()
        return report

    def compute_geo_score(self, responses: List[Dict]) -> float:
        # Simple average score placeholder
        scores = [r.get('score', 0) for r in responses]
        return sum(scores) / len(scores) if scores else 0.0