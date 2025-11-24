from models.report_model import Report, LLMResponse
from sqlalchemy.orm import Session
from typing import Optional

class ReportGenerator:
    def __init__(self, db: Session):
        self.db = db

    def generate_report(self, brand_name: str, user_id: Optional[int], llm_responses: list):
        # Create the report object
        report = Report(
            brand_name=brand_name,
            user_id=user_id
        )
        self.db.add(report)
        self.db.flush()  # Get report.id

        # Add LLM responses
        for resp in llm_responses:
            llm_response = LLMResponse(
                report_id=report.id,
                prompt_template=resp.get("prompt_template", ""),
                region=resp.get("region", ""),
                language_code=resp.get("language_code", ""),
                keyword=resp.get("keyword", ""),
                model=resp.get("model", ""),
                prompt_text=resp.get("prompt_text", ""),
                response=resp.get("response", "")
            )
            self.db.add(llm_response)

        self.db.commit()
        self.db.refresh(report)
        return report