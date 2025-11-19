from sqlalchemy import Column, Integer, String, Float, JSON, DateTime
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime, timezone
import json

Base = declarative_base()

class Report(Base):
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, index=True)
    brand_name = Column(String, nullable=False)
    keywords = Column(JSON, nullable=False)  # List of keywords
    overall_score = Column(Float, nullable=False)
    provider_analyses = Column(JSON, nullable=False)  # Detailed analysis per provider
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    def to_dict(self):
        # Parse JSON fields if they're strings
        keywords = self.keywords if isinstance(self.keywords, list) else json.loads(self.keywords) if isinstance(self.keywords, str) else []
        provider_analyses = self.provider_analyses if isinstance(self.provider_analyses, list) else json.loads(self.provider_analyses) if isinstance(self.provider_analyses, str) else []
        
        return {
            "id": self.id,
            "brand_name": self.brand_name,
            "keywords": keywords,
            "overall_score": self.overall_score,
            "provider_analyses": provider_analyses,
            "created_at": self.created_at.isoformat() if self.created_at is not None else None
        }