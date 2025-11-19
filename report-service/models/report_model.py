from sqlalchemy import Column, Integer, String, Float, JSON, DateTime
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime, timezone

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
        return {
            "id": self.id,
            "brand_name": self.brand_name,
            "keywords": self.keywords,
            "overall_score": self.overall_score,
            "provider_analyses": self.provider_analyses,
            "created_at": self.created_at.isoformat() if self.created_at is not None else None
        }