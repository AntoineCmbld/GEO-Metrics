from sqlalchemy import Column, Integer, String, Float, JSON, DateTime
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class Report(Base):
    __tablename__ = 'reports'
    id = Column(Integer, primary_key=True)
    content = Column(String)
    score = Column(Float)
    llm_responses = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)