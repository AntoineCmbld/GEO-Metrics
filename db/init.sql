CREATE TABLE IF NOT EXISTS reports (
    id SERIAL PRIMARY KEY,
    content TEXT,
    score FLOAT,
    llm_responses JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);