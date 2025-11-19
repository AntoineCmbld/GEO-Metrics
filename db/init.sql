CREATE TABLE IF NOT EXISTS reports (
    id SERIAL PRIMARY KEY,
    brand_name VARCHAR(255) NOT NULL,
    keywords JSONB NOT NULL,
    overall_score FLOAT NOT NULL,
    provider_analyses JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_brand_name ON reports(brand_name);
CREATE INDEX idx_created_at ON reports(created_at);