import os
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from passlib.hash import bcrypt
from database import get_db
from models.user_model import User
from jose import jwt

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "changeme")

router = APIRouter(prefix="/auth", tags=["auth"])

class LoginRequest(BaseModel):
    username: str
    password: str

@router.post("/login")
def login(request: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter_by(username=request.username).first()
    if not user or not bcrypt.verify(request.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = jwt.encode({"user_id": user.id, "username": user.username}, SECRET_KEY, algorithm="HS256")
    return {"access_token": token, "token_type": "bearer"}