from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session
from passlib.hash import bcrypt
from database import get_db
from models.user_model import User

router = APIRouter(prefix="/auth", tags=["auth"])

class RegisterRequest(BaseModel):
    username: str
    email: EmailStr
    password: str

@router.post("/register")
def register(request: RegisterRequest, db: Session = Depends(get_db)):
    if db.query(User).filter_by(username=request.username).first():
        raise HTTPException(status_code=400, detail="Username already exists")
    if db.query(User).filter_by(email=request.email).first():
        raise HTTPException(status_code=400, detail="Email already exists")
    user = User(
        username=request.username,
        email=request.email,
        password_hash=bcrypt.hash(request.password)
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return {"id": user.id, "username": user.username, "email": user.email}