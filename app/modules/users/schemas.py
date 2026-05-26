from uuid import UUID
from pydantic import BaseModel, EmailStr, ConfigDict

# Token schemas
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenPayload(BaseModel):
    sub: str | None = None

# User schemas
class UserBase(BaseModel):
    email: EmailStr
    is_active: bool = True
    role: str = "user"

class UserCreate(UserBase):
    password: str

class UserResponse(UserBase):
    id: UUID

    model_config = ConfigDict(from_attributes=True)
