from pydantic import BaseModel


class TokenPayload(BaseModel):
    sub: str
    company_id: int
    type: str


class LoginRequest(BaseModel):
    employee_code: str
    password: str
    company_id: int


class RefreshRequest(BaseModel):
    refresh_token: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
