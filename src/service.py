import numpy as np
import bentoml
from bentoml.io import NumpyNdarray, JSON
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import jwt
from datetime import datetime, timedelta

# Secret key and algorithm for JWT authentication
JWT_SECRET_KEY = "your_jwt_secret_key_here"
JWT_ALGORITHM = "HS256"

# User credentials for authentication
USERS = {
    "user123": "password123",
    "user456": "password456"
}

class JWTAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if request.url.path == "/v1/models/rf_classifier/predict":
            token = request.headers.get("Authorization")
            if not token:
                return JSONResponse(status_code=401, content={"detail": "Missing authentication token"})

            try:
                token = token.split()[1]  # Remove 'Bearer ' prefix
                payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
            except jwt.ExpiredSignatureError:
                return JSONResponse(status_code=401, content={"detail": "Token has expired"})
            except jwt.InvalidTokenError:
                return JSONResponse(status_code=401, content={"detail": "Invalid token"})

            request.state.user = payload.get("sub")

        response = await call_next(request)
        return response

class InputModel(BaseModel):
    gre_score: int
    toefl_score: int
    university_rating: float
    sop: float
    lor: float
    cgpa: float
    research: float


# Get the model from the Model Store
runner = bentoml.sklearn.get("universities_reg:latest").to_runner()

# Create a service API
reg_service = bentoml.Service("reg_service", runners=[runner])

# Add the JWTAuthMiddleware to the service
reg_service.add_asgi_middleware(JWTAuthMiddleware)

# Create an API endpoint for the service
@reg_service.api(input=JSON(), output=JSON())
def login(credentials: dict) -> dict:
    username = credentials.get("username")
    password = credentials.get("password")

    if username in USERS and USERS[username] == password:
        token = create_jwt_token(username)
        return {"token": token}
    else:
        return JSONResponse(status_code=401, content={"detail": "Invalid credentials"})



@reg_service.api(
    input=JSON(pydantic_model=InputModel),
    output=JSON(),
    route='v1/models/reg/predict'
)
async def classify(input_data: InputModel) -> dict:
    # Convert the input data to a numpy array
    input_series = np.array([input_data.gre_score, input_data.toefl_score, input_data.university_rating, 
                             input_data.sop, 
                             input_data.lor, input_data.cgpa, input_data.research])

    result = await runner.predict.async_run(input_series.reshape(1, -1))

    return {
        "prediction": result.tolist()
    }

# Function to create a JWT token
def create_jwt_token(user_id: str):
    expiration = datetime.utcnow() + timedelta(hours=1)
    payload = {
        "sub": user_id,
        "exp": expiration
    }
    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return token





