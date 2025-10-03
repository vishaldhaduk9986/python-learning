from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class User(BaseModel):
    name: str
    age: int

users_db = {}

@app.post(
    "/user",
    response_model=User,
    status_code=201,
    summary="Create a new user",
    response_description="The created user object"
)
async def create_user(user: User):
    """
    Create a new user and store in the database.
    - **Request Body:** User object (name, age)
    - **Response:** The created user object
    """
    users_db[user.name] = user
    return user

@app.get(
    "/user/{name}",
    response_model=User,
    summary="Get user by name",
    response_description="The user object if found"
)
async def get_user(name: str):
    """
    Retrieve a user by name (case-insensitive).
    - **Path Parameter:** name (str)
    - **Response:** The user object if found, 404 if not found
    """
    user = next((v for k, v in users_db.items() if k.lower() == name.lower()), None)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user
