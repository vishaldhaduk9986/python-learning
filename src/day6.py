from fastapi import FastAPI, Depends
from sqlmodel import SQLModel, Session, Field, create_engine, select

class Task(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    description: str

sqlite_file = "tasks.db"
engine = create_engine(f"sqlite:///{sqlite_file}", echo=False)

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session

async def lifespan(app: FastAPI):
    create_db_and_tables()
    yield

app = FastAPI(lifespan=lifespan)

@app.post(
    "/task",
    response_model=Task,
    summary="Create a new task",
    response_description="The created task object"
)
def create_task(task: Task, session: Session = Depends(get_session)):
    """
    Create a new task and store it in the database.
    - **Request Body:** Task object (description)
    - **Response:** The created task object
    """
    session.add(task)
    session.commit()
    session.refresh(task)
    return task

@app.get(
    "/tasks",
    response_model=list[Task],
    summary="Get all tasks",
    response_description="List of all tasks"
)
def read_tasks(session: Session = Depends(get_session)):
    """
    Retrieve all tasks from the database.
    - **Response:** List of all task objects
    """
    return session.exec(select(Task)).all()
