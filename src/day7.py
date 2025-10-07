from fastapi import FastAPI, HTTPException, Depends
from sqlmodel import SQLModel, Session, Field, create_engine, select
from typing import Optional, List


class Book(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    title: str
    author: str


class BookUpdate(SQLModel):
    title: Optional[str] = None
    author: Optional[str] = None

sqlite_file = "books.db"
engine = create_engine(f"sqlite:///{sqlite_file}", echo=True)

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session

async def lifespan(app: FastAPI):
    create_db_and_tables()
    yield

app = FastAPI(lifespan=lifespan)

# --- ROUTES ---

@app.get(
    "/books",
    response_model=List[Book],
    summary="Get all books",
    response_description="List of all books"
)
def get_books(session: Session = Depends(get_session)):
    """
    Retrieve all books from the database.
    - **Response:** List of all book objects
    """
    return session.exec(select(Book)).all()

@app.post(
    "/books",
    response_model=Book,
    status_code=201,
    summary="Add a new book",
    response_description="The created book object"
)
def add_book(book: Book, session: Session = Depends(get_session)):
    """
    Add a new book to the database.
    - **Request Body:** Book object (title, author)
    - **Response:** The created book object
    """
    session.add(book)
    session.commit()
    session.refresh(book)
    return book

@app.delete(
    "/books/{id}",
    response_model=Book,
    summary="Delete a book",
    response_description="The deleted book object"
)
def delete_book(id: int, session: Session = Depends(get_session)):
    """
    Delete a book by its ID.
    - **Path Parameter:** id (int)
    - **Response:** The deleted book object, 404 if not found
    """
    book = session.get(Book, id)
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")
    deleted_book = book
    session.delete(book)
    session.commit()
    return deleted_book

@app.patch(
    "/books/{id}",
    response_model=Book,
    summary="Update a book",
    response_description="The updated book object"
)
def update_book(id: int, book_update: BookUpdate, session: Session = Depends(get_session)):
    """
    Update a book by its ID.
    - **Path Parameter:** id (int)
    - **Request Body:** BookUpdate object (title, author)
    - **Response:** The updated book object, 404 if not found
    """
    db_book = session.get(Book, id)
    if not db_book:
        raise HTTPException(status_code=404, detail="Book not found")
    update_data = book_update.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_book, key, value)
    session.add(db_book)
    session.commit()
    session.refresh(db_book)
    return db_book

@app.get(
    "/books/{id}",
    response_model=Book,
    summary="Get book by ID",
    response_description="The book object if found"
)
def get_book(id: int, session: Session = Depends(get_session)):
    """
    Retrieve a book by its ID.
    - **Path Parameter:** id (int)
    - **Response:** The book object if found, 404 if not found
    """
    book = session.get(Book, id)
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")
    return book

