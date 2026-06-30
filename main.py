import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from sqlalchemy import create_engine, String, Boolean, Integer
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session, sessionmaker

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql+psycopg://todo:todo_secret@localhost:5432/todos"
)
# Render (and many hosts) provide a "postgres://" URL; normalize it to the
# psycopg v3 driver SQLAlchemy expects.
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg://", 1)
elif DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False)


class Base(DeclarativeBase):
    pass


class TodoModel(Base):
    __tablename__ = "todos"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    done: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(engine)
    yield


app = FastAPI(title="Todo List API", version="1.0.0", lifespan=lifespan)

CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class TodoIn(BaseModel):
    title: str
    description: Optional[str] = None
    done: bool = False


class Todo(TodoIn):
    id: int

    class Config:
        from_attributes = True


@app.get("/")
def root():
    return {"message": "Todo List API", "docs": "/docs"}


@app.get("/todos", response_model=list[Todo])
def list_todos(db: Session = Depends(get_db)):
    return db.query(TodoModel).order_by(TodoModel.id).all()


@app.post("/todos", response_model=Todo, status_code=201)
def create_todo(item: TodoIn, db: Session = Depends(get_db)):
    todo = TodoModel(**item.model_dump())
    db.add(todo)
    db.commit()
    db.refresh(todo)
    return todo


@app.get("/todos/{todo_id}", response_model=Todo)
def get_todo(todo_id: int, db: Session = Depends(get_db)):
    todo = db.get(TodoModel, todo_id)
    if todo is None:
        raise HTTPException(status_code=404, detail="Todo not found")
    return todo


@app.put("/todos/{todo_id}", response_model=Todo)
def update_todo(todo_id: int, item: TodoIn, db: Session = Depends(get_db)):
    todo = db.get(TodoModel, todo_id)
    if todo is None:
        raise HTTPException(status_code=404, detail="Todo not found")
    for key, value in item.model_dump().items():
        setattr(todo, key, value)
    db.commit()
    db.refresh(todo)
    return todo


@app.delete("/todos/{todo_id}", status_code=204)
def delete_todo(todo_id: int, db: Session = Depends(get_db)):
    todo = db.get(TodoModel, todo_id)
    if todo is None:
        raise HTTPException(status_code=404, detail="Todo not found")
    db.delete(todo)
    db.commit()
