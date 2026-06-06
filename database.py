from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# This creates a local file named "lost_found.db" to store your data
SQLALCHEMY_DATABASE_URL = "sqlite:///./lost_found.db"

# The engine is what actually connects to the database
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()