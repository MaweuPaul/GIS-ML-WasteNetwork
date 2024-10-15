import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()


DATABASE_URL = os.getenv("DATABASE_URL")  # Format: postgresql://user:password@localhost:5432/yourdb

# Create the SQLAlchemy engine
engine = create_engine(DATABASE_URL, echo=True) 


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)