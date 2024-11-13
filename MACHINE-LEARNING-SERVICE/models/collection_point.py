from sqlalchemy import Column, Integer, String, DateTime, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from geoalchemy2 import Geometry

Base = declarative_base()

class CollectionPoint(Base):
    """SQLAlchemy model for collection points matching Prisma schema"""
    __tablename__ = 'collection_points'

    id = Column(Integer, primary_key=True, autoincrement=True)
    point_id = Column(Integer, unique=True, nullable=False)
    description = Column(String, nullable=True)
    geom = Column(Geometry('POINT', srid=21037))
    created_at = Column(
        DateTime, 
        server_default=func.now(), 
        nullable=False
    )
    updated_at = Column(
        DateTime,
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )

    # Relationship with Route model (if needed)
    routes = relationship(
        "Route",
        secondary="collection_point_routes",
        back_populates="collection_points"
    )

    def __repr__(self):
        return f"<CollectionPoint(id={self.id}, point_id={self.point_id})>"