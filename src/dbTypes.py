from sqlalchemy import Column, Integer, String, LargeBinary
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class Embedders(Base):

    __tablename__ = 'meta_embedders'

    modelId = Column(Integer, unique=True, nullable=False, primary_key=True, autoincrement=False)
    shape = Column(Integer, nullable=False)
    dtype = Column(String, nullable=False)
    name = Column(String, nullable=False)

    def __repr__(self) -> str:
        return f"Embedder(model={self.modelId}, name={self.name}, dtype={self.dtype}, shape={self.shape})"


class EmbeddingData(Base):

    __tablename__ = 'embeddings'

    id = Column(Integer, primary_key=True)
    fileName = Column(String, unique=True, nullable=False)
    usedModelId = Column(Integer, nullable=False)
    values = Column(LargeBinary, nullable=False)

    def __repr__(self) -> str:
        return f"EmbeddingData(file={self.fileName}, model={self.usedModelId}, values=...)"
