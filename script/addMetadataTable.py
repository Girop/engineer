from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import numpy as np
from dbTypes import Embedders, Base



engine = create_engine('sqlite:///engineer.db')
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()


# bert_embedder = Embedders(modelId=1, shape=768, dtype=str(np.float16), name="Bert")
bert_embedder = Embedders(modelId=2, shape=768, dtype=str(np.float16), name="Bert")
session.add(bert_embedder)
session.commit()

session.close()
