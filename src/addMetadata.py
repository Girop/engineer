import json
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.dbTypes import Base, Metadata
from tqdm import tqdm


def get_line_count():
    lc = 0
    with open("metadata.json") as fp:
        for _ in fp:
            lc += 1
    return lc


def json_gen(batch_size=100):
    with open("metadata.json") as fp:
        batch = []
        for line in fp:
            batch.append(json.loads(line))
            if len(batch) >= batch_size:
                yield batch
                batch = []

        if batch:
            yield batch


if __name__ == '__main__':
    print("Adding metadata to database:")
    pbar = tqdm(total=get_line_count())
    engine = create_engine('sqlite:///engineer.db')
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    for batch in json_gen():
        for doc in batch:
            data = Metadata(
                article_id=doc['id'],
                authors=doc['authors'],
                title=doc['title'],
                comments=doc.get('comments'),
                journal_ref=doc.get('journal-ref'),
                doi=doc.get('doi'),
                report_no=doc['report-no'],
                categories=doc['categories'],
                license=doc.get('license'),
                abstract=doc['abstract'],
                versions=str(doc['versions']),
                update_date=doc["update_date"]
            )
            session.add(data)
            pbar.update(1)
        session.commit()


    session.close()
