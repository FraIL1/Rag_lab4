import argparse
import time
import uuid
from rag.loaders import read_path
from rag.store import Store

def main():
    parser = argparse.ArgumentParser(description='Индексировать корпус локально без вызовов API')
    parser.add_argument('--source', required=True, choices=['habr', 'science', 'news'])
    parser.add_argument('--path', required=True)
    parser.add_argument('--db', default='data/rag.db')
    args = parser.parse_args()
    store = Store(args.db)
    run = str(uuid.uuid4())
    store.start(run, 'indexing', f'Индексация {args.source}')
    start = time.perf_counter()
    try:
        result = store.ingest(read_path(args.path), args.source, lambda n: print(f'{n} документов', flush=True))
        store.finish(run, 'ok', str(result))
        print(result)
    except Exception:
        store.finish(run, 'error')
        raise
    finally:
        store.cost(run, 'Индексация', model='SQLite FTS5', seconds=time.perf_counter()-start)
    print('Всего в базе:', store.counts())

if __name__ == '__main__':
    main()
