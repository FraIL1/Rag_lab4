import sqlite3
import hashlib
import re
from pathlib import Path

SOURCES = ('habr', 'science', 'news')
STOP = set('что как для или это при чем есть какие какой почему когда где the a an is are of in to and what how'.split())

def chunks(text, size=1800, overlap=250):
    text = re.sub(r'\s+', ' ', text).strip()
    for start in range(0, len(text), size - overlap):
        yield text[start:start + size]
        if start + size >= len(text):
            break

class Store:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = str(path)
        with self.connect() as db:
            db.executescript('''
            CREATE TABLE IF NOT EXISTS documents(id TEXT PRIMARY KEY, source TEXT, title TEXT, url TEXT);
            CREATE VIRTUAL TABLE IF NOT EXISTS passages USING fts5(doc_id UNINDEXED, source UNINDEXED, title, body, tokenize='unicode61');
            CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY, series TEXT, created TEXT DEFAULT CURRENT_TIMESTAMP, question TEXT, status TEXT, answer TEXT);
            CREATE TABLE IF NOT EXISTS costs(run_id TEXT, stage TEXT, model TEXT, input_tokens INTEGER, output_tokens INTEGER, usd REAL, seconds REAL, status TEXT);
            CREATE TABLE IF NOT EXISTS conversations(id TEXT PRIMARY KEY, title TEXT NOT NULL, created TEXT DEFAULT CURRENT_TIMESTAMP, updated TEXT DEFAULT CURRENT_TIMESTAMP);
            CREATE TABLE IF NOT EXISTS messages(id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id TEXT NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL, created TEXT DEFAULT CURRENT_TIMESTAMP);
            CREATE INDEX IF NOT EXISTS messages_conversation ON messages(conversation_id, id);
            ''')
            # Recover successful answers already stored in runs by earlier versions.
            for row in db.execute("SELECT series, question, answer FROM runs WHERE status='ok' AND series<>'indexing' AND answer<>'' ORDER BY created, rowid"):
                if db.execute('SELECT 1 FROM conversations WHERE id=?', (row['series'],)).fetchone():
                    continue
                db.execute('INSERT INTO conversations(id,title) VALUES(?,?)', (row['series'], row['question'][:70]))
                for old in db.execute("SELECT question, answer FROM runs WHERE series=? AND status='ok' AND answer<>'' ORDER BY created, rowid", (row['series'],)):
                    db.execute('INSERT INTO messages(conversation_id,role,content) VALUES(?,?,?)', (row['series'], 'user', old['question']))
                    db.execute('INSERT INTO messages(conversation_id,role,content) VALUES(?,?,?)', (row['series'], 'assistant', old['answer']))

    def connect(self):
        db = sqlite3.connect(self.path, timeout=60)
        db.row_factory = sqlite3.Row
        return db

    def ingest(self, records, source, progress=None):
        if source not in SOURCES:
            raise ValueError('Неизвестный источник')
        added = skipped = 0
        with self.connect() as db:
            for record in records:
                text = str(record.get('text') or '').strip()
                if not text:
                    skipped += 1
                    continue
                title = str(record.get('title') or 'Без названия')
                digest = hashlib.sha256((source + '\0' + title + '\0' + text).encode()).hexdigest()
                if db.execute('SELECT 1 FROM documents WHERE id=?', (digest,)).fetchone():
                    skipped += 1
                    continue
                db.execute('INSERT INTO documents VALUES(?,?,?,?)', (digest, source, title, str(record.get('url') or '')))
                db.executemany('INSERT INTO passages VALUES(?,?,?,?)', ((digest, source, title, chunk) for chunk in chunks(text)))
                added += 1
                if progress and added % 100 == 0:
                    progress(added)
        return {'added': added, 'skipped': skipped}

    def search(self, query, sources, limit=8):
        words = [w for w in re.findall(r'\w+', query.lower()) if len(w) > 2 and w not in STOP][:24]
        if not words:
            return []
        # Prefixes compensate for common Russian endings; the LLM subsequently grades relevance.
        expression = ' OR '.join('"' + (w[:-2] if len(w) > 6 else w) + '"*' for w in words)
        marks = ','.join('?' for _ in sources)
        if not marks:
            return []
        with self.connect() as db:
            rows = db.execute(f'''SELECT passages.rowid AS chunk_id, passages.doc_id, passages.source,
                passages.title, passages.body, documents.url, bm25(passages,0,0,3,1) AS rank
                FROM passages JOIN documents ON documents.id=passages.doc_id
                WHERE passages MATCH ? AND passages.source IN ({marks}) ORDER BY rank LIMIT ?''',
                [expression, *sources, limit * 3]).fetchall()
        result, counts = [], {}
        for row in rows:
            doc = dict(row)
            if counts.get(doc['doc_id'], 0) >= 2:
                continue
            counts[doc['doc_id']] = counts.get(doc['doc_id'], 0) + 1
            result.append(doc)
            if len(result) == limit:
                break
        return result

    def counts(self):
        with self.connect() as db:
            return {r['source']: r['n'] for r in db.execute('SELECT source, COUNT(*) n FROM documents GROUP BY source')}

    def start(self, run_id, series, question):
        with self.connect() as db:
            db.execute('INSERT INTO runs(id,series,question,status) VALUES(?,?,?,?)', (run_id, series, question, 'running'))

    def finish(self, run_id, status, answer=''):
        with self.connect() as db:
            db.execute('UPDATE runs SET status=?,answer=? WHERE id=?', (status, answer, run_id))

    def cost(self, run_id, stage, model='', input_tokens=0, output_tokens=0, usd=0.0, seconds=0.0, status='ok'):
        with self.connect() as db:
            db.execute('INSERT INTO costs VALUES(?,?,?,?,?,?,?,?)', (run_id, stage, model, input_tokens, output_tokens, usd, seconds, status))

    def dashboard(self):
        with self.connect() as db:
            return [dict(r) for r in db.execute('''SELECT runs.series,runs.created,runs.question,runs.status AS run_status,costs.*
                FROM costs JOIN runs ON costs.run_id=runs.id ORDER BY runs.created DESC''')]

    def new_conversation(self, conversation_id):
        with self.connect() as db:
            db.execute('INSERT INTO conversations(id,title) VALUES(?,?)', (conversation_id, 'Новый диалог'))

    def conversations(self):
        with self.connect() as db:
            return [dict(row) for row in db.execute('SELECT id,title FROM conversations ORDER BY updated DESC, rowid DESC')]

    def messages(self, conversation_id):
        with self.connect() as db:
            return [dict(row) for row in db.execute('SELECT role,content FROM messages WHERE conversation_id=? ORDER BY id', (conversation_id,))]

    def add_message(self, conversation_id, role, content):
        with self.connect() as db:
            db.execute('INSERT INTO messages(conversation_id,role,content) VALUES(?,?,?)', (conversation_id, role, content))
            if role == 'user' and db.execute('SELECT COUNT(*) FROM messages WHERE conversation_id=? AND role=?', (conversation_id, 'user')).fetchone()[0] == 1:
                db.execute('UPDATE conversations SET title=? WHERE id=?', (content[:70], conversation_id))
            db.execute('UPDATE conversations SET updated=CURRENT_TIMESTAMP WHERE id=?', (conversation_id,))
