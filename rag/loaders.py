"""Streaming CSV/JSONL and safe in-memory archive member readers."""
import bz2
import csv
import gzip
import io
import json
import tarfile
import zipfile
from pathlib import Path
from bs4 import BeautifulSoup

csv.field_size_limit(100_000_000)

def normalize(row):
    def first(*keys):
        for key in keys:
            if row.get(key) is not None and str(row[key]).strip():
                return str(row[key])
        return ''
    body = first('text', 'content', 'body', 'post_html', 'article', 'abstract', 'summary', 'full_text')
    if '<' in body and '>' in body:
        body = BeautifulSoup(body, 'html.parser').get_text(' ', strip=True)
    return {'text': body, 'title': first('title', 'post_title', 'headline', 'name'),
            'url': first('url', 'link') or (f"https://habr.com/ru/articles/{row['post_id']}/" if row.get('post_id') else '')}

def read_stream(stream, name):
    lower = name.lower()
    if lower.endswith('.tar.gz') or lower.endswith('.tgz') or lower.endswith('.tar'):
        with tarfile.open(fileobj=stream, mode='r:*') as archive:
            metadata = {}
            for member in archive.getmembers():
                if member.isfile() and '/json/' in member.name and member.size <= 1_000_000:
                    with archive.extractfile(member) as part:
                        value = json.load(part)
                    if 'meta' in value:
                        metadata[Path(member.name).stem] = value['meta']
            for member in archive:
                if member.isfile() and member.size <= 100_000_000:
                    key = Path(member.name).stem
                    if '/json/' in member.name and key in metadata:
                        continue
                    with archive.extractfile(member) as part:
                        if member.name.endswith('.txt') and key in metadata:
                            meta = metadata[key]
                            yield {'title': meta.get('202', key), 'url': meta.get('url', meta.get('217', '')),
                                   'text': part.read().decode('utf-8', errors='replace')}
                        else:
                            yield from read_stream(io.BytesIO(part.read()), member.name)
        return
    if lower.endswith('.zip'):
        with zipfile.ZipFile(stream) as archive:
            for member in archive.infolist():
                if not member.is_dir() and member.file_size <= 100_000_000:
                    with archive.open(member) as part:
                        yield from read_stream(io.BytesIO(part.read()), member.filename)
        return
    if lower.endswith('.bz2'):
        with bz2.BZ2File(stream) as part:
            yield from read_stream(part, name[:-4])
        return
    if lower.endswith('.gz'):
        with gzip.GzipFile(fileobj=stream) as part:
            yield from read_stream(part, name[:-3])
        return
    if lower.endswith('.pdf'):
        from pypdf import PdfReader
        text = '\n'.join(page.extract_text() or '' for page in PdfReader(stream).pages)
        yield {'title': Path(name).stem, 'text': text, 'url': ''}
        return
    if lower.endswith('.docx'):
        import xml.etree.ElementTree as ET
        with zipfile.ZipFile(stream) as archive:
            root = ET.fromstring(archive.read('word/document.xml'))
            text = ' '.join(el.text or '' for el in root.iter() if el.tag.endswith('}t'))
        yield {'title': Path(name).stem, 'text': text, 'url': ''}
        return
    supported = ('.csv', '.tsv', '.json', '.jsonl', '.txt', '.md', '.html', '.htm')
    if not lower.endswith(supported):
        return
    text_stream = io.TextIOWrapper(stream, encoding='utf-8-sig', errors='replace')
    try:
        if lower.endswith(('.csv', '.tsv')):
            for row in csv.DictReader(text_stream, delimiter='\t' if lower.endswith('.tsv') else ','):
                yield normalize(row)
        elif lower.endswith('.jsonl'):
            for line in text_stream:
                if line.strip():
                    yield normalize(json.loads(line))
        elif lower.endswith('.json'):
            data = json.load(text_stream)
            if isinstance(data, dict):
                data = data.get('documents', data.get('data', [data]))
            for row in data:
                yield normalize(row)
        else:
            content = text_stream.read()
            if lower.endswith(('.html', '.htm')):
                content = BeautifulSoup(content, 'html.parser').get_text(' ', strip=True)
            yield {'title': Path(name).stem, 'text': content, 'url': ''}
    finally:
        text_stream.detach()

def read_path(path):
    path = Path(path)
    if path.is_dir():
        for file in sorted(path.rglob('*')):
            if file.is_file():
                yield from read_path(file)
    else:
        with path.open('rb') as stream:
            yield from read_stream(stream, path.name)
