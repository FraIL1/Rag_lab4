"""Index already downloaded teaching corpora. Never downloads files."""
from pathlib import Path
import argparse
import subprocess
import sys

ROOT = Path(__file__).parent
parser = argparse.ArgumentParser()
parser.add_argument('--directory', type=Path, default=Path.home() / 'Downloads')
args = parser.parse_args()
corpora = {'habr': '10_000-постов-Хабр.csv.bz2', 'news': '10_000-постов-Лента.csv.bz2', 'science': '10_000-статей-arxiv2.tar.gz'}
for source, name in corpora.items():
    path = args.directory / name
    if not path.exists():
        raise SystemExit(f'Файл не найден: {path}. Укажите папку через --directory.')
for source, name in corpora.items():
    subprocess.run([sys.executable, str(ROOT / 'ingest.py'), '--source', source, '--path', str(args.directory / name), '--db', str(ROOT / 'data/rag.db')], check=True)
