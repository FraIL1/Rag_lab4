from pathlib import Path
import io
import os
import time
import uuid
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from rag.store import Store
from rag.loaders import read_stream
from rag.llm import Groq
from rag.graph import build_graph

ROOT = Path(__file__).parent
load_dotenv(ROOT / '.env')
st.set_page_config(page_title='Линза', page_icon='✳', layout='wide')
st.markdown('<style>' + (ROOT / 'style.css').read_text(encoding='utf-8') + '</style>', unsafe_allow_html=True)
store = Store(ROOT / 'data/rag.db')
if 'series' not in st.session_state:
    existing = store.conversations()
    st.session_state.series = existing[0]['id'] if existing else str(uuid.uuid4())
    if not existing:
        store.new_conversation(st.session_state.series)
    st.session_state.messages = store.messages(st.session_state.series)
    st.session_state.trace = []

STAGES = {'plan': 'Уточняю вопрос', 'chat': 'Готовлю ответ', 'retrieve': 'Ищу материалы',
    'rewrite': 'Уточняю поиск', 'generate': 'Собираю ответ',
    'verify': 'Проверяю факты', 'accept': 'Ответ проверен', 'fallback': 'Дополняю ответ',
    'advance': 'Продолжаю поиск', 'combine': 'Готово'}

def show_step(stage, value):
    st.caption(STAGES.get(stage, stage))
    if stage == 'plan':
        for part in value.get('parts', []):
            st.write(part['question'])
    elif stage == 'retrieve':
        docs = value.get('docs', [])
        st.write(f'Найдено фрагментов: {len(docs)}')
        for title in dict.fromkeys(d['title'] for d in docs):
            st.caption(title)
    elif stage == 'verify':
        st.write(value.get('feedback', 'Проверка завершена'))
    elif stage == 'rewrite':
        st.write('Пробую другую формулировку.')

with st.sidebar:
    st.markdown('<div class="brand"><span class="brand-symbol">✳</span>линза</div>', unsafe_allow_html=True)
    if st.button('＋  Новый диалог', width='stretch'):
        st.session_state.series = str(uuid.uuid4())
        store.new_conversation(st.session_state.series)
        st.session_state.messages = []
        st.session_state.trace = []
        st.rerun()
    st.markdown('<div class="side-label">ДИАЛОГИ</div>', unsafe_allow_html=True)
    for conversation in store.conversations()[:30]:
        title = conversation['title'][:28] + ('…' if len(conversation['title']) > 28 else '')
        label = ('◉  ' if conversation['id'] == st.session_state.series else '◯  ') + title
        if st.button(label, key='chat_' + conversation['id'], width='stretch'):
            st.session_state.series = conversation['id']
            st.session_state.messages = store.messages(conversation['id'])
            st.session_state.trace = []
            st.rerun()
    st.markdown('<div class="side-label">НАСТРОЙКИ</div>', unsafe_allow_html=True)
    with st.expander('Подключение'):
        st.text_input('Ключ Groq', type='password', key='groq_key', placeholder='gsk_…')
        st.caption('Ключ действует в текущем сеансе.')
    key = st.session_state.get('groq_key') or os.getenv('GROQ_API_KEY', '')
    st.markdown('<div class="side-note">Место для хороших вопросов.</div>', unsafe_allow_html=True)

st.markdown('<div class="topline"><span class="wordmark">ЛИНЗА / ПРОСТРАНСТВО ПОИСКА</span><span class="availability"><i></i>Ваш личный помощник</span></div>', unsafe_allow_html=True)
chat_tab, data_tab, cost_tab = st.tabs(['Поиск', 'Библиотека', 'Расходы'])

with chat_tab:
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
    empty = not st.session_state.messages
    if empty:
        st.markdown('''<div class="hero"><div class="hero-orbit">✳</div>
<h1>Любопытство —<br><span>хорошее начало.</span></h1>
<p>Задайте вопрос. Найдём главное.</p></div>''', unsafe_allow_html=True)
    question = st.chat_input('О чём хотите узнать?', disabled=not key)
    if empty:
        st.markdown('<div class="suggestion-label">С ЧЕГО НАЧНЁМ?</div>', unsafe_allow_html=True)
        for col, label, prompt in zip(st.columns(3),
                ['↗  Технологии', '✧  Научные открытия', '◷  Истории из новостей'],
                ['Как SoundCloud использует искусственный интеллект?',
                 'Что исследуется в статье A-dependence of hadronization in nuclei?',
                 'Чем закончился суд Шэрон Осборн против The Sun в 2009 году?']):
            if col.button(label, width='stretch', disabled=not key):
                question = prompt
        if not key:
            st.markdown('<div class="quiet">Добавьте ключ в «Подключение», чтобы начать диалог.</div>', unsafe_allow_html=True)
    if question:
        if len(question) > 6000:
            st.error('Слишком длинный вопрос. Попробуйте сократить его.')
            st.stop()
        run_id = str(uuid.uuid4())
        store.start(run_id, st.session_state.series, question)
        history = st.session_state.messages[-6:]
        st.session_state.messages.append({'role': 'user', 'content': question})
        store.add_message(st.session_state.series, 'user', question)
        with st.chat_message('user'):
            st.write(question)
        trace, final = [], {}
        with st.status('Ищу ответ…', expanded=False) as status:
            try:
                graph = build_graph(store, Groq(store, run_id, key=key), run_id)
                for update in graph.stream({'question': question, 'history': history}, stream_mode='updates', config={'recursion_limit': 80}):
                    for stage, value in update.items():
                        trace.append({'stage': stage, 'result': value})
                        final.update(value)
                        show_step(stage, value)
                        status.update(label=STAGES.get(stage, 'Ищу ответ…'))
                answer = final['answer']
                store.finish(run_id, 'ok', answer)
                st.session_state.messages.append({'role': 'assistant', 'content': answer})
                store.add_message(st.session_state.series, 'assistant', answer)
                status.update(label='Ответ готов', state='complete', expanded=False)
            except Exception as exc:
                store.finish(run_id, 'error')
                status.update(label='Не удалось получить ответ', state='error')
                st.error(str(exc))
        st.session_state.trace = trace
        if 'answer' in final:
            st.rerun()
    if st.session_state.trace:
        with st.expander('Ход поиска', expanded=False):
            for item in st.session_state.trace:
                show_step(item['stage'], item['result'])

with data_tab:
    st.markdown('<div class="section-head"><h2>Ваша библиотека</h2><p>Всё, к чему можно вернуться с вопросом.</p></div>', unsafe_allow_html=True)
    source = st.selectbox('Коллекция', ['habr', 'science', 'news'], format_func=lambda s: {'habr': 'Технологии', 'science': 'Наука', 'news': 'Новости'}[s])
    uploaded = st.file_uploader('Добавить материалы', accept_multiple_files=True,
        type=['txt', 'md', 'html', 'csv', 'tsv', 'json', 'jsonl', 'pdf', 'docx', 'zip', 'gz', 'bz2'])
    if st.button('Добавить в библиотеку', disabled=not uploaded):
        run_id = str(uuid.uuid4())
        store.start(run_id, 'indexing', f'Загрузка {len(uploaded)} файлов: {source}')
        started = time.perf_counter()
        try:
            with st.status('Добавляю материалы…', expanded=False) as status:
                added = 0
                for file in uploaded:
                    result = store.ingest(read_stream(io.BytesIO(file.getvalue()), file.name), source)
                    added += result['added']
                status.update(label=f'Добавлено материалов: {added}', state='complete')
            store.finish(run_id, 'ok')
        except Exception as exc:
            store.finish(run_id, 'error')
            st.error(str(exc))
        finally:
            store.cost(run_id, 'Индексация', model='SQLite FTS5', seconds=time.perf_counter()-started)

with cost_tab:
    st.markdown('<div class="section-head"><h2>Расходы</h2><p>Общая картина и детали каждого запроса.</p></div>', unsafe_allow_html=True)
    rows = store.dashboard()
    if rows:
        df = pd.DataFrame(rows)
        ids = list(dict.fromkeys(df['series']))
        labels = {v: ('Библиотека' if v == 'indexing' else ('Текущий диалог' if v == st.session_state.series else f'Диалог {i+1}')) for i, v in enumerate(ids)}
        scope = st.selectbox('Показать', ['all'] + ids, format_func=lambda v: 'Все диалоги' if v == 'all' else labels[v])
        selected = df if scope == 'all' else df[df.series == scope]
        a, b, c = st.columns(3)
        a.metric('Запросы', selected.run_id.nunique())
        b.metric('Токены', f'{selected.input_tokens.sum() + selected.output_tokens.sum():,}'.replace(',', ' '))
        c.metric('Оценка расходов', f'${selected.usd.sum():.4f}')
        st.caption('Расчёт по выбранным тарифам. При бесплатном доступе — $0.')
        names = {'input_tokens': 'Входные токены', 'output_tokens': 'Выходные токены', 'usd': 'USD', 'seconds': 'Время, с'}
        sums = list(names)
        stages = selected.groupby('stage')[sums].sum().rename(columns=names)
        st.bar_chart(stages[['Входные токены', 'Выходные токены']], color=['#a08bd9', '#d9cdef'])
        with st.expander('По этапам'):
            st.dataframe(stages, width='stretch')
        with st.expander('По запросам'):
            st.dataframe(selected.groupby(['run_id', 'question'])[sums].sum().rename(columns=names), width='stretch')
        with st.expander('По диалогам'):
            st.dataframe(df.groupby('series')[sums].sum().rename(index=labels, columns=names), width='stretch')
        with st.expander('История операций'):
            st.dataframe(selected[['created', 'stage', 'input_tokens', 'output_tokens', 'usd', 'seconds', 'status']].rename(columns={**names, 'created': 'Дата', 'stage': 'Этап', 'status': 'Статус'}), width='stretch')
        st.download_button('Скачать отчёт', selected.to_csv(index=False).encode('utf-8-sig'), 'costs.csv', 'text/csv')
    else:
        st.caption('Здесь появится статистика ваших диалогов.')
