import json
import os
import time
import requests

class Groq:
    def __init__(self, store, run_id, key=None):
        self.store, self.run_id = store, run_id
        self.key = (key or os.getenv('GROQ_API_KEY', '')).strip()
        if not self.key:
            raise ValueError('Добавьте GROQ_API_KEY в файл .env и перезапустите приложение.')

    def ask(self, stage, system, payload, schema=None, large=False):
        role = 'LARGE' if large else 'SMALL'
        model = os.getenv(f'{role}_MODEL', 'openai/gpt-oss-120b' if large else 'openai/gpt-oss-20b')
        if schema:
            system += '\nОтветь только JSON, соответствующим схеме: ' + json.dumps(schema.model_json_schema(), ensure_ascii=False)
        body = {'model': model, 'messages': [{'role': 'system', 'content': system},
                {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}],
                'temperature': 0.1}
        if model.startswith('openai/gpt-oss-'):
            body['reasoning_effort'] = 'low'
        if schema:
            body['response_format'] = {'type': 'json_object'}
        for attempt in range(4):
            started = time.perf_counter()
            try:
                response = requests.post('https://api.groq.com/openai/v1/chat/completions',
                    headers={'Authorization': f'Bearer {self.key}'}, json=body, timeout=90)
            except requests.RequestException:
                self.store.cost(self.run_id, stage, model, seconds=time.perf_counter()-started, status='network_error_usage_unknown')
                raise RuntimeError('Не удалось связаться с Groq. Проверьте сеть; расход этого вызова неизвестен.') from None
            elapsed = time.perf_counter() - started
            if response.status_code != 200:
                self.store.cost(self.run_id, stage, model, seconds=elapsed, status=f'http_{response.status_code}')
                if response.status_code == 429:
                    try:
                        delay = float(response.headers.get('retry-after', '0'))
                    except ValueError:
                        delay = 0
                    if attempt < 3 and 0 < delay < 45:
                        time.sleep(delay * 1.5 + 1)
                        continue
                    try:
                        detail = response.json().get('error', {}).get('message', '')
                    except ValueError:
                        detail = ''
                    wait = f' Повторите примерно через {delay:.0f} с.' if delay > 0 else ''
                    raise RuntimeError(f'Groq вернул 429: {detail or "превышен лимит запросов или токенов"}.{wait} '
                                       'Проверьте раздел Limits в console.groq.com/settings/limits.')
                raise RuntimeError(f'Groq вернул HTTP {response.status_code}. Проверьте ключ и названия моделей в .env.')
            result = response.json()
            usage = result.get('usage', {})
            incoming, outgoing = usage.get('prompt_tokens', 0), usage.get('completion_tokens', 0)
            price = (incoming * float(os.getenv(f'{role}_INPUT_PRICE', '0')) + outgoing * float(os.getenv(f'{role}_OUTPUT_PRICE', '0'))) / 1_000_000
            choice = result['choices'][0]
            truncated = choice.get('finish_reason') == 'length'
            self.store.cost(self.run_id, stage, model, incoming, outgoing, price, elapsed,
                            'truncated' if truncated else ('ok' if usage else 'usage_missing'))
            if truncated:
                raise RuntimeError('Groq оборвал генерацию по лимиту модели. Ответ не завершён; расход записан.')
            content = choice['message']['content']
            if schema:
                try:
                    return schema.model_validate_json(content)
                except (ValueError, TypeError):
                    if attempt == 0:
                        body['messages'].append({'role': 'user', 'content': 'Предыдущий JSON не прошёл проверку. Верни корректный JSON строго по схеме.'})
                        continue
                    raise RuntimeError(f'Модель вернула неверный формат на этапе {stage}. Расход записан.') from None
            return content
        raise RuntimeError('Не удалось получить ответ модели.')
