# App Store Review Analysis API

Сервіс для збору відгуків з Apple App Store, їх обробки та багаторівневого NLP/LLM-аналізу: метрики рейтингів, sentiment-аналіз (VADER, Transformer, LLM), видобування негативних ключових слів/фраз (TF-IDF, spaCy, KeyBERT) та генерація структурованих product-insights за допомогою LLM. Результати доступні через REST API (FastAPI) та візуальний dashboard.

> Проєкт написаний на Python, використовує FastAPI, Pydantic, HuggingFace Transformers, spaCy, KeyBERT, і підтримує два LLM-провайдери — Ollama (локально/self-hosted) та Groq (хмарний API).

---

## Зміст

1. [Project Overview](#1-project-overview)
2. [Features](#2-features)
3. [Architecture / Approach](#3-architecture--approach)
4. [Project Structure](#4-project-structure)
5. [Tech Stack](#5-tech-stack)
6. [Local Setup](#6-local-setup)
7. [Running with Ollama](#7-running-with-ollama-docker-compose)
8. [Running with Groq](#8-running-with-groq)
9. [API Documentation](#9-api-documentation)
10. [Example API Usage](#10-example-api-usage)
11. [Results](#11-results-виведені-файли)
12. [Sentiment Analysis](#12-sentiment-analysis)
13. [Keyword Extraction](#13-keyword-extraction)
14. [LLM Insights](#14-llm-insights)
15. [Docker](#15-docker)
16. [Deployment / Render](#16-deployment--render)
17. [Sample Report](#17-sample-report-illustrative)
18. [Example Output JSON](#18-example-output-json)
19. [Troubleshooting](#19-troubleshooting)
20. [Development Notes](#20-development-notes)

---

## 1. Project Overview

Система вирішує задачу: маючи ID застосунку в Apple App Store, автоматично зібрати пул відгуків та перетворити їх на структуровану аналітику, корисну продуктовій/інженерній команді.

**Вхід:** числовий App Store `app_id`, код країни сторефронту (`country`), бажана кількість відгуків (`limit`).

**Вихід:**
- агреговані метрики рейтингів (середній рейтинг, розподіл 1–5 зірок);
- sentiment-мітки відгуків за трьома незалежними методами;
- ранжовані негативні ключові слова та фрази (3 методи екстракції);
- LLM-згенерований звіт з проблемними зонами, доказами (цитатами з відгуків), впливом на користувачів та рекомендаціями;
- набір PNG-візуалізацій та HTML-dashboard;
- сирі відгуки, доступні для завантаження окремо.

**Pipeline (фактичний, за кодом `collect_reviews.py` / `api.py`):**

```
FetchLayer API (App Store reviews)
        ↓
review.json (сирі дані)
        ↓
Loading + Validation + Cleaning   (processing/loader.py, cleaner.py)
        ↓
Rating Metrics                    (processing/metrics.py)
        ↓
Sentiment Analysis (3 незалежні методи)
   ├── VADER            (processing/sentiment.py)
   ├── Transformer       (processing/transformer_sentiment.py)
   └── LLM (Ollama/Groq) (processing/llm_sentiment.py)
        ↓
Негативні відгуки (label == "negative", за Transformer-міткою)
        ↓
Keyword / Phrase Extraction (3 незалежні методи)
   ├── TF-IDF            (processing/keywords.py)
   ├── spaCy POS          (processing/spacy_keywords.py)
   └── KeyBERT            (processing/keybert_keywords.py)
        ↓
LLM Insights (Ollama/Groq)         (processing/llm_insights.py)
        ↓
results/*.json + review.json      (processing/results.py)
        ↓
FastAPI REST endpoints + PNG-візуалізації + /dashboard   (api.py)
```

---

## 2. Features

Реалізовані та реально присутні в коді можливості:

- Збір відгуків з App Store через сторонній API **FetchLayer** (`collector/fetchlayer_client.py`).
- Валідація та очищення сирих відгуків: перевірка обов'язкових полів, діапазону рейтингу 1–5, HTML/URL-очищення тексту (`processing/loader.py`, `processing/cleaner.py`).
- Розрахунок агрегованих метрик рейтингу (`processing/metrics.py`).
- Sentiment-аналіз трьома незалежними методами:
  - VADER (лексиконний, лише англійська) — `processing/sentiment.py`;
  - мультимовна Transformer-модель — `processing/transformer_sentiment.py`;
  - LLM-based sentiment (Ollama/Groq, по одному відгуку) — `processing/llm_sentiment.py`.
- Видобування негативних ключових слів і фраз трьома методами:
  - TF-IDF n-грами — `processing/keywords.py`;
  - spaCy POS-патерни — `processing/spacy_keywords.py`;
  - KeyBERT (мультимовні sentence-embeddings) — `processing/keybert_keywords.py`.
- LLM-генерація структурованого звіту з product-insights (`processing/llm_insights.py`), з валідацією схеми через Pydantic та retry-логікою.
- REST API на FastAPI з ендпоінтами для запуску пайплайну, отримання результатів, keyword-звітів, сирих даних та PNG-візуалізацій (`api.py`).
- HTML dashboard (`GET /dashboard`), що відображає всі візуалізації на одній сторінці.
- Docker-підтримка: окремий `Dockerfile` для API-сервісу та окремий `Dockerfile`/`start.sh` для Ollama-сервісу.
- `docker-compose.yml` для локального запуску API + Ollama разом.
- Вибір LLM-провайдера (`ollama` / `groq`) через змінну середовища `LLM_PROVIDER`, окремо для sentiment-шару та insights-шару.
- Конфігурація деплою на Render (`render.yaml`).
- CLI-orchestrator (`collect_reviews.py`) як альтернативний спосіб запустити весь пайплайн без HTTP-шару.

---

## 3. Architecture / Approach

### Чому окремий collector-шар

`collector/fetchlayer_client.py` інкапсулює єдину відповідальність — отримати сирі відгуки з зовнішнього джерела (FetchLayer API) і повернути список валідованих Pydantic-моделей `Review`. Решта пайплайну (`processing/*`) працює виключно зі списком `dict`/`Review`, не знаючи, звідки дані взялися. Це відповідає принципу з ТЗ:

```python
reviews = review_client.get_reviews(app_id=app_id, limit=100)
```

Завдяки цьому джерело даних можна замінити (наприклад, на інший скрейпер чи RSS), не чіпаючи `processing`, `metrics`, `sentiment` чи `api.py`.

> У `collector/__init__.py` також імпортуються `AppStoreReviewClient` (з `apple_client.py`), `AppleStoreScraper` (з `scraper.py`) та базовий інтерфейс `ReviewClient` (з `review_client.py`) — тобто в пакеті `collector` передбачена абстракція `ReviewClient` і, ймовірно, альтернативна реалізація збору напряму з Apple. Файли цих модулів не входили до наданого для аналізу коду, тому їх внутрішня логіка в цьому README не описується — фактично використовуваний у `api.py` та `collect_reviews.py` клієнт це **`FetchLayerReviewClient`**.

### Чому сирі відгуки зберігаються окремо (`review.json`)

`review.json` — це checkpoint між збором даних і обробкою. Це дозволяє:
- повторно запускати аналіз без повторного виклику платного зовнішнього API (`POST /api/reviews/analyze` та `--skip-collection` в CLI);
- незалежно віддавати сирі дані користувачу (`GET /api/reviews/download`).

### Чому є окреме поле `clean_review`

Оригінальний `review` зберігається без змін (для показу користувачу /аудиту), а `clean_review` (HTML/URL прибрані, нормалізований Unicode, згорнуті пробіли) — це те, що фактично подається у всі sentiment- та keyword-моделі. Розділення "сирий текст для людини" / "чистий текст для моделі" — стандартна NLP-практика.

### Чому три sentiment-методи, а не один

- **VADER** — швидкий, детермінований, безкоштовний бейзлайн, але лексикон лише англійською.
- **Transformer** (`tabularisai/multilingual-sentiment-analysis`) — мультимовна модель (в докстрінгах коду прямо вказано: підтримує українську, російську, англійську), тому саме вона обрана як "джерело правди" для фільтрації негативних відгуків при keyword-екстракції та LLM-insights (`sentiment_field="sentiment_transformer"` — дефолт у відповідних функціях).
- **LLM sentiment** — третій, незалежний сигнал з поясненням через score/label, корисний для порівняння з двома детермінованими методами.

Усі три повертають нормалізовану 3-класову мітку (`positive`/`neutral`/`negative`), що дозволяє порівнювати їх side-by-side (саме для цього в `results.py` є `label_distribution` для кожного методу окремо).

### Чому три методи keyword-екстракції

- **TF-IDF** (`processing/keywords.py`) — статистичний, швидкий, добре працює на великому корпусі, без потреби у мовних моделях.
- **spaCy POS** (`processing/spacy_keywords.py`) — лінгвістично обґрунтована екстракція за граматичними патернами (ADJ+NOUN, NOUN+NOUN тощо), але лише англійська модель (`en_core_web_sm`).
- **KeyBERT** (`processing/keybert_keywords.py`) — семантична екстракція на мультимовних embeddings (`paraphrase-multilingual-MiniLM-L12-v2`), не залежить від точних граматичних правил.

Три підходи дають взаємодоповнюючу картину (статистика / граматика / семантика) і дозволяють порівнювати, наскільки узгоджені результати між методами.

### Чому результати зберігаються у JSON

`processing/results.py` зберігає все у прості `*.json`-файли в `results/` замість БД. Для обсягу даних одного прогону аналізу (сотні відгуків, кілька звітів) файлова система — достатнє й найпростіше рішення (без over-engineering); FastAPI-ендпоінти читають ці файли напряму (`json.load`).

### Чому LLM використовується для insights

Keyword-екстракція дає список термінів, але не відповідь на питання "що з цим робити". `processing/llm_insights.py` перетворює вибірку негативних відгуків на структурований `InsightReport`: групує семантично споріднені скарги в проблемні зони, додає докази (цитати), опис впливу на користувачів і конкретну рекомендацію — тобто виконує саме той ланцюжок `дані → патерн → проблемна зона → рекомендація`, а не просто повертає ще один список слів. Схема виходу валідується Pydantic-моделлю, а промпт прямо забороняє моделі вигадувати факти чи цитати, яких немає у вхідних відгуках.

### Вибір LLM-провайдера

Обидва LLM-шари (`llm_sentiment.py`, `llm_insights.py`) читають змінну середовища `LLM_PROVIDER` (`"ollama"` або `"groq"`) і незалежно один від одного викликають або локальний Ollama-сервер (`ollama` python-пакет, host з `OLLAMA_HOST`), або хмарний Groq API (`groq` python SDK, ключ з `GROQ_API_KEY`). Логіка виклику, парсингу відповіді та обробки помилок ізольована в приватних функціях (`_analyze_with_ollama`/`_analyze_with_groq`, `_call_ollama`/`_call_groq`), а публічний інтерфейс (`analyze_sentiment`, `generate_insight_report`) не залежить від того, який провайдер обрано.

### Чому background task для збору/аналізу

Повний пайплайн (мережевий запит до FetchLayer + завантаження ML-моделей + інференс на сотнях відгуків + LLM-виклики) може тривати десятки секунд і більше. Ендпоінт `POST /api/reviews/{app_id}/collect` та `POST /api/reviews/analyze` використовують `fastapi.BackgroundTasks`, щоб одразу повернути HTTP-відповідь `{"status": "processing"}`, а важку роботу виконати асинхронно в тому ж процесі. Результат потім читається окремим `GET`-запитом (`/api/analysis` тощо), коли пайплайн завершиться.

---

## 4. Project Structure

Структура, реконструйована з наданого коду (шляхи виведені з `import`-виразів у `api.py`/`collect_reviews.py`, а не вигадані):

```
app-store-review-analyzer/
├── collector/
│   ├── __init__.py            # публічний API пакета: FetchLayerReviewClient,
│   │                           # AppStoreReviewClient, AppleStoreScraper, ReviewClient,
│   │                           # Review, набір винятків
│   ├── fetchlayer_client.py    # клієнт до FetchLayer API (реально використовується)
│   ├── models.py                # Pydantic-модель Review
│   ├── exceptions.py            # ReviewCollectionError та підкласи
│   ├── apple_client.py          # присутній у __init__.py, код не входив до аналізу
│   ├── review_client.py         # базовий інтерфейс ReviewClient, код не входив до аналізу
│   └── scraper.py               # AppleStoreScraper, код не входив до аналізу
│
├── processing/
│   ├── __init__.py              # публічний API: load_reviews, clean_review,
│   │                             # compute_metrics, SentimentResult, analyze_sentiment,
│   │                             # attach_sentiment
│   ├── loader.py                # завантаження та валідація review.json
│   ├── cleaner.py                # очищення тексту відгуку (clean_review)
│   ├── metrics.py                # рейтингові метрики
│   ├── sentiment.py              # VADER sentiment
│   ├── transformer_sentiment.py  # Transformer sentiment (multilingual)
│   ├── llm_sentiment.py          # LLM sentiment (Ollama/Groq), per-review
│   ├── keywords.py               # TF-IDF keyword/phrase extraction + спільні стоп-слова
│   ├── spacy_keywords.py         # spaCy POS-based keyword/phrase extraction
│   ├── keybert_keywords.py       # KeyBERT keyword/phrase extraction
│   ├── llm_insights.py           # генерація InsightReport через LLM
│   ├── results.py                # збереження всіх результатів у results/*.json
│   └── visualization.py          # matplotlib PNG-графіки
│
├── ollama/
│   ├── Dockerfile                # окремий образ Ollama-сервісу (FROM ollama/ollama)
│   └── start.sh                  # entrypoint: ollama serve + ollama pull <model>
│
├── results/                      # створюється під час виконання пайплайну (не в git)
├── review.json                   # створюється під час виконання (сирі відгуки)
│
├── api.py                        # FastAPI-застосунок і всі REST-ендпоінти
├── collect_reviews.py            # CLI-orchestrator усього пайплайну (альтернатива API)
├── Dockerfile                    # образ основного API-сервісу
├── docker-compose.yml            # локальний запуск app + ollama
├── render.yaml                   # конфігурація деплою на Render
├── requirements.txt              # Python-залежності
├── .dockerignore
├── .env.example                  # шаблон змінних середовища
└── README.md
```

> У наданому коді немає файлу `main.py`. Точками входу є `api.py` (запускається через `uvicorn api:app`, див. `Dockerfile`) та `collect_reviews.py` (окремий CLI-скрипт, `python collect_reviews.py <app_id>`).

---

## 5. Tech Stack

Реально використані в `requirements.txt` та коді залежності:

| Категорія | Технологія |
|---|---|
| Мова | Python 3.11 (`Dockerfile`: `FROM python:3.11-slim`) |
| Web-фреймворк | FastAPI, Uvicorn (`fastapi`, `uvicorn[standard]`) |
| Валідація даних | Pydantic v2 (`pydantic>=2.0.0`) |
| HTTP-клієнт | `requests` (виклики до FetchLayer) |
| Sentiment (лексиконний) | `vaderSentiment` |
| Sentiment (transformer) | HuggingFace `transformers`, `torch` (модель `tabularisai/multilingual-sentiment-analysis`) |
| Keyword extraction (статистика) | `scikit-learn` (`TfidfVectorizer`), `numpy` |
| Keyword extraction (лінгвістика) | `spacy` (модель `en_core_web_sm`) |
| Keyword extraction (семантика) | `keybert`, `sentence-transformers` (модель `paraphrase-multilingual-MiniLM-L12-v2`) |
| LLM-провайдери | `groq` (Groq Cloud API), `ollama` (self-hosted / локальний сервер) |
| Візуалізація | `matplotlib` (backend `Agg`) |
| Конфігурація | `python-dotenv` |
| Тестування (залежність присутня) | `pytest` |
| Скрейпінг (залежність присутня) | `playwright`, `beautifulsoup4` — присутні в `requirements.txt`, але в наданому коді (`api.py`, `collect_reviews.py`) не викликаються; ймовірно використовуються в `collector/scraper.py`, який не входив до аналізу |
| Контейнеризація | Docker, Docker Compose |
| Деплой | Render (`render.yaml`) |
| Зовнішній API збору даних | FetchLayer (`api.fetchlayer.dev`) |

---

## 6. Local Setup

### Clone

```bash
git clone <repo-url>
cd app-store-review-analyzer
```

### Virtual environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS:**

```bash
python -m venv .venv
source .venv/bin/activate
```

### Dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

(другу команду видно з `Dockerfile` — модель spaCy встановлюється окремою командою, не через `requirements.txt`, оскільки `spacy_keywords.py` завантажує її за іменем `en_core_web_sm` під час першого виклику.)

### Environment variables

Скопіюйте шаблон:

```bash
cp .env.example .env
```

Фактичний вміст `.env.example`:

```env
FETCHLAYER_API_KEY=
GROQ_API_KEY=
HF_TOKEN=


LLM_PROVIDER=ollama
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.1:latest

GROQ_MODEL=openai/gpt-oss-20b
```

Що обов'язково для чого:

| Змінна | Обов'язкова для | Примітка |
|---|---|---|
| `FETCHLAYER_API_KEY` | Будь-якого збору відгуків (`FetchLayerReviewClient`) | Без неї конструктор клієнта одразу кидає `ValueError` |
| `LLM_PROVIDER` | LLM sentiment + LLM insights | `ollama` або `groq`; **дефолти різняться між модулями** — див. розділ 20 |
| `OLLAMA_HOST` | Якщо `LLM_PROVIDER=ollama` | Адреса Ollama-сервера; в Docker Compose перевизначається на `http://ollama:11434` |
| `OLLAMA_MODEL` | Якщо `LLM_PROVIDER=ollama` | Модель для sentiment-шару (дефолт коду: `llama3.2:3b`) і insights-шару (дефолт коду: `llama3.1:latest`), якщо змінна не задана |
| `GROQ_API_KEY` | Якщо `LLM_PROVIDER=groq` | Без нього — `LLMConfigError`/`ValueError` |
| `GROQ_MODEL` | Якщо `LLM_PROVIDER=groq` | Sentiment-шар дефолтить на `openai/gpt-oss-20b`, insights-шар — на `openai/gpt-oss-120b`, якщо змінна не задана |
| `HF_TOKEN` | — | Присутня в `.env.example`, але в наданому коді жодного явного використання (`os.environ["HF_TOKEN"]` тощо) не знайдено — можливо, потрібна для приватних/gated моделей HuggingFace у майбутньому |

### Запуск API локально (без Docker)

```bash
uvicorn api:app --reload --port 8000
```

### Запуск CLI-пайплайну напряму

```bash
python collect_reviews.py <APP_ID> --country us --limit 100
```

або обробка вже наявного `review.json` без нового збору:

```bash
python collect_reviews.py --skip-collection
```

---

## 7. Running with Ollama (Docker Compose)

Проєкт підтримує локальний self-hosted LLM через Ollama, зібраний у `docker-compose.yml`.

```bash
docker compose up -d --build
```

Це піднімає два сервіси:

```
app      — FastAPI-застосунок (порт 8000:8000), env з .env + OLLAMA_HOST=http://ollama:11434
ollama   — офіційний образ ollama/ollama (порт 11434:11434), том ollama_data для персистентності моделей
```

Перевірка статусу:

```bash
docker compose ps
```

Ollama-сервер стартує без попередньо завантаженої моделі — модель треба завантажити вручну:

```bash
docker compose exec ollama ollama pull llama3.1:latest
```

(або будь-яку іншу модель, вказану у `OLLAMA_MODEL`/`GROQ_MODEL`-еквіваленті для sentiment-шару, наприклад `llama3.2:3b`).

Перевірити, які моделі вже завантажені:

```bash
docker compose exec ollama ollama list
```

**Важливо:** всередині Docker Compose-мережі застосунок звертається до Ollama за іменем сервіса — `http://ollama:11434`, а не `http://localhost:11434`. Це явно перевизначено в `docker-compose.yml` (`environment: OLLAMA_HOST: http://ollama:11434`), незалежно від того, що прописано в `.env`.

---

## 8. Running with Groq

Замість локального Ollama можна використати хмарний Groq API — просто змінивши середовище, без змін коду:

```env
LLM_PROVIDER=groq
GROQ_API_KEY=<ваш ключ>
GROQ_MODEL=openai/gpt-oss-20b
```

Різниця між провайдерами:

| | Ollama | Groq |
|---|---|---|
| Розташування | Локально / self-hosted сервер | Хмарний API |
| Потрібен ключ | Ні | Так (`GROQ_API_KEY`) |
| Вартість | Безкоштовно (свій compute) | За токенами, зовнішній rate limit |
| Швидкість/якість | Залежить від локального заліза та обраної моделі | Стабільна швидкість інференсу, хмарні моделі |
| Обробка помилок у коді | `ConnectionError`/`TimeoutError`/`OSError` → `ProviderError`/`LLMRequestError` | `RateLimitError`/`APITimeoutError`/`APIConnectionError`/`APIStatusError` → ті самі типи винятків |

Дефолтні моделі, зашиті в коді (використовуються, якщо `GROQ_MODEL`/`OLLAMA_MODEL` не задані):

- `llm_sentiment.py`: Groq — `openai/gpt-oss-20b`, Ollama — `llama3.2:3b`.
- `llm_insights.py`: Groq — `openai/gpt-oss-120b`, Ollama — `llama3.1:latest`.

---

## 9. API Documentation

Базова адреса локально: `http://localhost:8000`.

### `POST /api/reviews/{app_id}/collect`

Запускає повний пайплайн (збір → обробка → аналіз → збереження) у фоновому режимі.

- **Path params:** `app_id` (`str`, regex `^\d+$` — лише цифри).
- **Request body** (`CollectRequest`):
  ```json
  { "country": "us", "limit": 100 }
  ```
  `country` — рядок довжиною 2 символи (дефолт `"us"`), `limit` — ціле число `> 0` (дефолт `100`).
- **Response 200:**
  ```json
  {
    "status": "processing",
    "message": "Collection and analysis started for app_id ... in background.",
    "details": "Country: us, Limit: 100"
  }
  ```
- Фінальний результат пайплайну сюди **не повертається** — його треба отримувати окремим `GET /api/analysis` після завершення фонової задачі.
- Помилки всередині фонової задачі (наприклад, невірний `app_id`, відсутність відгуків, збій LLM) лише логуються (`logging.exception`) і **не** повертаються клієнту — див. розділ Troubleshooting.

### `POST /api/reviews/{app_id}/fetch`

Тільки збирає відгуки та зберігає їх у `review.json`, **без** запуску аналізу.

- **Path params:** `app_id` (`^\d+$`).
- **Request body:** `CollectRequest` (той самий, що вище).
- **Response 200 (є дані):**
  ```json
  {
    "status": "success",
    "app_id": "368677368",
    "country": "us",
    "count": 100,
    "reviews": [ { "review_id": "...", "title": "...", "review": "...", "rating": 5, "author": "...", "date": "2026-01-15" } ],
    "saved_to": "/app/review.json"
  }
  ```
- **Response 200 (нуль відгуків):** те саме без `reviews`/`saved_to`, `count: 0`.
- **Response 500:** при будь-якому винятку в клієнті/API — `{"detail": "Failed to fetch reviews: ..."}`.

### `POST /api/reviews/analyze`

Запускає аналіз уже наявного `review.json` (без нового збору) у фоновому режимі.

- **Response 404:** якщо `review.json` не існує — `"review.json not found. Collect reviews first."`
- **Response 200:** `{"status": "processing", "message": "Analysis of existing reviews started in background."}`

### `GET /api/analysis`

Повертає повний зведений звіт (`results/analysis.json`).

- **Response 404:** якщо файл ще не створений — `"Analysis not found. Please run the collection endpoint first and wait for it to finish."`
- **Response 200:** вміст `analysis.json` (див. розділ 18).

### `GET /api/keywords/{method}`

Повертає повний звіт по одному з методів keyword-екстракції.

- **Path params:** `method` — один із `tfidf`, `spacy`, `keybert`.
- **Response 400:** якщо метод не з переліку.
- **Response 404:** якщо файл `negative_keywords_{method}.json` відсутній.
- **Response 200:** вміст відповідного `NegativeTermsReport`.

### `GET /api/reviews/download`

Віддає сирий `review.json` як файл для завантаження.

- **Response 404:** якщо файл відсутній.
- **Response 200:** `FileResponse`, `media_type=application/json`, `filename=review.json`.

### `GET /api/visualizations/rating-distribution`

PNG bar chart розподілу оцінок 1–5 (`processing/visualization.render_rating_distribution`).

- **Response 404:** якщо `metrics.json` відсутній.
- **Response 200:** `image/png`.

### `GET /api/visualizations/sentiment-distribution`

PNG grouped bar chart порівняння sentiment-міток між VADER / Transformer / LLM.

- **Response 404:** якщо `analysis.json` відсутній.
- **Response 200:** `image/png`.

### `GET /api/visualizations/sentiment-by-rating`

PNG stacked bar chart: розподіл sentiment (за Transformer-міткою) всередині кожної оцінки 1–5.

- **Response 404:** якщо `sentiment_transformer.json` відсутній.
- **Response 200:** `image/png`.

### `GET /api/visualizations/top-negative-terms`

PNG horizontal bar chart топ-N (15) термінів.

- **Query params:** `method` (`tfidf` | `keybert` | `spacy-pos`, дефолт `tfidf`), `kind` (`keywords` | `phrases`, дефолт `keywords`).
- **Response 400:** невалідний `method` або `kind`.
- **Response 404:** якщо відповідний `negative_keywords_*.json` відсутній.
- **Response 200:** `image/png`.

### `GET /dashboard`

HTML-сторінка, що вбудовує всі п'ять PNG-візуалізацій вище через `<img src="...">`. Не залежить від `analysis.json` напряму — кожен `<img>` сам зверне запит до відповідного ендпоінта і сам поверне 404 як зображення, якщо дані ще не готові.

> **Health check:** ендпоінта `/health`, зазначеного в `render.yaml` (`healthCheckPath: /health`), у `api.py` **не реалізовано**. Див. розділ Troubleshooting/Deployment Notes.

---

## 10. Example API Usage

Запуск повного пайплайну для App Store ID:

```bash
curl -X POST \
  http://localhost:8000/api/reviews/368677368/collect \
  -H "Content-Type: application/json" \
  -d '{"country":"us","limit":100}'
```

Відповідь одразу (пайплайн ще виконується у фоні):

```json
{
  "status": "processing",
  "message": "Collection and analysis started for app_id 368677368 in background.",
  "details": "Country: us, Limit: 100"
}
```

Через деякий час (коли фонова задача завершиться) забрати результат:

```bash
curl http://localhost:8000/api/analysis
```

Якщо запит зроблено занадто рано — прийде `404 Analysis not found. Please run the collection endpoint first and wait for it to finish.` — потрібно повторити пізніше.

Отримати лише результати одного keyword-методу:

```bash
curl http://localhost:8000/api/keywords/keybert
```

Завантажити сирі відгуки:

```bash
curl -O http://localhost:8000/api/reviews/download
```

Отримати PNG-графік:

```bash
curl http://localhost:8000/api/visualizations/top-negative-terms?method=spacy-pos&kind=phrases -o phrases.png
```

---

## 11. Results (виведені файли)

`processing/results.py:save_pipeline_results()` зберігає результати одного прогону в директорію `results/` (створюється автоматично, якщо відсутня):

| Файл | Зміст |
|---|---|
| `metrics.json` | Вихід `compute_metrics()`: `total`, `valid_count`, `invalid_count`, `average_rating`, `rating_counts`, `rating_distribution` |
| `sentiment_vader.json` | Per-review експорт (`index`, `rating`, `title`, `review`, `clean_review`, `sentiment`) для VADER |
| `sentiment_transformer.json` | Те саме для Transformer-sentiment (поле `sentiment_transformer`) |
| `sentiment_llm.json` | Те саме для LLM-sentiment (поле `sentiment_llm`) |
| `negative_keywords_tfidf.json` | Повний `NegativeTermsReport` з TF-IDF |
| `negative_keywords_spacy.json` | Повний `NegativeTermsReport` зі spaCy |
| `negative_keywords_keybert.json` | Повний `NegativeTermsReport` з KeyBERT |
| `insights.json` | Повний `InsightReport` від LLM |
| `analysis.json` | Зведений звіт: `metrics`, `sentiment_distribution` (по 3 методах), `llm_sentiment_errors`, `top_negative_keywords`/`top_negative_phrases` (топ-5 на метод), `insights_summary` |

Окремо, поза `results/`, у корені проєкту зберігається `review.json` — сирі валідовані (Pydantic `Review`) відгуки, отримані з FetchLayer, до будь-якої NLP-обробки. Саме цей файл віддається через `GET /api/reviews/download` і використовується `POST /api/reviews/analyze`/`--skip-collection`.

Усі API-ендпоінти для читання результатів (`/api/analysis`, `/api/keywords/{method}`, всі `/api/visualizations/*`) читають ці файли напряму з диска — окремої БД немає.

---

## 12. Sentiment Analysis

### VADER (`processing/sentiment.py`)

- **Бібліотека:** `vaderSentiment.SentimentIntensityAnalyzer`, лексикон лише англійською.
- **Вхід:** `clean_review` (рядок).
- **Вихід (`SentimentResult`):** `label` (`positive`/`neutral`/`negative`), `compound`, `pos`, `neu`, `neg`, `method="vader-en"`.
- **Мітка:** за стандартними порогами VADER: `compound >= 0.05` → positive, `compound <= -0.05` → negative, інакше neutral.
- **Призначення:** швидкий, детермінований бейзлайн без залежності від ваги моделей.

### Transformer (`processing/transformer_sentiment.py`)

- **Модель:** `tabularisai/multilingual-sentiment-analysis` (HuggingFace `pipeline("text-classification")`, `top_k=None`, `truncation=True`, `max_length=512`), лениво завантажується при першому виклику.
- **Вхід:** `clean_review`.
- **Вихід (`TransformerSentimentResult`):** нормалізований `label` (3 класи), `score` (сумарна впевненість по бакету), `raw_label` (нативна 5-класова мітка моделі), `raw_scores` (повний розподіл по 5 класах), `method="transformer-mdistilbert-tabularisai"`.
- **Мапінг класів:** нативні `Very Negative`/`Negative` → `negative`, `Neutral` → `neutral`, `Positive`/`Very Positive` → `positive`; фінальна мітка — бакет з найбільшою сумою ймовірностей.
- **Призначення:** саме цей метод обраний як дефолтне джерело "чи негативний відгук" для keyword-екстракції та LLM-insights, оскільки модель мультимовна (на відміну від VADER та spaCy).

### LLM Sentiment (`processing/llm_sentiment.py`)

- **Провайдер:** Ollama або Groq, обирається через `LLM_PROVIDER` (в цьому модулі дефолт — `"ollama"`).
- **Вхід:** окремий виклик LLM на кожен відгук (`clean_review`), з системним промптом, що вимагає відповіді ЛИШЕ у форматі JSON `{"label": ..., "score": ...}`.
- **Вихід (`LLMSentimentResult`):** `label` (може бути `None` при помилці), `score` (0.0–1.0), `method` (`llm-ollama`/`llm-groq`, або з суфіксом `-error`), `error` (текст помилки, якщо є).
- **Паралелізм:** `ThreadPoolExecutor(max_workers=4)`; при Groq додатково `time.sleep(0.3)` між результатами для дотримання rate limit.
- **Обробка помилок:** мережеві/API-помилки (`ProviderError`) та непарсибельні відповіді (`MalformedResponseError`) не піднімають виняток на весь пайплайн — конкретний відгук отримує `label=None`, і виключається з `label_distribution`, але враховується в окремому лічильнику `llm_sentiment_errors` (`results.py:count_sentiment_errors`).

---

## 13. Keyword Extraction

Усі три методи працюють над одним і тим самим набором текстів — `extract_negative_texts()` (`processing/keywords.py`), який фільтрує відгуки за `sentiment_transformer.label == "negative"` (дефолтне поле, змінюване через параметр).

### TF-IDF (`processing/keywords.py`)

- `sklearn.TfidfVectorizer`, `ngram_range=(1,3)` — уніграми, біграми, триграми в одному проході.
- Кастомний стоп-список: базовий англійський список sklearn **мінус** слова заперечення (`not`, `no`, `never`, `cannot`...— навмисно залишені, бо "cannot log in" втрачає сенс без "cannot"), **плюс** невеликий generic filler-список (`guys`, `gonna`, `thanks`...) і домен-специфічні слова (`app`, `application`, `music`, `song`, `spotify`).
- `min_df`/`max_df` адаптуються до розміру корпусу (щоб не занулити словник на маленькій кількості негативних відгуків).
- Терміни без пробілу → `keywords`, з пробілом → `phrases`; чисто числові n-грами відкидаються.
- `score` = сума TF-IDF ваги терміна по корпусу; `document_frequency` = кількість відгуків, де термін зустрічається.

### spaCy (`processing/spacy_keywords.py`)

- Модель `en_core_web_sm` (`ner`, `parser` вимкнені — використовуються лише POS-теги та леми).
- Однослівні `keyword`-кандидати: токени з POS `NOUN`/`ADJ`, що пройшли фільтр `_is_meaningful_token` (не стоп-слово, не пунктуація/число, довжина ≥ 2).
- Двослівні `phrase`-кандидати: пари сусідніх токенів з патернами `ADJ+NOUN`, `NOUN+NOUN`, `VERB+NOUN`, `NOUN+VERB`.
- Терміни лематизуються (`crashes`/`crashed` → `crash`), рахуються як множина унікальних термінів на документ (щоб повторення всередині одного відгуку не роздувbřezало document frequency).
- `score` = частка негативних відгуків, що містять термін (`document_frequency / n_docs`).
- Обмеження, задокументоване прямо в коді: лише англійська модель — україномовний/російськомовний текст тегується без помилки, але POS-теги для нього недостовірні.

### KeyBERT (`processing/keybert_keywords.py`)

- Embedding-модель: `paraphrase-multilingual-MiniLM-L12-v2` (мультимовна, обрана заради економії пам'яті/CPU порівняно з більшою `mpnet`).
- Екстракція виконується **на кожному відгуку окремо** (`extract_keywords(docs=[...])`), а не на одному конкатенованому корпусі — це зберігає можливість рахувати `document_frequency` (скільки різних відгуків підняли фразу), а не отримати домінуючу тему одного найдовшого відгуку.
- `use_mmr=True`, `diversity=0.5` — Maximal Marginal Relevance, щоб один відгук не давав кілька майже однакових варіацій фрази.
- `keyphrase_ngram_range=(1,3)`, той самий стоп-список, що й у TF-IDF.
- Ранжування пріоритезує `document_frequency`, потім середній cosine similarity score.

**Чому три методи одразу:** статистичний (TF-IDF), лінгвістичний (spaCy) і семантичний (KeyBERT) підходи вловлюють різні патерни — TF-IDF добре масштабується і не залежить від мовних моделей, spaCy дає граматично цілісні фрази, KeyBERT працює мультимовно і на семантичній подібності, а не точних n-грамах. Порівняння результатів між методами (доступне через `/api/keywords/{method}`) дає ширшу картину, ніж будь-який один із них окремо.

---

## 14. LLM Insights

`processing/llm_insights.py:generate_insight_report()` перетворює вже класифіковані (sentiment) відгуки на структурований `InsightReport`.

**Відбір негативних відгуків:** фільтруються записи з `sentiment_transformer.label == "negative"` (параметризовано, дефолт саме такий), сортуються за зростанням `rating` (найнижчий рейтинг спочатку), і обрізаються до `_MAX_NEGATIVE_REVIEWS = 50`. Це навмисне обмеження — тримає розмір LLM-запиту передбачуваним і, якщо відгуків більше 50, гарантує, що в модель потраплять саме найгостріші скарги (найнижчі оцінки), а не довільна вибірка.

**Побудова payload:** для кожного відібраного відгуку в LLM-запит іде `rating`, `title` (обрізаний до 120 символів), `review` (обрізаний до `_MAX_EVIDENCE_CHARS = 300` символів через `_truncate`), і `sentiment_confidence`. Додатково передається `aggregate_counts` (загальна кількість відгуків, кількість негативних, скільки з них реально пішло в LLM, розбивка по рейтингах серед негативних).

**System prompt (концептуально, без копіювання повного тексту):** модель виступає продуктовим аналітиком; має згрупувати семантично споріднені скарги в невелику кількість проблемних зон, для кожної навести 1–5 майже дослівних цитат **лише** з наданих відгуків, описати вплив на користувачів **лише** на основі цих доказів, дати конкретну рекомендацію, і пріоритезувати проблеми, що повторюються в кількох відгуках (або є одиничними, але критичними — краші, втрата даних, білінг). Явно заборонено вигадувати факти/цифри/цитати, яких немає у вхідних даних; статистику (проценти, середні) модель рахувати не повинна — вона обчислюється окремо в `metrics.py`/`results.py`.

**Формування `InsightReport`:** відповідь LLM парситься як JSON і валідується Pydantic-моделлю `InsightReport { summary: str, insights: list[Insight{problem_area, evidence[1..5], impact, recommendation}], model, method, reviews_analyzed }`. Поле `evidence` в `Insight` описане в коді як "near-verbatim excerpts copied from the supplied reviews" — тобто це контроль проти галюцинацій на рівні схеми: якщо LLM не поверне валідний JSON цієї форми, `Pydantic.ValidationError` перехоплюється і трактується як помилка відповіді.

**Retry:** якщо LLM повернула невалідний за схемою JSON (`LLMResponseError`), запит повторюється до `_MAX_RETRIES = 3` разів; якщо жодна спроба не пройшла валідацію — піднімається `LLMResponseError` з описом останньої помилки.

**Провайдер і модель:** `LLM_PROVIDER` (дефолт у цьому модулі — `"groq"`) визначає, чи викликати Groq (`_call_groq`, JSON-схема через `response_format={"type": "json_schema", ...}`, `strict=False` — у коментарі коду зазначено, що `strict=True` мав регресії на моделі `gpt-oss-120b`) чи Ollama (`_call_ollama`, схема передається напряму в параметр `format`). Модель визначається за пріоритетом: явний аргумент `model` → змінна середовища (`GROQ_MODEL`/`OLLAMA_MODEL`) → хардкодний дефолт (`openai/gpt-oss-120b` для Groq, `llama3.1:latest` для Ollama).

**Порожній випадок:** якщо відгуків немає взагалі, або немає жодного негативного — LLM не викликається, повертається валідний, але порожній `InsightReport` (`summary` пояснює причину, `insights=[]`, `reviews_analyzed=0`).

---

## 15. Docker

### `Dockerfile` (основний API-сервіс)

- **Base image:** `python:3.11-slim`.
- **Встановлення залежностей:** `pip install -r requirements.txt`, одразу після цього — `python -m spacy download en_core_web_sm` (окрема команда, бо модель spaCy не є pip-пакетом і завантажується за іменем моделі під час білда, а не рантайму).
- **Порт:** `EXPOSE 8000` — інформаційно; реальний порт визначається змінною середовища `$PORT` (важливо для Render, який динамічно призначає порт).
- **Команда запуску:** `CMD ["sh", "-c", "exec uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]` — навмисно **shell form** (не JSON/exec form), щоб `${PORT}` реально підставлявся оболонкою при старті контейнера; якщо `PORT` не заданий — фолбек на `8000`.
- `PYTHONUNBUFFERED=1`, `PYTHONDONTWRITEBYTECODE=1`, `PIP_NO_CACHE_DIR=1` — стандартна гігієна для контейнерних Python-образів (логи одразу в стрім, без `.pyc`, без pip-кешу в шарах образу).

### `docker-compose.yml`

Два сервіси:

```
app ↔ ollama
```

- `app` — білдиться з кореневого `Dockerfile`, читає `.env`, форвардить порт `8000:8000`, залежить від `ollama` (`depends_on`), і **перевизначає** `OLLAMA_HOST=http://ollama:11434` (Docker Compose DNS-ім'я сервіса), незалежно від значення в `.env`.
- `ollama` — офіційний образ `ollama/ollama`, порт `11434:11434`, іменований том `ollama_data:/root/.ollama` — саме цей том забезпечує, що завантажені моделі (`ollama pull ...`) не втрачаються при перестворенні контейнера.

### `ollama/Dockerfile` + `ollama/start.sh`

Окремий, спрощений образ (не той, що описаний вище в `docker-compose.yml` — цей використовується для деплою на Render, див. `render.yaml`):

```dockerfile
FROM ollama/ollama
COPY start.sh /start.sh
RUN chmod +x /start.sh
ENTRYPOINT ["/start.sh"]
```

```sh
ollama serve &
sleep 5
ollama pull your-model-name
wait
```

> **Знайдена проблема:** `ollama pull your-model-name` містить буквальний плейсхолдер `your-model-name`, а не реальну назву моделі. У поточному вигляді цей entrypoint спробує завантажити неіснуючу модель `your-model-name` і зазнає невдачі. Перед використанням цього образу рядок треба замінити на реальний тег моделі (наприклад, `llama3.1:latest`, узгоджений з `OLLAMA_MODEL` в `render.yaml`).

---

## 16. Deployment / Render

`render.yaml` описує два сервіси:

### `appstore-review-analysis-api` (web service)

- `env: docker`, `dockerfilePath: ./Dockerfile`, `dockerContext: .` — використовує кореневий `Dockerfile` (той, що описаний у розділі 15).
- `plan: standard`.
- `healthCheckPath: /health`.
- Env vars:
  - `FETCHLAYER_API_KEY` — `sync: false` (треба задати вручну в Render dashboard, не в git).
  - `LLM_PROVIDER=ollama` — захардкоджено значенням у самому `render.yaml`.
  - `OLLAMA_HOST=http://ollama:11434` — захардкоджено; передбачає, що приватний сервіс `ollama` доступний за цим внутрішнім DNS-ім'ям в мережі Render.
  - `OLLAMA_MODEL` — `sync: false` (задається вручну).
  - `PYTHONUNBUFFERED=1`.

### `ollama` (private service, `pserv`)

- `env: docker`, `dockerfilePath: ./ollama/Dockerfile`, `dockerContext: ./ollama` — саме той образ з плейсхолдером моделі, описаний у розділі 15.
- `plan: standard`.
- Порти, health check і персистентний диск для цього сервісу в `render.yaml` **не задані**.

### Deployment Notes (знайдені прогалини/невідповідності)

- **`/health` не реалізовано.** У `api.py` немає ендпоінта `GET /health`, хоча `healthCheckPath: /health` вказаний для web-сервісу в `render.yaml`. Без нього Render-health-check для цього сервісу працюватиме некоректно (звертатиметься на неіснуючий шлях → FastAPI поверне `404`, що Render, ймовірно, інтерпретує як "unhealthy").
- **Плейсхолдер моделі.** `ollama/start.sh` тягне `your-model-name` замість реальної моделі (див. розділ 15) — деплой `ollama`-сервіса в поточному вигляді не завантажить робочу модель без ручного виправлення файлу.
- **Персистентність моделей на Render не налаштована.** На відміну від `docker-compose.yml` (том `ollama_data`), у `render.yaml` для `pserv`-сервіса `ollama` жодного персистентного диска не описано — тобто після кожного передеплою/рестарту сервіса завантажену модель, ймовірно, доведеться тягнути заново (в межах того, що видно з наданого `render.yaml`).
- **Groq як альтернатива на Render.** `render.yaml` конфігурує лише `LLM_PROVIDER=ollama`. Перехід на Groq у продакшені технічно можливий (код це підтримує — див. розділ 8), але вимагає ручної зміни `LLM_PROVIDER` та додавання `GROQ_API_KEY`/`GROQ_MODEL` в Render dashboard — у наданому `render.yaml` цей шлях не сконфігурований.

---

## 17. Sample Report (illustrative)

> Нижче — **ілюстративний**, синтетичний приклад того, як міг би виглядати звіт після реального запуску пайплайну. Значення нижче **вигадані для демонстрації формату** й не є результатом реального виклику FetchLayer/LLM. (Обраний застосунок — умовний музичний стрімінговий сервіс: у `processing/keywords.py` домен-специфічний стоп-список прямо містить слово `"spotify"`, що натякає, на якому типі застосунку тестувався проєкт; реальних даних Spotify в цьому README не використано.)

### Executive Summary

За вибіркою зі 100 відгуків (умовно) середній рейтинг склав **3.4 / 5**. LLM-based sentiment позначив **41%** відгуків як негативні, Transformer-модель — **37%**, VADER (лише англомовна частина корпусу) — **33%**. Основні скарги концентруються навколо стабільності застосунку та поведінки offline-режиму.

### Sentiment Distribution (ілюстративно)

| Метод | positive | neutral | negative |
|---|---|---|---|
| VADER | 44 | 23 | 33 |
| Transformer | 41 | 22 | 37 |
| LLM | 39 | 20 | 41 |

### Key Problems

1. **App crashes on playback resume**
   - Evidence: *"crashes every time I try to resume a podcast after a call"*, *"app closes itself mid-song randomly"*.
   - Impact: Втрата контексту прослуховування, повторні спроби запуску, зниження довіри до стабільності застосунку.
   - Recommendation: Пріоритезувати crash-репорти навколо lifecycle audio-сесії (resume після переривань дзвінком/сповіщенням) як P0-баг.

2. **Offline downloads disappear after update**
   - Evidence: *"downloaded playlists gone after the last update"*.
   - Impact: Користувачі з обмеженим трафіком/офлайн-сценаріями втрачають збережений контент без попередження.
   - Recommendation: Додати міграцію/збереження локального кешу завантажень при оновленнях застосунку та явне попередження, якщо кеш буде очищено.

3. **Payment/subscription confusion**
   - Evidence: *"charged twice this month for premium"*.
   - Impact: Фінансова недовіра, потенційний відтік підписників, збільшення навантаження на підтримку.
   - Recommendation: Аудит idempotency біллінгових webhook-ів та явний екран історії списань у застосунку.

### Top Negative Keywords / Phrases (ілюстративно)

`crash`, `offline`, `subscription`, `battery`, `ads` / `keeps crashing`, `lost playlist`, `double charge`.

---

## 18. Example Output JSON

Скорочений, але структурно точний приклад `analysis.json` (за схемою `build_analysis_summary` у `processing/results.py`):

```json
{
  "metrics": {
    "total": 100,
    "valid_count": 98,
    "invalid_count": 2,
    "average_rating": 3.42,
    "rating_counts": {"1": 20, "2": 10, "3": 15, "4": 18, "5": 35},
    "rating_distribution": {"1": 20.41, "2": 10.2, "3": 15.31, "4": 18.37, "5": 35.71}
  },
  "sentiment_distribution": {
    "vader": {"positive": 44, "neutral": 23, "negative": 31},
    "transformer": {"positive": 41, "neutral": 22, "negative": 35},
    "llm": {"positive": 39, "neutral": 20, "negative": 39}
  },
  "llm_sentiment_errors": 0,
  "top_negative_keywords": {
    "tfidf": [
      {"term": "crash", "score": 4.21, "document_frequency": 12}
    ],
    "spacy": [
      {"term": "crash", "score": 0.31, "document_frequency": 12}
    ],
    "keybert": [
      {"term": "crash", "score": 0.62, "document_frequency": 9}
    ]
  },
  "top_negative_phrases": {
    "tfidf": [
      {"term": "keeps crashing", "score": 2.11, "document_frequency": 6}
    ],
    "spacy": [
      {"term": "keep crash", "score": 0.15, "document_frequency": 6}
    ],
    "keybert": [
      {"term": "app keeps crashing", "score": 0.58, "document_frequency": 5}
    ]
  },
  "insights_summary": {
    "summary": "Negative reviews cluster around stability issues and offline content loss.",
    "problem_areas": ["App crashes on playback resume", "Offline downloads disappear after update"],
    "reviews_analyzed": 37,
    "model": "llama3.1:latest"
  }
}
```

Приклад повного `InsightReport` (`results/insights.json`), за Pydantic-схемою з `processing/llm_insights.py`:

```json
{
  "summary": "Two recurring stability issues dominate negative feedback...",
  "insights": [
    {
      "problem_area": "App crashes on playback resume",
      "evidence": [
        "crashes every time I try to resume a podcast after a call"
      ],
      "impact": "Users lose listening context and lose trust in app stability.",
      "recommendation": "Prioritize crash investigation around audio session resume after interruptions."
    }
  ],
  "model": "llama3.1:latest",
  "method": "llm-ollama",
  "reviews_analyzed": 37
}
```

---

## 19. Troubleshooting

| Симптом | Ймовірна причина | Що перевірити |
|---|---|---|
| `POST /api/reviews/{app_id}/collect` повертає `200`, але `GET /api/analysis` довго дає `404` | Пайплайн ще виконується у фоні (мережа + ML-моделі + LLM-виклики можуть тривати десятки секунд/хвилини) | Зачекати і повторити запит; це очікувана поведінка `BackgroundTasks` |
| `GET /api/analysis` весь час `404`, хоча минуло достатньо часу | Фонова задача впала з винятком — `run_pipeline`/`run_pipeline_from_file` у `api.py` лише логують помилку (`logging.exception`) і **не** зберігають статус помилки нікуди, де його можна прочитати через API | Перевірити логи процесу застосунку; помилки фонового пайплайну **не** видно через жоден HTTP-ендпоінт |
| `ValueError: FETCHLAYER_API_KEY environment variable is not set or empty` | Не задана/порожня змінна `FETCHLAYER_API_KEY` | Перевірити `.env`/env vars контейнера |
| `LLMConfigError`/`ValueError: GROQ_API_KEY is not set` | `LLM_PROVIDER=groq`, але не заданий `GROQ_API_KEY` | Додати ключ або перемкнутись на `LLM_PROVIDER=ollama` |
| `ProviderError`/`LLMRequestError: Could not connect to Ollama` | Ollama-сервер недоступний або невірний `OLLAMA_HOST` | Локально: `OLLAMA_HOST=http://localhost:11434`; у Docker Compose: `http://ollama:11434` (сервіс `ollama` має бути запущений і healthy) |
| Ollama повертає помилку "model not found" | Модель, вказана в `OLLAMA_MODEL`/дефолт коду, не завантажена на сервері | `docker compose exec ollama ollama pull <model>` (або виправити плейсхолдер `your-model-name` у `ollama/start.sh` для Render-деплою) |
| `raise ValueError(f"Unsupported LLM_PROVIDER ...")` | `LLM_PROVIDER` заданий значенням, відмінним від `ollama`/`groq` | Перевірити значення змінної (регістр не важливий — код робить `.lower()`) |
| `review.json not found. Collect reviews first.` (`POST /api/reviews/analyze`) | Ще не було жодного успішного збору | Спочатку викликати `/collect` або `/fetch` |
| `ReviewCollectionError: Unexpected FetchLayer response shape` | FetchLayer змінив формат відповіді або повернув щось нестандартне | Перевірити відповідь FetchLayer напряму; клієнт очікує список або `dict` з ключем `reviews`/`data`/`results` |
| `NoReviewsAvailableError` | FetchLayer не повернув жодного відгуку для `app_id`/`country` | Перевірити коректність `app_id` та код країни |
| `RuntimeError: spaCy model 'en_core_web_sm' is not installed` | Модель spaCy не завантажена в середовищі | `python -m spacy download en_core_web_sm` (в Docker-образі це вже робиться на етапі білда) |
| Повільний перший запит sentiment/keyword-ендпоінтів | Transformer/KeyBERT-моделі завантажуються лениво при першому виклику (`_pipeline`/`_kw_model` — module-level singleton) | Очікувана поведінка; наступні виклики в тому самому процесі швидші |
| Недостатньо RAM / контейнер падає під час аналізу | Одночасне завантаження Transformer + sentence-transformers моделей (`torch`) в одному процесі потребує суттєвої пам'яті | Збільшити ліміт пам'яті контейнера/інстансу (в наданому `render.yaml` — `plan: standard`, конкретні ліміти не вказані) |
| `render`-деплой "unhealthy" одразу після старту | `healthCheckPath: /health` вказує на ендпоінт, якого немає в `api.py` | Або додати `GET /health` у код, або прибрати/змінити `healthCheckPath` у `render.yaml` |

---

## 20. Development Notes

- **Зміна LLM-моделі:**
  - Sentiment-шар — `processing/llm_sentiment.py`: `OLLAMA_MODEL`/`GROQ_MODEL` env vars, або хардкодні дефолти `OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")` / `_GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")`.
  - Insights-шар — `processing/llm_insights.py`: константи `DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"`, `DEFAULT_OLLAMA_MODEL = "llama3.1:latest"`, або аргумент `model=` у `generate_insight_report()`.
- **Зміна провайдера:** змінна `LLM_PROVIDER` в `.env`/env vars контейнера.
  - **Зверніть увагу:** дефолти, якщо змінна взагалі не задана, відрізняються між модулями — `llm_sentiment.py` дефолтить на `"ollama"` (`_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama")`), а `llm_insights.py` — на `"groq"` (`_resolve_provider()`: `os.environ.get("LLM_PROVIDER", "groq")`). Якщо `LLM_PROVIDER` не заданий явно, sentiment- та insights-шари можуть звертатись до **різних** провайдерів. Рекомендація: завжди задавати `LLM_PROVIDER` явно в `.env`, щоб уникнути цієї неочевидної розбіжності.
- **Зміна API-конфігурації:** `api.py` — назва застосунку/опис у `FastAPI(title=..., description=..., version=...)`, шляхи до файлів — константи `PROJECT_ROOT`, `RAW_REVIEWS_PATH`, `RESULTS_DIR` на початку файлу.
- **Зміна лімітів:**
  - Кількість відгуків на сторінку FetchLayer — `_REVIEWS_PER_PAGE = 20` в `collector/fetchlayer_client.py`.
  - Ліміт негативних відгуків, що йдуть у LLM insights — `_MAX_NEGATIVE_REVIEWS = 50` в `processing/llm_insights.py`.
  - Довжина обрізки тексту для LLM — `_MAX_EVIDENCE_CHARS = 300` там само.
  - Кількість ретраїв LLM insights — `_MAX_RETRIES = 3` там само.
  - Кількість паралельних воркерів для LLM sentiment — параметр `max_workers` у `attach_sentiment_llm()` (дефолт 4).
- **Зміна sentiment/keyword-пайплайну:**
  - Яке sentiment-поле вважається "джерелом правди" для негативної фільтрації — параметр `sentiment_field` (дефолт `"sentiment_transformer"`) у `extract_negative_texts()` (`processing/keywords.py`) та в `generate_insight_report()` (`processing/llm_insights.py`).
  - Стоп-слова для TF-IDF/KeyBERT — `_STOP_WORDS`, `_DOMAIN_STOP_WORDS`, `_GENERIC_FILLER_WORDS`, `_NEGATION_WORDS` у `processing/keywords.py`.
  - POS-патерни для spaCy-фраз — `_TWO_TOKEN_PATTERNS` у `processing/spacy_keywords.py`.
- **Тестування:** `pytest` є в `requirements.txt`, але окремих тестових файлів у наданому для аналізу коді не було — структуру `tests/` потрібно створювати з нуля.
