# RAG System — Q&A по любому тексту (FastAPI + VoyageAI + Gemini + ChromaDB)

Простая RAG-система, которая принимает текст (любой: статья, документ, описание продукта и т.п.)  
и отвечает на вопросы по этому тексту, указывая источник ответа.  
Работает через **FastAPI**, **VoyageAI** (эмбеддинги), **Gemini** (LLM), и **ChromaDB** (векторное хранилище).

<img width="1598" height="771" alt="image" src="https://github.com/user-attachments/assets/a8780014-52b2-4c15-b0cd-171557464b29" />


## Что делает

1. Принимаешь любой текст через API.  
2. Текст автоматически разбивается на чанки, векторизуется через **VoyageAI** и кладётся в **ChromaDB**.  
3. При запросе система:
   - достаёт релевантные фрагменты из ChromaDB,
   - передаёт их в **Gemini**,
   - получает ответ и возвращает его вместе с источниками (чанками текста, откуда взята информация).

---

## Основные технологии

| Компонент | Назначение |
|------------|------------|
| **FastAPI** | HTTP API-интерфейс |
| **VoyageAI** | Создание эмбеддингов |
| **Gemini** | LLM для генерации ответов |
| **ChromaDB** | Векторная база данных |
| **uvicorn** | запуск |

---

## Установка
git clone https://github.com/serenitymea/docsscanAi
cd docsscanAi

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt

python web_i.py web --voyage-key .key here. --gemini-key .key here.

