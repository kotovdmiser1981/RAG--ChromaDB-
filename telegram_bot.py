"""
Модуль Telegram-бота, интегрированного с RAG-ассистентом.

Бот позволяет задавать вопросы ассистенту через Telegram,
получать ответы на основе векторного поиска и LLM,
а также сохранять статистику взаимодействий.
"""

import os
import time
import logging
from typing import Optional

from telegram import Update
from telegram.error import TelegramError
from telegram.request import HTTPXRequest
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from rag import RAGAssistant
from cache import ResponseCache
from db_logger import DatabaseLogger


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
# Не выводим HTTP-адреса Telegram API, содержащие токен бота.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


class TelegramRAGBot:
    """Telegram-бот для RAG-ассистента."""

    def __init__(
        self,
        token: str,
        rag_assistant: RAGAssistant,
        cache: ResponseCache,
        logger: DatabaseLogger,
    ):
        self.rag_assistant = rag_assistant
        self.cache = cache
        self.db_logger = logger

        # Основной HTTP-клиент Telegram API.
        # trust_env=True явно разрешает использование настроек
        # окружения, включая поддерживаемые HTTPX прокси.
        request = HTTPXRequest(
            connection_pool_size=8,
            connect_timeout=30,
            read_timeout=30,
            write_timeout=30,
            pool_timeout=10,
            httpx_kwargs={"trust_env": True},
        )

        # Отдельный клиент для получения обновлений Telegram.
        get_updates_request = HTTPXRequest(
            connection_pool_size=2,
            connect_timeout=30,
            read_timeout=40,
            write_timeout=30,
            pool_timeout=10,
            httpx_kwargs={"trust_env": True},
        )

        self.application = (
            Application.builder()
            .token(token)
            .request(request)
            .get_updates_request(get_updates_request)
            .build()
        )

        # Команды
        self.application.add_handler(
            CommandHandler("start", self.start_command)
        )
        self.application.add_handler(
            CommandHandler("help", self.help_command)
        )
        self.application.add_handler(
            CommandHandler("stats", self.stats_command)
        )
        self.application.add_handler(
            CommandHandler("logs", self.logs_command)
        )

        # Обычные текстовые сообщения
        self.application.add_handler(
            MessageHandler(
                filters.TEXT & ~filters.COMMAND,
                self.handle_message,
            )
        )

    async def start_command(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ):
        """Обработчик команды /start."""
        if not update.message:
            return

        welcome_message = """
🤖 Добро пожаловать в RAG-ассистента!

Я могу отвечать на ваши вопросы, используя базу знаний.

Доступные команды:
/help — показать справку
/stats — статистика системы
/logs — получить логи в CSV формате

Просто напишите мне вопрос!
"""
        await update.message.reply_text(welcome_message.strip())

    async def help_command(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ):
        """Обработчик команды /help."""
        if not update.message:
            return

        help_text = """
📚 Справка по использованию бота:

• Напишите вопрос — я отвечу на основе базы знаний.
• Использую RAG для поиска релевантной информации.
• Ответы кешируются для ускорения повторных запросов.

Команды:
/start — начать работу
/help — показать справку
/stats — статистика системы
/logs — экспорт логов в CSV

Примеры вопросов:
• Что такое Python?
• Расскажи про RAG.
• Что такое векторные базы данных?
"""
        await update.message.reply_text(help_text.strip())

    async def stats_command(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ):
        """Обработчик команды /stats."""
        if not update.message:
            return

        try:
            doc_count = self.rag_assistant.embedding_store.collection.count()
            cache_size = self.cache.size()
            model = self.rag_assistant.model
            log_stats = self.db_logger.get_stats()

            stats_message = f"""
📊 СТАТИСТИКА СИСТЕМЫ

📚 База знаний:
• Документов в ChromaDB: {doc_count}
• Модель LLM: {model}

💾 Кеш:
• Записей в кеше: {cache_size}

📝 Логи:
• Всего запросов: {log_stats['total_requests']}
• Из кеша: {log_stats['cached_requests']}
• Уникальных пользователей: {log_stats['unique_users']}
• Среднее время ответа: {log_stats['avg_response_time_ms']:.0f} мс
"""
            await update.message.reply_text(stats_message.strip())

        except Exception:
            logger.exception("Ошибка при получении статистики")
            await self._send_error(
                update,
                "Не удалось получить статистику системы.",
            )

    async def logs_command(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ):
        """Экспорт логов текущего пользователя в CSV."""
        if not update.message or not update.effective_user:
            return

        filename = None

        try:
            user_id = str(update.effective_user.id)
            csv_content = self.db_logger.export_to_csv(user_id=user_id)

            if not csv_content:
                await update.message.reply_text(
                    "📝 Логов для вашего пользователя не найдено."
                )
                return

            filename = f"logs_{user_id}_{int(time.time())}.csv"

            with open(filename, "w", encoding="utf-8-sig", newline="") as file:
                file.write(csv_content)

            with open(filename, "rb") as file:
                await update.message.reply_document(
                    document=file,
                    filename=filename,
                    caption="📊 Ваши логи взаимодействий с ботом",
                )

        except Exception:
            logger.exception("Ошибка при экспорте логов")
            await self._send_error(
                update,
                "Не удалось экспортировать логи.",
            )

        finally:
            if filename and os.path.exists(filename):
                try:
                    os.remove(filename)
                except OSError:
                    logger.warning(
                        "Не удалось удалить временный файл %s",
                        filename,
                    )

    async def handle_message(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ):
        """Обработка текстового сообщения пользователя."""
        if not update.message or not update.effective_user:
            return

        user_message = update.message.text
        user = update.effective_user
        user_id = str(user.id)
        username = user.username or user.first_name or "Unknown"
        start_time = time.time()

        try:
            await update.message.chat.send_action(action="typing")

            # Проверяем кеш.
            cached_answer = self.cache.get(user_message)
            from_cache = cached_answer is not None

            if from_cache:
                answer = cached_answer
            else:
                answer, _ = self.rag_assistant.generate_response(
                    query=user_message,
                    top_k=3,
                    verbose=False,
                )
                self.cache.set(user_message, answer)

            response_time_ms = int((time.time() - start_time) * 1000)

            # Сохраняем взаимодействие в базе данных.
            self.db_logger.log_interaction(
                query=user_message,
                response=answer,
                source="telegram",
                user_id=user_id,
                username=username,
                from_cache=from_cache,
                response_time_ms=response_time_ms,
            )

            # Telegram ограничивает длину текстового сообщения.
            max_length = 4000
            parts = [
                answer[i:i + max_length]
                for i in range(0, len(answer), max_length)
            ]

            for part in parts:
                await update.message.reply_text(part)

            if from_cache:
                await update.message.reply_text(
                    "💾 Ответ получен из кеша.",
                    do_quote=False,
                )

        except Exception:
            logger.exception("Ошибка при обработке сообщения пользователя")

            error_message = (
                "❌ Не удалось обработать запрос. "
                "Попробуйте ещё раз немного позже."
            )

            response_time_ms = int((time.time() - start_time) * 1000)

            # Ошибка логирования не должна мешать уведомлению пользователя.
            try:
                self.db_logger.log_interaction(
                    query=user_message,
                    response=error_message,
                    source="telegram",
                    user_id=user_id,
                    username=username,
                    from_cache=False,
                    response_time_ms=response_time_ms,
                )
            except Exception:
                logger.exception("Не удалось записать ошибку в журнал")

            await self._send_error(update, error_message)

    async def _send_error(self, update: Update, message: str):
        """Отправляет сообщение об ошибке, не скрывая исходный сбой."""
        if not update.effective_message:
            return

        try:
            await update.effective_message.reply_text(message)
        except TelegramError:
            logger.exception(
                "Не удалось отправить сообщение об ошибке в Telegram"
            )

    def run(self):
        """Запускает бота."""
        print("🤖 Запуск Telegram-бота...")
        print("При запуске выполняется проверка подключения к Telegram API.")

        try:
            self.application.run_polling(
                bootstrap_retries=5,
                poll_interval=1.0,
                timeout=20,
            )
        except KeyboardInterrupt:
            print("Остановка Telegram-бота...")
        except Exception:
            logger.exception("Критическая ошибка при запуске Telegram-бота")
            raise
