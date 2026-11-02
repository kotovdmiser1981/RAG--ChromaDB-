"""
Модуль Telegram-бота, интегрированного с RAG-ассистентом.

Бот позволяет задавать вопросы ассистенту через Telegram,
получать ответы на основе векторного поиска и LLM,
а также сохранять статистику взаимодействий.
"""

import os
import time
import asyncio
import logging

from telegram import Update
from telegram.error import TelegramError, NetworkError
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
        # trust_env=True позволяет HTTPX использовать настройки окружения,
        # в том числе поддерживаемые переменные прокси.
        request = HTTPXRequest(
            connection_pool_size=8,
            connect_timeout=30,
            read_timeout=30,
            write_timeout=30,
            pool_timeout=10,
            httpx_kwargs={"trust_env": True},
        )

        # Отдельный HTTP-клиент для long polling (получения обновлений).
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

        # Команды Telegram.
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

        # Обычные текстовые сообщения.
        self.application.add_handler(
            MessageHandler(
                filters.TEXT & ~filters.COMMAND,
                self.handle_message,
            )
        )

        # Общий обработчик необработанных ошибок.
        self.application.add_error_handler(self.error_handler)

    async def _retry_telegram_call(
        self,
        operation,
        operation_name: str,
        retries: int = 3,
    ):
        """
        Повторяет Telegram API-вызов при временной сетевой ошибке.

        Задержки между попытками: 1, 2 секунды.
        Повторная отправка после сетевого обрыва теоретически может
        привести к дубликату, если Telegram принял сообщение,
        но клиент не получил подтверждение.
        """
        for attempt in range(retries):
            try:
                return await operation()

            except NetworkError:
                if attempt == retries - 1:
                    logger.exception(
                        "Не удалось выполнить операцию '%s' после %s попыток",
                        operation_name,
                        retries,
                    )
                    raise

                delay = 2 ** attempt
                logger.warning(
                    "Сетевая ошибка Telegram при операции '%s'. "
                    "Повтор через %s сек. Попытка %s/%s",
                    operation_name,
                    delay,
                    attempt + 1,
                    retries,
                )
                await asyncio.sleep(delay)

    async def _safe_reply(
        self,
        message,
        text: str,
        retries: int = 3,
        **kwargs,
    ):
        """Отправляет текстовое сообщение с повторными попытками."""
        return await self._retry_telegram_call(
            operation=lambda: message.reply_text(text, **kwargs),
            operation_name="отправка текстового сообщения",
            retries=retries,
        )

    async def _safe_reply_document(
        self,
        message,
        filename: str,
        caption: str = None,
        retries: int = 3,
    ):
        """Отправляет документ; файл заново открывается для каждой попытки."""

        async def send_document():
            with open(filename, "rb") as file:
                return await message.reply_document(
                    document=file,
                    filename=os.path.basename(filename),
                    caption=caption,
                )

        return await self._retry_telegram_call(
            operation=send_document,
            operation_name="отправка документа",
            retries=retries,
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
        await self._safe_reply(
            update.message,
            welcome_message.strip(),
        )

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
        await self._safe_reply(
            update.message,
            help_text.strip(),
        )

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
            await self._safe_reply(
                update.message,
                stats_message.strip(),
            )

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
                await self._safe_reply(
                    update.message,
                    "📝 Логов для вашего пользователя не найдено.",
                )
                return

            filename = f"logs_{user_id}_{int(time.time())}.csv"

            with open(filename, "w", encoding="utf-8-sig", newline="") as file:
                file.write(csv_content)

            await self._safe_reply_document(
                update.message,
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
            # Индикатор набора текста. Если он временно не отправился,
            # продолжаем обработку вопроса.
            try:
                await update.message.chat.send_action(action="typing")
            except NetworkError:
                logger.warning(
                    "Не удалось отправить индикатор typing; "
                    "обработка вопроса продолжается."
                )

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
            # Оставляем запас относительно предельной длины.
            max_length = 4000
            parts = [
                answer[i:i + max_length]
                for i in range(0, len(answer), max_length)
            ]

            for part in parts:
                await self._safe_reply(update.message, part)

            if from_cache:
                await self._safe_reply(
                    update.message,
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
            await self._safe_reply(update.effective_message, message)
        except TelegramError:
            logger.exception(
                "Не удалось отправить сообщение об ошибке в Telegram"
            )

    async def error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE):
        """Обрабатывает ошибки, не перехваченные отдельными обработчиками."""
        logger.error(
            "Необработанная ошибка при обработке обновления: %s",
            context.error,
            exc_info=context.error,
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
