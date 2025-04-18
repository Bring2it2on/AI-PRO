import logging
from logging.handlers import RotatingFileHandler
import traceback
import asyncio
from functools import wraps
import os
import threading


logging.basicConfig(filename="execution_log.txt", level=logging.INFO, format="%(asctime)s - %(message)s", encoding="utf-8")

def async_log_function_call(func):
    """비동기 함수에서도 동작하는 비동기 로깅 데코레이터"""
    if asyncio.iscoroutinefunction(func):  # 함수가 비동기 함수인지 확인
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            stack = traceback.extract_stack()[-2]
            caller_file = stack.filename
            caller_line = stack.lineno
            caller_func = stack.name

            # 비동기 로깅
            await asyncio.sleep(0)  # 문맥 전환을 위한 짧은 대기 (필수 X)
            logging.info(f"Async Function Call: {func.__name__}()")
            logging.info(f" - Called from: {caller_file} (line {caller_line}) in function {caller_func}")

            return await func(*args, **kwargs)  # 원래 비동기 함수 실행

        return async_wrapper

    else:  # 동기 함수 처리
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            stack = traceback.extract_stack()[-2]
            caller_file = stack.filename
            caller_line = stack.lineno
            caller_func = stack.name

            logging.info(f"Sync Function Call: {func.__name__}()")
            logging.info(f" - Called from: {caller_file} (line {caller_line}) in function {caller_func}")

            return func(*args, **kwargs)  # 원래 동기 함수 실행

        return sync_wrapper


class LoggerSingleton:
    _instance = None
    _lock = threading.Lock()
    _loggers = {}

    @classmethod
    def get_logger(cls, logger_name, level=logging.INFO , log_file="data/logs/test/Log.txt"):
        with cls._lock:
            if logger_name not in cls._loggers:
                logger = logging.getLogger(logger_name)
                logger.setLevel(level)

                # propagate 비활성화로 로그 전파 방지
                logger.propagate = False

                # 핸들러가 이미 있으면 반환
                if logger.handlers:
                    cls._loggers[logger_name] = logger
                    return logger

                # 로그 디렉토리 생성
                os.makedirs(os.path.dirname(log_file), exist_ok=True)

                # RotatingFileHandler 사용 (파일 크기 제한 및 백업)
                file_handler = RotatingFileHandler(
                    log_file,
                    maxBytes=10 * 1024 * 1024,  # 10MB
                    backupCount=5,
                    encoding="utf-8",
                )
                file_handler.setLevel(level)

                # 콘솔 출력을 위한 StreamHandler 추가
                console_handler = logging.StreamHandler()
                console_handler.setLevel(level)

                # 포맷터 설정
                formatter = logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
                )
                file_handler.setFormatter(formatter)
                console_handler.setFormatter(formatter)

                # 기존 핸들러 제거
                if logger.handlers:
                    logger.handlers.clear()

                # 새 핸들러 추가
                logger.addHandler(file_handler)
                logger.addHandler(console_handler)

                # 로거 캐시
                cls._loggers[logger_name] = logger

            return cls._loggers[logger_name]


HTTP_STATUS_MESSAGES = {
    200: "Success",
    400: "Bad Request",
    401: "Unauthorized",
    403: "Forbidden",
    404: "Not Found",
    500: "Internal Server Error",
}

# 상태(status) 값 정의
STATUS_TYPES = ["success", "error", "cancelled", "complete"]

# 로그 파일을 UTF-8로 읽어보는 예시
log_file_path = 'execution_log.txt'