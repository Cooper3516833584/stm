import os
import sys

from loguru import logger

_CONFIGURED = False


def setup_logging(log_dir: str | None = None, level: str = "DEBUG") -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    _CONFIGURED = True
    logger.remove()

    resolved_log_dir = log_dir or os.environ.get("FC_LOG_DIR", "fc_log")
    file_logging_error: Exception | None = None
    try:
        os.makedirs(resolved_log_dir, exist_ok=True)
        logger.add(
            os.path.join(resolved_log_dir, "{time}.log"),
            retention="1day",
            level=level,
            backtrace=True,
            diagnose=True,
            filter=lambda record: "debug" not in record["extra"],
        )
        logger.add(
            os.path.join(resolved_log_dir, "navigation_debug_{time}.log"),
            retention="1day",
            filter=lambda record: "debug" in record["extra"],
            level="DEBUG",
        )
    except Exception as exc:
        file_logging_error = exc

    logger.add(
        sys.stdout,
        filter=lambda record: "debug" not in record["extra"],
        level=level,
    )

    if file_logging_error is not None:
        logger.warning(f"[LOG] File logging disabled: {file_logging_error}")
