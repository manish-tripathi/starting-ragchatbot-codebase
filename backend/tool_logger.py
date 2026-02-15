import logging
import os
import time
from logging.handlers import RotatingFileHandler


class ToolLogger:
    """Logger for tool invocations with rotating file output."""

    def __init__(self, log_dir: str = None):
        if log_dir is None:
            log_dir = os.path.join(os.path.dirname(__file__), "logs")
        os.makedirs(log_dir, exist_ok=True)

        log_path = os.path.join(log_dir, "tool_calls.log")

        self.logger = logging.getLogger("tool_calls")
        self.logger.setLevel(logging.DEBUG)

        # Avoid adding duplicate handlers if instantiated multiple times
        if not self.logger.handlers:
            handler = RotatingFileHandler(
                log_path, maxBytes=2 * 1024 * 1024, backupCount=3
            )
            formatter = logging.Formatter(
                "%(asctime)s | %(levelname)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def log_tool_call(
        self,
        tool_name: str,
        inputs: dict,
        output: str,
        success: bool,
        duration_ms: float,
        error: str = None,
    ):
        """Log a single tool invocation."""
        truncated_output = (output[:500] + "...") if len(output) > 500 else output
        status = "OK" if success else "ERROR"
        msg = (
            f"tool={tool_name} | status={status} | "
            f"duration={duration_ms:.0f}ms | "
            f"input={inputs} | output={truncated_output}"
        )
        if error:
            msg += f" | error={error}"

        if success:
            self.logger.info(msg)
        else:
            self.logger.error(msg)
