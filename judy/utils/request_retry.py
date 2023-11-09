import time
import logging

from judy.config import (
    REQUEST_RETRY_BACKOFF,
    REQUEST_RETRY_MAX_ATTEMPTS,
    REQUEST_RETRY_WAIT_TIME,
)

_logger = logging.getLogger("app")


class Retry:
    def __init__(self, *args, **kwargs):
        self.func = None
        self.max_attempts = kwargs.get("max_attempts", REQUEST_RETRY_MAX_ATTEMPTS)
        self.wait_time = kwargs.get("wait_time", REQUEST_RETRY_WAIT_TIME)
        self.backoff = kwargs.get("backoff", REQUEST_RETRY_BACKOFF)

        if len(args) > 0 and callable(args[0]):
            self.func = args[0]

    def __call__(self, *args, **kwargs):
        def run_retry(*args, **kwargs):
            wait_time, backoff, attempts = self.wait_time, self.backoff, 0
            function_result = None
            while attempts < self.max_attempts:
                _logger.debug(
                    "Running %s, attempt %d", self.func.__name__, attempts + 1
                )
                try:
                    function_result = self.func(*args, **kwargs)
                except Exception as e:
                    _logger.error("Request failed for %s - %s", self.func.__name__, e)
                if function_result:
                    return function_result
                _logger.debug("Waiting %s seconds before next attempt", wait_time)
                time.sleep(self.wait_time)
                wait_time *= backoff
                attempts += 1
            if not function_result:
                raise SystemError(
                    f"{self.func.__name__} failed after {self.max_attempts} attempts"
                )
            return None

        if len(args) > 0 and callable(args[0]):
            self.func = args[0]
            return run_retry

        return run_retry(*args, **kwargs)
