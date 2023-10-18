import time
from gpt_eval.config import (
    REQUEST_RETRY_MAX_ATTEMPTS,
    REQUEST_RETRY_BACKOFF,
    REQUEST_RETRY_WAIT_TIME
)

class Retry:
    def __init__(self, max_attempts=REQUEST_RETRY_MAX_ATTEMPTS, wait_time=REQUEST_RETRY_WAIT_TIME, backoff=REQUEST_RETRY_BACKOFF):
        self.max_attempts = max_attempts
        self.wait_time = wait_time
        self.backoff = backoff
        self.func = None

    def __call__(self, *args, **kwargs):
        def run_retry(*args, **kwargs):
            wait_time, backoff, attempts = self.wait_time, self.backoff, 0
            while attempts < self.max_attempts:
                print(f"Running {self.func.__name__}, attempt {attempts+1}")
                function_result = self.func(*args, **kwargs)
                if function_result:
                    return function_result
                print(f"Waiting {wait_time} seconds before the next attempt")
                time.sleep(wait_time)
                wait_time *= backoff
                attempts += 1

        if callable(args[0]):
            self.func = args[0]

        return run_retry