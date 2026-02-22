"""Centralised crawler state — thread-safe job tracking."""
import threading


class CrawlerState:
    def __init__(self):
        self._lock = threading.Lock()
        self.is_started: bool = False
        self.is_running: bool = False
        self.current_operation: str = ""
        self.job_id: str | None = None

    def start_batch_operation(self, job_id: str | None = None):
        with self._lock:
            self.is_started = True
            self.is_running = True
            self.job_id = job_id

    def stop_operation(self):
        with self._lock:
            self.is_started = False

    def finish_operation(self):
        with self._lock:
            self.is_started = False
            self.is_running = False
            self.current_operation = ""
            self.job_id = None

    def is_cancelled(self) -> bool:
        """Return True when stop has been requested but run loop is still active."""
        with self._lock:
            return self.is_running and not self.is_started


# Singleton
state = CrawlerState()
