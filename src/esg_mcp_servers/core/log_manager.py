"""Shared log message queue."""
from collections import deque

# Thread-safe bounded deque (max 1000 messages to avoid unbounded growth)
log_messages: deque[str] = deque(maxlen=1000)
