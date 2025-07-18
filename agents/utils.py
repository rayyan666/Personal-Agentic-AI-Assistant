"""
Utility functions and classes for the agent system
"""
import uuid
import asyncio
import json
import hashlib
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from functools import wraps
import logging

logger = logging.getLogger(__name__)

def generate_task_id() -> str:
    """Generate a unique task ID"""
    return str(uuid.uuid4())

def hash_content(content: str) -> str:
    """Generate a hash for content deduplication"""
    return hashlib.md5(content.encode()).hexdigest()

def sanitize_input(text: str, max_length: int = 10000) -> str:
    """Sanitize and limit input text"""
    if not isinstance(text, str):
        text = str(text)
    
    # Remove potentially harmful characters
    sanitized = text.replace('\x00', '').strip()
    
    # Limit length
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length] + "..."
    
    return sanitized

def format_timestamp(dt: datetime) -> str:
    """Format datetime for display"""
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def parse_json_safely(json_str: str) -> Optional[Dict[str, Any]]:
    """Safely parse JSON string"""
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return None

def retry_async(max_retries: int = 3, delay: float = 1.0):
    """Decorator for retrying async functions"""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {delay}s...")
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"All {max_retries} attempts failed. Last error: {str(e)}")
            
            raise last_exception
        return wrapper
    return decorator

class RateLimiter:
    """Simple rate limiter for API calls"""
    
    def __init__(self, max_calls: int, time_window: int):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
    
    async def acquire(self):
        """Acquire permission to make a call"""
        now = datetime.now()
        
        # Remove old calls outside the time window
        self.calls = [call_time for call_time in self.calls 
                     if (now - call_time).seconds < self.time_window]
        
        # Check if we can make a new call
        if len(self.calls) >= self.max_calls:
            # Calculate wait time
            oldest_call = min(self.calls)
            wait_time = self.time_window - (now - oldest_call).seconds
            if wait_time > 0:
                await asyncio.sleep(wait_time)
        
        # Record the new call
        self.calls.append(now)

class TaskQueue:
    """Simple async task queue"""
    
    def __init__(self, max_size: int = 100):
        self.queue = asyncio.Queue(maxsize=max_size)
        self.processing = False
    
    async def put(self, item):
        """Add item to queue"""
        await self.queue.put(item)
    
    async def get(self):
        """Get item from queue"""
        return await self.queue.get()
    
    def empty(self) -> bool:
        """Check if queue is empty"""
        return self.queue.empty()
    
    def qsize(self) -> int:
        """Get queue size"""
        return self.queue.qsize()

class MemoryCache:
    """Simple in-memory cache with TTL"""
    
    def __init__(self, default_ttl: int = 3600):
        self.cache = {}
        self.default_ttl = default_ttl
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set cache value with TTL"""
        expiry = datetime.now() + timedelta(seconds=ttl or self.default_ttl)
        self.cache[key] = {
            'value': value,
            'expiry': expiry
        }
    
    def get(self, key: str) -> Optional[Any]:
        """Get cache value if not expired"""
        if key not in self.cache:
            return None
        
        entry = self.cache[key]
        if datetime.now() > entry['expiry']:
            del self.cache[key]
            return None
        
        return entry['value']
    
    def delete(self, key: str):
        """Delete cache entry"""
        if key in self.cache:
            del self.cache[key]
    
    def clear(self):
        """Clear all cache entries"""
        self.cache.clear()
    
    def cleanup_expired(self):
        """Remove expired entries"""
        now = datetime.now()
        expired_keys = [
            key for key, entry in self.cache.items()
            if now > entry['expiry']
        ]
        for key in expired_keys:
            del self.cache[key]

class EventEmitter:
    """Simple event emitter for agent communication"""
    
    def __init__(self):
        self.listeners = {}
    
    def on(self, event: str, callback: Callable):
        """Register event listener"""
        if event not in self.listeners:
            self.listeners[event] = []
        self.listeners[event].append(callback)
    
    def off(self, event: str, callback: Callable):
        """Remove event listener"""
        if event in self.listeners:
            try:
                self.listeners[event].remove(callback)
            except ValueError:
                pass
    
    async def emit(self, event: str, *args, **kwargs):
        """Emit event to all listeners"""
        if event in self.listeners:
            for callback in self.listeners[event]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(*args, **kwargs)
                    else:
                        callback(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Error in event listener for {event}: {str(e)}")

def validate_email(email: str) -> bool:
    """Simple email validation"""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def extract_urls(text: str) -> List[str]:
    """Extract URLs from text"""
    import re
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.findall(url_pattern, text)

def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to specified length"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"

class ConfigValidator:
    """Validate configuration settings"""
    
    @staticmethod
    def validate_required_fields(config: Dict[str, Any], required_fields: List[str]) -> List[str]:
        """Validate that required fields are present"""
        missing_fields = []
        for field in required_fields:
            if field not in config or config[field] is None:
                missing_fields.append(field)
        return missing_fields
    
    @staticmethod
    def validate_types(config: Dict[str, Any], type_map: Dict[str, type]) -> List[str]:
        """Validate field types"""
        type_errors = []
        for field, expected_type in type_map.items():
            if field in config and not isinstance(config[field], expected_type):
                type_errors.append(f"{field} should be of type {expected_type.__name__}")
        return type_errors

# Global instances
global_cache = MemoryCache()
global_event_emitter = EventEmitter()
global_rate_limiter = RateLimiter(max_calls=100, time_window=60)  # 100 calls per minute
