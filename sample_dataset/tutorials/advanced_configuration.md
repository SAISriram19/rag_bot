# Advanced Configuration Guide

## Overview

This guide covers advanced configuration options for optimizing performance, security, and scalability.

## Database Configuration

### Connection Pooling

Configure connection pooling for better performance:

```python
# config/database.py
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

DATABASE_CONFIG = {
    'url': 'postgresql://user:password@localhost/dbname',
    'pool_size': 20,
    'max_overflow': 30,
    'pool_timeout': 30,
    'pool_recycle': 3600,
    'pool_pre_ping': True
}

engine = create_engine(
    DATABASE_CONFIG['url'],
    poolclass=QueuePool,
    pool_size=DATABASE_CONFIG['pool_size'],
    max_overflow=DATABASE_CONFIG['max_overflow'],
    pool_timeout=DATABASE_CONFIG['pool_timeout'],
    pool_recycle=DATABASE_CONFIG['pool_recycle'],
    pool_pre_ping=DATABASE_CONFIG['pool_pre_ping']
)
```

### Read Replicas

Set up read replicas for scaling read operations:

```python
# config/database.py
DATABASES = {
    'write': {
        'url': 'postgresql://user:password@primary.db.example.com/dbname',
        'pool_size': 10
    },
    'read': [
        {
            'url': 'postgresql://user:password@replica1.db.example.com/dbname',
            'pool_size': 15,
            'weight': 1
        },
        {
            'url': 'postgresql://user:password@replica2.db.example.com/dbname',
            'pool_size': 15,
            'weight': 1
        }
    ]
}

class DatabaseRouter:
    def __init__(self, config):
        self.write_engine = create_engine(config['write']['url'])
        self.read_engines = [
            create_engine(replica['url']) 
            for replica in config['read']
        ]
        self.read_weights = [replica['weight'] for replica in config['read']]
    
    def get_read_engine(self):
        import random
        return random.choices(self.read_engines, weights=self.read_weights)[0]
```

## Caching Strategies

### Redis Configuration

Configure Redis for caching and session storage:

```python
# config/cache.py
import redis
from redis.sentinel import Sentinel

# Single Redis instance
REDIS_CONFIG = {
    'host': 'localhost',
    'port': 6379,
    'db': 0,
    'password': 'your-redis-password',
    'socket_timeout': 5,
    'socket_connect_timeout': 5,
    'retry_on_timeout': True,
    'health_check_interval': 30
}

# Redis Sentinel for high availability
SENTINEL_CONFIG = {
    'sentinels': [
        ('sentinel1.example.com', 26379),
        ('sentinel2.example.com', 26379),
        ('sentinel3.example.com', 26379)
    ],
    'service_name': 'mymaster',
    'socket_timeout': 0.1,
    'password': 'sentinel-password'
}

def get_redis_client():
    if SENTINEL_CONFIG['sentinels']:
        sentinel = Sentinel(
            SENTINEL_CONFIG['sentinels'],
            socket_timeout=SENTINEL_CONFIG['socket_timeout']
        )
        return sentinel.master_for(
            SENTINEL_CONFIG['service_name'],
            socket_timeout=REDIS_CONFIG['socket_timeout'],
            password=REDIS_CONFIG['password']
        )
    else:
        return redis.Redis(**REDIS_CONFIG)
```

### Multi-Level Caching

Implement multi-level caching strategy:

```python
# services/cache_service.py
import time
from typing import Any, Optional
from functools import wraps

class CacheService:
    def __init__(self, redis_client, local_cache_size=1000):
        self.redis = redis_client
        self.local_cache = {}
        self.local_cache_timestamps = {}
        self.max_local_size = local_cache_size
        self.local_ttl = 300  # 5 minutes
    
    def get(self, key: str) -> Optional[Any]:
        # Check local cache first
        if key in self.local_cache:
            if time.time() - self.local_cache_timestamps[key] < self.local_ttl:
                return self.local_cache[key]
            else:
                del self.local_cache[key]
                del self.local_cache_timestamps[key]
        
        # Check Redis cache
        value = self.redis.get(key)
        if value:
            # Store in local cache
            self._store_local(key, value)
            return value
        
        return None
    
    def set(self, key: str, value: Any, ttl: int = 3600):
        # Store in Redis
        self.redis.setex(key, ttl, value)
        # Store in local cache
        self._store_local(key, value)
    
    def _store_local(self, key: str, value: Any):
        if len(self.local_cache) >= self.max_local_size:
            # Remove oldest entry
            oldest_key = min(self.local_cache_timestamps.keys(),
                           key=lambda k: self.local_cache_timestamps[k])
            del self.local_cache[oldest_key]
            del self.local_cache_timestamps[oldest_key]
        
        self.local_cache[key] = value
        self.local_cache_timestamps[key] = time.time()

def cached(ttl=3600, key_prefix=""):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = f"{key_prefix}:{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Try to get from cache
            cached_result = cache_service.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_service.set(cache_key, result, ttl)
            return result
        return wrapper
    return decorator
```

## Security Configuration

### JWT Authentication

Configure JWT with proper security settings:

```python
# config/auth.py
import jwt
from datetime import datetime, timedelta
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

class JWTConfig:
    def __init__(self):
        # Generate RSA key pair for JWT signing
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        self.public_key = self.private_key.public_key()
        
        # JWT settings
        self.algorithm = 'RS256'
        self.access_token_expire = timedelta(minutes=15)
        self.refresh_token_expire = timedelta(days=7)
        self.issuer = 'api.example.com'
        self.audience = 'example-app'
    
    def create_access_token(self, user_id: str, scopes: list = None):
        now = datetime.utcnow()
        payload = {
            'sub': user_id,
            'iat': now,
            'exp': now + self.access_token_expire,
            'iss': self.issuer,
            'aud': self.audience,
            'type': 'access',
            'scopes': scopes or []
        }
        
        private_pem = self.private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        return jwt.encode(payload, private_pem, algorithm=self.algorithm)
    
    def verify_token(self, token: str):
        public_pem = self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        try:
            payload = jwt.decode(
                token, 
                public_pem, 
                algorithms=[self.algorithm],
                issuer=self.issuer,
                audience=self.audience
            )
            return payload
        except jwt.InvalidTokenError:
            return None
```

### Rate Limiting

Implement sophisticated rate limiting:

```python
# middleware/rate_limiter.py
import time
from collections import defaultdict, deque
from typing import Dict, Tuple

class RateLimiter:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.local_counters = defaultdict(deque)
    
    def is_allowed(self, key: str, limit: int, window: int) -> Tuple[bool, Dict]:
        """
        Check if request is allowed based on sliding window rate limiting
        
        Args:
            key: Unique identifier (user_id, IP, etc.)
            limit: Maximum requests allowed
            window: Time window in seconds
        
        Returns:
            (is_allowed, metadata)
        """
        now = time.time()
        window_start = now - window
        
        # Use Redis for distributed rate limiting
        pipe = self.redis.pipeline()
        
        # Remove old entries
        pipe.zremrangebyscore(f"rate_limit:{key}", 0, window_start)
        
        # Count current requests
        pipe.zcard(f"rate_limit:{key}")
        
        # Add current request
        pipe.zadd(f"rate_limit:{key}", {str(now): now})
        
        # Set expiration
        pipe.expire(f"rate_limit:{key}", window)
        
        results = pipe.execute()
        current_count = results[1]
        
        is_allowed = current_count < limit
        
        metadata = {
            'limit': limit,
            'remaining': max(0, limit - current_count - 1),
            'reset_time': int(now + window),
            'retry_after': window if not is_allowed else None
        }
        
        return is_allowed, metadata

# Usage in middleware
def rate_limit_middleware(limit_config):
    def middleware(request, response):
        # Determine rate limit key (user, IP, etc.)
        user_id = getattr(request, 'user_id', None)
        ip_address = request.remote_addr
        
        key = f"user:{user_id}" if user_id else f"ip:{ip_address}"
        
        # Get rate limit for this endpoint
        endpoint_limits = limit_config.get(request.endpoint, {})
        limit = endpoint_limits.get('requests', 100)
        window = endpoint_limits.get('window', 3600)
        
        is_allowed, metadata = rate_limiter.is_allowed(key, limit, window)
        
        # Add rate limit headers
        response.headers.update({
            'X-RateLimit-Limit': str(metadata['limit']),
            'X-RateLimit-Remaining': str(metadata['remaining']),
            'X-RateLimit-Reset': str(metadata['reset_time'])
        })
        
        if not is_allowed:
            response.status_code = 429
            response.headers['Retry-After'] = str(metadata['retry_after'])
            return {'error': 'Rate limit exceeded'}
        
        return None  # Continue processing
    
    return middleware
```

## Performance Optimization

### Connection Management

Optimize HTTP client connections:

```python
# services/http_client.py
import httpx
import asyncio
from typing import Optional

class HTTPClientManager:
    def __init__(self):
        self.client: Optional[httpx.AsyncClient] = None
        self.limits = httpx.Limits(
            max_keepalive_connections=20,
            max_connections=100,
            keepalive_expiry=30
        )
        self.timeout = httpx.Timeout(
            connect=5.0,
            read=30.0,
            write=10.0,
            pool=5.0
        )
    
    async def get_client(self) -> httpx.AsyncClient:
        if self.client is None or self.client.is_closed:
            self.client = httpx.AsyncClient(
                limits=self.limits,
                timeout=self.timeout,
                http2=True
            )
        return self.client
    
    async def close(self):
        if self.client and not self.client.is_closed:
            await self.client.aclose()

# Global client manager
http_manager = HTTPClientManager()

async def make_request(method: str, url: str, **kwargs):
    client = await http_manager.get_client()
    response = await client.request(method, url, **kwargs)
    return response
```

### Background Task Processing

Configure Celery for background tasks:

```python
# config/celery_config.py
from celery import Celery
from kombu import Queue

# Celery configuration
CELERY_CONFIG = {
    'broker_url': 'redis://localhost:6379/1',
    'result_backend': 'redis://localhost:6379/2',
    'task_serializer': 'json',
    'accept_content': ['json'],
    'result_serializer': 'json',
    'timezone': 'UTC',
    'enable_utc': True,
    'task_track_started': True,
    'task_time_limit': 30 * 60,  # 30 minutes
    'task_soft_time_limit': 25 * 60,  # 25 minutes
    'worker_prefetch_multiplier': 1,
    'task_acks_late': True,
    'worker_max_tasks_per_child': 1000,
}

# Queue configuration
CELERY_QUEUES = [
    Queue('high_priority', routing_key='high_priority'),
    Queue('normal_priority', routing_key='normal_priority'),
    Queue('low_priority', routing_key='low_priority'),
]

CELERY_ROUTES = {
    'tasks.send_email': {'queue': 'high_priority'},
    'tasks.process_document': {'queue': 'normal_priority'},
    'tasks.cleanup_old_data': {'queue': 'low_priority'},
}

def create_celery_app():
    celery = Celery('myapp')
    celery.config_from_object(CELERY_CONFIG)
    celery.conf.task_routes = CELERY_ROUTES
    return celery
```

## Monitoring and Logging

### Structured Logging

Set up comprehensive logging:

```python
# config/logging.py
import logging
import json
from datetime import datetime
from pythonjsonlogger import jsonlogger

class CustomJsonFormatter(jsonlogger.JsonFormatter):
    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)
        
        # Add timestamp
        log_record['timestamp'] = datetime.utcnow().isoformat()
        
        # Add service info
        log_record['service'] = 'api-service'
        log_record['version'] = '1.0.0'
        
        # Add request context if available
        if hasattr(record, 'request_id'):
            log_record['request_id'] = record.request_id
        if hasattr(record, 'user_id'):
            log_record['user_id'] = record.user_id

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'json': {
            '()': CustomJsonFormatter,
            'format': '%(levelname)s %(name)s %(message)s'
        },
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'json',
            'level': 'INFO'
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': 'logs/app.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,
            'formatter': 'json',
            'level': 'DEBUG'
        }
    },
    'loggers': {
        '': {  # Root logger
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': False
        },
        'sqlalchemy.engine': {
            'handlers': ['file'],
            'level': 'WARNING',
            'propagate': False
        }
    }
}
```

This advanced configuration guide provides production-ready patterns for scaling, security, and monitoring your application.