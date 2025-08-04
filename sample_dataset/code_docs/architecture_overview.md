# Architecture Overview

## System Architecture

The system follows a microservices architecture with clear separation of concerns and well-defined interfaces between components.

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Client    │    │  Mobile App     │    │  Third-party    │
│   (React)       │    │  (React Native) │    │  Integrations   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   API Gateway   │
                    │   (Kong/Nginx)  │
                    └─────────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Auth Service  │    │  User Service   │    │ Project Service │
│   (Node.js)     │    │   (Python)      │    │    (Go)         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │  Message Queue  │
                    │    (RabbitMQ)   │
                    └─────────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PostgreSQL    │    │     Redis       │    │   File Storage  │
│   (Primary DB)  │    │    (Cache)      │    │     (S3)        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Core Components

### API Gateway

The API Gateway serves as the single entry point for all client requests.

**Responsibilities:**
- Request routing and load balancing
- Authentication and authorization
- Rate limiting and throttling
- Request/response transformation
- API versioning
- Monitoring and logging

**Technology Stack:**
- Kong or Nginx Plus
- Lua scripts for custom logic
- Redis for rate limiting storage

**Configuration Example:**
```yaml
# kong.yml
_format_version: "2.1"

services:
  - name: user-service
    url: http://user-service:8080
    routes:
      - name: user-routes
        paths:
          - /api/v1/users
        methods:
          - GET
          - POST
          - PUT
          - DELETE

  - name: project-service
    url: http://project-service:8080
    routes:
      - name: project-routes
        paths:
          - /api/v1/projects

plugins:
  - name: rate-limiting
    config:
      minute: 100
      hour: 1000
  
  - name: jwt
    config:
      secret_is_base64: false
      key_claim_name: iss
```

### Authentication Service

Handles user authentication, authorization, and token management.

**Key Features:**
- JWT token generation and validation
- OAuth 2.0 / OpenID Connect support
- Multi-factor authentication
- Session management
- Password policies and security

**Core Classes:**

```javascript
// auth-service/src/models/User.js
class User {
  constructor(id, email, passwordHash, roles = [], mfaEnabled = false) {
    this.id = id;
    this.email = email;
    this.passwordHash = passwordHash;
    this.roles = roles;
    this.mfaEnabled = mfaEnabled;
    this.createdAt = new Date();
    this.lastLogin = null;
  }

  hasRole(role) {
    return this.roles.includes(role);
  }

  hasPermission(permission) {
    return this.roles.some(role => 
      ROLE_PERMISSIONS[role]?.includes(permission)
    );
  }
}

// auth-service/src/services/AuthService.js
class AuthService {
  constructor(userRepository, tokenService, passwordService) {
    this.userRepository = userRepository;
    this.tokenService = tokenService;
    this.passwordService = passwordService;
  }

  async authenticate(email, password) {
    const user = await this.userRepository.findByEmail(email);
    if (!user) {
      throw new AuthenticationError('Invalid credentials');
    }

    const isValidPassword = await this.passwordService.verify(
      password, 
      user.passwordHash
    );
    
    if (!isValidPassword) {
      throw new AuthenticationError('Invalid credentials');
    }

    if (user.mfaEnabled) {
      return { requiresMFA: true, userId: user.id };
    }

    const tokens = await this.tokenService.generateTokens(user);
    await this.userRepository.updateLastLogin(user.id);
    
    return { tokens, user: this.sanitizeUser(user) };
  }

  async refreshToken(refreshToken) {
    const payload = await this.tokenService.verifyRefreshToken(refreshToken);
    const user = await this.userRepository.findById(payload.userId);
    
    if (!user) {
      throw new AuthenticationError('Invalid refresh token');
    }

    return await this.tokenService.generateTokens(user);
  }
}
```

### User Service

Manages user profiles, preferences, and user-related operations.

**Architecture Pattern:** Domain-Driven Design (DDD)

```python
# user-service/src/domain/entities/user.py
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List
from enum import Enum

class UserStatus(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"
    SUSPENDED = "suspended"

@dataclass
class User:
    id: str
    email: str
    name: str
    status: UserStatus
    created_at: datetime
    updated_at: datetime
    last_login: Optional[datetime] = None
    profile: Optional['UserProfile'] = None
    
    def activate(self):
        if self.status == UserStatus.PENDING:
            self.status = UserStatus.ACTIVE
            self.updated_at = datetime.utcnow()
        else:
            raise ValueError("User must be in pending status to activate")
    
    def suspend(self, reason: str):
        self.status = UserStatus.SUSPENDED
        self.updated_at = datetime.utcnow()
        # Log suspension reason

# user-service/src/domain/repositories/user_repository.py
from abc import ABC, abstractmethod
from typing import List, Optional
from ..entities.user import User

class UserRepository(ABC):
    @abstractmethod
    async def save(self, user: User) -> User:
        pass
    
    @abstractmethod
    async def find_by_id(self, user_id: str) -> Optional[User]:
        pass
    
    @abstractmethod
    async def find_by_email(self, email: str) -> Optional[User]:
        pass
    
    @abstractmethod
    async def find_all(self, limit: int = 10, offset: int = 0) -> List[User]:
        pass

# user-service/src/application/services/user_service.py
class UserService:
    def __init__(self, user_repository: UserRepository, 
                 event_publisher: EventPublisher):
        self.user_repository = user_repository
        self.event_publisher = event_publisher
    
    async def create_user(self, email: str, name: str) -> User:
        # Check if user already exists
        existing_user = await self.user_repository.find_by_email(email)
        if existing_user:
            raise UserAlreadyExistsError(f"User with email {email} already exists")
        
        # Create new user
        user = User(
            id=generate_uuid(),
            email=email,
            name=name,
            status=UserStatus.PENDING,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        # Save user
        saved_user = await self.user_repository.save(user)
        
        # Publish event
        await self.event_publisher.publish(UserCreatedEvent(
            user_id=saved_user.id,
            email=saved_user.email,
            name=saved_user.name
        ))
        
        return saved_user
```

### Project Service

Manages projects, collaborations, and project-related operations.

**Architecture Pattern:** Clean Architecture with CQRS

```go
// project-service/internal/domain/project.go
package domain

import (
    "time"
    "errors"
)

type ProjectStatus string

const (
    ProjectStatusActive   ProjectStatus = "active"
    ProjectStatusArchived ProjectStatus = "archived"
    ProjectStatusDraft    ProjectStatus = "draft"
)

type Project struct {
    ID            string        `json:"id"`
    Name          string        `json:"name"`
    Description   string        `json:"description"`
    OwnerID       string        `json:"owner_id"`
    Status        ProjectStatus `json:"status"`
    CreatedAt     time.Time     `json:"created_at"`
    UpdatedAt     time.Time     `json:"updated_at"`
    Settings      ProjectSettings `json:"settings"`
    Collaborators []string      `json:"collaborators"`
}

type ProjectSettings struct {
    Public           bool `json:"public"`
    MaxCollaborators int  `json:"max_collaborators"`
    AllowComments    bool `json:"allow_comments"`
}

func (p *Project) AddCollaborator(userID string) error {
    if len(p.Collaborators) >= p.Settings.MaxCollaborators {
        return errors.New("maximum collaborators reached")
    }
    
    for _, collaborator := range p.Collaborators {
        if collaborator == userID {
            return errors.New("user is already a collaborator")
        }
    }
    
    p.Collaborators = append(p.Collaborators, userID)
    p.UpdatedAt = time.Now()
    return nil
}

// project-service/internal/application/commands/create_project.go
package commands

type CreateProjectCommand struct {
    Name        string          `json:"name" validate:"required,min=1,max=100"`
    Description string          `json:"description" validate:"max=500"`
    OwnerID     string          `json:"owner_id" validate:"required"`
    Settings    ProjectSettings `json:"settings"`
}

type CreateProjectHandler struct {
    projectRepo ProjectRepository
    eventBus    EventBus
}

func (h *CreateProjectHandler) Handle(cmd CreateProjectCommand) (*Project, error) {
    // Validate command
    if err := validate.Struct(cmd); err != nil {
        return nil, NewValidationError(err)
    }
    
    // Create project
    project := &Project{
        ID:          generateUUID(),
        Name:        cmd.Name,
        Description: cmd.Description,
        OwnerID:     cmd.OwnerID,
        Status:      ProjectStatusDraft,
        CreatedAt:   time.Now(),
        UpdatedAt:   time.Now(),
        Settings:    cmd.Settings,
        Collaborators: []string{},
    }
    
    // Save project
    if err := h.projectRepo.Save(project); err != nil {
        return nil, err
    }
    
    // Publish event
    event := ProjectCreatedEvent{
        ProjectID:   project.ID,
        Name:        project.Name,
        OwnerID:     project.OwnerID,
        CreatedAt:   project.CreatedAt,
    }
    
    if err := h.eventBus.Publish(event); err != nil {
        // Log error but don't fail the operation
        log.Error("Failed to publish ProjectCreatedEvent", err)
    }
    
    return project, nil
}
```

## Data Layer

### Database Design

**Primary Database:** PostgreSQL with read replicas

```sql
-- Users table
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_login TIMESTAMP WITH TIME ZONE
);

-- Projects table
CREATE TABLE projects (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    owner_id UUID NOT NULL REFERENCES users(id),
    status VARCHAR(20) NOT NULL DEFAULT 'draft',
    settings JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Project collaborators (many-to-many)
CREATE TABLE project_collaborators (
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role VARCHAR(50) NOT NULL DEFAULT 'collaborator',
    added_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (project_id, user_id)
);

-- Indexes for performance
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_status ON users(status);
CREATE INDEX idx_projects_owner_id ON projects(owner_id);
CREATE INDEX idx_projects_status ON projects(status);
CREATE INDEX idx_project_collaborators_user_id ON project_collaborators(user_id);
```

### Caching Strategy

**Redis Configuration:**

```python
# Caching layers
CACHE_LAYERS = {
    'L1': {  # Application cache (in-memory)
        'type': 'memory',
        'ttl': 300,  # 5 minutes
        'max_size': 1000
    },
    'L2': {  # Redis cache
        'type': 'redis',
        'ttl': 3600,  # 1 hour
        'cluster': True
    },
    'L3': {  # Database query cache
        'type': 'database',
        'ttl': 86400  # 24 hours
    }
}

# Cache key patterns
CACHE_PATTERNS = {
    'user': 'user:{user_id}',
    'user_projects': 'user:{user_id}:projects',
    'project': 'project:{project_id}',
    'project_collaborators': 'project:{project_id}:collaborators'
}
```

## Message Queue Architecture

**RabbitMQ Configuration:**

```yaml
# rabbitmq.conf
exchanges:
  - name: user.events
    type: topic
    durable: true
  
  - name: project.events
    type: topic
    durable: true

queues:
  - name: user.created
    exchange: user.events
    routing_key: user.created
    durable: true
    
  - name: project.created
    exchange: project.events
    routing_key: project.created
    durable: true
    
  - name: notification.email
    exchange: user.events
    routing_key: "*.created"
    durable: true
```

## Monitoring and Observability

### Metrics Collection

```python
# Prometheus metrics
from prometheus_client import Counter, Histogram, Gauge

# Request metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint']
)

# Business metrics
ACTIVE_USERS = Gauge(
    'active_users_total',
    'Number of active users'
)

PROJECTS_CREATED = Counter(
    'projects_created_total',
    'Total projects created'
)
```

### Distributed Tracing

```python
# OpenTelemetry configuration
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Configure tracer
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

jaeger_exporter = JaegerExporter(
    agent_host_name="jaeger",
    agent_port=6831,
)

span_processor = BatchSpanProcessor(jaeger_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# Usage in service methods
async def create_user(self, user_data):
    with tracer.start_as_current_span("create_user") as span:
        span.set_attribute("user.email", user_data.email)
        
        # Service logic here
        user = await self.user_repository.save(user_data)
        
        span.set_attribute("user.id", user.id)
        return user
```

## Security Architecture

### Security Layers

1. **Network Security**: WAF, DDoS protection, VPC isolation
2. **API Security**: Rate limiting, input validation, CORS
3. **Authentication**: JWT tokens, OAuth 2.0, MFA
4. **Authorization**: RBAC, resource-level permissions
5. **Data Security**: Encryption at rest and in transit
6. **Audit Logging**: Comprehensive security event logging

### Security Implementation

```python
# Security middleware
class SecurityMiddleware:
    def __init__(self):
        self.rate_limiter = RateLimiter()
        self.input_validator = InputValidator()
        self.audit_logger = AuditLogger()
    
    async def __call__(self, request, call_next):
        # Rate limiting
        if not await self.rate_limiter.is_allowed(request):
            return Response(status_code=429)
        
        # Input validation
        if not self.input_validator.validate(request):
            return Response(status_code=400)
        
        # Process request
        response = await call_next(request)
        
        # Audit logging
        await self.audit_logger.log_request(request, response)
        
        return response
```

This architecture provides a scalable, maintainable, and secure foundation for the application with clear separation of concerns and well-defined interfaces between components.