# Python SDK Reference

## Installation

```bash
pip install example-sdk
```

## Quick Start

```python
from example_sdk import Client

# Initialize client
client = Client(api_key="your-api-key")

# Create a user
user = client.users.create(
    email="user@example.com",
    name="John Doe"
)

print(f"Created user: {user.id}")
```

## Client Configuration

### Basic Configuration

```python
from example_sdk import Client

client = Client(
    api_key="your-api-key",
    base_url="https://api.example.com/v1",  # Optional
    timeout=30,  # Request timeout in seconds
    max_retries=3,  # Number of retry attempts
    retry_delay=1.0  # Delay between retries
)
```

### Advanced Configuration

```python
from example_sdk import Client, Config
import httpx

# Custom HTTP client configuration
http_client = httpx.Client(
    limits=httpx.Limits(max_connections=100),
    timeout=httpx.Timeout(30.0)
)

config = Config(
    api_key="your-api-key",
    http_client=http_client,
    debug=True,  # Enable debug logging
    user_agent="MyApp/1.0"
)

client = Client(config=config)
```

## User Management

### User Class

```python
class User:
    """Represents a user in the system."""
    
    def __init__(self, id: str, email: str, name: str, status: str, 
                 created_at: str, last_login: str = None):
        self.id = id
        self.email = email
        self.name = name
        self.status = status
        self.created_at = created_at
        self.last_login = last_login
    
    def __repr__(self):
        return f"User(id='{self.id}', email='{self.email}', name='{self.name}')"
    
    def to_dict(self) -> dict:
        """Convert user to dictionary representation."""
        return {
            'id': self.id,
            'email': self.email,
            'name': self.name,
            'status': self.status,
            'created_at': self.created_at,
            'last_login': self.last_login
        }
```

### UserManager

```python
class UserManager:
    """Manages user-related operations."""
    
    def __init__(self, client):
        self._client = client
    
    def create(self, email: str, name: str, password: str) -> User:
        """
        Create a new user.
        
        Args:
            email: User's email address
            name: User's full name
            password: User's password
            
        Returns:
            User: The created user object
            
        Raises:
            ValidationError: If input validation fails
            APIError: If the API request fails
        """
        data = {
            'email': email,
            'name': name,
            'password': password
        }
        
        response = self._client._post('/users', json=data)
        return User(**response)
    
    def get(self, user_id: str) -> User:
        """
        Retrieve a user by ID.
        
        Args:
            user_id: The user's unique identifier
            
        Returns:
            User: The user object
            
        Raises:
            NotFoundError: If user doesn't exist
            APIError: If the API request fails
        """
        response = self._client._get(f'/users/{user_id}')
        return User(**response)
    
    def list(self, limit: int = 10, offset: int = 0, 
             status: str = None) -> List[User]:
        """
        List users with optional filtering.
        
        Args:
            limit: Maximum number of users to return (1-100)
            offset: Number of users to skip
            status: Filter by user status ('active', 'inactive', 'pending')
            
        Returns:
            List[User]: List of user objects
        """
        params = {'limit': limit, 'offset': offset}
        if status:
            params['status'] = status
            
        response = self._client._get('/users', params=params)
        return [User(**user_data) for user_data in response['users']]
    
    def update(self, user_id: str, **kwargs) -> User:
        """
        Update a user's information.
        
        Args:
            user_id: The user's unique identifier
            **kwargs: Fields to update (name, email, status)
            
        Returns:
            User: The updated user object
        """
        response = self._client._patch(f'/users/{user_id}', json=kwargs)
        return User(**response)
    
    def delete(self, user_id: str) -> bool:
        """
        Delete a user.
        
        Args:
            user_id: The user's unique identifier
            
        Returns:
            bool: True if deletion was successful
        """
        self._client._delete(f'/users/{user_id}')
        return True
```

## Project Management

### Project Class

```python
from datetime import datetime
from typing import List, Optional, Dict, Any

class Project:
    """Represents a project in the system."""
    
    def __init__(self, id: str, name: str, description: str = None,
                 owner_id: str = None, status: str = 'active',
                 created_at: str = None, updated_at: str = None,
                 settings: Dict[str, Any] = None):
        self.id = id
        self.name = name
        self.description = description
        self.owner_id = owner_id
        self.status = status
        self.created_at = created_at
        self.updated_at = updated_at
        self.settings = settings or {}
    
    @property
    def is_public(self) -> bool:
        """Check if project is public."""
        return self.settings.get('public', False)
    
    @property
    def max_collaborators(self) -> int:
        """Get maximum number of collaborators allowed."""
        return self.settings.get('max_collaborators', 10)
    
    def add_collaborator(self, user_id: str) -> bool:
        """
        Add a collaborator to the project.
        
        Args:
            user_id: ID of the user to add as collaborator
            
        Returns:
            bool: True if successful
        """
        # This would typically make an API call
        pass
    
    def remove_collaborator(self, user_id: str) -> bool:
        """
        Remove a collaborator from the project.
        
        Args:
            user_id: ID of the user to remove
            
        Returns:
            bool: True if successful
        """
        pass
```

### ProjectManager

```python
class ProjectManager:
    """Manages project-related operations."""
    
    def __init__(self, client):
        self._client = client
    
    def create(self, name: str, description: str = None, 
               settings: Dict[str, Any] = None) -> Project:
        """
        Create a new project.
        
        Args:
            name: Project name
            description: Optional project description
            settings: Project settings dictionary
            
        Returns:
            Project: The created project object
        """
        data = {'name': name}
        if description:
            data['description'] = description
        if settings:
            data['settings'] = settings
            
        response = self._client._post('/projects', json=data)
        return Project(**response)
    
    def get(self, project_id: str) -> Project:
        """Get project by ID."""
        response = self._client._get(f'/projects/{project_id}')
        return Project(**response)
    
    def list(self, owner_id: str = None, status: str = None) -> List[Project]:
        """
        List projects with optional filtering.
        
        Args:
            owner_id: Filter by project owner
            status: Filter by project status
            
        Returns:
            List[Project]: List of project objects
        """
        params = {}
        if owner_id:
            params['owner_id'] = owner_id
        if status:
            params['status'] = status
            
        response = self._client._get('/projects', params=params)
        return [Project(**proj_data) for proj_data in response['projects']]
```

## Error Handling

### Exception Classes

```python
class ExampleSDKError(Exception):
    """Base exception for SDK errors."""
    pass

class APIError(ExampleSDKError):
    """Raised when API returns an error response."""
    
    def __init__(self, message: str, status_code: int = None, 
                 response_data: dict = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data or {}

class ValidationError(ExampleSDKError):
    """Raised when input validation fails."""
    
    def __init__(self, message: str, field: str = None):
        super().__init__(message)
        self.field = field

class NotFoundError(APIError):
    """Raised when a resource is not found."""
    pass

class RateLimitError(APIError):
    """Raised when rate limit is exceeded."""
    
    def __init__(self, message: str, retry_after: int = None):
        super().__init__(message)
        self.retry_after = retry_after

class AuthenticationError(APIError):
    """Raised when authentication fails."""
    pass
```

### Error Handling Examples

```python
from example_sdk import Client
from example_sdk.exceptions import (
    APIError, ValidationError, NotFoundError, 
    RateLimitError, AuthenticationError
)

client = Client(api_key="your-api-key")

try:
    user = client.users.create(
        email="invalid-email",  # Invalid email format
        name="John Doe",
        password="weak"  # Weak password
    )
except ValidationError as e:
    print(f"Validation error: {e}")
    if e.field:
        print(f"Field: {e.field}")

except AuthenticationError as e:
    print(f"Authentication failed: {e}")
    # Handle invalid API key

except RateLimitError as e:
    print(f"Rate limit exceeded: {e}")
    if e.retry_after:
        print(f"Retry after {e.retry_after} seconds")

except NotFoundError as e:
    print(f"Resource not found: {e}")

except APIError as e:
    print(f"API error: {e}")
    print(f"Status code: {e.status_code}")
    print(f"Response data: {e.response_data}")

except Exception as e:
    print(f"Unexpected error: {e}")
```

## Async Support

### Async Client

```python
import asyncio
from example_sdk import AsyncClient

async def main():
    async with AsyncClient(api_key="your-api-key") as client:
        # Create user asynchronously
        user = await client.users.create(
            email="async@example.com",
            name="Async User"
        )
        
        # Get user details
        user_details = await client.users.get(user.id)
        print(f"User: {user_details.name}")
        
        # List projects concurrently
        projects_task = client.projects.list()
        users_task = client.users.list(limit=50)
        
        projects, users = await asyncio.gather(projects_task, users_task)
        print(f"Found {len(projects)} projects and {len(users)} users")

# Run async code
asyncio.run(main())
```

## Utilities

### Pagination Helper

```python
from typing import Iterator, TypeVar, Generic

T = TypeVar('T')

class PaginatedResult(Generic[T]):
    """Helper for handling paginated API responses."""
    
    def __init__(self, client, endpoint: str, item_class: type, 
                 params: dict = None, limit: int = 50):
        self.client = client
        self.endpoint = endpoint
        self.item_class = item_class
        self.params = params or {}
        self.limit = limit
    
    def __iter__(self) -> Iterator[T]:
        """Iterate through all items across all pages."""
        offset = 0
        
        while True:
            current_params = {**self.params, 'limit': self.limit, 'offset': offset}
            response = self.client._get(self.endpoint, params=current_params)
            
            items = response.get('items', [])
            if not items:
                break
                
            for item_data in items:
                yield self.item_class(**item_data)
            
            if not response.get('has_more', False):
                break
                
            offset += self.limit

# Usage
for user in PaginatedResult(client, '/users', User, {'status': 'active'}):
    print(f"Processing user: {user.name}")
```

## Testing

### Mock Client

```python
from unittest.mock import Mock
from example_sdk import Client

def test_user_creation():
    # Create mock client
    mock_client = Mock(spec=Client)
    mock_client.users.create.return_value = User(
        id="test_123",
        email="test@example.com",
        name="Test User",
        status="active",
        created_at="2024-01-01T00:00:00Z"
    )
    
    # Test user creation
    user = mock_client.users.create(
        email="test@example.com",
        name="Test User",
        password="password123"
    )
    
    assert user.id == "test_123"
    assert user.email == "test@example.com"
    
    # Verify the method was called with correct arguments
    mock_client.users.create.assert_called_once_with(
        email="test@example.com",
        name="Test User",
        password="password123"
    )
```