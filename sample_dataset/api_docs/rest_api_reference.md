# REST API Reference

## Authentication

All API requests require authentication using an API key in the header:

```
Authorization: Bearer YOUR_API_KEY
```

## Base URL

```
https://api.example.com/v1
```

## Endpoints

### Users

#### GET /users

Retrieve a list of users.

**Parameters:**
- `limit` (integer, optional): Maximum number of users to return (default: 10, max: 100)
- `offset` (integer, optional): Number of users to skip (default: 0)
- `status` (string, optional): Filter by user status (`active`, `inactive`, `pending`)

**Response:**
```json
{
  "users": [
    {
      "id": "user_123",
      "email": "user@example.com",
      "name": "John Doe",
      "status": "active",
      "created_at": "2024-01-15T10:30:00Z",
      "last_login": "2024-01-20T14:22:00Z"
    }
  ],
  "total": 150,
  "has_more": true
}
```

#### POST /users

Create a new user.

**Request Body:**
```json
{
  "email": "newuser@example.com",
  "name": "Jane Smith",
  "password": "secure_password_123"
}
```

**Response:**
```json
{
  "id": "user_456",
  "email": "newuser@example.com",
  "name": "Jane Smith",
  "status": "pending",
  "created_at": "2024-01-21T09:15:00Z"
}
```

### Projects

#### GET /projects/{project_id}

Retrieve project details by ID.

**Path Parameters:**
- `project_id` (string, required): The unique project identifier

**Response:**
```json
{
  "id": "proj_789",
  "name": "My Project",
  "description": "A sample project for demonstration",
  "owner_id": "user_123",
  "status": "active",
  "created_at": "2024-01-10T08:00:00Z",
  "updated_at": "2024-01-20T16:45:00Z",
  "settings": {
    "public": false,
    "max_collaborators": 10
  }
}
```

## Error Handling

The API uses standard HTTP status codes:

- `200` - Success
- `201` - Created
- `400` - Bad Request
- `401` - Unauthorized
- `403` - Forbidden
- `404` - Not Found
- `429` - Rate Limited
- `500` - Internal Server Error

Error responses include details:

```json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "The email field is required",
    "details": {
      "field": "email",
      "reason": "missing_required_field"
    }
  }
}
```

## Rate Limiting

API requests are limited to 1000 requests per hour per API key. Rate limit headers are included in responses:

```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1642781400
```