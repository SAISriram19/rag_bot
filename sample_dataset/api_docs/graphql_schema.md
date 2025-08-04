# GraphQL API Schema

## Overview

This GraphQL API provides access to user and project data with real-time subscriptions.

**Endpoint:** `https://api.example.com/graphql`

## Schema Definition

### Types

#### User
```graphql
type User {
  id: ID!
  email: String!
  name: String!
  status: UserStatus!
  createdAt: DateTime!
  lastLogin: DateTime
  projects: [Project!]!
  profile: UserProfile
}

enum UserStatus {
  ACTIVE
  INACTIVE
  PENDING
  SUSPENDED
}

type UserProfile {
  avatar: String
  bio: String
  location: String
  website: String
}
```

#### Project
```graphql
type Project {
  id: ID!
  name: String!
  description: String
  owner: User!
  collaborators: [User!]!
  status: ProjectStatus!
  createdAt: DateTime!
  updatedAt: DateTime!
  settings: ProjectSettings!
}

enum ProjectStatus {
  ACTIVE
  ARCHIVED
  DRAFT
}

type ProjectSettings {
  public: Boolean!
  maxCollaborators: Int!
  allowComments: Boolean!
}
```

### Queries

#### Get User
```graphql
query GetUser($id: ID!) {
  user(id: $id) {
    id
    email
    name
    status
    createdAt
    projects {
      id
      name
      status
    }
  }
}
```

#### List Users
```graphql
query ListUsers($limit: Int = 10, $offset: Int = 0, $status: UserStatus) {
  users(limit: $limit, offset: $offset, status: $status) {
    nodes {
      id
      email
      name
      status
      createdAt
    }
    totalCount
    hasNextPage
  }
}
```

#### Search Projects
```graphql
query SearchProjects($query: String!, $limit: Int = 10) {
  searchProjects(query: $query, limit: $limit) {
    id
    name
    description
    owner {
      name
      email
    }
    createdAt
  }
}
```

### Mutations

#### Create User
```graphql
mutation CreateUser($input: CreateUserInput!) {
  createUser(input: $input) {
    user {
      id
      email
      name
      status
    }
    errors {
      field
      message
    }
  }
}

input CreateUserInput {
  email: String!
  name: String!
  password: String!
}
```

#### Update Project
```graphql
mutation UpdateProject($id: ID!, $input: UpdateProjectInput!) {
  updateProject(id: $id, input: $input) {
    project {
      id
      name
      description
      updatedAt
    }
    errors {
      field
      message
    }
  }
}

input UpdateProjectInput {
  name: String
  description: String
  settings: ProjectSettingsInput
}

input ProjectSettingsInput {
  public: Boolean
  maxCollaborators: Int
  allowComments: Boolean
}
```

### Subscriptions

#### Project Updates
```graphql
subscription ProjectUpdated($projectId: ID!) {
  projectUpdated(projectId: $projectId) {
    id
    name
    updatedAt
    updatedBy {
      name
      email
    }
  }
}
```

## Example Queries

### Fetch user with projects
```graphql
{
  user(id: "user_123") {
    name
    email
    projects {
      name
      status
      collaborators {
        name
      }
    }
  }
}
```

### Create a new project
```graphql
mutation {
  createProject(input: {
    name: "New Project"
    description: "A test project"
    settings: {
      public: false
      maxCollaborators: 5
    }
  }) {
    project {
      id
      name
    }
    errors {
      field
      message
    }
  }
}
```

## Error Handling

GraphQL errors are returned in the `errors` array:

```json
{
  "data": null,
  "errors": [
    {
      "message": "User not found",
      "locations": [{"line": 2, "column": 3}],
      "path": ["user"],
      "extensions": {
        "code": "NOT_FOUND",
        "userId": "invalid_id"
      }
    }
  ]
}
```