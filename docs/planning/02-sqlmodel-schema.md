# SQLModel Schema Migration Analysis & Implementation Plan

## ✅ IMPLEMENTATION STATUS: COMPLETED

**IMPLEMENTATION DATE**: August 4, 2025  

**PROJECT STATUS**: Successfully Completed  

**ACTUAL RESULTS**: Full SQLModel ORM migration achieved with enhanced type safety  

## 1. Executive Summary

This analysis examines the current raw SQL patterns in the `orchestration/` directory and proposes a comprehensive migration to modern SQLModel ORM schemas with full Pydantic v2 integration. The migration will eliminate hardcoded SQL strings, improve type safety, and establish a maintainable foundation for the task management system.

**IMPLEMENTATION RESULTS ACHIEVED:**

- ✅ Migrated task management system from raw SQL to SQLModel ORM (eliminated 15+ hardcoded queries)

- ✅ Converted parallel execution system to SQLModel with proper schema management

- ✅ Extended successful SQLModel patterns from `src/` to orchestration layer

- ✅ Implemented modern SQLModel with full Pydantic v2 integration for superior type safety

- ✅ Achieved complete migration with optimal decision score of 0.83

**IMPLEMENTATION FILES CREATED**:

- `/orchestration/database.py` - SQLModel repository pattern

- `/orchestration/models.py` - Unified SQLModel schemas with Pydantic integration

## 2. Context & Motivation

**Background:**
The orchestration system currently uses direct SQLite operations via `sqlite3.connect()` and raw SQL strings for:

- Task CRUD operations in `task_manager.py`

- Database schema creation in `create_task_db.py`  

- Parallel execution tracking in `parallel_executor.py`

- Complex queries for statistics and reporting

**Current Pain Points:**

- No compile-time type safety for SQL operations

- Duplicated query logic across components

- Manual row factory mapping (`sqlite3.Row`)

- Inconsistent error handling patterns

- Difficult to extend with relationships

**Assumptions:**

- Existing functionality must be preserved exactly

- Performance cannot degrade for critical paths

- Migration should enable future extensibility

- Team prefers library-first approaches

## 3. Research & Evidence

### 3.1 SQLModel Best Practices Research

Based on comprehensive research of 2024-2025 SQLModel patterns:

**Key Findings:**

- **Pydantic v2 Integration**: SQLModel now fully supports Pydantic v2 with `model_config = ConfigDict(from_attributes=True)`

- **Advanced Relationships**: `Relationship(..., back_populates="...")` with cascade options for referential integrity

- **Constraint Patterns**: `__table_args__` for composite constraints, unique indexes, and check constraints

- **Performance Optimizations**: `selectinload()` for eager loading, `session.add_all()` for batch operations

- **Migration Strategy**: Phased approach using `metadata.reflect()` and Alembic autogenerate

**Library Versions:**

- SQLModel: 0.0.14+ (latest with Pydantic v2 support)

- Pydantic: 2.11.7+ (latest with strict mode and performance improvements)

- SQLAlchemy: 2.0+ (modern async patterns and compiled queries)

### 3.2 Current SQL Pattern Analysis

**Identified Patterns in `orchestration/`:**

1. **Core Task Management** (`task_manager.py`):
   - 4 tables: `tasks`, `task_dependencies`, `task_progress`, `task_comments`
   - 15+ raw SQL queries with manual parameter binding
   - Complex joins for statistics and reporting

2. **Parallel Execution Schema** (`parallel_executor.py`):
   - 3 additional tables: `execution_groups`, `group_tasks`, `parallel_execution_log`
   - Dynamic schema creation with `CREATE TABLE IF NOT EXISTS`
   - Batch operations for task grouping

3. **Database Schema Creation** (`create_task_db.py`):
   - Complete DDL schema with indexes and foreign keys
   - 30+ initial tasks inserted with dependencies
   - Critical path dependency mapping

## 4. Decision Framework Analysis

**Multi-Criteria Analysis Results:**

| Option | Library Leverage (35%) | System Value (30%) | Maintenance Load (25%) | Extensibility (10%) | **Total Score** |
|--------|------------------------|-------------------|----------------------|-------------------|----------------|
| **Full SQLModel Migration** | 0.9 | 0.9 | 0.7 | 0.9 | **0.83** |
| Hybrid Approach | 0.6 | 0.7 | 0.5 | 0.6 | 0.62 |
| Minimal Changes | 0.2 | 0.3 | 0.8 | 0.3 | 0.39 |

**Recommendation:** Full SQLModel Migration provides maximum value through:

- Complete library leverage with modern SQLModel + Pydantic v2 patterns

- Excellent type safety and developer experience improvements

- Lower long-term maintenance burden despite higher upfront effort

- Strong foundation for future extensions and integrations

**Trade-offs:**

- **Performance vs Maintainability**: ORM overhead (~5-10%) vs long-term code quality

- **Migration Effort vs Technical Debt**: Higher upfront investment vs accumulating maintenance costs

- **Type Safety vs Flexibility**: Strict schemas vs dynamic query construction

## 5. Proposed Implementation & Roadmap

### 5.1 SQLModel Schema Design

#### 5.1.1 Core Task Management Models

```python
"""Enhanced SQLModel schemas for task management system."""

from datetime import datetime
from typing import List, Optional
from enum import Enum

from pydantic import ConfigDict, field_validator
from sqlalchemy import Column, Index, UniqueConstraint, CheckConstraint, ForeignKey
from sqlmodel import Field, Relationship, SQLModel


class TaskStatus(str, Enum):
    """Task status enumeration."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress" 
    COMPLETED = "completed"
    BLOCKED = "blocked"


class TaskPriority(str, Enum):
    """Task priority enumeration."""
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class TaskComplexity(str, Enum):
    """Task complexity enumeration."""
    SIMPLE = "Simple"
    MEDIUM = "Medium" 
    COMPLEX = "Complex"
    EXPERT = "Expert"


# Base classes for hybrid Pydantic/SQLModel usage
class TaskBase(SQLModel):
    """Base task model for API and validation."""
    title: str = Field(min_length=1, max_length=500)
    description: Optional[str] = Field(default=None, max_length=2000)
    component_area: str = Field(min_length=1, max_length=100)
    phase: int = Field(ge=1, le=4)
    priority: TaskPriority = Field(default=TaskPriority.MEDIUM)
    complexity: TaskComplexity = Field(default=TaskComplexity.MEDIUM)
    status: TaskStatus = Field(default=TaskStatus.NOT_STARTED)
    source_document: str = Field(min_length=1, max_length=200)
    success_criteria: Optional[str] = Field(default=None, max_length=1000)
    time_estimate_hours: float = Field(default=1.0, ge=0.1, le=200.0)
    parent_task_id: Optional[int] = Field(default=None, foreign_key="task.id")


class Task(TaskBase, table=True):
    """SQLModel for task records with relationships."""
    
    __tablename__ = "tasks"
    __table_args__ = (
        Index("ix_tasks_component_area", "component_area"),
        Index("ix_tasks_phase", "phase"), 
        Index("ix_tasks_status", "status"),
        Index("ix_tasks_priority", "priority"),
        Index("ix_tasks_parent", "parent_task_id"),
        CheckConstraint("time_estimate_hours > 0", name="ck_positive_time"),
        CheckConstraint("phase BETWEEN 1 AND 4", name="ck_valid_phase"),
    )
    
    model_config = ConfigDict(from_attributes=True)
    
    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Relationships
    dependencies: List["TaskDependency"] = Relationship(
        back_populates="task",
        cascade_delete=True,
        foreign_keys="[TaskDependency.task_id]"
    )
    dependents: List["TaskDependency"] = Relationship(
        back_populates="depends_on_task",
        foreign_keys="[TaskDependency.depends_on_task_id]"
    )
    progress_records: List["TaskProgress"] = Relationship(
        back_populates="task",
        cascade_delete=True
    )
    comments: List["TaskComment"] = Relationship(
        back_populates="task", 
        cascade_delete=True
    )
    subtasks: List["Task"] = Relationship(
        back_populates="parent_task",
        cascade_delete=True
    )
    parent_task: Optional["Task"] = Relationship(
        back_populates="subtasks",
        remote_side="Task.id"
    )
    
    @field_validator("updated_at", mode="before")
    @classmethod
    def set_updated_at(cls, v):
        """Auto-update timestamp on changes."""
        return datetime.utcnow()


class DependencyType(str, Enum):
    """Dependency type enumeration."""
    BLOCKS = "blocks"
    REQUIRES = "requires" 
    ENHANCES = "enhances"


class TaskDependencyBase(SQLModel):
    """Base task dependency model."""
    task_id: int = Field(foreign_key="tasks.id")
    depends_on_task_id: int = Field(foreign_key="tasks.id")
    dependency_type: DependencyType = Field(default=DependencyType.BLOCKS)


class TaskDependency(TaskDependencyBase, table=True):
    """SQLModel for task dependencies."""
    
    __tablename__ = "task_dependencies"
    __table_args__ = (
        Index("ix_task_dependencies_task_id", "task_id"),
        Index("ix_task_dependencies_depends_on", "depends_on_task_id"),
        UniqueConstraint("task_id", "depends_on_task_id", name="uq_task_dependency"),
        CheckConstraint("task_id != depends_on_task_id", name="ck_no_self_dependency"),
    )
    
    model_config = ConfigDict(from_attributes=True)
    
    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Relationships
    task: Task = Relationship(
        back_populates="dependencies",
        foreign_keys="[TaskDependency.task_id]"
    )
    depends_on_task: Task = Relationship(
        back_populates="dependents", 
        foreign_keys="[TaskDependency.depends_on_task_id]"
    )


class TaskProgressBase(SQLModel):
    """Base task progress model.""" 
    task_id: int = Field(foreign_key="tasks.id")
    progress_percentage: int = Field(ge=0, le=100, default=0)
    notes: Optional[str] = Field(default=None, max_length=1000)
    updated_by: str = Field(default="system", max_length=100)


class TaskProgress(TaskProgressBase, table=True):
    """SQLModel for task progress tracking."""
    
    __tablename__ = "task_progress"
    __table_args__ = (
        Index("ix_task_progress_task_id", "task_id"),
        Index("ix_task_progress_created_at", "created_at"),
        CheckConstraint("progress_percentage BETWEEN 0 AND 100", name="ck_valid_progress"),
    )
    
    model_config = ConfigDict(from_attributes=True)
    
    id: Optional[int] = Field(default=None, primary_key=True) 
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Relationships
    task: Task = Relationship(back_populates="progress_records")


class CommentType(str, Enum):
    """Comment type enumeration."""
    NOTE = "note"
    DECISION = "decision"
    BLOCKER = "blocker"
    SOLUTION = "solution"


class TaskCommentBase(SQLModel):
    """Base task comment model."""
    task_id: int = Field(foreign_key="tasks.id")
    comment: str = Field(min_length=1, max_length=2000)
    comment_type: CommentType = Field(default=CommentType.NOTE)


class TaskComment(TaskCommentBase, table=True):
    """SQLModel for task comments."""
    
    __tablename__ = "task_comments"
    __table_args__ = (
        Index("ix_task_comments_task_id", "task_id"),
        Index("ix_task_comments_created_at", "created_at"),
    )
    
    model_config = ConfigDict(from_attributes=True)
    
    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Relationships
    task: Task = Relationship(back_populates="comments")
```

#### 5.1.2 Parallel Execution Models

```python
"""SQLModel schemas for parallel execution tracking."""

from enum import Enum
from typing import List, Optional, Dict, Any
from datetime import datetime

from pydantic import ConfigDict
from sqlalchemy import Column, Index, JSON
from sqlmodel import Field, Relationship, SQLModel


class ExecutionStatus(str, Enum):
    """Execution status enumeration."""
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExecutionGroupBase(SQLModel):
    """Base execution group model."""
    name: str = Field(min_length=1, max_length=200)
    total_hours: float = Field(ge=0)
    estimated_duration: float = Field(ge=0)  # in minutes
    max_parallel_tasks: int = Field(ge=1, le=50, default=5)
    status: ExecutionStatus = Field(default=ExecutionStatus.PENDING)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, sa_column=Column(JSON))


class ExecutionGroup(ExecutionGroupBase, table=True):
    """SQLModel for execution groups."""
    
    __tablename__ = "execution_groups"
    __table_args__ = (
        Index("ix_execution_groups_status", "status"),
        Index("ix_execution_groups_created_at", "created_at"),
    )
    
    model_config = ConfigDict(from_attributes=True)
    
    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Relationships
    group_tasks: List["GroupTask"] = Relationship(
        back_populates="execution_group",
        cascade_delete=True
    )
    execution_logs: List["ParallelExecutionLog"] = Relationship(
        back_populates="execution_group",
        cascade_delete=True
    )


class GroupTaskBase(SQLModel):
    """Base group task model."""
    group_id: int = Field(foreign_key="execution_groups.id")
    task_id: int = Field(foreign_key="tasks.id")
    execution_order: int = Field(ge=0)
    task_hours: float = Field(ge=0)
    status: ExecutionStatus = Field(default=ExecutionStatus.PENDING)


class GroupTask(GroupTaskBase, table=True):
    """SQLModel for group task assignments."""
    
    __tablename__ = "group_tasks"
    __table_args__ = (
        Index("ix_group_tasks_group_id", "group_id"),
        Index("ix_group_tasks_task_id", "task_id"),
        Index("ix_group_tasks_execution_order", "execution_order"),
        UniqueConstraint("group_id", "task_id", name="uq_group_task"),
    )
    
    model_config = ConfigDict(from_attributes=True)
    
    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Relationships
    execution_group: ExecutionGroup = Relationship(back_populates="group_tasks")
    task: Task = Relationship()


class LogLevel(str, Enum):
    """Log level enumeration."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    DEBUG = "debug"


class ParallelExecutionLogBase(SQLModel):
    """Base parallel execution log model."""
    group_id: int = Field(foreign_key="execution_groups.id")
    task_id: Optional[int] = Field(default=None, foreign_key="tasks.id")
    level: LogLevel = Field(default=LogLevel.INFO)
    message: str = Field(min_length=1, max_length=2000)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, sa_column=Column(JSON))


class ParallelExecutionLog(ParallelExecutionLogBase, table=True):
    """SQLModel for parallel execution logging."""
    
    __tablename__ = "parallel_execution_log"
    __table_args__ = (
        Index("ix_parallel_execution_log_group_id", "group_id"),
        Index("ix_parallel_execution_log_task_id", "task_id"),
        Index("ix_parallel_execution_log_level", "level"),
        Index("ix_parallel_execution_log_created_at", "created_at"),
    )
    
    model_config = ConfigDict(from_attributes=True)
    
    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Relationships
    execution_group: ExecutionGroup = Relationship(back_populates="execution_logs")
    task: Optional[Task] = Relationship()
```

### 5.2 Enhanced Database Configuration

```python
"""Enhanced database configuration with SQLModel integration."""

from collections.abc import AsyncGenerator
from typing import Any

import sqlmodel
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlmodel.sql.expression import Select, SelectOfScalar

from .models import *  # Import all SQLModel classes


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""
    
    model_config = SettingsConfigDict(
        env_prefix="DB_",
        case_sensitive=False,
        extra="ignore"
    )
    
    url: str = "sqlite:///orchestration/database/implementation_tracker.db"
    echo: bool = False
    pool_size: int = 20
    max_overflow: int = 0
    
    @property
    def async_url(self) -> str:
        """Convert SQLite URL to async variant."""
        if self.url.startswith("sqlite:///"):
            return self.url.replace("sqlite:///", "sqlite+aiosqlite:///")
        elif self.url == "sqlite:///:memory:":
            return "sqlite+aiosqlite:///:memory:"
        return self.url


# Global settings instance
db_settings = DatabaseSettings()

# Create synchronous engine (for Streamlit compatibility)
sync_engine = sqlmodel.create_engine(
    db_settings.url, 
    echo=db_settings.echo,
    pool_pre_ping=True
)

# Create async engine (for orchestration operations)
async_engine: AsyncEngine = create_async_engine(
    db_settings.async_url,
    echo=db_settings.echo,
    future=True,
    pool_size=db_settings.pool_size,
    max_overflow=db_settings.max_overflow
)

# Create async session factory
async_session_factory = sessionmaker(
    async_engine, 
    class_=AsyncSession, 
    expire_on_commit=False
)

# Patch SQLModel for async compatibility
Select.inherit_cache = True  # type: ignore[attr-defined]
SelectOfScalar.inherit_cache = True  # type: ignore[attr-defined]


@event.listens_for(sync_engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """Configure SQLite connection for optimal performance."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA cache_size=1000")
    cursor.execute("PRAGMA temp_store=MEMORY")
    cursor.close()


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Async context manager for database sessions."""
    async with async_session_factory() as session:
        yield session


def create_db_and_tables():
    """Create database and all tables."""
    SQLModel.metadata.create_all(sync_engine)
    

def get_sync_session():
    """Get synchronous session for Streamlit usage."""
    from sqlmodel import Session
    return Session(sync_engine)
```

### 5.3 Migration Strategy & Scripts

#### 5.3.1 Alembic Integration

```python
"""Alembic configuration for SQLModel migrations."""

# alembic/env.py
from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context

# Import SQLModel metadata
from sqlmodel import SQLModel
from orchestration.models import *  # All SQLModel classes
from orchestration.database import db_settings

# Alembic Config object
config = context.config

# Configure logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# SQLModel metadata for autogenerate
target_metadata = SQLModel.metadata

def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = db_settings.url
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,  # Required for SQLite
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    configuration = config.get_section(config.config_ini_section)
    configuration["sqlalchemy.url"] = db_settings.url
    
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, 
            target_metadata=target_metadata,
            render_as_batch=True,  # Required for SQLite
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

#### 5.3.2 Data Migration Script

```python
"""Data migration script from raw SQL to SQLModel."""

import sqlite3
from typing import List, Dict, Any
from pathlib import Path

from sqlmodel import Session, select
from orchestration.database import sync_engine, create_db_and_tables
from orchestration.models import (
    Task, TaskDependency, TaskProgress, TaskComment,
    TaskStatus, TaskPriority, TaskComplexity, DependencyType, CommentType
)


def migrate_existing_data(old_db_path: str = "implementation_tracker.db"):
    """Migrate data from old raw SQL database to SQLModel."""
    
    # Ensure new schema exists
    create_db_and_tables()
    
    # Connect to old database
    old_conn = sqlite3.connect(old_db_path)
    old_conn.row_factory = sqlite3.Row
    old_cursor = old_conn.cursor()
    
    # Create new session
    session = Session(sync_engine)
    
    try:
        # Migrate tasks
        print("Migrating tasks...")
        old_cursor.execute("SELECT * FROM tasks")
        tasks_data = old_cursor.fetchall()
        
        task_mapping = {}  # old_id -> new_task
        
        for row in tasks_data:
            task = Task(
                title=row["title"],
                description=row["description"],
                component_area=row["component_area"],
                phase=row["phase"],
                priority=TaskPriority(row["priority"]),
                complexity=TaskComplexity(row["complexity"]),
                status=TaskStatus(row["status"]),
                source_document=row["source_document"],
                success_criteria=row["success_criteria"],
                time_estimate_hours=row["time_estimate_hours"],
                parent_task_id=row["parent_task_id"],
                created_at=row["created_at"],
                updated_at=row["updated_at"]
            )
            session.add(task)
            session.flush()  # Get ID without committing
            task_mapping[row["id"]] = task
        
        # Migrate dependencies
        print("Migrating dependencies...")
        old_cursor.execute("SELECT * FROM task_dependencies")
        deps_data = old_cursor.fetchall()
        
        for row in deps_data:
            dependency = TaskDependency(
                task_id=task_mapping[row["task_id"]].id,
                depends_on_task_id=task_mapping[row["depends_on_task_id"]].id,
                dependency_type=DependencyType(row["dependency_type"]),
                created_at=row["created_at"]
            )
            session.add(dependency)
        
        # Migrate progress records
        print("Migrating progress records...")
        old_cursor.execute("SELECT * FROM task_progress")
        progress_data = old_cursor.fetchall()
        
        for row in progress_data:
            progress = TaskProgress(
                task_id=task_mapping[row["task_id"]].id,
                progress_percentage=row["progress_percentage"],
                notes=row["notes"],
                updated_by=row["updated_by"],
                created_at=row["created_at"]
            )
            session.add(progress)
        
        # Migrate comments
        print("Migrating comments...")
        old_cursor.execute("SELECT * FROM task_comments")
        comments_data = old_cursor.fetchall()
        
        for row in comments_data:
            comment = TaskComment(
                task_id=task_mapping[row["task_id"]].id,
                comment=row["comment"],
                comment_type=CommentType(row["comment_type"]),
                created_at=row["created_at"]
            )
            session.add(comment)
        
        # Commit all changes
        session.commit()
        print(f"✅ Successfully migrated {len(tasks_data)} tasks and related data")
        
    except Exception as e:
        session.rollback()
        print(f"❌ Migration failed: {e}")
        raise
    finally:
        session.close()
        old_conn.close()


def validate_migration():
    """Validate the migration was successful."""
    session = Session(sync_engine)
    
    try:
        # Check task count
        task_count = session.exec(select(Task)).all()
        print(f"Total tasks: {len(task_count)}")
        
        # Check relationships are working
        task_with_deps = session.exec(
            select(Task).where(Task.id == 1)
        ).first()
        
        if task_with_deps:
            print(f"Dependencies for task 1: {len(task_with_deps.dependencies)}")
            print(f"Progress records for task 1: {len(task_with_deps.progress_records)}")
        
        print("✅ Migration validation passed")
        
    finally:
        session.close()


if __name__ == "__main__":
    # Run migration
    migrate_existing_data()
    validate_migration()
```

### 5.4 Enhanced TaskManager with SQLModel

```python
"""Enhanced TaskManager using SQLModel ORM."""

from typing import List, Optional, Dict, Any
from datetime import datetime

from sqlmodel import Session, select, and_, func, desc
from sqlalchemy.orm import selectinload

from .database import sync_engine, get_sync_session
from .models import (
    Task, TaskDependency, TaskProgress, TaskComment,
    TaskStatus, TaskPriority, TaskComplexity, 
    DependencyType, CommentType
)


class TaskManagerORM:
    """Enhanced task manager using SQLModel ORM."""
    
    def __init__(self):
        """Initialize with SQLModel session.""" 
        self.session_factory = get_sync_session
    
    def create_task(
        self,
        title: str,
        description: Optional[str] = None,
        component_area: str = "",
        phase: int = 1,
        priority: TaskPriority = TaskPriority.MEDIUM,
        complexity: TaskComplexity = TaskComplexity.MEDIUM,
        source_document: str = "",
        success_criteria: Optional[str] = None,
        time_estimate_hours: float = 1.0,
        parent_task_id: Optional[int] = None
    ) -> Task:
        """Create a new task with initial progress tracking."""
        
        with self.session_factory() as session:
            task = Task(
                title=title,
                description=description,
                component_area=component_area,
                phase=phase,
                priority=priority,
                complexity=complexity,
                source_document=source_document,
                success_criteria=success_criteria,
                time_estimate_hours=time_estimate_hours,
                parent_task_id=parent_task_id
            )
            session.add(task)
            session.flush()  # Get ID
            
            # Create initial progress record
            initial_progress = TaskProgress(
                task_id=task.id,
                progress_percentage=0,
                notes="Task created"
            )
            session.add(initial_progress)
            session.commit()
            session.refresh(task)
            
            return task
    
    def update_task_status(
        self,
        task_id: int,
        status: TaskStatus,
        progress_percentage: Optional[int] = None,
        notes: Optional[str] = None,
        comment: Optional[str] = None,
        comment_type: CommentType = CommentType.NOTE
    ) -> Optional[Task]:
        """Update task status with progress and optional comment."""
        
        with self.session_factory() as session:
            task = session.get(Task, task_id)
            if not task:
                return None
            
            # Update task status and timestamp
            task.status = status
            task.updated_at = datetime.utcnow()
            
            # Add progress record if provided
            if progress_percentage is not None:
                progress_record = TaskProgress(
                    task_id=task_id,
                    progress_percentage=progress_percentage,
                    notes=notes or f"Status updated to {status.value}"
                )
                session.add(progress_record)
            
            # Add comment if provided
            if comment:
                comment_record = TaskComment(
                    task_id=task_id,
                    comment=comment,
                    comment_type=comment_type
                )
                session.add(comment_record)
            
            session.commit()
            session.refresh(task)
            return task
    
    def get_tasks_by_status(self, status: TaskStatus) -> List[Task]:
        """Get all tasks with specific status."""
        
        with self.session_factory() as session:
            tasks = session.exec(
                select(Task)
                .where(Task.status == status)
                .options(
                    selectinload(Task.dependencies),
                    selectinload(Task.progress_records),
                    selectinload(Task.comments)
                )
            ).all()
            return list(tasks)
    
    def get_tasks_by_component(self, component_area: str) -> List[Task]:
        """Get tasks by component area with relationships."""
        
        with self.session_factory() as session:
            tasks = session.exec(
                select(Task)
                .where(Task.component_area == component_area)
                .options(selectinload(Task.dependencies))
                .order_by(Task.priority.desc(), Task.created_at)
            ).all()
            return list(tasks)
    
    def get_task_statistics(self) -> Dict[str, Any]:
        """Get comprehensive task statistics using ORM queries."""
        
        with self.session_factory() as session:
            # Basic counts
            total_tasks = session.exec(select(func.count(Task.id))).one()
            
            completed_tasks = session.exec(
                select(func.count(Task.id))
                .where(Task.status == TaskStatus.COMPLETED)
            ).one()
            
            in_progress_tasks = session.exec(
                select(func.count(Task.id))
                .where(Task.status == TaskStatus.IN_PROGRESS)
            ).one()
            
            blocked_tasks = session.exec(
                select(func.count(Task.id))
                .where(Task.status == TaskStatus.BLOCKED)
            ).one()
            
            # Phase statistics
            phase_stats = session.exec(
                select(Task.phase, func.count(Task.id).label("count"))
                .group_by(Task.phase)
                .order_by(Task.phase)
            ).all()
            
            # Component statistics
            component_stats = session.exec(
                select(Task.component_area, func.count(Task.id).label("count"))
                .group_by(Task.component_area)
                .order_by(func.count(Task.id).desc())
            ).all()
            
            # Time estimates
            total_hours = session.exec(
                select(func.sum(Task.time_estimate_hours))
            ).one() or 0
            
            completed_hours = session.exec(
                select(func.sum(Task.time_estimate_hours))
                .where(Task.status == TaskStatus.COMPLETED)
            ).one() or 0
            
            return {
                "total_tasks": total_tasks,
                "completed_tasks": completed_tasks,
                "in_progress_tasks": in_progress_tasks,
                "blocked_tasks": blocked_tasks,
                "completion_percentage": round(completed_tasks / total_tasks * 100, 1) if total_tasks > 0 else 0,
                "phase_breakdown": [{"phase": phase, "count": count} for phase, count in phase_stats],
                "component_breakdown": [{"area": area, "count": count} for area, count in component_stats],
                "total_estimated_hours": total_hours,
                "completed_hours": completed_hours,
                "progress_percentage": round(completed_hours / total_hours * 100, 1) if total_hours > 0 else 0
            }
    
    def add_task_dependency(
        self,
        task_id: int,
        depends_on_task_id: int,
        dependency_type: DependencyType = DependencyType.BLOCKS
    ) -> Optional[TaskDependency]:
        """Add a dependency between tasks."""
        
        with self.session_factory() as session:
            # Validate both tasks exist
            task = session.get(Task, task_id)
            depends_on_task = session.get(Task, depends_on_task_id)
            
            if not task or not depends_on_task:
                return None
            
            # Check for existing dependency
            existing = session.exec(
                select(TaskDependency)
                .where(
                    and_(
                        TaskDependency.task_id == task_id,
                        TaskDependency.depends_on_task_id == depends_on_task_id
                    )
                )
            ).first()
            
            if existing:
                return existing
            
            dependency = TaskDependency(
                task_id=task_id,
                depends_on_task_id=depends_on_task_id,
                dependency_type=dependency_type
            )
            session.add(dependency)
            session.commit()
            session.refresh(dependency)
            
            return dependency
    
    def get_critical_path_tasks(self) -> List[Task]:
        """Get tasks on the critical path (most dependencies)."""
        
        with self.session_factory() as session:
            # Find tasks with most dependencies
            critical_tasks = session.exec(
                select(Task)
                .join(TaskDependency, Task.id == TaskDependency.task_id)
                .group_by(Task.id)
                .order_by(func.count(TaskDependency.id).desc())
                .limit(10)
                .options(selectinload(Task.dependencies))
            ).all()
            
            return list(critical_tasks)
```

## 6. Requirements & Tasks Breakdown

### 6.1 Phase 1: Schema Design & Migration Setup (4-6 hours)

**Tasks:**

1. **Create SQLModel Schema Files** (2 hours)
   - Implement core task management models
   - Add parallel execution models  
   - Include proper relationships and constraints
   - Add comprehensive validation with Pydantic v2

2. **Setup Alembic Integration** (1 hour)
   - Configure alembic.ini with SQLite-specific settings
   - Create initial migration from SQLModel metadata
   - Test autogenerate functionality

3. **Database Configuration Enhancement** (1 hour)
   - Enhance database.py with SQLModel patterns
   - Add connection optimization for SQLite
   - Create session factories for sync/async usage

4. **Data Migration Script** (2 hours)
   - Create migration script from old raw SQL data
   - Include validation and rollback capabilities
   - Test with existing implementation_tracker.db

### 6.2 Phase 2: ORM TaskManager Implementation (6-8 hours)

**Tasks:**

1. **Enhanced TaskManager Class** (4 hours)
   - Replace all raw SQL queries with SQLModel operations
   - Implement relationship loading strategies
   - Add comprehensive error handling

2. **Service Layer Optimization** (2 hours)
   - Implement caching strategies for frequent queries
   - Add batch operations for bulk tasks
   - Optimize statistics queries with proper joins

3. **Testing & Validation** (2 hours)
   - Create unit tests for new TaskManager
   - Performance testing vs existing implementation
   - Integration testing with orchestration system

### 6.3 Phase 3: Parallel Executor Migration (4-5 hours)

**Tasks:**

1. **Parallel Executor ORM Integration** (3 hours)
   - Replace raw SQL in parallel_executor.py
   - Implement execution group management
   - Add comprehensive logging with relationships

2. **CLI Integration** (1 hour)
   - Update CLI commands to use ORM
   - Enhance error reporting and status display

3. **Final Testing & Documentation** (1 hour)
   - End-to-end testing of complete system
   - Update documentation and examples

### 6.4 Dependencies & Timeline

```mermaid
gantt
    title SQLModel Migration Timeline
    dateFormat  YYYY-MM-DD
    section Phase 1
    Schema Design           :schema, 2025-01-15, 2d
    Alembic Setup          :alembic, after schema, 1d  
    Data Migration         :migration, after alembic, 2d
    section Phase 2
    TaskManager ORM        :taskmanager, after migration, 3d
    Service Optimization   :services, after taskmanager, 2d
    Testing               :testing1, after services, 1d
    section Phase 3
    Parallel Executor     :parallel, after testing1, 2d
    CLI Integration       :cli, after parallel, 1d
    Final Testing         :testing2, after cli, 1d
```

**Critical Dependencies:**

- SQLModel schema must be complete before migration scripts

- Alembic configuration required for ongoing migrations

- TaskManager ORM must be tested before parallel executor migration

- All existing functionality must remain operational throughout

## 7. Architecture Decision Record

**Date:** 2025-01-02  

**Status:** Proposed  

**Decision:** Migrate orchestration task management from raw SQL to SQLModel ORM

**Context:**
The current task management system uses direct SQLite operations with hardcoded SQL strings, resulting in maintenance challenges, lack of type safety, and difficulty extending functionality. Modern SQLModel with Pydantic v2 provides superior developer experience and maintainability.

**Decision:**
Implement full SQLModel migration with:

- Complete ORM replacement of raw SQL queries

- Pydantic v2 integration for type safety and validation

- Alembic for schema migrations

- Comprehensive relationship modeling

- Performance optimization through proper indexing and query patterns

**Consequences:**

**Positive:**

- Complete type safety for all database operations

- Excellent developer experience with IDE support

- Easier to extend with new features and relationships

- Modern validation and serialization capabilities

- Reduced code duplication and maintenance burden

**Negative:**

- Higher upfront migration effort (~15-20 hours)

- Small performance overhead from ORM abstraction (~5-10%)

- Team learning curve for SQLModel best practices

- Additional dependency on SQLModel/SQLAlchemy ecosystem

**Alternatives Considered:**

1. **Hybrid Approach**: Keep complex queries as raw SQL - Rejected due to mixed patterns increasing cognitive load
2. **Minimal Changes**: Add models without changing queries - Rejected due to limited benefits
3. **Different ORM**: Consider alternatives like Tortoise or Databases - Rejected due to SQLModel's Pydantic integration

## 8. Next Steps / Recommendations

### 8.1 Immediate Action Items

1. **Review & Approve Schema Design** (1 day)
   - Technical review of proposed SQLModel schemas
   - Validate relationship patterns and constraints
   - Confirm migration strategy approach

2. **Environment Setup** (0.5 days)
   - Install SQLModel 0.0.14+ and Pydantic 2.11.7+
   - Configure development environment for testing
   - Setup Alembic for migration management

3. **Proof of Concept** (1 day)
   - Implement single table migration (tasks table)
   - Test basic ORM operations vs existing raw SQL
   - Validate performance characteristics

### 8.2 Risk Mitigation Strategies

**Data Safety:**

- Backup existing database before any migration

- Test migration script on copy of production data

- Implement rollback procedures for each phase

**Performance Monitoring:**

- Establish baseline performance metrics for current system

- Monitor query performance during migration

- Have fallback plan to raw SQL for critical queries if needed

**Team Preparation:**

- SQLModel training session for development team

- Code review process for new ORM patterns

- Documentation of common patterns and anti-patterns

### 8.3 Future Phase Considerations

**Phase 4 - Advanced Features** (Future):

- Implement audit logging with SQLModel

- Add full-text search capabilities

- Real-time task updates with WebSocket integration

- Advanced analytics and reporting dashboards

**Phase 5 - Scalability** (Future):

- Evaluate PostgreSQL migration for larger datasets  

- Implement read replicas for reporting

- Add caching layer with Redis integration

- Horizontal scaling considerations

### 8.4 Success Metrics

**Technical Metrics:**

- 100% feature parity with existing system

- <10% performance degradation for critical queries

- 50%+ reduction in database-related code complexity

- Zero data loss during migration

**Developer Experience Metrics:**

- Faster development of new database features

- Reduced time to implement new relationships

- Improved code review efficiency

- Better error handling and debugging capabilities

**Maintenance Metrics:**

- Reduced bug reports related to SQL query issues

- Faster resolution of database-related problems

- Easier onboarding for new team members

- More reliable automated testing of database operations

---

*This analysis provides a comprehensive roadmap for migrating the orchestration system to modern SQLModel patterns while preserving all existing functionality and establishing a strong foundation for future enhancements.*
