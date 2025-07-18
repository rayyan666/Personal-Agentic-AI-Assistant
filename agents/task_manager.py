import json
import asyncio
from datetime import datetime, timedelta, time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
from croniter import croniter

from .base_agent import BaseAgent, AgentTask
from .utils import sanitize_input, generate_task_id, format_timestamp
from config import config

class TaskStatus(Enum):
    """Task status options"""
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"
    DEFERRED = "deferred"

class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5

class RecurrenceType(Enum):
    """Task recurrence types"""
    NONE = "none"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    YEARLY = "yearly"
    CUSTOM = "custom"

@dataclass
class Task:
    """Represents a task"""
    id: str
    title: str
    description: str
    status: TaskStatus
    priority: TaskPriority
    created_at: datetime
    due_date: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    estimated_duration: Optional[int] = None  # in minutes
    actual_duration: Optional[int] = None  # in minutes
    tags: List[str] = None
    dependencies: List[str] = None  # Task IDs this task depends on
    recurrence: RecurrenceType = RecurrenceType.NONE
    recurrence_pattern: Optional[str] = None  # Cron pattern for custom recurrence
    project_id: Optional[str] = None
    assigned_to: Optional[str] = None
    notes: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.dependencies is None:
            self.dependencies = []
        if self.notes is None:
            self.notes = []

@dataclass
class Project:
    """Represents a project containing multiple tasks"""
    id: str
    name: str
    description: str
    created_at: datetime
    deadline: Optional[datetime] = None
    status: str = "active"  # active, completed, archived
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []

@dataclass
class TimeBlock:
    """Represents a time block for scheduling"""
    id: str
    title: str
    start_time: datetime
    end_time: datetime
    task_id: Optional[str] = None
    description: Optional[str] = None
    is_break: bool = False

@dataclass
class ProductivityMetrics:
    """Represents productivity metrics for a time period"""
    date: datetime
    tasks_completed: int
    tasks_created: int
    total_time_spent: int  # in minutes
    focus_time: int  # in minutes
    break_time: int  # in minutes
    productivity_score: float  # 0-100

class TaskManagerAgent(BaseAgent):
    """Task Manager Agent for productivity and task management"""
    
    def __init__(self):
        super().__init__(
            name="task_manager",
            description="Manages daily tasks, scheduling, and productivity tracking"
        )
        
        self.tasks: List[Task] = []
        self.projects: List[Project] = []
        self.time_blocks: List[TimeBlock] = []
        self.productivity_metrics: List[ProductivityMetrics] = []
        self.db_path = config.task_db_path
        
        # Initialize database
        self._init_database()
        
        # Load existing data
        self._load_data()
        
        # Start background task for recurring tasks and reminders
        asyncio.create_task(self._background_task_manager())
    
    def _init_database(self):
        """Initialize SQLite database for task management"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create tasks table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT,
                    status TEXT NOT NULL,
                    priority INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    due_date TEXT,
                    completed_at TEXT,
                    estimated_duration INTEGER,
                    actual_duration INTEGER,
                    tags TEXT,
                    dependencies TEXT,
                    recurrence TEXT DEFAULT 'none',
                    recurrence_pattern TEXT,
                    project_id TEXT,
                    assigned_to TEXT,
                    notes TEXT
                )
            ''')
            
            # Create projects table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    created_at TEXT NOT NULL,
                    deadline TEXT,
                    status TEXT DEFAULT 'active',
                    tags TEXT
                )
            ''')
            
            # Create time blocks table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS time_blocks (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT NOT NULL,
                    task_id TEXT,
                    description TEXT,
                    is_break BOOLEAN DEFAULT FALSE
                )
            ''')
            
            # Create productivity metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS productivity_metrics (
                    date TEXT PRIMARY KEY,
                    tasks_completed INTEGER DEFAULT 0,
                    tasks_created INTEGER DEFAULT 0,
                    total_time_spent INTEGER DEFAULT 0,
                    focus_time INTEGER DEFAULT 0,
                    break_time INTEGER DEFAULT 0,
                    productivity_score REAL DEFAULT 0.0
                )
            ''')
            
            conn.commit()
            conn.close()
            
            self.logger.info("Task manager database initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize task database: {str(e)}")
    
    def _load_data(self):
        """Load tasks, projects, and other data from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Load tasks
            cursor.execute("SELECT * FROM tasks")
            for row in cursor.fetchall():
                task = Task(
                    id=row[0],
                    title=row[1],
                    description=row[2],
                    status=TaskStatus(row[3]),
                    priority=TaskPriority(row[4]),
                    created_at=datetime.fromisoformat(row[5]),
                    due_date=datetime.fromisoformat(row[6]) if row[6] else None,
                    completed_at=datetime.fromisoformat(row[7]) if row[7] else None,
                    estimated_duration=row[8],
                    actual_duration=row[9],
                    tags=json.loads(row[10]) if row[10] else [],
                    dependencies=json.loads(row[11]) if row[11] else [],
                    recurrence=RecurrenceType(row[12]) if row[12] else RecurrenceType.NONE,
                    recurrence_pattern=row[13],
                    project_id=row[14],
                    assigned_to=row[15],
                    notes=json.loads(row[16]) if row[16] else []
                )
                self.tasks.append(task)
            
            # Load projects
            cursor.execute("SELECT * FROM projects")
            for row in cursor.fetchall():
                project = Project(
                    id=row[0],
                    name=row[1],
                    description=row[2],
                    created_at=datetime.fromisoformat(row[3]),
                    deadline=datetime.fromisoformat(row[4]) if row[4] else None,
                    status=row[5],
                    tags=json.loads(row[6]) if row[6] else []
                )
                self.projects.append(project)
            
            # Load time blocks
            cursor.execute("SELECT * FROM time_blocks")
            for row in cursor.fetchall():
                time_block = TimeBlock(
                    id=row[0],
                    title=row[1],
                    start_time=datetime.fromisoformat(row[2]),
                    end_time=datetime.fromisoformat(row[3]),
                    task_id=row[4],
                    description=row[5],
                    is_break=bool(row[6])
                )
                self.time_blocks.append(time_block)
            
            # Load productivity metrics
            cursor.execute("SELECT * FROM productivity_metrics")
            for row in cursor.fetchall():
                metrics = ProductivityMetrics(
                    date=datetime.fromisoformat(row[0]),
                    tasks_completed=row[1],
                    tasks_created=row[2],
                    total_time_spent=row[3],
                    focus_time=row[4],
                    break_time=row[5],
                    productivity_score=row[6]
                )
                self.productivity_metrics.append(metrics)
            
            conn.close()
            self.logger.info(f"Loaded {len(self.tasks)} tasks, {len(self.projects)} projects, {len(self.time_blocks)} time blocks")
            
        except Exception as e:
            self.logger.error(f"Failed to load task data: {str(e)}")
    
    async def _process_task_impl(self, task: AgentTask) -> str:
        """Process task manager tasks"""
        task_type = task.task_type.lower()
        content = sanitize_input(task.content)
        
        if task_type == "create_task":
            return await self._create_task(content, task.metadata)
        elif task_type == "list_tasks":
            return await self._list_tasks(task.metadata)
        elif task_type == "update_task":
            return await self._update_task(content, task.metadata)
        elif task_type == "complete_task":
            return await self._complete_task(content, task.metadata)
        elif task_type == "delete_task":
            return await self._delete_task(content, task.metadata)
        elif task_type == "create_project":
            return await self._create_project(content, task.metadata)
        elif task_type == "schedule_time":
            return await self._schedule_time_block(content, task.metadata)
        elif task_type == "daily_schedule":
            return await self._get_daily_schedule(task.metadata)
        elif task_type == "productivity_report":
            return await self._generate_productivity_report(task.metadata)
        elif task_type == "task_dependencies":
            return await self._manage_task_dependencies(content, task.metadata)
        elif task_type == "recurring_tasks":
            return await self._manage_recurring_tasks(content, task.metadata)
        elif task_type == "time_tracking":
            return await self._track_time(content, task.metadata)
        elif task_type == "task_suggestions":
            return await self._get_task_suggestions(content, task.metadata)
        elif task_type == "health_check":
            return "Task Manager Agent is ready to help you stay organized and productive!"
        else:
            return await self._create_task(content, task.metadata)
    
    async def _create_task(self, description: str, metadata: Dict[str, Any]) -> str:
        """Create a new task"""
        try:
            title = metadata.get("title", description[:50])
            priority = metadata.get("priority", "medium")
            due_date_str = metadata.get("due_date")
            estimated_duration = metadata.get("estimated_duration")
            tags = metadata.get("tags", [])
            project_id = metadata.get("project_id")
            
            # Parse priority
            priority_map = {
                "low": TaskPriority.LOW,
                "medium": TaskPriority.MEDIUM,
                "high": TaskPriority.HIGH,
                "urgent": TaskPriority.URGENT,
                "critical": TaskPriority.CRITICAL
            }
            task_priority = priority_map.get(priority.lower(), TaskPriority.MEDIUM)
            
            # Parse due date
            due_date = None
            if due_date_str:
                try:
                    due_date = datetime.fromisoformat(due_date_str)
                except:
                    # Try to parse natural language dates
                    due_date = self._parse_natural_date(due_date_str)
            
            # Create task
            new_task = Task(
                id=generate_task_id(),
                title=title,
                description=description,
                status=TaskStatus.TODO,
                priority=task_priority,
                created_at=datetime.now(),
                due_date=due_date,
                estimated_duration=estimated_duration,
                tags=tags,
                project_id=project_id
            )
            
            # Save task
            self._save_task(new_task)
            self.tasks.append(new_task)
            
            # Update daily metrics
            self._update_daily_metrics("tasks_created", 1)
            
            result = f"Task Created Successfully\n\n"
            result += f"**Task ID:** {new_task.id}\n"
            result += f"**Title:** {new_task.title}\n"
            result += f"**Priority:** {new_task.priority.name.title()}\n"
            result += f"**Status:** {new_task.status.value.replace('_', ' ').title()}\n"
            if due_date:
                result += f"**Due Date:** {format_timestamp(due_date)}\n"
            if estimated_duration:
                result += f"**Estimated Duration:** {estimated_duration} minutes\n"
            if tags:
                result += f"**Tags:** {', '.join(tags)}\n"
            if project_id:
                project = self._find_project_by_id(project_id)
                if project:
                    result += f"**Project:** {project.name}\n"
            
            result += f"\n**Description:**\n{description}"
            
            # Add suggestions
            suggestions = self._get_task_creation_suggestions(new_task)
            if suggestions:
                result += f"\n\n**Suggestions:**\n"
                for suggestion in suggestions:
                    result += f"• {suggestion}\n"
            
            return result

        except Exception as e:
            self.logger.error(f"Failed to create task: {str(e)}")
            return f"Sorry, I couldn't create the task. Error: {str(e)}"

    async def _list_tasks(self, metadata: Dict[str, Any]) -> str:
        """List tasks with filtering options"""
        try:
            status_filter = metadata.get("status", "all")
            priority_filter = metadata.get("priority", "all")
            project_filter = metadata.get("project_id")
            tag_filter = metadata.get("tag")
            limit = metadata.get("limit", 20)
            sort_by = metadata.get("sort_by", "priority")  # priority, due_date, created_at

            # Filter tasks
            filtered_tasks = self.tasks

            if status_filter != "all":
                filtered_tasks = [t for t in filtered_tasks if t.status.value == status_filter]

            if priority_filter != "all":
                priority_map = {"low": 1, "medium": 2, "high": 3, "urgent": 4, "critical": 5}
                if priority_filter in priority_map:
                    filtered_tasks = [t for t in filtered_tasks if t.priority.value == priority_map[priority_filter]]

            if project_filter:
                filtered_tasks = [t for t in filtered_tasks if t.project_id == project_filter]

            if tag_filter:
                filtered_tasks = [t for t in filtered_tasks if tag_filter in t.tags]

            # Sort tasks
            if sort_by == "priority":
                filtered_tasks = sorted(filtered_tasks, key=lambda t: t.priority.value, reverse=True)
            elif sort_by == "due_date":
                filtered_tasks = sorted(filtered_tasks, key=lambda t: t.due_date or datetime.max)
            else:  # created_at
                filtered_tasks = sorted(filtered_tasks, key=lambda t: t.created_at, reverse=True)

            result = f"Task List\n\n"
            result += f"**Total Tasks:** {len(self.tasks)}\n"
            result += f"**Filtered Results:** {len(filtered_tasks)}\n"
            if status_filter != "all":
                result += f"**Status Filter:** {status_filter.replace('_', ' ').title()}\n"
            if priority_filter != "all":
                result += f"**Priority Filter:** {priority_filter.title()}\n"
            result += f"**Sorted by:** {sort_by.replace('_', ' ').title()}\n\n"

            if not filtered_tasks:
                result += "No tasks match your criteria."
                return result

            # Group by status for better organization
            tasks_by_status = {}
            for task in filtered_tasks[:limit]:
                status = task.status.value
                if status not in tasks_by_status:
                    tasks_by_status[status] = []
                tasks_by_status[status].append(task)

            status_emojis = {
                "todo": "📋",
                "in_progress": "🔄",
                "completed": "✅",
                "cancelled": "❌",
                "blocked": "🚫",
                "deferred": "⏸️"
            }

            for status, tasks in tasks_by_status.items():
                emoji = status_emojis.get(status, "📝")
                result += f"**{emoji} {status.replace('_', ' ').title()} ({len(tasks)}):**\n"

                for task in tasks:
                    priority_indicator = "🔴" if task.priority.value >= 4 else "🟡" if task.priority.value == 3 else "🟢"
                    result += f"{priority_indicator} **{task.title}**\n"
                    result += f"   ID: {task.id}\n"
                    result += f"   Priority: {task.priority.name.title()}\n"
                    if task.due_date:
                        days_until_due = (task.due_date - datetime.now()).days
                        if days_until_due < 0:
                            result += f"   Due: {format_timestamp(task.due_date)} (⚠️ OVERDUE)\n"
                        elif days_until_due == 0:
                            result += f"   Due: {format_timestamp(task.due_date)} (📅 TODAY)\n"
                        else:
                            result += f"   Due: {format_timestamp(task.due_date)} ({days_until_due} days)\n"
                    if task.estimated_duration:
                        result += f"   Estimated: {task.estimated_duration} minutes\n"
                    if task.tags:
                        result += f"   Tags: {', '.join(task.tags)}\n"
                    result += "\n"

            if len(filtered_tasks) > limit:
                result += f"\n... and {len(filtered_tasks) - limit} more tasks"

            return result

        except Exception as e:
            self.logger.error(f"Failed to list tasks: {str(e)}")
            return f"Sorry, I couldn't list the tasks. Error: {str(e)}"

    async def _update_task(self, task_identifier: str, metadata: Dict[str, Any]) -> str:
        """Update an existing task"""
        try:
            # Find task
            task = self._find_task_by_id_or_title(task_identifier)
            if not task:
                return f"Task '{task_identifier}' not found."

            # Update fields
            updated_fields = []

            if "title" in metadata:
                task.title = metadata["title"]
                updated_fields.append("title")

            if "description" in metadata:
                task.description = metadata["description"]
                updated_fields.append("description")

            if "status" in metadata:
                try:
                    task.status = TaskStatus(metadata["status"])
                    updated_fields.append("status")
                except ValueError:
                    pass

            if "priority" in metadata:
                priority_map = {
                    "low": TaskPriority.LOW,
                    "medium": TaskPriority.MEDIUM,
                    "high": TaskPriority.HIGH,
                    "urgent": TaskPriority.URGENT,
                    "critical": TaskPriority.CRITICAL
                }
                if metadata["priority"].lower() in priority_map:
                    task.priority = priority_map[metadata["priority"].lower()]
                    updated_fields.append("priority")

            if "due_date" in metadata:
                if metadata["due_date"]:
                    try:
                        task.due_date = datetime.fromisoformat(metadata["due_date"])
                    except:
                        task.due_date = self._parse_natural_date(metadata["due_date"])
                else:
                    task.due_date = None
                updated_fields.append("due_date")

            if "estimated_duration" in metadata:
                task.estimated_duration = metadata["estimated_duration"]
                updated_fields.append("estimated_duration")

            if "tags" in metadata:
                task.tags = metadata["tags"]
                updated_fields.append("tags")

            if "add_note" in metadata:
                task.notes.append(f"{datetime.now().isoformat()}: {metadata['add_note']}")
                updated_fields.append("notes")

            # Save updated task
            self._save_task(task)

            result = f"Task Updated Successfully\n\n"
            result += f"**Task:** {task.title}\n"
            result += f"**ID:** {task.id}\n"
            result += f"**Updated Fields:** {', '.join(updated_fields)}\n\n"

            result += f"**Current Status:**\n"
            result += f"• Status: {task.status.value.replace('_', ' ').title()}\n"
            result += f"• Priority: {task.priority.name.title()}\n"
            if task.due_date:
                result += f"• Due Date: {format_timestamp(task.due_date)}\n"
            if task.estimated_duration:
                result += f"• Estimated Duration: {task.estimated_duration} minutes\n"
            if task.tags:
                result += f"• Tags: {', '.join(task.tags)}\n"

            return result

        except Exception as e:
            self.logger.error(f"Failed to update task: {str(e)}")
            return f"Sorry, I couldn't update the task. Error: {str(e)}"

    async def _complete_task(self, task_identifier: str, metadata: Dict[str, Any]) -> str:
        """Mark a task as completed"""
        try:
            task = self._find_task_by_id_or_title(task_identifier)
            if not task:
                return f"Task '{task_identifier}' not found."

            if task.status == TaskStatus.COMPLETED:
                return f"Task '{task.title}' is already completed."

            # Mark as completed
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()

            # Record actual duration if provided
            if "actual_duration" in metadata:
                task.actual_duration = metadata["actual_duration"]

            # Save task
            self._save_task(task)

            # Update daily metrics
            self._update_daily_metrics("tasks_completed", 1)
            if task.actual_duration:
                self._update_daily_metrics("total_time_spent", task.actual_duration)

            # Handle recurring tasks
            if task.recurrence != RecurrenceType.NONE:
                next_task = self._create_recurring_task(task)
                if next_task:
                    self._save_task(next_task)
                    self.tasks.append(next_task)

            result = f"Task Completed! 🎉\n\n"
            result += f"**Task:** {task.title}\n"
            result += f"**Completed At:** {format_timestamp(task.completed_at)}\n"

            if task.estimated_duration and task.actual_duration:
                variance = task.actual_duration - task.estimated_duration
                if variance > 0:
                    result += f"**Time:** {task.actual_duration} minutes ({variance} minutes over estimate)\n"
                elif variance < 0:
                    result += f"**Time:** {task.actual_duration} minutes ({abs(variance)} minutes under estimate)\n"
                else:
                    result += f"**Time:** {task.actual_duration} minutes (exactly as estimated!)\n"
            elif task.actual_duration:
                result += f"**Time Spent:** {task.actual_duration} minutes\n"

            # Check for dependent tasks
            dependent_tasks = [t for t in self.tasks if task.id in t.dependencies]
            if dependent_tasks:
                result += f"\n**Unblocked Tasks:**\n"
                for dep_task in dependent_tasks:
                    if dep_task.status == TaskStatus.BLOCKED:
                        dep_task.status = TaskStatus.TODO
                        self._save_task(dep_task)
                        result += f"• {dep_task.title}\n"

            if task.recurrence != RecurrenceType.NONE:
                result += f"\n**Next Occurrence:** A new recurring task has been created."

            return result

        except Exception as e:
            self.logger.error(f"Failed to complete task: {str(e)}")
            return f"Sorry, I couldn't complete the task. Error: {str(e)}"

    async def _get_daily_schedule(self, metadata: Dict[str, Any]) -> str:
        """Get daily schedule and task overview"""
        try:
            date_str = metadata.get("date", datetime.now().strftime("%Y-%m-%d"))
            try:
                target_date = datetime.fromisoformat(date_str).date()
            except:
                target_date = datetime.now().date()

            # Get tasks due today
            today_tasks = [
                t for t in self.tasks
                if t.due_date and t.due_date.date() == target_date and t.status != TaskStatus.COMPLETED
            ]

            # Get scheduled time blocks for today
            today_blocks = [
                b for b in self.time_blocks
                if b.start_time.date() == target_date
            ]

            # Get overdue tasks
            overdue_tasks = [
                t for t in self.tasks
                if t.due_date and t.due_date.date() < target_date and t.status not in [TaskStatus.COMPLETED, TaskStatus.CANCELLED]
            ]

            result = f"Daily Schedule for {target_date}\n\n"

            # Overdue tasks warning
            if overdue_tasks:
                result += f"⚠️ **Overdue Tasks ({len(overdue_tasks)}):**\n"
                for task in overdue_tasks[:5]:
                    days_overdue = (target_date - task.due_date.date()).days
                    result += f"• {task.title} ({days_overdue} days overdue)\n"
                result += "\n"

            # Today's tasks
            if today_tasks:
                result += f"📋 **Tasks Due Today ({len(today_tasks)}):**\n"
                for task in sorted(today_tasks, key=lambda t: t.priority.value, reverse=True):
                    priority_indicator = "🔴" if task.priority.value >= 4 else "🟡" if task.priority.value == 3 else "🟢"
                    status_indicator = "🔄" if task.status == TaskStatus.IN_PROGRESS else "📋"
                    result += f"{priority_indicator}{status_indicator} {task.title}"
                    if task.estimated_duration:
                        result += f" ({task.estimated_duration}m)"
                    result += "\n"
                result += "\n"

            # Scheduled time blocks
            if today_blocks:
                result += f"🕐 **Scheduled Time Blocks ({len(today_blocks)}):**\n"
                for block in sorted(today_blocks, key=lambda b: b.start_time):
                    time_range = f"{block.start_time.strftime('%H:%M')} - {block.end_time.strftime('%H:%M')}"
                    block_type = "☕ Break" if block.is_break else "💼 Work"
                    result += f"{block_type} {time_range}: {block.title}\n"
                result += "\n"

            # Daily summary
            total_estimated_time = sum(t.estimated_duration or 0 for t in today_tasks)
            scheduled_time = sum((b.end_time - b.start_time).total_seconds() / 60 for b in today_blocks if not b.is_break)

            result += f"📊 **Daily Summary:**\n"
            result += f"• Tasks due: {len(today_tasks)}\n"
            result += f"• Estimated work time: {total_estimated_time} minutes\n"
            result += f"• Scheduled time: {int(scheduled_time)} minutes\n"

            if total_estimated_time > scheduled_time:
                result += f"⚠️ You may need {total_estimated_time - int(scheduled_time)} more minutes\n"

            # Productivity tips
            if today_tasks:
                result += f"\n💡 **Productivity Tips:**\n"
                high_priority_tasks = [t for t in today_tasks if t.priority.value >= 3]
                if high_priority_tasks:
                    result += f"• Focus on {len(high_priority_tasks)} high-priority tasks first\n"

                if total_estimated_time > 480:  # More than 8 hours
                    result += f"• Consider breaking down large tasks or deferring some to tomorrow\n"

                result += f"• Take regular breaks to maintain productivity\n"

            return result

        except Exception as e:
            self.logger.error(f"Failed to get daily schedule: {str(e)}")
            return f"Sorry, I couldn't generate the daily schedule. Error: {str(e)}"

    # Helper methods
    def _find_task_by_id_or_title(self, identifier: str) -> Optional[Task]:
        """Find task by ID or partial title match"""
        # First try exact ID match
        for task in self.tasks:
            if task.id == identifier:
                return task

        # Then try partial title match
        for task in self.tasks:
            if identifier.lower() in task.title.lower():
                return task

        return None

    def _find_project_by_id(self, project_id: str) -> Optional[Project]:
        """Find project by ID"""
        for project in self.projects:
            if project.id == project_id:
                return project
        return None

    def _parse_natural_date(self, date_str: str) -> Optional[datetime]:
        """Parse natural language dates"""
        now = datetime.now()
        date_str = date_str.lower().strip()

        if date_str in ["today"]:
            return now.replace(hour=23, minute=59, second=59)
        elif date_str in ["tomorrow"]:
            return now + timedelta(days=1)
        elif date_str in ["next week"]:
            return now + timedelta(weeks=1)
        elif "days" in date_str:
            try:
                days = int(date_str.split()[0])
                return now + timedelta(days=days)
            except:
                pass

        return None

    def _save_task(self, task: Task):
        """Save task to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT OR REPLACE INTO tasks
                (id, title, description, status, priority, created_at, due_date, completed_at,
                 estimated_duration, actual_duration, tags, dependencies, recurrence,
                 recurrence_pattern, project_id, assigned_to, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                task.id, task.title, task.description, task.status.value, task.priority.value,
                task.created_at.isoformat(),
                task.due_date.isoformat() if task.due_date else None,
                task.completed_at.isoformat() if task.completed_at else None,
                task.estimated_duration, task.actual_duration,
                json.dumps(task.tags), json.dumps(task.dependencies),
                task.recurrence.value, task.recurrence_pattern,
                task.project_id, task.assigned_to, json.dumps(task.notes)
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            self.logger.error(f"Failed to save task: {str(e)}")

    def _update_daily_metrics(self, metric: str, value: int):
        """Update daily productivity metrics"""
        try:
            today = datetime.now().date()

            # Find or create today's metrics
            today_metrics = None
            for metrics in self.productivity_metrics:
                if metrics.date.date() == today:
                    today_metrics = metrics
                    break

            if not today_metrics:
                today_metrics = ProductivityMetrics(
                    date=datetime.now(),
                    tasks_completed=0,
                    tasks_created=0,
                    total_time_spent=0,
                    focus_time=0,
                    break_time=0,
                    productivity_score=0.0
                )
                self.productivity_metrics.append(today_metrics)

            # Update the specific metric
            if metric == "tasks_completed":
                today_metrics.tasks_completed += value
            elif metric == "tasks_created":
                today_metrics.tasks_created += value
            elif metric == "total_time_spent":
                today_metrics.total_time_spent += value
            elif metric == "focus_time":
                today_metrics.focus_time += value
            elif metric == "break_time":
                today_metrics.break_time += value

            # Recalculate productivity score
            today_metrics.productivity_score = self._calculate_productivity_score(today_metrics)

            # Save to database
            self._save_productivity_metrics(today_metrics)

        except Exception as e:
            self.logger.error(f"Failed to update daily metrics: {str(e)}")

    def _calculate_productivity_score(self, metrics: ProductivityMetrics) -> float:
        """Calculate productivity score based on various factors"""
        score = 0.0

        # Base score from completed tasks
        if metrics.tasks_completed > 0:
            score += min(metrics.tasks_completed * 10, 50)  # Max 50 points for tasks

        # Bonus for focus time
        if metrics.focus_time > 0:
            score += min(metrics.focus_time / 10, 30)  # Max 30 points for focus time

        # Penalty for too much break time
        if metrics.break_time > metrics.focus_time:
            score -= 10

        # Bonus for balanced work
        if 240 <= metrics.total_time_spent <= 480:  # 4-8 hours
            score += 20

        return max(0, min(100, score))

    def _save_productivity_metrics(self, metrics: ProductivityMetrics):
        """Save productivity metrics to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT OR REPLACE INTO productivity_metrics
                (date, tasks_completed, tasks_created, total_time_spent, focus_time, break_time, productivity_score)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                metrics.date.isoformat(), metrics.tasks_completed, metrics.tasks_created,
                metrics.total_time_spent, metrics.focus_time, metrics.break_time,
                metrics.productivity_score
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            self.logger.error(f"Failed to save productivity metrics: {str(e)}")

    def _get_task_creation_suggestions(self, task: Task) -> List[str]:
        """Get suggestions for a newly created task"""
        suggestions = []

        if not task.due_date:
            suggestions.append("Consider setting a due date to help with prioritization")

        if not task.estimated_duration:
            suggestions.append("Add an estimated duration to help with time planning")

        if not task.tags:
            suggestions.append("Add tags to help categorize and find this task later")

        if task.priority == TaskPriority.HIGH and not task.due_date:
            suggestions.append("High priority tasks should typically have due dates")

        return suggestions

    def _create_recurring_task(self, completed_task: Task) -> Optional[Task]:
        """Create next occurrence of a recurring task"""
        try:
            if completed_task.recurrence == RecurrenceType.NONE:
                return None

            next_due_date = None
            if completed_task.due_date:
                if completed_task.recurrence == RecurrenceType.DAILY:
                    next_due_date = completed_task.due_date + timedelta(days=1)
                elif completed_task.recurrence == RecurrenceType.WEEKLY:
                    next_due_date = completed_task.due_date + timedelta(weeks=1)
                elif completed_task.recurrence == RecurrenceType.MONTHLY:
                    next_due_date = completed_task.due_date + timedelta(days=30)  # Approximate
                elif completed_task.recurrence == RecurrenceType.YEARLY:
                    next_due_date = completed_task.due_date + timedelta(days=365)  # Approximate
                elif completed_task.recurrence == RecurrenceType.CUSTOM and completed_task.recurrence_pattern:
                    # Use croniter for custom patterns
                    try:
                        cron = croniter(completed_task.recurrence_pattern, completed_task.due_date)
                        next_due_date = cron.get_next(datetime)
                    except:
                        pass

            if next_due_date:
                next_task = Task(
                    id=generate_task_id(),
                    title=completed_task.title,
                    description=completed_task.description,
                    status=TaskStatus.TODO,
                    priority=completed_task.priority,
                    created_at=datetime.now(),
                    due_date=next_due_date,
                    estimated_duration=completed_task.estimated_duration,
                    tags=completed_task.tags.copy(),
                    dependencies=completed_task.dependencies.copy(),
                    recurrence=completed_task.recurrence,
                    recurrence_pattern=completed_task.recurrence_pattern,
                    project_id=completed_task.project_id,
                    assigned_to=completed_task.assigned_to
                )
                return next_task

            return None

        except Exception as e:
            self.logger.error(f"Failed to create recurring task: {str(e)}")
            return None

    async def _background_task_manager(self):
        """Background task for handling recurring tasks and reminders"""
        while True:
            try:
                # Check for overdue tasks and send reminders
                now = datetime.now()
                overdue_tasks = [
                    t for t in self.tasks
                    if t.due_date and t.due_date < now and t.status not in [TaskStatus.COMPLETED, TaskStatus.CANCELLED]
                ]

                if overdue_tasks:
                    self.logger.info(f"Found {len(overdue_tasks)} overdue tasks")

                # Sleep for 1 hour before next check
                await asyncio.sleep(3600)

            except Exception as e:
                self.logger.error(f"Error in background task manager: {str(e)}")
                await asyncio.sleep(3600)
