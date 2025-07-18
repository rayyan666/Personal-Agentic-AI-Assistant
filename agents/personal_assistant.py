import asyncio
import json
import re
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import sqlite3
import schedule
from dataclasses import dataclass

from .base_agent import BaseAgent, AgentTask
from .utils import generate_task_id, sanitize_input, format_timestamp, validate_email

@dataclass
class Reminder:
    """Represents a reminder"""
    id: str
    title: str
    description: str
    due_date: datetime
    is_recurring: bool = False
    recurrence_pattern: Optional[str] = None
    is_completed: bool = False
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

@dataclass
class Appointment:
    """Represents an appointment"""
    id: str
    title: str
    description: str
    start_time: datetime
    end_time: datetime
    location: Optional[str] = None
    attendees: List[str] = None
    is_cancelled: bool = False
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.attendees is None:
            self.attendees = []

class PersonalAssistantAgent(BaseAgent):
    """Personal Assistant Agent for day-to-day tasks"""
    
    def __init__(self):
        super().__init__(
            name="personal_assistant",
            description="Handles day-to-day tasks, scheduling, reminders, and general assistance"
        )
        
        self.reminders: List[Reminder] = []
        self.appointments: List[Appointment] = []
        self.db_path = "personal_assistant.db"
        
        # Initialize database
        self._init_database()
        
        # Load existing data
        self._load_data()
        
        # Start reminder checker
        asyncio.create_task(self._reminder_checker())
    
    def _init_database(self):
        """Initialize SQLite database for persistent storage"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create reminders table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS reminders (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT,
                    due_date TEXT NOT NULL,
                    is_recurring BOOLEAN DEFAULT FALSE,
                    recurrence_pattern TEXT,
                    is_completed BOOLEAN DEFAULT FALSE,
                    created_at TEXT NOT NULL
                )
            ''')
            
            # Create appointments table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS appointments (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT,
                    start_time TEXT NOT NULL,
                    end_time TEXT NOT NULL,
                    location TEXT,
                    attendees TEXT,
                    is_cancelled BOOLEAN DEFAULT FALSE,
                    created_at TEXT NOT NULL
                )
            ''')
            
            conn.commit()
            conn.close()
            
            self.logger.info("Database initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {str(e)}")
    
    def _load_data(self):
        """Load reminders and appointments from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Load reminders
            cursor.execute("SELECT * FROM reminders WHERE is_completed = FALSE")
            for row in cursor.fetchall():
                reminder = Reminder(
                    id=row[0],
                    title=row[1],
                    description=row[2],
                    due_date=datetime.fromisoformat(row[3]),
                    is_recurring=bool(row[4]),
                    recurrence_pattern=row[5],
                    is_completed=bool(row[6]),
                    created_at=datetime.fromisoformat(row[7])
                )
                self.reminders.append(reminder)
            
            # Load appointments
            cursor.execute("SELECT * FROM appointments WHERE is_cancelled = FALSE")
            for row in cursor.fetchall():
                attendees = json.loads(row[6]) if row[6] else []
                appointment = Appointment(
                    id=row[0],
                    title=row[1],
                    description=row[2],
                    start_time=datetime.fromisoformat(row[3]),
                    end_time=datetime.fromisoformat(row[4]),
                    location=row[5],
                    attendees=attendees,
                    is_cancelled=bool(row[7]),
                    created_at=datetime.fromisoformat(row[8])
                )
                self.appointments.append(appointment)
            
            conn.close()
            self.logger.info(f"Loaded {len(self.reminders)} reminders and {len(self.appointments)} appointments")
            
        except Exception as e:
            self.logger.error(f"Failed to load data: {str(e)}")
    
    async def _process_task_impl(self, task: AgentTask) -> str:
        """Process personal assistant tasks"""
        task_type = task.task_type.lower()
        content = sanitize_input(task.content)
        
        if task_type == "general_question":
            return await self._handle_general_question(content)
        elif task_type == "create_reminder":
            return await self._create_reminder(content, task.metadata)
        elif task_type == "list_reminders":
            return await self._list_reminders()
        elif task_type == "create_appointment":
            return await self._create_appointment(content, task.metadata)
        elif task_type == "list_appointments":
            return await self._list_appointments()
        elif task_type == "daily_schedule":
            return await self._get_daily_schedule(task.metadata)
        elif task_type == "weather_info":
            return await self._get_weather_info(task.metadata)
        elif task_type == "health_check":
            return "Personal Assistant Agent is healthy and ready to help!"
        else:
            return await self._handle_general_question(content)
    
    async def _handle_general_question(self, question: str) -> str:
        """Handle general questions and provide assistance"""
        # Use the LLM to generate a helpful response
        prompt = f"As a helpful personal assistant, please provide a useful and friendly response to this question: {question}"
        
        response = self.generate_response(prompt)
        
        # If no LLM response, provide a basic fallback
        if not response or "fallback mode" in response.lower():
            if "time" in question.lower() or "date" in question.lower():
                return f"The current time is {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            elif "help" in question.lower():
                return self._get_help_message()
            else:
                return f"I understand you're asking about: {question}. I'm here to help with scheduling, reminders, appointments, and general assistance. How can I help you today?"
        
        return response
    
    async def _create_reminder(self, content: str, metadata: Dict[str, Any]) -> str:
        """Create a new reminder"""
        try:
            # Extract reminder details from content or metadata
            title = metadata.get("title", content[:50])
            description = metadata.get("description", content)
            due_date_str = metadata.get("due_date")
            
            if not due_date_str:
                # Try to extract date from content
                due_date = self._extract_date_from_text(content)
                if not due_date:
                    due_date = datetime.now() + timedelta(hours=1)  # Default to 1 hour from now
            else:
                due_date = datetime.fromisoformat(due_date_str)
            
            reminder = Reminder(
                id=generate_task_id(),
                title=title,
                description=description,
                due_date=due_date,
                is_recurring=metadata.get("is_recurring", False),
                recurrence_pattern=metadata.get("recurrence_pattern")
            )
            
            # Save to database
            self._save_reminder(reminder)
            self.reminders.append(reminder)
            
            return f"Reminder created successfully: '{title}' scheduled for {format_timestamp(due_date)}"
            
        except Exception as e:
            self.logger.error(f"Failed to create reminder: {str(e)}")
            return f"Sorry, I couldn't create the reminder. Error: {str(e)}"
    
    async def _list_reminders(self) -> str:
        """List all active reminders"""
        if not self.reminders:
            return "You have no active reminders."
        
        reminder_list = ["Your active reminders:"]
        for reminder in sorted(self.reminders, key=lambda r: r.due_date):
            status = "⏰" if reminder.due_date > datetime.now() else "🔔"
            reminder_list.append(
                f"{status} {reminder.title} - Due: {format_timestamp(reminder.due_date)}"
            )
        
        return "\n".join(reminder_list)
    
    async def _create_appointment(self, content: str, metadata: Dict[str, Any]) -> str:
        """Create a new appointment"""
        try:
            title = metadata.get("title", content[:50])
            description = metadata.get("description", content)
            start_time_str = metadata.get("start_time")
            end_time_str = metadata.get("end_time")
            location = metadata.get("location")
            attendees = metadata.get("attendees", [])
            
            if not start_time_str:
                return "Please provide a start time for the appointment."
            
            start_time = datetime.fromisoformat(start_time_str)
            end_time = datetime.fromisoformat(end_time_str) if end_time_str else start_time + timedelta(hours=1)
            
            appointment = Appointment(
                id=generate_task_id(),
                title=title,
                description=description,
                start_time=start_time,
                end_time=end_time,
                location=location,
                attendees=attendees
            )
            
            # Save to database
            self._save_appointment(appointment)
            self.appointments.append(appointment)
            
            return f"Appointment created: '{title}' from {format_timestamp(start_time)} to {format_timestamp(end_time)}"
            
        except Exception as e:
            self.logger.error(f"Failed to create appointment: {str(e)}")
            return f"Sorry, I couldn't create the appointment. Error: {str(e)}"
    
    async def _list_appointments(self) -> str:
        """List upcoming appointments"""
        now = datetime.now()
        upcoming = [apt for apt in self.appointments if apt.start_time > now and not apt.is_cancelled]
        
        if not upcoming:
            return "You have no upcoming appointments."
        
        appointment_list = ["Your upcoming appointments:"]
        for apt in sorted(upcoming, key=lambda a: a.start_time):
            location_str = f" at {apt.location}" if apt.location else ""
            appointment_list.append(
                f"📅 {apt.title} - {format_timestamp(apt.start_time)}{location_str}"
            )
        
        return "\n".join(appointment_list)
    
    async def _get_daily_schedule(self, metadata: Dict[str, Any]) -> str:
        """Get daily schedule for a specific date"""
        date_str = metadata.get("date", datetime.now().strftime("%Y-%m-%d"))
        try:
            target_date = datetime.fromisoformat(date_str).date()
        except:
            target_date = datetime.now().date()
        
        # Get appointments for the day
        day_appointments = [
            apt for apt in self.appointments
            if apt.start_time.date() == target_date and not apt.is_cancelled
        ]
        
        # Get reminders for the day
        day_reminders = [
            rem for rem in self.reminders
            if rem.due_date.date() == target_date and not rem.is_completed
        ]
        
        schedule_parts = [f"Schedule for {target_date}:"]
        
        if day_appointments:
            schedule_parts.append("\nAppointments:")
            for apt in sorted(day_appointments, key=lambda a: a.start_time):
                schedule_parts.append(f"  📅 {format_timestamp(apt.start_time)} - {apt.title}")
        
        if day_reminders:
            schedule_parts.append("\nReminders:")
            for rem in sorted(day_reminders, key=lambda r: r.due_date):
                schedule_parts.append(f"  ⏰ {format_timestamp(rem.due_date)} - {rem.title}")
        
        if not day_appointments and not day_reminders:
            schedule_parts.append("\nNo appointments or reminders scheduled for this day.")
        
        return "\n".join(schedule_parts)
    
    async def _get_weather_info(self, metadata: Dict[str, Any]) -> str:
        """Get weather information (placeholder - would integrate with weather API)"""
        location = metadata.get("location", "your location")
        return f"I'd love to help you with weather information for {location}, but I need to be connected to a weather service. Please check your local weather app or website for current conditions."
    
    def _extract_date_from_text(self, text: str) -> Optional[datetime]:
        """Extract date/time from natural language text"""
        # Simple date extraction - in production, use more sophisticated NLP
        now = datetime.now()
        
        if "tomorrow" in text.lower():
            return now + timedelta(days=1)
        elif "next week" in text.lower():
            return now + timedelta(weeks=1)
        elif "in an hour" in text.lower():
            return now + timedelta(hours=1)
        elif "tonight" in text.lower():
            return now.replace(hour=20, minute=0, second=0, microsecond=0)
        
        # Try to find time patterns
        time_pattern = r'(\d{1,2}):(\d{2})\s*(am|pm)?'
        match = re.search(time_pattern, text.lower())
        if match:
            hour = int(match.group(1))
            minute = int(match.group(2))
            ampm = match.group(3)
            
            if ampm == 'pm' and hour != 12:
                hour += 12
            elif ampm == 'am' and hour == 12:
                hour = 0
            
            return now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        
        return None
    
    def _save_reminder(self, reminder: Reminder):
        """Save reminder to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO reminders 
                (id, title, description, due_date, is_recurring, recurrence_pattern, is_completed, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                reminder.id, reminder.title, reminder.description,
                reminder.due_date.isoformat(), reminder.is_recurring,
                reminder.recurrence_pattern, reminder.is_completed,
                reminder.created_at.isoformat()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to save reminder: {str(e)}")
    
    def _save_appointment(self, appointment: Appointment):
        """Save appointment to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO appointments 
                (id, title, description, start_time, end_time, location, attendees, is_cancelled, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                appointment.id, appointment.title, appointment.description,
                appointment.start_time.isoformat(), appointment.end_time.isoformat(),
                appointment.location, json.dumps(appointment.attendees),
                appointment.is_cancelled, appointment.created_at.isoformat()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to save appointment: {str(e)}")
    
    async def _reminder_checker(self):
        """Background task to check for due reminders"""
        while True:
            try:
                now = datetime.now()
                for reminder in self.reminders:
                    if not reminder.is_completed and reminder.due_date <= now:
                        self.logger.info(f"Reminder due: {reminder.title}")
                        # In a real implementation, you might send notifications here
                        
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error in reminder checker: {str(e)}")
                await asyncio.sleep(60)
    
    def _get_help_message(self) -> str:
        """Get help message for the personal assistant"""
        return """
I'm your Personal Assistant! Here's how I can help you:

📅 **Scheduling & Appointments**
- Create appointments with specific times
- View your daily schedule
- List upcoming appointments

⏰ **Reminders**
- Set reminders for important tasks
- View all active reminders
- Recurring reminders support

❓ **General Assistance**
- Answer questions
- Provide information
- Help with daily tasks

Just ask me naturally, like:
- "Remind me to call John tomorrow at 2pm"
- "What's my schedule for today?"
- "Create an appointment for lunch with Sarah"
- "What time is it?"

I'm here to make your day more organized and productive!
        """.strip()
