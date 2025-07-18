"""
Resource Finder Agent - Finds the best resources, tutorials, and materials for any topic
"""
import json
import re
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3

from .base_agent import BaseAgent, AgentTask
from .utils import sanitize_input, generate_task_id, format_timestamp, extract_urls

class ResourceType(Enum):
    """Types of resources"""
    TUTORIAL = "tutorial"
    COURSE = "course"
    BOOK = "book"
    ARTICLE = "article"
    VIDEO = "video"
    DOCUMENTATION = "documentation"
    TOOL = "tool"
    LIBRARY = "library"
    FRAMEWORK = "framework"
    BLOG_POST = "blog_post"
    PODCAST = "podcast"
    CONFERENCE_TALK = "conference_talk"

class ResourceLevel(Enum):
    """Resource difficulty levels"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"

class ResourceFormat(Enum):
    """Resource formats"""
    TEXT = "text"
    VIDEO = "video"
    AUDIO = "audio"
    INTERACTIVE = "interactive"
    HANDS_ON = "hands_on"

@dataclass
class Resource:
    """Represents a learning resource"""
    id: str
    title: str
    description: str
    url: Optional[str] = None
    resource_type: ResourceType = ResourceType.ARTICLE
    level: ResourceLevel = ResourceLevel.BEGINNER
    format: ResourceFormat = ResourceFormat.TEXT
    author: Optional[str] = None
    rating: float = 0.0  # 0-5 scale
    duration: Optional[str] = None  # e.g., "2 hours", "5 days"
    cost: str = "free"  # "free", "paid", "$X"
    tags: List[str] = None
    prerequisites: List[str] = None
    learning_outcomes: List[str] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.prerequisites is None:
            self.prerequisites = []
        if self.learning_outcomes is None:
            self.learning_outcomes = []
        if self.created_at is None:
            self.created_at = datetime.now()

@dataclass
class LearningPath:
    """Represents a structured learning path"""
    id: str
    title: str
    description: str
    topic: str
    level: ResourceLevel
    resources: List[Resource] = None
    estimated_duration: Optional[str] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.resources is None:
            self.resources = []
        if self.created_at is None:
            self.created_at = datetime.now()

class ResourceFinderAgent(BaseAgent):
    """Resource Finder Agent for finding learning materials"""
    
    def __init__(self, skip_model_init: bool = False):
        super().__init__(
            name="resource_finder",
            description="Finds the best resources, tutorials, and materials for any topic",
            skip_model_init=skip_model_init
        )
        
        self.resources: List[Resource] = []
        self.learning_paths: List[LearningPath] = []
        self.db_path = "resource_finder.db"
        
        # Initialize database
        self._init_database()
        
        # Load existing data
        self._load_data()
        
        # Resource databases and sources
        self.resource_sources = {
            "programming": [
                "MDN Web Docs", "Stack Overflow", "GitHub", "freeCodeCamp",
                "Codecademy", "Coursera", "edX", "Udemy", "Khan Academy"
            ],
            "data_science": [
                "Kaggle", "Towards Data Science", "DataCamp", "Coursera",
                "edX", "Fast.ai", "Papers with Code"
            ],
            "design": [
                "Dribbble", "Behance", "Adobe Creative Suite", "Figma Community",
                "Design+Code", "Interaction Design Foundation"
            ],
            "business": [
                "Harvard Business Review", "McKinsey Insights", "Coursera Business",
                "LinkedIn Learning", "MasterClass"
            ]
        }
    
    def _init_database(self):
        """Initialize SQLite database for resource data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create resources table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS resources (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT,
                    url TEXT,
                    resource_type TEXT NOT NULL,
                    level TEXT NOT NULL,
                    format TEXT NOT NULL,
                    author TEXT,
                    rating REAL DEFAULT 0.0,
                    duration TEXT,
                    cost TEXT DEFAULT 'free',
                    tags TEXT,
                    prerequisites TEXT,
                    learning_outcomes TEXT,
                    created_at TEXT NOT NULL
                )
            ''')
            
            # Create learning paths table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS learning_paths (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT,
                    topic TEXT NOT NULL,
                    level TEXT NOT NULL,
                    estimated_duration TEXT,
                    created_at TEXT NOT NULL
                )
            ''')
            
            # Create learning path resources junction table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS learning_path_resources (
                    path_id TEXT,
                    resource_id TEXT,
                    order_index INTEGER,
                    FOREIGN KEY (path_id) REFERENCES learning_paths (id),
                    FOREIGN KEY (resource_id) REFERENCES resources (id)
                )
            ''')
            
            conn.commit()
            conn.close()
            
            self.logger.info("Resource database initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize resource database: {str(e)}")
    
    def _load_data(self):
        """Load resources and learning paths from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Load resources
            cursor.execute("SELECT * FROM resources")
            for row in cursor.fetchall():
                resource = Resource(
                    id=row[0],
                    title=row[1],
                    description=row[2],
                    url=row[3],
                    resource_type=ResourceType(row[4]),
                    level=ResourceLevel(row[5]),
                    format=ResourceFormat(row[6]),
                    author=row[7],
                    rating=row[8],
                    duration=row[9],
                    cost=row[10],
                    tags=json.loads(row[11]) if row[11] else [],
                    prerequisites=json.loads(row[12]) if row[12] else [],
                    learning_outcomes=json.loads(row[13]) if row[13] else [],
                    created_at=datetime.fromisoformat(row[14])
                )
                self.resources.append(resource)
            
            # Load learning paths
            cursor.execute("SELECT * FROM learning_paths")
            for row in cursor.fetchall():
                path = LearningPath(
                    id=row[0],
                    title=row[1],
                    description=row[2],
                    topic=row[3],
                    level=ResourceLevel(row[4]),
                    estimated_duration=row[5],
                    created_at=datetime.fromisoformat(row[6])
                )
                
                # Load resources for this path
                cursor.execute('''
                    SELECT r.* FROM resources r
                    JOIN learning_path_resources lpr ON r.id = lpr.resource_id
                    WHERE lpr.path_id = ?
                    ORDER BY lpr.order_index
                ''', (path.id,))
                
                for resource_row in cursor.fetchall():
                    resource = Resource(
                        id=resource_row[0],
                        title=resource_row[1],
                        description=resource_row[2],
                        url=resource_row[3],
                        resource_type=ResourceType(resource_row[4]),
                        level=ResourceLevel(resource_row[5]),
                        format=ResourceFormat(resource_row[6]),
                        author=resource_row[7],
                        rating=resource_row[8],
                        duration=resource_row[9],
                        cost=resource_row[10],
                        tags=json.loads(resource_row[11]) if resource_row[11] else [],
                        prerequisites=json.loads(resource_row[12]) if resource_row[12] else [],
                        learning_outcomes=json.loads(resource_row[13]) if resource_row[13] else [],
                        created_at=datetime.fromisoformat(resource_row[14])
                    )
                    path.resources.append(resource)
                
                self.learning_paths.append(path)
            
            conn.close()
            self.logger.info(f"Loaded {len(self.resources)} resources and {len(self.learning_paths)} learning paths")
            
        except Exception as e:
            self.logger.error(f"Failed to load resource data: {str(e)}")
    
    async def _process_task_impl(self, task: AgentTask) -> str:
        """Process resource finder tasks"""
        task_type = task.task_type.lower()
        content = sanitize_input(task.content)
        
        if task_type == "find_resources":
            return await self._find_resources(content, task.metadata)
        elif task_type == "create_learning_path":
            return await self._create_learning_path(content, task.metadata)
        elif task_type == "recommend_resources":
            return await self._recommend_resources(content, task.metadata)
        elif task_type == "compare_resources":
            return await self._compare_resources(content, task.metadata)
        elif task_type == "filter_resources":
            return await self._filter_resources(content, task.metadata)
        elif task_type == "get_learning_path":
            return await self._get_learning_path(content, task.metadata)
        elif task_type == "list_resources":
            return await self._list_resources(task.metadata)
        elif task_type == "add_resource":
            return await self._add_resource(content, task.metadata)
        elif task_type == "rate_resource":
            return await self._rate_resource(content, task.metadata)
        elif task_type == "health_check":
            return "Resource Finder Agent is ready to help you find the best learning materials!"
        else:
            return await self._find_resources(content, task.metadata)
    
    async def _find_resources(self, topic: str, metadata: Dict[str, Any]) -> str:
        """Find resources for a specific topic"""
        try:
            level = metadata.get("level", "all")
            resource_type = metadata.get("resource_type", "all")
            format_type = metadata.get("format", "all")
            cost_filter = metadata.get("cost", "all")  # free, paid, all
            limit = metadata.get("limit", 10)
            
            # Generate resource recommendations using LLM
            resources_content = await self._generate_resource_recommendations(
                topic, level, resource_type, format_type, cost_filter
            )
            
            # Filter existing resources if any match
            matching_resources = self._search_existing_resources(topic, level, resource_type)
            
            result = f"Resources for: {topic}\n\n"
            
            if metadata.get("level") != "all":
                result += f"**Level:** {level.title()}\n"
            if metadata.get("resource_type") != "all":
                result += f"**Type:** {resource_type.title()}\n"
            if metadata.get("format") != "all":
                result += f"**Format:** {format_type.title()}\n"
            if metadata.get("cost") != "all":
                result += f"**Cost:** {cost_filter.title()}\n"
            
            result += "\n" + resources_content
            
            if matching_resources:
                result += f"\n\n**From Our Database ({len(matching_resources)} matches):**\n"
                for resource in matching_resources[:5]:  # Show top 5
                    result += self._format_resource_display(resource)
            
            # Add learning path suggestion
            if metadata.get("suggest_path", True):
                result += f"\n\n💡 **Tip:** Would you like me to create a structured learning path for {topic}? Just ask!"
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to find resources: {str(e)}")
            return f"Sorry, I couldn't find resources for {topic}. Error: {str(e)}"
    
    async def _create_learning_path(self, topic: str, metadata: Dict[str, Any]) -> str:
        """Create a structured learning path for a topic"""
        try:
            level = metadata.get("level", "beginner")
            duration = metadata.get("duration", "flexible")
            focus_areas = metadata.get("focus_areas", [])
            
            try:
                target_level = ResourceLevel(level)
            except ValueError:
                target_level = ResourceLevel.BEGINNER
            
            # Generate learning path structure
            path_content = await self._generate_learning_path_structure(
                topic, target_level, duration, focus_areas
            )
            
            # Create learning path object
            learning_path = LearningPath(
                id=generate_task_id(),
                title=f"Learning Path: {topic}",
                description=f"Comprehensive learning path for {topic} at {level} level",
                topic=topic,
                level=target_level,
                estimated_duration=duration
            )
            
            # Generate and add resources to the path
            path_resources = await self._generate_path_resources(topic, target_level, focus_areas)
            learning_path.resources.extend(path_resources)
            
            # Save to database
            self._save_learning_path(learning_path)
            self.learning_paths.append(learning_path)
            
            result = f"Learning Path Created: {topic}\n\n"
            result += f"**Path ID:** {learning_path.id}\n"
            result += f"**Level:** {level.title()}\n"
            result += f"**Estimated Duration:** {duration}\n"
            if focus_areas:
                result += f"**Focus Areas:** {', '.join(focus_areas)}\n"
            
            result += f"\n{path_content}\n"
            
            if learning_path.resources:
                result += f"\n**Resources ({len(learning_path.resources)}):**\n"
                for i, resource in enumerate(learning_path.resources, 1):
                    result += f"{i}. **{resource.title}** ({resource.resource_type.value})\n"
                    result += f"   {resource.description[:100]}...\n"
                    if resource.url:
                        result += f"   🔗 {resource.url}\n"
                    result += "\n"
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to create learning path: {str(e)}")
            return f"Sorry, I couldn't create a learning path for {topic}. Error: {str(e)}"
    
    async def _recommend_resources(self, description: str, metadata: Dict[str, Any]) -> str:
        """Recommend resources based on user's specific needs"""
        try:
            current_level = metadata.get("current_level", "beginner")
            goals = metadata.get("goals", [])
            time_available = metadata.get("time_available", "flexible")
            preferred_format = metadata.get("preferred_format", "any")
            
            # Generate personalized recommendations
            recommendation_prompt = f"""
            Based on the following user profile, recommend the best learning resources:
            
            User Description: {description}
            Current Level: {current_level}
            Goals: {', '.join(goals) if goals else 'General learning'}
            Time Available: {time_available}
            Preferred Format: {preferred_format}
            
            Provide personalized recommendations with explanations for why each resource is suitable.
            """
            
            recommendations = self.generate_response(recommendation_prompt, max_length=1024)
            
            result = f"Personalized Resource Recommendations\n\n"
            result += f"**Your Profile:**\n"
            result += f"• Current Level: {current_level.title()}\n"
            if goals:
                result += f"• Goals: {', '.join(goals)}\n"
            result += f"• Time Available: {time_available}\n"
            result += f"• Preferred Format: {preferred_format}\n\n"
            
            result += recommendations
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to recommend resources: {str(e)}")
            return f"Sorry, I couldn't generate recommendations. Error: {str(e)}"
    
    async def _compare_resources(self, resources_description: str, metadata: Dict[str, Any]) -> str:
        """Compare different resources or learning options"""
        try:
            comparison_criteria = metadata.get("criteria", ["quality", "cost", "time", "difficulty"])
            
            # Extract resource names/topics from description
            resources_to_compare = self._extract_resources_from_text(resources_description)
            
            if len(resources_to_compare) < 2:
                return "Please provide at least 2 resources or topics to compare."
            
            # Generate comparison
            comparison_prompt = f"""
            Compare the following learning resources/options: {', '.join(resources_to_compare)}
            
            Comparison Criteria: {', '.join(comparison_criteria)}
            
            Provide a detailed comparison table or analysis covering:
            1. Strengths and weaknesses of each
            2. Best use cases for each
            3. Recommendations based on different learner profiles
            """
            
            comparison_result = self.generate_response(comparison_prompt, max_length=1024)
            
            result = f"Resource Comparison\n\n"
            result += f"**Comparing:** {', '.join(resources_to_compare)}\n"
            result += f"**Criteria:** {', '.join(comparison_criteria)}\n\n"
            result += comparison_result
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to compare resources: {str(e)}")
            return f"Sorry, I couldn't compare the resources. Error: {str(e)}"
    
    async def _filter_resources(self, query: str, metadata: Dict[str, Any]) -> str:
        """Filter resources based on specific criteria"""
        try:
            filters = {
                "level": metadata.get("level"),
                "type": metadata.get("resource_type"),
                "format": metadata.get("format"),
                "cost": metadata.get("cost"),
                "rating_min": metadata.get("min_rating", 0),
                "duration_max": metadata.get("max_duration")
            }
            
            # Apply filters to existing resources
            filtered_resources = self._apply_filters(self.resources, filters)
            
            result = f"Filtered Resources\n\n"
            result += f"**Query:** {query}\n"
            
            active_filters = [f"{k}: {v}" for k, v in filters.items() if v is not None]
            if active_filters:
                result += f"**Filters:** {', '.join(active_filters)}\n"
            
            result += f"**Found:** {len(filtered_resources)} resources\n\n"
            
            if filtered_resources:
                for resource in filtered_resources[:10]:  # Show top 10
                    result += self._format_resource_display(resource)
            else:
                result += "No resources match your criteria. Try adjusting your filters."
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to filter resources: {str(e)}")
            return f"Sorry, I couldn't filter the resources. Error: {str(e)}"
    
    async def _get_learning_path(self, path_identifier: str, metadata: Dict[str, Any]) -> str:
        """Get details of a specific learning path"""
        try:
            # Find learning path by ID or title
            learning_path = None
            for path in self.learning_paths:
                if path.id == path_identifier or path_identifier.lower() in path.title.lower():
                    learning_path = path
                    break
            
            if not learning_path:
                return f"Learning path '{path_identifier}' not found."
            
            result = f"Learning Path: {learning_path.title}\n\n"
            result += f"**ID:** {learning_path.id}\n"
            result += f"**Topic:** {learning_path.topic}\n"
            result += f"**Level:** {learning_path.level.value.title()}\n"
            if learning_path.estimated_duration:
                result += f"**Duration:** {learning_path.estimated_duration}\n"
            result += f"**Created:** {format_timestamp(learning_path.created_at)}\n\n"
            
            result += f"**Description:**\n{learning_path.description}\n\n"
            
            if learning_path.resources:
                result += f"**Learning Path ({len(learning_path.resources)} steps):**\n\n"
                for i, resource in enumerate(learning_path.resources, 1):
                    result += f"**Step {i}: {resource.title}**\n"
                    result += f"Type: {resource.resource_type.value.title()}\n"
                    result += f"Level: {resource.level.value.title()}\n"
                    if resource.duration:
                        result += f"Duration: {resource.duration}\n"
                    result += f"Description: {resource.description}\n"
                    if resource.url:
                        result += f"Link: {resource.url}\n"
                    result += "\n"
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to get learning path: {str(e)}")
            return f"Sorry, I couldn't retrieve the learning path. Error: {str(e)}"
    
    async def _list_resources(self, metadata: Dict[str, Any]) -> str:
        """List available resources"""
        try:
            category = metadata.get("category", "all")
            limit = metadata.get("limit", 20)
            sort_by = metadata.get("sort_by", "created_at")  # created_at, rating, title
            
            resources_to_show = self.resources
            
            if category != "all":
                resources_to_show = [r for r in resources_to_show if category.lower() in [tag.lower() for tag in r.tags]]
            
            # Sort resources
            if sort_by == "rating":
                resources_to_show = sorted(resources_to_show, key=lambda r: r.rating, reverse=True)
            elif sort_by == "title":
                resources_to_show = sorted(resources_to_show, key=lambda r: r.title)
            else:  # created_at
                resources_to_show = sorted(resources_to_show, key=lambda r: r.created_at, reverse=True)
            
            result = f"Resource Library\n\n"
            result += f"**Total Resources:** {len(self.resources)}\n"
            if category != "all":
                result += f"**Category:** {category.title()}\n"
            result += f"**Showing:** {min(limit, len(resources_to_show))} resources\n"
            result += f"**Sorted by:** {sort_by.replace('_', ' ').title()}\n\n"
            
            for resource in resources_to_show[:limit]:
                result += self._format_resource_display(resource)
            
            if len(resources_to_show) > limit:
                result += f"\n... and {len(resources_to_show) - limit} more resources"
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to list resources: {str(e)}")
            return f"Sorry, I couldn't list the resources. Error: {str(e)}"
    
    # Helper methods
    def _format_resource_display(self, resource: Resource) -> str:
        """Format resource for display"""
        display = f"📚 **{resource.title}**\n"
        display += f"   Type: {resource.resource_type.value.title()} | Level: {resource.level.value.title()}\n"
        if resource.author:
            display += f"   Author: {resource.author}\n"
        if resource.rating > 0:
            display += f"   Rating: {'⭐' * int(resource.rating)} ({resource.rating}/5)\n"
        if resource.duration:
            display += f"   Duration: {resource.duration}\n"
        display += f"   Cost: {resource.cost.title()}\n"
        if resource.url:
            display += f"   🔗 {resource.url}\n"
        display += f"   {resource.description[:100]}...\n\n"
        return display

    async def _generate_resource_recommendations(self, topic: str, level: str, resource_type: str,
                                               format_type: str, cost_filter: str) -> str:
        """Generate resource recommendations using LLM"""

        # Check for common topics and provide curated responses
        curated_response = self._get_curated_resources(topic.lower(), level, cost_filter)
        if curated_response:
            return curated_response

        prompt = f"""
        Find and recommend the best learning resources for: {topic}

        Criteria:
        - Level: {level}
        - Type: {resource_type}
        - Format: {format_type}
        - Cost: {cost_filter}

        Provide a curated list of high-quality resources including:
        1. Title and brief description
        2. Why it's recommended
        3. Estimated time commitment
        4. Prerequisites (if any)
        5. Link or where to find it

        Focus on practical, actionable resources that provide real value.
        """

        try:
            llm_response = self.generate_response(prompt, max_length=1024)
            # If LLM response is too short or empty, use fallback
            if not llm_response or len(llm_response.strip()) < 50:
                return self._get_fallback_resources(topic, level, cost_filter)
            return llm_response
        except Exception as e:
            self.logger.error(f"LLM generation failed: {e}")
            return self._get_fallback_resources(topic, level, cost_filter)

    def _search_existing_resources(self, topic: str, level: str, resource_type: str) -> List[Resource]:
        """Search existing resources in database"""
        matching_resources = []

        for resource in self.resources:
            # Check if topic matches tags or title
            topic_match = (
                topic.lower() in resource.title.lower() or
                any(topic.lower() in tag.lower() for tag in resource.tags)
            )

            # Check level filter
            level_match = level == "all" or resource.level.value == level

            # Check type filter
            type_match = resource_type == "all" or resource.resource_type.value == resource_type

            if topic_match and level_match and type_match:
                matching_resources.append(resource)

        # Sort by rating
        return sorted(matching_resources, key=lambda r: r.rating, reverse=True)

    def _get_curated_resources(self, topic: str, level: str, cost_filter: str) -> str:
        """Get curated resources for common topics"""

        if "machine learning" in topic or "ml" in topic:
            return self._get_machine_learning_resources(level, cost_filter)
        elif "python" in topic:
            return self._get_python_resources(level, cost_filter)
        elif "javascript" in topic or "js" in topic:
            return self._get_javascript_resources(level, cost_filter)
        elif "data science" in topic:
            return self._get_data_science_resources(level, cost_filter)
        elif "web development" in topic:
            return self._get_web_development_resources(level, cost_filter)

        return None

    def _get_machine_learning_resources(self, level: str, cost_filter: str) -> str:
        """Curated machine learning resources"""

        resources = """## 🤖 Machine Learning Resources

### 📚 **Comprehensive Courses**

**1. Andrew Ng's Machine Learning Course (Coursera)**
   - 📖 The gold standard for ML education
   - ⏱️ 11 weeks, 5-7 hours/week
   - 💰 Free audit, $49/month for certificate
   - 🎯 Perfect for beginners with some math background
   - 🔗 coursera.org/learn/machine-learning

**2. Fast.ai Practical Deep Learning**
   - 📖 Top-down approach, code-first learning
   - ⏱️ 7 lessons, self-paced
   - 💰 Completely free
   - 🎯 Great for programmers who want to build quickly
   - 🔗 course.fast.ai

**3. CS229 Stanford Machine Learning**
   - 📖 Rigorous mathematical foundation
   - ⏱️ Full semester course materials
   - 💰 Free (lecture videos and notes)
   - 🎯 Advanced, requires strong math background
   - 🔗 cs229.stanford.edu

### 📖 **Essential Books**

**4. "Hands-On Machine Learning" by Aurélien Géron**
   - 📖 Practical approach with Python code
   - ⏱️ 2-3 months to complete
   - 💰 ~$45 (book)
   - 🎯 Perfect balance of theory and practice

**5. "Pattern Recognition and Machine Learning" by Bishop**
   - 📖 Mathematical foundations and theory
   - ⏱️ Graduate-level textbook
   - 💰 ~$80 (book)
   - 🎯 For deep theoretical understanding

### 💻 **Interactive Platforms**

**6. Kaggle Learn**
   - 📖 Micro-courses with hands-on exercises
   - ⏱️ 1-4 hours per course
   - 💰 Completely free
   - 🎯 Great for practical skills and competitions
   - 🔗 kaggle.com/learn

**7. Google's Machine Learning Crash Course**
   - 📖 TensorFlow-focused practical course
   - ⏱️ 15 hours total
   - 💰 Free
   - 🎯 Good introduction with Google's tools
   - 🔗 developers.google.com/machine-learning/crash-course

### 🛠️ **Practical Projects**

**8. Machine Learning Mastery**
   - 📖 Step-by-step tutorials and projects
   - ⏱️ Various project lengths
   - 💰 Free tutorials, paid books
   - 🎯 Learn by building real projects
   - 🔗 machinelearningmastery.com

### 📺 **YouTube Channels**

**9. 3Blue1Brown - Neural Networks Series**
   - 📖 Visual explanations of ML concepts
   - ⏱️ 1 hour total for neural network series
   - 💰 Free
   - 🎯 Excellent for understanding intuition

**10. Two Minute Papers**
   - 📖 Latest ML research explained simply
   - ⏱️ 2-5 minutes per video
   - 💰 Free
   - 🎯 Stay updated with cutting-edge research"""

        if level == "beginner":
            resources += "\n\n### 🎯 **Recommended Path for Beginners:**\n1. Start with Andrew Ng's course\n2. Read first half of Hands-On ML book\n3. Complete Kaggle Learn micro-courses\n4. Work on 2-3 Kaggle competitions"

        elif level == "intermediate":
            resources += "\n\n### 🎯 **Recommended Path for Intermediate:**\n1. Fast.ai course for practical skills\n2. CS229 for mathematical depth\n3. Advanced Kaggle competitions\n4. Implement papers from scratch"

        elif level == "advanced":
            resources += "\n\n### 🎯 **Recommended Path for Advanced:**\n1. Latest research papers on arXiv\n2. Contribute to open-source ML libraries\n3. Attend conferences (NeurIPS, ICML, ICLR)\n4. Specialize in specific domains (NLP, CV, RL)"

        if cost_filter == "free":
            resources += "\n\n💡 **Free Resources Highlighted:** Fast.ai, Kaggle Learn, Google ML Crash Course, YouTube channels, and audit versions of Coursera courses."

        return resources

    def _get_fallback_resources(self, topic: str, level: str, cost_filter: str) -> str:
        """Fallback resources when LLM generation fails"""

        return f"""## 📚 Learning Resources for {topic.title()}

### 🎯 **Getting Started**

**1. Online Courses**
   - 📖 Search for "{topic}" on Coursera, edX, or Udemy
   - ⏱️ Typically 4-12 weeks
   - 💰 Free audit options available
   - 🎯 Structured learning with certificates

**2. Documentation & Tutorials**
   - 📖 Official documentation and getting started guides
   - ⏱️ Self-paced
   - 💰 Free
   - 🎯 Authoritative and up-to-date information

**3. YouTube Learning**
   - 📖 Search for "{topic} tutorial" or "{topic} course"
   - ⏱️ Varies (1 hour to 20+ hours)
   - 💰 Free
   - 🎯 Visual learning with practical examples

**4. Books & eBooks**
   - 📖 Search for highly-rated books on Amazon or O'Reilly
   - ⏱️ Several weeks to months
   - 💰 $20-60 typically
   - 🎯 Comprehensive and detailed coverage

**5. Practice Platforms**
   - 📖 Hands-on coding and projects
   - ⏱️ Ongoing practice
   - 💰 Many free options available
   - 🎯 Learn by doing

### 💡 **Next Steps**
1. Start with free resources to gauge interest
2. Choose one primary course or book to follow
3. Supplement with practical projects
4. Join communities (Reddit, Discord, Stack Overflow)
5. Build a portfolio of projects

### 🔍 **Search Tips**
- Look for resources updated within the last 2 years
- Check reviews and ratings before committing
- Start with beginner-friendly content even if you have experience
- Focus on practical, hands-on learning"""

    def _get_python_resources(self, level: str, cost_filter: str) -> str:
        """Curated Python learning resources"""
        return """## 🐍 Python Learning Resources

### 📚 **Beginner Courses**
1. **Python.org Official Tutorial** - Free, comprehensive
2. **Automate the Boring Stuff** - Practical Python for beginners
3. **Python Crash Course** - Book with hands-on projects
4. **Codecademy Python Course** - Interactive learning

### 🚀 **Intermediate/Advanced**
1. **Real Python** - In-depth tutorials and articles
2. **Effective Python** - Best practices and advanced techniques
3. **Python Tricks** - Clean code and Pythonic patterns
4. **Flask/Django tutorials** - Web development frameworks"""

    def _get_javascript_resources(self, level: str, cost_filter: str) -> str:
        """Curated JavaScript learning resources"""
        return """## 🟨 JavaScript Learning Resources

### 📚 **Fundamentals**
1. **MDN Web Docs** - Comprehensive JavaScript reference
2. **JavaScript.info** - Modern JavaScript tutorial
3. **Eloquent JavaScript** - Free online book
4. **freeCodeCamp** - Interactive coding challenges

### 🚀 **Advanced Topics**
1. **You Don't Know JS** - Deep dive into JavaScript
2. **JavaScript: The Good Parts** - Best practices
3. **Modern JavaScript frameworks** - React, Vue, Angular
4. **Node.js tutorials** - Server-side JavaScript"""

    def _get_data_science_resources(self, level: str, cost_filter: str) -> str:
        """Curated Data Science learning resources"""
        return """## 📊 Data Science Learning Resources

### 📚 **Complete Programs**
1. **Kaggle Learn** - Free micro-courses
2. **DataCamp** - Interactive data science courses
3. **Coursera Data Science Specialization** - Johns Hopkins
4. **edX MicroMasters** - MIT and other universities

### 🛠️ **Tools & Libraries**
1. **Pandas documentation** - Data manipulation
2. **Matplotlib/Seaborn tutorials** - Data visualization
3. **Scikit-learn user guide** - Machine learning
4. **Jupyter Notebook tutorials** - Interactive computing"""

    def _get_web_development_resources(self, level: str, cost_filter: str) -> str:
        """Curated Web Development learning resources"""
        return """## 🌐 Web Development Learning Resources

### 📚 **Frontend Development**
1. **freeCodeCamp** - Complete web development curriculum
2. **The Odin Project** - Full-stack web development
3. **MDN Web Docs** - HTML, CSS, JavaScript reference
4. **CSS-Tricks** - CSS techniques and best practices

### 🚀 **Backend Development**
1. **Node.js tutorials** - JavaScript backend
2. **Django/Flask documentation** - Python web frameworks
3. **Express.js guides** - Node.js web framework
4. **Database tutorials** - SQL and NoSQL databases"""

    async def _generate_learning_path_structure(self, topic: str, level: ResourceLevel,
                                              duration: str, focus_areas: List[str]) -> str:
        """Generate learning path structure"""
        prompt = f"""
        Create a structured learning path for: {topic}

        Parameters:
        - Target Level: {level.value}
        - Duration: {duration}
        - Focus Areas: {', '.join(focus_areas) if focus_areas else 'General coverage'}

        Provide:
        1. Learning objectives
        2. Step-by-step progression
        3. Key milestones
        4. Recommended sequence
        5. Assessment points

        Make it practical and achievable.
        """

        return self.generate_response(prompt, max_length=1024)

    async def _generate_path_resources(self, topic: str, level: ResourceLevel,
                                     focus_areas: List[str]) -> List[Resource]:
        """Generate resources for a learning path"""
        # This would typically integrate with external APIs
        # For now, create some sample resources

        resources = []

        # Basic structure for any topic
        basic_resources = [
            {
                "title": f"Introduction to {topic}",
                "description": f"Comprehensive introduction covering the fundamentals of {topic}",
                "resource_type": ResourceType.TUTORIAL,
                "duration": "2-3 hours"
            },
            {
                "title": f"Hands-on {topic} Practice",
                "description": f"Practical exercises and projects to apply {topic} concepts",
                "resource_type": ResourceType.TUTORIAL,
                "format": ResourceFormat.HANDS_ON,
                "duration": "5-10 hours"
            },
            {
                "title": f"Advanced {topic} Techniques",
                "description": f"Advanced concepts and best practices in {topic}",
                "resource_type": ResourceType.COURSE,
                "level": ResourceLevel.ADVANCED,
                "duration": "10-15 hours"
            }
        ]

        for i, resource_data in enumerate(basic_resources):
            resource = Resource(
                id=generate_task_id(),
                title=resource_data["title"],
                description=resource_data["description"],
                resource_type=resource_data.get("resource_type", ResourceType.TUTORIAL),
                level=resource_data.get("level", level),
                format=resource_data.get("format", ResourceFormat.TEXT),
                duration=resource_data.get("duration", "1-2 hours"),
                tags=[topic.lower()] + focus_areas
            )
            resources.append(resource)

        return resources

    def _extract_resources_from_text(self, text: str) -> List[str]:
        """Extract resource names from text"""
        # Simple extraction - in production, use more sophisticated NLP
        resources = []

        # Look for common patterns
        patterns = [
            r'(?:compare|between)\s+([A-Za-z\s]+?)\s+(?:and|vs|versus)\s+([A-Za-z\s]+)',
            r'([A-Za-z\s]+?),\s*([A-Za-z\s]+?)(?:,|\s+and\s+)([A-Za-z\s]+)',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    resources.extend([m.strip() for m in match if m.strip()])
                else:
                    resources.append(match.strip())

        # Fallback: split by common separators
        if not resources:
            separators = [' vs ', ' and ', ', ', ' or ']
            for sep in separators:
                if sep in text.lower():
                    resources = [r.strip() for r in text.split(sep)]
                    break

        return resources[:5]  # Limit to 5 resources

    def _apply_filters(self, resources: List[Resource], filters: Dict[str, Any]) -> List[Resource]:
        """Apply filters to resource list"""
        filtered = resources

        if filters.get("level"):
            filtered = [r for r in filtered if r.level.value == filters["level"]]

        if filters.get("type"):
            filtered = [r for r in filtered if r.resource_type.value == filters["type"]]

        if filters.get("format"):
            filtered = [r for r in filtered if r.format.value == filters["format"]]

        if filters.get("cost"):
            if filters["cost"] == "free":
                filtered = [r for r in filtered if r.cost.lower() == "free"]
            elif filters["cost"] == "paid":
                filtered = [r for r in filtered if r.cost.lower() != "free"]

        if filters.get("rating_min"):
            filtered = [r for r in filtered if r.rating >= filters["rating_min"]]

        return filtered

    def _save_learning_path(self, learning_path: LearningPath):
        """Save learning path to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Save learning path
            cursor.execute('''
                INSERT OR REPLACE INTO learning_paths
                (id, title, description, topic, level, estimated_duration, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                learning_path.id, learning_path.title, learning_path.description,
                learning_path.topic, learning_path.level.value,
                learning_path.estimated_duration, learning_path.created_at.isoformat()
            ))

            # Save resources and link them to the path
            for i, resource in enumerate(learning_path.resources):
                # Save resource if not exists
                cursor.execute('''
                    INSERT OR IGNORE INTO resources
                    (id, title, description, url, resource_type, level, format, author, rating, duration, cost, tags, prerequisites, learning_outcomes, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    resource.id, resource.title, resource.description, resource.url,
                    resource.resource_type.value, resource.level.value, resource.format.value,
                    resource.author, resource.rating, resource.duration, resource.cost,
                    json.dumps(resource.tags), json.dumps(resource.prerequisites),
                    json.dumps(resource.learning_outcomes), resource.created_at.isoformat()
                ))

                # Link resource to learning path
                cursor.execute('''
                    INSERT OR REPLACE INTO learning_path_resources
                    (path_id, resource_id, order_index)
                    VALUES (?, ?, ?)
                ''', (learning_path.id, resource.id, i))

            conn.commit()
            conn.close()

        except Exception as e:
            self.logger.error(f"Failed to save learning path: {str(e)}")

    async def _add_resource(self, resource_description: str, metadata: Dict[str, Any]) -> str:
        """Add a new resource to the database"""
        try:
            title = metadata.get("title", resource_description[:50])
            url = metadata.get("url")
            resource_type = metadata.get("resource_type", "article")
            level = metadata.get("level", "beginner")
            author = metadata.get("author")
            rating = metadata.get("rating", 0.0)
            tags = metadata.get("tags", [])

            resource = Resource(
                id=generate_task_id(),
                title=title,
                description=resource_description,
                url=url,
                resource_type=ResourceType(resource_type),
                level=ResourceLevel(level),
                author=author,
                rating=float(rating),
                tags=tags
            )

            # Save to database
            self._save_resource(resource)
            self.resources.append(resource)

            return f"Resource added successfully: '{title}'"

        except Exception as e:
            self.logger.error(f"Failed to add resource: {str(e)}")
            return f"Sorry, I couldn't add the resource. Error: {str(e)}"

    async def _rate_resource(self, resource_identifier: str, metadata: Dict[str, Any]) -> str:
        """Rate a resource"""
        try:
            rating = metadata.get("rating")
            if not rating:
                return "Please provide a rating (1-5 scale)."

            rating = float(rating)
            if rating < 1 or rating > 5:
                return "Rating must be between 1 and 5."

            # Find resource
            resource = None
            for r in self.resources:
                if r.id == resource_identifier or resource_identifier.lower() in r.title.lower():
                    resource = r
                    break

            if not resource:
                return f"Resource '{resource_identifier}' not found."

            # Update rating (simple average for now)
            resource.rating = rating
            self._save_resource(resource)

            return f"Rating updated for '{resource.title}': {rating}/5 stars"

        except Exception as e:
            self.logger.error(f"Failed to rate resource: {str(e)}")
            return f"Sorry, I couldn't rate the resource. Error: {str(e)}"

    def _save_resource(self, resource: Resource):
        """Save resource to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT OR REPLACE INTO resources
                (id, title, description, url, resource_type, level, format, author, rating, duration, cost, tags, prerequisites, learning_outcomes, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                resource.id, resource.title, resource.description, resource.url,
                resource.resource_type.value, resource.level.value, resource.format.value,
                resource.author, resource.rating, resource.duration, resource.cost,
                json.dumps(resource.tags), json.dumps(resource.prerequisites),
                json.dumps(resource.learning_outcomes), resource.created_at.isoformat()
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            self.logger.error(f"Failed to save resource: {str(e)}")
