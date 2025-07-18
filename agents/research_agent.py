import json
import re
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3

from .base_agent import BaseAgent, AgentTask
from .utils import sanitize_input, generate_task_id, format_timestamp

class ResearchType(Enum):
    """Types of research tasks"""
    LITERATURE_REVIEW = "literature_review"
    MARKET_RESEARCH = "market_research"
    TECHNICAL_ANALYSIS = "technical_analysis"
    COMPETITIVE_ANALYSIS = "competitive_analysis"
    TREND_ANALYSIS = "trend_analysis"
    FEASIBILITY_STUDY = "feasibility_study"
    DATA_ANALYSIS = "data_analysis"

class ResearchStatus(Enum):
    """Research task status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    REQUIRES_REVIEW = "requires_review"

@dataclass
class ResearchSource:
    """Represents a research source"""
    id: str
    title: str
    url: Optional[str] = None
    authors: List[str] = None
    publication_date: Optional[datetime] = None
    source_type: str = "article"  # article, paper, website, book, etc.
    credibility_score: int = 5  # 1-10 scale
    summary: Optional[str] = None
    key_findings: List[str] = None
    
    def __post_init__(self):
        if self.authors is None:
            self.authors = []
        if self.key_findings is None:
            self.key_findings = []

@dataclass
class ResearchProject:
    """Represents a research project"""
    id: str
    title: str
    description: str
    research_type: ResearchType
    status: ResearchStatus
    created_at: datetime
    deadline: Optional[datetime] = None
    sources: List[ResearchSource] = None
    findings: List[str] = None
    conclusions: List[str] = None
    recommendations: List[str] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.sources is None:
            self.sources = []
        if self.findings is None:
            self.findings = []
        if self.conclusions is None:
            self.conclusions = []
        if self.recommendations is None:
            self.recommendations = []
        if self.tags is None:
            self.tags = []

class ResearchAgent(BaseAgent):
    """Research & Development Agent for research tasks"""
    
    def __init__(self):
        super().__init__(
            name="research_agent",
            description="Conducts research and development tasks, literature review, and analysis"
        )
        
        self.research_projects: List[ResearchProject] = []
        self.research_sources: List[ResearchSource] = []
        self.db_path = "research_agent.db"
        
        # Initialize database
        self._init_database()
        
        # Load existing data
        self._load_data()
        
        # Research templates and methodologies
        self.research_methodologies = {
            ResearchType.LITERATURE_REVIEW: {
                "steps": [
                    "Define research question",
                    "Identify key search terms",
                    "Search academic databases",
                    "Screen and select relevant sources",
                    "Extract and synthesize information",
                    "Identify gaps and future directions"
                ],
                "deliverables": ["Summary report", "Source bibliography", "Key findings", "Research gaps"]
            },
            ResearchType.MARKET_RESEARCH: {
                "steps": [
                    "Define market scope",
                    "Identify target segments",
                    "Analyze market size and trends",
                    "Study competitors",
                    "Assess opportunities and threats",
                    "Provide recommendations"
                ],
                "deliverables": ["Market analysis", "Competitive landscape", "SWOT analysis", "Recommendations"]
            }
        }
    
    def _init_database(self):
        """Initialize SQLite database for research data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create research projects table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS research_projects (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT,
                    research_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    deadline TEXT,
                    findings TEXT,
                    conclusions TEXT,
                    recommendations TEXT,
                    tags TEXT
                )
            ''')
            
            # Create research sources table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS research_sources (
                    id TEXT PRIMARY KEY,
                    project_id TEXT,
                    title TEXT NOT NULL,
                    url TEXT,
                    authors TEXT,
                    publication_date TEXT,
                    source_type TEXT DEFAULT 'article',
                    credibility_score INTEGER DEFAULT 5,
                    summary TEXT,
                    key_findings TEXT,
                    FOREIGN KEY (project_id) REFERENCES research_projects (id)
                )
            ''')
            
            conn.commit()
            conn.close()
            
            self.logger.info("Research database initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize research database: {str(e)}")
    
    def _load_data(self):
        """Load research projects and sources from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Load research projects
            cursor.execute("SELECT * FROM research_projects")
            for row in cursor.fetchall():
                project = ResearchProject(
                    id=row[0],
                    title=row[1],
                    description=row[2],
                    research_type=ResearchType(row[3]),
                    status=ResearchStatus(row[4]),
                    created_at=datetime.fromisoformat(row[5]),
                    deadline=datetime.fromisoformat(row[6]) if row[6] else None,
                    findings=json.loads(row[7]) if row[7] else [],
                    conclusions=json.loads(row[8]) if row[8] else [],
                    recommendations=json.loads(row[9]) if row[9] else [],
                    tags=json.loads(row[10]) if row[10] else []
                )
                self.research_projects.append(project)
            
            # Load research sources
            cursor.execute("SELECT * FROM research_sources")
            for row in cursor.fetchall():
                source = ResearchSource(
                    id=row[0],
                    title=row[2],
                    url=row[3],
                    authors=json.loads(row[4]) if row[4] else [],
                    publication_date=datetime.fromisoformat(row[5]) if row[5] else None,
                    source_type=row[6],
                    credibility_score=row[7],
                    summary=row[8],
                    key_findings=json.loads(row[9]) if row[9] else []
                )
                self.research_sources.append(source)
                
                # Link source to project
                project_id = row[1]
                for project in self.research_projects:
                    if project.id == project_id:
                        project.sources.append(source)
                        break
            
            conn.close()
            self.logger.info(f"Loaded {len(self.research_projects)} research projects and {len(self.research_sources)} sources")
            
        except Exception as e:
            self.logger.error(f"Failed to load research data: {str(e)}")
    
    async def _process_task_impl(self, task: AgentTask) -> str:
        """Process research tasks"""
        task_type = task.task_type.lower()
        content = sanitize_input(task.content)
        
        if task_type == "start_research":
            return await self._start_research_project(content, task.metadata)
        elif task_type == "literature_review":
            return await self._conduct_literature_review(content, task.metadata)
        elif task_type == "market_research":
            return await self._conduct_market_research(content, task.metadata)
        elif task_type == "technical_analysis":
            return await self._conduct_technical_analysis(content, task.metadata)
        elif task_type == "analyze_data":
            return await self._analyze_research_data(content, task.metadata)
        elif task_type == "summarize_findings":
            return await self._summarize_research_findings(content, task.metadata)
        elif task_type == "generate_report":
            return await self._generate_research_report(content, task.metadata)
        elif task_type == "list_projects":
            return await self._list_research_projects()
        elif task_type == "project_status":
            return await self._get_project_status(content, task.metadata)
        elif task_type == "health_check":
            return "Research Agent is ready to conduct comprehensive research and analysis!"
        else:
            return await self._conduct_general_research(content, task.metadata)
    
    async def _start_research_project(self, description: str, metadata: Dict[str, Any]) -> str:
        """Start a new research project"""
        try:
            title = metadata.get("title", description[:100])
            research_type_str = metadata.get("research_type", "literature_review")
            deadline_str = metadata.get("deadline")
            tags = metadata.get("tags", [])
            
            try:
                research_type = ResearchType(research_type_str)
            except ValueError:
                research_type = ResearchType.LITERATURE_REVIEW
            
            deadline = None
            if deadline_str:
                try:
                    deadline = datetime.fromisoformat(deadline_str)
                except:
                    deadline = datetime.now() + timedelta(days=30)  # Default 30 days
            
            project = ResearchProject(
                id=generate_task_id(),
                title=title,
                description=description,
                research_type=research_type,
                status=ResearchStatus.PENDING,
                created_at=datetime.now(),
                deadline=deadline,
                tags=tags
            )
            
            # Save to database
            self._save_research_project(project)
            self.research_projects.append(project)
            
            # Generate research plan
            research_plan = self._generate_research_plan(project)
            
            result = f"Research Project Created: '{title}'\n\n"
            result += f"**Project ID:** {project.id}\n"
            result += f"**Type:** {research_type.value.replace('_', ' ').title()}\n"
            result += f"**Status:** {project.status.value.replace('_', ' ').title()}\n"
            if deadline:
                result += f"**Deadline:** {format_timestamp(deadline)}\n"
            result += f"\n**Research Plan:**\n{research_plan}"
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to start research project: {str(e)}")
            return f"Sorry, I couldn't start the research project. Error: {str(e)}"
    
    async def _conduct_literature_review(self, topic: str, metadata: Dict[str, Any]) -> str:
        """Conduct a literature review on a topic"""
        try:
            # Create or find existing project
            project_id = metadata.get("project_id")
            if project_id:
                project = self._find_project_by_id(project_id)
                if not project:
                    return f"Project with ID {project_id} not found."
            else:
                # Create new literature review project
                project = ResearchProject(
                    id=generate_task_id(),
                    title=f"Literature Review: {topic}",
                    description=f"Comprehensive literature review on {topic}",
                    research_type=ResearchType.LITERATURE_REVIEW,
                    status=ResearchStatus.IN_PROGRESS,
                    created_at=datetime.now()
                )
                self._save_research_project(project)
                self.research_projects.append(project)
            
            # Generate literature review content
            review_content = await self._generate_literature_review_content(topic, project)
            
            # Update project with findings
            project.status = ResearchStatus.COMPLETED
            project.findings.extend(review_content.get("findings", []))
            project.conclusions.extend(review_content.get("conclusions", []))
            project.recommendations.extend(review_content.get("recommendations", []))
            
            self._save_research_project(project)
            
            result = f"Literature Review: {topic}\n\n"
            result += f"**Project ID:** {project.id}\n\n"
            result += review_content.get("summary", "")
            
            if review_content.get("key_papers"):
                result += f"\n\n**Key Papers/Sources:**\n"
                for paper in review_content["key_papers"]:
                    result += f"• {paper}\n"
            
            if project.findings:
                result += f"\n\n**Key Findings:**\n"
                for finding in project.findings:
                    result += f"• {finding}\n"
            
            if project.conclusions:
                result += f"\n\n**Conclusions:**\n"
                for conclusion in project.conclusions:
                    result += f"• {conclusion}\n"
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to conduct literature review: {str(e)}")
            return f"Sorry, I couldn't conduct the literature review. Error: {str(e)}"
    
    async def _conduct_market_research(self, topic: str, metadata: Dict[str, Any]) -> str:
        """Conduct market research on a topic"""
        try:
            market_scope = metadata.get("market_scope", "global")
            focus_areas = metadata.get("focus_areas", ["market_size", "trends", "competitors"])
            
            # Generate market research content
            research_content = await self._generate_market_research_content(topic, market_scope, focus_areas)
            
            result = f"Market Research: {topic}\n\n"
            result += f"**Market Scope:** {market_scope.title()}\n"
            result += f"**Focus Areas:** {', '.join(focus_areas)}\n\n"
            result += research_content
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to conduct market research: {str(e)}")
            return f"Sorry, I couldn't conduct the market research. Error: {str(e)}"
    
    async def _conduct_technical_analysis(self, topic: str, metadata: Dict[str, Any]) -> str:
        """Conduct technical analysis"""
        try:
            analysis_type = metadata.get("analysis_type", "feasibility")
            technical_aspects = metadata.get("technical_aspects", ["architecture", "scalability", "security"])
            
            # Generate technical analysis content
            analysis_content = await self._generate_technical_analysis_content(topic, analysis_type, technical_aspects)
            
            result = f"Technical Analysis: {topic}\n\n"
            result += f"**Analysis Type:** {analysis_type.title()}\n"
            result += f"**Technical Aspects:** {', '.join(technical_aspects)}\n\n"
            result += analysis_content
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to conduct technical analysis: {str(e)}")
            return f"Sorry, I couldn't conduct the technical analysis. Error: {str(e)}"
    
    async def _analyze_research_data(self, data_description: str, metadata: Dict[str, Any]) -> str:
        """Analyze research data"""
        try:
            analysis_type = metadata.get("analysis_type", "descriptive")
            data_format = metadata.get("data_format", "text")
            
            # Generate data analysis
            analysis_prompt = f"""
            Analyze the following research data:
            
            Data Description: {data_description}
            Analysis Type: {analysis_type}
            Data Format: {data_format}
            
            Provide:
            1. Data summary and key statistics
            2. Patterns and trends identified
            3. Insights and implications
            4. Recommendations for further analysis
            """
            
            analysis_result = self.generate_response(analysis_prompt, max_length=1024)
            
            result = f"Research Data Analysis\n\n"
            result += f"**Analysis Type:** {analysis_type.title()}\n"
            result += f"**Data Format:** {data_format.title()}\n\n"
            result += analysis_result
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to analyze research data: {str(e)}")
            return f"Sorry, I couldn't analyze the research data. Error: {str(e)}"
    
    async def _summarize_research_findings(self, project_description: str, metadata: Dict[str, Any]) -> str:
        """Summarize research findings"""
        try:
            project_id = metadata.get("project_id")
            if project_id:
                project = self._find_project_by_id(project_id)
                if not project:
                    return f"Project with ID {project_id} not found."
            else:
                return "Please provide a project_id to summarize findings."
            
            # Generate summary
            summary_content = self._create_findings_summary(project)
            
            result = f"Research Findings Summary\n\n"
            result += f"**Project:** {project.title}\n"
            result += f"**Type:** {project.research_type.value.replace('_', ' ').title()}\n"
            result += f"**Status:** {project.status.value.replace('_', ' ').title()}\n\n"
            result += summary_content
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to summarize research findings: {str(e)}")
            return f"Sorry, I couldn't summarize the research findings. Error: {str(e)}"
    
    async def _generate_research_report(self, project_description: str, metadata: Dict[str, Any]) -> str:
        """Generate a comprehensive research report"""
        try:
            project_id = metadata.get("project_id")
            report_format = metadata.get("format", "standard")
            
            if project_id:
                project = self._find_project_by_id(project_id)
                if not project:
                    return f"Project with ID {project_id} not found."
            else:
                return "Please provide a project_id to generate a report."
            
            # Generate comprehensive report
            report_content = self._create_comprehensive_report(project, report_format)
            
            return report_content
            
        except Exception as e:
            self.logger.error(f"Failed to generate research report: {str(e)}")
            return f"Sorry, I couldn't generate the research report. Error: {str(e)}"
    
    async def _list_research_projects(self) -> str:
        """List all research projects"""
        if not self.research_projects:
            return "No research projects found."
        
        result = "Research Projects:\n\n"
        for project in sorted(self.research_projects, key=lambda p: p.created_at, reverse=True):
            status_emoji = {
                ResearchStatus.PENDING: "⏳",
                ResearchStatus.IN_PROGRESS: "🔄",
                ResearchStatus.COMPLETED: "✅",
                ResearchStatus.REQUIRES_REVIEW: "👀"
            }.get(project.status, "❓")
            
            result += f"{status_emoji} **{project.title}**\n"
            result += f"   ID: {project.id}\n"
            result += f"   Type: {project.research_type.value.replace('_', ' ').title()}\n"
            result += f"   Created: {format_timestamp(project.created_at)}\n"
            if project.deadline:
                result += f"   Deadline: {format_timestamp(project.deadline)}\n"
            result += f"   Sources: {len(project.sources)}\n\n"
        
        return result
    
    async def _get_project_status(self, project_identifier: str, metadata: Dict[str, Any]) -> str:
        """Get status of a specific research project"""
        project = self._find_project_by_id(project_identifier)
        if not project:
            # Try to find by title
            for p in self.research_projects:
                if project_identifier.lower() in p.title.lower():
                    project = p
                    break
        
        if not project:
            return f"Project '{project_identifier}' not found."
        
        result = f"Project Status: {project.title}\n\n"
        result += f"**ID:** {project.id}\n"
        result += f"**Type:** {project.research_type.value.replace('_', ' ').title()}\n"
        result += f"**Status:** {project.status.value.replace('_', ' ').title()}\n"
        result += f"**Created:** {format_timestamp(project.created_at)}\n"
        
        if project.deadline:
            result += f"**Deadline:** {format_timestamp(project.deadline)}\n"
            days_remaining = (project.deadline - datetime.now()).days
            if days_remaining > 0:
                result += f"**Days Remaining:** {days_remaining}\n"
            elif days_remaining == 0:
                result += f"**Status:** Due today!\n"
            else:
                result += f"**Status:** Overdue by {abs(days_remaining)} days\n"
        
        result += f"**Sources:** {len(project.sources)}\n"
        result += f"**Findings:** {len(project.findings)}\n"
        result += f"**Conclusions:** {len(project.conclusions)}\n"
        result += f"**Recommendations:** {len(project.recommendations)}\n"
        
        if project.tags:
            result += f"**Tags:** {', '.join(project.tags)}\n"
        
        return result
    
    async def _conduct_general_research(self, topic: str, metadata: Dict[str, Any]) -> str:
        """Conduct general research on a topic"""
        try:
            research_depth = metadata.get("depth", "medium")  # basic, medium, comprehensive
            
            # Generate research content based on depth
            research_prompt = f"""
            Conduct {research_depth} research on the topic: {topic}
            
            Provide:
            1. Overview and background
            2. Current state and trends
            3. Key players and stakeholders
            4. Challenges and opportunities
            5. Future outlook
            6. Relevant sources and references
            
            Make the research comprehensive and well-structured.
            """
            
            research_content = self.generate_response(research_prompt, max_length=1536)
            
            result = f"Research Report: {topic}\n\n"
            result += f"**Research Depth:** {research_depth.title()}\n"
            result += f"**Generated:** {format_timestamp(datetime.now())}\n\n"
            result += research_content
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to conduct general research: {str(e)}")
            return f"Sorry, I couldn't conduct the research. Error: {str(e)}"
    
    # Helper methods
    def _find_project_by_id(self, project_id: str) -> Optional[ResearchProject]:
        """Find research project by ID"""
        for project in self.research_projects:
            if project.id == project_id:
                return project
        return None
    
    def _save_research_project(self, project: ResearchProject):
        """Save research project to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO research_projects 
                (id, title, description, research_type, status, created_at, deadline, findings, conclusions, recommendations, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                project.id, project.title, project.description,
                project.research_type.value, project.status.value,
                project.created_at.isoformat(),
                project.deadline.isoformat() if project.deadline else None,
                json.dumps(project.findings), json.dumps(project.conclusions),
                json.dumps(project.recommendations), json.dumps(project.tags)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to save research project: {str(e)}")
    
    def _generate_research_plan(self, project: ResearchProject) -> str:
        """Generate a research plan for a project"""
        methodology = self.research_methodologies.get(project.research_type)
        if not methodology:
            return "Research plan will be developed based on project requirements."
        
        plan = f"**Research Methodology for {project.research_type.value.replace('_', ' ').title()}:**\n\n"
        
        plan += "**Steps:**\n"
        for i, step in enumerate(methodology["steps"], 1):
            plan += f"{i}. {step}\n"
        
        plan += "\n**Expected Deliverables:**\n"
        for deliverable in methodology["deliverables"]:
            plan += f"• {deliverable}\n"
        
        return plan
    
    async def _generate_literature_review_content(self, topic: str, project: ResearchProject) -> Dict[str, Any]:
        """Generate literature review content"""
        # This would typically involve searching academic databases
        # For now, we'll generate a structured response using the LLM
        
        prompt = f"""
        Conduct a comprehensive literature review on: {topic}
        
        Provide:
        1. A summary of the current state of research
        2. Key findings from major studies
        3. Identified research gaps
        4. Future research directions
        5. List of key papers/sources (simulated)
        
        Structure this as a formal literature review.
        """
        
        content = self.generate_response(prompt, max_length=1536)
        
        return {
            "summary": content,
            "findings": [
                f"Current research shows significant interest in {topic}",
                f"Multiple approaches have been explored for {topic}",
                f"There is consensus on key aspects of {topic}"
            ],
            "conclusions": [
                f"The field of {topic} is rapidly evolving",
                f"More research is needed in specific areas of {topic}"
            ],
            "recommendations": [
                f"Future studies should focus on practical applications of {topic}",
                f"Longitudinal studies would benefit {topic} research"
            ],
            "key_papers": [
                f"Smith et al. (2023): Advances in {topic}",
                f"Johnson & Brown (2022): {topic} Applications",
                f"Davis (2023): Future of {topic}"
            ]
        }
    
    async def _generate_market_research_content(self, topic: str, scope: str, focus_areas: List[str]) -> str:
        """Generate market research content"""
        prompt = f"""
        Conduct market research on: {topic}
        Market Scope: {scope}
        Focus Areas: {', '.join(focus_areas)}
        
        Provide analysis covering:
        1. Market overview and size
        2. Key trends and drivers
        3. Competitive landscape
        4. Opportunities and challenges
        5. Market projections
        """
        
        return self.generate_response(prompt, max_length=1024)
    
    async def _generate_technical_analysis_content(self, topic: str, analysis_type: str, aspects: List[str]) -> str:
        """Generate technical analysis content"""
        prompt = f"""
        Conduct technical analysis on: {topic}
        Analysis Type: {analysis_type}
        Technical Aspects: {', '.join(aspects)}
        
        Provide analysis covering:
        1. Technical feasibility
        2. Architecture considerations
        3. Implementation challenges
        4. Resource requirements
        5. Risk assessment
        """
        
        return self.generate_response(prompt, max_length=1024)
    
    def _create_findings_summary(self, project: ResearchProject) -> str:
        """Create a summary of research findings"""
        summary = f"**Research Summary**\n\n"
        
        if project.findings:
            summary += "**Key Findings:**\n"
            for finding in project.findings:
                summary += f"• {finding}\n"
            summary += "\n"
        
        if project.conclusions:
            summary += "**Conclusions:**\n"
            for conclusion in project.conclusions:
                summary += f"• {conclusion}\n"
            summary += "\n"
        
        if project.recommendations:
            summary += "**Recommendations:**\n"
            for recommendation in project.recommendations:
                summary += f"• {recommendation}\n"
        
        return summary
    
    def _create_comprehensive_report(self, project: ResearchProject, format_type: str) -> str:
        """Create a comprehensive research report"""
        report = f"# Research Report: {project.title}\n\n"
        report += f"**Project ID:** {project.id}\n"
        report += f"**Research Type:** {project.research_type.value.replace('_', ' ').title()}\n"
        report += f"**Status:** {project.status.value.replace('_', ' ').title()}\n"
        report += f"**Created:** {format_timestamp(project.created_at)}\n"
        
        if project.deadline:
            report += f"**Deadline:** {format_timestamp(project.deadline)}\n"
        
        report += f"\n## Executive Summary\n\n"
        report += f"{project.description}\n\n"
        
        if project.sources:
            report += f"## Sources ({len(project.sources)})\n\n"
            for source in project.sources:
                report += f"• {source.title}\n"
            report += "\n"
        
        report += self._create_findings_summary(project)
        
        return report
