"""
Agent Coordinator - Central coordinator for all specialized agents
"""
import asyncio
import logging
from typing import Dict, Any, List, Optional, Type
from datetime import datetime

from agents.base_agent import BaseAgent, AgentTask, AgentResponse, TaskPriority
from agents.personal_assistant import PersonalAssistantAgent
from agents.code_generation import CodeGenerationAgent
from agents.research_agent import ResearchAgent
from agents.resource_finder import ResourceFinderAgent
from agents.email_agent import EmailAgent
from agents.web_scraper import WebScraperAgent
from agents.task_manager import TaskManagerAgent
from agents.utils import generate_task_id, global_event_emitter
from config import config, is_agent_enabled

class AgentCoordinator:
    """Central coordinator for managing all specialized agents"""
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.logger = logging.getLogger("AgentCoordinator")
        
        # Initialize all agents
        self._initialize_agents()
        
        # Setup event listeners
        self._setup_event_listeners()
        
        self.logger.info("Agent Coordinator initialized successfully")
    
    def _initialize_agents(self):
        """Initialize all available agents"""
        agent_classes = {
            "personal_assistant": PersonalAssistantAgent,
            "code_generation": CodeGenerationAgent,
            "research_agent": ResearchAgent,
            "resource_finder": ResourceFinderAgent,
            "email_agent": EmailAgent,
            "web_scraper": WebScraperAgent,
            "task_manager": TaskManagerAgent
        }
        
        for agent_name, agent_class in agent_classes.items():
            if is_agent_enabled(agent_name):
                try:
                    agent = agent_class()
                    self.agents[agent_name] = agent
                    self.logger.info(f"Initialized {agent_name} agent")
                except Exception as e:
                    self.logger.error(f"Failed to initialize {agent_name} agent: {str(e)}")
            else:
                self.logger.info(f"Agent {agent_name} is disabled")
    
    def _setup_event_listeners(self):
        """Setup event listeners for inter-agent communication"""
        global_event_emitter.on("task_completed", self._on_task_completed)
        global_event_emitter.on("agent_error", self._on_agent_error)
    
    async def _on_task_completed(self, agent_name: str, task_id: str, result: str):
        """Handle task completion events"""
        self.logger.info(f"Task {task_id} completed by {agent_name}")
    
    async def _on_agent_error(self, agent_name: str, error: str):
        """Handle agent error events"""
        self.logger.error(f"Error in {agent_name}: {error}")
    
    async def process_request(self, request: str, agent_name: Optional[str] = None, 
                            task_type: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Process a user request through the appropriate agent"""
        try:
            # Determine which agent to use
            if agent_name and agent_name in self.agents:
                target_agent = self.agents[agent_name]
            else:
                target_agent = self._route_request(request, task_type)
            
            if not target_agent:
                return "I'm sorry, I couldn't determine which agent should handle your request. Please specify an agent or rephrase your request."
            
            # Create task
            task = AgentTask(
                id=generate_task_id(),
                agent_name=target_agent.name,
                task_type=task_type or "general",
                content=request,
                priority=TaskPriority.MEDIUM,
                metadata=metadata or {}
            )
            
            # Process task
            response = await target_agent.process_task(task)
            
            # Emit completion event
            await global_event_emitter.emit("task_completed", target_agent.name, task.id, response.content)
            
            return response.content
            
        except Exception as e:
            self.logger.error(f"Error processing request: {str(e)}")
            return f"I encountered an error while processing your request: {str(e)}"
    
    def _route_request(self, request: str, task_type: Optional[str] = None) -> Optional[BaseAgent]:
        """Route request to the most appropriate agent"""
        request_lower = request.lower()
        
        # Task type based routing
        if task_type:
            task_type_routing = {
                "code": "code_generation",
                "research": "research_agent",
                "email": "email_agent",
                "scrape": "web_scraper",
                "task": "task_manager",
                "resource": "resource_finder",
                "schedule": "personal_assistant"
            }
            
            for key, agent_name in task_type_routing.items():
                if key in task_type.lower() and agent_name in self.agents:
                    return self.agents[agent_name]
        
        # Keyword based routing
        routing_keywords = {
            "personal_assistant": [
                "remind", "schedule", "appointment", "calendar", "meeting",
                "weather", "time", "date", "help", "assistant"
            ],
            "code_generation": [
                "code", "program", "function", "class", "debug", "review",
                "python", "javascript", "java", "programming", "algorithm"
            ],
            "research_agent": [
                "research", "study", "analyze", "literature", "paper",
                "market research", "analysis", "investigate", "findings"
            ],
            "resource_finder": [
                "learn", "tutorial", "course", "resource", "guide",
                "book", "material", "training", "education", "study"
            ],
            "email_agent": [
                "email", "compose", "write email", "send", "draft",
                "template", "message", "correspondence"
            ],
            "web_scraper": [
                "scrape", "extract", "website", "data", "crawl",
                "web data", "harvest", "collect from web"
            ],
            "task_manager": [
                "task", "todo", "productivity", "organize", "manage",
                "project", "deadline", "priority", "complete"
            ]
        }
        
        # Score each agent based on keyword matches
        agent_scores = {}
        for agent_name, keywords in routing_keywords.items():
            if agent_name in self.agents:
                score = sum(1 for keyword in keywords if keyword in request_lower)
                if score > 0:
                    agent_scores[agent_name] = score
        
        # Return agent with highest score
        if agent_scores:
            best_agent = max(agent_scores, key=agent_scores.get)
            return self.agents[best_agent]
        
        # Default to personal assistant if available
        return self.agents.get("personal_assistant")
    
    async def get_agent_status(self, agent_name: Optional[str] = None) -> Dict[str, Any]:
        """Get status of one or all agents"""
        if agent_name:
            if agent_name in self.agents:
                return self.agents[agent_name].get_status()
            else:
                return {"error": f"Agent {agent_name} not found"}
        else:
            return {
                agent_name: agent.get_status()
                for agent_name, agent in self.agents.items()
            }
    
    async def health_check(self) -> Dict[str, bool]:
        """Perform health check on all agents"""
        health_status = {}
        
        for agent_name, agent in self.agents.items():
            try:
                is_healthy = await agent.health_check()
                health_status[agent_name] = is_healthy
            except Exception as e:
                self.logger.error(f"Health check failed for {agent_name}: {str(e)}")
                health_status[agent_name] = False
        
        return health_status
    
    def list_agents(self) -> List[Dict[str, Any]]:
        """List all available agents and their capabilities"""
        agent_list = []
        
        for agent_name, agent in self.agents.items():
            agent_info = {
                "name": agent_name,
                "display_name": agent.name.replace("_", " ").title(),
                "description": agent.description,
                "status": agent.status.value,
                "capabilities": self._get_agent_capabilities(agent_name)
            }
            agent_list.append(agent_info)
        
        return agent_list
    
    def _get_agent_capabilities(self, agent_name: str) -> List[str]:
        """Get list of capabilities for an agent"""
        capabilities = {
            "personal_assistant": [
                "Schedule management", "Reminders", "General assistance",
                "Daily planning", "Time management"
            ],
            "code_generation": [
                "Code generation", "Code review", "Debugging",
                "Code optimization", "Documentation", "Testing"
            ],
            "research_agent": [
                "Literature review", "Market research", "Data analysis",
                "Technical analysis", "Report generation"
            ],
            "resource_finder": [
                "Learning resources", "Tutorial recommendations",
                "Course suggestions", "Resource comparison", "Learning paths"
            ],
            "email_agent": [
                "Email composition", "Template creation", "Email improvement",
                "Subject line generation", "Email formatting"
            ],
            "web_scraper": [
                "Web data extraction", "Website monitoring",
                "Search result scraping", "Data collection", "Content analysis"
            ],
            "task_manager": [
                "Task creation", "Project management", "Productivity tracking",
                "Schedule optimization", "Time blocking", "Progress monitoring"
            ]
        }
        
        return capabilities.get(agent_name, [])
    
    async def process_multi_agent_task(self, request: str, agent_sequence: List[str], 
                                     metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """Process a task that requires multiple agents in sequence"""
        results = []
        current_input = request
        
        for agent_name in agent_sequence:
            if agent_name not in self.agents:
                results.append(f"Error: Agent {agent_name} not available")
                break
            
            try:
                result = await self.process_request(
                    current_input, 
                    agent_name=agent_name, 
                    metadata=metadata
                )
                results.append(result)
                
                # Use the result as input for the next agent
                current_input = result
                
            except Exception as e:
                results.append(f"Error in {agent_name}: {str(e)}")
                break
        
        return results
    
    async def get_recommendations(self, user_context: str) -> List[str]:
        """Get recommendations for what the user might want to do"""
        recommendations = []
        
        context_lower = user_context.lower()
        
        # Context-based recommendations
        if "work" in context_lower or "project" in context_lower:
            recommendations.extend([
                "Create a task list for your project",
                "Set up time blocks for focused work",
                "Research best practices for your domain",
                "Find learning resources to improve skills"
            ])
        
        if "learn" in context_lower or "study" in context_lower:
            recommendations.extend([
                "Find comprehensive learning resources",
                "Create a structured learning path",
                "Set up study schedule and reminders",
                "Research the latest trends in your field"
            ])
        
        if "email" in context_lower or "communication" in context_lower:
            recommendations.extend([
                "Compose professional emails",
                "Create email templates for common scenarios",
                "Improve existing email drafts",
                "Generate compelling subject lines"
            ])
        
        if "data" in context_lower or "information" in context_lower:
            recommendations.extend([
                "Scrape data from websites",
                "Conduct research on specific topics",
                "Analyze and summarize findings",
                "Monitor websites for changes"
            ])
        
        # General recommendations if no specific context
        if not recommendations:
            recommendations = [
                "Ask me to help with daily task management",
                "Request code generation or review",
                "Get research assistance on any topic",
                "Find learning resources for new skills",
                "Compose emails or create templates",
                "Extract data from websites",
                "Set up reminders and schedules"
            ]
        
        return recommendations[:5]  # Return top 5 recommendations
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for all agents"""
        stats = {
            "total_agents": len(self.agents),
            "active_agents": len([a for a in self.agents.values() if a.status.value != "disabled"]),
            "agent_details": {}
        }
        
        for agent_name, agent in self.agents.items():
            agent_stats = {
                "tasks_processed": len(agent.task_history),
                "status": agent.status.value,
                "last_activity": None
            }
            
            if agent.task_history:
                last_task = max(agent.task_history, key=lambda t: t.created_at)
                agent_stats["last_activity"] = last_task.created_at.isoformat()
            
            stats["agent_details"][agent_name] = agent_stats
        
        return stats
