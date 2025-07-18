"""
Base Agent Framework for the Agentic AI Personal Agent System
"""
import asyncio
import logging
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

from config import config, get_agent_config

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.log_file),
        logging.StreamHandler()
    ]
)

class AgentStatus(Enum):
    """Agent status enumeration"""
    IDLE = "idle"
    PROCESSING = "processing"
    ERROR = "error"
    DISABLED = "disabled"

class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4

@dataclass
class AgentTask:
    """Represents a task for an agent"""
    id: str
    agent_name: str
    task_type: str
    content: str
    priority: TaskPriority = TaskPriority.MEDIUM
    metadata: Dict[str, Any] = None
    created_at: datetime = None
    completed_at: Optional[datetime] = None
    status: str = "pending"
    result: Optional[str] = None
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.metadata is None:
            self.metadata = {}

@dataclass
class AgentResponse:
    """Represents an agent's response"""
    agent_name: str
    task_id: str
    success: bool
    content: str
    metadata: Dict[str, Any] = None
    processing_time: float = 0.0
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}

class BaseAgent(ABC):
    """Base class for all agents in the system"""
    
    def __init__(self, name: str, description: str = "", skip_model_init: bool = False):
        self.name = name
        self.description = description
        self.status = AgentStatus.IDLE
        self.logger = logging.getLogger(f"Agent.{name}")
        self.config = get_agent_config(name.lower().replace(" ", "_"))
        self.task_history: List[AgentTask] = []
        self.model = None
        self.tokenizer = None
        self.pipeline = None

        # Initialize the LLM model (skip in test mode)
        if not skip_model_init:
            self._initialize_model()
        else:
            self.logger.info(f"Agent {self.name} initialized in test mode (model loading skipped)")

        self.logger.info(f"Agent {self.name} initialized successfully")
    
    def _initialize_model(self):
        """Initialize the LLM model for the agent"""
        try:
            model_name = self.config.get("model", config.default_model)
            self.logger.info(f"Loading model: {model_name}")
            
            # Use a lightweight model for demonstration
            # In production, you might want to use more powerful models
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Add padding token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Initialize text generation pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=model_name,
                tokenizer=self.tokenizer,
                max_length=self.config.get("max_tokens", config.max_tokens),
                temperature=self.config.get("temperature", config.temperature),
                do_sample=True,
                device=0 if torch.cuda.is_available() else -1
            )
            
            self.logger.info(f"Model {model_name} loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize model: {str(e)}")
            # Fallback to a simple text generation approach
            self.pipeline = None
    
    async def process_task(self, task: AgentTask) -> AgentResponse:
        """Process a task and return a response"""
        start_time = datetime.now()
        self.status = AgentStatus.PROCESSING
        
        try:
            self.logger.info(f"Processing task {task.id} of type {task.task_type}")
            
            # Add task to history
            self.task_history.append(task)
            
            # Process the task using the specific agent implementation
            result = await self._process_task_impl(task)
            
            # Update task status
            task.status = "completed"
            task.completed_at = datetime.now()
            task.result = result
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            response = AgentResponse(
                agent_name=self.name,
                task_id=task.id,
                success=True,
                content=result,
                processing_time=processing_time,
                metadata={"task_type": task.task_type}
            )
            
            self.status = AgentStatus.IDLE
            self.logger.info(f"Task {task.id} completed successfully in {processing_time:.2f}s")
            
            return response
            
        except Exception as e:
            self.status = AgentStatus.ERROR
            error_msg = f"Error processing task {task.id}: {str(e)}"
            self.logger.error(error_msg)
            
            # Update task with error
            task.status = "error"
            task.error = str(e)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return AgentResponse(
                agent_name=self.name,
                task_id=task.id,
                success=False,
                content=error_msg,
                processing_time=processing_time,
                metadata={"error": str(e)}
            )
    
    @abstractmethod
    async def _process_task_impl(self, task: AgentTask) -> str:
        """Implementation-specific task processing logic"""
        pass
    
    def generate_response(self, prompt: str, max_length: int = None) -> str:
        """Generate a response using the LLM"""
        try:
            if self.pipeline is None:
                # Fallback response generation
                return self._fallback_response(prompt)
            
            # Prepare the prompt with system context
            system_prompt = self.config.get("system_prompt", "")
            full_prompt = f"{system_prompt}\n\nUser: {prompt}\nAssistant:"
            
            max_len = max_length or self.config.get("max_tokens", 512)
            
            # Generate response
            outputs = self.pipeline(
                full_prompt,
                max_length=len(full_prompt.split()) + max_len,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Extract the generated text
            generated_text = outputs[0]['generated_text']
            
            # Remove the prompt from the response
            if "Assistant:" in generated_text:
                response = generated_text.split("Assistant:")[-1].strip()
            else:
                response = generated_text[len(full_prompt):].strip()
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            return self._fallback_response(prompt)
    
    def _fallback_response(self, prompt: str) -> str:
        """Fallback response when LLM is not available"""
        return f"I understand you're asking about: {prompt}. I'm currently operating in fallback mode. Please check the system configuration."
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        return {
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "tasks_processed": len(self.task_history),
            "last_activity": self.task_history[-1].created_at.isoformat() if self.task_history else None,
            "config": self.config
        }
    
    def get_task_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent task history"""
        recent_tasks = self.task_history[-limit:] if self.task_history else []
        return [asdict(task) for task in recent_tasks]
    
    async def health_check(self) -> bool:
        """Perform a health check on the agent"""
        try:
            # Simple test to ensure the agent is responsive
            test_task = AgentTask(
                id="health_check",
                agent_name=self.name,
                task_type="health_check",
                content="System health check"
            )
            
            response = await self.process_task(test_task)
            return response.success
            
        except Exception as e:
            self.logger.error(f"Health check failed: {str(e)}")
            return False
