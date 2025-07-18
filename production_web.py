#!/usr/bin/env python3
"""
Production Web Interface for AI Personal Agent System
Streamlined for production deployment
"""
import asyncio
import json
from datetime import datetime
from pathlib import Path
import sys
import logging

# Add the project directory to Python path
project_dir = Path(__file__).parent
sys.path.insert(0, str(project_dir))

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn

from agent_coordinator import AgentCoordinator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Personal Agent System",
    description="Production web interface for AI personal agents",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global coordinator
coordinator: Optional[AgentCoordinator] = None

# Pydantic models
class ChatMessage(BaseModel):
    message: str
    agent_name: Optional[str] = None

class AgentResponse(BaseModel):
    success: bool
    response: str
    agent_used: str
    processing_time: float

async def get_coordinator():
    """Get or create the agent coordinator"""
    global coordinator
    if coordinator is None:
        coordinator = AgentCoordinator()
    return coordinator

# Production HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Personal Agent System</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        
        .container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
            width: 100%;
            max-width: 900px;
            height: 80vh;
            display: flex;
            flex-direction: column;
        }
        
        .header {
            background: #f8f9fa;
            padding: 20px 30px;
            border-bottom: 1px solid #e9ecef;
            text-align: center;
        }
        
        .header h1 {
            color: #495057;
            font-size: 1.8em;
            font-weight: 700;
            margin-bottom: 5px;
        }
        
        .header p {
            color: #6c757d;
            font-size: 1em;
        }
        
        .agent-selector {
            padding: 15px 30px;
            background: #fff;
            border-bottom: 1px solid #e9ecef;
        }
        
        .agent-selector select {
            width: 100%;
            padding: 12px 15px;
            border: 2px solid #e9ecef;
            border-radius: 10px;
            font-size: 1em;
            background: white;
            cursor: pointer;
        }
        
        .agent-selector select:focus {
            outline: none;
            border-color: #2196f3;
        }
        
        .chat-messages {
            flex: 1;
            padding: 20px 30px;
            overflow-y: auto;
            background: #f8f9fa;
        }
        
        .message {
            margin-bottom: 20px;
            display: flex;
            align-items: flex-start;
        }
        
        .message.user {
            justify-content: flex-end;
        }
        
        .message-content {
            max-width: 80%;
            padding: 15px 20px;
            border-radius: 18px;
            font-size: 0.95em;
            line-height: 1.5;
        }
        
        .message.user .message-content {
            background: #2196f3;
            color: white;
            border-bottom-right-radius: 5px;
        }
        
        .message.agent .message-content {
            background: white;
            color: #495057;
            border: 1px solid #e9ecef;
            border-bottom-left-radius: 5px;
        }
        
        .message-meta {
            font-size: 0.8em;
            color: #6c757d;
            margin-top: 5px;
        }
        
        .chat-input {
            background: white;
            padding: 20px 30px;
            border-top: 1px solid #e9ecef;
        }
        
        .input-group {
            display: flex;
            gap: 15px;
            align-items: flex-end;
        }
        
        .input-field {
            flex: 1;
            min-height: 50px;
            max-height: 120px;
            padding: 15px 20px;
            border: 2px solid #e9ecef;
            border-radius: 25px;
            font-size: 0.95em;
            font-family: inherit;
            resize: none;
            outline: none;
        }
        
        .input-field:focus {
            border-color: #2196f3;
        }
        
        .send-button {
            background: #2196f3;
            color: white;
            border: none;
            padding: 15px 25px;
            border-radius: 25px;
            font-size: 0.95em;
            font-weight: 600;
            cursor: pointer;
            min-width: 80px;
        }
        
        .send-button:hover {
            background: #1976d2;
        }
        
        .send-button:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
            color: #6c757d;
        }
        
        .loading.show {
            display: block;
        }
        
        .welcome-message {
            text-align: center;
            padding: 40px 20px;
            color: #6c757d;
        }
        
        .welcome-message h2 {
            color: #495057;
            margin-bottom: 15px;
            font-weight: 600;
        }
        
        .examples {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin-top: 30px;
        }
        
        .example-item {
            padding: 15px;
            background: white;
            border-radius: 10px;
            cursor: pointer;
            border: 2px solid #e9ecef;
            transition: all 0.2s ease;
        }
        
        .example-item:hover {
            border-color: #2196f3;
            background: #f8f9fa;
        }
        
        @media (max-width: 768px) {
            .container {
                height: 100vh;
                border-radius: 0;
                max-width: 100%;
            }
            
            .header, .agent-selector, .chat-messages, .chat-input {
                padding-left: 20px;
                padding-right: 20px;
            }
            
            .examples {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 AI Personal Agent System</h1>
            <p>Your intelligent assistant for daily tasks</p>
        </div>

        <div class="agent-selector">
            <select id="agentSelect">
                <option value="">Auto-Select Agent</option>
                <option value="personal_assistant">👤 Personal Assistant</option>
                <option value="code_generation">💻 Code Generation</option>
                <option value="research_agent">🔬 Research Agent</option>
                <option value="resource_finder">📚 Resource Finder</option>
                <option value="email_agent">📧 Email Agent</option>
                <option value="web_scraper">🕷️ Web Scraper</option>
                <option value="task_manager">✅ Task Manager</option>
            </select>
        </div>

        <div class="chat-messages" id="chatMessages">
            <div class="welcome-message">
                <h2>Welcome to AI Personal Agent System</h2>
                <p>Choose an agent above or let the system auto-select the best one for your request.</p>

                <div class="examples">
                    <div class="example-item" onclick="sendExample('Generate a Python function to calculate factorial')">
                        <strong>Code Generation</strong><br>
                        Generate a Python function to calculate factorial
                    </div>
                    <div class="example-item" onclick="sendExample('Find learning resources for machine learning')">
                        <strong>Resource Finder</strong><br>
                        Find learning resources for machine learning
                    </div>
                    <div class="example-item" onclick="sendExample('Write a professional email requesting a meeting')">
                        <strong>Email Agent</strong><br>
                        Write a professional email requesting a meeting
                    </div>
                    <div class="example-item" onclick="sendExample('Research the latest trends in AI')">
                        <strong>Research Agent</strong><br>
                        Research the latest trends in AI
                    </div>
                    <div class="example-item" onclick="sendExample('Help me schedule my daily tasks')">
                        <strong>Personal Assistant</strong><br>
                        Help me schedule my daily tasks
                    </div>
                    <div class="example-item" onclick="sendExample('Create a task list for my project')">
                        <strong>Task Manager</strong><br>
                        Create a task list for my project
                    </div>
                </div>
            </div>
        </div>

        <div class="loading" id="loading">
            <p>🤖 Processing your request...</p>
        </div>

        <div class="chat-input">
            <div class="input-group">
                <textarea
                    id="messageInput"
                    class="input-field"
                    placeholder="Type your message here..."
                    rows="1"
                ></textarea>
                <button id="sendButton" class="send-button" onclick="sendMessage()">Send</button>
            </div>
        </div>
    </div>

    <script>
        let isProcessing = false;

        // Auto-resize textarea
        document.getElementById('messageInput').addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 120) + 'px';
        });

        // Send message on Enter (but allow Shift+Enter for new lines)
        document.getElementById('messageInput').addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        function sendExample(message) {
            document.getElementById('messageInput').value = message;
            sendMessage();
        }

        async function sendMessage() {
            if (isProcessing) return;

            const messageInput = document.getElementById('messageInput');
            const message = messageInput.value.trim();

            if (!message) return;

            const agentSelect = document.getElementById('agentSelect');
            const selectedAgent = agentSelect.value || null;

            // Clear input and show user message
            messageInput.value = '';
            messageInput.style.height = 'auto';
            addMessage(message, 'user');

            // Show loading
            showLoading(true);
            isProcessing = true;

            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        message: message,
                        agent_name: selectedAgent
                    })
                });

                const data = await response.json();

                if (data.success) {
                    addMessage(data.response, 'agent', data.agent_used, data.processing_time);
                } else {
                    addMessage('Sorry, I encountered an error processing your request.', 'agent');
                }

            } catch (error) {
                console.error('Error:', error);
                addMessage('Sorry, I encountered a network error. Please try again.', 'agent');
            } finally {
                showLoading(false);
                isProcessing = false;
            }
        }

        function addMessage(content, type, agentUsed = null, processingTime = null) {
            const messagesContainer = document.getElementById('chatMessages');
            const welcomeMessage = messagesContainer.querySelector('.welcome-message');

            // Remove welcome message on first interaction
            if (welcomeMessage) {
                welcomeMessage.remove();
            }

            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${type}`;

            const contentDiv = document.createElement('div');
            contentDiv.className = 'message-content';

            // Format content (basic markdown support)
            const formattedContent = formatMessage(content);
            contentDiv.innerHTML = formattedContent;

            const metaDiv = document.createElement('div');
            metaDiv.className = 'message-meta';

            if (type === 'user') {
                metaDiv.textContent = 'You • ' + new Date().toLocaleTimeString();
            } else {
                const agentName = agentUsed || 'AI Agent';
                const timeInfo = processingTime ? ` • ${processingTime.toFixed(1)}s` : '';
                metaDiv.textContent = `${agentName}${timeInfo} • ${new Date().toLocaleTimeString()}`;
            }

            messageDiv.appendChild(contentDiv);
            messageDiv.appendChild(metaDiv);
            messagesContainer.appendChild(messageDiv);

            // Scroll to bottom
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }

        function formatMessage(content) {
            // Basic markdown formatting
            return content
                .replace(/```([\\s\\S]*?)```/g, '<pre><code>$1</code></pre>')
                .replace(/`([^`]+)`/g, '<code style="background: #f1f3f4; padding: 2px 6px; border-radius: 4px;">$1</code>')
                .replace(/\\*\\*([^*]+)\\*\\*/g, '<strong>$1</strong>')
                .replace(/\\*([^*]+)\\*/g, '<em>$1</em>')
                .replace(/\\n/g, '<br>');
        }

        function showLoading(show) {
            const loading = document.getElementById('loading');
            const sendButton = document.getElementById('sendButton');

            if (show) {
                loading.classList.add('show');
                sendButton.disabled = true;
                sendButton.textContent = '...';
            } else {
                loading.classList.remove('show');
                sendButton.disabled = false;
                sendButton.textContent = 'Send';
            }
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the main web interface"""
    return HTMLResponse(content=HTML_TEMPLATE)

@app.post("/chat")
async def chat_endpoint(message_data: ChatMessage):
    """Handle chat messages from the web interface"""
    try:
        coordinator = await get_coordinator()

        # Process the message
        start_time = datetime.now()
        response_content = await coordinator.process_request(
            message_data.message,
            agent_name=message_data.agent_name
        )
        end_time = datetime.now()

        processing_time = (end_time - start_time).total_seconds()

        # Determine which agent was used
        agent_used = message_data.agent_name or "Auto-Selected Agent"
        if agent_used != "Auto-Selected Agent":
            agent_used = agent_used.replace('_', ' ').title()

        return AgentResponse(
            success=True,
            response=response_content,
            agent_used=agent_used,
            processing_time=processing_time
        )

    except Exception as e:
        logger.error(f"Error processing chat message: {str(e)}")
        return AgentResponse(
            success=False,
            response=f"I encountered an error: {str(e)}",
            agent_used="System",
            processing_time=0.0
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        coordinator = await get_coordinator()
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "agents_available": len(coordinator.agents) if coordinator.agents else 0
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/agents")
async def get_agents():
    """Get available agents"""
    try:
        coordinator = await get_coordinator()
        agents_info = []

        if coordinator.agents:
            for agent_name, agent in coordinator.agents.items():
                agents_info.append({
                    "name": agent_name,
                    "display_name": agent_name.replace('_', ' ').title(),
                    "description": getattr(agent, 'description', 'AI Agent'),
                    "status": "ready"
                })

        return {
            "agents": agents_info,
            "total": len(agents_info)
        }

    except Exception as e:
        logger.error(f"Error getting agents: {str(e)}")
        return {"agents": [], "total": 0, "error": str(e)}

async def main():
    """Main function to run the web interface"""
    try:
        logger.info("🚀 Starting AI Personal Agent System Web Interface...")
        logger.info("📱 Web interface will be available at: http://localhost:8800")

        # Initialize coordinator
        await get_coordinator()

        # Run the server
        config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=8800,
            log_level="info",
            access_log=True
        )
        server = uvicorn.Server(config)
        await server.serve()

    except Exception as e:
        logger.error(f"Failed to start web interface: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
