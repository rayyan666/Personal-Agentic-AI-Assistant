# AI Personal Agent System

A comprehensive artificial intelligence system featuring specialized agents for various tasks including code generation, research, personal assistance, and more. The system provides both web-based and terminal interfaces for flexible interaction with AI agents.

## Overview

This AI Personal Agent System is designed to provide intelligent assistance across multiple domains through specialized agents. Each agent is optimized for specific tasks and can work independently or in coordination with other agents to provide comprehensive solutions.

## System Architecture

The system consists of seven specialized AI agents:

1. **Personal Assistant Agent** - Manages schedules, reminders, and general assistance
2. **Code Generation Agent** - Generates, reviews, and explains code with task-specific optimization
3. **Research Agent** - Conducts research, analyzes information, and provides insights
4. **Resource Finder Agent** - Locates and recommends learning resources and materials
5. **Email Agent** - Generates and manages email communications
6. **Web Scraper Agent** - Extracts and processes data from web sources
7. **Task Manager Agent** - Organizes and tracks tasks and projects

## Installation

### Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)
- Required dependencies listed in requirements.txt

### Setup Instructions

1. Clone the repository and navigate to the project directory:
   ```bash
   cd projects/agentic_ai_project_2
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Terminal Interface (Recommended)

The terminal interface provides direct command-line access to all agents with full functionality.

#### Interactive Mode
```bash
# Activate virtual environment
source venv/bin/activate

# Start interactive terminal interface
python start_terminal.py
```

#### Single Command Mode
```bash
# Execute a single command with a specific agent
python terminal_app.py --agent code_generation --message "create a hello world function"

# List all available agents
python terminal_app.py --list
```

#### Terminal Commands
- `help` - Display available commands
- `clear` - Clear chat history
- `history` - Show conversation history
- `save` - Save chat history to file
- `quit`, `exit`, `back` - Return to agent selection

### Web Interface

The web interface provides a browser-based GUI for interacting with agents.

```bash
# Activate virtual environment
source venv/bin/activate

# Start web server
python start_production.py
```

Access the interface at: http://localhost:8800

## Agent Capabilities

### Code Generation Agent

**Specialized Features:**
- Task-specific code generation with 15+ categories
- Template-based responses for common requests (instant)
- Advanced AI-powered generation for complex tasks
- Support for multiple programming languages
- Automatic code validation and formatting

**Supported Task Types:**
- API Development (FastAPI, Flask, REST APIs)
- Data Processing (Pandas, CSV, JSON handling)
- Algorithm Implementation (Sorting, searching, mathematical)
- Object-Oriented Programming (Classes, inheritance)
- File I/O Operations (Text, JSON, CSV files)
- Database Operations (SQLite, generic database patterns)
- Unit Testing (Test cases, mocking, assertions)
- Machine Learning (Scikit-learn, neural networks)
- Web Development (Full-stack applications)
- Concurrency (Async/await, threading)

### Personal Assistant Agent

**Core Functions:**
- Schedule management and calendar integration
- Reminder and notification systems
- Task organization and prioritization
- General assistance and information retrieval

### Research Agent

**Research Capabilities:**
- Information gathering and analysis
- Source verification and citation
- Research project management
- Data synthesis and reporting

### Resource Finder Agent

**Resource Discovery:**
- Learning material recommendations
- Course and tutorial suggestions
- Documentation and reference finding
- Skill development path planning

### Email Agent

**Email Management:**
- Professional email composition
- Template-based email generation
- Email formatting and structure optimization
- Communication workflow automation

### Web Scraper Agent

**Data Extraction:**
- Web content extraction and parsing
- Structured data collection
- Automated data processing workflows
- Content monitoring and updates

### Task Manager Agent

**Project Management:**
- Task creation and tracking
- Project organization and planning
- Time management and scheduling
- Progress monitoring and reporting

## Configuration

### Model Configuration

The system uses configurable AI models for each agent. Default configuration uses Microsoft DialoGPT-medium for fast performance. Advanced users can configure specialized models:

```python
# config.py - Example configuration
"code_generation": {
    "model": "Salesforce/codet5p-770m-py",  # Specialized code generation
    "temperature": 0.2,
    "max_tokens": 2048
}
```

### Database Configuration

Each agent maintains its own SQLite database for persistent storage:
- Personal Assistant: Reminders and appointments
- Code Generation: Code snippets and reviews
- Research: Projects and sources
- Task Manager: Tasks and projects

## API Reference

### Web API Endpoints

- `GET /` - Web interface
- `POST /chat` - Send messages to agents
- `GET /health` - System health check
- `GET /agents` - List available agents and capabilities

### Request Format

```json
{
    "message": "Your request message",
    "agent": "agent_name",
    "task_type": "optional_task_type"
}
```

### Response Format

```json
{
    "response": "Agent response",
    "agent": "responding_agent",
    "processing_time": 0.05,
    "task_id": "unique_task_identifier"
}
```

## Performance Characteristics

### Response Times
- Template-based responses: < 0.01 seconds
- AI-generated responses: 1-5 seconds
- Complex multi-step tasks: 5-30 seconds

### Resource Usage
- Memory: 2-4 GB (depending on loaded models)
- CPU: Moderate usage during processing
- Storage: Minimal (databases grow with usage)

## Development

### Project Structure
```
projects/agentic_ai_project_2/
├── agents/                 # Agent implementations
├── config.py              # System configuration
├── agent_coordinator.py   # Central coordination logic
├── terminal_app.py        # Terminal interface
├── production_web.py      # Web interface
├── requirements.txt       # Dependencies
└── databases/             # SQLite databases
```

### Adding New Agents

1. Create agent class inheriting from `BaseAgent`
2. Implement required methods (`_process_task_impl`)
3. Add agent to `AgentCoordinator`
4. Update configuration in `config.py`

## Deployment

### Production Considerations

1. **Reverse Proxy Setup**: Configure Nginx or Apache for production
2. **SSL/TLS Configuration**: Implement HTTPS for secure communications
3. **Authentication**: Add user authentication and authorization
4. **Process Management**: Use PM2, Supervisor, or systemd for process management
5. **Monitoring**: Implement logging and monitoring solutions
6. **Scaling**: Consider load balancing for high-traffic deployments

### Environment Variables

```bash
export AI_AGENT_PORT=8800
export AI_AGENT_HOST=0.0.0.0
export AI_AGENT_DEBUG=false
```

## Troubleshooting

### Common Issues

1. **Model Loading Errors**: Ensure sufficient memory and stable internet connection
2. **Port Conflicts**: Change port in configuration if 8800 is occupied
3. **Permission Errors**: Verify file permissions for database directories
4. **Memory Issues**: Consider using lighter models for resource-constrained environments

### Logging

System logs are available through Python's logging module. Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## License

This project is provided as-is for educational and development purposes.

## Support

For technical support and questions, refer to the system logs and error messages for detailed debugging information.
