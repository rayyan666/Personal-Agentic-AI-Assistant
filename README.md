# AI Personal Agent System

AI personal agent system with both web and terminal interfaces.

## Quick Start

### Terminal Interface (Recommended)

```bash
# Activate virtual environment
source venv/bin/activate

# Start terminal interface
python start_terminal.py
```

### Web Interface

```bash
# Activate virtual environment
source venv/bin/activate

# Start production server
python start_production.py
```

Open http://localhost:8800 in your browser

## Features

- **Streamlined Interface**: Clean, mobile-friendly design
- **7 AI Agents**: Personal assistance, code generation, research, and more
- **Production-Ready**: Optimized for deployment
- **CORS Support**: Ready for cross-origin requests
- **Health Monitoring**: Built-in health check endpoint

## API Endpoints

- **GET /** - Web interface
- **POST /chat** - Send messages to agents
- **GET /health** - System health check
- **GET /agents** - List available agents

## Deployment

For production deployment, consider:

1. Setting up a reverse proxy (Nginx/Apache)
2. Configuring proper CORS settings
3. Adding authentication
4. Setting up SSL/TLS
5. Using a process manager (PM2/Supervisor)
