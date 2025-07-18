#!/usr/bin/env python3
"""
AI Personal Agent System - Terminal Interface
Terminal-based interface for interacting with AI agents directly.
"""

import asyncio
import sys
import os
import json
from typing import Dict, Any, Optional
from datetime import datetime
import argparse

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent_coordinator import AgentCoordinator
from agents.base_agent import AgentTask

class TerminalInterface:
    """Terminal-based interface for AI agents."""
    
    def __init__(self):
        self.coordinator = None
        self.current_agent = None
        self.session_history = []
        
    async def initialize(self):
        """Initialize the agent coordinator."""
        print("🚀 Initializing AI Personal Agent System...")
        print("=" * 50)

        try:
            self.coordinator = AgentCoordinator()
            print("✅ All agents loaded successfully!")
            print(f"📱 Available agents: {len(self.coordinator.agents)}")
            print("=" * 50)
            return True
        except Exception as e:
            print(f"❌ Failed to initialize agents: {e}")
            return False
    
    def display_agents(self):
        """Display available agents."""
        print("\n🤖 Available AI Agents:")
        print("-" * 30)
        
        for i, (agent_name, agent) in enumerate(self.coordinator.agents.items(), 1):
            status = "🟢 Ready" if agent else "🔴 Error"
            description = agent.description if agent else "Failed to load"
            print(f"{i}. {agent_name.replace('_', ' ').title()}")
            print(f"   Status: {status}")
            print(f"   Description: {description}")
            print()
    
    def select_agent(self) -> Optional[str]:
        """Allow user to select an agent."""
        self.display_agents()
        
        try:
            choice = input("Select an agent (number or name): ").strip()
            
            # Try to parse as number
            if choice.isdigit():
                agent_names = list(self.coordinator.agents.keys())
                idx = int(choice) - 1
                if 0 <= idx < len(agent_names):
                    return agent_names[idx]
            
            # Try to match by name
            choice_lower = choice.lower().replace(' ', '_')
            for agent_name in self.coordinator.agents.keys():
                if agent_name.lower() == choice_lower:
                    return agent_name
                if agent_name.lower().startswith(choice_lower):
                    return agent_name
            
            print(f"❌ Agent '{choice}' not found.")
            return None
            
        except (ValueError, KeyboardInterrupt):
            return None
    
    async def chat_with_agent(self, agent_name: str):
        """Start a chat session with the selected agent."""
        agent = self.coordinator.agents.get(agent_name)
        if not agent:
            print(f"❌ Agent '{agent_name}' not available.")
            return
        
        print(f"\n💬 Starting chat with {agent_name.replace('_', ' ').title()}")
        print(f"📝 Description: {agent.description}")
        print("-" * 50)
        print("Type 'quit', 'exit', or 'back' to return to agent selection")
        print("Type 'help' for available commands")
        print("Type 'clear' to clear chat history")
        print("-" * 50)
        
        chat_history = []
        
        while True:
            try:
                # Get user input
                user_input = input(f"\n[{agent_name}] You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle special commands
                if user_input.lower() in ['quit', 'exit', 'back']:
                    print("👋 Returning to agent selection...")
                    break
                
                elif user_input.lower() == 'help':
                    self.show_help()
                    continue
                
                elif user_input.lower() == 'clear':
                    chat_history.clear()
                    print("🧹 Chat history cleared.")
                    continue
                
                elif user_input.lower() == 'history':
                    self.show_chat_history(chat_history)
                    continue
                
                elif user_input.lower() == 'save':
                    self.save_chat_history(agent_name, chat_history)
                    continue
                
                # Process the message
                print(f"[{agent_name}] Agent: 🤔 Processing...")
                
                # Create task
                task = AgentTask(
                    id=f"terminal_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    agent_name=agent_name,
                    task_type="chat",
                    content=user_input,
                    priority=1
                )
                
                # Get response
                start_time = datetime.now()
                response = await self.coordinator.process_request(user_input, agent_name, "chat")
                end_time = datetime.now()
                
                # Display response
                processing_time = (end_time - start_time).total_seconds()
                print(f"[{agent_name}] Agent: {response}")
                print(f"⏱️  Processing time: {processing_time:.2f}s")
                
                # Add to history
                chat_history.append({
                    "timestamp": start_time.isoformat(),
                    "user": user_input,
                    "agent": response,
                    "processing_time": processing_time
                })
                
            except KeyboardInterrupt:
                print("\n👋 Chat interrupted. Returning to agent selection...")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                continue
    
    def show_help(self):
        """Show available commands."""
        print("\n📚 Available Commands:")
        print("-" * 20)
        print("help     - Show this help message")
        print("clear    - Clear chat history")
        print("history  - Show chat history")
        print("save     - Save chat history to file")
        print("quit     - Return to agent selection")
        print("exit     - Return to agent selection")
        print("back     - Return to agent selection")
    
    def show_chat_history(self, history):
        """Show chat history."""
        if not history:
            print("📝 No chat history available.")
            return
        
        print(f"\n📜 Chat History ({len(history)} messages):")
        print("-" * 40)
        
        for i, entry in enumerate(history[-10:], 1):  # Show last 10 messages
            timestamp = datetime.fromisoformat(entry["timestamp"]).strftime("%H:%M:%S")
            print(f"{i}. [{timestamp}] You: {entry['user'][:50]}{'...' if len(entry['user']) > 50 else ''}")
            print(f"   Agent: {entry['agent'][:50]}{'...' if len(entry['agent']) > 50 else ''}")
            print(f"   Time: {entry['processing_time']:.2f}s")
            print()
    
    def save_chat_history(self, agent_name: str, history):
        """Save chat history to file."""
        if not history:
            print("📝 No chat history to save.")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chat_history_{agent_name}_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump({
                    "agent": agent_name,
                    "session_start": history[0]["timestamp"],
                    "session_end": history[-1]["timestamp"],
                    "message_count": len(history),
                    "messages": history
                }, f, indent=2)
            
            print(f"💾 Chat history saved to: {filename}")
        except Exception as e:
            print(f"❌ Failed to save chat history: {e}")
    
    async def run_interactive(self):
        """Run the interactive terminal interface."""
        print("🎯 AI Personal Agent System - Terminal Interface")
        print("=" * 50)
        
        if not await self.initialize():
            return
        
        while True:
            try:
                print("\n🤖 Agent Selection")
                agent_name = self.select_agent()
                
                if agent_name:
                    await self.chat_with_agent(agent_name)
                else:
                    print("❌ Invalid selection. Please try again.")
                    continue
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
                continue
    
    async def run_single_command(self, agent_name: str, message: str):
        """Run a single command and exit."""
        if not await self.initialize():
            return
        
        agent = self.coordinator.agents.get(agent_name)
        if not agent:
            print(f"❌ Agent '{agent_name}' not found.")
            available = ", ".join(self.coordinator.agents.keys())
            print(f"Available agents: {available}")
            return
        
        print(f"🤖 Using {agent_name.replace('_', ' ').title()}")
        print(f"💬 Message: {message}")
        print("-" * 50)
        
        # Create and process task
        task = AgentTask(
            id=f"cmd_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            agent_name=agent_name,
            task_type="command",
            content=message,
            priority=1
        )
        
        start_time = datetime.now()
        response = await self.coordinator.process_request(message, agent_name, "command")
        end_time = datetime.now()
        
        print(f"🤖 Response: {response}")
        print(f"⏱️  Processing time: {(end_time - start_time).total_seconds():.2f}s")

async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="AI Personal Agent System - Terminal Interface")
    parser.add_argument("--agent", "-a", help="Agent name for single command mode")
    parser.add_argument("--message", "-m", help="Message for single command mode")
    parser.add_argument("--list", "-l", action="store_true", help="List available agents and exit")
    
    args = parser.parse_args()
    
    interface = TerminalInterface()
    
    # List agents mode
    if args.list:
        if await interface.initialize():
            interface.display_agents()
        return
    
    # Single command mode
    if args.agent and args.message:
        await interface.run_single_command(args.agent, args.message)
        return
    
    # Interactive mode
    await interface.run_interactive()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)
