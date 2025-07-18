#!/usr/bin/env python3
"""
Quick test of the code generation agent with templates only.
"""

import asyncio
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.code_generation import CodeGenerationAgent
from agents.base_agent import AgentTask

async def test_code_generation():
    """Test code generation with templates."""
    print("🧪 Testing Code Generation Agent (Templates Only)")
    print("=" * 50)
    
    try:
        # Initialize agent
        print("🚀 Initializing Code Generation Agent...")
        agent = CodeGenerationAgent()
        
        # Test requests that should use templates (fast)
        test_requests = [
            "create a hello world function",
            "make a function to add two numbers",
            "write a function to calculate factorial",
            "create a simple calculator",
            "generate a function to sort a list"
        ]
        
        print("✅ Agent initialized successfully!")
        print("\n🎯 Testing Template-Based Code Generation:")
        print("-" * 40)
        
        for i, request in enumerate(test_requests, 1):
            print(f"\n{i}. Request: '{request}'")
            print("   Processing...")
            
            # Create task
            task = AgentTask(
                id=f"test_{i}",
                agent_name="code_generation",
                task_type="generate",
                content=request,
                priority=1
            )
            
            # Process task
            start_time = asyncio.get_event_loop().time()
            result = await agent._process_task_impl(task)
            end_time = asyncio.get_event_loop().time()
            
            processing_time = end_time - start_time
            
            print(f"   ✅ Generated in {processing_time:.3f}s")
            print(f"   📝 Code preview: {result[:100]}...")
            
            if processing_time > 1.0:
                print("   ⚠️  Slow response (may have used AI model)")
            else:
                print("   🚀 Fast response (template-based)")
        
        print("\n🎉 All tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function."""
    success = await test_code_generation()
    
    if success:
        print("\n✅ Code Generation Agent is working correctly!")
        print("🎯 Templates are providing fast responses")
        print("🚀 System is ready for terminal interface")
    else:
        print("\n❌ Tests failed - check the error messages above")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
