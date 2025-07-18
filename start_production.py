import asyncio
import sys
import webbrowser
import time
from pathlib import Path

# Add the project directory to Python path
project_dir = Path(__file__).parent
sys.path.insert(0, str(project_dir))

def show_startup_info():
    """Show startup information"""
    print(" AI Personal Agent System - Production Mode")
    print("=" * 50)
    print(" Web URL: http://localhost:8800")
    print("📱 Mobile-friendly interface")
    print(" 7 AI agents available")
    print("=" * 50)
    print("Starting server...")

async def main():
    """Main launcher function"""
    try:
        show_startup_info()
        
        # Import and start the production web interface
        from production_web import main as web_main
        
        # Try to open browser automatically after a delay
        def open_browser():
            time.sleep(3)  # Give server time to start
            try:
                webbrowser.open('http://localhost:8800')
                print("✅ Browser opened automatically")
            except:
                print("⚠️  Please open http://localhost:8800 in your browser")
        
        # Start browser opening in background
        import threading
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        # Start the web interface
        await web_main()
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting server: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
