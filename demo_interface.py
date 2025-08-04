"""Demo script to test the Gradio interface without launching the full app."""

import sys
from pathlib import Path

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent / "src"))

from ui.gradio_interface import create_app

def main():
    """Demo the interface creation."""
    print("🚀 Creating Gradio interface...")
    
    try:
        app = create_app()
        print("✅ Interface created successfully!")
        print(f"📊 Available models: {app.available_models}")
        print(f"🎯 Current model: {app.current_model}")
        
        # Test some basic functionality
        print("\n🧪 Testing basic functionality...")
        
        # Test file upload with empty list
        result = app.upload_files([])
        print(f"📁 Empty upload test: {result}")
        
        # Test chat with empty message
        input_result, history_result = app.handle_chat("", [], app.current_model)
        print(f"💬 Empty chat test: input='{input_result}', history={history_result}")
        
        # Test chat with valid message
        input_result, history_result = app.handle_chat("Hello", [], app.current_model)
        print(f"💬 Valid chat test: input='{input_result}', history_length={len(history_result)}")
        
        # Test model change
        if len(app.available_models) > 0:
            result = app.change_model(app.available_models[0])
            print(f"🔄 Model change test: {result}")
        
        # Test clear chat
        result = app.clear_chat()
        print(f"🧹 Clear chat test: {len(result)} items in history")
        
        print("\n✅ All basic functionality tests passed!")
        print("🎉 Interface is ready to launch!")
        
    except Exception as e:
        print(f"❌ Error creating interface: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())