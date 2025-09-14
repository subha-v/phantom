#!/usr/bin/env python3
"""
Helper script to automatically add VS Code tasks for new Arduino sketches
Usage: python3 add_sketch_tasks.py <sketch_name>
"""

import json
import sys
import os

def add_sketch_tasks(sketch_name):
    tasks_file = ".vscode/tasks.json"
    
    # Load existing tasks
    with open(tasks_file, 'r') as f:
        tasks_config = json.load(f)
    
    # Create new tasks for the sketch
    compile_task = {
        "label": f"Arduino: Compile {sketch_name.title()}",
        "type": "shell",
        "command": "arduino-cli",
        "args": [
            "compile",
            "--fqbn",
            "arduino:renesas_uno:unor4wifi",
            f"${{workspaceFolder}}/hardware/{sketch_name}"
        ],
        "group": "build",
        "presentation": {
            "echo": True,
            "reveal": "always",
            "focus": False,
            "panel": "shared",
            "showReuseMessage": True,
            "clear": False
        },
        "problemMatcher": []
    }
    
    upload_task = {
        "label": f"Arduino: Upload {sketch_name.title()}",
        "type": "shell",
        "command": "arduino-cli",
        "args": [
            "upload",
            "--fqbn",
            "arduino:renesas_uno:unor4wifi",
            "--port",
            "/dev/cu.usbmodem34B7DA631B182",
            f"${{workspaceFolder}}/hardware/{sketch_name}"
        ],
        "group": "build",
        "presentation": {
            "echo": True,
            "reveal": "always",
            "focus": False,
            "panel": "shared",
            "showReuseMessage": True,
            "clear": False
        },
        "problemMatcher": [],
        "dependsOn": f"Arduino: Compile {sketch_name.title()}"
    }
    
    # Add tasks to the configuration
    tasks_config["tasks"].extend([compile_task, upload_task])
    
    # Write back to file
    with open(tasks_file, 'w') as f:
        json.dump(tasks_config, f, indent=4)
    
    print(f"✅ Added tasks for '{sketch_name}':")
    print(f"   - Arduino: Compile {sketch_name.title()}")
    print(f"   - Arduino: Upload {sketch_name.title()}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 add_sketch_tasks.py <sketch_name>")
        sys.exit(1)
    
    sketch_name = sys.argv[1]
    add_sketch_tasks(sketch_name)
