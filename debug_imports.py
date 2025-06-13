#!/usr/bin/env python3
"""
Debug script to check which modules are available and their contents
"""

import os
import sys
import importlib
import inspect

# Add src to path
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

print("Python Path:")
for path in sys.path:
    print(f"  {path}")
print()

def check_module(module_path, class_names=None):
    """Check if a module exists and what it contains"""
    print(f"\nChecking module: {module_path}")
    print("-" * 50)
    
    try:
        # Check if the file exists
        parts = module_path.split('.')
        file_path = os.path.join(src_path, *parts) + '.py'
        
        if os.path.exists(file_path):
            print(f"✓ File exists: {file_path}")
            
            # Try to import the module
            try:
                module = importlib.import_module(module_path)
                print(f"✓ Module imported successfully")
                
                # List all classes in the module
                classes = inspect.getmembers(module, inspect.isclass)
                if classes:
                    print(f"\nClasses found in {module_path}:")
                    for name, cls in classes:
                        if cls.__module__ == module_path:  # Only show classes defined in this module
                            print(f"  - {name}")
                            
                            # Show __init__ signature
                            try:
                                sig = inspect.signature(cls.__init__)
                                print(f"    {sig}")
                            except:
                                pass
                
                # List all functions
                functions = inspect.getmembers(module, inspect.isfunction)
                if functions:
                    print(f"\nFunctions found in {module_path}:")
                    for name, func in functions:
                        if func.__module__ == module_path:
                            print(f"  - {name}")
                
                # Check for specific classes if provided
                if class_names:
                    print(f"\nChecking for specific classes:")
                    for class_name in class_names:
                        if hasattr(module, class_name):
                            print(f"  ✓ {class_name} found")
                        else:
                            print(f"  ✗ {class_name} NOT found")
                            # Suggest similar names
                            similar = [name for name, _ in classes if class_name.lower() in name.lower()]
                            if similar:
                                print(f"    Did you mean: {', '.join(similar)}?")
                
            except ImportError as e:
                print(f"✗ Failed to import module: {e}")
                
                # Try to read the file and look for syntax errors
                print(f"\nReading file content to check for issues...")
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                        # Check for common issues
                        lines = content.split('\n')
                        for i, line in enumerate(lines[:20], 1):
                            if 'class ' in line and not line.strip().startswith('#'):
                                print(f"  Line {i}: {line.strip()}")
                except Exception as e:
                    print(f"  Could not read file: {e}")
        else:
            print(f"✗ File does not exist: {file_path}")
            
            # Check if directory exists
            dir_path = os.path.join(src_path, *parts[:-1])
            if os.path.exists(dir_path):
                print(f"\nDirectory exists: {dir_path}")
                print("Files in directory:")
                for file in os.listdir(dir_path):
                    print(f"  - {file}")
            else:
                print(f"✗ Directory does not exist: {dir_path}")
                
    except Exception as e:
        print(f"✗ Error checking module: {e}")

# Check each module
modules_to_check = [
    ('data.fetcher', ['DataFetcher', 'Fetcher']),
    ('optimization.mean_variance', ['MeanVarianceOptimizer', 'MeanVarianceOptimization']),
    ('backtesting.engine', ['BacktestingEngine', 'BacktestEngine', 'Backtester']),
    ('risk.metrics', ['RiskMetrics', 'RiskCalculator']),
    ('visualization.plots', ['PortfolioVisualizer', 'Visualizer', 'Plotter'])
]

print("="* 70)
print("MODULE AVAILABILITY CHECK")
print("="* 70)

for module_path, possible_classes in modules_to_check:
    check_module(module_path, possible_classes)

# Additional checks
print("\n\n" + "="* 70)
print("DIRECTORY STRUCTURE")
print("="* 70)

def show_tree(path, prefix="", max_depth=3, current_depth=0):
    """Show directory tree"""
    if current_depth >= max_depth:
        return
        
    try:
        items = sorted(os.listdir(path))
        for i, item in enumerate(items):
            if item.startswith('.'):
                continue
                
            item_path = os.path.join(path, item)
            is_last = i == len(items) - 1
            
            print(prefix + ("└── " if is_last else "├── ") + item)
            
            if os.path.isdir(item_path) and item not in ['__pycache__', '.git', 'venv', 'env']:
                extension = "    " if is_last else "│   "
                show_tree(item_path, prefix + extension, max_depth, current_depth + 1)
    except PermissionError:
        pass

print(f"\nProject structure from: {project_root}")
show_tree(project_root)

print("\n" + "="* 70)
print("RECOMMENDATIONS")
print("="* 70)

print("""
Based on the above checks, here are some recommendations:

1. If modules are missing, you need to create them:
   - Each module should be a .py file in the appropriate directory
   - Each directory needs an __init__.py file (can be empty)

2. If class names don't match, either:
   - Rename the classes in your files to match the imports
   - Update the imports in main.py to use the correct class names

3. To create a missing module, you can use the stub implementations from the 
   placeholder classes in the updated main.py

4. Common issues:
   - Missing __init__.py files in directories
   - Typos in class names (BacktestEngine vs BacktestingEngine)
   - Syntax errors in module files
   - Circular imports
""")