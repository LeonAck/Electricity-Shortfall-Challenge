import os
import sys
import pytest
import joblib

# Debug: Let's see what's actually available
print(f"=== FULL DEBUG INFO ===")
print(f"Current file: {__file__}")
print(f"Absolute path: {os.path.abspath(__file__)}")
print(f"Current working directory: {os.getcwd()}")

# Check multiple directories
directories_to_check = [
    os.path.dirname(os.path.abspath(__file__)),  # tests directory
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),  # parent
    os.getcwd(),  # current working dir
    "/home/runner/work/Electricity-Shortfall-Challenge",  # potential root
    "/home/runner/work/Electricity-Shortfall-Challenge/Electricity-Shortfall-Challenge",  # double nested
]

for i, directory in enumerate(directories_to_check):
    print(f"\n--- Directory {i+1}: {directory} ---")
    if os.path.exists(directory):
        try:
            contents = os.listdir(directory)
            print(f"Contents: {contents}")
            if 'configs' in contents:
                print(f"✓ FOUND configs directory here!")
            else:
                print("✗ No configs directory here")
        except PermissionError:
            print("Permission denied")
    else:
        print("Directory does not exist")

print(f"========================")

# Temporarily use a simple fallback to let the script continue
project_root = os.getcwd()  # Just use current directory for now
sys.path.insert(0, project_root)

# Keep the imports but don't run tests yet
try:
    from scripts.data_loading import load_data
    print("✓ Successfully imported scripts.data_loading")
except ImportError as e:
    print(f"✗ Failed to import scripts.data_loading: {e}")

# Simple test to see if this works
def test_debug():
    assert True  # This will always pass, just to see the debug output