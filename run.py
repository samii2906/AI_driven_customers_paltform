#!/usr/bin/env python3
"""
PulseIQ — AI-Driven Customer Analytics Platform
================================================
Run this file to start the application.
"""

import subprocess
import sys
import os

def check_install(pkg):
    try:
        __import__(pkg)
        return True
    except ImportError:
        return False

required = {'flask':'flask','sklearn':'scikit-learn','pandas':'pandas','numpy':'numpy'}
missing = [pip for mod,pip in required.items() if not check_install(mod)]
if missing:
    print(f"Installing: {', '.join(missing)}")
    subprocess.check_call([sys.executable,'-m','pip','install']+missing)

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Generate dataset if not present
if not os.path.exists('customers.csv'):
    print("Generating dataset (350 customers)...")
    exec(open('generate_data.py').read())

from app import load_and_train, app
print("\n" + "="*50)
print("  PulseIQ Customer Analytics Platform")
print("="*50)
load_and_train()
print("\n  Open http://127.0.0.1:5000 in your browser")
print("="*50 + "\n")
app.run(debug=False, port=5000, host='0.0.0.0')
