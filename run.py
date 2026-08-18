#!/usr/bin/env python3
"""
Root entry point alias calling parmanu/run.py for Team Parmanu
"""
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
team_run_py = os.path.join(script_dir, "parmanu", "run.py")

if __name__ == "__main__":
    import subprocess
    cmd = [sys.executable, team_run_py] + sys.argv[1:]
    sys.exit(subprocess.call(cmd))
