#!/bin/bash
echo "starting..."
cd /home/ilyes/agents
echo "in agents dir"
/home/ilyes/.venvs/agents/bin/python main.py
echo "python exited with code $?"
