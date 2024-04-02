#!/bin/bash

# Default port values
local_port=2287
remote_port=2287

# Parse command-line options
OPTIONS=$(getopt -o l:r: --long local-port:,remote-port: -- "$@")
eval set -- "$OPTIONS"

while true; do
    case "$1" in
        -l|--local-port)
            local_port=$2
            shift 2
            ;;
        -r|--remote-port)
            remote_port=$2
            shift 2
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Invalid option: $1"
            exit 1
            ;;
    esac
done

# Allocate a compute node with GPU resources using salloc
echo "Allocating a compute node with GPU resources..."
output=$(timeout 10s salloc --partition=single --gres=gpu:1)
echo "$output"

# Extract the assigned compute node from the salloc output
compute_node=$(echo "$output" | grep 'Nodes' | awk '{print $6}')

if [ -n "$compute_node" ]; then
    echo "Assigned compute node: $compute_node"
else
    echo "Failed to allocate a compute node."
    exit 1
fi

# Set up port forwarding using ssh
echo "Setting up port forwarding from local port $local_port to remote port $remote_port on $compute_node..."
ssh -L $local_port:localhost:$remote_port $compute_node &
ssh_pid=$!

# Wait for a few seconds to establish the SSH connection
sleep 5

# Run bash commands on the compute node
echo "Running bash commands on the compute node..."
bash_commands=(
    "echo 'Hello from the compute node!'"
    "ls -l"
)
for command in "${bash_commands[@]}"; do
    ssh $compute_node "$command"
done

# Start an interactive Python session on the compute node
echo "Starting an interactive Python session on the compute node..."
python_startup_commands=(
    "import numpy as np"
    "import torch"
    "print('Python session is ready!')"
)
python_command="python3 -c \"$(printf "%s; " "${python_startup_commands[@]}")import code; code.interact(local=locals())\""
ssh $compute_node "$python_command"

# Clean up the SSH process
kill $ssh_pid
