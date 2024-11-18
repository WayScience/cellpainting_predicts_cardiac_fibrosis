#!/bin/bash

# Get the directory where this script is located
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Loop through each subdirectory
find "$script_dir" -type d | while read -r dir; do
    # Check if there are any bash scripts in the directory
    for script in "$dir"/*.sh; do
        # Only run if the script actually exists
        if [[ -f "$script" ]]; then
            echo "Running script: $script in directory: $dir"
            # Run the script
            bash "$script"
        fi
    done
done
