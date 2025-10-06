#!/bin/bash

# Usage: ./rename_folders.sh /path/to/parent

PARENT_DIR="${1:-.}"   # Default to current directory if no arg given

for dir in "$PARENT_DIR"/*; do
    if [ -d "$dir" ] && [[ "$dir" == *.narc_output ]]; then
        new_name="${dir%.narc_output}"
        echo "Renaming: $dir -> $new_name"
        mv "$dir" "$new_name"
    fi
done