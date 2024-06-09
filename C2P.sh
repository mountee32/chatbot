#!/bin/bash
# ----------------------------
# Run with clear && chmod +x C2P.sh && ./C2P.sh
# ----------------------------
#
# This script combines multiple files into one output file.
# It starts by copying the contents of C2Pinput.txt into C2Poutput.txt as an initial prompt.
# Then it reads the names of files and directories listed in C2Pexclude.txt and excludes them.
# All other files in the current directory and its subdirectories are added to C2Poutput.txt with the full relative path as a subheading.

# Input prompt file and output file
prompt_file="C2Pinput.txt"
output_file="C2Poutput.txt"
exclude_file="C2Pexclude.txt"

# Clear the output file initially and prepend the prompt
cp "$prompt_file" "$output_file"
echo "" >> "$output_file"  # Adds a blank line for better formatting after the prompt

# Read the list of files and directories to exclude into an array
excluded_items=()
while IFS= read -r item; do
    # Remove trailing slash if it exists
    item=${item%/}
    excluded_items+=("$item")
done < "$exclude_file"

# Process all files and directories in the current directory recursively, excluding specified paths
find . -type f -print0 | while IFS= read -r -d '' item; do
    # Remove the leading './' from the item
    item=${item#./}

    # Check if the item should be excluded
    exclude=false
    for excluded_item in "${excluded_items[@]}"; do
        if [[ "$item" == "$excluded_item" || "$item" == "$excluded_item"/* ]]; then
            exclude=true
            break
        fi
    done

    if [[ "$exclude" == true ]]; then
        continue
    fi

    # Skip if the item is the output file itself or the prompt file
    if [[ "$item" != "$output_file" ]] && [[ "$item" != "$prompt_file" ]]; then
        # Print the filename being included
        echo "Including file: $item"

        # Write the full path as a subheading
        echo "## $item" >> "$output_file"
        echo "" >> "$output_file"  # Adds a blank line for better formatting before the file content

        # Append the contents of the file to C2Poutput.txt
        cat "$item" >> "$output_file"

        # Add a newline for separation between file contents
        echo "" >> "$output_file"
    fi
done
