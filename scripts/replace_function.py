#!/usr/bin/env python3
import sys

# Read the original file
with open('/Users/jing/PycharmProjects/ELVIS/scripts/analysis_results.py', 'r') as f:
    lines = f.readlines()

# Read the new function from temp file
with open('/Users/jing/PycharmProjects/ELVIS/scripts/temp_new_function.py', 'r') as f:
    new_function_lines = f.readlines()

# Find start and end of the function to replace
start_idx = None
end_idx = None

for i, line in enumerate(lines):
    if 'def analysis_all_principles_merged(args):' in line:
        start_idx = i
        print(f"Found function start at line {i}")
    if start_idx is not None and i > start_idx and line.strip().startswith('def main'):
        end_idx = i
        print(f"Found function end at line {i}")
        break

if start_idx is None or end_idx is None:
    print("Could not find function boundaries")
    sys.exit(1)

# Replace the function
new_lines = lines[:start_idx] + new_function_lines + ['\n\n'] + lines[end_idx:]

# Write back
with open('/Users/jing/PycharmProjects/ELVIS/scripts/analysis_results.py', 'w') as f:
    f.writelines(new_lines)

print(f"Successfully replaced function from lines {start_idx} to {end_idx}")
print(f"Old function had {end_idx - start_idx} lines")
print(f"New function has {len(new_function_lines)} lines")

