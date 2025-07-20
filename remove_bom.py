# remove_bom.py
"""
Removes BOM (Byte Order Mark) from requirements.txt and saves as plain UTF-8.
Run this script once in your project root if you see pip errors about BOM.
"""

input_file = 'requirements.txt'

with open(input_file, 'r', encoding='utf-8-sig') as f:
    content = f.read()

with open(input_file, 'w', encoding='utf-8') as f:
    f.write(content)

print('BOM removed from requirements.txt (if present).') 