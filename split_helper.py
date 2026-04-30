import re

with open("epls_kaggle.py", "r") as f:
    lines = f.readlines()

new_lines = []

in_step1 = False
in_step2 = False
in_step3 = False

# Insert RUN_PHASE config at line 9 (after imports)
for i, line in enumerate(lines):
    if i == 14: # Before numpy patch
        new_lines.append("# ==============================================================================\n")
        new_lines.append("# KAGGLE EXECUTION CONTROL\n")
        new_lines.append("# Change RUN_PHASE to \"1\", \"2\", \"3\", or \"all\" to run specific parts.\n")
        new_lines.append("RUN_PHASE = \"all\"\n")
        new_lines.append("# ==============================================================================\n\n")
    
    if "# STEP 1 —" in line:
        new_lines.append(f"if RUN_PHASE in ['all', '1']:\n")
        in_step1 = True
        
    if "# STEP 2 —" in line:
        in_step1 = False
        new_lines.append(f"if RUN_PHASE in ['all', '2']:\n")
        in_step2 = True
        
    if "# STEP 3 —" in line:
        in_step2 = False
        new_lines.append(f"if RUN_PHASE in ['all', '3']:\n")
        in_step3 = True

    if in_step1 or in_step2 or in_step3:
        new_lines.append("    " + line if line.strip() else line)
    else:
        new_lines.append(line)

with open("epls_kaggle.py", "w") as f:
    f.writelines(new_lines)

print("Done wrapping phases!")
