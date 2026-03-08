import sys

lines = open(r'..\test_tuning_results.txt', 'r', encoding='utf-16le').read().splitlines()

failed = []
success = 0
for line in lines:
    if "SUCCESS" in line:
        success += 1
    if line.startswith("- "):
        failed.append(line)

with open(r'build_report.txt', 'w', encoding='utf-8') as f:
    f.write(f"Success Count: {success}\n")
    f.write("Failed Tuners:\n")
    for failure in failed:
         f.write(failure + "\n")
