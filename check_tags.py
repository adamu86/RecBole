import os
import re
import glob

UUID_PATTERN = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$')

for item_path in sorted(glob.glob("dataset/**/*.item", recursive=True)):
    alias = os.path.basename(os.path.dirname(item_path))
    total = 0
    uuid_count = 0

    with open(item_path, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            total += 1
            tags = parts[1].strip()
            if UUID_PATTERN.match(tags):
                uuid_count += 1

    pct = (uuid_count / total * 100) if total else 0
    print(f"{alias}: {uuid_count}/{total} UUID ({pct:.1f}%)")
