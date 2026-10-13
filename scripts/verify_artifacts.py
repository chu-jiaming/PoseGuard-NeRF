"""Check the release snapshot using only Python's standard library."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    records = json.loads((ROOT / 'artifacts/manifest.json').read_text())['files']
    errors = []
    for row in records:
        path = ROOT / row['archive']
        if not path.is_file() or path.is_symlink():
            errors.append(row['archive'] + ': missing or symlink')
            continue
        data = path.read_bytes()
        if len(data) != row['bytes'] or hashlib.sha256(data).hexdigest() != row['sha256']:
            errors.append(row['archive'] + ': size/hash mismatch')
    if errors:
        raise SystemExit('\n'.join(errors))
    print('Verified %d archived files (%d bytes).' %
          (len(records), sum(row['bytes'] for row in records)))


if __name__ == '__main__':
    main()
