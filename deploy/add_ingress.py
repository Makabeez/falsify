"""Add falsify.baserep.xyz -> localhost:8097 to a cloudflared config, before the catch-all rule.

    python3 deploy/add_ingress.py /home/vps/.cloudflared/baserep-config.yml [--port 8097]

Idempotent: does nothing if the hostname is already there. Writes a .bak first.
"""
import shutil
import sys

HOST = "falsify.baserep.xyz"
args = sys.argv[1:]
PORT = int(args[args.index("--port") + 1]) if "--port" in args else 8097
path = args[0]
text = open(path).read()
if HOST in text:
    print(f"{HOST} already in {path}; nothing to do")
    sys.exit(0)
lines = text.splitlines(keepends=True)
idx = next((i for i, l in enumerate(lines) if l.strip().startswith("- service:") and "http_status" in l), None)
if idx is None:
    sys.exit("no catch-all '- service: http_status:404' rule found; add the ingress by hand")
indent = lines[idx][: len(lines[idx]) - len(lines[idx].lstrip())]
shutil.copy(path, path + ".bak")
lines[idx:idx] = [f"{indent}- hostname: {HOST}\n", f"{indent}  service: http://localhost:{PORT}\n"]
open(path, "w").write("".join(lines))
print(f"added {HOST} -> localhost:{PORT} in {path} (backup: {path}.bak)")
