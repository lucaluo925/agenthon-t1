"""Run the actual image offline under published Development restrictions before pushing."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

image = sys.argv[1]
expected = hashlib.sha256(Path("agent.py").read_bytes()).hexdigest()
config = json.loads(subprocess.check_output(["docker", "image", "inspect", image], text=True))[0]
assert config["Architecture"] == "amd64" and config["Os"] == "linux"
c = config["Config"]
assert not c.get("Volumes")
assert c["Labels"]["qfbench2.interface_version"] == "2.0"
assert c["Entrypoint"] == ["python", "/agent/agent.py"]
assert c["Cmd"] == ["solve", "--task-dir", "/input", "--out", "/app/output"]

# These are launcher restrictions, not a measurement of the House service or 128-GiB workloads.
restrictions = ["--network=none", "--read-only", "--user", "65534:65534", "--cap-drop=ALL",
                "--security-opt", "no-new-privileges", "--tmpfs", "/tmp:rw,noexec,nosuid,nodev,size=64m",
                "--pids-limit", "256", "--ulimit", "nofile=1024:1024", "--ulimit", "nproc=256:256",
                "--ulimit", "fsize=67108864:67108864", "--cpus", "16", "--memory", "128g",
                "--memory-swap", "128g", "-e", "HOME=/nonexistent"]
probe = (
    "import hashlib,httpx,openai,numpy,pandas,scipy,pyarrow;from openai import OpenAI;"
    f"assert hashlib.sha256(open('/agent/agent.py','rb').read()).hexdigest()=={expected!r};"
    "assert openai.__version__=='1.40.0' and httpx.__version__=='0.27.2';"
    "OpenAI(base_url='http://127.0.0.1:1/v1',api_key='unused',max_retries=0);"
    "print('actual image source and House client construction PASS')"
)
subprocess.run(["docker", "run", "--rm", *restrictions, "--entrypoint", "python", image, "-c", probe],
               check=True, timeout=90)

cases = []
with tempfile.TemporaryDirectory(prefix="t1-image-check-") as temporary:
    root = Path(temporary)
    # Every parent must be traversable by the actual launcher UID and the independent reader.
    root.chmod(0o755)
    task, fixture = root / "input", root / "fixtures"
    task.mkdir(mode=0o755)
    fixture.mkdir(mode=0o755)
    (task / "instruction.md").write_text("Write results.csv containing one row x=1.\n")
    replies = {
        "normal": "```python\nfrom pathlib import Path\nPath('results.csv').write_text('x\\n1\\n')\n```",
        "no_code": "No usable response",
    }
    for name, reply in replies.items():
        out = root / name
        out.mkdir()
        out.chmod(0o777)
        (fixture / "reply.txt").write_text(reply)
        command = ["docker", "run", "--rm", *restrictions,
                   "-v", f"{task}:/input:ro", "-v", f"{fixture}:/fixtures:ro",
                   "-v", f"{out}:/output", "-v", f"{out}:/app/output",
                   "-e", "AGENT_STUB_FILE=/fixtures/reply.txt", "-e", "QFBENCH_NETWORK=none",
                   image, "solve", "--task-dir", "/input", "--out", "/app/output"]
        result = subprocess.run(command, text=True, capture_output=True, timeout=90)
        if result.returncode:
            print(result.stdout, result.stderr)
        assert result.returncode == 0
        files = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())
        if name == "normal":
            assert files == ["results.csv"]
            assert (out / "results.csv").read_bytes() == b"x\n1\n"
            assert "6 requests used" in result.stderr
            # An unrelated non-root identity must also be able to read the output.
            reader = ["docker", "run", "--rm", "--network=none", "--read-only", "--user", "1000:1000",
                      "--cap-drop=ALL", "--security-opt", "no-new-privileges", "-v", f"{out}:/output:ro",
                      "--entrypoint", "python", image, "-c",
                      "assert open('/output/results.csv','rb').read()==b'x\\n1\\n'"]
            subprocess.run(reader, check=True, timeout=90)
        else:
            assert files == []
            assert "request cap reached (8)" in result.stderr
        cases.append({"case": name, "exit_code": result.returncode, "files": files})

receipt = {"image": image, "source_sha256": expected, "config_gate": "PASS", "runtime_gate": "PASS",
           "launcher_uid": 65534, "independent_output_reader_uid": 1000, "cases": cases,
           "network": "none", "real_house_calls": 0,
           "limitations": "Local fixture and client construction; not House quality or competition score."}
Path("ci-image-acceptance.json").write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt), flush=True)
