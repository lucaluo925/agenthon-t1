# Agenthon 2026 Track 1 submission image. linux/amd64 by contract.
#
# FROM the shared sandbox base, as baselines/README.md requires. That base matters for a
# specific reason: the model's generated script runs INSIDE THIS IMAGE, and every unit's
# instruction.md promises it numpy/pandas/scipy/pyarrow/numba (and the wider list in
# docker/requirements-sandbox.txt: statsmodels, scikit-learn, arch, polars, ta-lib, ...).
# If this image's package set differs from that promise, the model writes correct code that
# fails on an import. So the base is rebuilt in CI from the public kit rather than approximated.
#
# The verifier does NOT run in this image -- the harness runs the organizer's own
# finance-bench-sandbox against /input/checks after our container exits.
FROM finance-bench-sandbox:latest

# The only addition the base does not provide: an OpenAI-compatible client for the House route
# ($MODEL_ENDPOINT/v1 through the audited proxy). Vendor SDKs are useless here -- their
# endpoints are refused by the proxy and no vendor keys exist.
RUN pip install --no-cache-dir "openai==1.40.0"

COPY agent.py /agent/agent.py

LABEL qfbench2.interface_version="2.0"
LABEL org.opencontainers.image.title="agenthon-t1-coding-agent"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0

RUN python -c "import numpy, pandas, scipy, pyarrow, openai" \
 && python /agent/agent.py --help > /dev/null 2>&1 || true

# The harness runs: <IMAGE> solve --task-dir /input --out /app/output
# ENTRYPOINT takes the verb and flags as appended arguments, which is the kit's own sketch.
ENTRYPOINT ["python", "/agent/agent.py"]
CMD ["solve", "--task-dir", "/input", "--out", "/app/output"]
