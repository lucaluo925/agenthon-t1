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
#
# httpx is pinned deliberately; do not drop the pin. openai 1.40.0 builds its HTTP client with
# `proxies=proxies` written unconditionally into the call, and httpx 0.28 removed that
# parameter, so openai 1.40.0 + httpx 0.28.x raises
#     TypeError: Client.__init__() got an unexpected keyword argument 'proxies'
# at OpenAI(...) construction -- before any request, and whether or not HTTP_PROXY is set.
# The image built 2026-10-06 (sha256:90e59631...) shipped exactly that pair: the agent would
# have crashed on every unit, written nothing, failed g1_schema and scored 0.0000, and it
# would never have made the House call rule 9 requires. Confirmed by reading that image's pip
# layer straight from the registry (httpx-0.28.1.dist-info beside openai-1.40.0.dist-info) and
# reproduced locally. Left unpinned, pip resolves to 0.28.x again.
RUN pip install --no-cache-dir "openai==1.40.0" "httpx==0.27.2"

COPY agent.py /agent/agent.py

LABEL qfbench2.interface_version="2.0"
LABEL org.opencontainers.image.title="agenthon-t1-coding-agent"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0

# This check must CONSTRUCT the client, not merely import openai: `import openai` succeeds
# even against an incompatible httpx, which is exactly why the previous import-only check
# passed on an image that could never reach the model. Deliberately no `|| true` -- an image
# that cannot build a client must fail the build rather than ship.
RUN python -c "import httpx, openai; from openai import OpenAI; \
OpenAI(base_url='http://127.0.0.1:1/v1', api_key='unused', max_retries=0); \
print('house client constructs: openai', openai.__version__, '/ httpx', httpx.__version__)"

RUN python -c "import numpy, pandas, scipy, pyarrow, openai" \
 && python /agent/agent.py --help > /dev/null 2>&1 || true

# The harness runs: <IMAGE> solve --task-dir /input --out /app/output
# ENTRYPOINT takes the verb and flags as appended arguments, which is the kit's own sketch.
ENTRYPOINT ["python", "/agent/agent.py"]
CMD ["solve", "--task-dir", "/input", "--out", "/app/output"]
