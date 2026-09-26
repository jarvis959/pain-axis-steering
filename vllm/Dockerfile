# Derived image: pinned vLLM + the pain-axis steering patch.
#
# The base is pinned by digest so the patch context cannot drift. Build from the
# repository root:
#
#   docker build -f vllm/Dockerfile -t vllm-pain:qwen38-27b .
#
FROM vllm/vllm-openai:qwen38-flash-next@sha256:3b0e188ffceb3d07e09c3cb5215433a0020eacf02d7f882ed3a8bfd15454477e

COPY patches/vllm-pain-steering.patch /tmp/vllm-pain-steering.patch

RUN cd /usr/local/lib/python3.12/dist-packages \
    && patch -p1 --dry-run < /tmp/vllm-pain-steering.patch \
    && patch -p1 < /tmp/vllm-pain-steering.patch \
    && rm /tmp/vllm-pain-steering.patch \
    && python3 -c "import vllm.pain_axis, vllm.pain_axis_api; print('pain_axis imports OK')"
