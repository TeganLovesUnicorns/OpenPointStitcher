# OpenPointStitcher
#
# One container carrying the whole tool: Python, Open3D, the pipeline and the placement bench. Every package is
# pinned (requirements-lock.txt) so a rebuild gives the same results.
FROM python:3.11.15-slim-bookworm

# Open3D links against these even when nothing is being drawn. libegl1 is only
# needed if you ever ask it to render offscreen, which the pipeline does not,
# but it is small and it keeps that door open.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libgomp1 \
        libegl1 \
        libglib2.0-0 \
        gosu \
    && rm -rf /var/lib/apt/lists/*

# Not root: a root process writing into a bind mount leaves root-owned files in the user's folder on Linux.
# UID 1000 matches the usual first Linux user; Windows ignores ownership.
RUN useradd --create-home --uid 1000 stitch

WORKDIR /app

COPY requirements-lock.txt ./
RUN pip install --no-cache-dir -r requirements-lock.txt

COPY open_point_stitch/ ./open_point_stitch/
COPY tools/ ./tools/
COPY tests/ ./tests/
COPY placement_bench.html splash.png ./
COPY README.md CHANGELOG.md FIELD_GUIDE.md FAILURE_MODES.md LICENSE THIRD_PARTY_LICENSES.md ./
COPY docker-entrypoint.sh /usr/local/bin/
# A Windows checkout or editor can give the script CRLF endings, which breaks its first line.
RUN sed -i 's/\r$//' /usr/local/bin/docker-entrypoint.sh

# BUILD_TIME comes from outside (the launchers pass --build-arg BUILD_TIME=<UTC timestamp>) so it changes on every
# build; a RUN date step would be cached.
ARG BUILD_TIME=unknown
ENV OPENPOINT_BUILD_TIME=${BUILD_TIME}
RUN echo "${BUILD_TIME}" > /app/BUILD_TIME

# /scans: your scans, read only. /out: results. /scans-regression: the pair the regression test checks.
# /projects: where bench keeps every project, read-write.
RUN mkdir -p /scans /scans-regression /out /projects && chown stitch:stitch /out /projects
VOLUME ["/scans", "/scans-regression", "/out", "/projects"]

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OPENPOINT_TEST_SCANS=/scans-regression

# No USER here: the entrypoint starts as root only to make the mounted folders writable, then drops to the
# unprivileged user with gosu. Pass --user yourself and the entrypoint stands aside.
EXPOSE 8080

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["help"]
