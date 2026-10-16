# Third-party components

OpenPointStitcher is built on the open-source components below. Their licenses are permissive (MIT, BSD, Apache 2.0,
ISC, PSF, MPL 2.0) unless noted. Versions are the ones pinned in `requirements-lock.txt` and `ui-src/package-lock.json`.
None of them is GPL-2.0-only or otherwise incompatible with OpenPointStitcher's AGPL v3 license. This list is a
guide, not legal advice; each component's own license text, shipped with the component, is what
applies.

## Python packages (in the Docker image)

| Component | License |
|---|---|
| Open3D 0.19.0 | MIT |
| NumPy 2.4.4 | BSD-3-Clause (bundles code under 0BSD, MIT, Zlib, CC0-1.0) |
| SciPy 1.17.1 | BSD-3-Clause |
| opencv-python-headless 5.0.0.93 | Apache 2.0 (see bundled native libraries below) |
| scikit-learn, pandas, matplotlib, Pillow, Flask, Werkzeug, Jinja2, click, psutil, joblib | BSD-style, PSF (matplotlib) or MIT-CMU (Pillow) |
| plotly, dash, pydantic, PyYAML, urllib3, attrs, jsonschema and the other small dependencies | MIT, BSD, ISC or Apache 2.0 |
| certifi, tqdm | MPL 2.0 (tqdm: MPL 2.0 and MIT) |

`pip freeze` inside the image lists every package; each package's own license file is in its `.dist-info` folder.

## Native libraries bundled inside Python wheels

| Component | Comes with | License |
|---|---|---|
| FFmpeg (libavcodec, libavformat, libavutil, libswscale, libswresample) | opencv-python-headless | LGPL 2.1 or later |
| libvpx, libaom, libavif, libpng, OpenBLAS | opencv-python-headless | BSD-style |
| OpenSSL 1.1.1 | opencv-python-headless | OpenSSL / SSLeay |
| libgfortran, libquadmath | opencv-python-headless | GPL 3 with the GCC Runtime Library Exception |
| TBB, libc++ | open3d | Apache 2.0 |
| NVIDIA CUDA runtime | open3d (CUDA build) | NVIDIA CUDA Toolkit EULA (redistributable components) |

The full third-party license texts for OpenCV's bundled libraries are in the image at
`/usr/local/lib/python3.11/site-packages/cv2/LICENSE-3RD-PARTY.txt`.

**If you distribute the Docker image itself** (rather than this source code), the LGPL components above are shipped
as separate shared libraries, which is the arrangement the LGPL permits; include their license texts and make their
source available on request. Publishing this repository's source does not redistribute any of them.

## Browser bench (bundled into `placement_bench.html`)

| Component | License |
|---|---|
| React 18.3.1, React DOM 18.3.1 | MIT |
| Mantine 8.3 (@mantine/core, @mantine/hooks) and their dependencies | MIT, ISC, BSD or Apache 2.0 |
| Feather icons (paths used inline) | MIT |

Build-only tools (Vite and its plugins, caniuse-lite) are not shipped in the page.

## Base image

`python:3.11.15-slim-bookworm` (Debian 12), whose packages carry their usual Debian licenses.
