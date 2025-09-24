## Tsukiyomi Vision

Author: anudit

Tsukiyomi Vision provides local image and video inference utilities built on top of the moondream-station toolkit. It supports:

- Batch processing of image directories
- Real-time processing from cameras and video files
- Multiple inference functions: caption, query, detect, point

### Repository Structure

- `tools/batch_inference.py` — Batch image processing CLI
- `tools/realtime_inference.py` — Real-time camera/video processing CLI
- `examples/` — Example scripts for batch and real-time usage
- `setup/moondream-station/` — Local moondream-station sources used by the CLIs
- `requirements.txt` — Python dependencies

### Prerequisites

- Python 3.10+
- A virtual environment is recommended

### Setup

1. Create and activate a virtual environment
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Install moondream-station (choose one):

   From PyPI:
   ```bash
   pip install moondream-station
   ```

   From source:
   ```bash
   git clone https://github.com/m87-labs/moondream-station.git
   cd moondream-station
   pip install -e .
   ```

   Or use the provided installer after activating venv:
   ```bash
   bash setup/install_moondream_station.sh
   ```

### Usage: Batch Image Processing

Run captions on an image directory and save results as JSON:
```bash
python3 tools/batch_inference.py /path/to/images -f caption -l short -o results.json
```

Other functions:
```bash
# Query
python3 tools/batch_inference.py /path/to/images -f query -q "What is shown?" -o answers.json

# Detect
python3 tools/batch_inference.py /path/to/images -f detect -obj "car" -o detections.json

# Point
python3 tools/batch_inference.py /path/to/images -f point -obj "person" -o points.json
```

### Usage: Real-time Video/Camera Processing

Default camera captioning and optional JSON output:
```bash
python3 tools/realtime_inference.py --fps 1 --save-results -o live_results.json
```

Video file processing:
```bash
python3 tools/realtime_inference.py /path/to/video.mp4 -f caption --fps 2 --save-results -o video_results.json
```

Headless mode (no window):
```bash
python3 tools/realtime_inference.py --no-display --save-results -o results.json
```

### Examples

See `examples/` for scripted examples of batch and real-time operations.

### Notes

- Default model: `moondream-3-preview`. You can override with `-m`.
- The scripts will attempt to load the production manifest first, and fall back to the local manifest.
- Results are saved as structured JSON with timestamps and success flags.
- To launch the moondream-station local server manually, run:
  ```bash
  moondream-station
  ```

### License and Credits

This repository integrates and bundles components from the moondream-station project. Please refer to the moondream-station license files inside `setup/moondream-station/`.

Credits:
- Moondream and the moondream-station team for the local inference toolkit.

Unless otherwise noted, code in this repository is provided under the MIT License. See `LICENSE` for details.