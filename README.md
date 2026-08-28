# SimpleClick REST API for Vast.ai

This service exposes the SimpleClick interactive image segmentation model as a
FastAPI application. It accepts an image as base64 plus foreground and optional
background pixel points, and returns a binary PNG mask as base64.

## Run with Bash

```bash
cp .env.example .env
# Edit .env and set SIMPLECLICK_API_KEY before public exposure.
set -a; source .env; set +a
bash deploy/setup-vastai.sh
bash scripts/start-simpleclick.sh
```

`setup-vastai.sh` installs Git when needed, creates a virtual environment using
the host's PyTorch installation, and installs this service. The launcher clones
SimpleClick and downloads the checkpoint to `SIMPLECLICK_ROOT` on first startup.
Keep that directory on persistent Vast.ai storage so restarts reuse the model
files. Model loading can take several minutes.

Leave `bash scripts/start-simpleclick.sh` running in a tmux panel. It stays in
the foreground and forwards the Uvicorn process logs to the terminal.

The service is available at `http://HOST:8000`:

- Swagger UI: `/docs`
- Liveness: `GET /health/live`
- Readiness: `GET /health/ready`
- Segmentation: `POST /v1/segment`

Readiness returns `503` until the checkpoint and model are loaded. Keep one
Uvicorn worker per GPU because every worker loads its own model copy.

## Segmentation request

```bash
IMAGE_B64="$(base64 -w 0 ./image.jpg)"
curl -X POST http://localhost:8000/v1/segment \
  -H 'Content-Type: application/json' \
  -H "X-API-Key: ${SIMPLECLICK_API_KEY}" \
  -d "{\"image\":\"${IMAGE_B64}\",\"positive_points\":[[320,240]],\"negative_points\":[],\"threshold\":0.49}"
```

Points are `[x, y]` pixel coordinates from the top-left corner. At least one
positive point is required. Long strokes are sampled to a bounded number of
points before inference. The response contains `mask`, `mask_format`, and
`mask_shape` (`[height, width]`) along with the points actually used.

Errors use one stable shape:

```json
{
  "error": {
    "code": "point_out_of_bounds",
    "message": "positive_points[0] is outside the image bounds"
  },
  "request_id": "..."
}
```

## Vast.ai notes

Use a CUDA-enabled Vast.ai instance and publish TCP port `8000`.
Attach persistent storage and set `SIMPLECLICK_ROOT` there if you want to retain
the SimpleClick checkout and checkpoint between instance recreations. Configure
the public port mapping in Vast.ai to port `8000`, and set
`SIMPLECLICK_API_KEY` in `.env` or the instance environment. Do not put
credentials in this repository.

For a private deployment, the API key may be left empty, but network access
should then be restricted at the Vast.ai or reverse-proxy layer.

## Configuration

Settings are environment variables with the `SIMPLECLICK_` prefix. Relevant
options include `API_KEY`, `MODEL_DEVICE` (`auto`, `cuda`, or `cpu`),
`SIMPLECLICK_ROOT`, `CHECKPOINT_PATH`, `CHECKPOINT_ID`, `MAX_IMAGE_BYTES`,
`MAX_IMAGE_PIXELS`, `MAX_POINTS_USED`, `DEFAULT_THRESHOLD`, `CORS_ORIGINS`
(comma-separated), `ENABLE_DOCS`, and `WORKERS`.
