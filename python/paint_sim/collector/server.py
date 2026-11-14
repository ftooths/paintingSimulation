"""FastAPI server + WebSocket live progress for the Multi-Site Paint Sample Collector (Sections 1.2, 5.2, 5.3)."""
from __future__ import annotations

import argparse
import asyncio
import threading
import webbrowser
from pathlib import Path
from typing import Any, Dict, List, Optional

# Official FastAPI & WebSocket docs:
#   - https://fastapi.tiangolo.com/reference/websockets/
#   - https://fastapi.tiangolo.com/advanced/testing-websockets/
from fastapi import Body, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, Response

from paint_sim.collector.jobs import CollectorJobManager
from paint_sim.collector.sources import (
    SourceRegistry,
    load_collector_settings,
    save_collector_settings,
)
from paint_sim.collector.vision.gemini_detect import (
    get_public_secrets_status,
    list_available_vision_models,
    redact_secrets,
    save_secrets,
    test_genai_connection,
)
from paint_sim.db.sample_ingestor import (
    VALID_BRAND_IDS,
    VALID_PAINT_ROLES,
    VALID_SOLVENT_TYPES,
    STANDARD_SAMPLE_DIRECTORY_TREE,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
FRONTEND_DIR = _REPO_ROOT / "frontend"


def create_collector_app(
    job_manager: Optional[CollectorJobManager] = None,
    source_registry: Optional[SourceRegistry] = None,
    settings_path: Optional[Path] = None,
    secrets_path: Optional[Path] = None,
) -> FastAPI:
    """Create the FastAPI application serving the Sample Collector UI, REST API, and WebSockets."""
    mgr = job_manager or CollectorJobManager(secrets_path=secrets_path)
    registry = source_registry or SourceRegistry()
    app = FastAPI(title="Multi-Site Paint Sample Collector", version="1.0.0")

    # Event-driven WebSocket subscribers `(loop, asyncio.Queue)` keyed by job_id (zero idle polling)
    ws_subscribers: Dict[str, List[tuple[asyncio.AbstractEventLoop, asyncio.Queue]]] = {}
    ws_lock = threading.Lock()

    def on_job_event(job_id: str, event: Dict[str, Any]) -> None:
        with ws_lock:
            subs = list(ws_subscribers.get(job_id, [])) + list(ws_subscribers.get("*", []))
        for loop, aq in subs:
            try:
                loop.call_soon_threadsafe(aq.put_nowait, event)
            except Exception:
                pass

    mgr.add_event_listener(on_job_event)

    # ------------------------------------------------------------------------
    # Static UI Routes
    # ------------------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    def serve_index() -> HTMLResponse:
        html_path = FRONTEND_DIR / "sample_collector.html"
        if not html_path.is_file():
            return HTMLResponse("<h1>sample_collector.html not found</h1>", status_code=404)
        return HTMLResponse(html_path.read_text(encoding="utf-8"))

    @app.get("/src/collector/{filename}")
    def serve_collector_js(filename: str) -> Response:
        safe_name = Path(filename).name
        js_path = FRONTEND_DIR / "src" / "collector" / safe_name
        if not js_path.is_file():
            raise HTTPException(status_code=404, detail=f"Asset not found: {safe_name}")
        return FileResponse(js_path, media_type="application/javascript")

    # ------------------------------------------------------------------------
    # Source Registry & Taxonomy Constants
    # ------------------------------------------------------------------------

    @app.get("/api/taxonomy")
    def api_get_taxonomy() -> Dict[str, Any]:
        return {
            "brands": list(VALID_BRAND_IDS),
            "solvents": list(VALID_SOLVENT_TYPES),
            "roles": list(VALID_PAINT_ROLES),
            "tree": STANDARD_SAMPLE_DIRECTORY_TREE,
        }

    @app.get("/api/sources")
    def api_list_sources() -> Dict[str, Any]:
        return {"sources": registry.list_sources()}

    @app.post("/api/sources")
    def api_add_source(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        try:
            entry = registry.add_source(str(payload.get("url", "")), payload.get("defaults"))
            return {"source": entry, "sources": registry.list_sources()}
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.delete("/api/sources")
    def api_remove_source(url: str = Query(...)) -> Dict[str, Any]:
        try:
            removed = registry.remove_source(url)
            return {"removed": removed, "sources": registry.list_sources()}
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    # ------------------------------------------------------------------------
    # Section 5.3 Minimum REST Endpoints
    # ------------------------------------------------------------------------

    @app.post("/api/jobs")
    def api_create_job(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        try:
            sources = payload.get("sources") or payload.get("urls") or []
            defaults = payload.get("defaults") or {}
            if sources and isinstance(sources[0], str):
                sources = [{"url": u, "defaults": defaults} for u in sources if str(u).strip()]
            job = mgr.create_job(
                sources,
                vision_mode=str(payload.get("vision_mode", "manual")),
                model=payload.get("model"),
            )
            return job
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/jobs")
    def api_list_jobs() -> Dict[str, Any]:
        return {"jobs": mgr.list_jobs()}

    @app.get("/api/jobs/{job_id}")
    def api_get_job(job_id: str) -> Dict[str, Any]:
        try:
            return mgr.get_job(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/jobs/{job_id}/fetch")
    def api_fetch_job(job_id: str) -> Dict[str, Any]:
        try:
            return mgr.fetch_job_sources(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=redact_secrets(str(exc), secrets_path)
            ) from exc

    @app.get("/api/images/{image_id}")
    def api_get_image(image_id: str) -> Response:
        try:
            data = mgr.get_image_bytes(image_id)
            return Response(content=data, media_type="image/png")
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/samples/{sample_id}/crop")
    def api_get_sample_crop(sample_id: str) -> Response:
        try:
            data = mgr.get_sample_crop_bytes(sample_id)
            return Response(content=data, media_type="image/png")
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/samples/{sample_id}/mask")
    def api_get_sample_mask(sample_id: str) -> Response:
        try:
            data = mgr.get_sample_mask_bytes(sample_id)
            return Response(content=data, media_type="image/png")
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/images/{image_id}/segment/manual")
    def api_segment_manual(image_id: str, payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        try:
            return mgr.segment_image_manual(
                image_id,
                sample_prompts=payload.get("sample_prompts"),
                pos_points=payload.get("pos_points") or payload.get("pos"),
                neg_points=payload.get("neg_points") or payload.get("neg"),
                box=payload.get("box"),
                engine=payload.get("engine"),
                oklab_tol=payload.get("oklab_tol"),
                grabcut_iters=payload.get("grabcut_iters"),
                replace_existing_for_image=bool(payload.get("replace_existing", False)),
                override_validation=bool(payload.get("override_validation", False)),
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/images/{image_id}/segment/auto")
    def api_segment_auto(image_id: str, payload: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
        try:
            return mgr.segment_image_auto(
                image_id,
                model_id=payload.get("model") or payload.get("model_id"),
                segmentation_engine=payload.get("engine"),
                use_cache=bool(payload.get("use_cache", True)),
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.patch("/api/samples/{sample_id}")
    def api_patch_sample(sample_id: str, payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        try:
            return mgr.update_sample(sample_id, payload)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.delete("/api/samples/{sample_id}")
    def api_delete_sample(sample_id: str) -> Dict[str, Any]:
        try:
            deleted = mgr.delete_sample(sample_id)
            return {"deleted": deleted, "sample_tmp_id": sample_id}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/jobs/{job_id}/ingest")
    def api_ingest_job(job_id: str, payload: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
        try:
            rows = mgr.ingest_job_samples(
                job_id,
                sample_tmp_ids=payload.get("sample_tmp_ids"),
                overwrite=bool(payload.get("overwrite", False)),
            )
            return {
                "job_id": job_id,
                "ingested_count": len(rows),
                "sample_ids": [r["sample_id"] for r in rows],
                "rows": rows,
            }
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/settings")
    def api_get_settings() -> Dict[str, Any]:
        settings = load_collector_settings(settings_path)
        settings["auth_status"] = get_public_secrets_status(secrets_path)
        return settings

    @app.put("/api/settings")
    def api_put_settings(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        saved = save_collector_settings(payload, settings_path)
        saved["auth_status"] = get_public_secrets_status(secrets_path)
        return saved

    @app.put("/api/secrets")
    def api_put_secrets(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        try:
            status = save_secrets(payload, secrets_path)
            return {"ok": True, "auth_status": status}
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=redact_secrets(str(exc), secrets_path)
            ) from exc

    @app.get("/api/models")
    def api_get_models() -> Dict[str, Any]:
        cfg = load_collector_settings(settings_path)
        models = list_available_vision_models(secrets_path)
        return {
            "models": models,
            "default_model": cfg.get("default_vision_model", "gemini-3.8-flash"),
        }

    @app.post("/api/models/test")
    def api_test_model_connection(payload: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
        return test_genai_connection(
            model_id=payload.get("model"),
            secrets_path=secrets_path,
        )

    # ------------------------------------------------------------------------
    # Section 5.2 WebSocket Progress Channel (`/ws/jobs/{job_id}`)
    # ------------------------------------------------------------------------

    @app.websocket("/ws/jobs/{job_id}")
    async def ws_job_events(websocket: WebSocket, job_id: str) -> None:
        # Source: https://fastapi.tiangolo.com/reference/websockets/
        await websocket.accept()
        loop = asyncio.get_running_loop()
        aq: asyncio.Queue = asyncio.Queue(maxsize=256)
        sub_entry = (loop, aq)
        with ws_lock:
            ws_subscribers.setdefault(job_id, []).append(sub_entry)

        # Send initial snapshot event so connected clients immediately know status
        await websocket.send_json({"job_id": job_id, "type": "ws_connected"})
        try:
            while True:
                ev = await aq.get()
                await websocket.send_json(ev)
        except WebSocketDisconnect:
            pass
        finally:
            with ws_lock:
                if job_id in ws_subscribers and sub_entry in ws_subscribers[job_id]:
                    ws_subscribers[job_id].remove(sub_entry)

    return app


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Multi-Site Paint Sample Collector Server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (defaults to 127.0.0.1 only)")
    parser.add_argument("--port", type=int, default=8765, help="Port number (default: 8765)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open browser")
    args = parser.parse_args(argv)

    import uvicorn

    url = f"http://127.0.0.1:{args.port}/"
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()

    app = create_collector_app()
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
