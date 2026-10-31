"""First real contact with the Meta Model API. Run from the project root:

    python scripts/smoke_test.py            # all steps
    python scripts/smoke_test.py --no-video # skip the video step (saves ~$0.01 and a minute)

Needs .env with MODEL_API_KEY. Writes everything to runs/smoke-<timestamp>/ :
  api/*.json            every request/response
  synthetic.png         test image (three coloured squares + a small grey 'charger')
  sam_concept_*.png     overlays from concept prompts
  sam_box_overlay.png   overlay from a box prompt (+ mask-local dims printed)
  spark_grounding.json  Spark's structured object list for the image
  video_tracks.json     SAM video tracking summary (if run)
  summary.json          pass/fail per step, timings, usage
Nothing here is Last-Seen-specific; it only validates assumptions the pipeline relies on.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from lastseen import config, sam_parse  # noqa: E402
from lastseen.meta_client import MetaClient, MetaAPIError, image_to_data_url  # noqa: E402

W, H = 960, 540
SQUARES = {  # name: (x1, y1, x2, y2, colour)
    "red square": (120, 120, 300, 300, (220, 30, 30)),
    "blue square": (420, 160, 620, 360, (30, 60, 220)),
    "yellow square": (700, 80, 900, 280, (240, 210, 20)),
    "grey charger": (650, 400, 760, 450, (70, 70, 75)),
}


def make_image(path: Path) -> None:
    img = Image.new("RGB", (W, H), (235, 232, 225))
    d = ImageDraw.Draw(img)
    for name, (x1, y1, x2, y2, col) in SQUARES.items():
        d.rectangle([x1, y1, x2 - 1, y2 - 1], fill=col)
    # a "table" line for context
    d.rectangle([40, 470, 920, 480], fill=(120, 90, 60))
    img.save(path)


def make_video(path: Path, nframes: int = 90, fps: int = 30) -> None:
    """Red square moves left->right; grey charger stays; blue square leaves the frame."""
    tmp = path.parent / "vframes"
    tmp.mkdir(exist_ok=True)
    for i in range(nframes):
        img = Image.new("RGB", (W, H), (235, 232, 225))
        d = ImageDraw.Draw(img)
        dx = int(i * 4)
        d.rectangle([120 + dx, 120, 300 + dx, 300], fill=(220, 30, 30))
        bx = 420 + int(i * 8)
        d.rectangle([bx, 160, bx + 200, 360], fill=(30, 60, 220))
        d.rectangle([650, 400, 760, 450], fill=(70, 70, 75))
        img.save(tmp / f"{i:05d}.png")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", str(fps), "-i", str(tmp / "%05d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)], check=True)


def overlay(img: Image.Image, frames: list[sam_parse.SamFrame], out: Path) -> dict:
    """Draw boxes + (if decodable) masks; return info about mask dims."""
    base = img.convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    info: dict = {"records": []}
    colours = [(255, 80, 0, 110), (0, 160, 255, 110), (0, 200, 90, 110), (200, 0, 200, 110)]
    for fr in frames[:1]:
        for i, rec in enumerate(fr.records):
            x1, y1, x2, y2 = rec.box
            c = colours[i % len(colours)]
            rec_info = {"object_id": rec.object_id, "box": rec.box, "conf": rec.confidence,
                        "mask_hw": rec.mask_hw, "box_hw": (y2 - y1, x2 - x1), "decoded": False}
            try:
                local = sam_parse.decode_mask(rec)
                full = sam_parse.full_frame_mask(rec, local)
                rec_info["decoded"] = True
                rec_info["mask_local_shape"] = list(local.shape)
                rec_info["mask_area"] = int(full.sum())
                m = Image.fromarray((full * c[3]).astype(np.uint8), mode="L")
                colour_img = Image.new("RGBA", base.size, c[:3] + (0,))
                colour_img.putalpha(m)
                layer = Image.alpha_composite(layer, colour_img)
                d = ImageDraw.Draw(layer)
            except Exception as e:  # noqa: BLE001
                rec_info["decode_error"] = f"{type(e).__name__}: {e}"[:200]
            d.rectangle([x1, y1, x2 - 1, y2 - 1], outline=c[:3] + (255,), width=3)
            info["records"].append(rec_info)
    Image.alpha_composite(base, layer).convert("RGB").save(out)
    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--only-video", action="store_true", help="skip image/Spark steps, run only the SAM video step")
    ap.add_argument("--image", help="use your own JPG/PNG instead of the synthetic image")
    ap.add_argument("--concept", default=None, help="concept prompt for your own image (e.g. 'coffee mug')")
    args = ap.parse_args()

    run_dir = config.RUNS / time.strftime("smoke-%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True)
    client = MetaClient(run_dir=run_dir)
    summary: dict = {"run_dir": str(run_dir), "steps": {}}

    def step(name: str, fn):
        t0 = time.time()
        try:
            out = fn()
            summary["steps"][name] = {"ok": True, "s": round(time.time() - t0, 1), "out": out}
            print(f"[ok]   {name} ({time.time() - t0:.1f}s)")
        except Exception as e:  # noqa: BLE001
            summary["steps"][name] = {"ok": False, "s": round(time.time() - t0, 1), "error": f"{type(e).__name__}: {e}"[:1500]}
            print(f"[FAIL] {name}: {type(e).__name__}: {str(e)[:600]}")
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))

    # 0. models
    if not args.only_video:
        step("list_models", lambda: client.list_models())

    # 1. image
    if args.image:
        img_path = Path(args.image)
        concepts = [args.concept or "cup"]
    else:
        img_path = run_dir / "synthetic.png"
        make_image(img_path)
        concepts = ["red square", "square", "charger"]
    img = Image.open(img_path).convert("RGB")
    data_url = image_to_data_url(img_path)
    iw, ih = img.size

    # 2. SAM concept prompts
    concept_results: dict = {}

    def sam_concept(concept: str):
        text = client.sam_responses(concept, image_data_url=data_url, mask_encoding="one_bit",
                                    include_confidence=True, stream=True, kind=f"sam_concept_{concept.replace(' ', '_')}")
        frames = sam_parse.parse_lines(text)
        (run_dir / f"sam_concept_{concept.replace(' ', '_')}.txt").write_text(text)
        info = overlay(img, frames, run_dir / f"sam_concept_{concept.replace(' ', '_')}.png") if frames else {"records": []}
        concept_results[concept] = info
        return {"n_records": len(frames[0].records) if frames else 0, "raw_len": len(text), **info}

    for c in ([] if args.only_video else concepts):
        step(f"sam_concept:{c}", lambda c=c: sam_concept(c))

    # 3. SAM box prompt (two objects) — on synthetic image we know the boxes
    def sam_box():
        if args.image:
            # use the first concept result's boxes as prompts (re-segment what we found)
            recs = concept_results.get(concepts[0], {}).get("records", [])
            objs = [{"id": i + 1, "prompts": [{"frame": 0, "box": r["box"]}]} for i, r in enumerate(recs[:16])]
            if not objs:
                raise RuntimeError("no boxes from concept step to re-prompt with")
        else:
            objs = [{"id": 1, "prompts": [{"frame": 0, "box": SQUARES["red square"][:4]}]},
                    {"id": 2, "prompts": [{"frame": 0, "box": SQUARES["grey charger"][:4]}]},
                    {"id": 7, "prompts": [{"frame": 0, "points": [(520, 260, 1)]}]}]
        prompt = sam_parse.build_box_prompt(iw, ih, objs)
        text = client.sam_responses(prompt, image_data_url=data_url, include_confidence=True, stream=False,
                                    kind="sam_box_image")
        frames = sam_parse.parse_lines(text)
        (run_dir / "sam_box_image.txt").write_text(text)
        info = overlay(img, frames, run_dir / "sam_box_overlay.png") if frames else {"records": []}
        ids = [r.object_id for r in frames[0].records] if frames else []
        return {"prompt": prompt[:300], "ids_returned": ids, **info}

    if not args.only_video:
        step("sam_box_image", sam_box)

    # 4. Spark grounding with structured output
    def spark_grounding():
        schema = {
            "type": "object", "additionalProperties": False,
            "required": ["room", "objects"],
            "properties": {
                "room": {"type": "string"},
                "objects": {"type": "array", "items": {
                    "type": "object", "additionalProperties": False,
                    "required": ["label", "description", "bbox", "confidence"],
                    "properties": {
                        "label": {"type": "string"},
                        "description": {"type": "string"},
                        "bbox": {"type": "object", "additionalProperties": False,
                                 "required": ["x_min", "y_min", "x_max", "y_max"],
                                 "properties": {k: {"type": "integer"} for k in ("x_min", "y_min", "x_max", "y_max")}},
                        "confidence": {"type": "number"}}}}}}
        prompt = ("You are an object grounding expert. List every distinct object in the image (for this test, "
                  "coloured squares count as objects). For each give a short label, a one-line description, and the "
                  "bounding box as x_min, y_min, x_max, y_max on a normalized 0-1000 grid with (0,0) at the top-left. "
                  "Do not miss any objects. Also name the kind of place this is in 'room'.")
        res = client.spark_json(prompt, images=[data_url], schema=schema, schema_name="grounding",
                                max_tokens=4000, kind="spark_grounding")
        (run_dir / "spark_grounding.json").write_text(json.dumps(res, indent=1))
        # convert to pixels and compare to known boxes
        px = []
        for o in res.get("objects", []):
            b = o["bbox"]
            box = (round(b["x_min"] * iw / 1000), round(b["y_min"] * ih / 1000),
                   round(b["x_max"] * iw / 1000), round(b["y_max"] * ih / 1000))
            px.append({"label": o["label"], "box": box})
        return {"room": res.get("room"), "n": len(px), "boxes": px}

    if not args.only_video:
        step("spark_grounding", spark_grounding)

    # 5. Spark pairwise same-object question (two crops of the same image region)
    def spark_pair():
        a = img.crop(SQUARES["grey charger"][:4]).resize((220, 100))
        b = img.crop(SQUARES["red square"][:4]).resize((200, 200))
        from lastseen.meta_client import bytes_to_data_url
        import io
        def du(im):
            buf = io.BytesIO(); im.save(buf, format="PNG"); return bytes_to_data_url(buf.getvalue(), "image/png")
        schema = {"type": "object", "additionalProperties": False, "required": ["same", "confidence", "reason"],
                  "properties": {"same": {"type": "boolean"}, "confidence": {"type": "number"}, "reason": {"type": "string"}}}
        res = client.spark_json("Image 1 and image 2 are crops of household objects taken at different times. "
                                "Are they the SAME physical object? Answer with same=true/false, confidence 0-1, and a short reason.",
                                images=[du(a), du(b)], schema=schema, schema_name="same_object", max_tokens=1500,
                                kind="spark_pair")
        return res

    if not args.only_video:
        step("spark_pairwise", spark_pair)

    # 6. SAM video with box prompts (forward)
    if not args.no_video:
        def sam_video():
            vpath = run_dir / "synthetic.mp4"
            make_video(vpath, nframes=90, fps=30)
            fid = client.upload_file(vpath)
            objs = [{"id": 1, "prompts": [{"frame": 0, "box": (120, 120, 300, 300)}]},
                    {"id": 2, "prompts": [{"frame": 0, "box": (650, 400, 760, 450)}]},
                    {"id": 3, "prompts": [{"frame": 0, "box": (420, 160, 620, 360)}]}]
            prompt = sam_parse.build_box_prompt(W, H, objs)
            t0 = time.time()
            text = client.sam_responses(prompt, video_file_id=fid, include_confidence=True,
                                        propagation_direction="forward", stream=True, kind="sam_video_box")
            el = time.time() - t0
            frames = sam_parse.parse_lines(text)
            (run_dir / "sam_video_box.txt").write_text(text)
            per_obj: dict[int, list[int]] = {}
            for fr in frames:
                for r in fr.records:
                    per_obj.setdefault(r.object_id, []).append(fr.frame)
            # track the red square's x over time (expect ~ +4px/frame)
            red = sorted((fr.frame, r.box[0]) for fr in frames for r in fr.records if r.object_id == 1)
            usage = client.calls[-1].usage
            out = {"file_id": fid, "n_frames_returned": len(frames), "elapsed_s": round(el, 1),
                   "frames_per_object": {k: [min(v), max(v), len(v)] for k, v in per_obj.items()},
                   "red_x_first_last": [red[0], red[-1]] if red else None, "usage": usage,
                   "frame_order_ascending": [f.frame for f in frames] == sorted(f.frame for f in frames)}
            (run_dir / "video_tracks.json").write_text(json.dumps(out, indent=1, default=str))
            return out

        step("sam_video_box", sam_video)

    # 7. cost summary
    summary["calls"] = [{"kind": c.kind, "status": c.status, "s": c.elapsed_s, "usage": c.usage} for c in client.calls]
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    ok = sum(1 for s in summary["steps"].values() if s["ok"])
    print(f"\n{ok}/{len(summary['steps'])} steps passed. Details: {run_dir}/summary.json")
    return 0 if ok == len(summary["steps"]) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except MetaAPIError as e:
        print("API error:", e)
        sys.exit(2)
