# Last Seen

Object-permanence memory for your home, built on Meta Model API.

Walk through your rooms with your phone for a minute. Later, ask:

> where did I leave my passport?  ·  what moved off the counter since this morning?

and get the sentence, the frame with the object masked, the place it was before, and a SAM-tracked
clip of the moment. Nothing is labelled by hand: Muse Spark finds and names every movable thing,
SAM 3.1 cuts it out, DINO embeddings plus Spark pairwise judgement decide which sightings are the
same physical object across days, and a small state machine turns that into *stationary / moved /
new / not seen*.

Built solo for the Meta Global AI Developer Hackathon (Oct 26 – Nov 4, 2026), indie track.

## What Meta models do here

| Model | Used for | Calls per 60 s clip |
|---|---|---|
| **Muse Spark 1.3** (vision, structured JSON, `reasoning_effort=low`) | keyframe grounding: room, surfaces, every movable object with a 0–1000 box · one attribute record per object cluster (canonical name, category, colours, brand, marks) · pairwise "same object?" judgement with a single-home prior · question → intent JSON | ~30 + ~40 + 5–20 + 1 per question |
| **SAM 3.1** (image box prompts, ≤16 per request) | pixel masks + tight boxes for every grounded object → clean crops for embedding and the answer overlay | ~30 |
| **SAM 3.1** (video box prompt, 150-frame window) | "Play the moment": tracked mask over the seconds around the answer | 1 per answer, on demand |
| **DINOv2-small** (Meta FAIR, Apache-2.0; DINOv3 ViT-S/16 wired in behind its gated login) | three embeddings per sighting on Apple MPS (object, mask-pooled patches, background ring), 13 ms each | local |

Every API call is saved under `runs/<run>/api/NNN_<kind>.json` (request without media, response,
usage, timing), and cached under `data/api_cache/` so re-runs are free.

## Setup (macOS, Apple Silicon; Linux with CUDA or CPU also works)

```bash
git clone https://github.com/<you>/last-seen && cd last-seen
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
brew install ffmpeg                          # if `ffmpeg -version` fails
echo "MODEL_API_KEY=..." > .env              # key from dev.meta.ai — never commit it
hf auth login                                # optional: unlocks DINOv3 (gated); thresholds were tuned on DINOv2 — re-tune with scripts/replay.py
```

## Run

```bash
python scripts/smoke_test.py                 # first contact: 8 API calls on a synthetic image/video
python scripts/ingest.py clips/before.mov --session before --stride 1
python scripts/ingest.py clips/after.mov  --session after  --stride 1   # 1 s keyframes: handheld clips are blurry
python scripts/ask.py "where is my massage gun?"
python scripts/ask.py "what moved since this morning?"
python scripts/ask.py --window "where is my charger?"   # also builds the SAM-tracked answer clip
uvicorn app.server:app --port 8008           # web app at http://localhost:8008
```

## Filming for good results

The system is only as good as what the camera saw. What worked, and what did not, on the clips here:

* **Pause 1–2 s on every surface**, framing the whole surface, then move on slowly. Keyframes are taken
  every second (`--stride 1`); a pause gives two sharp frames of everything on it. Fast pans are the
  blurry crops the system keeps but never trusts.
* **Film the same route from roughly the same angles each time.** Same viewpoint → embedding
  similarity 0.8–0.97; a flat-on-the-table view vs. a standing one can drop to 0.2 and then only
  Spark's pairwise judgement can link them.
* **Same lighting every session** (daylight, or all lights on). Colour names feed the matching; a
  dim room turns grey metal black.
* **Keep hands out of frame.** An object in hand is a separate place.
* **Hero objects:** anything hand-sized or bigger with a distinct shape or colour (keys with a fob,
  wallet, watch, remote, bottle, headphones). Weak: cables, paper, socks, anything that looks like its
  neighbours — and two identical items are one object to the system.
* **Name sessions by what they are** (`--session before`, `--session after`, `--session tuesday-pm`):
  the names appear in answers ("what did the *after* walk-through change?").

Expect the first run in a new home to need a few taps: look-alikes the system could not settle land in
*Needs your call* with both crops and Spark's reason, and anything it only glimpsed once is reported as
*less sure* rather than as fact. If a pair is settled wrongly, `scripts/explain.py <sighting_id>` shows
which gate decided and `scripts/replay.py` re-runs the whole re-ID on stored embeddings without
spending API calls.

Tuning without spending API calls:

```bash
python scripts/replay.py --requality --retrack   # re-run clustering → re-ID → dedupe → states on stored embeddings
REID_T_MATCH=0.75 python scripts/replay.py       # any threshold in lastseen/reid.py is an env var
python scripts/explain.py 143                    # why a sighting matched (or didn't): candidates, scores, gates
python scripts/sheet.py objects                  # contact sheet: one row per object, crops per session
python scripts/pair_test.py --a 6 --b 152 --object 8 --session 2   # one Spark pairwise call, as the pipeline would ask it
python scripts/export_static.py docs/demo        # the app as a static site with pre-computed answers (served at /demo by the app, and by GitHub Pages)
```

## Pipeline

![architecture](docs/architecture.png)

```
clip ─ffmpeg─▶ 1280px CFR keyframes (2 s stride + scene cuts, sharpest of ±3 frames)
keyframe ─Spark grounding─▶ room, surfaces, movable objects (stickers / fragments / blurry things filtered)
boxes ─SAM 3.1 image prompts─▶ masks  ─▶ crops: object (bg blanked), background ring, full
crops ─DINOv2─▶ e_obj, e_patch, e_ring  +  mean CIELAB of the object's pixels  +  crop quality
quality = contrast-normalised sharpness × size; blurry/tiny crops are kept but never drive a match
tracklets (same-name boxes across keyframes) ─▶ clusters (embedding ≥ 0.65, or ≥ 0.45 with the same noun; never co-seen)
cluster ─Spark attributes─▶ canonical name, category, colours, brand, marks
re-ID cascade vs the object table: co-occurrence veto → colour veto → thresholds → Spark pairwise → ask the user
dedupe: looks alike · same name never co-seen · "not seen at P" + "new at P" ─▶ Spark decides
objects + places (room, surface) ─▶ per-session modal place ─▶ stationary / moved / new / not seen / possibly hidden
question ─Spark intent over the catalog─▶ SQL ─▶ answer card (+ "before that it was on …", + confidence tiers)
answer ─SAM 3.1 video, 150 frames both directions─▶ tracked overlay clip
```

Two structural rules do more than any threshold:

* **co-occurrence** — two sightings on one keyframe are different objects, always;
* **single-home prior** — in one home there is usually one of each thing: two sightings of the same
  kind and colour that were never seen together are the same object unless a pairwise look finds a
  real difference (Spark is told that blur, viewpoint and lighting are not evidence).

Why keyframes + short tracking windows instead of tracking the whole clip: SAM 3.1 video is billed
per frame with a 50-frame minimum and track ids do not survive objects leaving and re-entering the
frame, which a handheld walk-through does constantly. Keyframe sightings are the identity record;
video tracking is spent only where the demo needs motion.

## What the first real clips taught us (measured, Oct 27–28)

Two 60 s iPhone walk-throughs of the same apartment, minutes apart, after moving a massage gun
(kitchen counter → dining table) and a pair of scissors (dining table → kitchen counter).

* DINOv2 similarity (0.6·obj + 0.4·patch) of the same object across the two clips, best exemplar:
  massage gun 0.91, jacket 0.97, laptop 0.82, steel bottle 0.88, scissors 0.64. Different large
  objects: ≤ 0.54. Same-kind different objects (soap bottle vs steel bottle): ≤ 0.39.
* The same scissors flat on a table vs. standing in a cup: 0.2. Embeddings alone cannot link those;
  the single-home prior + Spark can.
* DINO is nearly colour-blind: a peach soap bottle and a steel bottle embed alike; mean CIELAB of the
  masked pixels separates them (a/b distance > 22 is a veto).
* The first run produced 147 objects for ~40 real ones. Root causes, all fixed: a loose
  "nearby + same noun" tracklet rule chained neighbours on a counter; Spark reasoning ate the
  output budget (null JSON) until `max_tokens ≥ 4000` with `reasoning_effort=low`; blurry crops
  auto-matched on looks alone; colour names contradict for the same item ("blue" vs "dark gray"
  jacket) so colours are compared as sets and a strong same-name match overrides them.
* After the fixes (replayed on the same embeddings): massage gun = one object, 13 sightings, *moved
  kitchen counter → dining table*; scissors *moved dining table → kitchen counter*; bottle, laptop,
  jacket stationary. "What moved" lists them first with high confidence; single-glimpse changes are
  shown separately as "less sure".

## The demo clips (Oct 30): five planted moves, five found

Two ~55 s walk-throughs 35 minutes apart (`before`, `after`), after moving five everyday things:
a smartwatch (kitchen counter → sofa), the TV remote (sofa → living-room dresser), a wallet (kitchen
counter → top of the washing machine), a massage gun (dining table → laundry room) and the car keys
(hallway hook → kitchen counter). 182 sightings, 30 objects. "What did the after walk-through change?"
lists exactly those five as *moved* with high confidence; nothing that stayed put is reported as moved.

The keys were the hard one, and they taught three rules that are now in the code:

* **Absence is evidence, but only in a re-filmed spot.** The hook was never in Spark's scene surfaces
  (a wall of hooks is not a "surface"), so the system believed it had not looked there and refused to
  claim a move. A surface where *any* object was seen in a session now counts as scanned — the other
  two key sets were seen on that hook, so the hook was looked at and the black keys were not on it.
* **A partial mask is worse than no mask.** SAM cut the keys away from their fob; the object was
  named "silver keys" and Spark, judging half an object, called the counter sighting a different set.
  When a mask covers under 35 % of its box the plain crop is used instead, for naming and for
  pairwise checks.
* **Tell the judge what the footage knows — carefully.** The pairwise prompt now carries "A was last
  seen on the hallway hook; the later walk-through filmed the hook again and A was not there; B
  appeared on the kitchen counter". That flipped the keys. It also flipped a white cap into a grey
  one, so the context is only added when appearance already makes "same thing" plausible (similarity
  ≥ 0.5, compatible colours), Spark's answers are remembered and a confident "different" is never
  re-asked with a leading question, and "not seen" claims built only on blurry glimpses are demoted
  to *less sure*.

## Honesty rules in the answers

* A "moved" needs two or more sightings on each side; otherwise it is reported as *less sure*.
* "Not seen" means: the place was scanned later and the object was not found there — never "gone".
* When two objects match the question, the one seen many times wins; the other is offered, not asked.
* Ambiguous same-object decisions go to *Needs your call* with both crops and Spark's reason.

## Cost (60 s clip at 30 fps, ~30 keyframes, ~40 objects)

| Call | Count | ≈ Cost |
|---|---|---|
| Spark grounding (one image each) | 30 | $0.35 |
| SAM image box prompts | 30 | $0.08 |
| Spark attributes (per cluster) | ~40 | $0.15 |
| Spark pairwise verification | 5–20 | ≤ $0.10 |
| SAM video window (150 frames) | per answer | $0.03 |

About $0.70 per clip; re-runs hit the cache.

## Layout

```
lastseen/   meta_client · video · sam_parse · rle · db · prompts · ingest · reid · state · windows · render · query · api
scripts/    smoke_test · ingest · ask · replay · explain · sheet · eval · export_static · bench_embed
app/        FastAPI server + single-page UI (app/static/index.html); the same UI runs as a static site
docs/       architecture figure; demo/ = the exported static demo (real data, no key needed)
tests/      offline end-to-end test with a mock API (python tests/test_pipeline_mock.py)
data/       clips/ frames/ crops/ api_cache/ lastseen.sqlite   (gitignored)
runs/       per-run API logs                                   (gitignored)
```

## Licenses

SAM 3.1, Muse Spark and Muse Voice Transcribe are used through Meta Model API under its terms.
`facebook/dinov2-small` is Apache-2.0. If you switch to DINOv3, its weights are under the DINOv3
License ("Built with DINOv3"). `meta-sam-parser` is under the SAM License. This code is MIT.
