# LocalMediaRAG

A fully **offline** retrieval app over your personal photo gallery and document folder. It recognizes and clusters the *people* in your photos, builds a searchable text index of your documents, and answers natural-language queries — no internet, no cloud, nothing leaves your machine.

```
python server.py
# open http://localhost:8747
```

---

## What it does

| Query | Result |
|---|---|
| `photos of me with Mr.X` | Photo(s) containing **both** people |
| `photos of me` | Every photo where "me" appears |
| `my python certificates` | Matching documents, certificates ranked first |
| `show me photos of mr.x` | Mr.X's photos ("me" here is the requester, not the subject) |

If the phrase means "the two of us together" ("me **with** Mr.X"), only photos containing everyone named are returned; a single person matches all photos containing them.

## Folders

```
media-rag/
  server.py            # stdlib HTTP server: API + static files + thumbnails
  index.html           # the web UI (single file)
  models/              # bundled OpenCV YuNet + SFace ONNX models (offline)
  media/
    photos/            # ✏️ your photo gallery  (jpg/png/webp/bmp, any nesting)
    documents/         # ✏️ txt, md, docx, pdf, xlsx, csv, ...
    labels/            # optional per-person reference photos
  index/               # generated index — safe to delete, rebuilt on rescan
```

**Reference photos (optional).** Drop one or more photos of a person into `labels/<name>/` and the matching faces will be named *<name>* automatically on the next rescan — useful for anchoring people who only appear once in the gallery.

## Getting started

1. Put your photos in `media/photos/` and documents in `media/documents/`.
2. Run `python server.py` (needs `numpy` + `opencv-python`; two ONNX models bundled in `models/`).
3. Open **http://localhost:8747**.
4. Faces cluster automatically as **Unnamed group** bubbles in the **People** panel — type a name and press Enter. That name is now searchable ("me", "Mr.X", "Alex", anything).
5. Use the **Rescan media** button after adding or removing files. Cluster sensitivity is adjustable via the slider in the People panel.

## How it works — briefly

- **Faces** — YuNet detects faces; SFace embeds each face (512-d, L2-normalized); average-linkage agglomerative clustering groups them into people (the tolerance slider trades merge-together vs. split-apart).
- **Names survive rescans** — user labels are carried across re-clusters by matching a new cluster's centroid to the previously named one, and `labels/` reference photos act as authoritative anchors.
- **Documents** — pure-stdlib extraction: `docx`/`xlsx` via `zipfile`+XML, legacy text `pdf` via content-stream parsing, plain text read directly. A TF-IDF vector per document (numpy, no sklearn); filename terms are boosted and certificate-type keywords get a bonus.
- **Query routing** — rule-based: recognized person names route to photos; anything else routes to documents. `to`/"with"/"and" differences, "me" vs. "show me", and category words are handled heuristically so the phrasing above works.

## Notes & limits

- **No OCR.** A scanned/photo-of-text certificate isn't searchable by content — only by filename, and only through the photo pipeline. Adding Tesseract later would lift this.
- Clustering quality depends a lot on pose/lighting/faces-per-photo. The tolerance slider and `labels/` reference photos are the tuning tools.
- The HTTP server binds `127.0.0.1` only — it's local-first by design.
- Safe to delete `index/`; the next start or a **Rescan media** rebuilds everything.

## Your data

Personal photos, documents, face labels, and generated indexes are not included in this repository. Add your files under `media/photos/` and `media/documents/`, then run the app and click **Rescan media**. These paths are ignored by Git so your local data and generated indexes are not accidentally committed.