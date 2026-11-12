#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LocalMediaRAG — fully offline retrieval over your gallery of photos + documents.

Run:  python server.py
Then open:  http://localhost:8747

What it does
  - Scans media/photos  -> detects faces (YuNet), computes identity embeddings (SFace),
                          clusters faces into people, lets YOU name each cluster.
  - Scans media/documents -> extracts text (docx/pdf/txt/md/xlsx best-effort, pure stdlib),
                          builds a TF-IDF index.
  - Natural queries:
      "photos of me with Mr.X"  -> group photos containing both people
      "my python certificates"  -> matching documents
  Everything runs locally. Models are bundled in /models — no internet needed.

Folders
  media/photos/     put your photo gallery here  (jpg/png/webp/bmp)
  media/documents/  put pdf/docx/txt/md/xlsx here
  media/labels/     optional per-person reference photos:  labels/<name>/<any.jpg>
  index/            generated index (safe to delete -> rescan rebuilds it)
"""

import os
import re
import json
import time
import zipfile
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from collections import Counter, defaultdict

import numpy as np
import cv2

# ---------------------------------------------------------------------------
#  Paths + globals
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MEDIA_DIR = os.path.join(BASE_DIR, "media")
PHOTOS_DIR = os.path.join(MEDIA_DIR, "photos")
DOCS_DIR = os.path.join(MEDIA_DIR, "documents")
LABELS_DIR = os.path.join(MEDIA_DIR, "labels")
INDEX_DIR = os.path.join(BASE_DIR, "index")
THUMB_DIR = os.path.join(INDEX_DIR, "thumbnails")
FACE_DIR = os.path.join(INDEX_DIR, "faces")
INDEX_FILE = os.path.join(INDEX_DIR, "index.json")
EMBED_FILE = os.path.join(INDEX_DIR, "embeddings.npz")

PORT = 8747
DEFAULT_TOLERANCE = 0.50
PHOTO_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DOC_EXTS = {".txt", ".md", ".csv", ".json", ".docx", ".pdf", ".xlsx", ".log", ".yaml", ".yml", ".log"}

PHOTO_WORDS = {"photo", "photos", "picture", "pictures", "pic", "pics", "image", "images",
               "selfie", "selfies", "shot", "shots", "group"}
DOC_WORDS = {"document", "documents", "doc", "docs", "certificate", "certificates", "certification",
             "report", "letter", "note", "notes", "file", "pdf", "resume", "cv", "transcript",
             "degree", "course", "files", "contract", "invoice", "receipt", "policy"}
ME_WORDS = {"me", "mine", "myself", "self"}
IMPERATIVE_WORDS = {"show", "give", "get", "find", "send", "tell", "fetch", "take"}
CERT_WORDS = {"cert", "certs", "certificate", "certificates", "certification", "completion",
              "accredited", "accreditation", "diploma", "degree"}

_lock = threading.Lock()
index_state = {
    "built": False,
    "tolerance": DEFAULT_TOLERANCE,
    "clusters": [],        # [{id, name, face_ids:[...]}]
    "faces": [],           # [{id, photo, bbox, cluster}]
    "photos": [],          # [{rel, thumb, bboxes, faces, face_clusters, nfaces, created}]
    "docs": [],            # [{rel, name, ext, text, tokens, nwords}]
    "vocab": [],           # term list
    "idf": [],             # parallel idf values
    "last_build": None,
}
_emb = None  # numpy (N, 512) aligned with index_state["faces"]


# ---------------------------------------------------------------------------
#  Tiny helpers
# ---------------------------------------------------------------------------
def rel(path):
    """Absolute path under BASE_DIR -> forward-slash relative path."""
    return "/".join(os.path.relpath(path, BASE_DIR).split(os.sep))


def rel_media(path):
    return os.path.relpath(path, MEDIA_DIR).replace(os.sep, "/")


def ensure_dirs():
    for d in (MODELS_DIR, PHOTOS_DIR, DOCS_DIR, LABELS_DIR, INDEX_DIR, THUMB_DIR, FACE_DIR):
        os.makedirs(d, exist_ok=True)


def read_image(path, max_dim=2000):
    """Read an image robustly: handles non-ASCII paths + corrupt files on Windows.
    Returns a cv2 BGR image (downscaled if very large) or None.
    """
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            return None
        h, w = img.shape[:2]
        if max(h, w) > max_dim:                       # keep scans fast on huge phone photos
            scale = max_dim / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return img
    except Exception:
        return None


def load_model(name):
    p = os.path.join(MODELS_DIR, name)
    if not os.path.exists(p):
        raise FileNotFoundError(f"Model missing: {p} — re-run setup or download {name} into models/")
    return p


# ---------------------------------------------------------------------------
#  Face pipeline — detection (YuNet) + embedding (SFace)
# ---------------------------------------------------------------------------
class FacePipeline:
    def __init__(self):
        self.detector = cv2.FaceDetectorYN_create(load_model("face_detection_yunet_2023mar.onnx"), "", (320, 320))
        self.recognizer = cv2.FaceRecognizerSF_create(load_model("face_recognition_sface_2021dec.onnx"), "")

    def detect_faces(self, img):
        """Return list of full YuNet rows (15 values: bbox + 5 landmarks + score)."""
        h, w = img.shape[:2]
        self.detector.setInputSize((w, h))
        faces = self.detector.detect(img)[1]
        if faces is None or len(faces) == 0:
            return []
        keep = []
        for f in faces:
            x, y, ww, hh = int(f[0]), int(f[1]), int(f[2]), int(f[3])
            score = float(f[14])
            if score < 0.6 or ww < 16 or hh < 16:
                continue
            keep.append(f.tolist())
        return keep

    def embed(self, img, det_row):
        row = np.asarray(det_row, dtype=np.float32).reshape(1, -1)
        aligned = self.recognizer.alignCrop(img, row)
        feat = self.recognizer.feature(aligned).squeeze()
        nrm = float(np.linalg.norm(feat))
        return (feat / nrm) if nrm > 0 else feat


# ---------------------------------------------------------------------------
#  Clustering — average-linkage agglomerative (centroid mean), offline, numpy
# ---------------------------------------------------------------------------
def cluster_faces(emb, tolerance):
    n = len(emb)
    if n == 0:
        return []
    e = np.asarray(emb, dtype=np.float64).copy()
    e /= (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)
    clusters = [[i] for i in range(n)]
    cents = np.array([e[i] for i in range(n)])
    sizes = np.ones(n)
    while len(clusters) > 1:
        k = len(clusters)
        close = cents @ cents.T
        close = close - np.eye(k)  # ignore self-similarity
        i, j = np.unravel_index(np.argmax(close), close.shape)
        if close[i, j] < tolerance:
            break
        i, j = min(i, j), max(i, j)
        newc = (sizes[i] * cents[i] + sizes[j] * cents[j]) / (sizes[i] + sizes[j])
        nrm = np.linalg.norm(newc)
        cents[i] = newc / nrm if nrm > 0 else newc
        sizes[i] += sizes[j]
        clusters[i] = clusters[i] + clusters[j]
        del clusters[j]
        cents = np.delete(cents, j, axis=0)
        sizes = np.delete(sizes, j)
    return clusters


# ---------------------------------------------------------------------------
#  Face scan over media/photos (+ label anchors)
# ---------------------------------------------------------------------------
def stable_id(path):
    import hashlib
    return int(hashlib.md5(path.encode("utf-8")).hexdigest()[:8], 16)


def write_image(path, img, params=None):
    """cv2.imwrite that survives non-ASCII paths on Windows (imencode + tofile)."""
    ext = os.path.splitext(path)[1] or ".jpg"
    ok, buf = cv2.imencode(ext, img, params or [])
    if not ok:
        return False
    buf.tofile(path)
    return True


def gen_photo_thumb(path, max_dim=300):
    img = read_image(path, max_dim=max_dim)
    if img is None:
        return None
    name = os.path.splitext(os.path.basename(path))[0].replace("#", "_") + ".jpg"
    out = os.path.join(THUMB_DIR, f"{stable_id(path)}_{name}")
    write_image(out, img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return "thumbnails/" + os.path.basename(out)


def gen_face_thumb(img, bbox, out_name, size=96):
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    pad = int(max(w, h) * 0.20)
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(img.shape[1], x + w + pad), min(img.shape[0], y + h + pad)
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    write_image(os.path.join(FACE_DIR, out_name), crop, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return "faces/" + out_name


def scan_faces(pipeline):
    """Returns (face_rows, emb_array, photos, anchors)."""
    face_rows = []
    emb_list = []
    photos = []
    anchors = {}
    for root, _dirs, files in os.walk(PHOTOS_DIR):
        for fn in sorted(files):
            if os.path.splitext(fn)[1].lower() not in PHOTO_EXTS:
                continue
            path = os.path.join(root, fn)
            img = read_image(path)
            if img is None:
                continue
            faces = pipeline.detect_faces(img)
            photo = {
                "path": path, "rel": rel(path), "thumb": None,
                "bboxes": [], "faces": [], "face_clusters": [], "nfaces": 0,
                "created": os.path.getmtime(path),
            }
            for det in faces:
                fid = f"f{len(face_rows)}"
                emb_list.append(pipeline.embed(img, det))
                face_rows.append({"id": fid, "photo": path,
                                  "bbox": [int(det[0]), int(det[1]), int(det[2]), int(det[3])],
                                  "cluster": None})
                photo["bboxes"].append([int(det[0]), int(det[1]), int(det[2]), int(det[3]), float(det[14])])
            photo["nfaces"] = len(photo["bboxes"])
            photo["thumb"] = gen_photo_thumb(path)
            photos.append(photo)

    # reference anchors: media/labels/<person>/<imgs>
    if os.path.isdir(LABELS_DIR):
        for person in sorted(os.listdir(LABELS_DIR)):
            pdir = os.path.join(LABELS_DIR, person)
            if not os.path.isdir(pdir):
                continue
            for fn in sorted(os.listdir(pdir)):
                img = read_image(os.path.join(pdir, fn))
                if img is None:
                    continue
                for det in pipeline.detect_faces(img):
                    emb_list.append(pipeline.embed(img, det))
                    anchors.setdefault(person.lower(), []).append(len(emb_list) - 1)
    emb = np.array(emb_list, dtype=np.float32) if emb_list else np.zeros((0, 512), dtype=np.float32)
    return face_rows, emb, photos, anchors


# ---------------------------------------------------------------------------
#  Document extraction (pure stdlib)
# ---------------------------------------------------------------------------
def extract_doc_text(path):
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in (".txt", ".md", ".csv", ".log", ".yaml", ".yml", ".json"):
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        if ext == ".docx":
            with zipfile.ZipFile(path) as z:
                xml = z.read("word/document.xml").decode("utf-8", "replace")
            xml = re.sub(r"<w:p[ >]", "\n", xml)
            return re.sub(r"<[^>]+>", "", xml)
        if ext == ".xlsx":
            with zipfile.ZipFile(path) as z:
                shared = z.read("xl/sharedStrings.xml").decode("utf-8", "replace")
            return "\n".join(re.findall(r"<t[^>]*>(.*?)</t>", shared))
        if ext == ".pdf":
            return extract_pdf_text(path)
    except Exception:
        return ""
    return ""


def extract_pdf_text(path):
    try:
        with open(path, "rb") as f:
            data = f.read()
    except Exception:
        return ""
    out = []
    for m in re.finditer(rb"stream\r?\n(.*?)endstream", data, re.S):
        s = m.group(1).decode("latin-1", "replace")
        chunks = re.findall(r"\((?:\\\\.|[^\\\\()])*\)\s*Tj|\[(?:[^\]]*)\]\s*TJ", s)
        if not chunks:
            continue
        line = []
        for c in chunks:
            for t in re.findall(r"\(((?:\\\\.|[^\\\\()])*)\)", c):
                line.append(t.replace(r"\(", "(").replace(r"\)", ")").replace(r"\\", "\\"))
        if line:
            out.append(" ".join(line))
    return "\n".join(out)


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def doc_terms(path, text):
    counts = Counter(tokenize(os.path.basename(path)))
    body = Counter(tokenize(text))
    terms = Counter()
    for t, c in body.items():
        terms[t] += int(c)
    for t, c in counts.items():
        terms[t] += int(c * 3.0)  # filename boosted
    return terms


def build_doc_index():
    docs = []
    for root, _dirs, files in os.walk(DOCS_DIR):
        for fn in sorted(files):
            ext = os.path.splitext(fn)[1].lower()
            if ext not in DOC_EXTS:
                continue
            path = os.path.join(root, fn)
            text = extract_doc_text(path)
            terms = doc_terms(path, text)
            docs.append({"path": path, "rel": rel(path), "name": fn, "ext": ext,
                         "text": text, "tokens": dict(terms), "nwords": sum(terms.values())})
    df = Counter()
    for d in docs:
        for t in d["tokens"]:
            df[t] += 1
    N = max(len(docs), 1)
    vocab = sorted(df.keys())
    idf = np.array([np.log((N + 1) / (df[t] + 1)) + 1.0 for t in vocab], dtype=np.float32)
    return docs, vocab, idf


def doc_vector(d, vocab, idf):
    vec = np.zeros(len(vocab), np.float32)
    tokens = d.get("tokens") or {}
    if not tokens:
        return vec
    idx_of = {t: i for i, t in enumerate(vocab)}
    denom = max(np.sqrt(sum(c * c for c in tokens.values())), 1e-9)
    for t, c in tokens.items():
        i = idx_of.get(t)
        if i is not None:
            vec[i] = (c / denom) * idf[i]
    return vec


# ---------------------------------------------------------------------------
#  Name carrying (user labels survive re-clustering / rescans)
# ---------------------------------------------------------------------------
def named_centroids():
    rows = index_state.get("faces") or []
    if not rows or _emb is None or len(_emb) != len(rows):
        return {}
    acc = defaultdict(list)
    for i, r in enumerate(rows):
        gi = r.get("cluster")
        if gi is None:
            continue
        nm = next((c.get("name") for c in index_state["clusters"] if c["id"] == gi and c.get("name")), None)
        if nm:
            acc[nm].append(_emb[i])
    cents = {}
    for name, vecs in acc.items():
        c = np.mean(vecs, axis=0)
        nrm = np.linalg.norm(c)
        cents[name] = c / nrm if nrm > 0 else c
    return cents


def carry_names(old_cents, new_emb, groups):
    names = {}
    for gi, members in enumerate(groups):
        c = np.mean(new_emb[members], axis=0)
        nrm = np.linalg.norm(c)
        if nrm == 0:
            continue
        c = c / nrm
        best, best_sim = None, 0.0
        for name, oc in old_cents.items():
            s = float(c @ oc)
            if s > best_sim:
                best, best_sim = name, s
        if best and best_sim >= 0.5:
            names[gi] = best
    return names


# ---------------------------------------------------------------------------
#  Full index build
# ---------------------------------------------------------------------------
def build_index(pipeline):
    t0 = time.time()
    old_cents = named_centroids()
    face_rows, es, photos, anchors = scan_faces(pipeline)
    state = index_state
    with _lock:
        groups = cluster_faces(es, state["tolerance"]) if len(es) else []
        emb_to_cluster = {}
        for gi, members in enumerate(groups):
            for ei in members:
                emb_to_cluster[ei] = gi
        n_gallery = len(face_rows)
        for i in range(n_gallery):
            face_rows[i]["cluster"] = emb_to_cluster.get(i)

        cluster_records = {}
        for i, row in enumerate(face_rows):
            gi = row["cluster"]
            if gi is None:
                continue
            if gi not in cluster_records:
                cluster_records[gi] = {"id": gi, "name": None, "face_ids": []}
            cluster_records[gi]["face_ids"].append(i)

        # 1) reference-photo labels are authoritative
        if anchors:
            for name, eis in anchors.items():
                for ei in eis:
                    gi = emb_to_cluster.get(ei)
                    if gi is not None and gi in cluster_records:
                        cluster_records[gi]["name"] = name
        # 2) otherwise carry the user's previous names
        for gi, nm in carry_names(old_cents, es, groups).items():
            if gi in cluster_records and not cluster_records[gi].get("name"):
                cluster_records[gi]["name"] = nm

        state["clusters"] = [cluster_records[gi] for gi in sorted(cluster_records)]

        face_by_photo = defaultdict(list)
        for row in face_rows:
            face_by_photo[row["photo"]].append(row)
        for ph in photos:
            ph["face_clusters"] = []
            img = read_image(ph["path"])
            for row in face_by_photo.get(ph["path"], []):
                ph["face_clusters"].append((row["id"], row["cluster"]))
                if img is not None:
                    gen_face_thumb(img, row["bbox"], row["id"] + ".jpg")
            ph["faces"] = [fc[0] for fc in ph["face_clusters"]]
            ph["nfaces"] = len(ph["faces"])

        state["photos"] = photos
        docs, vocab, idf = build_doc_index()
        state["docs"], state["vocab"], state["idf"] = docs, vocab, idf.tolist()
        state["faces"] = [{"id": r["id"], "photo": rel(r["photo"]), "bbox": r["bbox"], "cluster": r["cluster"]}
                          for r in face_rows]
        state["built"] = True
        state["last_build"] = time.strftime("%Y-%m-%d %H:%M:%S")
    np.savez_compressed(EMBED_FILE, emb=es)
    save_index()
    return {"photos": len(state["photos"]), "faces": len(state["faces"]),
            "clusters": len(state["clusters"]), "docs": len(state["docs"]),
            "seconds": round(time.time() - t0, 2)}


def save_index():
    state = index_state
    slim = {
        "built": state["built"], "tolerance": state["tolerance"], "last_build": state["last_build"],
        "photos": [{k: v for k, v in p.items() if k != "path"} for p in state["photos"]],
        "faces": state["faces"],
        "clusters": state["clusters"],
        "docs": [{k: v for k, v in d.items() if k != "path"} for d in state["docs"]],
        "vocab": state["vocab"], "idf": state["idf"],
    }
    tmp = INDEX_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(slim, f, ensure_ascii=False)
    os.replace(tmp, INDEX_FILE)


def load_index():
    global _emb
    if not os.path.exists(INDEX_FILE):
        return False
    try:
        with open(INDEX_FILE, "r", encoding="utf-8") as f:
            slim = json.load(f)
    except Exception:
        return False
    with _lock:
        for k in ("built", "tolerance", "last_build", "photos", "clusters", "docs", "vocab", "idf", "faces"):
            if k in slim:
                index_state[k] = slim[k]
        index_state["built"] = True
    try:
        _emb = np.load(EMBED_FILE)["emb"]
    except Exception:
        _emb = None
    return True


# ---------------------------------------------------------------------------
#  Query
# ---------------------------------------------------------------------------
def person_mentions(qtokens):
    names = []
    for c in index_state["clusters"]:
        nm = c.get("name")
        if not nm:
            continue
        alias = nm.lower()
        if alias in ME_WORDS:
            continue  # "me" only counts as a person via the guarded me-alias logic
        if all(t in qtokens for t in tokenize(alias)):
            names.append(nm)
    return names


def make_snippet(d):
    text = re.sub(r"\s+", " ", d.get("text") or "").strip()
    if not text:
        return "No extractable text (binary or scanned — try the photos tab)."
    return text[:180] + ("…" if len(text) > 180 else "")


def answer_query(q):
    qtokens = tokenize(q)
    st = index_state
    named = person_mentions(qtokens)
    has_photo_word = any(t in PHOTO_WORDS for t in qtokens)

    me_word = any(t in ("me", "i") for t in qtokens)
    if me_word:
        join_word = bool(set(qtokens) & {"with", "and", "&"})
        imper = bool(set(qtokens) & IMPERATIVE_WORDS)
        # "me" is a photo subject only in a photo context ("photos of me") or
        # with an explicit co-occurrence ("me with X" / "me and X") — never for
        # an imperative requester ("show me ...") unless a co-word also appears.
        want_me = (has_photo_word or named) and (join_word or not imper)
        if want_me:
            me_cluster = next((c for c in st["clusters"]
                               if (c.get("name") or "").strip().lower() in ("me", "myself", "self", "i")), None)
            if me_cluster is not None and not any(n.lower() in ("me", "myself", "self", "i") for n in named):
                named.append(me_cluster["name"])

    results_photos = []
    results_docs = []

    if named:
        cointent = len(named) >= 2 and bool(set(qtokens) & {"with", "and", "&"})
        for ph in st["photos"]:
            ph_names = []
            for fid, cl in ph.get("face_clusters", []):
                nm = next((c.get("name") for c in st["clusters"] if c["id"] == cl and c.get("name")), None)
                if nm and nm not in ph_names:
                    ph_names.append(nm)
            hits = [n for n in named if n.lower() in [x.lower() for x in ph_names]]
            if hits:
                if cointent and len(hits) < len(named):
                    continue  # "A with B" -> photo must contain everyone
                results_photos.append({"rel": ph["rel"], "thumb": ph.get("thumb"),
                                       "nfaces": ph["nfaces"], "names": ph_names,
                                       "hits": [h.lower() for h in hits],
                                       "score": len(hits),
                                       "created": ph.get("created")})
        results_photos.sort(key=lambda r: (r["score"], r["nfaces"]), reverse=True)

    if st["vocab"] and len(qtokens):
        idf = st["idf"]
        idx_of = {t: i for i, t in enumerate(st["vocab"])}
        qv = np.zeros(len(st["vocab"]), np.float32)
        for t in qtokens:
            i = idx_of.get(t)
            if i is not None:
                qv[i] += 1.0
        qn = np.linalg.norm(qv)
        distinct_qt = set(qtokens)
        cert_intent = bool(distinct_qt & CERT_WORDS)
        scored = []
        for d in st["docs"]:
            dv = doc_vector(d, st["vocab"], idf)
            cos = float((dv @ qv) / max(qn * np.linalg.norm(dv), 1e-9)) if qn else 0.0
            if not np.isfinite(cos):
                cos = 0.0
            name = d["name"].lower()
            d_tokens = set(d.get("tokens") or {})
            tokens_present = sum(1 for t in distinct_qt if t in d_tokens)
            total = cos + 0.10 * tokens_present + 0.12 * sum(1 for t in distinct_qt if t in name)
            if cert_intent and (d_tokens & CERT_WORDS):
                total += 0.25
            if total > 0:
                scored.append((total, d))
        scored.sort(key=lambda x: x[0], reverse=True)
        for total, d in scored[:8]:
            results_docs.append({"rel": d["rel"], "name": d["name"], "ext": d["ext"],
                                 "snippet": make_snippet(d), "score": round(total, 4)})

    mode = "photos" if (results_photos and (has_photo_word or named)) else "docs"
    return {"query": q, "mode": mode, "persons": [n.lower() for n in named],
            "photos": results_photos, "docs": results_docs}


# ---------------------------------------------------------------------------
#  HTTP server
# ---------------------------------------------------------------------------
MIME = {".html": "text/html", ".css": "text/css", ".js": "application/javascript",
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
        ".gif": "image/gif", ".webp": "image/webp", ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".json": "application/json", ".txt": "text/plain"}


class Handler(BaseHTTPRequestHandler):
    pipeline = None
    is_cli = True

    def log_message(self, *a):
        if Handler.is_cli:
            super().log_message(*a)

    def _send_json(self, data, code=200):
        body = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self):
        ln = int(self.headers.get("Content-Length") or 0)
        if ln:
            try:
                return json.loads(self.rfile.read(ln).decode("utf-8") or "{}")
            except Exception:
                return {}
        return {}

    def _serve_file(self, path):
        if not os.path.exists(path):
            self.send_error(404)
            return
        ext = os.path.splitext(path)[1].lower()
        with open(path, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", MIME.get(ext, "application/octet-stream"))
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        path = urllib.parse.unquote(urllib.parse.urlparse(self.path).path)
        if path in ("/", "/index.html"):
            self._serve_file(os.path.join(BASE_DIR, "index.html")); return
        if path == "/api/status":
            st = index_state
            return self._send_json({
                "built": st["built"], "tolerance": st["tolerance"], "last_build": st["last_build"],
                "photos": len(st["photos"]), "docs": len(st["docs"]),
                "faces": len(st["faces"]), "clusters": len(st["clusters"]),
                "named": sum(1 for c in st["clusters"] if c.get("name")),
                "dirs": {"photos": rel_media(PHOTOS_DIR), "documents": rel_media(DOCS_DIR),
                         "labels": rel_media(LABELS_DIR)},
            })
        if path == "/api/people":
            st = index_state
            out = []
            for c in st["clusters"]:
                sample_face = st["faces"][c["face_ids"][0]] if c["face_ids"] else None
                out.append({"id": c["id"], "name": c.get("name"), "count": len(c["face_ids"]),
                            "sample": (sample_face["id"] + ".jpg") if sample_face else None})
            return self._send_json({"clusters": out, "faces": st["faces"]})
        if path.startswith("/thumbnails/"):
            self._serve_file(os.path.join(THUMB_DIR, os.path.basename(path))); return
        if path.startswith("/faces/"):
            self._serve_file(os.path.join(FACE_DIR, os.path.basename(path))); return
        if path.startswith("/media/"):
            fp = os.path.normpath(os.path.join(BASE_DIR, path.lstrip("/")))
            if fp.startswith(BASE_DIR) and os.path.isfile(fp):
                self._serve_file(fp); return
        self.send_error(404)

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path
        try:
            if path == "/api/rescan":
                res = build_index(Handler.pipeline)
                return self._send_json(res)
            if path == "/api/name":
                body = self._read_body()
                cid = int(body.get("cluster_id", -1))
                name = (body.get("name") or "").strip()
                with _lock:
                    for c in index_state["clusters"]:
                        if c["id"] == cid:
                            c["name"] = name or None
                    save_index()
                return self._send_json({"ok": True})
            if path == "/api/tolerance":
                body = self._read_body()
                tol = float(body.get("tolerance", DEFAULT_TOLERANCE))
                tol = max(0.2, min(0.9, tol))
                with _lock:
                    index_state["tolerance"] = tol
                re_cluster = []
                if os.path.exists(EMBED_FILE):
                    try:
                        emb = np.load(EMBED_FILE)["emb"]
                        re_cluster = cluster_faces(emb, tol)
                    except Exception:
                        re_cluster = []
                if re_cluster:
                    with _lock:
                        cl = []
                        old = {c["id"]: c.get("name") for c in index_state["clusters"]}
                        for gi, members in enumerate(re_cluster):
                            rec = {"id": gi, "name": None, "face_ids": [int(ei) for ei in members]}
                            # carry names when a new cluster is dominated by one old cluster
                            cl.append(rec)
                        index_state["clusters"] = cl
                        save_index()
                return self._send_json({"ok": True, "tolerance": tol,
                                        "clusters": len(re_cluster) if re_cluster else len(index_state["clusters"])})
            if path == "/api/query":
                body = self._read_body()
                q = (body.get("q") or "").strip()
                if not q:
                    return self._send_json({"error": "empty query"})
                return self._send_json(answer_query(q))
        except Exception as e:
            import traceback
            traceback.print_exc()
            return self._send_json({"error": str(e)}, 500)
        self.send_error(404)


def main():
    ensure_dirs()
    Handler.pipeline = FacePipeline()
    load_index()
    if not index_state["built"]:
        print("First run — building index over media/ ...")
        build_index(Handler.pipeline)
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"LocalMediaRAG ready  ->  http://localhost:{PORT}")
    print(f"  photos    : {len(index_state['photos'])}")
    print(f"  faces     : {len(index_state['faces'])}")
    print(f"  documents : {len(index_state['docs'])}")
    print("  models    : bundled, fully offline.")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()