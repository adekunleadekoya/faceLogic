import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk
import faiss
import insightface
from insightface.app import FaceAnalysis


# ─── Global state ────────────────────────────────────────────────────────────
PICS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pics")

face_app: FaceAnalysis = None
embeddings: list[np.ndarray] = []   # parallel to `labels`
labels: list[str] = []              # filenames
index: faiss.IndexFlatIP = None     # inner-product (cosine) FAISS index

# Number of parallel workers for index building.
# ONNX Runtime releases the GIL during inference, so these threads run truly in parallel.
_INDEX_WORKERS = min(os.cpu_count() or 1, 8)


# ─── InsightFace helpers ──────────────────────────────────────────────────────

def load_model() -> FaceAnalysis:
    """Load InsightFace buffalo_l model (downloads on first run).

    intra/inter_op_num_threads are set to 1 so that ONNX Runtime does not
    internally parallelise a single inference call. Core distribution is
    handled externally by ThreadPoolExecutor in build_index(), which means
    N worker threads each occupy 1 core cleanly with no over-subscription.
    """
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1   # one thread per ONNX op
    opts.inter_op_num_threads = 1   # one thread across ops
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"],
                       session_options=opts)
    app.prepare(ctx_id=0, det_thresh=0.3, det_size=(640, 640))
    return app


def get_embedding(img_path: str) -> np.ndarray | None:
    """Return the L2-normalised 512-d embedding for the first face found."""
    img = cv2.imdecode(
        np.frombuffer(open(img_path, "rb").read(), np.uint8),
        cv2.IMREAD_COLOR
    )
    if img is None:
        print(f"[debug] cv2 could not decode image: {img_path}")
        return None

    # Resize large images so close-cropped faces fall within the detector's anchor range
    max_side = 640
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    print(f"[debug] Image loaded OK: {img.shape}  path={img_path}")
    faces = face_app.get(img)
    print(f"[debug] Faces detected: {len(faces)}")
    if not faces:
        return None
    emb = faces[0].normed_embedding          # already unit-norm
    return emb.astype(np.float32)


# ─── Index building ───────────────────────────────────────────────────────────

def _embed_one(fname: str) -> tuple[str, np.ndarray | None]:
    """Worker: decode and embed a single image. Safe to call from multiple threads
    because ONNX Runtime's InferenceSession.run() releases the GIL and is thread-safe."""
    path = os.path.join(PICS_DIR, fname)
    return fname, get_embedding(path)


def build_index(progress_cb=None) -> tuple[list, list, faiss.IndexFlatIP]:
    """Read every image in PICS_DIR in parallel, extract embeddings, build FAISS index.

    Args:
        progress_cb: optional callable(done: int, total: int) called after each
                     image completes — suitable for updating a progress indicator.
    """
    supported = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = [
        f for f in os.listdir(PICS_DIR)
        if os.path.splitext(f)[1].lower() in supported
    ]

    total = len(files)
    done = 0
    # Map filename -> embedding, populated as futures complete (out-of-order)
    result_map: dict[str, np.ndarray | None] = {}

    with ThreadPoolExecutor(max_workers=_INDEX_WORKERS) as pool:
        futures = {pool.submit(_embed_one, fname): fname for fname in files}
        for future in as_completed(futures):
            fname, emb = future.result()
            result_map[fname] = emb
            done += 1
            if progress_cb:
                progress_cb(done, total)

    # Re-order results to match the original file listing (deterministic index order)
    embs: list[np.ndarray] = []
    names: list[str] = []
    for fname in files:
        emb = result_map.get(fname)
        if emb is not None:
            embs.append(emb)
            names.append(fname)

    if embs:
        matrix = np.vstack(embs)            # (N, 512)
        idx = faiss.IndexFlatIP(matrix.shape[1])
        idx.add(matrix)
    else:
        idx = faiss.IndexFlatIP(512)

    return embs, names, idx


# ─── Matching ─────────────────────────────────────────────────────────────────

def find_match(query_path: str, top_k: int = 5) -> tuple[str | None, list[tuple[str, float]]]:
    """Return (error_msg, results). error_msg is None on success."""
    emb = get_embedding(query_path)
    if emb is None:
        print(f"[debug] No face detected in query image: {query_path}")
        return "No face detected in the query image.", []
    print(f"[debug] Query embedding extracted OK. Index size: {index.ntotal}")
    if index.ntotal == 0:
        return "Index is empty — no faces were found in the pics folder.", []
    q = emb.reshape(1, -1)
    k = min(top_k, index.ntotal)
    scores, idxs = index.search(q, k)
    results = []
    for score, i in zip(scores[0], idxs[0]):
        if i >= 0:
            print(f"[debug] match: {labels[i]}  score={score:.4f}")
            results.append((labels[i], float(score)))
    return None, results


# ─── UI ───────────────────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Face Match — InsightFace + FAISS")
        self.resizable(True, True)
        self.geometry("860x620")
        self._build_ui()
        self._status("Loading InsightFace model and building index…")
        threading.Thread(target=self._init_backend, daemon=True).start()

    # ── layout ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Top toolbar
        toolbar = tk.Frame(self, pady=6)
        toolbar.pack(fill=tk.X, padx=10)

        self.btn_rebuild = tk.Button(
            toolbar, text="↺  Rebuild Index",
            command=self._rebuild, state=tk.DISABLED, width=18
        )
        self.btn_rebuild.pack(side=tk.LEFT, padx=4)

        self.btn_search = tk.Button(
            toolbar, text="🔍  Select Query Image",
            command=self._search, state=tk.DISABLED, width=22
        )
        self.btn_search.pack(side=tk.LEFT, padx=4)

        self.lbl_index = tk.Label(toolbar, text="Index: –", anchor="w")
        self.lbl_index.pack(side=tk.LEFT, padx=12)

        # Main pane: query image (left) + results (right)
        pane = tk.PanedWindow(self, orient=tk.HORIZONTAL, sashwidth=6)
        pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)

        # Left: query image
        left = tk.LabelFrame(pane, text="Query Image", padx=4, pady=4)
        pane.add(left, minsize=280)
        self.query_canvas = tk.Label(
            left, text="(no image selected)",
            width=36, height=18, relief=tk.SUNKEN, bg="#1e1e1e", fg="#aaa"
        )
        self.query_canvas.pack(fill=tk.BOTH, expand=True)
        self.lbl_query_name = tk.Label(left, text="", wraplength=260)
        self.lbl_query_name.pack()

        # Right: results grid
        right = tk.LabelFrame(pane, text="Top Matches", padx=4, pady=4)
        pane.add(right, minsize=420)

        # Scrollable canvas for image cards
        self._results_canvas = tk.Canvas(right, bg="#2b2b2b", highlightthickness=0)
        vsb = ttk.Scrollbar(right, orient=tk.VERTICAL, command=self._results_canvas.yview)
        self._results_canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self._results_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._cards_frame = tk.Frame(self._results_canvas, bg="#2b2b2b")
        self._cards_window = self._results_canvas.create_window(
            (0, 0), window=self._cards_frame, anchor="nw"
        )
        self._cards_frame.bind("<Configure>", self._on_cards_configure)
        self._results_canvas.bind("<Configure>", self._on_canvas_resize)
        self._results_canvas.bind("<MouseWheel>", lambda e: self._results_canvas.yview_scroll(-1 * (e.delta // 120), "units"))

        self._card_images: list = []   # keep PhotoImage references alive

        # Status bar
        self.status_var = tk.StringVar(value="Initialising…")
        tk.Label(self, textvariable=self.status_var, anchor="w",
                 relief=tk.SUNKEN, padx=6).pack(
            side=tk.BOTTOM, fill=tk.X
        )

    # ── backend init ──────────────────────────────────────────────────────────

    def _init_backend(self):
        global face_app, embeddings, labels, index
        import time
        try:
            self.after(0, lambda: self._status("Loading InsightFace model…"))
            t0 = time.perf_counter()
            face_app = load_model()
            t1 = time.perf_counter()
            model_secs = t1 - t0
            self.after(0, lambda: self._status(f"Model loaded in {model_secs:.2f}s — building index…"))

            t2 = time.perf_counter()

            def _index_progress(done, total):
                self.after(0, lambda: self._status(
                    f"Building index… {done}/{total} images  "
                    f"({_INDEX_WORKERS} parallel workers)"
                ))

            embeddings, labels, index = build_index(progress_cb=_index_progress)
            t3 = time.perf_counter()
            index_secs = t3 - t2

            print(f"[timing] model load:  {model_secs:.2f}s")
            print(f"[timing] index build: {index_secs:.2f}s")
            print(f"[timing] total:       {t3 - t0:.2f}s")

            self.after(0, self._on_ready)
        except Exception as exc:
            self.after(0, lambda: self._status(f"Error: {exc}"))
            self.after(0, lambda: messagebox.showerror("Startup error", str(exc)))

    def _on_ready(self):
        n = index.ntotal if index else 0
        self.lbl_index.config(text=f"Index: {n} face(s) from {len(labels)} image(s)")
        self.btn_rebuild.config(state=tk.NORMAL)
        self.btn_search.config(state=tk.NORMAL)
        self._status("Ready.")

    # ── callbacks ─────────────────────────────────────────────────────────────

    def _rebuild(self):
        self._status("Rebuilding index…")
        self.btn_rebuild.config(state=tk.DISABLED)
        self.btn_search.config(state=tk.DISABLED)
        threading.Thread(target=self._do_rebuild, daemon=True).start()

    def _do_rebuild(self):
        global embeddings, labels, index
        try:
            def _rebuild_progress(done, total):
                self.after(0, lambda: self._status(
                    f"Rebuilding index… {done}/{total} images  "
                    f"({_INDEX_WORKERS} parallel workers)"
                ))
            embeddings, labels, index = build_index(progress_cb=_rebuild_progress)
            self.after(0, self._on_ready)
        except Exception as exc:
            self.after(0, lambda: self._status(f"Rebuild error: {exc}"))

    def _search(self):
        path = filedialog.askopenfilename(
            title="Select query image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")]
        )
        if not path:
            return
        self._show_query_image(path)
        self.lbl_query_name.config(text=os.path.basename(path))
        self._status("Searching…")
        threading.Thread(target=self._do_search, args=(path,), daemon=True).start()

    def _do_search(self, path: str):
        try:
            err, results = find_match(path, top_k=10)
            self.after(0, lambda: self._show_results(err, results, path))
        except Exception as exc:
            self.after(0, lambda: self._status(f"Search error: {exc}"))

    def _show_results(self, err: str | None, results: list, query_path: str):
        # Clear existing cards
        for w in self._cards_frame.winfo_children():
            w.destroy()
        self._card_images.clear()

        if err:
            self._status(f"Error: {err}")
            tk.Label(self._cards_frame, text=err, bg="#2b2b2b", fg="#ff6b6b",
                     wraplength=380, justify=tk.LEFT).pack(padx=12, pady=12)
            return
        if not results:
            self._status("No results returned.")
            return

        THUMB = 160
        COLS = 3

        for rank, (fname, score) in enumerate(results, 1):
            col = (rank - 1) % COLS
            row = (rank - 1) // COLS

            card = tk.Frame(self._cards_frame, bg="#3c3c3c", bd=1, relief=tk.RIDGE)
            card.grid(row=row, column=col, padx=6, pady=6, sticky="nsew")

            img_path = os.path.join(PICS_DIR, fname)
            lbl_img = tk.Label(card, bg="#3c3c3c")
            lbl_img.pack(padx=4, pady=(6, 2))
            try:
                img = Image.open(img_path)
                img.thumbnail((THUMB, THUMB), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                lbl_img.config(image=photo)
                lbl_img.image = photo
                self._card_images.append(photo)
            except Exception:
                lbl_img.config(text="?", width=10, height=6, fg="#aaa")

            tk.Label(card, text=f"#{rank}  {score:.4f}", bg="#3c3c3c", fg="#7ec8e3",
                     font=("Consolas", 10, "bold")).pack()
            tk.Label(card, text=fname, bg="#3c3c3c", fg="#cccccc",
                     font=("TkDefaultFont", 8), wraplength=THUMB + 10).pack(padx=4, pady=(0, 6))

        for c in range(COLS):
            self._cards_frame.columnconfigure(c, weight=1)

        self._status(f"Done. {len(results)} match(es) returned.")

    def _on_cards_configure(self, _event):
        self._results_canvas.configure(scrollregion=self._results_canvas.bbox("all"))

    def _on_canvas_resize(self, event):
        self._results_canvas.itemconfig(self._cards_window, width=event.width)

    # ── image helpers ─────────────────────────────────────────────────────────

    def _show_query_image(self, path: str):
        self._show_preview(path, self.query_canvas, size=(280, 280))

    def _show_preview(self, path: str, widget: tk.Label, size: tuple):
        try:
            img = Image.open(path)
            img.thumbnail(size, Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            widget.config(image=photo, text="")
            widget.image = photo          # keep reference
        except Exception:
            widget.config(text="(could not load image)", image="")

    def _status(self, msg: str):
        self.status_var.set(msg)
        self.update_idletasks()


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(PICS_DIR, exist_ok=True)
    app = App()
    app.mainloop()
