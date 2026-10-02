"""Precompute response embeddings for `pair_coverage --embedding_cache_path`.

Embeds every response of every model on every pool item of each requested
dataset, once, and writes them in the cache format read by
`jhu_ta1.magnet._embedding_cache.CachedResponseEmbeddings`:

    <cache_dpath>/index.json          provider, model, dimensions, per-item records
    <cache_dpath>/chunks/<hash>.npy   float arrays, one row per embedded response

The cache is incremental. Datasets already complete in the index are skipped, so
a dataset can be added to the manifest and this command rerun. Chunk files are
named by a hash of their provider, model and input texts, so an interrupted run
resumes without re-embedding finished chunks. The index is rewritten, atomically,
only when a whole dataset is done.
"""

import argparse
import contextlib
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np

from jhu_ta1.magnet._embedding_cache import response_hash
from jhu_ta1.magnet._helm_pair_data import file_hash, load_panel

FORMAT_VERSION = 1
CHUNK_SIZE = 50
PROVIDER = "sentence-transformers"
DEFAULT_MODEL = "nomic-ai/nomic-embed-text-v2-moe"


def _sha256(text):
    return hashlib.sha256(text.encode()).hexdigest()


def chunk_name(provider, model, texts):
    """Name a chunk by what produced it, so identical inputs reuse the same file."""
    payload = json.dumps([provider, model, texts], separators=(",", ":"))
    return f"{_sha256(payload)}.npy"


def pool_entries(panel):
    """Yield (model, item_id, item_index) in the order texts are embedded."""
    item_ids = [json.loads(key)[1] for key in panel.item_ids]
    if len(set(item_ids)) != len(item_ids):
        raise ValueError(
            f"{panel.dataset}: item ids repeat across splits, which the cache "
            "format cannot index"
        )
    for model in panel.models:
        for index, item_id in enumerate(item_ids):
            yield model, item_id, index


def load_index(cache_dpath, provider, model):
    path = Path(cache_dpath) / "index.json"
    if not path.is_file():
        return {
            "format_version": FORMAT_VERSION,
            "provider": provider,
            "model": model,
            "dimensions": None,
            "datasets": {},
            "chunk_sha256": {},
            "text_preprocessing": "str(response), embedded as-is with the passage prompt",
        }
    index = json.loads(path.read_text())
    if index["format_version"] != FORMAT_VERSION:
        raise ValueError(f"Unsupported embedding cache format in {path}")
    if index["provider"] != provider or index["model"] != model:
        raise ValueError(
            f"{path} holds {index['provider']} / {index['model']} embeddings; "
            f"refusing to mix in {provider} / {model}. Use a new cache directory."
        )
    return index


def write_index(cache_dpath, index):
    path = Path(cache_dpath) / "index.json"
    scratch = path.with_suffix(".json.tmp")
    scratch.write_text(json.dumps(index) + "\n")
    os.replace(scratch, path)


def is_complete(cache_dpath, index, panel):
    """True when the index already covers this exact panel, byte for byte."""
    records = index["datasets"].get(panel.dataset)
    if records is None:
        return False
    for model, item_id, position in pool_entries(panel):
        record = records.get(model, {}).get(item_id)
        if record is None or record["response_sha256"] != response_hash(
            panel.responses[model][position]
        ):
            return False
        if record["chunk"] not in index["chunk_sha256"]:
            return False
        if not (Path(cache_dpath) / "chunks" / record["chunk"]).is_file():
            return False
    return True


def _save_chunk(path, vectors):
    scratch = path.with_suffix(".tmp")
    with scratch.open("wb") as stream:
        np.save(stream, vectors, allow_pickle=False)
    os.replace(scratch, path)


def embed_panel(panel, cache_dpath, index, embed, *, provider, model, chunk_size):
    """Embed one panel into the cache and record it in `index` (in memory)."""
    entries = list(pool_entries(panel))
    texts = [str(panel.responses[m][i]) for m, _, i in entries]
    chunks_dpath = Path(cache_dpath) / "chunks"
    records, hashes = {}, {}
    dimensions = index["dimensions"]
    embedded, started, next_percent = 0, time.monotonic(), 1

    for start in range(0, len(texts), chunk_size):
        chunk_texts = texts[start : start + chunk_size]
        name = chunk_name(provider, model, chunk_texts)
        path = chunks_dpath / name
        if not path.is_file():
            vectors = np.asarray(embed(chunk_texts))
            if vectors.shape[0] != len(chunk_texts) or vectors.ndim != 2:
                raise ValueError(
                    f"Embedder returned {vectors.shape} for {len(chunk_texts)} texts"
                )
            if not np.isfinite(vectors).all():
                raise ValueError(f"{panel.dataset}: non-finite embeddings")
            _save_chunk(path, vectors)
            embedded += len(chunk_texts)
        vectors = np.load(path, mmap_mode="r", allow_pickle=False)
        if vectors.shape != (len(chunk_texts), vectors.shape[-1]):
            raise ValueError(f"Chunk {name} has the wrong shape {vectors.shape}")
        if dimensions is None:
            dimensions = int(vectors.shape[1])
        elif vectors.shape[1] != dimensions:
            raise ValueError(
                f"Chunk {name} has {vectors.shape[1]} dimensions, expected {dimensions}"
            )
        hashes[name] = file_hash(path)
        for offset, (entry, text) in enumerate(
            zip(entries[start : start + chunk_size], chunk_texts)
        ):
            entry_model, item_id, position = entry
            records.setdefault(entry_model, {})[item_id] = {
                "response_sha256": response_hash(panel.responses[entry_model][position]),
                "embedding_input_sha256": _sha256(text),
                "chunk": name,
                "row": offset,
            }

        done = start + len(chunk_texts)
        percent = 100 * done // len(texts)
        if percent >= next_percent or done == len(texts):
            elapsed = time.monotonic() - started
            rate = embedded / elapsed if elapsed and embedded else 0.0
            remaining = (len(texts) - done) / rate / 60 if rate else 0.0
            print(
                f"{panel.dataset}: [{percent:3d}%] {done}/{len(texts)} responses | "
                f"{embedded} embedded this run"
                + (f" at {rate:.1f}/s, ~{remaining:.1f}m left" if rate else ""),
                flush=True,
            )
            next_percent = percent + 1

    index["dimensions"] = dimensions
    index["datasets"][panel.dataset] = records
    index["chunk_sha256"].update(hashes)


def precompute(
    suite_path,
    manifest,
    cache_dpath,
    embed_factory,
    uses_onehot,
    *,
    provider=PROVIDER,
    model=DEFAULT_MODEL,
    datasets=None,
    chunk_size=CHUNK_SIZE,
    force=False,
):
    """Embed the requested datasets. `embed_factory()` builds the embedder lazily."""
    entries = manifest["datasets"]
    names = [entry["dataset"] for entry in entries]
    if not names or len(names) != len(set(names)):
        raise ValueError("Dataset names must be nonempty and unique")
    if datasets:
        unknown = sorted(set(datasets) - set(names))
        if unknown:
            raise ValueError(f"Not in the manifest: {', '.join(unknown)}")
        entries = [entry for entry in entries if entry["dataset"] in datasets]

    cache_dpath = Path(cache_dpath)
    (cache_dpath / "chunks").mkdir(parents=True, exist_ok=True)
    index = load_index(cache_dpath, provider, model)
    embed = None
    summary = {"embedded": [], "cached": [], "onehot": []}

    for entry in entries:
        dataset = entry["dataset"]
        if uses_onehot(dataset):
            print(f"{dataset}: one-hot embeddings, nothing to cache", flush=True)
            summary["onehot"].append(dataset)
            continue
        panel = load_panel(suite_path, entry)
        if not force and is_complete(cache_dpath, index, panel):
            print(f"{dataset}: already cached", flush=True)
            summary["cached"].append(dataset)
            continue
        if embed is None:
            embedder = embed_factory()
            embed = lambda texts: embedder.embed(texts, model)  # noqa: E731
        embed_panel(
            panel,
            cache_dpath,
            index,
            embed,
            provider=provider,
            model=model,
            chunk_size=chunk_size,
        )
        write_index(cache_dpath, index)
        summary["embedded"].append(dataset)
        print(f"{dataset}: cache updated ({cache_dpath})", flush=True)
    return summary


def _load_dkps():
    """Import dkps from a scratch directory.

    dkps.embed creates ./.cache/embed/* relative to the working directory at
    import time. That must not land in the checkout.
    """
    with tempfile.TemporaryDirectory() as scratch, contextlib.chdir(scratch):
        from dkps.embed import SentenceTransformersClient
        from dkps.helm import uses_onehot
    return SentenceTransformersClient, uses_onehot


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--helm_suite_path", required=True)
    parser.add_argument("--dataset_manifest", required=True)
    parser.add_argument(
        "--cache_dpath",
        required=True,
        help="Cache directory; created if missing, extended if it exists",
    )
    parser.add_argument("--embed_provider", default=PROVIDER, choices=[PROVIDER])
    parser.add_argument("--embed_model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Only these manifest datasets (default: all of them)",
    )
    parser.add_argument("--chunk_size", type=int, default=CHUNK_SIZE)
    parser.add_argument(
        "--force", action="store_true", help="Re-embed datasets already cached"
    )
    args = parser.parse_args(argv)
    if args.chunk_size < 1:
        parser.error("chunk_size must be positive")

    manifest = json.loads(Path(args.dataset_manifest).read_text())
    client_class, uses_onehot = _load_dkps()
    summary = precompute(
        Path(args.helm_suite_path).resolve(),
        manifest,
        Path(args.cache_dpath).resolve(),
        client_class,
        uses_onehot,
        provider=args.embed_provider,
        model=args.embed_model,
        datasets=args.datasets,
        chunk_size=args.chunk_size,
        force=args.force,
    )
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
