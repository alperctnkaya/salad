"""
OpenHotels evaluation pipeline.

Uses local JSON metadata + on-disk images (no HuggingFace dependency).
Computes gallery and query embeddings, runs FAISS retrieval, and prints
Recall@K metrics.

Results are saved to a timestamped directory under eval_results/ including:
  - metadata.json  : full model config, checkpoint path, recall metrics, timestamps
  - predictions_<split>.npz : retrieved indices and gallery hotel IDs per query
  - embeddings_<split>.npz  : raw embeddings (optional, --save_embeddings)
"""

import argparse
import json
import os
import time
from datetime import datetime

import faiss
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloaders.OpenHotelsDataset import OpenHotelsDataset, tensor_collate
from dataloaders.config import (
    GALLERY_METADATA,
    IMAGE_ROOT,
    TEST_NON_OBJECT_METADATA,
    TEST_OBJECT_METADATA,
)
from vpr_model import VPRModel

# ============================
# CONFIGURATION
# ============================

BATCH_SIZE = 128
NUM_WORKERS = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K_METRICS = [1, 5, 10, 100]
SAVE_PREDICTIONS_K = 100  # Number of predictions to save per query
SEARCH_BATCH_SIZE = 256  # Number of queries to search at once (avoids OOM / timeouts)
DEFAULT_OUTPUT_ROOT = "eval_results"

# Standard ImageNet mean/std
IMAGENET_MEAN_STD = {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}


from typing import Dict, List


def compute_recalls(
    query_ids: np.ndarray,
    gallery_ids: np.ndarray,
    retrieved_indices: np.ndarray,
    top_k_list: List[int],
) -> Dict[int, float]:
    recalls = {k: 0 for k in top_k_list}
    n_queries = len(query_ids)

    for i in range(n_queries):
        true_id = query_ids[i]
        retrieved_ids = gallery_ids[retrieved_indices[i]]

        for k in top_k_list:
            if true_id in retrieved_ids[:k]:
                recalls[k] += 1

    return {k: (count / n_queries) * 100 for k, count in recalls.items()}


def format_results_table(
    model_name: str,
    split_name: str,
    recalls: Dict[int, float],
) -> str:
    width = 40
    lines = [
        "=" * width,
        f"  {model_name.upper()} — {split_name}",
        "=" * width,
    ]
    for k, acc in sorted(recalls.items()):
        lines.append(f"  Recall@{k:<4}  {acc:>7.2f}%")
    lines.append("=" * width)
    return "\n".join(lines)


class EvaluationPipeline:
    def __init__(
        self,
        checkpoint_path,
        model_config=None,
        output_dir=None,
        save_embeddings=False,
        hf_repo=None,
    ):
        print(f"🚀 Initializing Benchmark on {DEVICE}...")

        self.checkpoint_path = os.path.abspath(checkpoint_path)
        self.hf_repo = hf_repo
        self.save_embeddings = save_embeddings

        # Default config if none provided (matches main.py)
        if model_config is None:
            model_config = {
                "backbone_arch": "dinov2_vitb14",
                "backbone_config": {
                    "num_trainable_blocks": 4,
                    "return_token": False,
                    "return_attention": False,
                    "norm_layer": True,
                },
                "agg_arch": "salad",
                "agg_config": {
                    "num_channels": 768,
                    "num_clusters": 64,
                    "cluster_dim": 128,
                    "token_dim": 256,
                },
                "loss_name": "MultiSimilarityLoss",
                "miner_name": None,
                "miner_margin": 0.1,
                "faiss_gpu": False,
            }

        self.model_config = model_config

        # ── Setup output directory ────────────────────────────────────
        # When loading from HF, use a local directory based on the repo name;
        # otherwise, eval results live next to the checkpoint file.
        if output_dir is None:
            if hf_repo:
                repo_tag = hf_repo.replace("/", "_")
                ckpt_stem = os.path.splitext(os.path.basename(self.checkpoint_path))[0]
                output_dir = os.path.join(DEFAULT_OUTPUT_ROOT, f"{repo_tag}_{ckpt_stem}")
            else:
                ckpt_dir = os.path.dirname(self.checkpoint_path)
                ckpt_stem = os.path.splitext(os.path.basename(self.checkpoint_path))[0]
                output_dir = os.path.join(ckpt_dir, f"eval_{ckpt_stem}")
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"📁 Results will be saved to: {self.output_dir}")

        # ── Filter config to only VPRModel constructor args ───────────
        import inspect
        valid_keys = set(inspect.signature(VPRModel.__init__).parameters.keys()) - {"self"}
        filtered_config = {k: v for k, v in model_config.items() if k in valid_keys}
        self.model = VPRModel(**filtered_config)

        # ── Load Checkpoint ───────────────────────────────────────────
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

        # Support multiple checkpoint formats:
        #   1. {"model_state_dict": ...}  — custom training loop
        #   2. {"state_dict": ...}        — PyTorch Lightning save_weights_only=True
        #   3. raw state dict (OrderedDict with tensor values)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict) and all(
            isinstance(v, torch.Tensor) for v in list(checkpoint.values())[:5]
        ):
            # Raw state dict
            state_dict = checkpoint
        else:
            raise ValueError(
                f"Unrecognised checkpoint format. "
                f"Top-level keys: {list(checkpoint.keys())[:10] if isinstance(checkpoint, dict) else type(checkpoint)}"
            )

        # Strip 'module.' prefix (DataParallel) if present
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v

        self.model.load_state_dict(new_state_dict)
        self.model.to(DEVICE)
        self.model.eval()

        # Transforms
        res = (224, 224) if self.model_config.get("agg_arch", "").lower() == "mixvpr" else (322, 322)
        self.transform = T.Compose(
            [
                T.Resize(res, interpolation=T.InterpolationMode.BILINEAR),
                T.ToTensor(),
                T.Normalize(
                    mean=IMAGENET_MEAN_STD["mean"], std=IMAGENET_MEAN_STD["std"]
                ),
            ]
        )

    # -----------------------------------------------------------------

    @torch.no_grad()
    def compute_embeddings(self, metadata_path: str, desc: str = "Embedding"):
        """
        Compute normalised embeddings for every image described in
        *metadata_path*.

        Returns:
            embeddings : ``np.ndarray[N, D]``
            hotel_ids  : ``np.ndarray[N]``  (string hotel IDs)
        """

        dataset = OpenHotelsDataset(
            metadata_path=metadata_path,
            image_root=IMAGE_ROOT,
            transform=self.transform,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            shuffle=False,
            pin_memory=True,
            collate_fn=tensor_collate,
        )

        embeddings = []
        hotel_ids = []

        print(f"🔄 Computing embeddings for {desc} ({len(dataset)} images)...")

        for images, batch_hotel_ids in tqdm(dataloader, desc=desc):
            images = images.to(DEVICE)

            with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
                descriptors = self.model(images)

            # L2 Normalise
            if descriptors.ndim == 3:
                descriptors = F.normalize(descriptors, p=2, dim=2)
            else:
                descriptors = F.normalize(descriptors, p=2, dim=1)

            embeddings.append(descriptors.cpu().numpy())
            hotel_ids.extend(batch_hotel_ids)

        embeddings = np.vstack(embeddings)
        hotel_ids = np.array(hotel_ids, dtype=np.int64)

        return embeddings, hotel_ids

    # -----------------------------------------------------------------

    def _evaluate(
        self, index, query_emb, query_ids, gallery_ids, query_name, gallery_emb=None
    ):
        """Run FAISS retrieval or MaxSim and print Recall@K."""
        # ── Search ────────────────────────────────────────────────────
        print(
            f"⚡ Searching Top-{SAVE_PREDICTIONS_K} candidates for {len(query_ids)} queries..."
        )
        t1 = time.time()

        n_queries = query_emb.shape[0]
        I_list = []

        if query_emb.ndim == 3 and gallery_emb is not None:
            # MaxSim Search
            query_tensor = torch.tensor(query_emb, device=DEVICE)
            batch_q = 64
            batch_g = 1024
            G = gallery_emb.shape[0]

            for start in tqdm(
                range(0, n_queries, batch_q), desc=f"MaxSim Search {query_name}"
            ):
                end = min(start + batch_q, n_queries)
                q_batch = query_tensor[start:end]

                scores_batch = []
                for j in range(0, G, batch_g):
                    g_batch = torch.tensor(gallery_emb[j : j + batch_g], device=DEVICE)
                    # sim: [bq, V, bg, V]
                    sim = torch.einsum("qid,rjd->qirj", q_batch, g_batch)
                    max_sim = sim.max(dim=3).values  # [bq, V, bg]
                    score = max_sim.sum(dim=1)  # [bq, bg]
                    scores_batch.append(score)

                sim_scores = torch.cat(scores_batch, dim=1)
                topk = torch.topk(sim_scores, SAVE_PREDICTIONS_K, dim=1)
                I_list.append(topk.indices.cpu().numpy())
        else:
            # FAISS Search
            for start in tqdm(
                range(0, n_queries, SEARCH_BATCH_SIZE),
                desc=f"FAISS Search {query_name}",
            ):
                end = min(start + SEARCH_BATCH_SIZE, n_queries)
                _, i_batch = index.search(query_emb[start:end], SAVE_PREDICTIONS_K)
                I_list.append(i_batch)

        I = np.vstack(I_list)

        search_time = time.time() - t1
        print(f"   Search done  ({search_time:.1f}s)")

        # ── Recall@K ──────────────────────────────────────────────────
        recalls = compute_recalls(query_ids, gallery_ids, I, TOP_K_METRICS)

        table = format_results_table(
            f"{self.model_config['backbone_arch']}-{self.model_config['agg_arch']}",
            query_name,
            recalls
        )
        print(f"\n{table}")

        # ── Save predictions ──────────────────────────────────────────
        split_tag = query_name.lower().replace(" ", "_")
        pred_path = os.path.join(self.output_dir, f"predictions_{split_tag}.npz")
        np.savez_compressed(
            pred_path,
            retrieved_indices=I,
            query_ids=query_ids,
            gallery_ids=gallery_ids,
        )
        print(f"   💾 Predictions saved to {pred_path}")

        return I, recalls, search_time

    # -----------------------------------------------------------------

    def _save_embeddings(self, embeddings, hotel_ids, name):
        """Save embeddings to .npz if --save_embeddings is set."""
        if not self.save_embeddings:
            return
        emb_path = os.path.join(self.output_dir, f"embeddings_{name}.npz")
        np.savez_compressed(emb_path, embeddings=embeddings, hotel_ids=hotel_ids)
        print(f"   💾 Embeddings saved to {emb_path} ({embeddings.shape})")

    def _save_metadata(self, all_recalls, timings):
        """Save a metadata.json with full experiment info."""
        meta = {
            "checkpoint_path": self.checkpoint_path,
            "hf_repo": self.hf_repo,
            "model_config": self.model_config,
            "device": DEVICE,
            "batch_size": BATCH_SIZE,
            "num_workers": NUM_WORKERS,
            "top_k_metrics": TOP_K_METRICS,
            "save_predictions_k": SAVE_PREDICTIONS_K,
            "save_embeddings": self.save_embeddings,
            "results": {},
            "timings": timings,
            "timestamp": datetime.now().isoformat(),
        }

        for split_name, recalls in all_recalls.items():
            meta["results"][split_name] = {f"recall@{k}": v for k, v in recalls.items()}

        meta_path = os.path.join(self.output_dir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)
        print(f"\n📋 Metadata saved to {meta_path}")

    def run(self):
        run_start = time.time()
        all_recalls = {}
        timings = {}

        # 1. Compute gallery embeddings (shared across both query sets)
        t0 = time.time()
        gallery_emb, gallery_ids = self.compute_embeddings(
            GALLERY_METADATA, desc="Processing Gallery"
        )
        timings["gallery_embedding_s"] = round(time.time() - t0, 1)
        self._save_embeddings(gallery_emb, gallery_ids, "gallery")

        is_multivector = gallery_emb.ndim == 3
        index = None

        if is_multivector:
            print(
                "\n🔍 Multivector embeddings detected. Bypassing FAISS for exhaustive MaxSim search..."
            )
        else:
            # Build FAISS index ONCE
            print("\n🔍 Building FAISS Index for the Gallery...")
            t0 = time.time()
            d = gallery_emb.shape[1]
            index = faiss.IndexFlatIP(d)
            index.add(gallery_emb)
            indexing_time = time.time() - t0
            timings["faiss_indexing_s"] = round(indexing_time, 1)
            print(f"   Indexed {index.ntotal} gallery vectors ({indexing_time:.1f}s)")

        # 2. Evaluate test_non_object queries
        t0 = time.time()
        non_obj_emb, non_obj_ids = self.compute_embeddings(
            TEST_NON_OBJECT_METADATA, desc="Processing Test Non-Object Queries"
        )
        timings["non_object_embedding_s"] = round(time.time() - t0, 1)
        self._save_embeddings(non_obj_emb, non_obj_ids, "test_non_object")

        _, recalls_non_obj, search_time = self._evaluate(
            index,
            non_obj_emb,
            non_obj_ids,
            gallery_ids,
            "TEST_NON_OBJECT",
            gallery_emb if is_multivector else None,
        )
        all_recalls["test_non_object"] = recalls_non_obj
        timings["non_object_search_s"] = round(search_time, 1)
        del non_obj_emb

        # 3. Evaluate test_object queries
        t0 = time.time()
        obj_emb, obj_ids = self.compute_embeddings(
            TEST_OBJECT_METADATA, desc="Processing Test Object Queries"
        )
        timings["object_embedding_s"] = round(time.time() - t0, 1)
        self._save_embeddings(obj_emb, obj_ids, "test_object")

        _, recalls_obj, search_time = self._evaluate(
            index,
            obj_emb,
            obj_ids,
            gallery_ids,
            "TEST_OBJECT",
            gallery_emb if is_multivector else None,
        )
        all_recalls["test_object"] = recalls_obj
        timings["object_search_s"] = round(search_time, 1)
        del obj_emb

        if not is_multivector:
            del gallery_emb

        timings["total_s"] = round(time.time() - run_start, 1)

        # ── Save metadata with all results ────────────────────────────
        self._save_metadata(all_recalls, timings)

        print("\n✅ Evaluation complete.")
        print(f"   📂 All results saved to: {self.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a VPR model on Hotel-50K.")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to the model checkpoint to evaluate (required if not using --hf_repo)",
    )
    parser.add_argument(
        "--model_config_path",
        type=str,
        default=None,
        help="Path to model_config.json saved during training (overrides default model_config)",
    )
    parser.add_argument(
        "--hf_repo",
        type=str,
        default=None,
        help="HuggingFace repository ID to download from (e.g., 'username/repo')",
    )
    parser.add_argument(
        "--hf_checkpoint_filename",
        type=str,
        default=None,
        help="Filename of the checkpoint in the HuggingFace repository (auto-detected if None)",
    )
    parser.add_argument(
        "--hf_config_filename",
        type=str,
        default=None,
        help="Filename of the config in the HuggingFace repository (optional, auto-detected if None)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Custom output directory for results (default: auto-generated under eval_results/)",
    )
    parser.add_argument(
        "--save_embeddings",
        action="store_true",
        help="Also save raw embeddings as .npz files (can be large)",
    )
    args = parser.parse_args()

    if args.hf_repo:
        try:
            from huggingface_hub import hf_hub_download, list_repo_files
        except ImportError:
            print("❌ huggingface_hub not installed. Please install it with `pip install huggingface_hub`")
            exit(1)
            
        print(f"🔍 Inspecting HuggingFace repo '{args.hf_repo}'...")
        repo_files = list_repo_files(repo_id=args.hf_repo)
        
        # Auto-detect checkpoint
        if not args.hf_checkpoint_filename:
            pth_files = [f for f in repo_files if f.endswith(".pth")]
            if len(pth_files) == 0:
                print(f"❌ No .pth checkpoint found in HF repo '{args.hf_repo}'")
                exit(1)
            elif len(pth_files) > 1:
                print(f"⚠️ Multiple .pth files found. Defaulting to: {pth_files[0]}")
            args.hf_checkpoint_filename = pth_files[0]
            
        # Auto-detect config
        if not args.hf_config_filename:
            if "model_config.json" in repo_files:
                args.hf_config_filename = "model_config.json"
            else:
                json_files = [f for f in repo_files if f.endswith(".json")]
                if len(json_files) == 1:
                    args.hf_config_filename = json_files[0]

        print(f"⬇️ Downloading checkpoint '{args.hf_checkpoint_filename}' from HF repo '{args.hf_repo}'...")
        args.checkpoint_path = hf_hub_download(repo_id=args.hf_repo, filename=args.hf_checkpoint_filename)
        
        if args.hf_config_filename:
            print(f"⬇️ Downloading config '{args.hf_config_filename}' from HF repo '{args.hf_repo}'...")
            args.model_config_path = hf_hub_download(repo_id=args.hf_repo, filename=args.hf_config_filename)
            
    if not args.checkpoint_path:
        parser.error("You must provide either --checkpoint_path or --hf_repo")

    # Load model config from model_config.json if provided
    model_config = None
    if args.model_config_path:
        with open(args.model_config_path, "r") as f:
            model_config = json.load(f)
        print(f"📄 Loaded model config from {args.model_config_path}")

    pipeline = EvaluationPipeline(
        checkpoint_path=args.checkpoint_path,
        model_config=model_config,
        output_dir=args.output_dir,
        save_embeddings=args.save_embeddings,
        hf_repo=args.hf_repo,
    )
    pipeline.run()
