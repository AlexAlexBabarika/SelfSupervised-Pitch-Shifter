import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

from config import DataConfig, DatasetSpec

DATASETS_DIR = Path("./datasets")


def download_dataset(
    repo_id: str,
    local_dir: Path,
    allow_patterns: list[str],
    token: str | None,
) -> Path:
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
        token=token,
        allow_patterns=allow_patterns,
        max_workers=8,
    )
    return local_dir


# GTSinger audio is 48 kHz, 24-bit, mono.
GTSINGER_BYTES_PER_S = 144_000


def select_gtsinger_files(spec: DatasetSpec, token: str | None) -> list[str]:
    # The full set is far larger than the per-singer cap, so pick a random
    # subset of each singer's singing takes: 1.3x the cap, to leave room for
    # what silence trimming removes.
    api = HfApi(token=token)
    budget = spec.group_minutes * 60.0 * 1.3 * GTSINGER_BYTES_PER_S
    picked: list[str] = []

    def dirs(path: str | None) -> list[str]:
        tree = api.list_repo_tree(spec.hf_name, path_in_repo=path, repo_type="dataset")
        return sorted(e.path for e in tree if not hasattr(e, "size"))

    for lang in dirs(None):
        for singer in dirs(lang):
            tree = api.list_repo_tree(
                spec.hf_name, path_in_repo=singer, recursive=True, repo_type="dataset"
            )
            wavs = sorted(
                (e.path, e.size)
                for e in tree
                if hasattr(e, "size")
                and e.path.endswith(".wav")
                and "Paired_Speech_Group" not in e.path
            )
            random.Random(0).shuffle(wavs)
            total = 0
            for path, size in wavs:
                if total >= budget:
                    break
                picked.append(path)
                total += size
    return picked


def download_gtsinger(spec: DatasetSpec, local_dir: Path, token: str | None) -> Path:
    local_dir.mkdir(parents=True, exist_ok=True)
    files = select_gtsinger_files(spec, token)
    print(f"  gtsinger: {len(files)} files selected")

    def fetch(path: str) -> None:
        hf_hub_download(
            spec.hf_name, path, repo_type="dataset", local_dir=local_dir, token=token
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(fetch, files))
    return local_dir


def main() -> None:
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")

    data_cfg = DataConfig()

    DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    keys = data_cfg.datasets_to_load
    print(f"Downloading {keys} in parallel into {DATASETS_DIR}/")
    with ThreadPoolExecutor(max_workers=len(keys)) as pool:
        futures = {}
        for key in keys:
            spec = data_cfg.datasets[key]
            path = DATASETS_DIR / spec.subdir
            if key == "gtsinger":
                fut = pool.submit(download_gtsinger, spec, path, hf_token)
            else:
                fut = pool.submit(
                    download_dataset,
                    spec.hf_name,
                    path,
                    [spec.parquet_glob],
                    hf_token,
                )
            futures[fut] = key
        for fut in as_completed(futures):
            name = futures[fut]
            fut.result()
            print(f"  done: {name}")


if __name__ == "__main__":
    main()
