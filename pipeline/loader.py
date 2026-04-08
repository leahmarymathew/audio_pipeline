import os

import datasets
from datasets import Audio, load_dataset
import httpx
from huggingface_hub import set_client_factory


languages = ["hindi", "tamil", "bengali", "marathi", "gujarati"]

# Keep streaming retries short to avoid long stalls on broken network/proxy paths.
datasets.config.STREAMING_READ_MAX_RETRIES = 2
datasets.config.STREAMING_READ_RETRY_INTERVAL = 1


def _configure_hf_client():
    def client_factory():
        return httpx.Client(
            verify=False,
            follow_redirects=True,
            timeout=httpx.Timeout(60.0, connect=30.0, read=60.0, write=60.0, pool=60.0),
        )

    set_client_factory(client_factory)


_configure_hf_client()


def _load_hf_streaming_datasets():
    datasets_dict = {}

    for lang in languages:
        try:
            ds = load_dataset(
                "ai4bharat/IndicVoices",
                lang,
                split="train",
                streaming=True,
            )
            ds = ds.cast_column("audio_filepath", Audio(decode=False))
            datasets_dict[lang] = ds
        except Exception as exc:
            print(f"[{lang}] dataset load warning: {exc}")
            datasets_dict[lang] = []

    return datasets_dict


def get_datasets():
    print("Loading IndicVoices from Hugging Face streaming...")
    return _load_hf_streaming_datasets()
