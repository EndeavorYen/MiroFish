"""Convert multilingual-e5-small to GGUF with its real tokenizer (#61).

The model's ``config.json`` says ``BertModel``, but it tokenises with
XLM-R's SentencePiece Unigram model. llama.cpp's converter writes a
WordPiece vocabulary for every ``BertModel``, so the GGUF it produces turns
most English words into ``<unk>`` and splits Chinese into single characters
(vectors at cosine 0.78-0.95 of the original). This keeps the BERT tensors
(position ids from 0, nothing trimmed) and writes the Unigram vocabulary
instead; the result matches transformers token for token (cosine 1.0000).

Needs a llama.cpp checkout (only ``convert_hf_to_gguf.py``, ``conversion/``
and ``gguf-py/``) and the Hugging Face model directory. The converter needs
torch, so run it in a throwaway environment rather than the backend's:

    git clone --depth 1 https://github.com/ggml-org/llama.cpp
    uv run --no-project --python 3.12 \\
        --with torch --with transformers --with sentencepiece --with protobuf \\
        --with safetensors --with numpy --with pyyaml --with requests \\
        python backend/scripts/convert_e5_gguf.py --llama-cpp llama.cpp \\
        --out multilingual-e5-small-ugm-F16.gguf

``--model`` defaults to the copy in the Hugging Face cache.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

REPO = "intfloat/multilingual-e5-small"


def cached_model() -> str | None:
    root = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"), "hub")
    found = sorted(glob.glob(os.path.join(root, "models--intfloat--multilingual-e5-small", "snapshots", "*")))
    return found[-1] if found else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--llama-cpp", required=True, type=Path, help="a llama.cpp checkout")
    parser.add_argument("--model", help=f"the {REPO} directory (default: the Hugging Face cache)")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--outtype", default="f16")
    args = parser.parse_args()

    model = args.model or cached_model()
    if not model:
        parser.error(f"{REPO} is not in the Hugging Face cache; pass --model")
    sys.path.insert(0, str(args.llama_cpp))
    sys.path.insert(1, str(args.llama_cpp / "gguf-py"))
    import conversion.bert as bert

    # Tensors as BertModel, vocabulary as XLM-R (SentencePiece Unigram).
    bert.BertModel.set_vocab = bert.BertModel._xlmroberta_set_vocab
    sys.argv = ["convert_hf_to_gguf.py", model, "--outtype", args.outtype, "--outfile", str(args.out)]
    import convert_hf_to_gguf

    convert_hf_to_gguf.main()

    from gguf import GGUFReader

    field = GGUFReader(str(args.out)).fields["tokenizer.ggml.model"]
    tokenizer = bytes(field.parts[field.data[0]]).decode()
    if tokenizer != "t5":
        print(f"unexpected tokenizer.ggml.model {tokenizer!r} (want 't5', SentencePiece Unigram)")
        return 1
    print(f"wrote {args.out} (tokenizer.ggml.model = t5)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
