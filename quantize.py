from llm_rs.repository import Repository
from llm_rs import ContainerType,QuantizationType,AutoQuantizer
import sys
import os
import logging 
from pathlib import Path
from huggingface_hub import snapshot_download

ORGANIZATION="rustformers"
REMOVE = True

QUANTIZATION_TASKS = [
    (ContainerType.GGML,QuantizationType.Q4_0),
    (ContainerType.GGJT,QuantizationType.Q4_0),
    (ContainerType.GGML,QuantizationType.Q5_1),
    (ContainerType.GGJT,QuantizationType.Q5_1),
]

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 4:
        print("Usage: python quantize.py <repo_name> <output_directory> <source_file1> <source_file2> ....")
        sys.exit(1)
        
    repo_name = sys.argv[1]
    output_dir = sys.argv[2]
    targets = sys.argv[3:]
    token = os.environ.get("HUGGINGFACE_TOKEN",None)
    repo = Repository(repo_name,ORGANIZATION,token=token)
    print(f"Quantizing {'|'.join(targets)} into {repo.name}...")

    cache = Path(output_dir).resolve()
    cache.mkdir(parents=True,exist_ok=True)

    for target in targets:
        model_file = cache / target
        assert model_file.suffix == ".bin", "Only .bin files are supported"
        meta_file = model_file.with_suffix(".meta")

        snapshot_download(repo.name,allow_patterns=[model_file.name,meta_file.name],token=token,local_dir=cache,local_dir_use_symlinks=False)

        assert model_file.exists(), f"Failed to download {model_file.name} from {repo.name}."
        assert meta_file.exists(), f"Failed to download {meta_file.name} from {repo.name}."

        for container_type,quantization_type in QUANTIZATION_TASKS:
            quantized = AutoQuantizer.quantize(model_file,container=container_type,quantization=quantization_type)
            repo.upload(quantized)
            if REMOVE:
                os.remove(quantized)

        if REMOVE:
            os.remove(model_file)
            os.remove(meta_file)

    print("Done!")