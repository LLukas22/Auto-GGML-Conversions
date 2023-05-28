from llm_rs.repository import Repository
from llm_rs.auto import CURRENT_QUANTIZATION_VERSION
from llm_rs import ContainerType,QuantizationType,AutoQuantizer,ModelMetadata
import sys
import os
import logging 
from huggingface_hub import snapshot_download
from pathlib import Path
import shutil
import json 
from tabulate import tabulate

ORGANIZATION="rustformers"
README_CACHE = "./README_CACHE"

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) != 2:
        print("Usage: python generate_readme.py <repo_name>")
        sys.exit(1)
        
    repo_name = sys.argv[1]
    token = os.environ.get("HUGGINGFACE_TOKEN",None)
    repo = Repository(repo_name,ORGANIZATION,token=token)
    cache = Path(README_CACHE).resolve()
    if cache.exists():
        shutil.rmtree(cache)

    cache.mkdir(parents=True,exist_ok=True)

    #download the readme template from the repo
    try:
        snapshot_download(repo.name,allow_patterns=["README_TEMPLATE.md"],token=token,local_dir=cache,local_dir_use_symlinks=False)
        template = cache / "README_TEMPLATE.md"
        readme = template.read_text()
    except Exception as e:
        print(f"Failed to download README_TEMPLATE.md from {repo.name}.")
        print(e)
        sys.exit(1)

    #get all the models in the repo
    snapshot_download(repo.name,allow_patterns=["*.meta"],token=token,local_dir=cache,local_dir_use_symlinks=False)
    metadata_files = [x for x in cache.glob("*.meta") if x.is_file()]
    print(f"Found {len(metadata_files)} models in {repo.name}.")
    #get the model name from the meta file

    models:dict[str,ModelMetadata] = {}
    for file in metadata_files:
        metadata = ModelMetadata.deserialize(json.loads(file.read_text()))
        models[file.stem] = metadata

    # generate the readme

    
    # generate the model table
    table_data = []
    for model_name,metadata in models.items():
        filename = f"{model_name}.bin"

        table_data.append([
            f"[{filename}](https://huggingface.co/{repo.name}/blob/main/{filename})",
            f"[{metadata.base_model}](https://huggingface.co/{metadata.base_model})" if metadata.base_model else "-",
            metadata.quantization.__repr__().split(".")[-1],
            metadata.container.__repr__().split(".")[-1],
            "V3"
        ])

    headers = ["Name", "Based on", "Type", "Container", "GGML Version"]
    table = tabulate(table_data, headers=headers, tablefmt="pipe")

    readme = readme.replace("$MODELS$",table)

    new_readme = cache / "README.md"
    new_readme.write_text(readme)

    repo.api.upload_file(repo_id=repo.name,path_in_repo="README.md",path_or_fileobj=new_readme,token=token,commit_message="Generated README.md")
    print("Done!")