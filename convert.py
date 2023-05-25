from llm_rs.convert import AutoConverter
from llm_rs.repository import Repository
from llm_rs import ContainerType,QuantizationType,AutoQuantizer
import sys
import os
import logging 


ORGANIZATION="Rustformers"
REMOVE_CONVERTED = True

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 4:
        print("Usage: python convert.py <repo_name> <output_directory> <source_repo1> <source_repo2> ....")
        sys.exit(1)
        
    repo_name = sys.argv[1]
    output_dir = sys.argv[2]
    targets = sys.argv[3:]
    token = os.environ.get("HUGGINGFACE_TOKEN",None)
    repo = Repository(repo_name,ORGANIZATION,token=token)
    print(f"Converting {'|'.join(targets)} into {repo.name}...")

    container_types = [ContainerType.GGML,ContainerType.GGJT]

    for target in targets:
        converted_model = AutoConverter.convert(target, output_dir)
        for container_type in container_types:
            quantized = AutoQuantizer.quantize(converted_model,container=container_type)
            repo.upload(quantized)
            if REMOVE_CONVERTED:
                os.remove(quantized)
        repo.upload(converted_model)
        if REMOVE_CONVERTED:
            os.remove(converted_model)

    print("Done!")