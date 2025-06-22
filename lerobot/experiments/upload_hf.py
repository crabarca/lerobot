from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path="/Users/cristobal/Proyectos/lerobotmuc/outputs/pickup_v1",
    repo_id="crabarca/trash_pickup_v1",
    repo_type="model",
)