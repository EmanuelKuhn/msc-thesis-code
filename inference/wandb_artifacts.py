import wandb


def get_artifact(reference: str, type: str | None = None, init_wandb=True) -> wandb.Artifact:
    if init_wandb:
        global wandb
        if not "wandb" in globals() or wandb.run is None:
            import wandb
            wandb.init(project="wandb_artifacts.py", name="get_artifact", resume=True)

    artifact = wandb.run.use_artifact(reference, type=type)
    
    return artifact
