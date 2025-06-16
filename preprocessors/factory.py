from preprocessors.s3e import S3E

PREPROCESSOR_REGISTRY = {
    "S3E": S3E,
}

def load_preprocessor_model(cfg):
    cls = PREPROCESSOR_REGISTRY.get(cfg.model_name)
    if cls is None:
        raise ValueError(f"Unknown preprocessor {cfg.model_name}")
    return cls(cfg)
