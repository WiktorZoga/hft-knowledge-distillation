import torch

def resolve_device(config_device: str = "cuda") -> torch.device:
    if config_device == "cpu":
        return torch.device("cpu")
        
    if torch.cuda.is_available():
        return torch.device("cuda")
        
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
        
    return torch.device("cpu")