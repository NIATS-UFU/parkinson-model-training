"""Content hashes for pretrained model tensors used by paper pipelines."""

import hashlib


def torch_module_sha256(module) -> str:
    """Return a deterministic SHA-256 over a PyTorch module state dictionary."""
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(str(value.dtype).encode())
        digest.update(b"\0")
        digest.update(str(tuple(value.shape)).encode())
        digest.update(b"\0")
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def keras_model_sha256(model) -> str:
    """Return a deterministic SHA-256 over a Keras model's weights."""
    digest = hashlib.sha256()
    for weight in sorted(model.weights, key=lambda item: item.path):
        value = weight.numpy()
        digest.update(weight.path.encode())
        digest.update(b"\0")
        digest.update(str(value.dtype).encode())
        digest.update(b"\0")
        digest.update(str(value.shape).encode())
        digest.update(b"\0")
        digest.update(value.tobytes())
    return digest.hexdigest()
