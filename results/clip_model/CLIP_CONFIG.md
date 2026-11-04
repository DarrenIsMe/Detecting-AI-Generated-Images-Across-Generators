# Frozen CLIP configuration

- Model: OpenAI CLIP ViT-B/32
- OpenCLIP model name: `ViT-B-32-quickgelu`
- Pretrained tag: `openai`
- Activation: QuickGELU (required by the OpenAI checkpoint)
- Embedding dimension: 512
- Feature dtype on disk: float32
- Feature normalization: L2-normalized
- Checkpoint provider/repository: `timm/vit_base_patch32_clip_224.openai`
- Checkpoint repository revision: `a6f597a30f7b82c51704746581f9a4e41421e878`
- Checkpoint file: `open_clip_model.safetensors`
- Checkpoint size: 605,143,284 bytes
- Checkpoint SHA-256: `E6D1BD7789AA45192B3BF90570A789B478BAE1B74EBCCE7EDDD908E83A2B7C31`
- OpenCLIP: 3.3.0
- PyTorch: 2.11.0+cu128
- torchvision: 0.26.0+cu128
- Device used: NVIDIA GeForce GTX 1660 SUPER (6,144 MiB)
- Preprocess: bicubic resize to 224, center crop 224x224, RGB conversion, OpenAI mean `(0.48145466, 0.4578275, 0.40821073)`, OpenAI std `(0.26862954, 0.26130258, 0.27577711)`

The checkpoint is stored under `data/models/open_clip/`. Extraction metadata records the configuration hash, requested/effective batch sizes, ordered image IDs, and checkpoint cache path.
