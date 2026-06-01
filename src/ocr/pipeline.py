"""OCR pipeline with pluggable model support."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union
from PIL import Image
from .models import load_checkpoint

# Model registry
_MODEL_REGISTRY: dict[str, type["BaseOCRModel"]] = {}

def register_model(name: str):
    """Decorator to register a custom OCR model."""
    def decorator(cls):
        _MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def list_models() -> list[str]:
    """List all registered model names."""
    return list(_MODEL_REGISTRY.keys())


class BaseOCRModel(ABC):
    """Base class for OCR models."""

    @abstractmethod
    def __init__(self, model_id: str = None):
        pass

    @abstractmethod
    def recognize(self, image: Image.Image) -> str:
        """Recognize text from a PIL Image."""
        pass


@register_model("trocr_smi")
@register_model("trocr_smi_nor")
@register_model("trocr_smi_pred")
@register_model("trocr_smi_nor_pred")
@register_model("trocr_smi_synth")
@register_model("trocr_smi_pred_synth")
@register_model("trocr_smi_nor_pred_synth")
class TrOCRModel(BaseOCRModel):
    """TrOCR models fine-tuned on Sami data from Sprakbanken."""

    MODEL_MAP = {
        "trocr_smi": "Sprakbanken/trocr_smi",
        "trocr_smi_nor": "Sprakbanken/trocr_smi_nor",
        "trocr_smi_pred": "Sprakbanken/trocr_smi_pred",
        "trocr_smi_nor_pred": "Sprakbanken/trocr_smi_nor_pred",
        "trocr_smi_synth": "Sprakbanken/trocr_smi_synth",
        "trocr_smi_pred_synth": "Sprakbanken/trocr_smi_pred_synth",
        "trocr_smi_nor_pred_synth": "Sprakbanken/trocr_smi_nor_pred_synth",
    }

    def __init__(self, model_id: str = None):
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        hf_model = self.MODEL_MAP.get(model_id, model_id)
        self.processor = TrOCRProcessor.from_pretrained(hf_model)
        self.model = VisionEncoderDecoderModel.from_pretrained(hf_model)

    def recognize(self, image: Image.Image) -> str:
        image = image.convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
        generated_ids = self.model.generate(pixel_values)
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


# Register all modular OCR architectures
_MODULAR_ARCHITECTURES = [
    # CRNN variants
    "crnn_vgg16", "crnn_vgg19", "crnn_resnet50", "crnn_resnet101", "crnn_simple",
    # CTC-only variants
    "ctc_vgg16", "ctc_vgg19", "ctc_resnet50", "ctc_resnet101", "ctc_simple",
    # Transformer variants
    "transformer_vgg16", "transformer_vgg19", "transformer_resnet50",
    "transformer_resnet101", "transformer_simple",
]


class ModularOCRModel(BaseOCRModel):
    """
    Modular OCR model supporting all backbone/encoder combinations.

    Uses the new unified checkpoint format from train_unified.py.
    Requires explicit weights_path parameter - trained models are stored
    in trained_models/ directory from train_queue.py.
    """

    DEFAULT_VOCAB = " !%'(),-./0123456789:;?ABCDEFGHIJKLMNOPRSTUVWZabcdefghijklmnoprstuvwyz|§ÁÂÄÅÏáâäåæïöøČčĐđŠšŧŊŋž–"

    def __init__(self, model_id: str = None, weights_path: str = None):
        import torch
        from torchvision import transforms

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = model_id

        if weights_path is None:
            raise ValueError(
                f"No weights provided for {model_id}. "
                f"Specify weights_path or train with: python src/ocr/train_queue.py --models {model_id}"
            )

        # Load checkpoint
        checkpoint = torch.load(weights_path, map_location=self.device)

        self.charset = checkpoint["charset"]
        self.char_to_idx = checkpoint["char_to_idx"]
        self.idx_to_char = checkpoint["idx_to_char"]

        config = checkpoint.get("config", {})
        self.img_height = config.get("img_height", 32)
        self.img_width = config.get("img_width", 800)

        self.model, _, _, _ = load_checkpoint(weights_path, device=str(self.device))

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

    def recognize(self, image: Image.Image) -> str:
        import torch

        image = self._preprocess_image(image)
        x = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)  # (B, T, C)
            preds = logits.argmax(dim=2).squeeze(0).cpu().tolist()

        # CTC decode (greedy)
        result = []
        prev = -1
        for idx in preds:
            if idx != prev and idx != 0:
                char = self.idx_to_char.get(idx)
                if char and char != "<blank>":
                    result.append(char)
            prev = idx
        return "".join(result)

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Resize image maintaining aspect ratio, then pad/crop to target size."""
        w, h = image.size
        new_h = self.img_height
        new_w = int(w * (new_h / h))

        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        if new_w < self.img_width:
            padded = Image.new("L", (self.img_width, self.img_height), 255)
            padded.paste(image.convert("L"), (0, 0))
            return padded
        elif new_w > self.img_width:
            return image.crop((0, 0, self.img_width, self.img_height))
        return image


# Register all modular architectures
for arch in _MODULAR_ARCHITECTURES:
    _MODEL_REGISTRY[arch] = ModularOCRModel


class OCRPipeline:
    """Main OCR pipeline interface."""

    def __init__(self, model_name: str = "trocr_smi_pred_synth", weights_path: str = None):
        """
        Initialize the OCR pipeline.

        Args:
            model_name: Name of the model to use. See list_models() for options.
                        Default is 'trocr_smi_pred_synth' (best performing).
            weights_path: Path to model weights (required for custom models like ctc_simple).
        """
        if model_name not in _MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model: {model_name}. Available: {list_models()}"
            )
        self.model_name = model_name
        self.weights_path = weights_path
        self._model = None

    @property
    def model(self) -> BaseOCRModel:
        """Lazy-load the model."""
        if self._model is None:
            model_cls = _MODEL_REGISTRY[self.model_name]
            if self.weights_path:
                self._model = model_cls(model_id=self.model_name, weights_path=self.weights_path)
            else:
                self._model = model_cls(model_id=self.model_name)
        return self._model

    def recognize(self, image: Union[str, Path, Image.Image]) -> str:
        """
        Recognize text from an image.

        Args:
            image: Path to image file or PIL Image object.

        Returns:
            Recognized text string.
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        return self.model.recognize(image)

    def __call__(self, image: Union[str, Path, Image.Image]) -> str:
        """Shorthand for recognize()."""
        return self.recognize(image)


# Example of adding a custom model:
#
# @register_model("my_custom_model")
# class MyCustomModel(BaseOCRModel):
#     def __init__(self, model_id: str = None):
#         # Load your model here
#         pass
#
#     def recognize(self, image: Image.Image) -> str:
#         # Run inference
#         return "recognized text"
