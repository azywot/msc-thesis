"""Image inspector tool for analyzing image files.

This tool allows vision-language models to analyze and answer questions about images.
Only enabled in non-direct mode (requires VLM).
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from PIL import Image

from ..core.tool import BaseTool, ToolResult
from ..utils.logging import get_logger
from ..utils.parsing import strip_thinking_tags
from ..utils.prompting import append_step_by_step_instruction, should_append_step_by_step_instruction

logger = get_logger(__name__)

# Supported image extensions
SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


class ImageInspectorTool(BaseTool):
    """Tool for inspecting image files using vision-language models.

    This tool requires a vision model (VLM) and only works in non-direct/sub-agent mode.
    It loads images, converts them to RGB, and uses the VLM to answer questions about them.
    """

    def __init__(
        self,
        model_provider: Any,
        use_thinking: bool = False
    ):
        """Initialize image inspector tool.

        Args:
            model_provider: Model provider with VLM support (must support multimodal generation)
            use_thinking: Whether to use thinking mode for VLM
        """
        self.model_provider = model_provider
        self.use_thinking = use_thinking

    @property
    def name(self) -> str:
        return "image_inspector"

    @property
    def description(self) -> str:
        return "Analyze and answer questions about image files using a vision-language model"

    def get_schema(self) -> Dict[str, Any]:
        """Return Qwen3 JSON Schema.

        Image inspector requires only a question. The system injects full_file_path
        from attachments (no LLM-provided path).
        """
        return {
            "type": "function",
            "function": {
                "name": "image_inspector",
                "description": (
                    "Inspect the attached image file for this question. "
                    "Important: Do NOT guess or provide file paths. The system will automatically use the "
                    "question's attached image. You MUST include a question about the image."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "Question about the attached image."
                        }
                    },
                    "required": ["question"]
                }
            }
        }

    def execute(self, question: str, full_file_path: str) -> ToolResult:
        """Analyze image file using VLM.

        Args:
            question: Question about the image
            full_file_path: Path to image file (injected by framework from attachments)

        Returns:
            ToolResult with VLM analysis
        """
        logger.info(f"Inspecting image file: {full_file_path} with question: {question}")

        path = Path(full_file_path)

        # Check if file exists
        if not path.exists():
            return ToolResult(
                success=False,
                output="",
                metadata={"file_path": full_file_path},
                error=f"File not found: {full_file_path}"
            )

        # Check if it's a file
        if not path.is_file():
            return ToolResult(
                success=False,
                output="",
                metadata={"file_path": full_file_path},
                error=f"Path is not a file: {full_file_path}"
            )

        # Get file extension
        file_ext = path.suffix.lower()

        # Check if supported
        if file_ext not in SUPPORTED_IMAGE_EXTS:
            return ToolResult(
                success=False,
                output="",
                metadata={"file_path": full_file_path, "file_type": file_ext},
                error=f"Unsupported image type: {file_ext}. Supported types: {', '.join(sorted(SUPPORTED_IMAGE_EXTS))}"
            )

        # Validate question
        if not question or not question.strip():
            return ToolResult(
                success=False,
                output="",
                metadata={"file_path": full_file_path},
                error="Question parameter is required for image inspection"
            )

        try:
            image = self._load_image_rgb(path)
            analysis, usage = self._analyze_with_vlm(image, question)

            return ToolResult(
                success=True,
                output=analysis,
                metadata={
                    "file_path": full_file_path,
                    "file_name": path.name,
                    "file_type": file_ext,
                    "image_size": image.size,
                    "image_mode": image.mode,
                    "question": question
                },
                usage=usage,
            )

        except Exception as e:
            logger.error(f"Error analyzing image: {e}", exc_info=True)
            return ToolResult(
                success=False,
                output="",
                metadata={"file_path": full_file_path, "file_type": file_ext},
                error=f"Error analyzing image: {str(e)}"
            )

    def _load_image_rgb(self, path: Path) -> Image.Image:
        """Load image and convert to RGB format.

        Args:
            path: Path to image file

        Returns:
            PIL Image in RGB mode

        Raises:
            ValueError: If image cannot be loaded or converted
        """
        try:
            img = Image.open(path)
            # Normalize to RGB for broad model compatibility
            if img.mode != "RGB":
                img = img.convert("RGB")
            return img
        except Exception as e:
            raise ValueError(f"Failed to load/convert image: {e}")

    def _analyze_with_vlm(self, image: Image.Image, question: str) -> Tuple[str, Optional[Dict[str, int]]]:
        """Use vision-language model to analyze image and answer question.

        Args:
            image: PIL Image object
            question: Question about the image

        Returns:
            Tuple of (VLM analysis/answer, token usage dict or None)
        """
        if not self.model_provider:
            return "[Note] No image-inspector model configured; cannot analyze the image.", None

        system_prompt = (
            "You are given an image attached to the user's question. "
            "Answer the question using only the image content. "
            "If the image does not contain enough information, say so."
        )
        if should_append_step_by_step_instruction(self.model_provider, self.use_thinking):
            system_prompt = append_step_by_step_instruction(system_prompt)

        prompt_messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"Question:\n{question}\n"},
                ],
            },
        ]

        try:
            prompt = self.model_provider.apply_chat_template(prompt_messages, use_thinking=self.use_thinking)

            result = self.model_provider.generate([
                {
                    "prompt": prompt,
                    "multi_modal_data": {"image": image},
                }
            ])[0]

            output = strip_thinking_tags(result.text)
            return output, result.usage

        except Exception as e:
            logger.error(f"VLM generation failed: {e}", exc_info=True)
            raise RuntimeError(f"Image inspection failed (model may not support multimodal inputs via vLLM): {e}")

    def validate_args(self, **kwargs) -> bool:
        """Validate arguments.

        Args:
            **kwargs: Tool arguments (question from LLM; full_file_path injected by orchestrator)

        Returns:
            True if valid
        """
        if 'question' not in kwargs:
            return False

        question = kwargs['question']

        return isinstance(question, str) and len(question.strip()) > 0
