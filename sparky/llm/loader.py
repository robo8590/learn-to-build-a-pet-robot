"""Load the selected local GGUF model and optional vision projector."""

import os

from llama_cpp import Llama

from sparky.config.llm_registry import (
    available_llm_keys,
    get_llm_config,
    is_vision_capable,
    is_vision_ready,
    llm_path,
    mmproj_path,
    resolve_llm_model_key,
)
from sparky.config.settings import LLM_N_GPU_LAYERS, LLM_N_THREADS


def initialize_llm(model_key=None):
    """Load the selected local GGUF model.

    Selection precedence: ``model_key`` (CLI) -> ``ROBOT_LLM_MODEL`` env var ->
    ``DEFAULT_LLM_MODEL``.

    Returns ``(llm, vision_enabled)`` where ``vision_enabled`` is True when the
    model supports vision and its mmproj projector was loaded, so image input
    is usable for the "look and ..." feature.
    """
    try:
        resolved_key = resolve_llm_model_key(
            cli_value=model_key,
            env_value=os.getenv("ROBOT_LLM_MODEL"),
        )
    except KeyError as e:
        raise SystemExit(str(e))

    config = get_llm_config(resolved_key)
    path = llm_path(resolved_key)

    if not path.is_file():
        raise SystemExit(
            f"LLM model '{resolved_key}' not found at '{path}'.\n"
            f"Download it with: python download_models.py --llm {resolved_key}\n"
            f"Available models: {', '.join(available_llm_keys())}"
        )

    print(f"Loading local LLM: {config['label']} ({resolved_key})...")

    chat_handler = None
    if is_vision_ready(resolved_key):
        projector = mmproj_path(resolved_key)
        try:
            # MTMDChatHandler renders the GGUF's own tokenizer.chat_template,
            # so it works for Gemma 4, Qwen3.5, and other vision families.
            # Gemma4ChatHandler is an empty subclass of MTMDChatHandler.
            from llama_cpp.llama_chat_format import MTMDChatHandler

            print(f"Loading vision projector: {projector}...")
            chat_handler = MTMDChatHandler(
                clip_model_path=str(projector),
                use_gpu=LLM_N_GPU_LAYERS != 0,
                verbose=False,
            )
        except Exception as e:
            print(f"[Vision] Warning: could not load vision projector ({e}). "
                  "Vision will be disabled for this session.")
    elif is_vision_capable(resolved_key):
        print(
            f"[Vision] disabled: mmproj projector missing for {resolved_key}. "
            f"Run: python download_models.py --mmproj {resolved_key}"
        )

    llm = Llama(
        model_path=str(path),
        n_ctx=config["n_ctx"],
        n_threads=LLM_N_THREADS,
        n_gpu_layers=LLM_N_GPU_LAYERS,
        chat_handler=chat_handler,
        verbose=False,
    )

    vision_enabled = chat_handler is not None
    if vision_enabled:
        print("[Vision] enabled: image input ready.")
    return llm, vision_enabled
