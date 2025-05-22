import logging
from typing import Dict, Any, Tuple
import torch
import gradio as gr
from infer import ModelLoader, DEVICE, Translator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Store models and tokenizers
MODELS: Dict[str, Tuple[Any, Any]] = {
    "mbart50": (None, None),
    "mt5": (None, None),
    "rbmt": (None, None)
}

def initialize_models(model_types: list[str] = ["mbart50", "mt5", "rbmt"]) -> None:
    """Initialize translation models and store them in MODELS dictionary.

    Args:
        model_types: List of model types to initialize.
    """
    global MODELS
    for model_type in model_types:
        try:
            if model_type == "mbart50":
                logger.info("Loading MBart50 model...")
                MODELS["mbart50"] = ModelLoader.load_mbart50()
                logger.info(f"MBart50 model loaded on {DEVICE}")
            elif model_type == "mt5":
                logger.info("Loading MT5 model...")
                MODELS["mt5"] = ModelLoader.load_mt5()
                logger.info(f"MT5 model loaded on {DEVICE}")
            elif model_type == "rbmt":
                logger.info("Initializing RBMT...")
                from models.rule_based_mt import TransferBasedMT
                MODELS["rbmt"] = (TransferBasedMT(), None)
                logger.info("RBMT initialized")
            elif model_type == "smt":
                logger.warning("SMT not implemented, skipping...")
        except Exception as e:
            logger.error(f"Failed to initialize {model_type}: {str(e)}")
            MODELS[model_type] = (None, None)

def translate_text(model_type: str, input_text: str) -> str:
    """Translate input text using the selected model.

    Args:
        model_type: Type of model to use ('rbmt', 'smt', 'mbart50', 'mt5').
        input_text: English text to translate.

    Returns:
        Translated text or error message.
    """
    try:
        model, tokenizer = MODELS.get(model_type, (None, None))
        if model is None:
            return f"Error: Model '{model_type}' not loaded or not supported."
        if model_type == "rbmt":
            return Translator.translate_rbmt(input_text)
        elif model_type == "smt":
            return Translator.translate_smt(input_text)
        elif model_type == "mbart50":
            return Translator.translate_mbart50(input_text, model, tokenizer)
        else:  # mt5
            return Translator.translate_mt5(input_text, model, tokenizer)
    except Exception as e:
        return f"Error during translation: {str(e)}"

# Initialize models before launching the app
logger.info("Starting model initialization...")
initialize_models()
logger.info("Model initialization complete.")

# Define Gradio interface
with gr.Blocks(theme="soft", title="English to Vietnamese Translator", css="""
    .gr-dropdown .options {
        background: #f8f9fa;
    }
    .gr-dropdown .options .item:hover {
        background-color: #e3f2fd;
        cursor: pointer;
    }
    .gr-dropdown .options .item.selected {
        background-color: #bbdefb;
    }
    .container {
        background-color: #f5f7fa;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .textbox {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
    }
    .gr-button {
        background-color: #2196f3 !important;
        color: white !important;
    }
    .gr-button:hover {
        background-color: #1976d2 !important;
    }
    .markdown {
        color: #2c3e50;
    }
""") as demo:
    gr.Markdown(
        "# English to Vietnamese Machine Translation Demo",
        elem_classes=["markdown"]
    )
    gr.Markdown(
        "Select a model and enter English text to translate into Vietnamese.",
        elem_classes=["markdown"]
    )

    with gr.Column(elem_classes=["container"]):
        model_choice = gr.Dropdown(
            choices=["rbmt", "smt", "mbart50", "mt5"],
            label="Model Type",
            value="mbart50",
            elem_classes=["gr-dropdown"]
        )
        input_text = gr.Textbox(
            label="Input Text (English)",
            placeholder="Enter English text to translate...",
            lines=5,
            elem_classes=["textbox"]
        )
        output_text = gr.Textbox(
            label="Translated Text (Vietnamese)",
            lines=5,
            elem_classes=["textbox"],
            interactive=False
        )
        translate_button = gr.Button("Translate", elem_classes=["gr-button"])
    
    gr.Markdown(
        "_Note_: SMT is not implemented. RBMT requires a working TransferBasedMT class.",
        elem_classes=["markdown"]
    )

    # Bind the translation function to the button
    translate_button.click(
        fn=translate_text,
        inputs=[model_choice, input_text],
        outputs=output_text,
        show_progress=True
    )

# Launch the app
if __name__ == "__main__":
    demo.launch()