import logging
from typing import Dict, Any, Tuple
import torch
import gradio as gr
from infer import ModelLoader, DEVICE, Translator
from models.statistical_mt import LanguageModel

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
    "rbmt": (None, None),
    "smt": (None, None)
}

def initialize_models(model_types: list[str] = ["mbart50", "mt5", "rbmt", "smt"]) -> None:
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
                logger.info("Initializing SMT...")
                MODELS["smt"] = (ModelLoader.load_smt(), None)
                logger.info("SMT initialized")
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
            return Translator.translate_smt(input_text, model)
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
with gr.Blocks(
    theme="soft", 
    title="English to Vietnamese Translator",
    css="""
    /* Root variables for consistent theming */
    :root {
        --primary-color: #2563eb;
        --primary-hover: #1d4ed8;
        --secondary-color: #64748b;
        --success-color: #10b981;
        --error-color: #ef4444;
        --warning-color: #f59e0b;
        --background-primary: #ffffff;
        --background-secondary: #f8fafc;
        --background-tertiary: #f1f5f9;
        --text-primary: #1e293b;
        --text-secondary: #64748b;
        --border-color: #e2e8f0;
        --border-radius: 12px;
        --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        --transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* Global styles */
    * {
        box-sizing: border-box;
    }

    body {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
        line-height: 1.6;
        color: var(--text-primary);
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }

    /* Main container */
    .gradio-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 2rem;
    }

    /* Header styling */
    .header {
        text-align: center;
        margin-bottom: 3rem;
        padding: 2rem;
        background: var(--background-primary);
        border-radius: var(--border-radius);
        box-shadow: var(--shadow-lg);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    .header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--primary-color);
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 4px rgba(37, 99, 235, 0.2);
        position: relative;
        z-index: 1;
    }

    /* Enhanced gradient text effect for supported browsers */
    @supports (-webkit-background-clip: text) {
        .header h1 {
            background: linear-gradient(135deg, var(--primary-color), #7c3aed, #ec4899, var(--primary-color));
            background-size: 200% 200%;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            animation: gradientShift 4s ease-in-out infinite;
        }
    }

    @keyframes gradientShift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }

    .header p {
        color: var(--text-secondary);
        font-size: 1.1rem;
        margin: 0;
    }

    /* Main content container */
    .main-container {
        background: var(--background-primary);
        border-radius: var(--border-radius);
        padding: 2.5rem;
        box-shadow: var(--shadow-lg);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: var(--transition);
    }

    .main-container:hover {
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
    }

    /* Model selection styling */
    .model-section {
        margin-bottom: 2rem;
    }

    .model-label {
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
        display: block;
    }

    .gr-dropdown {
        border-radius: var(--border-radius) !important;
        border: 2px solid var(--border-color) !important;
        transition: var(--transition) !important;
        background: var(--background-primary) !important;
    }

    .gr-dropdown:focus-within {
        border-color: var(--primary-color) !important;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1) !important;
    }

    .gr-dropdown .options {
        background: var(--background-primary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: var(--border-radius) !important;
        box-shadow: var(--shadow-lg) !important;
    }

    .gr-dropdown .options .item {
        padding: 0.75rem 1rem !important;
        transition: var(--transition) !important;
        border-radius: 8px !important;
        margin: 0.25rem !important;
    }

    .gr-dropdown .options .item:hover {
        background-color: var(--background-secondary) !important;
        cursor: pointer;
        transform: translateY(-1px);
    }

    .gr-dropdown .options .item.selected {
        background-color: var(--primary-color) !important;
        color: white !important;
    }

    /* Input/Output sections */
    .io-section {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 2rem;
        margin-bottom: 2rem;
    }

    @media (max-width: 768px) {
        .io-section {
            grid-template-columns: 1fr;
            gap: 1.5rem;
        }
    }

    .input-section, .output-section {
        background: var(--background-secondary);
        padding: 1.5rem;
        border-radius: var(--border-radius);
        border: 1px solid var(--border-color);
        transition: var(--transition);
    }

    .input-section:hover, .output-section:hover {
        border-color: var(--primary-color);
        box-shadow: var(--shadow-md);
    }

    .section-title {
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .section-title::before {
        content: "";
        width: 4px;
        height: 20px;
        background: var(--primary-color);
        border-radius: 2px;
    }

    /* Textbox styling */
    .gr-textbox {
        border-radius: var(--border-radius) !important;
        border: 2px solid var(--border-color) !important;
        transition: var(--transition) !important;
        background: var(--background-primary) !important;
        font-size: 1rem !important;
        line-height: 1.5 !important;
    }

    .gr-textbox:focus {
        border-color: var(--primary-color) !important;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1) !important;
        outline: none !important;
    }

    .gr-textbox textarea {
        resize: vertical !important;
        min-height: 120px !important;
    }

    /* Button styling */
    .translate-button {
        background: linear-gradient(135deg, var(--primary-color), #7c3aed) !important;
        color: white !important;
        border: none !important;
        border-radius: var(--border-radius) !important;
        padding: 1rem 2rem !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        cursor: pointer !important;
        transition: var(--transition) !important;
        box-shadow: var(--shadow-md) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        position: relative !important;
        overflow: hidden !important;
    }

    .translate-button:hover {
        transform: translateY(-2px) !important;
        box-shadow: var(--shadow-lg) !important;
    }

    .translate-button:active {
        transform: translateY(0) !important;
    }

    .translate-button::before {
        content: "";
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        transition: left 0.5s;
    }

    .translate-button:hover::before {
        left: 100%;
    }

    /* Loading animation */
    .loading {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid rgba(255, 255, 255, 0.3);
        border-radius: 50%;
        border-top-color: white;
        animation: spin 1s ease-in-out infinite;
        margin-right: 0.5rem;
    }

    @keyframes spin {
        to { transform: rotate(360deg); }
    }

    /* Progress bar styling */
    .progress-bar {
        background: var(--primary-color) !important;
        border-radius: 4px !important;
        height: 4px !important;
    }

    /* Model info cards */
    .model-info {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin-top: 2rem;
        padding-top: 2rem;
        border-top: 1px solid var(--border-color);
    }

    .model-card {
        background: var(--background-secondary);
        padding: 1rem;
        border-radius: var(--border-radius);
        border: 1px solid var(--border-color);
        transition: var(--transition);
        text-align: center;
    }

    .model-card:hover {
        border-color: var(--primary-color);
        transform: translateY(-2px);
        box-shadow: var(--shadow-md);
    }

    .model-card h3 {
        color: var(--primary-color);
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
    }

    .model-card p {
        color: var(--text-secondary);
        font-size: 0.9rem;
        margin: 0;
    }

    /* Responsive design */
    @media (max-width: 1024px) {
        .gradio-container {
            padding: 1rem;
        }
        
        .main-container {
            padding: 1.5rem;
        }
        
        .header h1 {
            font-size: 2rem;
        }
    }

    @media (max-width: 640px) {
        .header {
            padding: 1.5rem;
            margin-bottom: 2rem;
        }
        
        .header h1 {
            font-size: 1.8rem;
        }
        
        .main-container {
            padding: 1rem;
        }
        
        .translate-button {
            width: 100% !important;
            padding: 0.875rem 1.5rem !important;
        }
    }

    /* Accessibility improvements */
    .sr-only {
        position: absolute;
        width: 1px;
        height: 1px;
        padding: 0;
        margin: -1px;
        overflow: hidden;
        clip: rect(0, 0, 0, 0);
        white-space: nowrap;
        border: 0;
    }

    /* Focus styles for accessibility */
    *:focus {
        outline: 2px solid var(--primary-color);
        outline-offset: 2px;
    }

    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-track {
        background: var(--background-secondary);
    }

    ::-webkit-scrollbar-thumb {
        background: var(--primary-color);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: var(--primary-hover);
    }
    """
) as demo:
    # Header section
    with gr.Column(elem_classes=["header"]):
        gr.HTML("""
            <h1>🌐 English to Vietnamese Machine Translation</h1>
            <p>Advanced AI-powered translation with multiple model options</p>
        """)

    # Main content
    with gr.Column(elem_classes=["main-container"]):
        # Model selection
        with gr.Row(elem_classes=["model-section"]):
            model_choice = gr.Dropdown(
                choices=[
                    ("Rule-Based MT (RBMT)", "rbmt"),
                    ("Statistical MT (SMT)", "smt"),
                    ("MBart50 (Neural)", "mbart50"),
                    ("mT5 (Neural)", "mt5")
                ],
                label="🤖 Select Translation Model",
                value="mbart50",
                elem_classes=["gr-dropdown"],
                info="Choose the translation approach that best fits your needs"
            )

        # Input/Output section
        with gr.Row(elem_classes=["io-section"]):
            with gr.Column(elem_classes=["input-section"]):
                gr.HTML('<div class="section-title">📝 Input Text (English)</div>')
                input_text = gr.Textbox(
                    placeholder="Enter your English text here...\n\nExample: Hello, how are you today?",
                    lines=6,
                    elem_classes=["gr-textbox"],
                    show_label=False,
                    container=False
                )
            
            with gr.Column(elem_classes=["output-section"]):
                gr.HTML('<div class="section-title">🇻🇳 Translation (Vietnamese)</div>')
                output_text = gr.Textbox(
                    placeholder="Translation will appear here...",
                    lines=6,
                    elem_classes=["gr-textbox"],
                    interactive=False,
                    show_label=False,
                    container=False
                )

        # Translate button
        translate_button = gr.Button(
            "🚀 Translate Text",
            elem_classes=["translate-button"],
            variant="primary",
            size="lg"
        )

        # Model information cards
        gr.HTML("""
            <div class="model-info">
                <div class="model-card">
                    <h3>RBMT</h3>
                    <p>Rule-based approach using linguistic rules and dictionaries</p>
                </div>
                <div class="model-card">
                    <h3>SMT</h3>
                    <p>Statistical model trained on parallel corpora</p>
                </div>
                <div class="model-card">
                    <h3>MBart50</h3>
                    <p>Facebook's multilingual BART model</p>
                </div>
                <div class="model-card">
                    <h3>mT5</h3>
                    <p>Google's multilingual T5 transformer</p>
                </div>
            </div>
        """)

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