"""
English to Vietnamese Machine Translation Web Application

This module provides a Gradio-based web interface for translating English text
to Vietnamese using multiple translation approaches including rule-based,
statistical, and neural machine translation models.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import torch
import gradio as gr

from infer import ModelLoader, DEVICE, Translator
from models.statistical_mt import LanguageModel

# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================

# Application metadata
APP_TITLE = "English to Vietnamese Machine Translation"
APP_DESCRIPTION = "Advanced AI-powered translation with multiple model options"
MAX_INPUT_LENGTH = 1000
DEFAULT_MODEL = "mbart50"

# Example questions for users to try
EXAMPLE_QUESTIONS = [
    "Hello, how are you today?",
    "What time is it now?",
    "I would like to order some food.",
    "Where is the nearest hospital?",
    "Can you help me with directions to the airport?",
    "The weather is beautiful today.",
    "I am learning Vietnamese language.",
    "Thank you for your help.",
    "How much does this cost?",
    "I need to book a hotel room.",
    "What is your favorite Vietnamese dish?",
    "I love traveling to Vietnam.",
    "Could you please speak more slowly?",
    "I don't understand what you're saying.",
    "Have a wonderful day!"
]

# Model configuration
MODEL_CHOICES = [
    ("Rule-Based MT (RBMT)", "rbmt"),
    ("Statistical MT (SMT)", "smt"),
    ("MBart50 (Neural)", "mbart50"),
    ("mT5 (Neural)", "mt5")
]

MODEL_DESCRIPTIONS = {
    "rbmt": "Rule-based approach using linguistic rules and dictionaries",
    "smt": "Statistical model trained on parallel corpora",
    "mbart50": "Facebook's multilingual BART model",
    "mt5": "Google's multilingual T5 transformer"
}

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

def setup_logging() -> logging.Logger:
    """Configure and return logger instance."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# =============================================================================
# MODEL MANAGEMENT
# =============================================================================

class ModelManager:
    """Manages loading and storage of translation models."""
    
    def __init__(self):
        """Initialize model storage."""
        self.models: Dict[str, Tuple[Any, Any]] = {
            "mbart50": (None, None),
            "mt5": (None, None),
            "rbmt": (None, None),
            "smt": (None, None)
        }
    
    def initialize_models(self, model_types: list[str] = None) -> None:
        """Initialize specified translation models.
        
        Args:
            model_types: List of model types to initialize. 
                        Defaults to all available models.
        """
        if model_types is None:
            model_types = ["mbart50", "mt5", "rbmt", "smt"]
        
        logger.info("Starting model initialization...")
        
        for model_type in model_types:
            try:
                self._load_single_model(model_type)
            except Exception as e:
                logger.error(f"Failed to initialize {model_type}: {str(e)}")
                self.models[model_type] = (None, None)
        
        logger.info("Model initialization complete.")
    
    def _load_single_model(self, model_type: str) -> None:
        """Load a single model based on its type.
        
        Args:
            model_type: Type of model to load.
        """
        logger.info(f"Loading {model_type.upper()} model...")
        
        if model_type == "mbart50":
            self.models["mbart50"] = ModelLoader.load_mbart50()
            logger.info(f"MBart50 model loaded on {DEVICE}")
            
        elif model_type == "mt5":
            self.models["mt5"] = ModelLoader.load_mt5()
            logger.info(f"MT5 model loaded on {DEVICE}")
            
        elif model_type == "rbmt":
            from models.rule_based_mt import TransferBasedMT
            self.models["rbmt"] = (TransferBasedMT(), None)
            logger.info("RBMT initialized")
            
        elif model_type == "smt":
            self.models["smt"] = (ModelLoader.load_smt(), None)
            logger.info("SMT initialized")
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def get_model(self, model_type: str) -> Tuple[Any, Any]:
        """Get model and tokenizer for specified type.
        
        Args:
            model_type: Type of model to retrieve.
            
        Returns:
            Tuple of (model, tokenizer).
        """
        return self.models.get(model_type, (None, None))
    
    def is_model_loaded(self, model_type: str) -> bool:
        """Check if a model is loaded and ready.
        
        Args:
            model_type: Type of model to check.
            
        Returns:
            True if model is loaded, False otherwise.
        """
        model, _ = self.get_model(model_type)
        return model is not None

# =============================================================================
# TRANSLATION SERVICE
# =============================================================================

class TranslationService:
    """Handles translation requests using different models."""
    
    def __init__(self, model_manager: ModelManager):
        """Initialize translation service.
        
        Args:
            model_manager: Instance of ModelManager for accessing models.
        """
        self.model_manager = model_manager
    
    def translate(self, model_type: str, input_text: str) -> str:
        """Translate input text using the selected model.
        
        Args:
            model_type: Type of model to use for translation.
            input_text: English text to translate.
            
        Returns:
            Translated Vietnamese text or error message.
        """
        # Input validation
        if not input_text or not input_text.strip():
            return "Please enter some text to translate."
        
        if len(input_text) > MAX_INPUT_LENGTH:
            return f"Input text too long. Maximum {MAX_INPUT_LENGTH} characters allowed."
        
        # Check if model is loaded
        if not self.model_manager.is_model_loaded(model_type):
            return f"Error: Model '{model_type}' is not loaded or not supported."
        
        try:
            return self._perform_translation(model_type, input_text.strip())
        except Exception as e:
            logger.error(f"Translation error with {model_type}: {str(e)}")
            return f"Translation failed: {str(e)}"
    
    def _perform_translation(self, model_type: str, text: str) -> str:
        """Perform the actual translation based on model type.
        
        Args:
            model_type: Type of model to use.
            text: Text to translate.
            
        Returns:
            Translated text.
        """
        model, tokenizer = self.model_manager.get_model(model_type)
        
        if model_type == "rbmt":
            return Translator.translate_rbmt(text)
        elif model_type == "smt":
            return Translator.translate_smt(text, model)
        elif model_type == "mbart50":
            return Translator.translate_mbart50(text, model, tokenizer)
        elif model_type == "mt5":
            return Translator.translate_mt5(text, model, tokenizer)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

# =============================================================================
# UI STYLING
# =============================================================================

def get_custom_css() -> str:
    """Return custom CSS styles for the application."""
    return """
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

    /* Clear button styling */
    .gr-button[variant="secondary"] {
        background: var(--background-secondary) !important;
        color: var(--text-secondary) !important;
        border: 2px solid var(--border-color) !important;
        border-radius: var(--border-radius) !important;
        transition: var(--transition) !important;
        font-weight: 500 !important;
    }

    .gr-button[variant="secondary"]:hover {
        background: var(--error-color) !important;
        color: white !important;
        border-color: var(--error-color) !important;
        transform: translateY(-1px) !important;
        box-shadow: var(--shadow-md) !important;
    }

    /* Examples section */
    .examples-section {
        margin: 2rem 0;
        padding: 1.5rem;
        background: var(--background-secondary);
        border-radius: var(--border-radius);
        border: 1px solid var(--border-color);
    }

    .examples-title {
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 1.1rem;
    }

    .examples-title::before {
        content: "💡";
        font-size: 1.2rem;
    }

    /* Style for example buttons */
    .gr-examples .gr-button {
        background: var(--background-primary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
        margin: 0.25rem !important;
        font-size: 0.9rem !important;
        color: var(--text-primary) !important;
        transition: var(--transition) !important;
        text-align: left !important;
        max-width: none !important;
        white-space: normal !important;
        word-wrap: break-word !important;
    }

    .gr-examples .gr-button:hover {
        background: var(--primary-color) !important;
        color: white !important;
        border-color: var(--primary-color) !important;
        transform: translateY(-1px) !important;
        box-shadow: var(--shadow-sm) !important;
    }

    .gr-examples .gr-button:active {
        transform: translateY(0) !important;
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

# =============================================================================
# UI COMPONENTS
# =============================================================================

def create_header() -> gr.HTML:
    """Create the application header."""
    return gr.HTML(f"""
        <h1>🌐 {APP_TITLE}</h1>
        <p>{APP_DESCRIPTION}</p>
    """)

def create_model_info_cards() -> gr.HTML:
    """Create model information cards."""
    cards_html = '<div class="model-info">'
    
    for display_name, model_key in MODEL_CHOICES:
        model_name = display_name.split(" ")[0]  # Extract short name
        description = MODEL_DESCRIPTIONS[model_key]
        
        cards_html += f"""
            <div class="model-card">
                <h3>{model_name}</h3>
                <p>{description}</p>
            </div>
        """
    
    cards_html += '</div>'
    return gr.HTML(cards_html)

def create_examples_section() -> gr.Examples:
    """Create examples section with sample questions."""
    return gr.Examples(
        examples=[[example] for example in EXAMPLE_QUESTIONS],
        inputs=None,  # Will be set when creating the interface
        label="💡 Try these example sentences:",
        examples_per_page=5
    )

# =============================================================================
# APPLICATION SETUP
# =============================================================================

def create_gradio_interface() -> gr.Blocks:
    """Create and configure the Gradio interface."""
    
    # Initialize services
    model_manager = ModelManager()
    translation_service = TranslationService(model_manager)
    
    # Initialize models
    model_manager.initialize_models()
    
    # Create Gradio interface
    with gr.Blocks(
        theme="soft",
        title=APP_TITLE,
        css=get_custom_css()
    ) as demo:
        
        # Header section
        with gr.Column(elem_classes=["header"]):
            create_header()
        
        # Main content
        with gr.Column(elem_classes=["main-container"]):
            
            # Model selection
            with gr.Row(elem_classes=["model-section"]):
                model_choice = gr.Dropdown(
                    choices=MODEL_CHOICES,
                    label="🤖 Select Translation Model",
                    value=DEFAULT_MODEL,
                    elem_classes=["gr-dropdown"],
                    info="Choose the translation approach that best fits your needs"
                )
            
            # Input/Output section
            with gr.Row(elem_classes=["io-section"]):
                with gr.Column(elem_classes=["input-section"]):
                    gr.HTML('<div class="section-title">📝 Input Text (English)</div>')
                    input_text = gr.Textbox(
                        placeholder="Enter your English text here or click on an example below...",
                        lines=6,
                        max_lines=10,
                        elem_classes=["gr-textbox"],
                        show_label=False,
                        container=False
                    )
                
                with gr.Column(elem_classes=["output-section"]):
                    gr.HTML('<div class="section-title">🇻🇳 Translation (Vietnamese)</div>')
                    output_text = gr.Textbox(
                        placeholder="Translation will appear here...",
                        lines=6,
                        max_lines=10,
                        elem_classes=["gr-textbox"],
                        interactive=False,
                        show_label=False,
                        container=False
                    )
            
            # Examples section
            with gr.Row(elem_classes=["examples-section"]):
                examples = gr.Examples(
                    examples=[[example] for example in EXAMPLE_QUESTIONS],
                    inputs=[input_text],
                    label="💡 Try these example sentences:",
                    examples_per_page=5,
                    run_on_click=False
                )
            
            # Action buttons
            with gr.Row():
                translate_button = gr.Button(
                    "🚀 Translate Text",
                    elem_classes=["translate-button"],
                    variant="primary",
                    size="lg",
                    scale=3
                )
                clear_button = gr.Button(
                    "🗑️ Clear",
                    variant="secondary",
                    size="lg",
                    scale=1
                )
            
            # Model information cards
            create_model_info_cards()
        
        # Bind the translation function to the button
        translate_button.click(
            fn=translation_service.translate,
            inputs=[model_choice, input_text],
            outputs=output_text,
            show_progress=True
        )
        
        # Clear button functionality
        clear_button.click(
            fn=lambda: ("", ""),
            inputs=[],
            outputs=[input_text, output_text]
        )
        
        # Also allow Enter key to trigger translation
        input_text.submit(
            fn=translation_service.translate,
            inputs=[model_choice, input_text],
            outputs=output_text,
            show_progress=True
        )
    
    return demo

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main function to run the application."""
    try:
        demo = create_gradio_interface()
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True
        )
    except Exception as e:
        logger.error(f"Failed to launch application: {str(e)}")
        raise

if __name__ == "__main__":
    main()