from diffusers_helper.hf_login import login

import os
import configparser # Added for state management

# Ensure the 'states' directory exists for saving settings
STATES_FOLDER = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './states')))
os.makedirs(STATES_FOLDER, exist_ok=True)
print(f"State settings will be saved in: {STATES_FOLDER}")

# Set HF_HOME relative to script location
#HF_HOME_FOLDER = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './hf_download')))
#os.environ['HF_HOME'] = HF_HOME_FOLDER

import gradio as gr
import torch
import traceback
import einops
import safetensors.torch as sf
import numpy as np
import argparse
import math
import time # Make sure time is imported
import atexit # For pynvml cleanup
import random # For random seed

# --- GPU Monitor Imports ---
try:
    import pynvml
    pynvml_available = True
except ImportError:
    pynvml_available = False
    print("Warning: pynvml not found. GPU stats will not be displayed. Install with: pip install nvidia-ml-py")
# --- End GPU Monitor Imports ---


from PIL import Image as PILImage # Use alias to avoid conflict
from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, state_dict_weighted_merge, state_dict_offset_merge, generate_timestamp
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
# Import necessary functions including the new unload_all_models
from diffusers_helper.memory import cpu, gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation, fake_diffusers_current_device, DynamicSwapInstaller, unload_complete_models, load_model_as_complete, unload_all_models
from diffusers_helper.thread_utils import AsyncStream, async_run
from diffusers_helper.gradio.progress_bar import make_progress_bar_css, make_progress_bar_html
from transformers import SiglipImageProcessor, SiglipVisionModel
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket

# Constants for Model Names
MODEL_ORIGINAL = "FramePack (Original)"
MODEL_F1 = "FramePack-F1"


# --- GPU Monitor Helper Function ---
def format_bytes(size_bytes):
    """Converts bytes to a human-readable format (MiB or GiB)."""
    if size_bytes is None or size_bytes == 0:
        return "0 MiB"
    size_mib = size_bytes / (1024**2)
    if size_mib >= 1024:
        size_gib = size_mib / 1024
        return f"{size_gib:.1f} GiB"
    else:
        # Display MiB with no decimal places for compactness
        return f"{size_mib:.0f} MiB"

# --- GPU Monitor Initialization ---
nvml_initialized = False
gpu_handle = None
gpu_name = "N/A" # Store GPU name here
if pynvml_available:
    try:
        print("Initializing NVML for GPU monitoring...")
        pynvml.nvmlInit()
        nvml_initialized = True
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count > 0:
            gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0) # Get handle for GPU 0
            gpu_name = pynvml.nvmlDeviceGetName(gpu_handle) # Get GPU name
            if isinstance(gpu_name, bytes):
                gpu_name = gpu_name.decode('utf-8')
            gpu_name = gpu_name.replace("NVIDIA GeForce ", "")
            print(f"NVML Initialized. Monitoring GPU 0: {gpu_name}")
        else:
            print("NVML Initialized, but no NVIDIA GPUs detected.")
            gpu_handle = None
            gpu_name = "No NVIDIA GPU"
            nvml_initialized = False
        atexit.register(pynvml.nvmlShutdown)
        print("Registered NVML shutdown hook.")
    except pynvml.NVMLError as error:
        print(f"Failed to initialize NVML: {error}. GPU stats disabled.")
        nvml_initialized = False
        gpu_handle = None
        gpu_name = "NVML Init Error"
# --- End GPU Monitor Initialization ---


parser = argparse.ArgumentParser()
parser.add_argument('--share', action='store_true')
parser.add_argument("--server", type=str, default='0.0.0.0')
parser.add_argument("--port", type=int, required=False)
parser.add_argument("--inbrowser", action='store_true')
args = parser.parse_args()

print(args)

# --- Determine VRAM Mode ---
try:
    free_mem_gb = get_cuda_free_memory_gb(gpu)
except Exception as e:
    print(f"Warning: Could not get free GPU memory: {e}. Assuming low VRAM mode.")
    free_mem_gb = 0
high_vram = free_mem_gb > 60

print(f'Free VRAM {free_mem_gb:.2f} GB')
print(f'High-VRAM Mode: {high_vram}')

# --- Constants ---
DEFAULT_SEED = 31337
MAX_SEED = 2**32 - 1
SEED_MODE_LAST = 'last'
SEED_MODE_RANDOM = 'random'
PREVIEW_TARGET_HEIGHT = 256

# --- State Management Configuration ---
SETTINGS_FILE = os.path.join(STATES_FOLDER, 'ui_settings_manager.ini')
DEFAULT_SECTION = '__Default__'
LAST_STATE_SECTION = '__LastState__'
PRESET_PREFIX = 'Preset_'
RESERVED_SECTIONS = [DEFAULT_SECTION, LAST_STATE_SECTION]


# Load models initially to CPU
print("Loading models to CPU...")
text_encoder = LlamaModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder', torch_dtype=torch.float16).cpu()
text_encoder_2 = CLIPTextModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder_2', torch_dtype=torch.float16).cpu()
tokenizer = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer')
tokenizer_2 = CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer_2')
vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16).cpu()
feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor')
image_encoder = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=torch.float16).cpu()

# Load BOTH transformer models
print("Loading Original FramePack Transformer...")
transformer_original = HunyuanVideoTransformer3DModelPacked.from_pretrained('lllyasviel/FramePackI2V_HY', torch_dtype=torch.bfloat16).cpu()
print("Loading FramePack-F1 Transformer...")
transformer_f1 = HunyuanVideoTransformer3DModelPacked.from_pretrained('lllyasviel/FramePack_F1_I2V_HY_20250503', torch_dtype=torch.bfloat16).cpu()
print("Models loaded.")

# Add both transformers to the list for potential unloading
all_core_models = [text_encoder, text_encoder_2, vae, image_encoder, transformer_original, transformer_f1]

# --- Model Setup ---
vae.eval()
text_encoder.eval()
text_encoder_2.eval()
image_encoder.eval()
transformer_original.eval()
transformer_f1.eval()

if not high_vram:
    print("Low VRAM: Enabling VAE slicing/tiling.")
    vae.enable_slicing()
    vae.enable_tiling()

# Apply settings to both transformers
for t in [transformer_original, transformer_f1]:
    t.high_quality_fp32_output_for_inference = True
    t.to(dtype=torch.bfloat16)
    t.requires_grad_(False)

print('transformer.high_quality_fp32_output_for_inference = True (applied to both)')

vae.to(dtype=torch.float16)
image_encoder.to(dtype=torch.float16)
text_encoder.to(dtype=torch.float16)
text_encoder_2.to(dtype=torch.float16)

vae.requires_grad_(False)
text_encoder.requires_grad_(False)
text_encoder_2.requires_grad_(False)
image_encoder.requires_grad_(False)

# --- Memory Management Setup ---
if not high_vram:
    print("Low VRAM mode: Enabling DynamicSwap for Transformers and Text Encoder.")
    # Install DynamicSwap on both transformers
    DynamicSwapInstaller.install_model(transformer_original, device=gpu)
    DynamicSwapInstaller.install_model(transformer_f1, device=gpu)
    DynamicSwapInstaller.install_model(text_encoder, device=gpu)
else:
    print("High VRAM mode: Preloading models to GPU.")
    try:
        text_encoder.to(gpu)
        text_encoder_2.to(gpu)
        image_encoder.to(gpu)
        vae.to(gpu)
        # Load both transformers to GPU in high VRAM mode
        transformer_original.to(gpu)
        transformer_f1.to(gpu)
        print("Models successfully moved to GPU.")
    except Exception as e:
         print(f"Error moving models to GPU: {e}. Check available VRAM.")
         high_vram = False
         print("Falling back to Low VRAM mode settings.")
         DynamicSwapInstaller.install_model(transformer_original, device=gpu)
         DynamicSwapInstaller.install_model(transformer_f1, device=gpu)
         DynamicSwapInstaller.install_model(text_encoder, device=gpu)


stream = AsyncStream()

outputs_folder = './outputs/'
os.makedirs(outputs_folder, exist_ok=True)

# --- Declare component variables globally BEFORE gr.Blocks() ---
input_image = None
end_image = None
model_selector = None # ADDED
resolution = None
prompt = None
n_prompt = None
keep_models_alive = None
keep_only_final_video = None
use_teacache = None
seed = None
random_seed_button = None
use_last_seed_button = None
total_second_length = None
steps = None
gs = None
mp4_crf = None
output_fps = None
gpu_memory_preservation = None
latent_window_size = None
cfg = None # Added for CFG Scale (Real)
rs = None  # Added for CFG Re-Scale
state_presets_dropdown = None
preset_name_textbox = None
result_video = None
preview_image = None
progress_desc = None
progress_bar = None
start_button = None
end_button = None
gpu_stats_display = None
inverted_sampling_note = None # ADDED


# --- State Management Functions --- START ---
# --- Helper Functions ---

def _read_config():
    """Safely reads the INI file from the states folder."""
    config = configparser.ConfigParser()
    os.makedirs(STATES_FOLDER, exist_ok=True)
    if os.path.exists(SETTINGS_FILE):
        try:
            config.read(SETTINGS_FILE)
        except configparser.Error as e:
            print(f"Error reading INI file '{SETTINGS_FILE}': {e}")
            gr.Warning(f"Could not read settings file: {e}")
    return config

def _write_config(config):
    """Safely writes the INI file to the states folder."""
    os.makedirs(STATES_FOLDER, exist_ok=True)
    try:
        with open(SETTINGS_FILE, 'w') as configfile:
            config.write(configfile)
    except IOError as e:
        print(f"Error writing INI file '{SETTINGS_FILE}': {e}")
        gr.Warning(f"Could not write settings file: {e}")

def _get_component_map():
    """Maps descriptive INI keys to actual Gradio component objects."""
    component_vars = {
        'model_selection': model_selector, # ADDED
        'resolution': resolution, 'prompt': prompt, 'n_prompt': n_prompt,
        'keep_models_alive': keep_models_alive, 'keep_only_final_video': keep_only_final_video,
        'use_teacache': use_teacache, 'seed': seed, 'total_second_length': total_second_length,
        'steps': steps, 'gs': gs, 'mp4_crf': mp4_crf, 'output_fps': output_fps,
        'gpu_memory_preservation': gpu_memory_preservation, 'latent_window_size': latent_window_size,
        'cfg_scale_real': cfg,
        'cfg_rescale': rs
    }
    initialized_components = {k: v for k, v in component_vars.items() if v is not None}
    if len(initialized_components) != len(component_vars):
        missing = [k for k, v in component_vars.items() if v is None]
        print(f"Warning: Components not mapped in _get_component_map (still None): {missing}")
    return initialized_components


def _get_current_ui_values(*args):
    """Captures current UI values into a settings dictionary based on component order."""
    component_map = _get_component_map()
    component_keys = list(component_map.keys())
    if len(args) != len(component_keys):
        print(f"Error: UI value count mismatch in _get_current_ui_values. Expected {len(component_keys)}, got {len(args)}")
        expected_comps = [comp.label if hasattr(comp, 'label') else str(comp) for comp in component_map.values()]
        print(f"Expected components (labels): {expected_comps}")
        raise ValueError("Internal error: Component count mismatch when getting UI values.")

    settings_dict = {}
    for key, value in zip(component_keys, args):
        if key == 'seed':
            try:
                 value = str(int(float(value)))
            except (ValueError, TypeError):
                 print(f"Warning: Could not convert seed value '{value}' to int for saving. Using default {DEFAULT_SEED}.")
                 value = str(DEFAULT_SEED)
        settings_dict[key] = str(value) if value is not None else ''
    return settings_dict

# def _apply_settings_to_ui(settings_dict):
    # """Generates Gradio updates from a settings dictionary."""
    # component_map = _get_component_map()
    # updates = {}
    # for key, component in component_map.items():
        # if key in settings_dict:
            # value_str = settings_dict[key]
            # try:
                # default_value = getattr(component, 'value', None)
                # target_type = type(default_value) if default_value is not None else str

                # # Handle model selector specifically if needed (e.g., ensure value is in choices)
                # if key == 'model_selection' and component is not None and hasattr(component, 'choices'):
                    # if value_str not in component.choices:
                        # print(f"Warning: Loaded model '{value_str}' not in available choices {component.choices}. Reverting to default '{MODEL_ORIGINAL}'.")
                        # value_str = MODEL_ORIGINAL # Fallback to default model

                # if value_str == '' and target_type != str:
                    # print(f"Info: Empty value loaded for '{key}'. Reverting to default '{default_value}'.")
                    # value = default_value
                # elif target_type == bool:
                    # value = value_str.lower() in ('true', '1', 't', 'yes', 'on')
                # elif target_type == int:
                    # value = int(float(value_str)) # Allow float conversion first for robustness
                # elif target_type == float:
                    # value = float(value_str)
                # else: # Includes string, potentially dropdown values
                    # value = value_str
                # updates[component] = gr.update(value=value)

            # except (ValueError, TypeError) as e:
                # print(f"Warning: Could not convert loaded value '{value_str}' for key '{key}' to type {target_type}. Using default. Error: {e}")
                # if default_value is not None:
                    # updates[component] = gr.update(value=default_value)
            # except Exception as e:
                 # print(f"Warning: Unexpected error processing loaded key '{key}': {e}. Skipping update.")
                 # traceback.print_exc()
        # else:
             # print(f"Info: Key '{key}' not found in loaded settings section for component '{component.label if hasattr(component,'label') else key}'.") # Improved logging
    # return updates
    
def _apply_settings_to_ui(settings_dict):
    """Generates Gradio updates from a settings dictionary."""
    component_map = _get_component_map()
    updates = {}
    for key, component in component_map.items():
        if key in settings_dict:
            value_str = settings_dict[key]
            try:
                # Use component's internal value if available, otherwise best guess
                # Note: Default value might be None for some components initially
                default_value = getattr(component, 'value', None) if component else None
                target_type = type(default_value) if default_value is not None else str

                # --- CORRECTED CHECK FOR DROPDOWN CHOICES ---
                if key == 'model_selection' and component is not None and hasattr(component, 'choices'):
                    # Extract the valid *values* from the choices list (which might be list of strings or list of tuples)
                    valid_choice_values = []
                    if component.choices:
                        if isinstance(component.choices[0], (tuple, list)): # Check if it's list of tuples/lists
                           valid_choice_values = [choice[1] for choice in component.choices if isinstance(choice, (tuple, list)) and len(choice) > 1]
                        else: # Assume it's a list of strings
                           valid_choice_values = component.choices

                    if value_str not in valid_choice_values:
                        print(f"Warning: Loaded model value '{value_str}' not in available choice values {valid_choice_values}. Reverting to default '{MODEL_ORIGINAL}'.")
                        value_str = MODEL_ORIGINAL # Fallback to default model if loaded value invalid
                # --- END OF CORRECTION ---


                # General type conversion logic
                if value_str == '' and target_type != str:
                    # Handle empty strings for non-string types by using default
                    # (avoids errors converting '' to int/float/bool)
                    print(f"Info: Empty value loaded for '{key}'. Using default '{default_value}'.")
                    value = default_value # Use component's default if possible
                elif target_type == bool:
                    # Robust boolean conversion
                    value = value_str.lower() in ('true', '1', 't', 'yes', 'on')
                elif target_type == int:
                    # Convert to int, allowing intermediate float conversion
                    value = int(float(value_str))
                elif target_type == float:
                    value = float(value_str)
                else: # Includes string, dropdown values (after validation above)
                    value = value_str

                # Create the update object only if the component exists
                if component:
                    updates[component] = gr.update(value=value)
                else:
                    print(f"Warning: Component for key '{key}' is None. Cannot create update.")


            except (ValueError, TypeError) as e:
                print(f"Warning: Could not convert loaded value '{value_str}' for key '{key}' to type {target_type}. Using default '{default_value}'. Error: {e}")
                if component and default_value is not None:
                    updates[component] = gr.update(value=default_value)
            except Exception as e:
                 print(f"Warning: Unexpected error processing loaded key '{key}': {e}. Skipping update.")
                 traceback.print_exc()
        else:
             # Log if a setting from the INI doesn't match any known component key
             print(f"Info: Key '{key}' from component map not found in loaded settings section.")

    return updates    


def _get_default_settings():
    """Returns a dictionary of the application's hardcoded default settings as strings."""
    default_preserved_mem_value = max(4.0, free_mem_gb * 0.2) if not high_vram else 8.0
    return {
        # Keys MUST match _get_component_map()
        'model_selection': MODEL_ORIGINAL, # ADDED Default Model
        'resolution': '512',
        'prompt': '',
        'n_prompt': '',
        'keep_models_alive': 'True',
        'keep_only_final_video': 'True',
        'use_teacache': 'True',
        'seed': str(DEFAULT_SEED),
        'total_second_length': '5.0',
        'steps': '25',
        'gs': '10.0',
        'mp4_crf': '18',
        'output_fps': '30',
        'gpu_memory_preservation': str(default_preserved_mem_value),
        'latent_window_size': '9',
        'cfg_scale_real': '1.0',
        'cfg_rescale': '0.0',
    }

def create_default_state_if_missing():
    """Ensures the [__Default__] section exists in the INI."""
    config = _read_config()
    needs_update = False
    default_settings_keys = set(_get_default_settings().keys()) # Use set for efficient comparison
    if DEFAULT_SECTION not in config:
        print(f"'{DEFAULT_SECTION}' not found in INI '{SETTINGS_FILE}'. Creating with defaults.")
        needs_update = True
    else:
        existing_keys = set(config[DEFAULT_SECTION].keys())
        missing_keys = default_settings_keys - existing_keys
        if missing_keys:
             print(f"'{DEFAULT_SECTION}' is missing keys: {missing_keys}. Updating with current defaults.")
             needs_update = True
        # Optional: Check for extra keys in INI default section (might indicate old settings)
        # extra_keys = existing_keys - default_settings_keys
        # if extra_keys:
        #     print(f"Warning: Extra keys found in '{DEFAULT_SECTION}': {extra_keys}. These will be ignored.")

    if needs_update:
        # Ensure all default keys are written, potentially overwriting the section
        config[DEFAULT_SECTION] = _get_default_settings()
        _write_config(config)
        return True
    return False


def get_saved_states():
    """Returns a list of available state names (Default + User Presets)."""
    config = _read_config()
    presets = ["Default"]
    user_presets = []
    for section in config.sections():
        if section.startswith(PRESET_PREFIX):
            user_presets.append(section[len(PRESET_PREFIX):])
    presets.extend(sorted(user_presets)) # Sort user presets alphabetically
    return presets

# --- Gradio Action Functions ---

def save_state_as(preset_name, *args):
    """Saves the current UI state as a new preset."""
    if not preset_name or preset_name.strip() == "":
        gr.Warning("Please enter a valid name for the preset.")
        return gr.update()

    preset_name = preset_name.strip()

    # Enhanced check for reserved/invalid names
    reserved_or_invalid = [s.lower().replace(PRESET_PREFIX, '') for s in RESERVED_SECTIONS] + ["default"]
    if preset_name.lower() in reserved_or_invalid:
         gr.Warning(f"Preset name '{preset_name}' is reserved or invalid. Please choose another name.")
         return gr.update()

    section_name = PRESET_PREFIX + preset_name
    config = _read_config()

    try:
        current_settings = _get_current_ui_values(*args)
        config[section_name] = current_settings
        _write_config(config)
        # gr.Info(f"Preset '{preset_name}' saved successfully.") # Info messages can be annoying
        new_choices = get_saved_states()
        # Return update for dropdown to refresh choices and select the newly saved preset
        return gr.update(choices=new_choices, value=preset_name)
    except ValueError as e:
         gr.Error(f"Failed to save preset: {e}")
         return gr.update() # Return update for dropdown without changing selection
    except Exception as e:
        gr.Error(f"An unexpected error occurred while saving preset '{preset_name}': {e}")
        traceback.print_exc()
        return gr.update() # Return update for dropdown without changing selection


def load_state(state_name_to_load):
    """Loads the selected state (Default or Preset) and updates the UI."""
    if not state_name_to_load:
        gr.Warning("No state selected to load.")
        return {} # Return empty dict, Gradio handles no updates

    config = _read_config()
    settings_dict = None
    section_to_load = "" # Initialize

    if state_name_to_load == "Default":
        section_to_load = DEFAULT_SECTION
        if section_to_load not in config:
            gr.Warning("Default settings section not found in file, using hardcoded defaults.")
            settings_dict = _get_default_settings()
        else:
             try:
                # Get default settings first to ensure all keys are present
                settings_dict = _get_default_settings()
                # Override with values from the INI file's default section
                settings_dict.update(dict(config.items(section_to_load)))
             except configparser.Error as e:
                 gr.Error(f"Error reading Default section: {e}. Using hardcoded defaults.")
                 settings_dict = _get_default_settings()
    else:
        section_to_load = PRESET_PREFIX + state_name_to_load
        if section_to_load not in config:
            gr.Error(f"Preset '{state_name_to_load}' not found in settings file.")
            return {}
        else:
            try:
                # Load preset values. Start with defaults to ensure all fields are present,
                # then overwrite with the preset's values. This handles presets saved
                # before new settings were added.
                settings_dict = _get_default_settings()
                settings_dict.update(dict(config.items(section_to_load)))
            except configparser.Error as e:
                 gr.Error(f"Error reading preset '{state_name_to_load}': {e}")
                 return {}

    if settings_dict:
        print(f"Loading state: {state_name_to_load} (Section: {section_to_load})")
        updates = _apply_settings_to_ui(settings_dict)
        # gr.Info(f"State '{state_name_to_load}' loaded.")
        return updates # Return the dictionary of updates
    else:
        gr.Error(f"Could not retrieve settings for '{state_name_to_load}'.")
        return {}

def delete_state(state_name_to_delete):
    """Deletes a user-saved preset."""
    if not state_name_to_delete or state_name_to_delete == "Default":
        gr.Warning("Cannot delete 'Default' or empty selection.")
        return gr.update() # Return update for dropdown without change

    section_to_delete = PRESET_PREFIX + state_name_to_delete
    config = _read_config()

    if section_to_delete not in config:
        gr.Warning(f"Preset '{state_name_to_delete}' not found, cannot delete.")
        new_choices = get_saved_states() # Refresh choices in case UI is out of sync
        return gr.update(choices=new_choices)

    try:
        config.remove_section(section_to_delete)
        _write_config(config)
        # gr.Info(f"Preset '{state_name_to_delete}' deleted.")
        new_choices = get_saved_states()
        # Refresh choices and set dropdown value back to Default
        return gr.update(choices=new_choices, value="Default")
    except Exception as e:
        gr.Error(f"Failed to delete preset '{state_name_to_delete}': {e}")
        traceback.print_exc()
        new_choices = get_saved_states() # Refresh choices even on error
        return gr.update(choices=new_choices) # Keep current selection on error

def save_last_state(*args):
    """Saves the current UI state to the [__LastState__] section."""
    config = _read_config()
    try:
        # Use the same function as saving presets to get current values
        current_settings = _get_current_ui_values(*args)
        config[LAST_STATE_SECTION] = current_settings
        _write_config(config)
        print("Last UI state saved.")
    except ValueError as e:
         print(f"Error saving last state (ValueError): {e}") # More specific error
    except Exception as e:
        print(f"Error saving last state: {e}")
        traceback.print_exc()

def load_last_or_default_state():
    """Loads the last state if available, otherwise loads default. Returns update dict and source."""
    config = _read_config()
    settings_dict = None
    loaded_source = "Unknown"

    if LAST_STATE_SECTION in config:
        print(f"Found '{LAST_STATE_SECTION}', attempting to load it.")
        try:
            # Start with defaults to ensure all keys exist
            settings_dict = _get_default_settings()
            # Update with last session values from INI
            settings_dict.update(dict(config.items(LAST_STATE_SECTION)))
            loaded_source = "Last Session"
            print(f"Successfully parsed '{LAST_STATE_SECTION}'.")
        except configparser.Error as e:
            print(f"Error reading {LAST_STATE_SECTION}: {e}. Falling back.")
            settings_dict = None # Ensure fallback if parsing fails
        except Exception as e: # Catch other potential errors during update
            print(f"Unexpected error processing {LAST_STATE_SECTION}: {e}. Falling back.")
            traceback.print_exc()
            settings_dict = None


    # Fallback to Default section if last state failed or wasn't present
    if settings_dict is None:
        if DEFAULT_SECTION in config:
            print(f"Loading '{DEFAULT_SECTION}' as fallback.")
            try:
                 # Start with hardcoded defaults again
                 settings_dict = _get_default_settings()
                 # Update with INI defaults
                 settings_dict.update(dict(config.items(DEFAULT_SECTION)))
                 loaded_source = "Default (from INI)"
                 print(f"Successfully parsed '{DEFAULT_SECTION}'.")
            except configparser.Error as e:
                print(f"Error reading {DEFAULT_SECTION}: {e}. Falling back to hardcoded.")
                settings_dict = None # Ensure fallback
            except Exception as e:
                 print(f"Unexpected error processing {DEFAULT_SECTION}: {e}. Falling back to hardcoded.")
                 traceback.print_exc()
                 settings_dict = None
        else:
             # Default section missing entirely
             print(f"'{DEFAULT_SECTION}' not found/readable in INI. Will use hardcoded defaults.")
             settings_dict = None # Ensure hardcoded defaults are used next

    # Final fallback: hardcoded defaults if everything else failed
    if settings_dict is None:
        print("Using hardcoded defaults as final fallback.")
        settings_dict = _get_default_settings()
        loaded_source = "Hardcoded Default"
        # Attempt to create the default INI section if we ended up here
        create_default_state_if_missing()

    # Apply the determined settings to the UI
    updates = _apply_settings_to_ui(settings_dict)
    return updates, loaded_source

def initialize_settings_and_ui():
    """Called on Gradio load. Ensures defaults, loads last state, updates UI."""
    print("Initializing UI settings...")
    create_default_state_if_missing() # Ensure default section exists first
    initial_updates, loaded_source = load_last_or_default_state() # Load state (dict of updates)
    saved_states_list = get_saved_states() # Get preset list for dropdown

    print(f"Initialization complete. Loaded state from: {loaded_source}. States available: {saved_states_list}")
    # gr.Info(f"Settings loaded from: {loaded_source}") # User feedback

    # Prepare the dropdown update separately
    # Default value should ideally be based on loaded state, but 'Default' is safe fallback
    dropdown_value = "Default"
    # Try to find the loaded preset name if applicable
    if loaded_source.startswith("Preset_"):
         dropdown_value = loaded_source[len(PRESET_PREFIX):]
    elif loaded_source == "Last Session":
         # If last session was loaded, dropdown should probably show 'Default'
         # unless we want to compare last session to presets and select match? Too complex.
         dropdown_value = "Default" # Or keep whatever value was loaded? 'Default' is less confusing.

    dropdown_update = gr.update(choices=saved_states_list, value=dropdown_value)

    # Add the dropdown update to the dictionary of updates
    # Ensure state_presets_dropdown is not None before assignment
    if state_presets_dropdown is not None:
        initial_updates[state_presets_dropdown] = dropdown_update
    else:
         print("Warning: state_presets_dropdown component is None during initialization.")


    # Return the complete dictionary of updates for all components
    return initial_updates

# --- State Management Functions --- END ---


@torch.no_grad()
def worker(
    selected_model_name, # ADDED: To know which logic to use
    input_image, end_image, prompt, n_prompt, seed, total_second_length,
    latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation,
    use_teacache, mp4_crf, output_fps, resolution, keep_models_alive,
    keep_only_final_video
):
    """The main generation worker function, adapted for model switching."""
    job_id = generate_timestamp()
    output_filename = None
    previous_output_filename = None
    dynamic_swap_installed = False # Track if swap was installed *for this run*

    # --- Select Active Model ---
    is_f1_model = selected_model_name == MODEL_F1
    if is_f1_model:
        active_transformer = transformer_f1
        print(f"Worker: Using FramePack-F1 model.")
    else:
        active_transformer = transformer_original
        print(f"Worker: Using FramePack (Original) model.")

    # --- Core Setup (Common) ---
    try:
        current_seed = int(seed)
        print(f"Worker using Seed: {current_seed}")

        # Calculate total sections based on the selected model's typical FPS or slider
        # F1 original code used 30fps hardcoded, but let's use the slider for flexibility
        total_latent_sections = (total_second_length * output_fps) / (latent_window_size * 4)
        total_latent_sections = int(max(round(total_latent_sections), 1))
        print(f"Planned generation: {total_second_length} seconds @ {output_fps} FPS, {total_latent_sections} sections.")
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Starting generation...'))))

        # --- Low VRAM Dynamic Swap Check ---
        # Check if swap needs to be applied specifically for the active model if not already done
        if not high_vram:
             # Check if the *active* transformer has the swap attribute
             if not hasattr(active_transformer, 'forge_backup_original_class'):
                 print(f"Applying DynamicSwap for {selected_model_name} (worker check)...")
                 DynamicSwapInstaller.install_model(active_transformer, device=gpu)
                 # Also ensure text encoder swap is applied if needed
                 if not hasattr(text_encoder, 'forge_backup_original_class'):
                      DynamicSwapInstaller.install_model(text_encoder, device=gpu)
                 dynamic_swap_installed = True # Mark swap installed for *this run* on the active model
             else:
                 print(f"DynamicSwap already present for {selected_model_name}.")
                 # Check text encoder separately if it wasn't installed above
                 if not hasattr(text_encoder, 'forge_backup_original_class'):
                     DynamicSwapInstaller.install_model(text_encoder, device=gpu)


        # --- Text Encoding (Common) ---
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Encoding text prompt...'))))
        if not high_vram:
            fake_diffusers_current_device(text_encoder, gpu) # Use swap if installed
            load_model_as_complete(text_encoder_2, target_device=gpu, unload=True) # Offload TE2
        else:
            text_encoder.to(gpu)
            text_encoder_2.to(gpu)

        llama_vec, clip_l_pooler = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)
        # Convert cfg (potentially float if loaded from state) to float for comparison
        if float(cfg) == 1.0:
            llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
        else:
            llama_vec_n, clip_l_pooler_n = encode_prompt_conds(n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)
        print("Text encoding complete.")

        # --- Image Processing (Common) ---
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Processing start frame...'))))
        if input_image is None:
            raise ValueError("Start Frame image is missing.")
        img_h, img_w, _ = input_image.shape
        # Use resolution slider value for bucket finding in both models
        height, width = find_nearest_bucket(img_h, img_w, resolution=resolution)
        print(f"Input resolution {img_w}x{img_h}, selected bucket {width}x{height}")
        input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)
        PILImage.fromarray(input_image_np).save(os.path.join(outputs_folder, f'{job_id}_start.png'))
        input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1.0
        input_image_pt = input_image_pt.permute(2, 0, 1).unsqueeze(0).unsqueeze(2) # Shape: [1, C, 1, H, W]
        print(f"Start frame processed. Shape: {input_image_pt.shape}")

        # End image processing (only for original model logic path)
        has_end_image = end_image is not None and not is_f1_model # F1 logic doesn't use end image
        if has_end_image:
            stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Processing end frame...'))))
            H_end, W_end, C_end = end_image.shape
            end_image_np = resize_and_center_crop(end_image, target_width=width, target_height=height)
            PILImage.fromarray(end_image_np).save(os.path.join(outputs_folder, f'{job_id}_end.png'))
            end_image_pt = torch.from_numpy(end_image_np).float() / 127.5 - 1.0
            end_image_pt = end_image_pt.permute(2, 0, 1).unsqueeze(0).unsqueeze(2)
            print(f"End frame processed. Shape: {end_image_pt.shape}")
        else:
             end_image_np = None # Ensure it's None if not used
             end_image_pt = None


        # --- VAE Encoding (Common) ---
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'VAE encoding frames...'))))
        if not high_vram:
            load_model_as_complete(vae, target_device=gpu, unload=True) # Unload VAE after use
        else:
            vae.to(gpu)

        # Note: VAE input expects [B, C, T, H, W], our input_image_pt is [1, C, 1, H, W]
        start_latent = vae_encode(input_image_pt, vae) # Output shape: [1, 16, 1, H//8, W//8]
        print(f"Start latent encoded. Shape: {start_latent.shape}")
        if has_end_image:
            end_latent = vae_encode(end_image_pt, vae)
            print(f"End latent encoded. Shape: {end_latent.shape}")
        else:
            end_latent = None

        # --- CLIP Vision Encoding (Common, but merging differs) ---
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'CLIP Vision encoding...'))))
        if not high_vram:
            load_model_as_complete(image_encoder, target_device=gpu, unload=True) # Unload ImgEncoder after use
        else:
            image_encoder.to(gpu)

        # Get start image embedding
        image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state

        # Handle end image embedding ONLY for original model
        if has_end_image:
            end_image_encoder_output = hf_clip_vision_encode(end_image_np, feature_extractor, image_encoder)
            end_image_encoder_last_hidden_state = end_image_encoder_output.last_hidden_state
            # Merge embeddings only if end image exists (original model path)
            image_encoder_last_hidden_state = (image_encoder_last_hidden_state + end_image_encoder_last_hidden_state) / 2.0
            print("Combined start/end frame CLIP Vision embeddings.")
        else:
            print("Using start frame CLIP Vision embedding.") # F1 always uses this path


        # --- Dtype Conversion (Common) ---
        target_transformer_dtype = active_transformer.dtype
        llama_vec = llama_vec.to(target_transformer_dtype)
        llama_vec_n = llama_vec_n.to(target_transformer_dtype)
        clip_l_pooler = clip_l_pooler.to(target_transformer_dtype)
        clip_l_pooler_n = clip_l_pooler_n.to(target_transformer_dtype)
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(target_transformer_dtype)

        # --- Sampling Setup (Common) ---
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Preparing for sampling...'))))
        rnd = torch.Generator("cpu").manual_seed(current_seed)
        num_frames_in_window = latent_window_size * 4 - 3
        history_pixels = None
        total_generated_latent_frames = 0 # Track generated frames across sections

        # ==============================================================
        # --- MODEL-SPECIFIC SAMPLING LOGIC ---
        # ==============================================================

        if is_f1_model:
            # --- FramePack-F1 Forward Sampling Logic ---
            print("Starting F1 Forward Sampling Loop...")
            # F1 history includes conditioning frames: 1 start + 2 context + 16 noise/history
            # We initialize with zeros for the history part and append the start latent
            # History shape: [B, C_latent, T_hist, H//8, W//8]
            # T_hist = 16 (4x) + 2 (2x) + 1 (1x/start) = 19 conditioning frames
            # history_latents = torch.zeros(size=(1, 16, 16 + 2 + 1, height // 8, width // 8), dtype=torch.float32, device=cpu) # Original F1 init - might be wrong

            # Append the actual start latent (needs correct slicing/indexing)
            # Start_latent shape: [1, 16, 1, H//8, W//8]
            # History_latents shape after cat: [1, 16, 19+1, H//8, W//8] ? No, should replace placeholder.
            # Let's re-evaluate F1 init: it seems history grows.
            # Initialize empty and append start latent first.
            history_latents = start_latent.to(device=cpu, dtype=torch.float32) # Shape: [1, 16, 1, H//8, W//8]
            total_generated_latent_frames = 1 # Start frame counts as first frame

            for section_index in range(total_latent_sections):
                if stream.input_queue.top() == 'end':
                    print("Stop signal received. Ending generation early.")
                    stream.output_queue.push(('end', output_filename)) # Send last successful file
                    return

                print(f"\n--- Starting F1 Section {section_index + 1}/{total_latent_sections} ---")

                stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, f'Loading Transformer (Section {section_index + 1})...'))))
                if not high_vram:
                    # Unload potentially inactive models first
                    inactive_transformer = transformer_original if is_f1_model else transformer_f1
                    unload_complete_models(vae, image_encoder, text_encoder, text_encoder_2, inactive_transformer)
                    move_model_to_device_with_memory_preservation(active_transformer, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)
                else:
                     active_transformer.to(gpu) # Ensure active model is on GPU

                if use_teacache:
                    active_transformer.initialize_teacache(enable_teacache=True, num_steps=steps)
                    print("TeaCache Enabled.")
                else:
                    active_transformer.initialize_teacache(enable_teacache=False)
                    print("TeaCache Disabled.")

                def callback_f1(d):
                    if stream.input_queue.top() == 'end':
                        stream.output_queue.push(('end', None))
                        raise KeyboardInterrupt('User requested stop during sampling.')

                    current_step = d['i'] + 1
                    percentage = int(100.0 * current_step / steps)
                    hint = f'Sampling Step {current_step}/{steps}'

                    # Calculate frames based on total_generated_latent_frames *before* this section + progress in current section
                    # Simpler F1 calc: total_generated_latent_frames * 4 - 3 (adjust for current preview)
                    current_preview_latent_frames = d['denoised'].shape[2] if d.get('denoised') is not None else 0
                    # Estimate frames including preview progress
                    current_total_frames = max(0, (total_generated_latent_frames + current_preview_latent_frames) * 4 - 3)

                    current_video_seconds = max(0, current_total_frames / float(output_fps))
                    desc = f'F1 Section {section_index + 1}/{total_latent_sections}. Est. Length: {current_video_seconds:.2f}s ({current_total_frames} frames). Extending video...'

                    preview_resized_np = None
                    if d.get('denoised') is not None:
                        try:
                            preview_tensor = vae_decode_fake(d['denoised']) # Use fake decode for speed
                            preview_np = (preview_tensor * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
                            preview_np = einops.rearrange(preview_np, 'b c t h w -> (b h) (t w) c')

                            # Resize preview
                            h_orig, w_orig, _ = preview_np.shape
                            if h_orig > 0 and w_orig > 0:
                                aspect_ratio = w_orig / h_orig
                                target_width = int(PREVIEW_TARGET_HEIGHT * aspect_ratio)
                                target_width = max(1, target_width)
                                pil_img = PILImage.fromarray(preview_np)
                                resized_pil_img = pil_img.resize((target_width, PREVIEW_TARGET_HEIGHT), PILImage.Resampling.LANCZOS)
                                preview_resized_np = np.array(resized_pil_img)
                            else:
                                preview_resized_np = preview_np
                        except Exception as e:
                             print(f"Warning: F1 Preview generation/resizing failed - {e}")
                             preview_resized_np = None # Don't show broken preview

                    stream.output_queue.push(('progress', (preview_resized_np, desc, make_progress_bar_html(percentage, hint))))
                    return

                # --- Prepare F1 Conditioning Latents ---
                # Indices splitting for F1 conditioning
                # Total indices = start_frame_idx + 4x_hist_indices + 2x_hist_indices + 1x_hist_indices + noise_indices
                indices = torch.arange(0, sum([1, 16, 2, 1, latent_window_size]), device=cpu).unsqueeze(0)

                clean_latent_indices_start, clean_latent_4x_indices, clean_latent_2x_indices, clean_latent_1x_indices, latent_indices = \
                    indices.split([1, 16, 2, 1, latent_window_size], dim=1)
                # Clean latents are start_frame + 1x_history
                clean_latent_indices = torch.cat([clean_latent_indices_start, clean_latent_1x_indices], dim=1)

                # Get history latents from the end of the current history_latents tensor
                # History needs 16 (4x), 2 (2x), 1 (1x) frames = 19 total conditioning frames from past
                num_cond_frames = 16 + 2 + 1
                if history_latents.shape[2] >= num_cond_frames:
                    conditioning_latents = history_latents[:, :, -num_cond_frames:, :, :]
                    clean_latents_4x, clean_latents_2x, clean_latents_1x = conditioning_latents.split([16, 2, 1], dim=2)
                else:
                    # Handle edge case: Not enough history yet (should only happen theoretically before first frame is fully added?)
                    # Pad with start frame latent. F1 code seems to assume history is sufficient.
                    print(f"Warning: History length {history_latents.shape[2]} < {num_cond_frames}. Padding conditioning latents with start frame.")
                    padding_needed = num_cond_frames - history_latents.shape[2]
                    # Repeat start latent along the time dimension
                    padding_tensor = start_latent.repeat(1, 1, padding_needed, 1, 1).to(history_latents.device, history_latents.dtype)
                    conditioning_latents = torch.cat([padding_tensor, history_latents], dim=2)
                    # Now split the padded conditioning latents
                    clean_latents_4x, clean_latents_2x, clean_latents_1x = conditioning_latents.split([16, 2, 1], dim=2)


                # Clean latents input = start_latent + 1x history frame
                # Ensure tensors are on the same device and dtype before concatenating
                clean_latents = torch.cat([start_latent.to(device=clean_latents_1x.device, dtype=clean_latents_1x.dtype), clean_latents_1x], dim=2)

                print(f"F1 Conditioning shapes: clean_latents {clean_latents.shape}, 2x {clean_latents_2x.shape}, 4x {clean_latents_4x.shape}")

                # --- F1 Sampling Call ---
                print(f"Starting F1 sampling for {num_frames_in_window} target output frames...")
                stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, f'Sampling F1 Section {section_index + 1}...'))))

                # Prepare args for sample_hunyuan
                device_kwargs = {'device': gpu, 'dtype': torch.bfloat16} # Default compute device/dtype
                text_kwargs = {'device': gpu, 'dtype': target_transformer_dtype} # Dtype for text embeds
                latent_kwargs = {'device': gpu, 'dtype': torch.bfloat16} # Dtype for latent inputs

                generated_latents = sample_hunyuan(
                    transformer=active_transformer, sampler='unipc', width=width, height=height,
                    frames=num_frames_in_window, # Target output frames
                    real_guidance_scale=float(cfg), distilled_guidance_scale=float(gs),
                    guidance_rescale=float(rs),
                    num_inference_steps=steps, generator=rnd,
                    prompt_embeds=llama_vec.to(**text_kwargs), prompt_embeds_mask=llama_attention_mask.to(gpu),
                    prompt_poolers=clip_l_pooler.to(**text_kwargs),
                    negative_prompt_embeds=llama_vec_n.to(**text_kwargs), negative_prompt_embeds_mask=llama_attention_mask_n.to(gpu),
                    negative_prompt_poolers=clip_l_pooler_n.to(**text_kwargs),
                    image_embeddings=image_encoder_last_hidden_state.to(**text_kwargs),
                    # F1 specific conditioning:
                    latent_indices=latent_indices.to(gpu),
                    clean_latents=clean_latents.to(**latent_kwargs),
                    clean_latent_indices=clean_latent_indices.to(gpu),
                    clean_latents_2x=clean_latents_2x.to(**latent_kwargs),
                    clean_latent_2x_indices=clean_latent_2x_indices.to(gpu),
                    clean_latents_4x=clean_latents_4x.to(**latent_kwargs),
                    clean_latent_4x_indices=clean_latent_4x_indices.to(gpu),
                    callback=callback_f1,
                    # Common args
                    device=gpu, dtype=torch.bfloat16, # Specify main compute dtype
                )

                # --- F1 Post-Sampling ---
                generated_latents = generated_latents.to(device=cpu, dtype=torch.float32) # Move to CPU for storage/decode
                print(f"F1 Sampling complete for section {section_index + 1}. Generated latent shape: {generated_latents.shape}") # Shape: [1, 16, T_gen, H//8, W//8]

                # Append generated latents to history
                history_latents = torch.cat([history_latents, generated_latents], dim=2)
                num_new_latent_frames = generated_latents.shape[2]
                total_generated_latent_frames += num_new_latent_frames # Accumulate count *after* generation
                print(f"F1 History updated. Total latent frames accumulated: {total_generated_latent_frames}. History tensor shape: {history_latents.shape}")

                # --- F1 VAE Decoding ---                 
                stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, f'Decoding Section {section_index + 1}...'))))
                if not high_vram:
                    offload_model_from_device_for_memory_preservation(active_transformer, target_device=gpu, preserved_memory_gb=8)
                    load_model_as_complete(vae, target_device=gpu, unload=False) # Keep VAE loaded
                else:
                     vae.to(gpu)                     
                    
                real_history_latents = history_latents.to(gpu, vae.dtype) # Move the relevant part to GPU

                if history_pixels is None:
                    # First section decode: Decode the entire history accumulated so far
                    print(f"F1 First decode: Decoding history of shape {real_history_latents.shape}")
                    history_pixels = vae_decode(real_history_latents, vae).cpu()
                    print(f"Initialized F1 pixel history with shape: {history_pixels.shape}")
                else:
                    # Subsequent sections: Decode only the *end slice* needed for soft append
                    # Slice size based on standalone worker: latent_window_size * 2 latents
                    num_latents_to_decode_slice = latent_window_size * 2
                    # Make sure slice index isn't negative
                    decode_start_index = max(0, real_history_latents.shape[2] - num_latents_to_decode_slice)
                    latents_to_decode_slice = real_history_latents[:, :, decode_start_index:, :, :]

                    print(f"F1 Subsequent decode: Decoding slice of shape {latents_to_decode_slice.shape} from index {decode_start_index}")
                    current_pixels_slice = vae_decode(latents_to_decode_slice, vae).cpu()
                    print(f"Decoded pixel slice shape: {current_pixels_slice.shape}")

                    # Soft append the *decoded slice* to the existing *pixel history*
                    # Overlap based on standalone worker/F1 paper
                    append_overlap_pixels = latent_window_size * 4 - 3
                    print(f"Soft appending F1 section with pixel overlap: {append_overlap_pixels} frames")
                    history_pixels = soft_append_bcthw(history_pixels, current_pixels_slice, append_overlap_pixels)
                    print(f"F1 Pixel history appended. New shape: {history_pixels.shape}")                    
                    
                if not high_vram:
                    vae.to(cpu) # Offload VAE after decode
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                    print("VAE offloaded.")

                # --- F1 Save Video ---
                # This part remains the same, saving the updated history_pixels
                output_filename = os.path.join(outputs_folder, f'{job_id}_f1_{history_pixels.shape[2]}_frames.mp4')
                final_num_frames = history_pixels.shape[2]
                print(f"Saving F1 video: {output_filename} ({final_num_frames} frames)")
                save_bcthw_as_mp4(history_pixels, output_filename, fps=output_fps, crf=mp4_crf)
                stream.output_queue.push(('file', output_filename))

                # Clean up intermediate files if requested
                if keep_only_final_video and previous_output_filename is not None:
                    try:
                        os.remove(previous_output_filename)
                        print(f"Deleted previous intermediate video: {previous_output_filename}")
                    except OSError as e:
                        print(f"Warning: Could not delete intermediate video {previous_output_filename}: {e}")
                previous_output_filename = output_filename

            # --- End of F1 Sampling Loop ---

        else:
            # --- Original FramePack Inverted Sampling Logic ---
            print("Starting Original Inverted Sampling Loop...")
            # Original history size calculation (different conditioning structure)
            # Shape: [B, C_latent, T_hist = 1(post) + 2(2x) + 16(4x), H//8, W//8]
            history_latents = torch.zeros(size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32, device=cpu)
            # total_generated_latent_frames initialized to 0

            # Padding calculation for inverted sampling
            latent_paddings = list(reversed(range(total_latent_sections)))
            if total_latent_sections > 4:
                latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]
            print(f"Using latent padding sequence: {latent_paddings}")

            for i, latent_padding in enumerate(latent_paddings):
                section_index = i + 1
                is_last_section = (latent_padding == 0)
                is_first_section = (i == 0)
                latent_padding_size = latent_padding * latent_window_size

                if stream.input_queue.top() == 'end':
                    print("Stop signal received. Ending generation early.")
                    stream.output_queue.push(('end', output_filename))
                    return

                print(f"\n--- Starting Original Section {section_index}/{len(latent_paddings)} ---")
                print(f"Padding: {latent_padding} ({latent_padding_size} frames), Last: {is_last_section}, First: {is_first_section}")

                stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, f'Loading Transformer (Section {section_index})...'))))
                if not high_vram:
                    # Unload potentially inactive models first
                    inactive_transformer = transformer_original if is_f1_model else transformer_f1
                    unload_complete_models(vae, image_encoder, text_encoder, text_encoder_2, inactive_transformer)
                    move_model_to_device_with_memory_preservation(active_transformer, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)
                else:
                     active_transformer.to(gpu)

                if use_teacache:
                    active_transformer.initialize_teacache(enable_teacache=True, num_steps=steps)
                    print("TeaCache Enabled.")
                else:
                    active_transformer.initialize_teacache(enable_teacache=False)
                    print("TeaCache Disabled.")

                def callback_original(d):
                    if stream.input_queue.top() == 'end':
                        stream.output_queue.push(('end', None))
                        raise KeyboardInterrupt('User requested stop during sampling.')
                    current_step = d['i'] + 1
                    percentage = int(100.0 * current_step / steps)
                    hint = f'Sampling Step {current_step}/{steps}'
                    # Original calculation: current total frames = previous history + current denoised (approx)
                    current_preview_pixel_frames = d['denoised'].shape[2] * 4 if d.get('denoised') is not None else 0
                    # This estimate is tricky with prepending, show based on *current* history_pixels length
                    current_total_frames = (history_pixels.shape[2] if history_pixels is not None else 0) + current_preview_pixel_frames

                    current_video_seconds = max(0, current_total_frames / float(output_fps))
                    desc = f'Original Section {section_index}/{len(latent_paddings)}. Est. Final Length: {current_video_seconds:.2f}s ({current_total_frames} frames).'

                    preview_resized_np = None
                    if d.get('denoised') is not None:
                        try:
                            preview_tensor = vae_decode_fake(d['denoised'])
                            preview_np = (preview_tensor * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
                            preview_np = einops.rearrange(preview_np, 'b c t h w -> (b h) (t w) c')
                            # Resize preview
                            h_orig, w_orig, _ = preview_np.shape
                            if h_orig > 0 and w_orig > 0:
                                aspect_ratio = w_orig / h_orig
                                target_width = int(PREVIEW_TARGET_HEIGHT * aspect_ratio)
                                target_width = max(1, target_width)
                                pil_img = PILImage.fromarray(preview_np)
                                resized_pil_img = pil_img.resize((target_width, PREVIEW_TARGET_HEIGHT), PILImage.Resampling.LANCZOS)
                                preview_resized_np = np.array(resized_pil_img)
                            else:
                                preview_resized_np = preview_np
                        except Exception as e:
                             print(f"Warning: Original Preview generation/resizing failed - {e}")
                             preview_resized_np = None
                    stream.output_queue.push(('progress', (preview_resized_np, desc, make_progress_bar_html(percentage, hint))))
                    return

                # --- Prepare Original Conditioning Latents ---
                # Total indices = pre_idx + padding + noise_idx + post_idx + 2x_hist_idx + 4x_hist_idx
                total_indices = sum([1, latent_padding_size, latent_window_size, 1, 2, 16])
                indices = torch.arange(0, total_indices, device=cpu).unsqueeze(0)
                clean_latent_indices_pre, _, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = \
                    indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
                # Original clean latents: pre (start) + post (history/end)
                clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

                clean_latents_pre = start_latent.to(device=cpu, dtype=history_latents.dtype)
                # Original history structure: [post_frame, 2x_hist, 4x_hist] sliced from the *start* of history tensor
                current_history_post, current_history_2x, current_history_4x = \
                    history_latents[:, :, :1 + 2 + 16].split([1, 2, 16], dim=2)

                # Use end latent for post conditioning ONLY if available AND first section
                if has_end_image and is_first_section:
                     print("Using end latent for conditioning in the first section.")
                     clean_latents_post_frame = end_latent.to(device=cpu, dtype=history_latents.dtype)
                else:
                     # Use the previously generated frame (which is the first frame in history_latents)
                     clean_latents_post_frame = current_history_post

                clean_latents = torch.cat([clean_latents_pre.to(clean_latents_post_frame.device, clean_latents_post_frame.dtype), clean_latents_post_frame], dim=2)
                clean_latents_2x = current_history_2x
                clean_latents_4x = current_history_4x
                print(f"Original Conditioning shapes: clean_latents {clean_latents.shape}, 2x {clean_latents_2x.shape}, 4x {clean_latents_4x.shape}")

                # --- Original Sampling Call ---
                print(f"Starting original sampling for {num_frames_in_window} target output frames...")
                stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, f'Sampling Original Section {section_index}...'))))

                device_kwargs = {'device': gpu, 'dtype': torch.bfloat16}
                text_kwargs = {'device': gpu, 'dtype': target_transformer_dtype}
                latent_kwargs = {'device': gpu, 'dtype': torch.bfloat16}

                generated_latents = sample_hunyuan(
                    transformer=active_transformer, sampler='unipc', width=width, height=height,
                    frames=num_frames_in_window,
                    real_guidance_scale=float(cfg), distilled_guidance_scale=float(gs),
                    guidance_rescale=float(rs),
                    num_inference_steps=steps, generator=rnd,
                    prompt_embeds=llama_vec.to(**text_kwargs), prompt_embeds_mask=llama_attention_mask.to(gpu),
                    prompt_poolers=clip_l_pooler.to(**text_kwargs),
                    negative_prompt_embeds=llama_vec_n.to(**text_kwargs), negative_prompt_embeds_mask=llama_attention_mask_n.to(gpu),
                    negative_prompt_poolers=clip_l_pooler_n.to(**text_kwargs),
                    image_embeddings=image_encoder_last_hidden_state.to(**text_kwargs),
                    # Original specific conditioning:
                    latent_indices=latent_indices.to(gpu),
                    clean_latents=clean_latents.to(**latent_kwargs), # Combined start + post
                    clean_latent_indices=clean_latent_indices.to(gpu),
                    clean_latents_2x=clean_latents_2x.to(**latent_kwargs),
                    clean_latent_2x_indices=clean_latent_2x_indices.to(gpu),
                    clean_latents_4x=clean_latents_4x.to(**latent_kwargs),
                    clean_latent_4x_indices=clean_latent_4x_indices.to(gpu),
                    callback=callback_original,
                    # Common args
                    device=gpu, dtype=torch.bfloat16,
                )

                # --- Original Post-Sampling ---
                generated_latents = generated_latents.to(device=cpu, dtype=torch.float32)
                print(f"Original Sampling complete for section {section_index}. Latent shape: {generated_latents.shape}")

                # Prepend generated latents to history (inverted)
                if is_last_section:
                    # In the very last section (which generates the *start* frames), prepend the true start latent
                    print("Prepending start latent to the final generated segment (Original - last section).")
                    generated_latents = torch.cat([start_latent.to(generated_latents.device, generated_latents.dtype), generated_latents], dim=2)

                history_latents = torch.cat([generated_latents, history_latents], dim=2) # Prepend
                # total_generated_latent_frames += generated_latents.shape[2] # This count isn't very useful here
                print(f"Original History updated by prepending. History tensor shape: {history_latents.shape[2]} latents.")

                # --- Original VAE Decoding ---
                stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, f'Decoding Section {section_index}...'))))
                if not high_vram:
                    offload_model_from_device_for_memory_preservation(active_transformer, target_device=gpu, preserved_memory_gb=8)
                    load_model_as_complete(vae, target_device=gpu, unload=False) # Keep VAE loaded
                else:
                     vae.to(gpu)

                # Decode appropriate part for soft append
                if is_first_section: # First section (generates end frames)
                    latents_to_decode = generated_latents # Decode just the newly generated end part
                else:
                    # Decode the newly generated part (now at the start of history) + overlap from previous (which is now later in the tensor)
                    overlap_latent_frames = latent_window_size # Number of latent frames for overlap
                    # Decode from the start of history up to generated_latents size + overlap
                    end_idx_for_overlap = generated_latents.shape[2] + overlap_latent_frames
                    end_idx_for_overlap = min(end_idx_for_overlap, history_latents.shape[2]) # Clamp to history size
                    latents_to_decode = history_latents[:, :, :end_idx_for_overlap, :, :]

                print(f"Original Decoding latents slice of shape: {latents_to_decode.shape} (from index 0)")
                current_pixels_section = vae_decode(latents_to_decode.to(gpu, vae.dtype), vae)
                current_pixels_section = current_pixels_section.cpu()
                print(f"VAE decoding complete for section {section_index}. Decoded pixel slice shape: {current_pixels_section.shape}")

                # --- Original Soft Append (Prepending) ---
                if history_pixels is None:
                    history_pixels = current_pixels_section
                    print("Initialized pixel history.")
                else:
                    append_overlap_pixels = latent_window_size * 4
                    new_pixel_frames_in_section = generated_latents.shape[2] * 4
                    actual_overlap = min(append_overlap_pixels, current_pixels_section.shape[2] - new_pixel_frames_in_section)
                    actual_overlap = max(0, actual_overlap)
                    print(f"Soft appending with pixel overlap: {actual_overlap} frames (Theoretical max: {append_overlap_pixels})")
                    history_pixels = soft_append_bcthw(current_pixels_section, history_pixels, actual_overlap)
                    print("Pixel history appended.")


                if not high_vram:
                    vae.to(cpu)
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                    print("VAE offloaded.")

                # --- Original Save Video ---
                # Save the *current* complete pixel history (which grows from end to start)
                output_filename = os.path.join(outputs_folder, f'{job_id}_orig_{history_pixels.shape[2]}_frames.mp4')
                final_num_frames = history_pixels.shape[2]
                print(f"Saving Original video: {output_filename} ({final_num_frames} frames)")
                save_bcthw_as_mp4(history_pixels, output_filename, fps=output_fps, crf=mp4_crf)
                stream.output_queue.push(('file', output_filename))

                if keep_only_final_video and previous_output_filename is not None:
                    try:
                        os.remove(previous_output_filename)
                        print(f"Deleted previous intermediate video: {previous_output_filename}")
                    except OSError as e:
                        print(f"Warning: Could not delete intermediate video {previous_output_filename}: {e}")
                previous_output_filename = output_filename

                if is_last_section:
                    print("\n--- Reached last section (Original). Generation finished. ---")
                    break
            # --- End of Original Sampling Loop ---

        # ==============================================================
        # --- END OF MODEL-SPECIFIC LOGIC ---
        # ==============================================================

    except KeyboardInterrupt:
         print("\nOperation cancelled by user during execution.")
    except Exception as e:
        print("\n--- ERROR DURING GENERATION ---")
        traceback.print_exc()
        stream.output_queue.push(('progress', (None, f'Error: {type(e).__name__}', make_progress_bar_html(100, 'Error! Check Logs.'))))
    finally:
        print("\n--- Running Final Cleanup ---")
        stream.output_queue.push(('end', output_filename)) # Send the *last generated* filename

        if not keep_models_alive:
            print("Unloading models as 'Keep Models in Memory' is unchecked.")
            # Pass *all* models that might be loaded
            unload_all_models(all_core_models)
        else:
            print("Keeping models in memory as requested.")
            # In high VRAM, ensure *all* core models are back on GPU if they were moved
            if high_vram:
                 print("High VRAM mode: Ensuring models are on GPU post-run...")
                 models_on_gpu = 0
                 for m in all_core_models:
                     try:
                         p = next(m.parameters(), None)
                         if p is not None and p.device != gpu:
                             m.to(gpu)
                             print(f"Moved {m.__class__.__name__} back to GPU.")
                         models_on_gpu += 1
                     except RuntimeError as e:
                          # Handle specific CUDA error in subprocesses gracefully
                          if "Cannot re-initialize CUDA in forked subprocess" in str(e):
                              print(f"Known issue: Cannot move {m.__class__.__name__} back to GPU in subprocess. Skipping.")
                          else:
                              # Re-raise other runtime errors or handle them
                              print(f"Warning: Runtime error ensuring {m.__class__.__name__} is on GPU: {e}")
                     except Exception as e:
                         print(f"Warning: Could not ensure {m.__class__.__name__} is on GPU: {e}")
                 # Check if the count matches (excluding potential subprocess skips)
                 # if models_on_gpu == len(all_core_models):
                 #     print("All models confirmed/moved to GPU (or skipped due to subprocess limitation).")
                 # else:
                 #     print(f"{models_on_gpu}/{len(all_core_models)} models confirmed/moved to GPU.")

                 if torch.cuda.is_available():
                     torch.cuda.empty_cache()
            else:
                 print("Low VRAM mode: Models managed by DynamicSwap or loaded/unloaded as needed.")

        # Uninstall DynamicSwap IF it was installed *during this specific run* for the *active* model
        # Check both transformers if swap was potentially installed on them
        if dynamic_swap_installed and not high_vram:
             print("Uninstalling DynamicSwap that was installed for this run...")
             uninstalled_count = 0
             try:
                 # Check and uninstall from active transformer
                 if hasattr(active_transformer, 'forge_backup_original_class'):
                     DynamicSwapInstaller.uninstall_model(active_transformer)
                     print(f"DynamicSwap uninstalled from active model: {selected_model_name}")
                     uninstalled_count += 1
                 # Check and uninstall from text encoder (only if installed this run - simple check)
                 if hasattr(text_encoder, 'forge_backup_original_class'):
                      # This assumes if swap exists, it was installed this run if dynamic_swap_installed is True
                      DynamicSwapInstaller.uninstall_model(text_encoder)
                      print("DynamicSwap uninstalled from text encoder.")
                      uninstalled_count += 1

                 # Check and uninstall from the *inactive* transformer if it somehow got installed too
                 # This is less likely with current logic but safer
                 inactive_transformer = transformer_original if is_f1_model else transformer_f1
                 if hasattr(inactive_transformer, 'forge_backup_original_class'):
                     print(f"Warning: DynamicSwap found on inactive transformer ({inactive_transformer.__class__.__name__}). Uninstalling.")
                     DynamicSwapInstaller.uninstall_model(inactive_transformer)
                     uninstalled_count += 1

             except Exception as e:
                 print(f"Warning: Error during DynamicSwap uninstall - {e}")
             print(f"DynamicSwap uninstall attempt finished. Uninstalled {uninstalled_count} components.")
        elif not high_vram:
            print("DynamicSwap was not marked as installed during this run, or high VRAM mode active. Skipping uninstall.")


        print("Cleanup finished.")

    return


# --- Seed Button Functions ---
def randomize_seed_internal():
    """Internal function to just get a random seed value."""
    return random.randint(0, MAX_SEED)

# Define CSS classes for button states
base_button_class = "seed-button"
active_button_class = "seed-button-active"
# inactive_button_class = "seed-button-inactive" # Optional: for specific inactive styling



def set_seed_mode_random():
    """Sets mode to random, updates button styles, and generates/sets new seed."""
    print("Seed mode set to RANDOM")
    new_seed = randomize_seed_internal()
    print(f"Generated and set random seed: {new_seed}")
    return {
        seed_mode_state: SEED_MODE_RANDOM,
        random_seed_button: gr.update(elem_classes=[base_button_class, active_button_class]),
        use_last_seed_button: gr.update(elem_classes=[base_button_class]), # Deactivate last seed button
        seed: new_seed # Update the seed number input
    }

def set_seed_mode_last(last_seed_value_state):
    """Sets mode to last, updates button styles, and sets seed input from state."""
    print("Seed mode set to LAST")
    seed_update = gr.update() # Default to no change
    try:
        # Ensure last_seed_value_state is treated as int if possible
        last_seed_int = int(float(last_seed_value_state)) # Allow conversion from float/str
        if 0 <= last_seed_int <= MAX_SEED:
            print(f"Setting seed input to last seed: {last_seed_int}")
            seed_update = gr.update(value=last_seed_int) # Update the seed number input
        else:
            print(f"Last seed value {last_seed_int} out of range [0, {MAX_SEED}]. Not setting input.")
    except (ValueError, TypeError):
        print(f"Invalid last seed value '{last_seed_value_state}' in state. Not setting input.")

    return {
        seed_mode_state: SEED_MODE_LAST,
        random_seed_button: gr.update(elem_classes=[base_button_class]), # Deactivate random button
        use_last_seed_button: gr.update(elem_classes=[base_button_class, active_button_class]), # Activate last seed button
        seed: seed_update # Apply the seed value update (or no change if error)
    }
# --- End Seed Button Functions ---


def process(
    # Standard inputs (order must match process_inputs definition exactly)
    model_selector_val, # ADDED
    input_image_val, end_image_val,
    resolution_val, prompt_val, n_prompt_val, keep_models_alive_val, keep_only_final_video_val,
    use_teacache_val, seed_input_val, total_second_length_val, steps_val, gs_val, mp4_crf_val, output_fps_val,
    gpu_memory_preservation_val, latent_window_size_val, cfg_val, rs_val,
    # State inputs
    current_seed_mode_val,
):
    """Handles the Gradio button click, starts the worker, yields updates, determines and stores the seed."""
    global stream
    if input_image_val is None:
        gr.Warning("Please provide a Start Frame image.")
        # Return updates for all outputs expected by the button click
        return {
            result_video: None, preview_image: gr.update(visible=False),
            progress_desc: '', progress_bar: '',
            start_button: gr.update(interactive=True), end_button: gr.update(interactive=False),
            seed: gr.update(), last_seed_value: gr.update()
        }

    # Handle optional end image based on model
    if model_selector_val == MODEL_F1 and end_image_val is not None:
        gr.Info("FramePack-F1 model selected. The End Frame image will be ignored.")
        end_image_val = None # Ensure end image is None for F1 worker path

    # Seed processing (remains the same)
    try:
        actual_seed = int(seed_input_val)
        if not (0 <= actual_seed <= MAX_SEED):
             gr.Warning(f"Seed '{seed_input_val}' must be between 0 and {MAX_SEED}. Using default: {DEFAULT_SEED}")
             actual_seed = DEFAULT_SEED
    except (ValueError, TypeError):
         gr.Warning(f"Invalid seed value '{seed_input_val}'. Using default: {DEFAULT_SEED}")
         actual_seed = DEFAULT_SEED

    print(f"--- Starting New Generation Request --- Model: {model_selector_val}, Seed: {actual_seed} (Mode: {current_seed_mode_val})")

    # Yield initial updates to disable start button, enable end button, set seed, store last seed
    yield {
        result_video: None, preview_image: gr.update(visible=False),
        progress_desc: '', progress_bar: '',
        start_button: gr.update(interactive=False), end_button: gr.update(interactive=True),
        seed: gr.update(value=actual_seed), # Show the actual seed being used
        last_seed_value: actual_seed # Store the actual seed used in state
    }

    stream = AsyncStream()

    # Pass all parameters to the worker - ORDER MATTERS
    async_run(worker,
              model_selector_val, # Pass selected model name
              input_image_val, end_image_val, prompt_val, n_prompt_val, actual_seed, total_second_length_val,
              latent_window_size_val, steps_val,
              cfg_val, gs_val, rs_val,
              gpu_memory_preservation_val,
              use_teacache_val, mp4_crf_val, output_fps_val, resolution_val, keep_models_alive_val,
              keep_only_final_video_val)

    output_filename = None

    # Stream handling loop (remains the same)
    while True:
        # Use timeout to prevent blocking indefinitely if worker hangs?
        # flag, data = stream.output_queue.next(timeout=1.0) # timeout in seconds
        flag, data = stream.output_queue.next() # No timeout

        if flag is None: # Handle timeout case if using timeout
            # print("Stream timeout...")
            continue

        if flag == 'file':
            output_filename = data
            # Yield update including the new filename
            yield {
                result_video: output_filename, # Show the latest generated file
                preview_image: gr.update(visible=True), # Keep preview potentially visible? Or hide? Let's keep it for now.
                progress_desc: gr.update(), # Keep last description
                progress_bar: gr.update(), # Keep last progress bar state
                start_button: gr.update(interactive=False), # Still generating
                end_button: gr.update(interactive=True), # Still stoppable
                seed: gr.update(), last_seed_value: gr.update() # No change to seeds
            }
        elif flag == 'progress':
            preview, desc, html = data
            # Yield update with progress details
            yield {
                result_video: output_filename, # Show last known video file
                preview_image: gr.update(visible=preview is not None, value=preview), # Update preview
                progress_desc: desc, # Update description
                progress_bar: html, # Update progress bar
                start_button: gr.update(interactive=False), # Still generating
                end_button: gr.update(interactive=True), # Still stoppable
                seed: gr.update(), last_seed_value: gr.update() # No change to seeds
            }
        elif flag == 'end':
             final_filename = data if data else output_filename # Use filename from 'end' if provided, else last 'file'
             print(f"Process received end signal. Final file: {final_filename}")
             # Yield final update: show final video, hide preview, clear progress, enable start, disable end
             yield {
                 result_video: final_filename,
                 preview_image: gr.update(visible=False),
                 progress_desc: '',
                 progress_bar: '',
                 start_button: gr.update(interactive=True),
                 end_button: gr.update(interactive=False),
                 seed: gr.update(), last_seed_value: gr.update() # No change to seeds
             }
             break # Exit the loop
        else:
            print(f"Warning: Unknown flag received from worker: {flag}")


def end_process():
    """Sends the stop signal when the End Generation button is clicked."""
    print("End button clicked. Sending 'end' signal to worker.")
    if 'stream' in globals() and stream is not None and hasattr(stream, 'input_queue'):
        try:
            stream.input_queue.push('end')
        except Exception as e:
            print(f"Error pushing 'end' signal: {e}")
    else:
        print("Warning: Stream object or input queue not found, cannot send end signal.")
    # Just update the button state, the worker will handle the actual stop
    return gr.update(interactive=False)

# --- GPU Monitor Function ---
def get_gpu_stats_text():
    """Fetches GPU stats using pynvml and formats them for stable display."""
    global gpu_name # Allow updating gpu_name if NVML init fails later
    if not nvml_initialized or gpu_handle is None:
        return f"GPU ({gpu_name}): N/A"

    try:
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
        util_rates = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)
        temp_gpu = pynvml.nvmlDeviceGetTemperature(gpu_handle, pynvml.NVML_TEMPERATURE_GPU)

        if mem_info.total > 0:
            mem_usage_percent = (mem_info.used / mem_info.total) * 100
        else:
            mem_usage_percent = 0

        used_mem_str = format_bytes(mem_info.used)
        total_mem_str = format_bytes(mem_info.total)

        # Format string with alignment for better readability
        stats_str = (
            f"{gpu_name:<15} | " # Left align GPU name
            f"Mem: {mem_usage_percent:>5.1f}% ({used_mem_str:>8s} / {total_mem_str:<8s}) | " # Right align %, used; left align total
            f"Util: {util_rates.gpu:>3d}% | " # Right align util
            f"Temp: {temp_gpu:>3d}°C" # Right align temp
        )
        return stats_str
    except pynvml.NVMLError as error:
        # Handle specific non-critical errors silently if needed
        if 'NVML_ERROR_GPU_IS_LOST' in str(error) or 'NVML_ERROR_TIMEOUT' in str(error):
             # Don't spam console for these potentially temporary errors
             pass
        else:
             # Log other NVML errors
             print(f"NVML Error fetching GPU stats: {error}")
        # Potentially try re-init or indicate error state
        # For now, just return error message
        gpu_name = "NVML Error" # Update status
        return f"GPU ({gpu_name}): Error"
    except Exception as e:
        print(f"Unexpected error fetching GPU stats: {e}")
        gpu_name = "Error" # Update status
        return f"GPU ({gpu_name}): Error"


# --- Gradio UI Definition ---

quick_prompts = [
    'The girl dances gracefully, with clear movements, full of charm.',
    'A character doing some simple body movements.',
    'Astronaut floating in space overlooking earth.',
    'Robot walking through a futuristic city street.',
    'A dog chasing its tail in a park.',
    'Time-lapse clouds moving across a blue sky.',
]
quick_prompts = [[x] for x in quick_prompts]

# CSS definition
css = make_progress_bar_css() + f"""
/* Style the GPU stats text box */
#gpu-stats-display textarea {{
    text-align: left !important; font-family: 'Consolas', 'Monaco', 'monospace';
    font-weight: bold !important; font-size: 0.85em !important; line-height: 1.2 !important;
    min-height: 20px !important; padding: 3px 6px !important; color: #c0c0c0 !important;
    border: none !important; background: transparent !important; white-space: pre;
}}
/* Adjust spacing for header row items */
#header-row > .wrap {{ gap: 5px !important; align-items: center !important; }}
/* Base style seed buttons */
.{base_button_class} {{
    min-width: 30px !important; max-width: 50px !important; height: 30px !important;
    margin: 0 2px !important; padding: 0 5px !important; line-height: 0 !important;
    border: 1px solid #555 !important; background-color: #333 !important; color: #eee !important;
    transition: background-color 0.2s ease, border-color 0.2s ease;
}}
/* Active style for seed buttons */
.{active_button_class} {{
    border: 1px solid #aef !important; background-color: #446 !important; color: #fff !important;
}}
/* Ensure preview image fits width and doesn't overflow */
#preview-image img {{
    width: 100% !important; height: {PREVIEW_TARGET_HEIGHT}px !important;
    object-fit: contain !important;
}}
/* Style state management group */
#state-management-group {{
    border: 1px solid #444; padding: 10px; margin-bottom: 15px; border-radius: 5px;
}}
/* Style the output column for state management placement */
#output-column {{
    display: flex;
    flex-direction: column;
}}
/* Style for the inverted sampling note */
#inverted-sampling-note {{
    font-size: 0.9em; color: #bbb; margin-top: 10px;
}}
"""


block = gr.Blocks(css=css, theme=gr.themes.Soft()).queue()
# block = gr.Blocks(css=css, theme=gr.themes.Monochrome()).queue() # Alternative Dark theme

with block:
    # --- Gradio State ---
    last_seed_value = gr.State(value=DEFAULT_SEED)
    seed_mode_state = gr.State(value=SEED_MODE_LAST) # Default to using last seed

    # --- Title (Optional: could be made dynamic) ---
    main_title = gr.Markdown("<h1><center>FramePack I2V Video Generation</center></h1>")

    # --- Header Row (GPU Stats - remains same) ---
    with gr.Row(elem_id="header-row"):
        with gr.Column(scale=5):
             # Adjusted description
             gr.Markdown("Generate video from start/end frames (Original model) or start frame (F1 model) and prompt.")
        with gr.Column(scale=4, min_width=400):
            gpu_stats_display = gr.Textbox(
                value=f"GPU ({gpu_name}): Initializing...", label="GPU Stats", show_label=False,
                interactive=False, elem_id="gpu-stats-display"
            )

    with gr.Row():
        # --- Input Column (Left Pane) ---
        with gr.Column(scale=1):
            gr.Markdown("## Inputs")

            # --- ADD Model Selector ---
            model_selector = gr.Dropdown(
                label="Select Model",
                choices=[MODEL_ORIGINAL, MODEL_F1],
                value=MODEL_ORIGINAL, # Default to original
                info="Choose between original FramePack (uses end frame) and F1 (ignores end frame)."
            )

            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(sources=['upload'], type="numpy", label="Start Frame", height=320)
                with gr.Column():
                    # Add info about when end frame is used
                    end_image = gr.Image(sources=['upload'], type="numpy", label="End Frame (Optional - Used by Original Model)", height=320)

            resolution = gr.Slider(label="Output Resolution (Width)", minimum=256, maximum=768, value=512, step=16, info="Nearest bucket (~WxH) will be used.")
            prompt = gr.Textbox(label="Prompt", placeholder="Describe the desired action...", lines=2)
            n_prompt = gr.Textbox(label="Negative Prompt", value="", placeholder="Describe elements to avoid...", lines=2)
            example_quick_prompts = gr.Dataset(samples=quick_prompts, label='Prompt Examples', samples_per_page=len(quick_prompts), components=[prompt])

            with gr.Row():
                start_button = gr.Button("Generate Video", variant="primary", scale=3)
                end_button = gr.Button("Stop Generation", interactive=False, scale=1)

            with gr.Accordion("Advanced Settings", open=True):
                keep_models_alive = gr.Checkbox(label="Keep Models in GPU Memory After Generation", value=True, info="Recommended for faster subsequent runs.")
                keep_only_final_video = gr.Checkbox(label="Keep Only Final Video", value=True, info="Deletes intermediate videos.")
                use_teacache = gr.Checkbox(label='Use TeaCache Optimization', value=True, info='Faster, might affect fine details.')

                with gr.Row(equal_height=True):
                    seed = gr.Number(label="Seed", value=DEFAULT_SEED, precision=0, minimum=0, maximum=MAX_SEED, step=1, info="Current seed value.", scale=4)
                    random_seed_button = gr.Button("🎲", scale=1, elem_classes=[base_button_class])
                    use_last_seed_button = gr.Button("♻️", scale=1, elem_classes=[base_button_class, active_button_class]) # Start active as default mode is LAST

                total_second_length = gr.Slider(label="Target Video Length (Seconds)", minimum=1.0, maximum=120.0, value=5.0, step=0.1)
                steps = gr.Slider(label="Sampling Steps", minimum=4, maximum=60, value=25, step=1)
                gs = gr.Slider(label="Guidance Scale (Distilled)", minimum=1.0, maximum=20.0, value=10.0, step=0.1)
                mp4_crf = gr.Slider(label="MP4 Quality (CRF)", minimum=0, maximum=51, value=18, step=1, info="Lower=better quality/larger file (18=high, 23=medium)")
                output_fps = gr.Slider(label="Output FPS", minimum=5, maximum=60, value=30, step=1)
                default_preserved_mem_value = max(4.0, free_mem_gb * 0.2) if not high_vram else 8.0
                gpu_memory_preservation = gr.Slider(label="Low VRAM: Min Free GPU Mem (GB)", minimum=1.0, maximum=20.0, value=default_preserved_mem_value, step=0.5, info="Stops loading parts if free mem drops below this.")

                latent_window_size = gr.Slider(label="Latent Window Size", minimum=1, maximum=33, value=9, step=1, visible=True)
                cfg = gr.Slider(label="CFG Scale (Real)", minimum=1.0, maximum=32.0, value=1.0, step=0.01, visible=True)
                rs = gr.Slider(label="CFG Re-Scale", minimum=0.0, maximum=1.0, value=0.0, step=0.01, visible=True)

        # --- Output Column (Right Pane) ---
        with gr.Column(scale=1, elem_id="output-column"):
            gr.Markdown("## Outputs")
            result_video = gr.Video(label="Generated Video", autoplay=True, show_share_button=False, height=512, loop=True, interactive=False, elem_id="result-video")
            preview_image = gr.Image(label="Sampling Preview", height=PREVIEW_TARGET_HEIGHT, visible=False, interactive=False, elem_id="preview-image")
            progress_desc = gr.Markdown("", elem_classes='progress-text', elem_id="progress-desc")
            progress_bar = gr.HTML("", elem_classes='progress-bar-container', elem_id="progress-bar")

            # --- State Management UI (Remains same place) ---
            with gr.Group(elem_id="state-management-group"):
                 gr.Markdown("#### Settings Presets")
                 with gr.Row():
                      state_presets_dropdown = gr.Dropdown(label="Load Preset", choices=["Default"], value="Default", scale=3)
                      load_preset_button = gr.Button("📂 Load", scale=1)
                      delete_preset_button = gr.Button("🗑️ Delete", scale=1, variant="stop")
                 with gr.Row():
                      preset_name_textbox = gr.Textbox(label="Save Preset As:", placeholder="Enter preset name...", scale=3)
                      save_preset_button = gr.Button("💾 Save", scale=1)

            # --- ADD Inverted Sampling Note (Visibility controlled by model) ---
            inverted_sampling_note = gr.Markdown(
                "ℹ️ **Note (Original Model):** Uses inverted sampling. End frame actions may appear during initial generation steps.",
                visible=True, # Start visible as Original is default
                elem_id="inverted-sampling-note"
            )


    # --- Footer  ---
    gr.HTML("""
    <div style="text-align:center; margin-top:20px; font-size:0.9em; color:#555;">
        Model: <a href="https://huggingface.co/lllyasviel/FramePackI2V_HY" target="_blank">FramePackI2V_HY</a> /
               <a href="https://huggingface.co/lllyasviel/FramePack_F1_I2V_HY_20250503" target="_blank">FramePack-F1</a> |
        Based on <a href="https://huggingface.co/Tencent-Hunyuan/HunyuanDiT" target="_blank">Hunyuan-DiT</a> | UI: Gradio
    </div>
    <div style="text-align:center; margin-top:5px; font-size:0.9em; color:#555;">
        <a href="https://x.com/search?q=framepack&f=live" target="_blank">#FramePack on X (Twitter)</a>
    </div>
    """)

# --- Define Component Lists (Update with model_selector) ---
    # List of components managed by state saving/loading
    components_for_state = [
        model_selector, # ADDED
        resolution, prompt, n_prompt, keep_models_alive, keep_only_final_video,
        use_teacache, seed, total_second_length, steps, gs, mp4_crf, output_fps,
        gpu_memory_preservation, latent_window_size,
        cfg, rs
    ]
    # List of ALL inputs to the 'process' function (order matters!)
    process_inputs = [
        model_selector, # ADDED
        input_image, end_image,
        resolution, prompt, n_prompt, keep_models_alive, keep_only_final_video,
        use_teacache, seed, total_second_length, steps, gs, mp4_crf, output_fps,
        gpu_memory_preservation, latent_window_size, cfg, rs, # These match components_for_state[1:]
        seed_mode_state, # State input
    ]
    # List of ALL outputs from the 'process' function (yield dictionary keys match these)
    process_outputs_list = [
        result_video, preview_image, progress_desc, progress_bar,
        start_button, end_button,
        seed, last_seed_value # Output the possibly updated seed and the last seed state
    ]
    # --- Define the ORDERED list of output components for initial load FIRST ---
    # Needs all state components + dropdown + components changed by model selection
    all_initial_load_outputs = components_for_state + [state_presets_dropdown, inverted_sampling_note, end_image]


    # --- Event Listeners / Connections ---

    # Quick prompts 
    example_quick_prompts.click(lambda x: x[0], inputs=[example_quick_prompts], outputs=prompt, show_progress=False, queue=False)

    # --- ADD Listener for Model Change ---
    def update_ui_on_model_change(selected_model):
        """Updates UI elements based on the selected model."""
        is_f1 = selected_model == MODEL_F1
        is_original = not is_f1

        # Update note visibility
        note_update = gr.update(visible=is_original)
        # Update end image label
        end_image_label_update = gr.update(label="End Frame (Optional - Used by Original Model)" if is_original else "End Frame (Ignored by F1 Model)")

        print(f"Model changed to: {selected_model}. Updating UI elements.") # Log change

        # Return dictionary of updates
        return {
            inverted_sampling_note: note_update,
            end_image: end_image_label_update,
            # Add other UI updates here if needed (e.g., disable end_image upload for F1?)
        }

    model_selector.change(
        fn=update_ui_on_model_change,
        inputs=model_selector,
        outputs=[inverted_sampling_note, end_image] # Match keys in returned dict
    )

    # State Management (now includes model_selector via components_for_state)
    save_preset_button.click(
        fn=save_state_as,
        inputs=[preset_name_textbox] + components_for_state, # Pass current values of all state components
        outputs=[state_presets_dropdown] # Update dropdown list
    )
    load_preset_button.click(
        fn=load_state,
        inputs=[state_presets_dropdown], # Input is the name of the state to load
        outputs=components_for_state # Outputs are the components to update
    ).then( # ADDED: Trigger UI update *after* loading state based on the loaded model value
        fn=update_ui_on_model_change,
        inputs=model_selector, # Input is the model_selector component (which now has the loaded value)
        outputs=[inverted_sampling_note, end_image] # Outputs are the UI elements affected by model choice
    )
    delete_preset_button.click(
        fn=delete_state,
        inputs=[state_presets_dropdown], # Input is the preset name to delete
        outputs=[state_presets_dropdown] # Output refreshes the dropdown
    )

    # Wrapper for generation to include saving last state before starting
    def run_generation_and_save_state(*all_process_inputs_tuple):
        """Saves the last UI state and then starts the generation process."""
        print("run_generation_and_save_state called.")
        # Map component objects in components_for_state to their indices in process_inputs
        process_inputs_map = {comp: idx for idx, comp in enumerate(process_inputs)}
        state_values_to_save = []
        error_occurred = False
        for comp in components_for_state:
            if comp in process_inputs_map:
                comp_index = process_inputs_map[comp]
                # Ensure index is within bounds of the passed tuple
                if comp_index < len(all_process_inputs_tuple):
                    state_values_to_save.append(all_process_inputs_tuple[comp_index])
                else:
                    print(f"Error: Index {comp_index} out of bounds for component {comp} in input tuple.")
                    error_occurred = True
                    break # Stop trying to gather values
            else:
                print(f"Error: Component {comp} not found in process_inputs map for state saving.")
                error_occurred = True
                break # Stop trying to gather values

        if not error_occurred and len(state_values_to_save) == len(components_for_state):
            save_last_state(*state_values_to_save) # Save state if values gathered successfully
        elif not error_occurred:
             print(f"Error: Mismatch saving state. Expected {len(components_for_state)} values, got {len(state_values_to_save)}.")
             # Optionally show a Gradio warning here?
             # gr.Warning("Failed to save current settings before generation.")
        else:
            # Error already printed
             pass
             # Optionally show warning
             # gr.Warning("Failed to save current settings due to internal error.")

        # Proceed with generation regardless of save state success? Or stop? Let's proceed.
        yield from process(*all_process_inputs_tuple)


    # Generate/Stop Buttons
    start_button.click(
        fn=run_generation_and_save_state, # Use the wrapper function
        inputs=process_inputs, # Pass all inputs needed by the wrapper (and subsequently by process)
        outputs=process_outputs_list # Outputs expected by the process function's yields
    )
    end_button.click(
        fn=end_process,
        inputs=None, # No input needed
        outputs=[end_button] # Just update the end button state
    )

    # Seed Buttons 
    random_seed_button.click(
        fn=set_seed_mode_random,
        inputs=None,
        outputs=[seed_mode_state, random_seed_button, use_last_seed_button, seed]
    )
    use_last_seed_button.click(
        fn=set_seed_mode_last,
        inputs=[last_seed_value], # Input is the state holding the last seed value
        outputs=[seed_mode_state, random_seed_button, use_last_seed_button, seed]
    )


# --- Scheduled Initial Load and GPU Monitor ---
    # Define the initial load function separately for clarity
    # REMOVE the @block.load decorator from here
    def initial_load_sequence():
        print("Running initial UI load sequence...")
        # 1. Load settings into a dictionary of component->update pairs
        initial_setting_updates_dict, loaded_source = load_last_or_default_state()
        print(f"Settings loaded from: {loaded_source}")

        # 2. Get saved presets and prepare dropdown update
        saved_states_list = get_saved_states()
        # Determine the value for the dropdown based on loaded settings
        loaded_dropdown_value = "Default" # Fallback
        if state_presets_dropdown in initial_setting_updates_dict:
             update_obj = initial_setting_updates_dict[state_presets_dropdown]
             if isinstance(update_obj, dict) and 'value' in update_obj:
                 loaded_dropdown_value = update_obj['value']
             elif hasattr(update_obj, 'value'): # Check older versions
                 loaded_dropdown_value = update_obj.value
        # Create the final dropdown update object
        dropdown_update = gr.update(choices=saved_states_list, value=loaded_dropdown_value)
        if state_presets_dropdown is not None:
             initial_setting_updates_dict[state_presets_dropdown] = dropdown_update
        else:
             print("Warning: state_presets_dropdown is None during initial load sequence.")


        # 3. Determine the loaded model value from the initial updates dict
        loaded_model_value = MODEL_ORIGINAL # Default assumption
        if model_selector is not None and model_selector in initial_setting_updates_dict:
            update_obj = initial_setting_updates_dict[model_selector]
            # Use the same corrected check for older Gradio update objects/dicts
            if isinstance(update_obj, dict) and 'value' in update_obj:
                loaded_model_value = update_obj['value']
                print(f"Model value loaded from settings dictionary: {loaded_model_value}")
            elif hasattr(update_obj, 'value'): # Check older versions
                 loaded_model_value = update_obj.value
                 print(f"Model value loaded from object attribute: {loaded_model_value}")
            else:
                 print(f"Warning: Could not extract loaded model value. Using default '{loaded_model_value}'.")
        else:
             print(f"Warning: Model selector update not found. Assuming default '{loaded_model_value}'.")

        # 4. Get UI updates based on the determined loaded model
        ui_updates_for_model_dict = update_ui_on_model_change(loaded_model_value)
        print("UI updates based on loaded model determined.")

        # 5. Merge all update dictionaries
        final_updates_dict = {**initial_setting_updates_dict, **ui_updates_for_model_dict}
        print("Initial settings and model-based UI updates merged into dictionary.")

        # 6. Create the ORDERED LIST of updates based on all_initial_load_outputs
        ordered_update_values = []
        print(f"Building ordered updates for {len(all_initial_load_outputs)} components...")
        for component in all_initial_load_outputs:
            # Find the update value for this component in the final dictionary
            # If a component wasn't explicitly updated, use a default gr.update() for no change.
            update_value = final_updates_dict.get(component, gr.update())
            ordered_update_values.append(update_value)
            # Debug print (optional)
            # print(f"  - Appending update for: {component.label if hasattr(component, 'label') else component}")


        print(f"Returning ordered list of {len(ordered_update_values)} updates.")
        # Ensure the length matches exactly
        if len(ordered_update_values) != len(all_initial_load_outputs):
             print(f"CRITICAL ERROR: Mismatch between output component list ({len(all_initial_load_outputs)}) and returned update list ({len(ordered_update_values)})!")
             # You might want to raise an error here or return a list of Nones of the correct length
             # to prevent the Gradio error, although the UI state will be wrong.
             # return [gr.update()] * len(all_initial_load_outputs) # Example recovery (bad state)

        return ordered_update_values # Return the ordered list
    # --- END of initial_load_sequence function definition ---


    # Connect the initial load sequence to the block load event
    # The outputs list MUST match the order of the returned list from the function
    block.load(
        fn=initial_load_sequence,
        inputs=None,
        outputs=all_initial_load_outputs # Pass the defined list of output components
    )

    # GPU Monitor 
    if nvml_initialized:
        try:
            # Use every= if available in the Gradio version
            block.load(fn=get_gpu_stats_text, inputs=None, outputs=gpu_stats_display, every=2)
            print("Using block.load(every=2) for GPU stats.")
        except TypeError:
            # Fallback to generator loop if 'every' is not supported
            print("Warning: block.load(every=N) not supported. Using generator fallback for GPU stats.")
            def update_gpu_display_generator():
                print("Starting GPU stats update loop (generator)...")
                while True:
                    stats_text = get_gpu_stats_text()
                    yield gr.update(value=stats_text)
                    time.sleep(2) # Update every 2 seconds
            block.load(update_gpu_display_generator, inputs=None, outputs=gpu_stats_display)


# --- Launch the Gradio App ---
block.launch(
    server_name=args.server,
    server_port=args.port,
    share=args.share,
    inbrowser=args.inbrowser,
)