from openai import OpenAI
from datasets import load_dataset
import random
import json
import os
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
import prompt
import utils
import time
import re

# Load configuration from YAML file
try:
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Extract configuration sections
    DATASET_CONFIG = config.get('dataset', {})
    OPENAI_CONFIG = config.get('openai', {})
    DEBUG_CONFIG = config.get('debug', {})
    PIPELINE_CONFIG = config.get('pipeline', {})
    NETWORK_CONFIG = config.get('network', {})
    
    # Print prompt type information
    print("Using standard prompt module (full-featured)")
    
    # Print configuration if verbose logging is enabled
    if PIPELINE_CONFIG.get('verbose_logging', True):
        print("="*50)
        print("CURRENT CONFIGURATION:")
        print("="*50)
        for section, settings in config.items():
            print(f"\n{section.upper()} CONFIG:")
            if isinstance(settings, dict):
                for key, value in settings.items():
                    print(f"   {key}: {value}")
            else:
                print(f"   {settings}")
        print("="*50)
        
except Exception as e:
    print(f" Error loading configuration: {e}")
    print("Please check your config.yaml file")
    exit(1)

# Configure network settings for Hugging Face Hub
try:
    # Set up Hugging Face mirror if specified
    hf_mirror = NETWORK_CONFIG.get('hf_mirror')
    if hf_mirror:
        print(f"Using Hugging Face mirror: {hf_mirror}")
        os.environ['HF_ENDPOINT'] = hf_mirror
    
    # Set connection timeout
    timeout = NETWORK_CONFIG.get('timeout', 30)
    print(f"Connection timeout set to: {timeout}s")
    
except Exception as e:
    print(f" Warning: Failed to configure network settings: {e}")

# Load datasets
try:
    dataset_name = DATASET_CONFIG.get('dataset_name')
    config_name = DATASET_CONFIG.get('config_name', '')
    
    print(f"Loading {dataset_name} dataset...")
    
    # Try to load dataset with config if specified
    dataset_kwargs = {
        'streaming': DATASET_CONFIG.get('use_streaming', True)
    }
    
    # Add config_name if specified and not empty
    if config_name and config_name.strip():
        dataset_kwargs['name'] = config_name.strip()
        print(f"Using config: {config_name}")
    
    try:
        # Attempt to load with specified config
        dataset = load_dataset(dataset_name, **dataset_kwargs)
        print("Dataset loaded successfully!")
        
    except Exception as e:
        if "Config name is missing" in str(e) or "Please pick one among" in str(e):
            print(f"❌ Dataset requires config selection. Error: {e}")
            
            # Try to get available configs and suggest the first one
            try:
                print("Attempting to auto-detect available configs...")
                from datasets import get_dataset_config_names
                available_configs = get_dataset_config_names(dataset_name)
                print(f"Available configs: {available_configs}")
                
                if available_configs:
                    auto_config = available_configs[0]
                    print(f"Auto-selecting first config: {auto_config}")
                    dataset_kwargs['name'] = auto_config
                    dataset = load_dataset(dataset_name, **dataset_kwargs)
                    print(f" Successfully loaded with config: {auto_config}")
                else:
                    raise Exception("No available configs found")
                    
            except Exception as inner_e:
                print(f"Failed to auto-detect configs: {inner_e}")
                raise e
        else:
            raise e
    
    # check the dataset splits
    print(f"Dataset splits: {dataset}")
    
    # Try to access dataset splits in a simple way
    data_stream = None
    split_used = None
    
    try:
        # Try each split in priority order
        for split_name in DATASET_CONFIG.get('split_priority', ['test', 'train', 'validation']):
            try:
                print(f"Trying to access {split_name} split...")
                
                # For IterableDatasetDict, use dictionary-style access
                try:
                    data_stream = dataset[split_name]
                    split_used = split_name
                    print(f"Successfully accessed {split_name} split")
                    break
                except (KeyError, TypeError):
                    print(f" {split_name} split not available")
                    continue
                    
            except Exception as e:
                print(f"Failed to access {split_name} split: {e}")
                continue

        # If no specific split found, try train directly
        if data_stream is None:
            print("Trying to access train split as fallback...")
            try:
                data_stream = dataset['train']
                split_used = "train"
                print(f"Using train split as fallback")
            except Exception as e:
                print(f"Error accessing train split: {e}")
            
        # Final fallback: iterate available splits and use the first one
        if data_stream is None:
            print("Using first available split as final fallback...")
            try:
                # Get the first available split from the dataset
                for key in dataset:
                    try:
                        data_stream = dataset[key]
                        split_used = key
                        print(f"Using first available split: {key}")
                        break
                    except Exception as e:
                        print(f"Failed to use split {key}: {e}")
                        continue
            except Exception as e:
                print(f"Error iterating splits: {e}")
            
    except Exception as e:
        print(f"Error during split access: {e}")
        data_stream = None
        split_used = "error"
    
    print(f"Final data stream: {type(data_stream)}")
    print(f"Split used: {split_used}")
    
    # Check if we have a valid data stream
    if data_stream is None:
        print("No valid data stream found! Cannot proceed.")
        samples = []
    else:
        samples = []
        max_samples = DATASET_CONFIG.get('max_samples_to_load', 5)
        print(f"Retrieving {max_samples} samples...")
        
        try:
            for i, sample in enumerate(data_stream):
                if i >= max_samples:
                    print(f"Reached maximum samples limit ({max_samples})")
                    break
                
                print(f"Processing sample {i+1}/{max_samples}...")
                
                # Debug: print sample type and basic info
                if PIPELINE_CONFIG.get('verbose_logging', True):
                    print(f"  Sample type: {type(sample)}")
                    if hasattr(sample, 'keys'):
                        print(f"  Sample keys: {list(sample.keys())[:5]}...")  # Show first 5 keys
                    elif isinstance(sample, str):
                        print(f"  Sample length: {len(sample)} characters")
                
                samples.append(sample)
                print(f"Retrieved sample {i+1}/{max_samples}")
                
            print(f"Total samples retrieved: {len(samples)}")
            
        except Exception as e:
            print(f"Error during sample retrieval: {e}")
            print(f"Retrieved {len(samples)} samples before error")
            import traceback
            traceback.print_exc()
        
        if len(samples) == 0:
            print("No samples retrieved from dataset!")
        else:
            print(f"Successfully retrieved {len(samples)} samples")

    
except Exception as e:
    print(f"Error loading dataset: {e}")
    samples = []

def extract_key_info_from_large_sample(sample_data: Dict, dataset_name: str) -> Dict:
    """
    Extract key information from large dataset samples to fit LLM token limits
    Specifically designed for Mind2Web and other large datasets
    """
    
    # Determine if smart truncation is needed
    sample_str = json.dumps(sample_data, indent=2, ensure_ascii=False)
    max_chars = DATASET_CONFIG.get('max_sample_chars', 50000)
    
    if len(sample_str) <= max_chars:
        # Sample is small enough, return as-is
        return sample_data
    
    print(f"Sample size ({len(sample_str):,} chars) exceeds limit ({max_chars:,} chars)")
    print("Applying intelligent data extraction...")
    
    # Create extracted sample with key information
    extracted_sample = {}
    
    # Always preserve basic metadata
    basic_fields = ['annotation_id', 'website', 'domain', 'subdomain', 'confirmed_task']
    for field in basic_fields:
        if field in sample_data:
            extracted_sample[field] = sample_data[field]
    
    # Handle Mind2Web specific structure
    if 'osunlp/Mind2Web' in dataset_name or 'Mind2Web' in dataset_name:
        
        # Extract ALL action representations (these are critical for Mind2Web validation)
        if 'action_reprs' in sample_data:
            action_reprs = sample_data['action_reprs']
            # Keep ALL action representations - they are the core of Mind2Web tasks
            extracted_sample['action_reprs'] = action_reprs
            print(f"   → Preserved all {len(action_reprs) if isinstance(action_reprs, list) else 1} action representations")
        
        # Extract minimal action structure (since action_reprs already contains the key info)
        if 'actions' in sample_data and isinstance(sample_data['actions'], list):
            actions = sample_data['actions']
            extracted_actions = []
            
            # Only take first 2 actions for structure understanding, since action_reprs has all the details
            indices_to_keep = [0, 1] if len(actions) >= 2 else [0] if len(actions) >= 1 else []
            
            for i in indices_to_keep:
                if i < len(actions):
                    action = actions[i].copy() if isinstance(actions[i], dict) else actions[i]
                    
                    # For each action, keep only the most essential fields
                    if isinstance(action, dict):
                        # Keep minimal essential action information
                        essential_action = {}
                        action_fields_to_keep = ['action_uid', 'operation']
                        
                        for field in action_fields_to_keep:
                            if field in action:
                                essential_action[field] = action[field]
                        
                        # Extremely aggressive HTML truncation (just keep a tiny sample for structure)
                        html_fields = ['raw_html', 'cleaned_html']
                        for html_field in html_fields:
                            if html_field in action:
                                html_content = action[html_field]
                                if isinstance(html_content, str) and len(html_content) > 200:
                                    # Keep only first 100 characters for structure understanding
                                    essential_action[html_field] = html_content[:100] + f"... [TRUNCATED {len(html_content)-100:,} chars for space efficiency]"
                                else:
                                    essential_action[html_field] = html_content
                        
                        # Keep only 1 candidate element as example
                        for candidate_field in ['pos_candidates', 'neg_candidates']:
                            if candidate_field in action:
                                candidates = action[candidate_field]
                                if isinstance(candidates, list) and len(candidates) > 0:
                                    essential_action[candidate_field] = [candidates[0]] if candidates else []
                                    if len(candidates) > 1:
                                        essential_action[candidate_field].append({"_note": f"... and {len(candidates)-1} more candidates (see action_reprs for details)"})
                                else:
                                    essential_action[candidate_field] = candidates
                        
                        extracted_actions.append(essential_action)
                    else:
                        extracted_actions.append(action)
            
            extracted_sample['actions'] = extracted_actions
            extracted_sample['_action_note'] = {
                'original_action_count': len(actions),
                'extracted_action_count': len(extracted_actions),
                'note': 'Only first 2 actions included for structure reference. Full action details preserved in action_reprs field.'
            }
            print(f"   → Compressed {len(actions)} actions to {len(extracted_actions)} sample actions (full details in action_reprs)")
    
    # For other dataset types, apply generic truncation strategy
    else:
        # Keep all fields but truncate string values that are too long
        for key, value in sample_data.items():
            if key not in extracted_sample:  # Don't overwrite already processed fields
                if isinstance(value, str) and len(value) > 2000:
                    # Truncate long strings
                    extracted_sample[key] = value[:1000] + f"\n... [TRUNCATED {len(value)-2000:,} chars] ...\n" + value[-1000:]
                elif isinstance(value, list) and len(value) > 10:
                    # Truncate long lists
                    extracted_sample[key] = value[:5] + [f"... and {len(value)-10} more items"] + value[-5:]
                elif isinstance(value, dict) and len(json.dumps(value)) > 5000:
                    # For large nested objects, keep only essential fields
                    if len(value) > 10:
                        essential_keys = list(value.keys())[:5]
                        extracted_sample[key] = {k: value[k] for k in essential_keys}
                        extracted_sample[key]['_truncated'] = f"... and {len(value)-5} more fields"
                    else:
                        extracted_sample[key] = value
                else:
                    extracted_sample[key] = value
    
    # Add metadata about the extraction
    extracted_sample['_extraction_metadata'] = {
        'original_size_chars': len(sample_str),
        'extraction_applied': True,
        'extraction_reason': f'Sample exceeded {max_chars:,} character limit',
        'dataset_name': dataset_name
    }
    
    # Verify the extracted sample is within limits
    extracted_str = json.dumps(extracted_sample, indent=2, ensure_ascii=False)
    print(f"Extracted sample size: {len(extracted_str):,} chars (reduced by {len(sample_str)-len(extracted_str):,} chars)")
    
    return extracted_sample


# OpenAI client configuration
client = OpenAI(
    api_key=OPENAI_CONFIG.get('api_key'),
    base_url=OPENAI_CONFIG.get('base_url', 'https://api.openai.com/v1')
)

class ValidationGenerator:
    """
    Universal validation script generator that works with any dataset
    """
    
    def __init__(self, client: OpenAI, config: Optional[Dict] = None):
        self.client = client
        self.config = config if config is not None else globals()['config']
        self.selected_tasks = []
        self.generated_scripts = {}
    
    def analyze_tasks(self, samples: List[Dict]) -> List[Dict]:
        """Analyze and select representative tasks from dataset (universal approach)"""
        print("Analyzing tasks and selecting representative samples...")
        
        max_scripts = DATASET_CONFIG.get('max_scripts_to_generate', 20)
        
        # If we have fewer or equal samples than max_scripts, return all
        if len(samples) <= max_scripts:
            self.selected_tasks = samples
            print(f"Selected all {len(samples)} samples (within limit)")
            return samples
        
        # Simple sampling strategy: evenly distribute across the dataset
        step = len(samples) // max_scripts
        selected = []
        
        for i in range(max_scripts):
            index = i * step
            if index < len(samples):
                selected.append(samples[index])
        
        # If we still have room, add some random samples to reach max_scripts
        remaining_samples = [s for s in samples if s not in selected]
        while len(selected) < max_scripts and remaining_samples:
            random_sample = random.choice(remaining_samples)
            selected.append(random_sample)
            remaining_samples.remove(random_sample)
        
        self.selected_tasks = selected
        print(f"Selected {len(selected)} representative tasks using universal sampling")
        return selected
    
    def generate_validation_script(self, task_sample: Dict) -> Optional[str]:
        """Use GPT to generate validation script for a specific task"""
        
        # Apply intelligent data extraction if enabled and needed
        processed_sample = task_sample
        if DATASET_CONFIG.get('enable_smart_truncation', True):
            processed_sample = extract_key_info_from_large_sample(task_sample, dataset_name)
        
        # Convert processed sample to string for LLM analysis
        dataset_sample_str = json.dumps(processed_sample, indent=2, ensure_ascii=False)
        
        # Get basic sample summary for logging
        sample_summary = utils.get_sample_summary(processed_sample)
        
        # Print sample information if verbose logging is enabled
        if PIPELINE_CONFIG.get('verbose_logging', True):
            print(f"Sample keys: {sample_summary['sample_keys']}")
            print(f"Sample summary: {sample_summary}")
            print(f"Final sample size: {len(dataset_sample_str)} characters")
            
            # Show extraction info if applied
            if '_extraction_metadata' in processed_sample:
                metadata = processed_sample['_extraction_metadata']
                print(f"Data extraction applied: {metadata['original_size_chars']:,} → {len(dataset_sample_str):,} chars")
        
        # Generate prompt using the standard prompt module
        prompt_text = prompt.get_initial_validation_prompt(
            dataset_sample_str,  # Pass sample as string for LLM to analyze
            dataset_name  # Pass dataset name for context
        )
        
        if prompt_text:
            print(f"Total prompt length: {len(prompt_text)} characters")
        
        if not prompt_text:
            print("Failed to generate prompt")
            return None
        
        try:
            messages = [
                {
                    "role": "system", 
                    "content": "You are an expert Python developer specializing in data validation and analysis. Generate clean, well-documented, and robust code that can intelligently analyze dataset samples and create meaningful validation logic. CRITICAL: The script must include a complete working example with the actual dataset sample and show validation results. IMPORTANT: Return ONLY executable Python code without any Markdown formatting, explanatory text, or code block markers. Start directly with imports or class definitions."
                },
                {
                    "role": "user",
                    "content": prompt_text
                }
            ]
            
            raw_script = self.client.chat.completions.create(
                model=OPENAI_CONFIG.get('model', 'gpt-4'),
                messages=messages,
                max_tokens=OPENAI_CONFIG.get('max_tokens', 4000),
                temperature=OPENAI_CONFIG.get('temperature', 0.3),
                n=1
            ).choices[0].message.content
            
            if raw_script is None:
                print("❌ Failed to get response from API")
                return None
            
            # Clean any potential Markdown formatting
            cleaned_script = utils.clean_markdown_script(raw_script)
            
            return cleaned_script
            
        except Exception as e:
            print(f"Error generating validation script: {e}")
            return None
    
    def self_debug_with_system_feedback(self, script_code: str, task_sample: Dict) -> str:
        """Implement self-debug with system feedback"""
        
        print("Starting self-debug with system feedback...")
        current_script = script_code
        max_iterations = DEBUG_CONFIG.get('max_system_feedback_iterations', 3)
        
        for iteration in range(max_iterations):
            print(f"Debug iteration {iteration + 1}/{max_iterations}")
            
            # Try to execute the script and capture errors
            error_message = utils.test_script_execution(current_script, task_sample)
            
            if not error_message:
                print("Script executed successfully!")
                break
                
            print(f"Found error: {error_message}")
            
            # Generate debugging prompt using the standard prompt module
            debug_prompt_text = prompt.get_debug_prompt(current_script, error_message)
            
            try:
                messages = [
                    {
                        "role": "system",
                        "content": "You are an expert Python debugger specializing in validation scripts. Fix the error while preserving the original functionality and dataset analysis capabilities. Return ONLY executable Python code without Markdown formatting."
                    },
                    {
                        "role": "user", 
                        "content": debug_prompt_text
                    }
                ]
                
                raw_script = self.client.chat.completions.create(
                    model=OPENAI_CONFIG.get('model', 'gpt-4'),
                    messages=messages,
                    max_tokens=OPENAI_CONFIG.get('max_tokens', 4000),
                    temperature=DEBUG_CONFIG.get('debug_temperature', 0.1),
                    n=1
                ).choices[0].message.content
                
                if raw_script is None:
                    print(f" Failed to get response during debugging iteration {iteration + 1}")
                    break
                
                current_script = utils.clean_markdown_script(raw_script)
                print("Script updated based on error feedback")
                
            except Exception as e:
                print(f"Error during debugging iteration {iteration + 1}: {e}")
                break
        
        return current_script
    
    def self_debug_with_reflection(self, script_code: str, task_sample: Dict) -> str:
        """Implement self-debug with self-reflection"""
        
        print("Starting self-debug with reflection...")
        
        # Generate reflection prompt using the standard prompt module
        reflection_prompt_text = prompt.get_reflection_prompt(script_code)
        
        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are a senior validation script expert specializing in dataset analysis and validation. Focus on improving the script's ability to intelligently analyze dataset samples and create meaningful validation logic. Return ONLY executable Python code without Markdown formatting."
                },
                {
                    "role": "user",
                    "content": reflection_prompt_text
                }
            ]
            
            raw_script = self.client.chat.completions.create(
                model=OPENAI_CONFIG.get('model', 'gpt-4'),
                messages=messages,
                max_tokens=OPENAI_CONFIG.get('max_tokens', 4000),
                temperature=DEBUG_CONFIG.get('debug_temperature', 0.1),
                n=1
            ).choices[0].message.content
            
            if raw_script is None:
                print(" Failed to get response during reflection")
                return script_code
            
            return utils.clean_markdown_script(raw_script)
            
        except Exception as e:
            print(f"Error during reflection: {e}")
            return script_code
    
    def generate_and_debug_script(self, task_sample: Dict) -> Optional[str]:
        """Complete pipeline: generate, debug, and refine validation script"""
        
        # Get basic sample info for display
        sample_summary = utils.get_sample_summary(task_sample)
        
        print(f"\n{'='*60}")
        print(f"Generating validation script for sample with keys: {sample_summary['sample_keys'][:3]}...")
        print(f"{'='*60}")
        
        # Step 1: Generate initial script
        print("Step 1: Generating initial validation script...")
        initial_script = self.generate_validation_script(task_sample)
        
        if not initial_script:
            print("Failed to generate initial script")
            return None
        
        # Step 2: Self-debug with system feedback
        print("Step 2: Self-debugging with system feedback...")
        debugged_script = self.self_debug_with_system_feedback(initial_script, task_sample)
        
        # Step 3: Self-debug with reflection
        print("Step 3: Self-debugging with reflection...")
        final_script = self.self_debug_with_reflection(debugged_script, task_sample)
        
        return final_script
    
    def run_validation_pipeline(self, samples: List[Dict]) -> Dict[str, Dict]:
        """Run the complete validation script generation pipeline"""
        
        print(f"Starting {dataset_name} validation script generation pipeline...")
        print("="*80)
        
        # Analyze and select representative tasks
        selected_tasks = self.analyze_tasks(samples)
        
        generated_scripts = {}
        
        for i, task in enumerate(selected_tasks):
            # Generate simple task ID
            task_id = utils.generate_simple_task_id(i)
            sample_info = utils.get_sample_summary(task)
            
            # Generate and debug validation script
            script = self.generate_and_debug_script(task)
            
            if script:
                generated_scripts[task_id] = {
                    'script': script,
                    'sample_info': sample_info
                }
                
                # Save script to file if enabled
                if PIPELINE_CONFIG.get('save_scripts_to_files', True):
                    filename = f"validation_script_{task_id}.py"
                    utils.save_validation_script(filename, script, task)
                    print(f" Generated and saved validation script: {filename}")
                else:
                    print(f" Generated validation script for {task_id}")
            else:
                print(f" Failed to generate script for {task_id}")
        
        self.generated_scripts = generated_scripts
        return generated_scripts


# Main execution
if __name__ == "__main__":
    print(f"{dataset_name} Validation Script Generator")
    print("Universal dataset validation script generation")
    print("="*50)
    
    # Check API status before proceeding (using new OpenAI client)
    print("🔍 Checking API status...")
    try:
        test_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=5
        )
        print("API is accessible")
    except Exception as e:
        print(f"API check failed: {e}")
        exit(1)
    
    # Initialize the validation generator
    generator = ValidationGenerator(client, config)
    
    # Run the complete pipeline
    results = generator.run_validation_pipeline(samples)
    
    # Print summary
    utils.print_generation_summary(results)