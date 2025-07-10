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
    
    # Print configuration if verbose logging is enabled
    if PIPELINE_CONFIG.get('verbose_logging', True):
        print("="*50)
        print("CURRENT CONFIGURATION:")
        print("="*50)
        for section, settings in config.items():
            print(f"\n📋 {section.upper()} CONFIG:")
            if isinstance(settings, dict):
                for key, value in settings.items():
                    print(f"   {key}: {value}")
            else:
                print(f"   {settings}")
        print("="*50)
        
except Exception as e:
    print(f"❌ Error loading configuration: {e}")
    print("Please check your config.yaml file")
    exit(1)

# Load datasets
try:
    dataset_name = DATASET_CONFIG.get('dataset_name')
    print(f"Loading {dataset_name} dataset...")
    
    # using streaming to avoid downloading the whole dataset
    dataset = load_dataset(
        dataset_name, 
        streaming=DATASET_CONFIG.get('use_streaming', True)
    )
    print("Dataset loaded successfully!")
    
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
                
                # For streaming datasets, access splits differently
                if hasattr(dataset, split_name):
                    data_stream = getattr(dataset, split_name)
                    split_used = split_name
                    print(f"✅ Successfully accessed {split_name} split")
                    break
                else:
                    print(f"❌ {split_name} split not found")
                    
            except (KeyError, AttributeError, TypeError) as e:
                print(f"❌ Failed to access {split_name} split: {e}")
                continue

        # If no specific split found, try to use the first available split
        if data_stream is None:
            print("Trying to access train split directly...")
            
            # Try common split names directly
            try:
                if hasattr(dataset, 'train'):
                    data_stream = dataset.train
                    split_used = "train"
                    print(f"✅ Using train split")
                else:
                    print("❌ No train split found")
                    
            except Exception as e:
                print(f"❌ Error accessing train split: {e}")
            
        # Final fallback: use dataset directly
        if data_stream is None:
            print("Using dataset directly as fallback...")
            data_stream = dataset
            split_used = "direct"
            
    except Exception as e:
        print(f"❌ Error during split access: {e}")
        data_stream = dataset
        split_used = "fallback"
        print(f"Using fallback dataset: {e}")
    
    print(f"Final data stream: {type(data_stream)}")
    print(f"Split used: {split_used}")
    
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
            print(f"✅ Retrieved sample {i+1}/{max_samples}")
            
        print(f"Total samples retrieved: {len(samples)}")
        
    except Exception as e:
        print(f"Error during sample retrieval: {e}")
        print(f"Retrieved {len(samples)} samples before error")
        import traceback
        traceback.print_exc()
    
    if len(samples) == 0:
        print("❌ No samples retrieved from dataset!")
    else:
        print(f"✅ Successfully retrieved {len(samples)} samples")

    
except Exception as e:
    print(f"Error loading dataset: {e}")
    samples = []

# OpenAI API configuration
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
        
        # Convert sample to string for LLM analysis
        dataset_sample_str = json.dumps(task_sample, indent=2, ensure_ascii=False)
        
        # Get basic sample summary for logging
        sample_summary = utils.get_sample_summary(task_sample)
        
        # Print sample information if verbose logging is enabled
        if PIPELINE_CONFIG.get('verbose_logging', True):
            print(f"Sample keys: {sample_summary['sample_keys']}")
            print(f"Sample summary: {sample_summary}")
        
        # Generate prompt using the prompt module with dataset sample string
        prompt_text = prompt.get_initial_validation_prompt(
            dataset_sample_str,  # Pass sample as string for LLM to analyze
            dataset_name  # Pass dataset name for context
        )
        
        try:
            response = self.client.chat.completions.create(
                model=OPENAI_CONFIG.get('model', 'gpt-4o'),
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert Python developer specializing in data validation and analysis. Generate clean, well-documented, and robust code that can intelligently analyze dataset samples and create meaningful validation logic. CRITICAL: The script must include a complete working example with the actual dataset sample and show validation results. IMPORTANT: Return ONLY executable Python code without any Markdown formatting, explanatory text, or code block markers. Start directly with imports or class definitions."
                    },
                    {
                        "role": "user",
                        "content": prompt_text
                    }
                ],
                max_tokens=OPENAI_CONFIG.get('max_tokens', 4000),
                temperature=OPENAI_CONFIG.get('temperature', 0.3),
                n = 1
            )
            
            raw_script = utils.validate_openai_response(response)
            
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
            
            # Generate debugging prompt using the prompt module
            debug_prompt_text = prompt.get_debug_prompt(current_script, error_message)
            
            try:
                response = self.client.chat.completions.create(
                    model=OPENAI_CONFIG.get('model', 'gpt-4o'),
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert Python debugger specializing in validation scripts. Fix the error while preserving the original functionality and dataset analysis capabilities. Return ONLY executable Python code without Markdown formatting."
                        },
                        {
                            "role": "user", 
                            "content": debug_prompt_text
                        }
                    ],
                    max_tokens=OPENAI_CONFIG.get('max_tokens', 4000),
                    temperature=DEBUG_CONFIG.get('debug_temperature', 0.1)
                )
                
                raw_script = utils.validate_openai_response(response)
                current_script = utils.clean_markdown_script(raw_script)
                print("Script updated based on error feedback")
                
            except Exception as e:
                print(f"Error during debugging iteration {iteration + 1}: {e}")
                break
        
        return current_script
    
    def self_debug_with_reflection(self, script_code: str, task_sample: Dict) -> str:
        """Implement self-debug with self-reflection"""
        
        print("Starting self-debug with reflection...")
        
        # Generate reflection prompt using the prompt module
        reflection_prompt_text = prompt.get_reflection_prompt(script_code)
        
        try:
            response = self.client.chat.completions.create(
                model=OPENAI_CONFIG.get('model', 'gpt-4o'),
                messages=[
                    {
                        "role": "system",
                        "content": "You are a senior validation script expert specializing in dataset analysis and validation. Focus on improving the script's ability to intelligently analyze dataset samples and create meaningful validation logic. Return ONLY executable Python code without Markdown formatting."
                    },
                    {
                        "role": "user",
                        "content": reflection_prompt_text
                    }
                ],
                max_tokens=OPENAI_CONFIG.get('max_tokens', 4000),
                temperature=DEBUG_CONFIG.get('debug_temperature', 0.1)
            )
            
            raw_script = utils.validate_openai_response(response)
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
                    print(f"✅ Generated and saved validation script: {filename}")
                else:
                    print(f"✅ Generated validation script for {task_id}")
            else:
                print(f"❌ Failed to generate script for {task_id}")
        
        self.generated_scripts = generated_scripts
        return generated_scripts


# Main execution
if __name__ == "__main__":
    print(f"{dataset_name} Validation Script Generator")
    print("Universal dataset validation script generation")
    print("="*50)
    
    # Initialize the validation generator
    generator = ValidationGenerator(client, config)
    
    # Run the complete pipeline
    results = generator.run_validation_pipeline(samples)
    
    # Print summary
    utils.print_generation_summary(results)