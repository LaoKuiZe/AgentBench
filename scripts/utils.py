import traceback
import re
from typing import Dict, Optional, Any

# Clean the markdown formmats
def clean_markdown_script(script_content: str) -> str:
    """Clean Markdown formatting from generated script"""
    
    print("Cleaning Markdown formatting from script...")
    
    lines = script_content.split('\n')
    cleaned_lines = []
    in_code_block = False
    skip_text = True  # Skip text until we find the first code block
    
    for line in lines:
        # Remove code block markers
        if line.strip().startswith('```'):
            if not in_code_block:
                # Starting a code block
                in_code_block = True
                skip_text = False  # Found code, stop skipping
            else:
                # Ending a code block
                in_code_block = False
                skip_text = True  # Skip text after code blocks
            continue
        
        if in_code_block:
            # Inside code block, keep as is
            cleaned_lines.append(line)
        elif not skip_text:
            # Outside code block but after we've seen code
            # Convert explanatory text to comments
            if line.strip():
                if line.strip().startswith('#'):
                    # Already a comment
                    cleaned_lines.append(line)
                elif line.strip().startswith('###') or line.strip().startswith('##'):
                    # Markdown header
                    cleaned_lines.append(f"# {line.strip().lstrip('#').strip()}")
                else:
                    # Regular text, convert to comment
                    cleaned_lines.append(f"# {line.strip()}")
            else:
                # Empty line
                cleaned_lines.append(line)
        # Skip everything else (text before code blocks)
    
    # If no code blocks were found, assume the entire content is code
    if skip_text and not cleaned_lines:
        print("No code blocks found, treating entire content as Python code")
        return script_content.strip()
    
    # Join lines and remove excessive blank lines
    cleaned_script = '\n'.join(cleaned_lines)
    
    # Remove multiple consecutive blank lines
    cleaned_script = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_script)
    
    return cleaned_script.strip()


def test_script_execution(script_code: str, task_sample: Dict) -> Optional[str]:
    """Test script execution and capture any errors"""
    
    try:
        # Create a safe environment for testing
        test_globals = {
            '__builtins__': __builtins__,
            'task_sample': task_sample
        }
        
        # Execute the script in controlled environment  
        exec(script_code, test_globals)
        
        return None  # No error
        
    except Exception as e:
        return f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"


def validate_openai_response(response) -> str:
    """Validate and extract content from OpenAI API response"""
    
    if not response or not response.choices:
        raise ValueError("Empty or invalid response from OpenAI API")
    
    content = response.choices[0].message.content
    if not content:
        raise ValueError("Empty content in OpenAI response")
    
    return content


def get_sample_summary(task_sample: Any) -> Dict[str, Any]:
    """Get a basic summary of the dataset sample without assuming field names"""
    
    # Handle different input types
    if isinstance(task_sample, str):
        # If input is a string, try to parse it as JSON
        try:
            import json
            parsed_sample = json.loads(task_sample)
            if isinstance(parsed_sample, dict):
                task_sample = parsed_sample
            else:
                # If it's not a dict after parsing, treat as string
                return {
                    'sample_type': 'string',
                    'sample_size': len(task_sample),
                    'sample_keys': [],
                    'has_list_fields': False,
                    'has_dict_fields': False,
                    'has_string_fields': True,
                    'potential_description_fields': []
                }
        except (json.JSONDecodeError, TypeError):
            # If JSON parsing fails, treat as plain string
            return {
                'sample_type': 'string',
                'sample_size': len(task_sample),
                'sample_keys': [],
                'has_list_fields': False,
                'has_dict_fields': False,
                'has_string_fields': True,
                'potential_description_fields': []
            }
    
    # Handle non-dict types
    if not isinstance(task_sample, dict):
        return {
            'sample_type': type(task_sample).__name__,
            'sample_size': 1,
            'sample_keys': [],
            'has_list_fields': isinstance(task_sample, list),
            'has_dict_fields': False,
            'has_string_fields': isinstance(task_sample, str),
            'potential_description_fields': []
        }
    
    # Generic analysis for dict input
    summary = {
        'sample_type': 'dict',
        'sample_keys': list(task_sample.keys()),
        'sample_size': len(task_sample),
        'has_list_fields': any(isinstance(v, list) for v in task_sample.values()),
        'has_dict_fields': any(isinstance(v, dict) for v in task_sample.values()),
        'has_string_fields': any(isinstance(v, str) for v in task_sample.values())
    }
    
    # Try to find potential description field (generic approach)
    potential_desc_fields = []
    for key, value in task_sample.items():
        if isinstance(value, str) and len(value) > 10:  # Likely a description
            potential_desc_fields.append(key)
    
    summary['potential_description_fields'] = potential_desc_fields
    
    return summary


def generate_simple_task_id(index: int) -> str:
    """Generate simple task ID based on index"""
    return f"task_{index+1}"


def save_validation_script(filename: str, script_content: str, sample_data: Dict) -> None:
    """Save validation script to file with basic sample info"""
    
    # Get basic sample summary
    summary = get_sample_summary(sample_data)
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"# Validation script for {filename}\n")
        f.write(f"# Sample keys: {summary['sample_keys']}\n")
        f.write(f"# Sample analysis: {summary}\n\n")
        f.write(script_content)


def print_generation_summary(results: Dict) -> None:
    """Print generation pipeline summary"""
    
    print("\n" + "="*80)
    print("GENERATION COMPLETE - SUMMARY")
    print("="*80)
    
    for task_id, data in results.items():
        sample_info = data.get('sample_info', {})
        
        print(f"\n📋 {task_id.upper()}:")
        print(f"   Sample Keys: {sample_info.get('sample_keys', 'Unknown')}")
        print(f"   Sample Size: {sample_info.get('sample_size', 'Unknown')} fields")
        print(f"   Script Length: {len(data['script'])} characters")
        print(f"   File: validation_script_{task_id}.py")
    
    print(f"\n✅ Successfully generated {len(results)} validation scripts")
    print("📁 All scripts saved to current directory")


def check_api_status(client, api_key: str, base_url: str) -> bool:
    """
    Check if API is accessible and has quota remaining
    """
    try:
        print("Checking API status...")
        
        # Try a simple API call to check status
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=5
        )
        
        print("✅ API is accessible")
        return True
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ API check failed: {error_msg}")
        
        if "429" in error_msg:
            print("⚠️  Possible causes:")
            print("   - API quota exceeded")
            print("   - Rate limit reached")
            print("   - Need to wait before making more requests")
        elif "401" in error_msg:
            print("⚠️  Authentication failed - check API key")
        elif "403" in error_msg:
            print("⚠️  Access forbidden - check permissions")
        
        return False

def suggest_solutions_for_429():
    """
    Suggest solutions for 429 errors
    """
    print("\nSolutions for 429 errors:")
    print("1. Check API quota/billing status")
    print("2. Wait 10-15 minutes before retrying")
    print("3. Increase delay between requests")
