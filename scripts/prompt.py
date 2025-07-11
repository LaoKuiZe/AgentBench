# Dataset structure information for different datasets
DATASET_STRUCTURES = {
    "osunlp/Mind2Web": """Dataset Structure
Data Fields
"annotation_id" (str): unique id for each task
"website" (str): website name
"domain" (str): website domain
"subdomain" (str): website subdomain
"confirmed_task" (str): task description
"action_reprs" (list[str]): human readable string representation of the action sequence
"actions" (list[dict]): list of actions (steps) to complete the task
"action_uid" (str): unique id for each action (step)
"raw_html" (str): raw html of the page before the action is performed
"cleaned_html" (str): cleaned html of the page before the action is performed
"operation" (dict): operation to perform
"op" (str): operation type, one of CLICK, TYPE, SELECT
"original_op" (str): original operation type, contain additional HOVER and ENTER that are mapped to CLICK, not used
"value" (str): optional value for the operation, e.g., text to type, option to select
"pos_candidates" (list[dict]): ground truth elements. Here we only include positive elements that exist in "cleaned_html" after our preprocessing, so "pos_candidates" might be empty. The original labeled element can always be found in the "raw_html".
"tag" (str): tag of the element
"is_original_target" (bool): whether the element is the original target labeled by the annotator
"is_top_level_target" (bool): whether the element is a top level target find by our algorithm. please see the paper for more details.
"backend_node_id" (str): unique id for the element
"attributes" (str): serialized attributes of the element, use json.loads to convert back to dict
"neg_candidates" (list[dict]): other candidate elements in the page after preprocessing, has similar structure as "pos_candidates"
""",
    "tasksource/planbench": """
task: string
prompt_type: string
domain: string
instance_id: int64
example_instance_ids: list
query: string
ground_truth_plan: string
    """,
    "princeton-nlp/SWE-bench_Verified": """
instance_id: (str) - A formatted instance identifier, usually as repo_owner__repo_name-PR-number.
patch: (str) - The gold patch, the patch generated by the PR (minus test-related code), that resolved the issue.
repo: (str) - The repository owner/name identifier from GitHub.
base_commit: (str) - The commit hash of the repository representing the HEAD of the repository before the solution PR is applied.
hints_text: (str) - Comments made on the issue prior to the creation of the solution PR’s first commit creation date.
created_at: (str) - The creation date of the pull request.
test_patch: (str) - A test-file patch that was contributed by the solution PR.
problem_statement: (str) - The issue title and body.
version: (str) - Installation version to use for running evaluation.
environment_setup_commit: (str) - commit hash to use for environment setup and installation.
FAIL_TO_PASS: (str) - A json list of strings that represent the set of tests resolved by the PR and tied to the issue resolution.
PASS_TO_PASS: (str) - A json list of strings that represent tests that should pass before and after the PR application.
    """,
    
    # Add more datasets here as needed
    # "MMInstruction/OSWorld-G": "OSWorld dataset structure...",
}

def get_dataset_structure_info(dataset_name: str) -> str:
    """Get dataset structure information based on dataset name"""
    
    # Direct lookup first
    if dataset_name in DATASET_STRUCTURES:
        return DATASET_STRUCTURES[dataset_name]
    
    # Fuzzy matching for partial names
    for known_name, structure_info in DATASET_STRUCTURES.items():
        if any(part in dataset_name for part in known_name.split("/")):
            return structure_info
    
    # No structure information found
    return ""

def get_initial_validation_prompt(dataset_sample_str: str, dataset_name: str) -> str:
    """Generate detailed prompt for LLM to create validation script"""
    
    # Get dataset-specific structure information if available
    dataset_structure_info = get_dataset_structure_info(dataset_name)
    
    # Format structure information section
    structure_section = ""
    if dataset_structure_info:
        structure_section = f"""
**DATASET STRUCTURE REFERENCE:**
The dataset "{dataset_name}" has the following known structure:
```
{dataset_structure_info}
```
Use this as a reference, but always analyze the actual sample provided to understand the real structure.
"""
    else:
        structure_section = f"""
**DATASET ANALYSIS:**
No predefined structure information available for "{dataset_name}". 
You must analyze the provided sample completely to understand its structure.
"""
    
    prompt = f"""
You are an expert code generator specializing in creating validation scripts for agents datasets. 
I need you to generate a comprehensive validation script that can analyze and validate this specific dataset sample.

**DATASET INFORMATION:**
- Dataset Name: {dataset_name}
{structure_section}

**ACTUAL DATASET SAMPLE TO ANALYZE:**
```
{dataset_sample_str}
```

**YOUR MISSION:**
Please analyze the above dataset sample and create a validation script that:

1. **INTELLIGENTLY ANALYZES AND EXTRACTS DATA** - Automatically inspect the sample to:
   - Identify available fields and their purposes (task description, actions, HTML content, etc.)
   - Extract and parse all relevant information from appropriate fields
   - Handle different data formats and nested structures gracefully

2. **CREATES A COMPREHENSIVE SCORING SYSTEM** that evaluates multiple dimensions:

   ** DATA QUALITY METRICS (0-1 points each):**
   - **Completeness Score**: Are all expected fields present and non-empty?
   - **Format Consistency Score**: Do data formats follow expected patterns?
   - **Content Validity Score**: Is the content meaningful and properly structured?
   - **Action Sequence Quality**: Are actions/steps logically ordered and complete?
   - **Domain-Specific Accuracy**: Does content meet domain requirements?

   ** WEIGHTED SCORING SYSTEM:**
   - Assign different weights to each metric based on dataset importance
   - Calculate weighted average for final composite score
   - Provide clear scoring criteria and thresholds
   - Include penalty system for critical failures

   ** INTELLIGENT EVALUATION LOGIC:**
   - Automatically determine relevant quality dimensions for this dataset type
   - Apply dataset-specific validation rules (web actions, planning steps, code quality, etc.)
   - Cross-validate information consistency across different fields
   - Detect common data quality issues and anti-patterns

3. **PROVIDES ACTIONABLE REPORTING** including:
   - **Overall Quality Score** (0-100) with clear interpretation
   - **Detailed breakdown** of each scoring dimension
   - **Specific issues found** with severity levels and fix suggestions
   - **Quality trends** and patterns identified in the data
   - **Recommendations** for data improvement
**CRITICAL REQUIREMENTS:**

**SMART DATA ANALYSIS**: Your script must intelligently analyze the provided sample to understand its structure
**FLEXIBLE EXTRACTION**: Work with whatever fields and formats are actually present
**ROBUST VALIDATION**: Create meaningful validation logic based on the discovered data structure
**COMPREHENSIVE SCORING**: Implement detailed scoring with explanations
**ERROR HANDLING**: Handle missing, malformed, or unexpected data gracefully

**VALIDATION FRAMEWORK TEMPLATE:**

```python
import json
from bs4 import BeautifulSoup
import re
from typing import Dict, List, Any, Optional

class UniversalDatasetValidator:
    def __init__(self, sample_data):
        \"\"\"Initialize with a dataset sample for structure analysis\"\"\"
        self.sample = sample_data
        self.structure_analysis = self.analyze_data_structure()
        
    def analyze_data_structure(self) -> Dict[str, Any]:
        \"\"\"Analyze the dataset sample to understand its structure\"\"\"
        # TODO: Implement intelligent analysis of the sample data
        # Inspect fields, understand data types, identify key information
        # Return analysis results including field mappings
        pass
        
    def extract_task_info(self) -> Dict[str, str]:
        \"\"\"Extract task information based on structure analysis\"\"\"
        # TODO: Use structure analysis to find and extract task description
        # Look for fields that contain task/instruction information
        pass
        
    def extract_actions(self) -> List[Any]:
        \"\"\"Extract action sequence based on structure analysis\"\"\"
        # TODO: Use structure analysis to find and extract actions/steps
        # Handle different action formats (list of strings, list of dicts, etc.)
        pass

    # if the dataset is a web dataset, we need to extract the html content
    def extract_html_content(self) -> Optional[str]:
        \"\"\"Extract HTML content if available\"\"\"
        # TODO: Use structure analysis to find HTML content
        # Look for fields containing HTML data
        pass
        
    def generate_selector(self, action_info: Any) -> str:
        \"\"\"Generate CSS selector from action information\"\"\"
        # TODO: Implement smart selector generation based on action format
        # Handle different action representations
        pass
        
    def validate_sample(self) -> Dict[str, Any]:
        \"\"\"Main validation function\"\"\"
        # TODO: Implement comprehensive validation logic
        # 1. Validate data structure completeness
        # 2. Validate action sequence logical consistency
        # 3. Validate HTML-action compatibility (if applicable)
        # 4. Calculate detailed scores with explanations
        # Return comprehensive validation results
        pass

# Example usage that demonstrates the validator working with the actual sample
def main():
    # Use the actual dataset sample provided
    sample_data = {dataset_sample_str}
    
    # Create validator and analyze the sample
    validator = UniversalDatasetValidator(sample_data)
    
    print("=== DATASET SAMPLE ANALYSIS ===")
    print(f"Dataset: {dataset_name}")
    print(f"Sample Keys: {{list(sample_data.keys())}}")
    print(f"Structure Analysis: {{validator.structure_analysis}}")
    
    # Perform validation
    results = validator.validate_sample()
    
    print("\\n=== VALIDATION RESULTS ===")
    print(f"Validation Results: {{results}}")
    
if __name__ == "__main__":
    main()
```

**IMPLEMENTATION GUIDELINES:**

1. **ANALYZE FIRST**: Look at the actual sample data structure before writing extraction logic
2. **BE ADAPTIVE**: Don't assume specific field names - discover them from the data
3. **HANDLE VARIATIONS**: Use try-catch and .get() methods for safe data access
4. **VALIDATE MEANINGFULLY**: Create validation criteria that make sense for this data type
5. **SCORE COMPREHENSIVELY**: Provide detailed scores with clear explanations
6. **USE AVAILABLE DATA**: Work with whatever fields are actually present in the sample

**SPECIAL CONSIDERATIONS FOR WEB INTERACTION DATA:**
- If the sample contains HTML and actions, validate HTML-action compatibility
- Generate appropriate CSS selectors for element targeting
- Validate action sequences for logical consistency
- Check for missing or malformed action data
- Ensure actions can realistically be performed on the given HTML

**OUTPUT REQUIREMENTS:**
- Generate ONLY executable Python code
- Do NOT include any Markdown formatting (no ```)
- Start directly with Python imports
- Include complete working example with the actual sample data
- Make the script immediately executable and show comprehensive results
- Ensure the script works with the exact sample provided

**CRITICAL SUCCESS CRITERIA:**
1. Script must work with the provided dataset sample structure
2. Intelligently analyze and use the actual fields in the sample
3. Generate meaningful validation scores based on the actual data
4. Handle variations in field values gracefully
5. Include working example with the actual dataset sample
6. Be immediately executable and show detailed results

Please analyze the provided dataset sample and generate a complete validation script that intelligently works with this specific data structure.
"""
    return prompt.strip()


def get_debug_prompt(script_code: str, error_message: str) -> str:
    """Generate debugging prompt for system feedback stage"""
    
    debug_prompt = f"""
The following Python validation script has a runtime error. Please fix the error and return the corrected complete script.

**ORIGINAL SCRIPT:**
```python
{script_code}
```

**ERROR MESSAGE:**
{error_message}

**DEBUGGING INSTRUCTIONS:**
1. Analyze the error message carefully
2. Identify the root cause of the issue
3. Fix the error while maintaining the dataset analysis functionality
4. Ensure all imports are included
5. Make sure the script can handle edge cases and field variations

**COMMON FIXES:**
- Add proper imports if missing
- Fix syntax errors and typos
- Add error handling for data access
- Use .get() methods for safe dictionary access
- Handle cases where expected fields might be missing
- Fix JSON parsing issues
- Handle different data types appropriately

**MAINTAIN CORE FUNCTIONALITY:**
- Keep the intelligent data structure analysis
- Preserve the adaptive validation logic
- Ensure the script works with the actual dataset sample
- Maintain comprehensive error handling

Please provide the complete corrected script:
"""
    return debug_prompt.strip()


def get_reflection_prompt(script_code: str) -> str:
    """Generate self-reflection prompt for code review stage"""
    
    reflection_prompt = f"""
You are reviewing a validation script that analyzes dataset samples intelligently. 
Please review and improve the script to ensure it's robust and comprehensive.

**SCRIPT TO REVIEW:**
```python
{script_code}
```

**REVIEW CHECKLIST:**

1. **DATA STRUCTURE ANALYSIS**:
   - Does the script properly analyze the sample data structure?
   - Can it handle different field names and data formats?
   - Is the structure analysis comprehensive and robust?

2. **DATA EXTRACTION**:
   - Are data extractions safe and error-handled?
   - Does it handle missing or malformed data gracefully?
   - Can it work with various data formats?

3. **VALIDATION LOGIC**:
   - Is the validation logic meaningful for this type of data?
   - Does it provide comprehensive scoring?
   - Are edge cases handled properly?

4. **CODE QUALITY**:
   - Are all necessary imports included?
   - Is error handling implemented properly?
   - Is the code well-structured and readable?

5. **EXAMPLE USAGE**:
   - Does the example actually work with real data?
   - Are results properly displayed?
   - Is the script immediately executable?

**IMPROVEMENT AREAS:**
- Enhance data structure analysis if needed
- Add better error handling
- Improve validation logic comprehensiveness
- Ensure robust field access
- Add more detailed scoring explanations
- Make the script more adaptive to different data formats

Please provide the COMPLETE improved script with better analysis and validation capabilities:
"""
    return reflection_prompt.strip()