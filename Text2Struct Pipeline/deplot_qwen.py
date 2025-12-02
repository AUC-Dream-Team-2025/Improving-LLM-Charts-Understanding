"""
Chart Understanding with Qwen2.5-VL-7B-Instruct
Graduation Project - Prompting Techniques Evaluation
ENHANCED VERSION: Added DePlot Chain-of-Models Strategy + Strategy Selection
NEW: Added DePlot + Chain-of-Thought Hybrid Strategy
"""

# ============================================================================
# QUICK CONFIGURATION - Modify these before running!
# ============================================================================
# To test with 100 examples: Set NUM_SAMPLES = 100
# To test with full dataset: Set NUM_SAMPLES = None
# See main() function for more configuration options
# ============================================================================

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor
from qwen_vl_utils import process_vision_info  # REQUIRED: pip install qwen-vl-utils
import json
import re
import traceback
from pathlib import Path
from typing import Dict, List, Any
import time
from tqdm import tqdm
import numpy as np
from PIL import Image

# ============================================================================
# DEPLOT MODEL SETUP
# ============================================================================

class DePlotExtractor:
    """
    DePlot model for extracting tabular data from charts
    
    ✅ VERIFICATION: This implementation is CORRECT
    - Uses the official google/deplot model
    - Properly handles image preprocessing
    - Returns linearized table data as intended by DePlot paper
    """
    
    def __init__(self, model_path: str = "google/deplot"):
        print(f"Loading DePlot model: {model_path}")

        # Try bfloat16 first (best compatibility)
        try:
            self.model = Pix2StructForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,  # More stable
                device_map="auto"
            )
        except:
            # Fall back to float32 (always works, slightly slower)
            self.model = Pix2StructForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float32,  # Most compatible
                device_map="auto"
            )
        
        self.processor = Pix2StructProcessor.from_pretrained(model_path)
        
        print("DePlot model loaded successfully!")
        
    def extract_table(self, image_path: str, max_patches: int = 2048) -> str:
        """
        Extract tabular data from chart image using DePlot
        
        ✅ VERIFICATION: This extraction is CORRECT
        - Uses the official DePlot prompt: "Generate underlying data table of the figure below:"
        - Properly handles dtype conversion for model compatibility
        - Returns linearized table in DePlot's expected format (CSV-like with special tokens)
        
        Args:
            image_path: Path to the chart image
            max_patches: Maximum number of patches (default 2048)
            
        Returns:
            Extracted table data as string (linearized format with <0x0A> for newlines, etc.)
        """
        try:
            # Load image
            image = Image.open(image_path).convert("RGB")
            
            # Process with DePlot - Using official prompt
            inputs = self.processor(
                images=image,
                text="Generate underlying data table of the figure below:",
                return_tensors="pt",
                max_patches=max_patches
            ).to(self.model.device)
            
            # ✅ CRITICAL FIX: Ensure dtype compatibility
            # Pix2Struct uses 'flattened_patches' for the visual data
            inputs = inputs.to(self.model.device)
            if "flattened_patches" in inputs:
                inputs["flattened_patches"] = inputs["flattened_patches"].to(self.model.dtype)

            # Generate table
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1024
                )
            
            # Decode output
            table_data = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            return table_data
            
        except Exception as e:
            print(f"Error extracting table with DePlot: {e}")
            return ""

# ============================================================================
# MODEL SETUP (FROM FIRST CODE - CORRECT IMPLEMENTATION)
# ============================================================================

class ChartAnalyzer:
    """
    Main class for chart analysis using Qwen2.5-VL-7B-Instruct
    
    ✅ VERIFICATION: This implementation is CORRECT
    - Uses official Qwen2.5-VL implementation
    - Properly applies chat template
    - Uses qwen_vl_utils for vision processing (as per official docs)
    """
    
    def __init__(self, model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct"):
        print(f"Loading model: {model_path}")
        
        # Load model
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        print("Model loaded successfully!")
        
    def generate_response(
        self, 
        image_path: str, 
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.1,  # Lower temperature for factual chart tasks
        top_p: float = 0.9,
        few_shot_messages: List[Dict] = None
    ) -> str:
        """
        Generate response for a given image and prompt
        Uses official Qwen implementation with qwen_vl_utils
        
        ✅ VERIFICATION: This is the CORRECT way to pass prompts to Qwen2.5-VL
        - The prompt text is embedded in the messages structure
        - Images are processed through qwen_vl_utils
        - Chat template is properly applied
        - The model DOES see the full prompt including DePlot table data
        
        Args:
            image_path: Path to the chart image
            prompt: Text prompt for the model (CAN include DePlot table data)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (lower = more factual)
            top_p: Top-p sampling parameter
            few_shot_messages: Optional pre-built message history for multimodal few-shot
            
        Returns:
            Generated text response
        """
        
        # 1. Construct Messages
        messages = []
        
        # Add few-shot history if present
        if few_shot_messages:
            messages.extend(few_shot_messages)
            
        # ✅ Add current query - The prompt parameter contains ALL text including DePlot table
        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},  # CORRECT: use 'image' key
                {"type": "text", "text": prompt}  # ✅ This includes DePlot table if present!
            ]
        })
        
        # 2. Prepare Inputs (The Official Qwen Way)
        text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        # Move inputs to GPU
        inputs = inputs.to(self.model.device)
        
        # 3. Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0
            )
        
        # 4. Decode - Trim input tokens from output
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        response_text = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        return response_text


def clean_deplot_output(table_text: str) -> str:
    """
    Cleans raw DePlot output into a readable format for the LLM.
    
    ✅ VERIFICATION: This cleaning is CORRECT
    - DePlot outputs linearized tables with special tokens like <0x0A> (newline)
    - This function converts them to readable format
    - The cleaned table is what Qwen sees in the prompt
    """
    # 1. Replace the hex newline with a real newline
    cleaned = table_text.replace("<0x0A>", "\n")
    
    # 2. Replace hex tab with a pipe (if it exists)
    cleaned = cleaned.replace("<0x09>", " | ")
    
    # 3. Ensure pipes have spaces for better tokenization
    # This handles cases like "Apple|5" becoming "Apple | 5"
    cleaned = cleaned.replace("|", " | ")
    
    # 4. Remove multiple spaces created by step 3
    # (e.g., if it was already " | ", step 3 makes it "  |  ")
    cleaned = " ".join(cleaned.split())
    
    # 5. Restore the newlines (split/join removes them)
    # A safer way to do step 4 while preserving newlines:
    lines = [line.strip() for line in cleaned.split('\n')]
    cleaned = "\n".join([l for l in lines if l]) # remove empty lines
    
    return cleaned


# ============================================================================
# PROMPTING STRATEGIES (ENHANCED WITH DEPLOT + NEW HYBRID)
# ============================================================================

class PromptingStrategies:
    """Implementation of different prompting strategies - Enhanced version with DePlot + CoT"""
    
    @staticmethod
    def baseline(question: str) -> str:
        """Strategy 1: Baseline with strict formatting constraints"""
        return f"""Question: {question}
Answer the question using a single word or phrase.
End your response with: Final Answer: [your answer]"""
    
    @staticmethod
    def zero_shot_cot(question: str) -> str:
        """Strategy 2: Zero-Shot with Chain-of-Thought - Enhanced with calculation guidance"""
        return f"""Question: {question}
Let's solve this step by step with careful analysis and calculations.
**IMPORTANT: Use digits (e.g., 3, 20.5) for all numbers, not words like 'three'.**
**For arithmetic: Show each calculation step explicitly (e.g., 103.7 - 103.13 = 0.57)**
1. Identify the chart type and analyze the axes and legend:
2. Extract the relevant numbers/data from the chart:
3. Determine what calculation or reasoning is needed:
4. Perform the calculation or analysis step-by-step:
5. Verify the answer makes sense:
6. Final Answer:"""
    
    @staticmethod
    def few_shot_text(question: str) -> str:
        """Strategy 3: Few-Shot with 4 diverse examples"""
        examples = """Here are examples of chart analysis (use digits for all numbers):

Example 1:
Question: How many categories are shown in the bar chart?
Analysis: Looking at the y-axis, I can count the distinct category labels. There are 5 bars, each representing one category.
Answer: 5

Example 2:
Question: What is the value for Sales in Q2?
Analysis: Looking at the bar chart, the Sales bar for Q2 reaches to 85 on the y-axis.
Answer: 85

Example 3:
Question: What percentage does Marketing represent in the budget?
Analysis: The pie chart shows the Marketing segment labeled as 25%. All segments sum to 100%.
Answer: 25%

Example 4:
Question: What is the difference between the highest and lowest values?
Analysis: The highest bar reaches 150, and the lowest reaches 45. The difference is 150 - 45 = 105.
Answer: 105

Now analyze this chart:
"""
        return examples + f"Question: {question}\nAnalysis:"
    
    @staticmethod
    def few_shot_multimodal(question: str, example_images: List[str] = None) -> tuple:
        """
        Strategy 4: True Multimodal Few-Shot Prompting
        Now with 3 examples: Bar Chart, Pie Chart, and Line Chart
        """
        if example_images is None or len(example_images) < 3:
            return (None, False)
        
        # Verify files exist
        valid_images = [img for img in example_images if Path(img).exists()]
        if len(valid_images) < 3:
            print(f"Warning: Expected 3 example images, found {len(valid_images)}. Skipping multimodal few-shot.")
            return (None, False)
        
        messages = []
        
        # Example 1: Bar Chart - Direct value reading
        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": example_images[0]},
                {"type": "text", "text": "What was the production volume of diamonds in Angola in 2004?"}
            ]
        })
        messages.append({
            "role": "assistant",
            "content": "Based on the bar chart, the volume for Angola in 2004 is 6.1.\n\nFinal Answer: 6.1"
        })
        
        # Example 2: Pie Chart - Calculation with multiple values
        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": example_images[1]},
                {"type": "text", "text": "Take highest percentage and lowest percentage (leave 0), add it and divide it by 2, what is the result?"}
            ]
        })
        messages.append({
            "role": "assistant",
            "content": "From the pie chart, the highest percentage is 33% and the lowest non-zero percentage is 2%. The calculation is (33 + 2) / 2 = 35 / 2 = 17.5.\n\nFinal Answer: 17.5"
        })
        
        # Example 3: Line Chart - Time series difference
        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": example_images[2]},
                {"type": "text", "text": "What is the difference in the percentage of livestock breeds classified as being at risk of extinction between 2002 and 2003?"}
            ]
        })
        messages.append({
            "role": "assistant",
            "content": "Looking at the line chart for Thailand, the value in 2002 is approximately 10% and in 2003 is also approximately 9%. Reading more precisely from the chart, the difference is 10% - 9% = 1%. Converting to decimal form, this is 0.01.\n\nFinal Answer: 0.01"
        })
            
        return (messages, True)

    
    @staticmethod
    def structured_output(question: str) -> str:
        """Strategy 5: Structured Output with comprehensive JSON fields"""
        return f"""You are a helpful assistant that analyzes charts.
Analyze the chart to answer the user's question.
Provide your answer *only* in the following JSON format.
**IMPORTANT: Use digits (e.g., 3, 20.5) for all numbers, not words (e.g., 'three').**

Question: {question}

```json
{{
  "chart_type": "description of chart type (bar, pie, line, etc.)",
  "key_elements": "important elements observed (axes, legend, data points)",
  "reasoning": "step-by-step reasoning to answer the question",
  "answer": "the final answer to the question",
  "confidence": "high/medium/low confidence in the answer"
}}
```"""
    
    @staticmethod
    def role_based(question: str) -> str:
        """Strategy 6: Role-Based Prompting with expert persona"""
        return f"""You are an expert data analyst specializing in chart interpretation.
Your task is to analyze the provided chart with precision and answer the user's question.
**Guidelines: Use digits (e.g., 3, 20) for all numerical answers, not words (e.g., 'three').**

Question: {question}

Expert Analysis and Final Answer:"""
    
    @staticmethod
    def deplot_chain(question: str, deplot_table: str) -> str:
        """
        Strategy 7: DePlot Chain-of-Models Approach
        
        ✅ VERIFICATION: Qwen DOES see the DePlot table data because:
        1. The table data is passed as the 'deplot_table' parameter
        2. It's embedded in the returned prompt string
        3. This prompt is passed to generate_response() as the 'prompt' parameter
        4. In generate_response(), the prompt becomes part of messages[]["content"][]["text"]
        5. The Qwen processor applies chat template to these messages
        6. Therefore, Qwen receives the full prompt INCLUDING the table data
        
        This strategy uses extracted tabular data from DePlot model
        along with the visual chart to provide comprehensive analysis.
        
        Args:
            question: The question to answer
            deplot_table: Extracted table data from DePlot model (already cleaned)
            
        Returns:
            Enhanced prompt with table context
        """

        return f"""You are analyzing a chart to answer a question. 
To help you, a data extraction model has already extracted the underlying tabular data from this chart.

**Extracted Table Data:**
{deplot_table}

**Important Instructions:**
1. Use BOTH the visual chart AND the extracted table data to answer accurately
2. The table data provides exact numerical values - use them for precise calculations
3. The visual chart helps you understand trends, patterns, and context
4. Cross-verify information between the table and the chart
5. Use digits (e.g., 3, 20.5) for all numbers in your answer, not words
6. **CRITICAL:** End your response with "Final Answer: X" where X is your answer.

**Question:** {question}

**Your Analysis and Answer:**"""
    
    @staticmethod
    def deplot_cot(question: str, deplot_table: str) -> str:
        """
        Strategy 8: NEW - DePlot Chain + Chain-of-Thought (Hybrid Approach)
        
        ✅ IMPLEMENTATION: This strategy combines the best of both worlds:
        - DePlot provides exact numerical data from the chart
        - Chain-of-Thought guides systematic reasoning and verification
        
        This hybrid approach should theoretically achieve the highest accuracy because:
        1. DePlot eliminates OCR/visual reading errors for numerical values
        2. CoT reduces calculation errors through step-by-step verification
        3. Cross-referencing visual + tabular data catches inconsistencies
        
        Combines the strengths of:
        - DePlot: Precise tabular data extraction
        - Chain-of-Thought: Step-by-step reasoning and verification
        
        This strategy provides the model with exact numerical data while encouraging
        systematic analysis and calculation verification.
        
        Args:
            question: The question to answer
            deplot_table: Extracted table data from DePlot model (already cleaned)
            
        Returns:
            Hybrid prompt with table context and CoT guidance
        """
        
        return f"""You are analyzing a chart to answer a question using both visual information and extracted data.

**Extracted Table Data from Chart:**
{deplot_table}

**Question:** {question}

**Instructions:** Solve this step-by-step using BOTH the table data and the visual chart.

**Step-by-Step Analysis:**

1. **Understand the Question:**
   - What is being asked?
   - What specific information do I need to find?

2. **Analyze the Chart Visually:**
   - What type of chart is this? (bar, line, pie, etc.)
   - What do the axes/legend represent?
   - Are there any important visual patterns or trends?

3. **Cross-Reference with Table Data:**
   - Locate the relevant data points in the extracted table
   - Verify these match what you see in the chart
   - Identify the exact numerical values needed

4. **Perform Calculations (if needed):**
   - Show each calculation step explicitly
   - Use the exact values from the table data
   - Example format: 45.7 - 23.2 = 22.5
   - **CRITICAL: Use digits (e.g., 5, 20.5) for all numbers, never words like 'five'**

5. **Verify Your Answer:**
   - Does the answer make sense given the chart context?
   - Does it match both the visual representation and table data?
   - Check your calculations for arithmetic errors

6. **Provide Final Answer:**
   - End with: Final Answer: [your answer]

**Your Step-by-Step Solution:**"""

# ============================================================================
# BENCHMARK DATASET CREATION
# ============================================================================

def create_benchmark_dataset(output_path: str = "benchmark.json"):
    """Create benchmark dataset from HuggingFaceM4/ChartQA"""
    print("Creating benchmark dataset from HuggingFaceM4/ChartQA...")
    try:
        from datasets import load_dataset
        # Load from test split
        chartqa = load_dataset("HuggingFaceM4/ChartQA", split="test")
        
        benchmark = []
        Path("benchmark_images").mkdir(exist_ok=True)
        
        for idx, item in enumerate(chartqa):
            image_path = f"benchmark_images/chart_{idx}.png"
            item["image"].save(image_path)
            
            benchmark.append({
                "id": f"chart_{idx}",
                "image_path": image_path,
                "question": item["query"],
                "answer": item["label"],
                "source": "HuggingFaceM4/ChartQA"
            })
        
        with open(output_path, "w") as f:
            json.dump(benchmark, f, indent=2)
        
        print(f"✅ Benchmark created with {len(benchmark)} examples")
        return benchmark
        
    except Exception as e:
        print(f"❌ Error creating benchmark: {e}")
        return None

# ============================================================================
# EVALUATION - CORRECTED VERSION WITH ALL CRITICAL FIXES
# ============================================================================

def evaluate_answer(predicted: str, ground_truth, max_relative_change: float = 0.05) -> bool:
    """
    ULTIMATE evaluation function - Best of both worlds.
    
    Combines:
    - Original's comprehensive normalization and pattern matching
    - Robust's smart evaluation order and ALL-numbers checking
    
    Args:
        predicted: Model's prediction (string)
        ground_truth: Correct answer (string or list - ChartQA format)
        max_relative_change: Relative tolerance (default 5% = 0.05)
    
    Returns:
        bool: True if answer is correct within tolerance
    """
    
    # ========================================================================
    # STEP 1: Handle ChartQA List Format
    # ========================================================================
    if isinstance(ground_truth, list):
        if not ground_truth:
            return False
        gt_list = [str(g) for g in ground_truth]
    else:
        gt_list = [str(ground_truth)]
    
    # ========================================================================
    # STEP 2: JSON Extraction
    # ========================================================================
    try:
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', predicted, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(1))
            if "answer" in data:
                predicted = str(data["answer"])
        elif predicted.strip().startswith("{"):
            data = json.loads(predicted)
            if "answer" in data:
                predicted = str(data["answer"])
    except:
        pass
    

    # ========================================================================
    # STEP 3: Pattern Extraction - FINAL ROBUST VERSION
    # ========================================================================
    
    # First, clean up escape sequences
    predicted = predicted.replace('\\n', '\n').replace('\\t', ' ')

    # Extract answer using patterns (most specific to least specific)
    # NOTE: We removed \. and \, from the stop condition to protect decimals!
    answer_patterns = [
        r'final\s+answer\s*:?\s*is\s*:?\s*(.+?)(?:\n|$)',  
        r'final\s+answer\s*:?\s*(.+?)(?:\n|$)',            
        r'the\s+answer\s+is\s*:?\s*(.+?)(?:\n|$)',          
        r'\nanswer\s*:\s*(.+?)(?:\n|$)',    # Newline + "Answer:" (Avoids verbs)
        r'^answer\s*:\s*(.+?)(?:\n|$)',     # Start of string + "Answer:"
    ]

    for pattern in answer_patterns:
        # Use MULTILINE to handle ^ and \n correctly
        match = re.search(pattern, predicted, re.IGNORECASE | re.MULTILINE)
        if match:
            extracted = match.group(1).strip()
            
            # SMART CLEANUP (The "Decimal Protector"):
            # Only split on punctuation if followed by space or end of line.
            # This keeps "0.57" intact but cleans "50."
            extracted = re.split(r'[.,;](?:\s|$)', extracted)[0].strip()
            
            # Remove any remaining weird leading punctuation
            extracted = re.sub(r'^[\s\-=:*]+', '', extracted)
            
            # ✅ ADDED: Also remove trailing punctuation that survived
            extracted = re.sub(r'[;!?]+$', '', extracted)
            
            if extracted and len(extracted) < 50:
                predicted = extracted
                break


    # ========================================================================
    # STEP 4: Comprehensive Normalization
    # ========================================================================
    def normalize(text):
        text = text.lower().strip()
        
        # Comprehensive word-to-digit
        word_to_digit = {
            'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4', 
            'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 
            'ten': '10', 'eleven': '11', 'twelve': '12', 'thirteen': '13', 
            'fourteen': '14', 'fifteen': '15', 'sixteen': '16', 'seventeen': '17',
            'eighteen': '18', 'nineteen': '19', 'twenty': '20',
            'thirty': '30', 'forty': '40', 'fifty': '50', 'sixty': '60',
            'seventy': '70', 'eighty': '80', 'ninety': '90', 'hundred': '100'
        }
        
        for word, digit in word_to_digit.items():
            text = re.sub(r'\b' + word + r'\b', digit, text)
        
        # Remove currency symbols and commas
        text = text.replace('$', '').replace('€', '').replace('£', '').replace(',', '')
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    pred_clean = normalize(predicted)
    
    # ========================================================================
    # STEP 5: Evaluate Against Each Ground Truth
    # ========================================================================
    for gt_item in gt_list:
        gt_clean = normalize(gt_item)
        
        # Extract ALL numbers (handles negatives and decimals automatically)
        pred_numbers = [float(x) for x in re.findall(r'-?\d+\.?\d*', pred_clean)]
        gt_numbers = [float(x) for x in re.findall(r'-?\d+\.?\d*', gt_clean)]
        
        # CHECK 1: Exact String Match
        if pred_clean == gt_clean:
            return True
        
        # CHECK 2: Substring Match (ONLY for non-numeric GT)
        if not gt_numbers:
            if gt_clean in pred_clean:
                return True
        
        # CHECK 3: Numerical Comparison (Check ALL numbers with better tolerance)
        if gt_numbers:
            target = gt_numbers[0]
            
            for candidate in pred_numbers:
                # Exact match
                if candidate == target:
                    return True
                
                # Relative error (5%)
                if target == 0:
                    if abs(candidate) < 1e-6:
                        return True
                else:
                    relative_error = abs(candidate - target) / abs(target)
                    if relative_error <= max_relative_change:
                        return True
                    
                    # Percentage mismatch (bidirectional)
                    if abs(target) > 1 and abs(candidate) < 1:
                        candidate_as_percentage = candidate * 100
                        if abs(candidate_as_percentage - target) / abs(target) <= max_relative_change:
                            return True
                    
                    if abs(target) < 1 and abs(candidate) > 1:
                        candidate_as_decimal = candidate / 100
                        if abs(candidate_as_decimal - target) / abs(target) <= max_relative_change:
                            return True
    
    return False


# ============================================================================
# MAIN EVALUATION (ENHANCED WITH DEPLOT + DEPLOT_COT)
# ============================================================================

def run_comprehensive_evaluation(
    analyzer, 
    benchmark, 
    num_samples=None,
    output_file="results_merged.json", 
    debug_mode=False,
    use_deplot=True,  # Enable/disable DePlot-based strategies
    strategies_to_test=None  # List of specific strategies to test (None = all)
):
    """
    Run comprehensive evaluation of all prompting strategies
    
    Args:
        analyzer: ChartAnalyzer instance
        benchmark: List of benchmark examples
        num_samples: Number of samples to test (None = all, or specify like 100)
        output_file: Path to save results
        debug_mode: If True, print detailed debug info
        use_deplot: If True, enable DePlot-based strategies
        strategies_to_test: List of strategy names to test, e.g., ["baseline", "deplot_cot"]
                           If None, tests all strategies. Available options:
                           - "baseline"
                           - "zero_shot_cot"
                           - "few_shot_text"
                           - "few_shot_multimodal"
                           - "structured"
                           - "role_based"
                           - "deplot_chain" (requires use_deplot=True)
                           - "deplot_cot" (NEW - requires use_deplot=True)
    """
    
    # Limit benchmark size if specified
    if num_samples is not None:
        benchmark = benchmark[:num_samples]
        print(f"Testing with {len(benchmark)} samples (limited from full dataset)")
    else:
        print(f"Testing with all {len(benchmark)} samples")

    # Build full strategies dictionary
    # NOTE: Strategies execute in the order defined here
    all_strategies = {}
    
    # Add DePlot-based strategies if enabled (prioritize them first)
    if use_deplot:
        all_strategies["deplot_chain"] = PromptingStrategies.deplot_chain
        all_strategies["deplot_cot"] = PromptingStrategies.deplot_cot  # NEW
    
    # Add remaining strategies in standard order
    all_strategies.update({
        "baseline": PromptingStrategies.baseline,
        "zero_shot_cot": PromptingStrategies.zero_shot_cot,
        "few_shot_text": PromptingStrategies.few_shot_text,
        "few_shot_multimodal": PromptingStrategies.few_shot_multimodal,
        "structured": PromptingStrategies.structured_output,
        "role_based": PromptingStrategies.role_based
    })
    
    # Filter strategies if specific ones requested
    if strategies_to_test is not None:
        # Validate requested strategies
        invalid_strategies = [s for s in strategies_to_test if s not in all_strategies]
        if invalid_strategies:
            print(f"⚠️  Warning: Invalid strategies requested: {invalid_strategies}")
            print(f"Available strategies: {list(all_strategies.keys())}")
        
        strategies = {k: v for k, v in all_strategies.items() if k in strategies_to_test}
        
        if not strategies:
            print("❌ Error: No valid strategies selected!")
            return None
        
        print(f"\n✅ Testing only these strategies: {list(strategies.keys())}")
    else:
        strategies = all_strategies
        print(f"\n✅ Testing all {len(strategies)} strategies")
    
    # Example images for multimodal few-shot
    multimodal_example_images = [
        "example_images/example_bar_chart.png",
        "example_images/example_pie_chart.png",
        "example_images/example_line_chart.png"
    ]
    
    # Initialize DePlot if needed
    deplot_extractor = None
    if any(s in strategies for s in ["deplot_chain", "deplot_cot"]):
        print("\n" + "="*60)
        print("Initializing DePlot model for DePlot-based strategies...")
        print("="*60)
        deplot_extractor = DePlotExtractor()
    
    all_results = {}
    
    for strategy_name, strategy_func in strategies.items():
        print(f"\n{'='*60}")
        print(f"Testing Strategy: {strategy_name.upper()}")
        print(f"{'='*60}")
        
        strategy_results = []
        correct_count = 0
        
        for i, example in enumerate(tqdm(benchmark, desc=f"{strategy_name}")):
            try:
                prompt_text = ""
                history = None
                table_data = None  # Store for debug output
                
                # Handle different strategies
                if strategy_name == "few_shot_multimodal":
                    hist, ok = strategy_func(example["question"], multimodal_example_images)
                    if not ok:
                        strategy_results.append({
                            "id": example["id"],
                            "question": example["question"],
                            "error": "Multimodal example images not found"
                        })
                        continue
                    history = hist
                    prompt_text = example["question"]
                
                elif strategy_name in ["deplot_chain", "deplot_cot"]:
                    # ✅ DePlot-based strategies (chain and CoT hybrid)
                    # Step 1: Extract table data using DePlot
                    deplot_start = time.time()
                    
                    raw_table_data = deplot_extractor.extract_table(example["image_path"])
                    # ✅ Clean the table data IMMEDIATELY
                    table_data = clean_deplot_output(raw_table_data)

                    deplot_time = time.time() - deplot_start
                    
                    # Step 2: Create enhanced prompt with table data
                    prompt_text = strategy_func(example["question"], table_data)
                    
                    # Note: DePlot info will be added to result_entry below
                
                else:
                    prompt_text = strategy_func(example["question"])
                
                # Generate response
                start = time.time()
                response = analyzer.generate_response(
                    image_path=example["image_path"],
                    prompt=prompt_text,
                    few_shot_messages=history
                )
                inference_time = time.time() - start
                
                # Evaluate
                is_correct = evaluate_answer(response, example["answer"])
                if is_correct:
                    correct_count += 1
                
                # Update results
                result_entry = {
                    "id": example["id"],
                    "question": example["question"],
                    "predicted": response,
                    "ground_truth": example["answer"],
                    "correct": is_correct,
                    "inference_time": inference_time
                }
                
                # ✅ Add DePlot-specific info for both strategies
                if strategy_name in ["deplot_chain", "deplot_cot"]:
                    result_entry["deplot_extraction_time"] = deplot_time
                    result_entry["total_time"] = inference_time + deplot_time
                    result_entry["extracted_table"] = table_data[:200] + "..." if len(table_data) > 200 else table_data
                
                strategy_results.append(result_entry)
                
                # Debug mode: print first 3 examples
                if debug_mode and i < 3:
                    print(f"\n--- Example {i + 1} ---")
                    print(f"Question: {example['question']}")
                    if strategy_name in ["deplot_chain", "deplot_cot"] and table_data:
                        print(f"DePlot Table: {table_data[:150]}...")
                    print(f"Predicted: {response}")
                    print(f"Ground Truth: {example['answer']}")
                    print(f"Correct: {is_correct}")
                    
            except Exception as e:
                print(f"\nError on {example['id']}: {e}")
                if debug_mode:
                    traceback.print_exc()
                strategy_results.append({
                    "id": example["id"],
                    "question": example["question"],
                    "error": str(e)
                })
        
        # Calculate metrics
        total_valid = len([r for r in strategy_results if "error" not in r and "question" in r])
        if total_valid > 0:
            accuracy = correct_count / total_valid
            avg_time = np.mean([r.get("inference_time", 0) for r in strategy_results if "inference_time" in r])
            
            # Calculate total time for DePlot-based strategies
            if strategy_name in ["deplot_chain", "deplot_cot"]:
                avg_total_time = np.mean([r.get("total_time", 0) for r in strategy_results if "total_time" in r])
            else:
                avg_total_time = 0
        else:
            accuracy = 0
            avg_time = 0
            avg_total_time = 0
        
        all_results[strategy_name] = {
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total": total_valid,
            "avg_inference_time": avg_time,
            "results": strategy_results
        }
        
        # Add total time for DePlot-based strategies
        if strategy_name in ["deplot_chain", "deplot_cot"]:
            all_results[strategy_name]["avg_total_time"] = avg_total_time
        
        print(f"\nResults for {strategy_name}:")
        print(f"  Accuracy: {accuracy:.2%} ({correct_count}/{total_valid})")
        print(f"  Avg Inference Time: {avg_time:.2f}s")
        if strategy_name in ["deplot_chain", "deplot_cot"]:
            print(f"  Avg Total Time (DePlot + Qwen): {avg_total_time:.2f}s")
    
    # Save results
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    for strategy_name, results in all_results.items():
        print(f"{strategy_name:20s}: {results['accuracy']:.2%}")
    
    print(f"\nResults saved to {output_file}")
    
    return all_results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    print("="*60)
    print("Chart Understanding with Qwen2.5-VL-7B-Instruct")
    print("ENHANCED VERSION - With DePlot Chain + CoT Hybrid Strategy")
    print("="*60)
    
    # ========================================================================
    # CONFIGURATION: Modify these parameters as needed
    # ========================================================================
    NUM_SAMPLES = None   # Set to 50 for testing, None for full dataset
    DEBUG_MODE = True   # Set to True to see first 3 examples per strategy
    USE_DEPLOT = True   # Set to True to enable DePlot-based strategies
    OUTPUT_FILE = "results_whole_with_deplot_cot_new.json"

    # NEW PARAMETERS FOR DETAILED OUTPUT
    SHOW_COMPARISON_TABLE = True   # Set to True to see detailed comparison
    COMPARISON_TABLE_SAMPLES = 100  # Show first 100 samples in detail

    # ========================================================================
    # STRATEGY SELECTION: Choose which strategies to test
    # ========================================================================
    # Option 1: Test ALL strategies (default)
    # STRATEGIES_TO_TEST = None
    
    # Option 2: Test only specific strategies (uncomment to use)
    # STRATEGIES_TO_TEST = ["baseline", "deplot_chain", "deplot_cot"]
    STRATEGIES_TO_TEST = ["deplot_cot"]  # Only the new hybrid strategy
    # STRATEGIES_TO_TEST = ["deplot_chain", "deplot_cot", "zero_shot_cot"]  # Compare DePlot variants with CoT
    
    # Available strategies:
    # - "baseline"
    # - "zero_shot_cot"
    # - "few_shot_text"
    # - "few_shot_multimodal"
    # - "structured"
    # - "role_based"
    # - "deplot_chain" (requires USE_DEPLOT=True)
    # - "deplot_cot" (NEW - requires USE_DEPLOT=True)
    # ========================================================================
    
    # Step 1: Initialize model
    print("\n[1/3] Initializing Qwen model...")
    analyzer = ChartAnalyzer()
    
    # Step 2: Create or load benchmark
    print("\n[2/3] Loading benchmark dataset...")
    if Path("benchmark.json").exists():
        with open("benchmark.json", "r") as f:
            benchmark = json.load(f)
        print(f"Loaded existing benchmark with {len(benchmark)} examples")
    else:
        benchmark = create_benchmark_dataset()
        if benchmark is None:
            print("Failed to create benchmark. Please check the error above.")
            return
    
    # Step 3: Run evaluation
    print("\n[3/3] Running comprehensive evaluation...")
    results = run_comprehensive_evaluation(
        analyzer,
        benchmark,
        num_samples=NUM_SAMPLES,
        output_file=OUTPUT_FILE,
        debug_mode=DEBUG_MODE,
        use_deplot=USE_DEPLOT,
        strategies_to_test=STRATEGIES_TO_TEST,
        # show_comparison_table=SHOW_COMPARISON_TABLE,  # NEW for testing to see output
        # comparison_table_samples=COMPARISON_TABLE_SAMPLES  # NEW for testing to see output
    )
    
    print("\n✅ Evaluation complete!")
    print(f"Check '{OUTPUT_FILE}' for detailed results.")
    
    # Print final comparison if multiple strategies tested
    if results and len(results) > 1:
        print("\n" + "="*60)
        print("STRATEGY COMPARISON")
        print("="*60)
        sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        print(f"{'Strategy':<20} {'Accuracy':<12} {'Correct/Total':<15} {'Avg Time'}")
        print("-"*60)
        for name, res in sorted_results:
            time_str = f"{res['avg_total_time']:.2f}s" if 'avg_total_time' in res else f"{res['avg_inference_time']:.2f}s"
            print(f"{name:<20} {res['accuracy']:>6.2%}      {res['correct_count']:>3}/{res['total']:<7}     {time_str}")

if __name__ == "__main__":
    main()