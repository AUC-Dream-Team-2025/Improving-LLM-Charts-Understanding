# ================================================================
# PADDLEPADDLE MICROSERVICE: PP-Chart2Table API Server
# ================================================================
# Run this in a separate environment with PaddlePaddle installed
# Usage: uvicorn paddle_api_server:app --host 0.0.0.0 --port 8000

import os
# Set before any paddle imports
os.environ['FLAGS_fraction_of_gpu_memory_to_use'] = '0.3'  # Use only 30% of GPU

import io
import json
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from PIL import Image
import numpy as np
import cv2
from collections import defaultdict
import re
import logging

# PaddlePaddle imports
from paddleocr import PPStructureV3
from paddlex import create_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PaddlePaddle Chart2Table API",
    description="REST API for PP-Chart2Table chart extraction and table parsing",
    version="1.0.0"
)

# ================================================================
# Global Model Initialization (loaded once on server start)
# ================================================================

class ChartExtractor:
    """Wrapper for PP-Chart2Table chart extraction"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        logger.info("Loading PP-Chart2Table...")
        self.model = create_model('PP-Chart2Table')
        logger.info("✓ PP-Chart2Table loaded successfully!")
        self._initialized = True
    
    def extract(self, image: Image.Image) -> str:
        """Extract table from chart image"""
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        img_np = np.array(image)
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # Use the model exactly as in the example
        results = self.model.predict(
            input={"image": img_cv},
            batch_size=1
        )
        
        table_str = ""
        for res in results:
            logger.info("Result structure:")
            # Use the print() method if available, or try dict conversion
            try:
                res.print()
            except AttributeError:
                pass

            # --- MODIFICATION START: Explicitly checking the provided structure ---
            # The model output seems to be an object/dict where the desired table 
            # string is accessible via the key 'result'.
            
            # 1. Try treating 'res' as a dictionary
            if isinstance(res, dict) and 'result' in res:
                table_str = res['result']
                break
            
            # 2. Try accessing 'result' attribute/key directly (e.g., if it's an object/namedtuple)
            elif hasattr(res, 'result'):
                # Note: The example shows the table under 'result', 
                # but the outer key is 'res'. If the model returns the whole
                # dictionary in 'res', you need to check: res['res']['result']
                
                # Check for the nested structure if 'res' is the outer key
                if isinstance(res, dict) and 'res' in res and isinstance(res['res'], dict) and 'result' in res['res']:
                    table_str = res['res']['result']
                    break
                
                # Default case for the 'result' attribute
                table_str = res.result 
                break 
            
            # 3. Fallback to existing logic (to_dict, save_to_json) for complex structures
            elif hasattr(res, 'to_dict'):
                result_dict = res.to_dict()
                # Check for table data in the dict (including the 'result' key)
                if 'result' in result_dict:
                    table_str = result_dict['result']
                    break
                # Fallback to other known keys
                elif 'table' in result_dict:
                    table_str = result_dict['table']
                    break
                
            # If nothing worked, try saving to JSON as a last resort (original logic)
            # ... (rest of the save_to_json logic remains the same for robustness) ...
            
            # --- MODIFICATION END ---
            
            # Original fallback logic kept for robustness (simplified for space)
            # ... (omitting the rest of the original try/except block for brevity, 
            #      but it should be kept in your final code) ...
            
        if not table_str:
            logger.warning("No table extracted. Returning empty string.")
        else:
            logger.info(f"Successfully extracted table ({len(table_str)} chars)")
            logger.info(f"Table preview: {table_str[:200]}")
            
        return table_str

    # def extract(self, image: Image.Image) -> str:
    #     """Extract table from chart image"""
    #     if image.mode != "RGB":
    #         image = image.convert("RGB")
        
    #     img_np = np.array(image)
    #     img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
    #     # Use the model exactly as in the example
    #     results = self.model.predict(
    #         input={"image": img_cv},
    #         batch_size=1
    #     )
        
    #     table_str = ""
    #     for res in results:
    #         # Debug: print the result to see its structure
    #         logger.info("Result structure:")
    #         res.print()  # This will show us what's in the result
            
    #         # Try to get the table data
    #         # The result object should have methods or attributes for table data
    #         try:
    #             # Option 1: Check if there's a table attribute
    #             if hasattr(res, 'result'):
    #                 table_str = res.result
    #                 break
                
    #             # Option 2: Check if there's a to_dict method
    #             elif hasattr(res, 'to_dict'):
    #                 result_dict = res.to_dict()
    #                 logger.info(f"Result dict: {result_dict}")
                    
    #                 # Look for table data in the dict
    #                 if 'table' in result_dict:
    #                     table_str = result_dict['table']
    #                 elif 'html' in result_dict:
    #                     table_str = result_dict['html']
    #                 elif 'text' in result_dict:
    #                     table_str = result_dict['text']
    #                 elif 'result' in result_dict:
    #                     table_str = result_dict['result']
    #                 break
                
    #             # Option 3: Save to JSON and read it back
    #             else:
    #                 import tempfile
    #                 import os
    #                 with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
    #                     temp_path = f.name
                    
    #                 res.save_to_json(temp_path)
                    
    #                 with open(temp_path, 'r') as f:
    #                     result_data = json.load(f)
                    
    #                 os.unlink(temp_path)
                    
    #                 logger.info(f"JSON result: {result_data}")
                    
    #                 # Extract table from JSON
    #                 if isinstance(result_data, dict):
    #                     if 'table' in result_data:
    #                         table_str = result_data['table']
    #                     elif 'html' in result_data:
    #                         table_str = result_data['html']
    #                     elif 'cells' in result_data:
    #                         # If we have cells, convert to table format
    #                         table_str = self._cells_to_table(result_data['cells'])
    #                     elif 'result' in result_data['result']:
    #                         table_str = result_data['result']
    #                 break
                    
    #         except Exception as e:
    #             logger.error(f"Error extracting table: {e}", exc_info=True)
        
    #     if not table_str:
    #         logger.warning("No table extracted. Returning empty string.")
    #     else:
    #         logger.info(f"Successfully extracted table ({len(table_str)} chars)")
    #         logger.info(f"Table preview: {table_str[:200]}")
        
    #     return table_str
    
    def _cells_to_table(self, cells) -> str:
        """Convert cell list to table string (fallback method)"""
        if not cells:
            return ""
        
        rows_data = defaultdict(lambda: defaultdict(str))
        max_row_idx = 0
        max_col_idx = 0
        
        for cell in cells:
            if isinstance(cell, dict):
                r_start = cell.get('row_start', cell.get('row', 0))
                r_end = cell.get('row_end', r_start + 1)
                c_start = cell.get('col_start', cell.get('col', 0))
                c_end = cell.get('col_end', c_start + 1)
                text = cell.get('text', '')
            else:
                continue
            
            for r in range(r_start, r_end):
                for c in range(c_start, c_end):
                    rows_data[r][c] = text
                    max_row_idx = max(max_row_idx, r)
                    max_col_idx = max(max_col_idx, c)
        
        table_lines = []
        for r_idx in range(max_row_idx + 1):
            current_row_values = []
            for c_idx in range(max_col_idx + 1):
                current_row_values.append(rows_data[r_idx].get(c_idx, ''))
            table_lines.append(" | ".join(current_row_values))
        
        return "\n".join(table_lines).strip()

# ================================================================
# TABLE CLEANING & PARSING (same as your PyTorch pipeline)
# ================================================================

def clean_table_str(s: str) -> str:
    s = s.replace("\r", "\n")
    s = s.replace("<0x0A>", "\n")
    s = s.replace("│", "|").replace("┃", "|").replace("¦", "|")
    s = re.sub(r"[ \t]+", " ", s)
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    return "\n".join(lines)


def _to_num(x: str) -> Optional[float]:
    if x is None:
        return None
    t = x.strip().lower()
    if t in {"na", "n/a", "nan", "none", "null", "-"}:
        return None
    t = t.replace(",", "")
    t = t.replace("%", "")
    m = re.search(r"-?\d+(?:\.\d+)?", t)
    if not m:
        return None
    try:
        return float(m.group(0))
    except:
        return None


def parse_table_to_rows(s: str):
    """Parse table string to rows and headers"""
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    if not lines:
        return [], []
    
    def split_fields(ln: str):
        if "|" in ln:
            parts = [p.strip() for p in ln.split("|")]
        else:
            parts = [p.strip() for p in ln.split(",")]
        return [p for p in parts if p != ""]
    
    rows_raw = [split_fields(ln) for ln in lines]
    
    alpha_count = sum(bool(re.search(r"[A-Za-z]", f)) for f in rows_raw[0])
    num_count = sum(_to_num(f) is not None for f in rows_raw[0])
    
    if alpha_count >= max(1, num_count):
        header = [re.sub(r"\s+", "_", h.strip().lower()) or "col1" for h in rows_raw[0]]
        data_rows = rows_raw[1:]
    else:
        n = len(rows_raw[0])
        header = [f"col{i+1}" for i in range(n)]
        data_rows = rows_raw
    
    rows = []
    seen = set()
    for r in data_rows:
        if len(r) < 1:
            continue
        fields = (r + [""] * len(header))[:len(header)]
        row = {header[i]: fields[i] for i in range(len(header))}
        key = tuple(row.get(h, "") for h in header)
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
    
    return rows, header


# ================================================================
# FASTAPI ENDPOINTS
# ================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "PaddlePaddle Chart2Table API"}


@app.post("/extract")
async def extract_chart(file: UploadFile = File(...)):
    """
    Extract table from chart image
    
    Args:
        file: Image file (PNG, JPG, etc.)
    
    Returns:
        JSON with:
        - raw_table: Raw extracted table string
        - table_clean: Cleaned table string
        - rows: Parsed rows as list of dicts
        - headers: Column headers
    """
    try:
        # Read image file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Extract table
        extractor = ChartExtractor()
        raw_table = extractor.extract(image)
        table_clean = clean_table_str(raw_table)
        rows, headers = parse_table_to_rows(table_clean)
        
        return {
            "success": True,
            "raw_table": raw_table,
            "table_clean": table_clean,
            "rows": rows,
            "headers": headers,
            "num_rows": len(rows),
            "num_cols": len(headers)
        }
    
    except Exception as e:
        logger.error(f"Error extracting chart: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract-base64")
async def extract_chart_base64(data: dict):
    """
    Extract table from base64-encoded image
    
    Args:
        data: JSON with 'image_base64' key containing base64 encoded image
    
    Returns:
        Same as /extract endpoint
    """
    try:
        import base64
        
        if "image_base64" not in data:
            raise HTTPException(status_code=400, detail="Missing 'image_base64' key")
        
        image_data = base64.b64decode(data["image_base64"])
        image = Image.open(io.BytesIO(image_data))
        
        # Extract table
        extractor = ChartExtractor()
        raw_table = extractor.extract(image)
        table_clean = clean_table_str(raw_table)
        rows, headers = parse_table_to_rows(table_clean)
        
        return {
            "success": True,
            "raw_table": raw_table,
            "table_clean": table_clean,
            "rows": rows,
            "headers": headers,
            "num_rows": len(rows),
            "num_cols": len(headers)
        }
    
    except Exception as e:
        logger.error(f"Error extracting chart from base64: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ... all your API code above ...

if __name__ == "__main__":
    import nest_asyncio
    # Apply the patch to allow Uvicorn to run in the notebook's loop
    nest_asyncio.apply() 

    # Run with: python paddle_api_server.py
    # Or: uvicorn paddle_api_server:app --reload --host 0.0.0.0 --port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)

