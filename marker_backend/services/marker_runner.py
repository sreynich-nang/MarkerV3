import subprocess  
from pathlib import Path  
from typing import List, Tuple  
import pypdfium2 as pdfium  
from ..core.config import (  
    MARKER_CLI,  
    MARKER_FLAGS,  
    OUTPUTS_DIR,  
    OUTPUT_FORMAT,  
    GPU_TEMP_THRESHOLD_C,  
    GPU_MEM_FREE_MB,  
    GPU_WAIT_TIMEOUT_SEC,  
    GPU_POLL_INTERVAL_SEC,  
)  
from ..core.logger import get_logger  
from ..core.exceptions import MarkerError  
import shlex  
import time  
import os  
  
logger = get_logger(__name__)  
  
  
def _get_total_pages(pdf_path: Path) -> int:  
    """Get total page count from PDF using pypdfium2"""  
    try:  
        pdf_doc = pdfium.PdfDocument(pdf_path)  
        total_pages = len(pdf_doc)  
        pdf_doc.close()  
        return total_pages  
    except Exception as e:  
        logger.error(f"Failed to get page count for {pdf_path}: {e}")  
        raise MarkerError(f"Could not read PDF: {e}")  
  
  
def _expected_output_for(input_path: Path, chunk_suffix: str = "") -> Path:  
    """Generate expected output path for chunk or combined file"""  
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)  
    if chunk_suffix:  
        return OUTPUTS_DIR / f"{input_path.stem}_chunk_{chunk_suffix}.md"  
    return OUTPUTS_DIR / f"{input_path.stem}.md"  
  
  
def _query_nvidia_smi() -> List[Tuple[int, int, int, int]]:  
    """Return list of tuples (index, temp_c, mem_total_mb, mem_used_mb) for each GPU.  
    If nvidia-smi is not available or fails, return empty list.  
    """  
    try:  
        cmd = [  
            "nvidia-smi",  
            "--query-gpu=index,temperature.gpu,memory.total,memory.used",  
            "--format=csv,noheader,nounits",  
        ]  
        res = subprocess.run(cmd, capture_output=True, text=True)  
        if res.returncode != 0:  
            logger.debug(f"nvidia-smi returned non-zero: {res.stderr}")  
            return []  
  
        lines = [ln.strip() for ln in res.stdout.splitlines() if ln.strip()]  
        out = []  
        for ln in lines:  
            parts = [p.strip() for p in ln.split(",")]  
            if len(parts) >= 4:  
                idx = int(parts[0])  
                temp = int(parts[1])  
                mem_total = int(parts[2])  
                mem_used = int(parts[3])  
                out.append((idx, temp, mem_total, mem_used))  
        return out  
    except FileNotFoundError:  
        logger.debug("nvidia-smi not found; skipping GPU queries")  
        return []  
    except Exception as e:  
        logger.debug(f"Error querying nvidia-smi: {e}")  
        return []  
  
  
def _gpu_state_ok() -> bool:  
    """Return True if all GPUs are below temp threshold and have sufficient free memory.  
    If no GPUs are present or nvidia-smi unavailable, return True (no GPU to wait on).  
    """  
    gpus = _query_nvidia_smi()  
    if not gpus:  
        return True  
  
    for idx, temp, mem_total, mem_used in gpus:  
        mem_free = mem_total - mem_used  
        if temp >= GPU_TEMP_THRESHOLD_C:  
            logger.debug(f"GPU {idx} temp {temp}C >= threshold {GPU_TEMP_THRESHOLD_C}C")  
            return False  
        if mem_free < GPU_MEM_FREE_MB:  
            logger.debug(f"GPU {idx} free mem {mem_free}MB < required {GPU_MEM_FREE_MB}MB")  
            return False  
    return True  
  
  
def wait_for_gpu_ready(timeout: int = GPU_WAIT_TIMEOUT_SEC, poll: int = GPU_POLL_INTERVAL_SEC):  
    """Block until GPU(s) are below thresholds or timeout reached. Raises MarkerError on timeout.  
    If no GPUs detected, returns immediately.  
    """  
    start = time.time()  
    # quick check  
    if _gpu_state_ok():  
        return  
  
    logger.info("Waiting for GPU(s) to cool down and free memory before starting next chunk")  
    while True:  
        if _gpu_state_ok():  
            logger.info("GPU(s) are ready")  
            return  
        if time.time() - start > timeout:  
            msg = f"Timeout waiting for GPU to become available after {timeout}s"  
            logger.error(msg)  
            raise MarkerError(msg)  
        time.sleep(poll)  
  
  
def run_marker_for_chunk_with_range(pdf_path: Path, page_range: str, chunk_id: int) -> Path:  
    """Run marker_single for a specific page range chunk"""  
    out_path = _expected_output_for(pdf_path, str(chunk_id))  
  
    # If CUDA_VISIBLE_DEVICES is set in env, respect it; otherwise use system default  
    env = os.environ.copy()  
  
    # Wait for GPU to be in a safe state before launching heavy processing  
    try:  
        wait_for_gpu_ready()  
    except MarkerError:  
        # re-raise to stop processing  
        raise  
  
    # Build command with page range  
    cmd = [MARKER_CLI, str(pdf_path), "--page_range", page_range] + MARKER_FLAGS  
  
    logger.info(f"Processing chunk {chunk_id} (pages {page_range}) for {pdf_path}")  
    logger.info(f"Command: {' '.join(shlex.quote(p) for p in cmd)}")  
      
    start = time.time()  
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)  
    duration = time.time() - start  
  
    # Log summary info at INFO and full outputs at DEBUG so app.log captures details  
    logger.info(  
        "Marker finished for chunk %s of %s (exit=%s) in %.2fs",  
        chunk_id,  
        pdf_path.name,  
        res.returncode,  
        duration,  
    )  
    logger.debug("Marker stdout for chunk %s:\n%s", chunk_id, res.stdout or "<no stdout>")  
    logger.debug("Marker stderr for chunk %s:\n%s", chunk_id, res.stderr or "<no stderr>")  
  
    if res.returncode != 0:  
        logger.error("Marker failed for chunk %s of %s (exit=%s). See stderr in logs.",   
                    chunk_id, pdf_path.name, res.returncode)  
        raise MarkerError(f"Marker failed for chunk {chunk_id}: {res.stderr}")  
  
    # Discover the output file for this chunk  
    chunk_output = _discover_marker_output(pdf_path, out_path, res.stdout + "\n" + res.stderr)  
    return chunk_output  
  
  
def _discover_marker_output(pdf_path: Path, expected_path: Path, command_output: str) -> Path:  
    """Discover the actual output file from marker run"""  
    # First, check the expected path  
    if expected_path.exists():  
        return expected_path  
  
    logger.debug("Expected output not found at {expected_path}; attempting discovery heuristics.")  
      
    candidates = []  
    stem_pattern = f"{pdf_path.stem}*"  
  
    # 1) look in configured MARKER_OUTPUT_DIR  
    from ..core.config import MARKER_OUTPUT_DIR  
    try:  
        candidates.extend(list(MARKER_OUTPUT_DIR.glob(stem_pattern)))  
    except Exception:  
        logger.debug(f"Could not access MARKER_OUTPUT_DIR: {MARKER_OUTPUT_DIR}")  
  
    # 2) look in the input file's parent directory  
    try:  
        candidates.extend(list(pdf_path.parent.glob(stem_pattern)))  
    except Exception:  
        pass  
  
    # 3) look in current working directory  
    try:  
        candidates.extend(list(Path.cwd().glob(stem_pattern)))  
    except Exception:  
        pass  
  
    # 4) parse stdout/stderr for any .md path  
    import re  
    md_paths = re.findall(r"[A-Za-z0-9_:\\/.\- ]+(?:\.md)?", command_output)  
    for p in md_paths:  
        p = p.strip()  
        try:  
            pth = Path(p)  
            if pth.exists() and pth.is_file():  
                candidates.append(pth)  
        except Exception:  
            continue  
  
    # Deduplicate and sort by modification time (newest first)  
    unique = {}  
    for c in candidates:  
        try:  
            unique[str(c.resolve())] = c  
        except Exception:  
            unique[str(c)] = c  
  
    candidates = list(unique.values())  
    candidates.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)  
  
    if candidates:  
        chosen = candidates[0]  
        logger.info(f"Discovered Marker output at {chosen}")  
        return chosen  
  
    # Nothing found  
    logger.error("Marker finished but no markdown output discovered")  
    raise MarkerError(f"Expected output {expected_path} not found after Marker run")  
  
  
def combine_chunk_outputs(chunk_files: List[Path], pdf_path: Path) -> Path:  
    """Combine multiple chunk outputs into a single markdown file"""  
    combined_path = _expected_output_for(pdf_path, "combined")  
      
    logger.info(f"Combining {len(chunk_files)} chunks into {combined_path}")  
      
    try:  
        with open(combined_path, 'w', encoding='utf-8') as outfile:  
            for i, chunk_file in enumerate(chunk_files):  
                if not chunk_file.exists():  
                    logger.warning(f"Chunk file {chunk_file} does not exist, skipping")  
                    continue  
                  
                logger.debug(f"Adding chunk {i+1}: {chunk_file}")  
                  
                with open(chunk_file, 'r', encoding='utf-8') as infile:  
                    content = infile.read()  
                      
                    # Add chunk separator if not the first chunk  
                    if i > 0:  
                        outfile.write("\n\n--- Chunk " + str(i+1) + " ---\n\n")  
                      
                    outfile.write(content)  
          
        logger.info(f"Successfully combined chunks into {combined_path}")  
        return combined_path  
          
    except Exception as e:  
        logger.error(f"Failed to combine chunks: {e}")  
        raise MarkerError(f"Failed to combine chunk outputs: {e}")  
  
  
def run_marker_for_chunked_pdf(pdf_path: Path, chunk_size: int = 5) -> Path:  
    """Process a large PDF in chunks and combine the results"""  
    # Get total page count  
    total_pages = _get_total_pages(pdf_path)  
    logger.info(f"Processing {pdf_path.name} with {total_pages} pages in chunks of {chunk_size}")  
      
    chunk_files = []  
      
    # Process each chunk  
    for chunk_id, start_page in enumerate(range(0, total_pages, chunk_size)):  
        end_page = min(start_page + chunk_size - 1, total_pages - 1)  
        page_range = f"{start_page}-{end_page}"  
          
        logger.info(f"Processing chunk {chunk_id + 1}: pages {page_range} ({start_page + 1}-{end_page + 1} of {total_pages})")  
          
        try:  
            chunk_output = run_marker_for_chunk_with_range(pdf_path, page_range, chunk_id)  
            chunk_files.append(chunk_output)  
        except MarkerError as e:  
            logger.error(f"Failed to process chunk {chunk_id + 1}: {e}")  
            # Continue with other chunks or raise based on your requirements  
            continue  
      
    if not chunk_files:  
        raise MarkerError("No chunks were successfully processed")  
      
    # Combine all chunk outputs  
    combined_output = combine_chunk_outputs(chunk_files, pdf_path)  
      
    # Clean up individual chunk files (optional)  
    try:  
        for chunk_file in chunk_files:  
            if chunk_file.exists():  
                chunk_file.unlink()  
                logger.debug(f"Cleaned up chunk file: {chunk_file}")  
    except Exception as e:  
        logger.warning(f"Failed to cleanup some chunk files: {e}")  
      
    return combined_output  
  
  
# Keep the original function for backward compatibility  
def run_marker_for_chunk(chunk_path: Path) -> Path:  
    """Legacy function - processes entire file without chunking"""  
    out_path = _expected_output_for(chunk_path)  
  
    # If CUDA_VISIBLE_DEVICES is set in env, respect it; otherwise use system default  
    env = os.environ.copy()  
  
    # Wait for GPU to be in a safe state before launching heavy processing  
    try:  
        wait_for_gpu_ready()  
    except MarkerError:  
        # re-raise to stop processing  
        raise  
  
    cmd = [MARKER_CLI, str(chunk_path)] + MARKER_FLAGS  
  
    logger.info(f"Starting Marker for {chunk_path} with cmd: {' '.join(shlex.quote(p) for p in cmd)}")  
    start = time.time()  
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)  
    duration = time.time() - start  
  
    # Log summary info at INFO and full outputs at DEBUG so app.log captures details  
    logger.info(  
        "Marker finished for %s (exit=%s) in %.2fs",  
        chunk_path,  
        res.returncode,  
        duration,  
    )  
    logger.debug("Marker stdout for %s:\n%s", chunk_path, res.stdout or "<no stdout>")  
    logger.debug("Marker stderr for %s:\n%s", chunk_path, res.stderr or "<no stderr>")  
  
    if res.returncode != 0:  
        logger.error("Marker failed for %s (exit=%s). See stderr in logs.", chunk_path, res.returncode)  
        # ensure stderr is available in the exception message for immediate feedback  
        raise MarkerError(f"Marker failed for {chunk_path}: {res.stderr}")  
      
    # If marker outputs to stdout or writes file elsewhere, try to discover the produced markdown.  
    # First, check the canonical out_path  
    if out_path.exists():  
        return out_path  
  
    logger.debug("Expected output not found at canonical path; attempting discovery heuristics.")  
    # 1) look in configured MARKER_OUTPUT_DIR  
    from ..core.config import MARKER_OUTPUT_DIR  
  
    candidates = []  
    stem_pattern = f"{chunk_path.stem}*"  
  
    try:  
        candidates.extend(list(MARKER_OUTPUT_DIR.glob(stem_pattern)))  
    except Exception:  
        logger.debug(f"Could not access MARKER_OUTPUT_DIR: {MARKER_OUTPUT_DIR}")  
  
    # 2) look in the input file's parent (where marker may have placed outputs)  
    try:  
        candidates.extend(list(chunk_path.parent.glob(stem_pattern)))  
    except Exception:  
        pass  
  
    # 3) look in current working directory  
    try:  
        candidates.extend(list(Path.cwd().glob(stem_pattern)))  
    except Exception:  
        pass  
  
    # 4) parse stdout/stderr for any .md path  
    text = (res.stdout or "") + "\n" + (res.stderr or "")  
    import re  
    md_paths = re.findall(r"[A-Za-z0-9_:\\/.\- ]+(?:\.md)?", text)  
    for p in md_paths:  
        p = p.strip()  
        try:  
            pth = Path(p)  
            if pth.exists() and pth.is_file():  
                candidates.append(pth)  
        except Exception:  
            continue  
  
    # Deduplicate and sort by modification time (newest first)  
    unique = {}  
    for c in candidates:  
        try:  
            unique[str(c.resolve())] = c  
        except Exception:  
            unique[str(c)] = c  
  
    candidates = list(unique.values())  
    candidates.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)  
  
    if candidates:  
        chosen = candidates[0]  
        logger.info(f"Discovered Marker output at {chosen}")  
        # Ensure output dir exists and move/copy file to OUTPUTS_DIR if not already there  
        try:  
            OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)  
            dest = OUTPUTS_DIR / chosen.name  
            if chosen.resolve() != dest.resolve():  
                # Move the file so future runs are predictable  
                try:  
                    chosen.replace(dest)  
                    logger.info(f"Moved output {chosen} -> {dest}")  
                except Exception:  
                    # fallback to copy if replace fails  
                    import shutil  
                    shutil.copy2(chosen, dest)  
                    logger.info(f"Copied output {chosen} -> {dest}")  
            return dest  
        except Exception as e:  
            logger.error(f"Failed to relocate discovered output: {e}")  
            return chosen  
  
    # Nothing found  
    logger.error("Marker finished but no markdown output discovered; stdout/stderr below:\n%s", text)  
    raise MarkerError(f"Expected output {out_path} not found after Marker run")