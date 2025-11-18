from fastapi import APIRouter, UploadFile, File, HTTPException    
from fastapi.responses import FileResponse    
from ..core.logger import get_logger    
from ..core.config import ensure_dirs, UPLOADS_DIR, OUTPUTS_DIR  
from ..services.file_handler import save_upload    
from ..services.marker_runner import run_marker_for_chunk    
from ..models.schemas import UploadResponse    
from ..core.exceptions import InvalidFileError, MarkerError  # Removed ChunkingError  
import time    
    
router = APIRouter()    
logger = get_logger(__name__)    
    
    
@router.post("/upload", response_model=UploadResponse)    
async def upload_pdf(file: UploadFile = File(...)):    
    """Upload a PDF and process it with marker."""    
    ensure_dirs()    
    start = time.time()    
    try:    
        saved_path = await save_upload(file)    
        logger.info(f"Saved upload to {saved_path}")    
    
        output = run_marker_for_chunk(saved_path)    
        logger.info(f"Marker produced output file: {output}")    
    
        elapsed = time.time() - start    
    
        return UploadResponse(    
            status="success",    
            filename=saved_path.name,    
            merged_path=str(output),  
            processing_time_seconds=round(elapsed, 2),    
        )    
    
    except InvalidFileError as e:    
        logger.exception("Invalid file error")    
        raise HTTPException(status_code=400, detail=str(e))    
    except MarkerError as e:    
        logger.exception("Marker processing error")    
        raise HTTPException(status_code=500, detail=str(e))    
    except Exception as e:    
        logger.exception("Unexpected error processing upload")    
        raise HTTPException(status_code=500, detail=str(e))    
    
    
@router.get("/download/{filename:path}")    
def download(filename: str):    
    path = OUTPUTS_DIR / filename  
    if not path.exists():    
        raise HTTPException(status_code=404, detail="File not found")    
    return FileResponse(path, filename=path.name, media_type="text/markdown")
