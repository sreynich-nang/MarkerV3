from fastapi import APIRouter, UploadFile, File, HTTPException    
from fastapi.responses import FileResponse    
from ..core.logger import get_logger    
from ..core.config import ensure_dirs, UPLOADS_DIR, OUTPUTS_DIR, FILTERS_DIR  
from ..services.file_handler import save_upload    
from ..services.marker_runner import process_document    
from ..models.schemas import UploadResponse, TableExtractionResponse    
from ..core.exceptions import InvalidFileError, MarkerError  # Removed ChunkingError  
import time    
    
router = APIRouter()    
logger = get_logger(__name__)    
    
    
@router.post("/upload", response_model=UploadResponse)    
async def upload_pdf(file: UploadFile = File(...)):    
    """Upload a PDF or image and process it with marker (PDFs with chunking, images with OCR)."""    
    ensure_dirs()    
    start = time.time()    
    try:    
        saved_path = await save_upload(file)    
        logger.info(f"Saved upload to {saved_path}")    
    
        output = process_document(saved_path, chunk_size=5)    
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
    """Download markdown file from processed documents.
    
    Handles two output structures:
    - PDF uploads: outputs/document_name.md
    - Image uploads: outputs/document_name/document_name.md
    """
    # Try PDF structure first (direct file in outputs/)
    path = OUTPUTS_DIR / f"{filename}.md"
    if path.exists():
        logger.info(f"Found markdown at PDF path: {path}")
        return FileResponse(path, filename=path.name, media_type="text/markdown")
    
    # Try image structure (nested in document folder)
    path = OUTPUTS_DIR / filename / f"{filename}.md"
    if path.exists():
        logger.info(f"Found markdown at image path: {path}")
        return FileResponse(path, filename=path.name, media_type="text/markdown")
    
    # Neither path exists
    logger.warning(f"Markdown file not found for document: {filename}")
    raise HTTPException(status_code=404, detail="Markdown file not found")


@router.post("/filter_tables", response_model=TableExtractionResponse)
async def filter_tables(document: str, sheets_per_file: int = 30, store_in_filters: bool = False):
    """Extract tables from a processed document's markdown and save Excel batches.

    Expects marker output folder structure: outputs/<document>/<document>.md
    Returns metadata including created Excel files.
    """
    ensure_dirs()
    start = time.time()
    try:
        from ..services.table_extractor import extract_and_save_tables
        excel_base_dir = FILTERS_DIR if store_in_filters else None
        md_path, dfs, excel_files, excel_folder = extract_and_save_tables(
            document,
            OUTPUTS_DIR,
            sheets_per_file=sheets_per_file,
            excel_base_dir=excel_base_dir,
        )
        logger.info(f"Extracted {len(dfs)} tables for document '{document}' into {excel_folder}")
        return TableExtractionResponse(
            status="success",
            document=document,
            markdown_path=str(md_path),
            tables_count=len(dfs),
            excel_folder=str(excel_folder),
            excel_files=[str(p) for p in excel_files],
        )
    except FileNotFoundError as e:
        logger.exception("Markdown file not found for table extraction")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:  # noqa: BLE001
        logger.exception("Unexpected error during table extraction")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/download_table")
def download_table(document: str, filename: str, store_in_filters: bool = False):
    """Download a generated Excel table batch for a document.

    Parameters:
    - document: base document name (folder and markdown source name)
    - filename: Excel file name (e.g. `tables_1.xlsx`)
    - store_in_filters: if true, look under `filters/<document>/`, else under
      `outputs/<document>/tables_xlsx_<document>/`.

    Returns the Excel file or 404 if not found. Rejects path traversal attempts.
    """
    ensure_dirs()
    # Basic traversal protection
    if any(ch in filename for ch in ("..", "/", "\\")):
        raise HTTPException(status_code=400, detail="Invalid filename")

    if store_in_filters:
        base_dir = FILTERS_DIR / document
    else:
        base_dir = OUTPUTS_DIR / document / f"tables_xlsx_{document}"

    path = base_dir / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Excel file not found")

    logger.info(f"Downloading Excel table file {path}")
    return FileResponse(
        path,
        filename=path.name,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
