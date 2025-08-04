import os
import shutil
from typing import List, Dict, Any
from datetime import datetime


def create_directory_structure():
    """Create the required directory structure"""
    directories = [
        "data/documents/pdf",
        "data/documents/json",
        "data/documents/txt",
        "data/documents/docx",
        "data/documents/csv",
        "data/vectorstore",
        "config",
        "src",
        "utils",
        "tests"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("Directory structure created successfully")


def log_message(message: str, level: str = "INFO"):
    """Simple logging function"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {level}: {message}")


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"


def get_file_info(file_path: str) -> Dict[str, Any]:
    """Get information about a file"""
    if not os.path.exists(file_path):
        return None
    
    stat = os.stat(file_path)
    return {
        "name": os.path.basename(file_path),
        "path": file_path,
        "size": stat.st_size,
        "size_formatted": format_file_size(stat.st_size),
        "modified": datetime.fromtimestamp(stat.st_mtime),
        "extension": os.path.splitext(file_path)[1].lower()
    }


def scan_documents_directory(directory: str) -> List[Dict[str, Any]]:
    """Scan directory for supported document files"""
    supported_extensions = {'.pdf', '.txt', '.json', '.docx', '.csv'}
    files_info = []
    
    if not os.path.exists(directory):
        return files_info
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            ext = os.path.splitext(file)[1].lower()
            
            if ext in supported_extensions:
                info = get_file_info(file_path)
                if info:
                    files_info.append(info)
    
    return files_info


def backup_vector_store(source_dir: str, backup_dir: str):
    """Create backup of vector store"""
    if os.path.exists(source_dir):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{backup_dir}_backup_{timestamp}"
        shutil.copytree(source_dir, backup_path)
        log_message(f"Vector store backed up to {backup_path}")
        return backup_path
    else:
        log_message("No vector store found to backup", "WARNING")
        return None


def validate_openai_api_key(api_key: str) -> bool:
    """Validate OpenAI API key format"""
    if not api_key:
        return False
    
    # Basic format validation
    if not api_key.startswith("sk-"):
        return False
    
    if len(api_key) < 20:
        return False
    
    return True


def count_tokens_approximate(text: str) -> int:
    """Approximate token count for text"""
    # Rough approximation: 1 token ≈ 4 characters
    return len(text) // 4


def truncate_text(text: str, max_length: int = 1000) -> str:
    """Truncate text to maximum length"""
    if len(text) <= max_length:
        return text
    
    return text[:max_length] + "..."


def clean_filename(filename: str) -> str:
    """Clean filename for safe usage"""
    import re
    # Remove or replace invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove excessive dots
    filename = re.sub(r'\.+', '.', filename)
    return filename.strip()


def get_system_info() -> Dict[str, Any]:
    """Get system information"""
    import platform
    import psutil
    
    return {
        "platform": platform.system(),
        "platform_version": platform.version(),
        "python_version": platform.python_version(),
        "cpu_count": psutil.cpu_count(),
        "memory_total": format_file_size(psutil.virtual_memory().total),
        "memory_available": format_file_size(psutil.virtual_memory().available)
    }