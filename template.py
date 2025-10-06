import os
from pathlib import Path

project_name = "rag-groq-chatbot"

list_of_files = [
    f"{project_name}/utils/__init__.py",
    f"{project_name}/utils/custom_exceptions.py",
    f"{project_name}/utils/logger.py",
    f"{project_name}/docker-compose.yml",
    f"{project_name}/Dockerfile",
    f"{project_name}/app/__init__.py",    
    f"{project_name}/app/main.py",        
    f"{project_name}/app/ingest.py",      
    f"{project_name}/app/retriever.py",   
    f"{project_name}/app/llm_client.py",  
    f"{project_name}/app/schemas.py",     
    f"{project_name}/chainlit_app.py",
    "requirements.txt",
    "setup.py",
    ".env",
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    # Create directory if it doesn't exist
    if filedir:
        os.makedirs(filedir, exist_ok=True)
        print(f"Created directory: {filedir}")

    # Create empty file if it doesn't exist or is empty
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
        print(f"Created file: {filepath}")
    else:
        print(f"File already exists: {filepath}")

print(f"Project structure for '{project_name}' created successfully!")