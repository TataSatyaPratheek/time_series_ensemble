"""
Configuration Viewer Page
Displays the content of key configuration files for transparency.
"""
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Configuration Viewer", page_icon="⚙️", layout="wide")

st.title("⚙️ System Configuration Viewer")
st.info("This is a read-only view of the project's key configuration files.")

# --- Helper Function ---
def read_file_content(file_path: Path) -> str:
    """Reads the content of a file, returns an error message if not found."""
    try:
        return file_path.read_text()
    except FileNotFoundError:
        return f"File not found at: {file_path}"
    except Exception as e:
        return f"Error reading file: {e}"

# --- File Paths ---
# Assumes Streamlit is run from the project root.
# Adjust paths if you run it from within the `frontend` directory.
PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
ENV_FILE = PROJECT_ROOT / ".env"

# --- Display Configurations ---
tab1, tab2, tab3, tab4 = st.tabs(["Environment (.env)", "Agents (agents.yaml)", "Models (models.yaml)", "Tasks (tasks.yaml)"])

with tab1:
    st.subheader("Environment Variables (`.env`)")
    env_content = read_file_content(ENV_FILE)
    st.code(env_content, language='bash')
    
with tab2:
    st.subheader("Agent Configuration (`agents.yaml`)")
    agents_content = read_file_content(CONFIG_DIR / "agents.yaml")
    st.code(agents_content, language='yaml')

with tab3:
    st.subheader("Model Configuration (`models.yaml`)")
    models_content = read_file_content(CONFIG_DIR / "models.yaml")
    st.code(models_content, language='yaml')

with tab4:
    st.subheader("Task Configuration (`tasks.yaml`)")
    tasks_content = read_file_content(CONFIG_DIR / "tasks.yaml")
    st.code(tasks_content, language='yaml')
