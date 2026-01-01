import pytest
import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture(scope="session")
def project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def telemetry_csv(project_root):
    """Get the telemetry CSV path."""
    return project_root / "drone_telemetry.csv"


@pytest.fixture(scope="session")
def frames_dir(project_root):
    """Get the frames directory path."""
    return project_root / "frames"
