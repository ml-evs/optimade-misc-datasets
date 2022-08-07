# Load settings before importing app
import os

from utils import SETTINGS

os.environ.update(SETTINGS)

from optimade.server.main import app  # noqa

__all__ = "app"


@app.on_event("startup")
def load_structures():
    from utils import load_structures

    load_structures(insert=True, remove_archive=True)
