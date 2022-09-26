# Load settings before importing app
import os

SETTINGS: dict[str, str] = {
    "OPTIMADE_PROVIDER_FIELDS": '{"structures": ["hull_distance", "formation_energy", "space_group"]}',
    "OPTIMADE_PROVIDER": '{"prefix": "odbx", "name": "Open Database of Xtals", "description": "Miscellaneous materials discovery datasets, hosted as OPTIMADE APIs"}',
    "OPTIMADE_DATABASE_BACKEND": "mongomock",
    "OPTIMADE_INSERT_TEST_DATA": "false",
}

os.environ.update(SETTINGS)


from optimade.server.main import app  # noqa

__all__ = "app"


@app.on_event("startup")
def load_structures():
    from utils import camd_entry_to_optimade_model, load_structures

    load_structures(
        "10.6084/m9.figshare.19601956.v1",
        "./files/camd_data_to_release_wofeatures.json",
        camd_entry_to_optimade_model,
        insert=True,
        remove_archive=False,
    )
