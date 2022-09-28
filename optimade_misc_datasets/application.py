# Load settings before importing app
import logging
import os

SETTINGS: dict[str, str] = {
    "OPTIMADE_PROVIDER_FIELDS": '{"structures": ["hull_distance", "formation_energy", "space_group"]}',
    "OPTIMADE_PROVIDER": '{"prefix": "odbx", "name": "Open Database of Xtals", "description": "Miscellaneous materials discovery datasets, hosted as OPTIMADE APIs"}',
    "OPTIMADE_DATABASE_BACKEND": "mongodb",
    "OPTIMADE_MONGO_URI": "mongodb://localhost:27017/optimade",
    "OPTIMADE_MONGO_DATABASE": "optimade",
    "OPTIMADE_INSERT_TEST_DATA": "false",
}

os.environ.update(SETTINGS)


from optimade.server.main import app  # noqa

__all__ = "app"


@app.on_event("startup")
async def load_structures():
    from utils import (
        camd_entry_to_optimade_model,
        load_structures,
        wren_entry_to_optimade_model,
    )

    logging.info("Initialised ingestion event")

    # Law et al dataset
    load_structures(
        "10.5281/zenodo.7089031",
        "./upper-bound-energy-gnn-0.1/paper_results/relaxed_structures.tar.gz",
        metadata_csv="./upper-bound-energy-gnn-0.1/paper_results/dft_confirmation.csv",
        insert=True,
        remove_archive=False,
        id_prefix="law-gnn",
    )

    # CAMD dataset
    load_structures(
        "10.6084/m9.figshare.19601956.v1",
        "./34818031/files/camd_data_to_release_wofeatures.json",
        camd_entry_to_optimade_model,
        insert=True,
        remove_archive=False,
    )

    # Wren dataset
    load_structures(
        "10.5281/zenodo.6345276",
        "./prospective.json.gz",
        wren_entry_to_optimade_model,
        insert=True,
        remove_archive=False,
    )
