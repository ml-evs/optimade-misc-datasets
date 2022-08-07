import json
import logging
import math
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pymatgen.core
import tqdm
from bson.objectid import ObjectId

from optimade.models import StructureResource, StructureResourceAttributes
from optimade.models.utils import anonymous_element_generator

LOG = logging.getLogger("optimade")

SETTINGS = {
    "OPTIMADE_PROVIDER_FIELDS": '{"structures": ["hull_distance", "formation_energy"]}',
    "OPTIMADE_PROVIDER": '{"prefix": "odbx", "name": "Open Database of Xtals","description": "The CAMD dataset, hosted via OPTIMADE https://doi.org/10.6084/m9.figshare.19601956.v1"}',
    "OPTIMADE_DATABASE_BACKEND": "mongomock",
    "OPTIMADE_INSERT_TEST_DATA": "false",
}

os.environ.update(SETTINGS)


def download_from_figshare(
    doi: str, data_dir: Optional[Path] = None
) -> Tuple[List[Path], List[int], int, Path]:
    """Download all files associated with a given Figshare DOI.

    Files will be placed in a folder structure (under `data_dir`) by
    article ID then file ID, with an additional top-level metadata.json
    file containing the full response from the Figshare API.

    Returns:
        A list of file paths, a list of file ids,
        the article id and the article directory.

    """

    import hashlib

    import requests

    chunk_size = 1024**2
    figshare_api_url = "https://api.figshare.com/v2/"

    response = requests.get(f"{figshare_api_url}/articles?doi={doi}")
    if response.status_code != 200:
        raise RuntimeError(f"Bad response: {response.content!r}")
    article_id = response.json()[0]["id"]

    response = requests.get(f"{figshare_api_url}/articles/{article_id}")
    if response.status_code != 200:
        raise RuntimeError(f"Bad response: {response.content!r}")
    response_json = response.json()

    data_dir = data_dir or Path(__file__).parent
    article_dir = Path(data_dir) / f"figshare_{article_id}"
    if not article_dir.exists():
        os.makedirs(article_dir)

    with open(article_dir / "metadata.json", "w") as f:
        json.dump(response_json, f)

    filenames = []
    file_ids = []

    for files in response_json["files"]:
        download_url = files["download_url"]
        local_path = (
            Path(data_dir) / f"figshare_{article_id}" / str(files["id"]) / files["name"]
        )
        if local_path.exists():
            with open(local_path, "rb") as f:
                md5 = hashlib.md5(f.read()).hexdigest()
            if md5 != files["supplied_md5"]:
                LOG.info(
                    f"Downloaded file {local_path} ({md5!r}) does not match MD5 supplied by figshare ({files['supplied_md5']!r}), will move"
                )
                local_path.replace(Path(str(local_path) + ".old"))
            else:
                LOG.info(
                    f"{local_path} already exists locally ({md5!r}), not re-downlaoding..."
                )
                filenames.append(local_path)
                file_ids.append(files["id"])
                continue

        os.makedirs(local_path.parent, exist_ok=True)

        with requests.get(download_url, stream=True) as file_stream:
            LOG.info(
                f"Downloading file {files['name']!r} with size {files['size'] // 1024**2} MB"
            )
            with open(local_path, "wb") as f:
                for chunk in tqdm.tqdm(
                    file_stream.iter_content(chunk_size=chunk_size),
                    total=int(files["size"]) // chunk_size,
                    unit=" MB",
                ):
                    f.write(chunk)

            LOG.info(f"Downloaded file {files['name']!r} to {local_path}")

        filenames.append(local_path)
        file_ids.append(files["id"])

    return (filenames, file_ids, article_id, article_dir)


def pmg_structure_to_optimade_dict(
    s: pymatgen.core.Structure, object_id: Union[str, ObjectId, None] = None
) -> StructureResourceAttributes:
    """Convert the pymatgen Structure into an OPTIMADE StructureResourceAttributes object."""

    attributes = {}
    attributes["cartesian_site_positions"] = s.lattice.get_cartesian_coords(
        s.frac_coords
    ).tolist()
    attributes["fractional_site_positions"] = s.frac_coords.tolist()
    attributes["species_at_sites"] = [_.symbol for _ in s.species]
    attributes["species"] = [
        {"name": _.symbol, "chemical_symbols": [_.symbol], "concentration": [1]}
        for _ in set(s.composition.elements)
    ]
    attributes["dimension_types"] = [1, 1, 1]
    attributes["nperiodic_dimensions"] = 3
    attributes["nelements"] = len(s.composition.elements)
    if object_id:
        attributes["last_modified"] = ObjectId(object_id).generation_time
        attributes["immutable_id"] = str(object_id)
    else:
        attributes["last_modified"] = None
        attributes["immutable_id"] = None
    attributes["chemical_formula_descriptive"] = None
    attributes["chemical_formula_anonymous"] = "".join(
        [
            "".join(x)
            for x in zip(
                anonymous_element_generator(),
                reversed(re.split("[A-Z]", s.composition.anonymized_formula)[1:]),
            )
        ]
    )
    attributes["elements"] = sorted([_.symbol for _ in s.composition.elements])
    gcd = math.gcd(*[int(_) for _ in s.composition.to_reduced_dict.values()])
    attributes["chemical_formula_reduced"] = "".join(
        _
        + f"{int(s.composition.to_reduced_dict[_]) // gcd if s.composition.to_reduced_dict[_] // gcd > 1 else ''}"
        for _ in attributes["elements"]
    )
    attributes["elements_ratios"] = [
        s.composition.get_atomic_fraction(e) for e in attributes["elements"]
    ]
    attributes["nsites"] = len(attributes["species_at_sites"])
    attributes["lattice_vectors"] = s.lattice.matrix.tolist()
    attributes["structure_features"] = []

    return StructureResourceAttributes(**attributes)


def camd_entry_to_optimade_model(entry: Dict[str, Any]) -> StructureResource:
    """Convert an antry in the CAMD dataset to an OPTIMADE StructureResource model.

    Each entry is expected to have a MSON pymatgen structure under the key `structure`,
    a MongoDB ID under the key `_id` and stability, formation energy, space group and
    a CAMD ID under the keys `stability`, `delta_e`, `space_group`, `data_id` respectively.

    """
    s = entry["structure"]
    object_id = entry["_id"]["oid"]
    attributes = pmg_structure_to_optimade_dict(
        pymatgen.core.Structure.from_dict(s), object_id=object_id
    )
    attributes.hull_distance = entry["stability"]
    attributes.formation_energy = entry["delta_e"]
    attributes.space_group = entry["space_group"]
    _id = entry["data_id"]
    return StructureResource(id=_id, attributes=attributes)


def extract_files(files: List[Path], remove_archive: bool = False):
    """Extracts a tar.gz file into its root directory, optionally deleting the archive once complete."""
    import tarfile

    for f in files:
        if str(f).endswith(".tar.gz"):
            LOG.info(f"Extracting file {f} to {f.parent}")
            with tarfile.open(f, "r:gz") as tar:
                tar.extractall(f.parent)
            LOG.info(f"File {f} extracted.")

            if remove_archive:
                LOG.info(f"Removing archive {f}.")
                os.remove(f)


def load_structures(
    insert: bool = False,
    top_n: Optional[int] = 10000,
    remove_archive: bool = True,
) -> List[Dict[str, Any]]:
    """Download, extract and convert the CAMD Figshare dataset, optionally
    inserting it into a mock optimade-python-tools database.

    Returns:
        A list of 'flattened' OPTIMADE structure objects.

    """

    import bson.json_util

    from optimade.server.routers.structures import structures_coll

    # Download dataset if missing
    camd_doi = "https://doi.org/10.6084/m9.figshare.19601956.v1"
    files, file_ids, article_id, article_dir = download_from_figshare(camd_doi)

    structure_file = Path(
        f"{article_dir}/{file_ids[0]}/files/camd_data_to_release_wofeatures.json"
    )
    optimade_structure_file = Path(f"{article_dir}/optimade_structures.bson")
    if not optimade_structure_file.exists():
        if not structure_file.exists():
            extract_files(files, remove_archive=remove_archive)

        with open(structure_file, "r") as f:
            structure_data = json.load(f)

        if remove_archive:
            file_path = Path(f"{article_dir}/{file_ids[0]}")
            LOG.info("Removing extracted archive at %s", file_path)
            shutil.rmtree(file_path)

        optimade_structure_json = []
        if top_n:
            structure_data = structure_data[:top_n]
        for entry in tqdm.tqdm(structure_data):
            structure = camd_entry_to_optimade_model(entry).dict()
            structure.update(structure.pop("attributes"))
            optimade_structure_json.append(structure)

        with open(optimade_structure_file, "w") as f:
            f.write(bson.json_util.dumps(optimade_structure_json))

    else:
        with open(optimade_structure_file, "r") as f:
            optimade_structure_json = bson.json_util.loads(f.read())

    LOG.info(
        "Found %s structures in %s",
        len(optimade_structure_json),
        optimade_structure_file,
    )
    if insert:
        if top_n:
            optimade_structure_json = optimade_structure_json[:top_n]
        structures_coll.insert(optimade_structure_json)
        LOG.info(
            "Inserted %s structures into mock database", len(optimade_structure_json)
        )

    return optimade_structure_json


if __name__ == "__main__":
    load_structures()
