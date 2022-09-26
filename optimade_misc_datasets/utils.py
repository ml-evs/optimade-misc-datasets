import json
import logging
import os
from pathlib import Path
from typing import Any, Callable

import pymatgen.core
import tqdm

from optimade.adapters.structures.pymatgen import from_pymatgen
from optimade.models import StructureResource

LOG = logging.getLogger("optimade")

REQUEST_TIMEOUT = 9.3


def download_from_doi(
    doi: str, data_dir: Path = None
) -> tuple[list[Path], list[int], int, Path]:
    """Download all files associated with a given DOI from figshare or Zenodo.

    Files will be placed in a folder structure (under `data_dir`) by
    article ID then file ID, with an additional top-level metadata.json
    file containing the full response from the corresponding API.

    Returns:
        A list of file paths, a list of file ids,
        the article id and the article directory.

    """

    if "figshare" in doi.lower():
        return download_from_figshare(doi, data_dir=data_dir)
    elif "zenodo" in doi.lower():
        return download_from_zenodo(doi, data_dir=data_dir)
    else:
        raise RuntimeError(f"Could not recognize {doi=} as a Zenodo or Figshare DOI.")


def download_from_zenodo(
    doi: str, data_dir: Path = None
) -> tuple[list[Path], list[int], int, Path]:
    """Download all files associated with a given Zenodo DOI.

    Files will be placed in a folder structure (under `data_dir`) by
    article ID then file ID, with an additional top-level metadata.json
    file containing the full response from the Zenodo API.

    Returns:
        A list of file paths, a list of file ids,
        the article id and the article directory.

    """

    import requests

    if "zenodo" not in doi:
        raise RuntimeError(f"{doi=} does not look like a Zenodo DOI, exiting...")
    if doi.startswith("http"):
        doi_url = doi
    else:
        doi_url = f"https://doi.org/{doi}"

    r = requests.get(doi_url, timeout=REQUEST_TIMEOUT)
    redirected_url = r.url.strip()
    record_id = int(redirected_url.split("/")[-1])

    record_data = requests.get(
        f"https://zenodo.org/api/records/{record_id}", timeout=REQUEST_TIMEOUT
    ).json()

    data_dir = data_dir or Path(__file__).parent
    record_dir = Path(data_dir) / f"zenodo_{record_id}"
    if not record_dir.exists():
        os.makedirs(record_dir)

    with open(record_dir / "metadata.json", "w") as f:
        json.dump(record_data, f)

    filenames = []
    file_ids = []

    for files in record_data["files"]:
        download_url = files["links"]["self"]
        local_path = Path(data_dir) / f"zenodo_{record_id}" / files["key"]

        if local_path.exists():
            if check_existing_hash_with_remote(local_path, files["checksum"]):
                filenames.append(local_path)
                file_ids.append(files["key"])
                continue

        os.makedirs(local_path.parent, exist_ok=True)
        download_file(download_url, local_path, files["size"], files["key"])
        filenames.append(local_path)
        file_ids.append(files["key"])

    return (filenames, file_ids, record_id, record_dir)


def download_from_figshare(
    doi: str, data_dir: Path = None
) -> tuple[list[Path], list[int], int, Path]:
    """Download all files associated with a given Figshare DOI.

    Files will be placed in a folder structure (under `data_dir`) by
    article ID then file ID, with an additional top-level metadata.json
    file containing the full response from the Figshare API.

    Returns:
        A list of file paths, a list of file ids,
        the article id and the article directory.

    """

    import requests

    figshare_api_url = "https://api.figshare.com/v2/"

    response = requests.get(
        f"{figshare_api_url}/articles?doi={doi}", timeout=REQUEST_TIMEOUT
    )
    if response.status_code != 200:
        raise RuntimeError(f"Bad response: {response.content!r}")
    article_id = response.json()[0]["id"]

    response = requests.get(
        f"{figshare_api_url}/articles/{article_id}", timeout=REQUEST_TIMEOUT
    )
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
            if check_existing_hash_with_remote(local_path, files["supplied_md5"]):
                filenames.append(local_path)
                file_ids.append(files["id"])
                continue

        os.makedirs(local_path.parent, exist_ok=True)
        download_file(download_url, local_path, files["size"], files["name"])
        filenames.append(local_path)
        file_ids.append(files["id"])

    return (filenames, file_ids, article_id, article_dir)


def download_file(
    url: str, local_path: Path, size: int, name: str, chunk_size: int = 1024**2
) -> None:
    """Stream a file from a URL to a local path with a given chunk size.

    Arguments:
        url: The URL to download from.
        local_path: The local path to save to.
        size: The size of the file in bytes.
        name: The name of the file.
        chunk_size: The chunk size to use when streaming.

    """

    import requests

    with requests.get(url, stream=True, timeout=REQUEST_TIMEOUT) as file_stream:
        LOG.info(f"Downloading file {name!r} with size {int(size) // 1024**2} MB")
        with open(local_path, "wb") as f:
            for chunk in tqdm.tqdm(
                file_stream.iter_content(chunk_size=chunk_size),
                total=int(size) // chunk_size,
                unit=" MB",
            ):
                f.write(chunk)

        LOG.info(f"Downloaded file {name!r} to {local_path}")


def check_existing_hash_with_remote(local_path: Path, remote_hash: str) -> bool:
    """Check the hash of a local file against a remote hash; if they do not match,
    back up the local copy and return False.

    Arguments:
        local_path: The local path to check.
        remote_hash: The remote md5 hash to check against.

    Returns:
        True if the hashes match, False otherwise.

    """
    import hashlib

    with open(local_path, "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()
    if md5 != remote_hash.lstrip("md5:"):
        LOG.info(
            f"Downloaded file {local_path} ({md5!r}) does not match MD5 supplied by figshare ({remote_hash!r}), will move"
        )
        local_path.replace(Path(str(local_path) + ".old"))
        return False
    else:
        LOG.info(
            f"{local_path} already exists locally ('md5:{md5}'), not re-downlaoding..."
        )
        return True


def camd_entry_to_optimade_model(entry: dict[str, Any]) -> StructureResource:
    """Convert an entry in the CAMD dataset to an OPTIMADE StructureResource model.

    Each entry is expected to have a MSON pymatgen structure under the key `structure`,
    a MongoDB ID under the key `_id` and stability, formation energy, space group and
    a CAMD ID under the keys `stability`, `delta_e`, `space_group`, `data_id` respectively.

    Arguments:
        entry: The entry to convert.

    Returns:
        The converted OPTIMADE StructureResource.

    """
    from bson import ObjectId

    s = entry["structure"]
    object_id = entry["_id"]["oid"]
    attributes = from_pymatgen(
        pymatgen.core.Structure.from_dict(s),
    )
    attributes.immutable_id = object_id
    attributes.last_modified = ObjectId(object_id).generation_time
    attributes.hull_distance = entry["stability"]
    attributes.formation_energy = entry["delta_e"]
    attributes.space_group = entry["space_group"]
    _id = entry["data_id"]
    return StructureResource(id=_id, attributes=attributes)


def wren_entry_to_optimade_model(entry: tuple) -> StructureResource:
    """Convert a row from the Wren dataframe into an OPTIMADE structure resource.

    Arguments:
        entry: A row from the Wren dataframe, as a tuple.

    Returns:
        A StructureResource model.

    """
    (
        material_id,
        source_id,
        cse,
        wyckoff_final,
        wyckoff_initial,
        initial_structure,
        E_vasp_corrected,
        E_ref_terminal,
        E_f_dft,
        E_f_pred,
        E_f_std,
        E_hull_mp_wbm_dft,
        E_hull_mp_wbm_pred,
        E_hull_mp_dft,
        E_hull_mp_pred,
    ) = entry

    attributes = from_pymatgen(
        pymatgen.core.Structure.from_dict(cse["structure"]),
    )

    _id = material_id
    attributes.formation_energy = E_f_dft
    attributes.hull_distance = E_hull_mp_dft
    return StructureResource(id=_id, attributes=attributes)


def extract_files(files: list[Path], remove_archive: bool = False) -> None:
    """Extracts a tar.gz file into its root directory, optionally deleting the
    archive once complete.

    Arguments:
        files: A list of files to extract.
        remove_archive: Whether to remove the archive once complete.

    """
    import tarfile

    for f in files:
        if sorted(f.suffixes) == [".gz", ".tar"]:
            LOG.info(f"Extracting file {f} to {f.parent}")
            with tarfile.open(f, "r:gz") as tar:
                tar.extractall(f.parent)
            LOG.info(f"File {f} extracted.")

            if remove_archive:
                LOG.info(f"Removing archive {f}.")
                os.remove(f)


def load_structures(
    doi: str,
    target_file: str,
    adapter_function: Callable,
    insert: bool = False,
    top_n: int = None,
    remove_archive: bool = True,
) -> list[dict[str, Any]]:
    """Download, extract and convert a Zenodo or Figshare dataset, optionally
    inserting it into a mock optimade-python-tools database.

    The function expects to find a JSON or gzipped-JSON file at the `target_file`
    path, which itself should contain either a list of objects to pass through the
    `adapter_function`, or a dictionary with a key `data` that contains such a list.

    Arguments:
        doi: The DOI of the dataset to download and ingest.
        target_file: The path to the file in the dataset to extract and convert.
        adapter_function: The function to use to convert the dataset into OPTIMADE models.
        insert: Whether to insert the converted models into a mock database.
        top_n: The number of entries to convert and insert.
        remove_archive: Whether to remove the archive once complete.

    Returns:
        A list of 'flattened' OPTIMADE structure entries.

    """

    import gzip

    import bson.json_util

    from optimade.server.routers.structures import structures_coll

    # Download dataset if missing
    files, file_ids, article_id, article_dir = download_from_doi(doi)

    structure_file = Path(f"{article_dir}/{file_ids[0]}") / target_file

    optimade_structure_file = Path(article_dir) / "optimade_structures.bson"

    if not optimade_structure_file.exists():
        for file_id in file_ids:
            if not structure_file.exists():
                extract_files(files, remove_archive=remove_archive)

            if not structure_file.exists():
                continue

            mode = "r"
            if structure_file.suffix == ".gz":
                mode = "rb"

            with open(structure_file, mode) as f:
                if mode == "rb":
                    structure_data = json.loads(
                        gzip.decompress(f.read()).decode("utf-8")
                    )
                elif mode == "r":
                    structure_data = json.load(f)

            if isinstance(structure_data, dict):
                structure_data = structure_data["data"]

            optimade_structure_json = []
            if top_n:
                structure_data = structure_data[:top_n]
            for entry in tqdm.tqdm(structure_data):
                structure = adapter_function(entry).dict()
                structure.update(structure.pop("attributes"))
                optimade_structure_json.append(structure)

            with open(optimade_structure_file, "w") as f:
                f.write(bson.json_util.dumps(optimade_structure_json))

            break

        else:
            raise RuntimeError(f"Could not find {structure_file!r} in data archive.")

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
    # CAMD dataset
    load_structures(
        "10.6084/m9.figshare.19601956.v1",
        "./files/camd_data_to_release_wofeatures.json",
        camd_entry_to_optimade_model,
    )

    # Wren dataset
    load_structures(
        "10.5281/zenodo.6345276",
        "./",
        wren_entry_to_optimade_model,
    )
