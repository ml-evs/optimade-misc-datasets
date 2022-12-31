import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Union

import pandas as pd
import pymatgen.core
import pymatgen.io.vasp
import pymatgen.io.cif
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

def download_from_github(
    repo: str, data_dir: Path = None, commit_or_branch: str = "main",
) -> tuple[list[Path], list[int], int, Path]:
    """Download a single file from a URL. 

    Files will be placed in a folder structure (under `data_dir`) by
    repo and then path, with an additional top-level metadata.json
    file containing the full response from the corresponding API.

    Returns:
        A list of file paths, a list of file ids,
        the article id and the article directory.

    """
    repo = repo.replace("github/", "")
    data_dir = data_dir or Path(__file__).parent
    local_path = data_dir / f"github_{repo.replace('/', '__')}" / f"{commit_or_branch}.zip"
    os.makedirs(local_path.parent, exist_ok=True)
    
    download_file(f"https://github.com/{repo}/archive/{commit_or_branch}.zip", local_path)

    return [local_path], local_path.parent



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
    url: str, local_path: Path, size: int = None, name: str = None, chunk_size: int = 1024**2
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
        if size:
            LOG.info(f"Downloading file {name!r} with size {int(size) // 1024**2} MB")
        with open(local_path, "wb") as f:
            for chunk in tqdm.tqdm(
                file_stream.iter_content(chunk_size=chunk_size),
                total=int(size) // chunk_size if size else None,
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
    _id = "camd-" + entry["data_id"]
    return StructureResource(id=_id, attributes=attributes)


def _3dsc_row_to_optimade_model(entry: dict) -> StructureResource:
    return entry


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

    _id = "wren-" + material_id
    attributes.formation_energy = E_f_dft
    attributes.hull_distance = E_hull_mp_dft
    return StructureResource(id=_id, attributes=attributes)


def extract_files(files: list[Path], remove_archive: bool = False) -> None:
    """Extracts an archive tar.gz/zip file into its root directory, optionally deleting the
    archive once complete.

    Arguments:
        files: A list of files to extract.
        remove_archive: Whether to remove the archive once complete.

    """
    import tarfile
    import zipfile

    for f in files:
        if sorted(f.suffixes) == [".gz", ".tar"]:
            LOG.info(f"Extracting file {f} to {f.parent}")
            with tarfile.open(f, "r:gz") as tar:
                tar.extractall(f.parent)
            LOG.info(f"File {f} extracted.")

            if remove_archive:
                LOG.info(f"Removing archive {f}.")
                os.remove(f)

        elif f.suffix == ".zip":
            LOG.info(f"Extracting file {f} to {f.parent}")
            with zipfile.ZipFile(f, "r") as zip:
                zip.extractall(f.parent)
            LOG.info(f"File {f} extracted.")

            if remove_archive:
                LOG.info(f"Removing archive {f}.")
                os.remove(f)


def load_from_cifs(structure_file, metadata, id_prefix):
    structure_data = []
    data = metadata.to_dict(orient="index")
    for ind in tqdm.tqdm(data):
        row = data[ind]
        cif_fname = row["cif"]
        _id = f"{id_prefix}/{ind}"
        structure = pymatgen.io.cif.CifParser(structure_file.parent / "3DSC-main" / "superconductors_3D" / cif_fname).get_structures()[0]
        breakpoint()
        attributes = from_pymatgen(structure)
        attributes.critical_temperature = row["tc"]
        attributes.formation_energy = row["formation_energy_per_atom_2"]
        attributes.hull_distance = row["e_above_hull_2"]
        s = StructureResource(id=_id, attributes=attributes).dict()
        s.update(s.pop("attributes"))
        structure_data.append(s)

    return structure_data

            
def load_from_poscar(structure_file, metadata, id_prefix):
    optimade_structure_json = []
    for ff in os.walk(structure_file.parent, topdown=True):
        if ff[-1] == ["POSCAR"]:
            _id = ff[-3].split("/")[-1]
            structure = pymatgen.io.vasp.inputs.Poscar.from_file(
                f"{ff[0]}/POSCAR"
            ).structure
            attributes = from_pymatgen(structure)
            attributes.formation_energy = metadata[
                metadata["id"] == _id
            ]["decomp_energy"].values[0]
            if id_prefix:
                _id = f"{id_prefix}-{_id}"
            s = StructureResource(id=_id, attributes=attributes).dict()
            s.update(s.pop("attributes"))
            optimade_structure_json.append(s)
        else:
            raise RuntimeError(
                "Cannot read from folder directory without additional metadata file"
            )
    return optimade_structure_json

def law_to_optimade(entry):
    return entry


def load_structures(
    doi: str,
    target_file: str,
    adapter_function: Callable = None,
    data_load_function: Callable = None,
    metadata_csv: Union[str, Path] = None,
    insert: bool = False,
    top_n: int = None,
    remove_archive: bool = False,
    id_prefix: str = None,
) -> list[dict[str, Any]]:
    """Download, extract and convert a Zenodo or Figshare dataset, optionally
    inserting it into an optimade-python-tools database.

    The function expects to find a JSON or gzipped-JSON file at the `target_file`
    path, which itself should contain either a list of objects to pass through the
    `adapter_function`, or a dictionary with a key `data` that contains such a list.

    Arguments:
        doi: The DOI of the dataset to download and ingest.
        target_file: The path to the file in the dataset to extract and convert.
        adapter_function: The function to use to convert the dataset into OPTIMADE models.
        metadata_csv: The path to a CSV file containing metadata for the dataset; if present,
            will crawl the directory for POSCARs and grab energies from the metadata.
        insert: Whether to insert the converted models into a database.
        top_n: The number of entries to convert and insert.
        remove_archive: Whether to remove the archive once complete.
        id_prefix: An optional string to use as a prefix for OPTIMADE IDs

    Returns:
        A list of 'flattened' OPTIMADE structure entries.

    """

    import gzip
    import bson.json_util
    from optimade.server.routers.structures import structures_coll

    # Download dataset if missing
    if doi.startswith("github"):
        files, article_dir = download_from_github(doi)
    else:
        files, article_dir = download_from_doi(doi)

    structure_file = Path(article_dir) / target_file

    optimade_structure_file = Path(article_dir) / "optimade_structures.bson"

    if not optimade_structure_file.exists():

        extract_files(files, remove_archive=remove_archive)

        mode = "r"
        if structure_file.suffix == ".gz":
            mode = "rb"
        
        if ".json" in structure_file.suffixes:
            if mode == "rb":
                structure_data = json.loads(
                    gzip.decompress(f.read()).decode("utf-8")
                )
            elif mode == "r":
                structure_data = json.load(f)

        if metadata_csv:
            metadata = pd.read_csv(article_dir / metadata_csv, comment="#")
            structure_data = data_load_function(structure_file, metadata, id_prefix)

        optimade_structure_json = []

        if structure_data and isinstance(structure_data, dict):
            structure_data = structure_data["data"]
        if top_n:
            structure_data = structure_data[:top_n]
        for entry in tqdm.tqdm(structure_data):
            structure = adapter_function(entry).dict()
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
        LOG.info("Inserted %s structures into database", len(optimade_structure_json))

    return optimade_structure_json

def insert_all_datasets():

    # 3DSC dataset
    load_structures(
        "github/aimat-lab/3DSC",
        "main.zip",
        data_load_function=load_from_cifs,
        adapter_function=_3dsc_row_to_optimade_model,
        metadata_csv="./3DSC-main/superconductors_3D/data/final/MP/3DSC_MP.csv",
        insert=True,
        id_prefix="3dsc",
    )

    # Law et al dataset
    #load_structures(
    #    "10.5281/zenodo.7089031",
    #    "./upper-bound-energy-gnn-0.1/paper_results/relaxed_structures.tar.gz",
    #    metadata_csv="./upper-bound-energy-gnn-0.1/paper_results/dft_confirmation.csv",
    #    data_load_function=load_from_poscar,
    #    adapter_function=law_to_optimade,
    #    id_prefix="law-gnn",
    #    insert=True,
    #    remove_archive=True,
    #)

    ## CAMD dataset
    #load_structures(
    #    "10.6084/m9.figshare.19601956.v1",
    #    "./34818031/files/camd_data_to_release_wofeatures.json",
    #    adapter_function=camd_entry_to_optimade_model,
    #    insert=True,
    #    remove_archive=True,
    #)

    ## Wren dataset
    #load_structures(
    #    "10.5281/zenodo.6345276",
    #    "./prospective.json.gz",
    #    adapter_function=wren_entry_to_optimade_model,
    #    insert=True,
    #    remove_archive=True,
    #)
