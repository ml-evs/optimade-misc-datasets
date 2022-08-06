import os
import bson.json_util
import logging

os.environ["OPTIMADE_PROVIDER_FIELDS"] = '{"structures": ["hull_distance", "formation_energy"]}'
os.environ["OPTIMADE_PROVIDER"] = '{"prefix": "odbx", "name": "Open Database of Xtals","description": "The CAMD dataset, hosted via OPTIMADE https://doi.org/10.6084/m9.figshare.19601956.v1"}'
os.environ["OPTIMADE_DATABASE_BACKEND"] = "mongomock"
os.environ["OPTIMADE_INSERT_TEST_DATA"] = "false"

import uvicorn
from optimade.server.main import app
from optimade.server.routers.structures import structures_coll
LOG = logging.getLogger("optimade")

def download_from_figshare(doi, data_dir=None):
    import requests
    import tqdm
    import json
    import os
    import hashlib
    from pathlib import Path

    chunk_size = 1024**2
    figshare_api_url = "https://api.figshare.com/v2/"

    response = requests.get(f"{figshare_api_url}/articles?doi={doi}")
    if response.status_code != 200:
        raise RuntimeError(f"Bad response: {response.content}")
    response = response.json()
    article_id = response[0]["id"]

    response = requests.get(f"{figshare_api_url}/articles/{article_id}")
    if response.status_code != 200:
        raise RuntimeError(f"Bad response: {response.content}")
    response = response.json()

    data_dir = data_dir or "."
    article_dir = Path(data_dir) / f"figshare_{article_id}"
    if not article_dir.exists():
        os.makedirs(article_dir)

    with open(article_dir / "metadata.json", "w") as f:
        json.dump(response, f)

    filenames = []
    file_ids = []

    for files in response["files"]:
        download_url = files["download_url"]
        local_path = Path(data_dir) / f"figshare_{article_id}" / str(files["id"]) / files["name"]
        if local_path.exists():
            with open(local_path, "rb") as f:
                md5 = hashlib.md5(f.read()).hexdigest()
            if md5 != files["supplied_md5"]:
                print(f"Downloaded file {local_path} ({md5!r}) does not match MD5 supplied by figshare ({files['supplied_md5']!r}), will move")
                local_path.replace(Path(str(local_path) + ".old"))
            else:
                print(f"{local_path} already exists locally ({md5!r}), not re-downlaoding...")
                filenames.append(local_path)
                file_ids.append(files["id"])
                continue

        with requests.get(download_url, stream=True) as file_stream:
            print(f"Downloading file {files['name']!r} with size {files['size'] // 1024**2} MB")
            with open(local_path, "wb") as f:
                for chunk in tqdm.tqdm(file_stream.iter_content(chunk_size=chunk_size), total=int(files['size']) // chunk_size, unit=" MB"):
                    f.write(chunk)

        filenames.append(local_path)
        file_ids.append(files["id"])

    return (filenames, file_ids, article_id, article_dir)

def pmg_structure_to_optimade_dict(s, object_id=None):
    import re
    import math
    from optimade.models import StructureResourceAttributes
    from optimade.models.utils import anonymous_element_generator
    from bson.objectid import ObjectId

    attributes = {}
    attributes["cartesian_site_positions"] = s.lattice.get_cartesian_coords(s.frac_coords).tolist()
    attributes["fractional_site_positions"] = s.frac_coords.tolist()
    attributes["species_at_sites"] = [_.symbol for _ in s.species]
    attributes["species"] = [{"name": _.symbol, "chemical_symbols": [_.symbol], "concentration": [1]} for _ in set(s.composition.elements)]
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
    attributes["chemical_formula_anonymous"] = "".join(["".join(x) for x in zip(anonymous_element_generator(), reversed(re.split("[A-Z]", s.composition.anonymized_formula)[1:]))])
    attributes["elements"] = sorted([_.symbol for _ in s.composition.elements])
    gcd = math.gcd(*[int(_) for _ in s.composition.to_reduced_dict.values()])
    attributes["chemical_formula_reduced"] = "".join(_ + f"{int(s.composition.to_reduced_dict[_]) // gcd if s.composition.to_reduced_dict[_] // gcd > 1 else ''}" for _ in attributes["elements"])
    attributes["elements_ratios"] = [s.composition.get_atomic_fraction(e) for e in attributes["elements"]]
    attributes["nsites"] = len(attributes["species_at_sites"])
    attributes["lattice_vectors"] = s.lattice.matrix.tolist()
    attributes["structure_features"] = []
    return StructureResourceAttributes(**attributes)
    
def camd_entry_to_optimade_model(entry):
    from pymatgen.core import Structure
    from optimade.models import StructureResource
    s = entry["structure"]
    object_id = entry["_id"]["oid"]
    attributes = pmg_structure_to_optimade_dict(Structure.from_dict(s), object_id=object_id)
    attributes._odbx_hull_distance = entry["stability"]
    attributes._odbx_formation_energy = entry["delta_e"]
    attributes._odbx_space_group = entry["space_group"]
    _id = entry["data_id"]
    return StructureResource(id=_id, attributes=attributes)


def extract_files(files, remove_archive: bool = True):
    import tarfile
    for f in files:
        if str(f).endswith(".tar.gz"):
            with tarfile.open(f, "r:gz") as tar:
                tar.extractall(f.parent)

        if remove_archive:
            os.remove(f)


@app.on_event("startup")
async def load_structures():

    # Download dataset if missing
    camd_doi = "https://doi.org/10.6084/m9.figshare.19601956.v1"
    files, file_ids, article_id, article_dir = download_from_figshare(camd_doi)
    structure_file = f"figshare_{article_id}/{file_ids[0]}/files/camd_data_to_release_wofeatures.json"
    import json
    with open(structure_file, "r") as f:
        structure_data = json.load(f)

    import tqdm
    optimade_structure_json = []
    for entry in tqdm.tqdm(structure_data):
        structure = camd_entry_to_optimade_model(entry).dict()
        structure.update(structure.pop("attributes"))
        optimade_structure_json.append(structure)

    LOG.info("Found %s structures", len(optimade_structure_json))
    structures_coll.insert(optimade_structure_json)
    LOG.info("Inserted %s structures into database", len(optimade_structure_json))


if __name__ == "__main__":
    uvicorn.run("__main__:app", port=5001)
