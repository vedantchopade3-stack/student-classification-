
from pathlib import Path
import requests
import zipfile
import shutil
import io
import sys

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

RECORD_ID = "22015963"
ZIP_NAME = "ECOTEC_Online_Student_Risk_Dataset.zip"
EXPECTED_CSV = "ecotec_online_student_risk.csv"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) StudentProject/1.0",
    "Accept": "application/json"
}

def download_dataset():
    session = requests.Session()
    session.headers.update(HEADERS)

    print("Getting dataset information from Zenodo...")

    api_url = f"https://zenodo.org/api/records/{RECORD_ID}"
    response = session.get(api_url, timeout=60)

    if response.status_code != 200:
        raise RuntimeError(
            f"Zenodo API error: {response.status_code}\n"
            f"Response: {response.text[:500]}\n\n"
            "Open https://zenodo.org/records/22015963 in your browser "
            "and download the ZIP manually."
        )

    record = response.json()

    file_info = next(
        (f for f in record.get("files", [])
         if f.get("key") == ZIP_NAME),
        None
    )

    if file_info is None:
        raise FileNotFoundError(
            f"{ZIP_NAME} was not found in the record.\n"
            f"Available files: {[f.get('key') for f in record.get('files', [])]}"
        )

    download_url = file_info.get("links", {}).get("self")

    if not download_url:
        raise RuntimeError("Zenodo did not provide a download URL.")

    print("Downloading:", ZIP_NAME)

    download = session.get(
        download_url,
        timeout=120,
        stream=True
    )
    download.raise_for_status()

    content = io.BytesIO()
    for chunk in download.iter_content(chunk_size=1024 * 1024):
        if chunk:
            content.write(chunk)

    content.seek(0)

    if not zipfile.is_zipfile(content):
        raise RuntimeError(
            "Downloaded response is not a valid ZIP archive. "
            "Try the manual browser download."
        )

    print("Extracting CSV...")

    with zipfile.ZipFile(content) as archive:
        csv_files = [
            name for name in archive.namelist()
            if name.lower().endswith(".csv")
        ]

        if not csv_files:
            raise FileNotFoundError("No CSV file found in the archive.")

        selected = next(
            (name for name in csv_files
             if Path(name).name.lower() == EXPECTED_CSV),
            csv_files[0]
        )

        output_path = DATA_DIR / EXPECTED_CSV

        with archive.open(selected) as source:
            with output_path.open("wb") as target:
                shutil.copyfileobj(source, target)

    print("\nDataset downloaded successfully!")
    print("CSV file:", output_path)
    print("Archive CSV:", selected)


if __name__ == "__main__":
    try:
        download_dataset()
    except Exception as error:
        print("\nDOWNLOAD ERROR:", error)
        sys.exit(1)
