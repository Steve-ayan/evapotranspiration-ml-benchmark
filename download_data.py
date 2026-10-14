import os
import requests
import urllib3

# Suppress SSL verification warnings caused by local security policies/proxies
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

LOCATIONS = {
    "Sokoto": {"lat": 13.0059, "lon": 5.2476},
    "Kaduna": {"lat": 10.5105, "lon": 7.4165},
    "Ibadan": {"lat": 7.3775, "lon": 3.9470},
    "Port_Harcourt": {"lat": 4.8156, "lon": 7.0498},
}

# NASA POWER Daily Agroclimatology parameters
PARAMETERS = [
    "T2M_MAX",
    "T2M_MIN",
    "RH2M",
    "ALLSKY_SFC_SW_DWN",
    "WS2M",
]

START_DATE = "19900101"
END_DATE = "20231231"
BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"


def fetch_nasa_data():
    output_dir = os.path.join("data", "raw")
    os.makedirs(output_dir, exist_ok=True)

    with requests.Session() as session:
        # Disable SSL verification on session calls
        session.verify = False

        for name, coords in LOCATIONS.items():
            print(f"\nDownloading climate data for {name}...")

            params = {
                "parameters": ",".join(PARAMETERS),
                "community": "AG",
                "longitude": coords["lon"],
                "latitude": coords["lat"],
                "start": START_DATE,
                "end": END_DATE,
                "format": "CSV",
            }

            try:
                response = session.get(
                    BASE_URL,
                    params=params,
                    timeout=180,
                )

                if response.status_code == 200:
                    file_path = os.path.join(
                        output_dir,
                        f"{name}_raw_1990_2023.csv",
                    )

                    with open(file_path, "wb") as file:
                        file.write(response.content)

                    print(f"Download successful: {file_path}")

                else:
                    print(
                        f"Download failed for {name}. "
                        f"HTTP status: {response.status_code}"
                    )
                    print(f"Error details: {response.text[:1000]}")

            except requests.exceptions.RequestException as error:
                print(f"Network error for {name}: {error}")


if __name__ == "__main__":
    fetch_nasa_data()