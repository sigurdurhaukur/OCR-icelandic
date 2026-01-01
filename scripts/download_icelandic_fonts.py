import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests


def download_icelandic_fonts(overwrite=False):
    print("Fetching font list from Google Fonts API...")

    # Use the Google Fonts Developer API
    # This endpoint works without authentication
    url = "https://www.googleapis.com/webfonts/v1/webfonts?sort=popularity&key=AIzaSyDummyKey"

    # Create download directory
    download_dir = Path("icelandic_fonts")
    download_dir.mkdir(exist_ok=True)

    try:
        # Try without key first
        response = requests.get(
            "https://www.googleapis.com/webfonts/v1/webfonts?sort=popularity"
        )
        data = response.json()

        if "error" in data:
            raise Exception("Need API key")

        fonts_data = data

    except Exception as e:
        print(f"API error: {e}")
        print("Trying alternative method...")

        # Alternative: Use the fonts.google.com metadata endpoint
        response = requests.get("https://fonts.google.com/metadata/fonts")
        fonts_data = json.loads(response.text)

        # Transform to expected format
        if "familyMetadataList" in fonts_data:
            icelandic_fonts = []
            for font in fonts_data["familyMetadataList"]:
                if "latin-ext" in font.get("subsets", []):
                    # Need to construct download URLs manually
                    family = font["family"]
                    icelandic_fonts.append(
                        {
                            "family": family,
                            "variants": font.get("fonts", {}),
                            "subsets": font.get("subsets", []),
                        }
                    )

            print(
                f"Found {len(icelandic_fonts)} fonts supporting Icelandic characters\n"
            )

            def download_font_alt(font):
                family = font["family"]
                font_dir = download_dir / family.replace(" ", "_").replace("/", "_")

                # Skip if directory exists and overwrite is False
                if not overwrite and font_dir.exists():
                    print(f"  ⏭ {family}: Already exists, skipping")
                    return

                try:
                    # Download using Google Fonts CSS API
                    family_param = family.replace(" ", "+")
                    css_url = f"https://fonts.googleapis.com/css2?family={family_param}&subset=latin-ext&display=swap"

                    headers = {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                    }
                    css_response = requests.get(css_url, headers=headers, timeout=10)

                    # Extract font URLs
                    font_urls = re.findall(
                        r"src: url\((https://[^)]+)\)", css_response.text
                    )

                    if not font_urls:
                        print(f"  ✗ {family}: No font files found")
                        return

                    font_dir.mkdir(exist_ok=True)

                    downloaded = 0
                    for i, url in enumerate(font_urls):
                        ext = "woff2" if "woff2" in url else "ttf"
                        filename = f"variant_{i}.{ext}"

                        font_data = requests.get(url, timeout=10).content
                        with open(font_dir / filename, "wb") as f:
                            f.write(font_data)
                        downloaded += 1

                    print(f"  ✓ {family} ({downloaded} files)")

                except Exception as e:
                    print(f"  ✗ {family}: {e}")

            # Download fonts in parallel
            with ThreadPoolExecutor(max_workers=10) as executor:
                executor.map(download_font_alt, icelandic_fonts)

            print(f"\n✅ Download complete! Fonts saved to {download_dir.absolute()}")
            return

    # Original API path
    if "items" not in fonts_data:
        print("Unexpected API response format")
        print(json.dumps(fonts_data, indent=2)[:500])
        return

    icelandic_fonts = [
        font for font in fonts_data["items"] if "latin-ext" in font.get("subsets", [])
    ]

    print(f"Found {len(icelandic_fonts)} fonts supporting Icelandic characters\n")

    def download_font(font):
        family = font["family"]
        files = font.get("files", {})

        if not files:
            print(f"  ✗ {family}: No files available")
            return

        font_dir = download_dir / family.replace(" ", "_").replace("/", "_")

        # Skip if directory exists and overwrite is False
        if not overwrite and font_dir.exists():
            print(f"  ⏭ {family}: Already exists, skipping")
            return

        try:
            font_dir.mkdir(exist_ok=True)

            downloaded = 0
            for variant, url in files.items():
                url = url.replace("http://", "https://")
                ext = url.split(".")[-1]
                filename = f"{variant}.{ext}"

                font_data = requests.get(url, timeout=10).content
                with open(font_dir / filename, "wb") as f:
                    f.write(font_data)
                downloaded += 1

            print(f"  ✓ {family} ({downloaded} variants)")

        except Exception as e:
            print(f"  ✗ {family}: {e}")

    # Download fonts in parallel
    with ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(download_font, icelandic_fonts)

    print(f"\n✅ Download complete! Fonts saved to {download_dir.absolute()}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download Icelandic fonts from Google Fonts"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Redownload fonts even if they already exist",
    )
    args = parser.parse_args()

    download_icelandic_fonts(overwrite=args.overwrite)
