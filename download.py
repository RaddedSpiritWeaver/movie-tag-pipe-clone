import os
import shutil
import subprocess

ANNOTATIONS_FILE = "data/annotations.txt"
DOWNLOAD_DIR = "data/movies"
YTDLP_PATH = shutil.which("yt-dlp") or shutil.which("yt-dlp.exe") or "yt-dlp"
COOKIES_FROM_BROWSER = ["firefox", "brave", "chrome", "edge"]  # tried in order; set to [] to disable

os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def resolve_cookie_browser() -> str | None:
    for browser in COOKIES_FROM_BROWSER:
        result = subprocess.run(
            [YTDLP_PATH, "--cookies-from-browser", browser, "--simulate", "ytsearch1:test"],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
        )
        if "Could not copy" not in result.stderr and "error" not in result.stderr.lower():
            print(f"Using cookies from: {browser}")
            return browser
        print(f"Cookie extraction failed for {browser}, trying next...")
    print("WARNING: Could not extract cookies from any browser. Age-restricted videos will be skipped.")
    return None


COOKIE_BROWSER = resolve_cookie_browser()


def video_already_downloaded(video_id: str) -> bool:
    for f in os.listdir(DOWNLOAD_DIR):
        if video_id in f:
            return True
    return False

def main():
    with open(ANNOTATIONS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue

            video_id = line.split(":")[0].strip()

            youtube_url = f"https://www.youtube.com/watch?v={video_id}"

            if video_already_downloaded(video_id):
                print(f"Skipping {video_id} (already downloaded)")
                continue

            print(f"Downloading {youtube_url}")
            cmd = [
                YTDLP_PATH,
                youtube_url,
                "-f", "bestvideo+bestaudio/best",
                "--cookies", "cookies.txt",
                "--merge-output-format", "mkv",
                "--sleep-interval", str(5),
                "--remux-video", "mkv",
                "--user-agent", "Mozilla/5.0",
                # "--extractor-args", "youtube:player_client=android,mweb",
                "-o", os.path.join(DOWNLOAD_DIR, "%(id)s.%(ext)s"),
            ]
            # if COOKIE_BROWSER:
            #     cmd += ["--cookies-from-browser", COOKIE_BROWSER]
            subprocess.run(cmd, check=False)

if __name__ == "__main__":
    main()