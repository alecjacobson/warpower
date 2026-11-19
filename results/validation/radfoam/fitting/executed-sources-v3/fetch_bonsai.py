"""Fetch only Bonsai's calibrated low-resolution images from the official ZIP."""

import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path

URL = "https://storage.googleapis.com/gresearch/refraw360/360_v2.zip"
ROOT = Path("data/radfoam-training")


class RangeFile(io.RawIOBase):
    def __init__(self, url):
        self.url = url
        response = urllib.request.urlopen(urllib.request.Request(url, method="HEAD"))
        self.size = int(
            response.headers.get("x-goog-stored-content-length", response.headers["Content-Length"])
        )
        self.etag = response.headers["ETag"]
        response.close()
        self.position, self.start, self.cache, self.transferred = 0, 0, b"", 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset + (0 if whence == 0 else self.position if whence == 1 else self.size)
        return self.position

    def read(self, size=-1):
        size = self.size - self.position if size < 0 else min(size, self.size - self.position)
        if size <= 0:
            return b""
        if not (
            self.start <= self.position and self.position + size <= self.start + len(self.cache)
        ):
            stop = min(self.size, self.position + max(size, 1024**2))
            request = urllib.request.Request(
                self.url,
                headers={"Range": f"bytes={self.position}-{stop - 1}", "If-Match": self.etag},
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                assert response.status == 206
                assert response.headers["Content-Range"].startswith(
                    f"bytes {self.position}-{stop - 1}/"
                )
                self.cache = response.read(stop - self.position + 1)
            assert len(self.cache) == stop - self.position
            self.start = self.position
            self.transferred += len(self.cache)
        value = self.cache[self.position - self.start : self.position - self.start + size]
        self.position += len(value)
        return value


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    remote = RangeFile(URL)
    with zipfile.ZipFile(remote) as archive:
        selected = [
            x
            for x in archive.infolist()
            if not x.is_dir()
            and (
                x.filename.startswith("bonsai/images_4/")
                or x.filename.startswith("bonsai/sparse/0/")
                or x.filename == "bonsai/poses_bounds.npy"
            )
        ]
        assert selected, archive.namelist()[:20]
        print(
            json.dumps(
                {
                    "selected_files": len(selected),
                    "uncompressed_bytes": sum(x.file_size for x in selected),
                    "archive_bytes": remote.size,
                }
            ),
            flush=True,
        )
        entries = []
        for i, entry in enumerate(sorted(selected, key=lambda x: x.header_offset)):
            path = ROOT / entry.filename
            assert path.resolve().is_relative_to(ROOT.resolve())
            data = archive.read(entry)  # zipfile checks the ZIP CRC.
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            entries.append(
                {
                    "path": entry.filename,
                    "bytes": len(data),
                    "crc32": entry.CRC,
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
            if i % 40 == 0:
                print(
                    json.dumps({"downloaded_files": i + 1, "network_bytes": remote.transferred}),
                    flush=True,
                )
    report = {
        "source": URL,
        "archive_etag": remote.etag,
        "archive_bytes": remote.size,
        "network_bytes": remote.transferred,
        "files": entries,
    }
    (ROOT / "bonsai-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({"complete": True, "files": len(entries), "network_bytes": remote.transferred}),
        flush=True,
    )


if __name__ == "__main__":
    main()
