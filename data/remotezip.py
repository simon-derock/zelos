"""Minimal HTTP range-request file object so `zipfile` can read members of a
remote zip without downloading the whole archive.

Input: URL of a zip served with byte-range support. Output: a seekable,
read-only file-like object usable by `zipfile.ZipFile`.
"""
import io
import requests


class HTTPRangeFile(io.RawIOBase):
    def __init__(self, url: str, size: int | None = None, session=None):
        self.url = url
        self.s = session or requests.Session()
        if size is None:
            r = self.s.get(url, headers={"Range": "bytes=0-0"}, stream=True, timeout=60)
            cr = r.headers.get("Content-Range", "")
            size = int(cr.split("/")[-1]) if "/" in cr else int(r.headers["Content-Length"])
            r.close()
        self.size = size
        self.pos = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + n, self.size) - 1
        for _ in range(5):
            try:
                r = self.s.get(self.url, headers={"Range": f"bytes={self.pos}-{end}"}, timeout=120)
                if r.status_code == 206:
                    break
            except requests.RequestException:
                continue
        else:
            raise IOError("range request failed")
        data = r.content
        self.pos += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)
