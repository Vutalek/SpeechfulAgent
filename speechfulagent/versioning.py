import os
import re
import typing as tt


class VersioningMixin:
    def known_versions(self, dir: str) -> tt.List[str]:
        return [d for d in os.listdir(dir) if os.path.isdir(dir + '/' + d)]
    
    def get_latest(self, dir: str) -> str:
        version_numbers = []
        pattern = re.compile(r"^v[\d]+")
        for ver in self.known_versions(dir):
            if pattern.match(ver):
                version_numbers.append(int(ver[1:]))
            else:
                continue
        if len(version_numbers):
            return 'v' + str(max(version_numbers))
        else:
            return ""
        
    def get_next_version(self, dir: str) -> str:
        latest = self.get_latest(dir)
        if latest:
            return 'v' + str(int(latest[1:]) + 1)
        else:
            return 'v1'