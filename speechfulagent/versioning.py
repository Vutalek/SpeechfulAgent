import os
import re
from abc import ABC, abstractmethod
from typing import List, Dict, Any

import yaml


class VersioningMixin(ABC):
    """Mixin that handles versioning manipulations"""
    version = None

    def get_version(self) -> str | None:
        """Returns current version of model"""
        if self.version is not None:
            return self.version
        else:
            raise RuntimeError("Model not loaded!")

    def known_versions(self, dir: str) -> List[str]:
        """Returns list of found versions"""
        return [d for d in os.listdir(dir) if os.path.isdir(dir + '/' + d)]
    
    def get_latest(self, dir: str) -> str:
        """Returns version with highest number or empty string if dir is empty"""
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
        """Evaluates next version name in dir"""
        latest = self.get_latest(dir)
        if latest:
            return 'v' + str(int(latest[1:]) + 1)
        else:
            return 'v1'
    
    def save_model(self, dir: str, *args, **kwargs):
        """Saves model into dir.
        Creates subfolder with version name.
        This subfolder must contain info.yml file with metadata.

        method _save_model() must be implemented.

        _save_model() has path as it's argument, not dir!
        """
        version = self.get_next_version(dir)
        path = dir + '/' + version
        os.mkdir(path)
        self._save_model(path, version, *args, **kwargs)

    @abstractmethod
    def _save_model(self, path: str, version: str, *args, **kwargs):
        pass

    def load_model(self, dir: str, version: str = "latest", *args, **kwargs):
        """Loads model from dir.
        
        method _load_model() must be implemented.
        """
        known = self.known_versions(dir)
        if version == "latest":
            version = self.get_latest(dir)
        if version not in known:
            raise RuntimeError("Unknown version!")
        self.version = version

        with open(dir + '/' + version + '/' + "info.yml", "rt") as f:
            data = yaml.safe_load(f)
        self._load_model(dir + '/' + version, data, *args, **kwargs)

    @abstractmethod
    def _load_model(self, path: str, data: Dict[str, Any], *args, **kwargs):
        pass