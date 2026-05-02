"""Module with versioning mixin for handling version control."""

import os
import re
from abc import ABC, abstractmethod
from typing import List, Dict, Any

import yaml
from yamlmaker import generate


class VersioningMixin(ABC):
    """Mixin that handles versioning manipulations."""
    version = None

    def get_version(self) -> str | None:
        """Returns current version of model"""
        if self.version is not None:
            return self.version
        else:
            raise RuntimeError("Model not loaded!")

    def known_versions(self, root: str) -> List[str]:
        """Returns list of found versions"""
        return [d for d in os.listdir(root) if os.path.isdir(root + '/' + d)]

    def get_latest(self, root: str) -> str:
        """Returns version with highest number or empty string if root is empty"""
        version_numbers = []
        pattern = re.compile(r"^v[\d]+")
        for ver in self.known_versions(root):
            if pattern.match(ver):
                version_numbers.append(int(ver[1:]))
            else:
                continue
        if version_numbers:
            return 'v' + str(max(version_numbers))
        else:
            return ""

    def get_next_version(self, root: str) -> str:
        """Evaluates next version name in root"""
        latest = self.get_latest(root)
        if latest:
            return 'v' + str(int(latest[1:]) + 1)
        else:
            return 'v1'

    def save_model(self, path: str, *args, **kwargs):
        """Saves model into path.
        Creates subfolder with version name.
        This subfolder must contain info.yml file with metadata.

        method _save_model() must be implemented.

        _save_model() has path as it's argument, not path!

        _save_model() must return dictionary with metadata to save!
        """
        version = self.get_next_version(path)
        path = path + '/' + version
        os.makedirs(path, exist_ok=False)
        info = self._save_model(path, version, *args, **kwargs)
        generate(info, path + '/' + "info")
        self.version = version


    @abstractmethod
    def _save_model(self, path: str, version: str, *args, **kwargs) -> Dict[str, Any]:
        pass

    def load_model(self, path: str, *args, version: str="latest", **kwargs):
        """Loads model from path.
        
        method _load_model() must be implemented.
        """
        known = self.known_versions(path)
        if version == "latest":
            version = self.get_latest(path)
        if version not in known:
            raise RuntimeError("Unknown version!")
        self.version = version

        with open(path + '/' + version + '/' + "info.yml", "rt", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        self._load_model(path + '/' + version, data, *args, **kwargs)

    @abstractmethod
    def _load_model(self, path: str, data: Dict[str, Any], *args, **kwargs):
        pass
