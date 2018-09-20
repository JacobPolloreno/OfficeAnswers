import json
import logging


logger = logging.getLogger(__name__)


class Config(object):

    def __init__(self) -> None:
        pass

    def __str__(self) -> str:
        res = []
        for key in dir(self):
            if not key.startswith("__"):
                res.append((key, getattr(self, key)))
        return str(res)

    def _load_from_dict(self, obj: dict) -> None:
        for key, value in obj.items():
            setattr(self, key, value)

    def from_dict(self, obj) -> None:
        self._load_from_dict(obj)

    def from_json_file(self, file_path: str) -> None:
        with open(file_path, 'r') as f:
            try:
                obj = json.load(f)
                self._load_from_dict(obj)
            except json.decoder.JSONDecodeError:
                error_msg = f"Error reading `{file_path} file`\n" + \
                    "Expecting json file. See `data/sample_model.config'"
                logger.error(error_msg)
