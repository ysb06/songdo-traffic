# path_config.py
from __future__ import annotations
from pathlib import Path
import yaml
from typing import Any, Dict, Iterator, Mapping


class _PathSection(Mapping[str, Path]):
    """
    한 섹션(dataset, nodelink, imcrts, …)을 표현.
    - 속성 접근: section.metr_imc
    - 딕셔너리 접근: section["metr_imc"]
    """
    def __init__(self, base_dir: Path, raw: Dict[str, Any]) -> None:
        self._base = base_dir / raw["dir"]
        self._files: Dict[str, str] = dict(raw.get("filenames", {}))

    # ── 파일 경로 제공 ────────────────────────────────────────────────
    def _resolve(self, name: str) -> Path:
        try:
            filename = self._files[name]
        except KeyError as err:
            raise AttributeError(name) from err
        return self._base / filename

    # 속성 방식
    def __getattr__(self, item: str) -> Path:          # conf.dataset.metr_imc
        return self._resolve(item)

    # dict 방식
    def __getitem__(self, item: str) -> Path:          # conf["dataset"]["metr_imc"]
        return self._resolve(item)

    # ── Mapping 프로토콜 구현 (선택) ────────────────────────────────
    def __iter__(self) -> Iterator[str]:
        return iter(self._files)

    def __len__(self) -> int:
        return len(self._files)

    def __repr__(self) -> str:                         # 디버깅용
        return f"<PathSection base={self._base}>"


class PathConfig(Mapping[str, _PathSection]):
    """
    전체 YAML을 표현하는 최상위 객체.
      conf.dataset.metr_imc
      conf["imcrts"]["data"]
    둘 다 Path 객체를 반환한다.
    """
    def __init__(self, yaml_file: str | Path) -> None:
        with open(yaml_file, "r", encoding="utf-8") as f:
            data: dict[str, Any] = yaml.safe_load(f)

        # root_dir은 필수로 가정
        self._root = Path(data["root_dir"])
        self._sections: Dict[str, _PathSection] = {}

        for key, section in data.items():
            if key == "root_dir":
                continue
            self._sections[key] = _PathSection(self._root, section)

    # ── 섹션 접근 ──────────────────────────────────────────────────
    def __getattr__(self, item: str) -> _PathSection:   # conf.dataset
        try:
            return self._sections[item]
        except KeyError as err:
            raise AttributeError(item) from err

    def __getitem__(self, item: str) -> _PathSection:   # conf["dataset"]
        return self._sections[item]

    # ── Mapping 프로토콜 구현(선택) ────────────────────────────────
    def __iter__(self) -> Iterator[str]:
        return iter(self._sections)

    def __len__(self) -> int:
        return len(self._sections)

    def __repr__(self) -> str:
        return f"<ConfigPaths root={self._root}>"