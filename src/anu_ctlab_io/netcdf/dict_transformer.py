#!/usr/bin/env python3
from benedict import benedict
import collections.abc
from typing import Any, Self
from enum import Enum, auto
from dataclasses import dataclass
import base64
import zlib
from copy import deepcopy

class DictTransform(Enum):
    insert = auto()
    remove = auto()
    rekey = auto()
    update = auto()

    def __str__(self) -> str:
        return self.__dict__["_name_"]

    @classmethod
    def from_string(cls, string: str) -> Self:
        try:
            return cls[string]
        except KeyError as e:
            raise RuntimeError(f"DictTransform {string} not recognized.", e)


@dataclass
class DictTransformRecord:
    transform: DictTransform
    cache: Any

    def __str__(self):
        return str({"transform": str(self.transform), "cache": str(self.cache)})


class DictTransformerException(Exception):
    pass


class DictTransformer:
    _dict: dict
    _transform_list: list[DictTransformRecord]
    _transform_index: int

    def __init__(self, dictionary: dict):
        self._dict = dictionary
        self._transform_list = []
        self._transform_index = -1

    def _add_transform(self, transform: DictTransformRecord):
        if self._transform_index < len(self._transform_list) - 1:
            del self._transform_list[self._transform_index + 1 :]

        self._transform_list.append(transform)
        self._transform_index += 1

    def insert(self, kv_pairs: dict, *, track=True):
        if any([k in self._dict and not isinstance(v, dict) for k,v in kv_pairs.items()]):
            raise DictTransformerException(
                "Inserting at an already present key in a dict is invalid, use update instead."
            )
        d = benedict(self._dict)
        d.merge(kv_pairs)
        self._dict.update(d)
        if track:
            self._add_transform(DictTransformRecord(DictTransform.insert, kv_pairs))

    def remove(self, keys: list[str | list[str]], *, track=True):
        cache = benedict()
        d = benedict(self._dict)
        for k in keys:
            cache[k] = d.pop(k)
            d.clean()
        if track:
            self._add_transform(DictTransformRecord(DictTransform.remove, cache))

    def rekey(self, key_map: dict[str, str | list[str]], *, track=True):
        rekeyed_items = benedict()
        for k, v in key_map.items():
            rekeyed_items[v] = self._dict.pop(k)
        self._dict |= rekeyed_items
        if track:
            self._add_transform(DictTransformRecord(DictTransform.rekey, key_map))

    def update(self, update_map: dict[str, Any], *, track=True):
        d = benedict(self._dict)
        um = benedict(update_map)
        cache = benedict({"redo": um, "undo": {}})

        for kp in um.keypaths():
            if isinstance(um[kp], dict):
                pass
            else:
                cache["undo"][kp] = d[kp]
                d[kp] = um[kp]
        self._dict.update(d)

        if track:
            self._add_transform(DictTransformRecord(DictTransform.update, cache))

    def undo(self):
        tx = self._transform_list[self._transform_index]
        self._transform_index -= 1
        match tx.transform:
            case DictTransform.insert:
                self.remove(tx.cache.keys(), track=False)
            case DictTransform.remove:
                self.insert(tx.cache, track=False)
            case DictTransform.rekey:
                self.rekey({v: k for k, v in tx.cache.items()}, track=False)
            case DictTransform.update:
                self.update(tx.cache["undo"], track=False)

    def redo(self):
        if self._transform_index < len(self._transform_list) - 1:
            self._transform_index += 1
            tx = self._transform_list[self._transform_index]
            match tx.transform:
                case DictTransform.insert:
                    self.insert(tx.cache, track=False)
                case DictTransform.remove:
                    self.remove(tx.cache.keys(), track=False)
                case DictTransform.rekey:
                    self.rekey(tx.cache, track=False)
                case DictTransform.update:
                    self.update(tx.cache["redo"], track=False)

    def undo_all(self):
        while self._transform_index >= 0:
            self.undo()

    def redo_all(self):
        while self._transform_index < len(self._transform_list) - 1:
            self.redo()

    def __repr__(self):
        return "<DictTransformer>" + self._dict.__repr__()

    def serialize(self):
        return base64.a85encode(
            zlib.compress(str(self._transform_list).encode("utf-8"))
        )

    @classmethod
    def from_existing_transformer(cls, d: dict, tx: Self) -> Self:
        new_tx = cls(d)
        new_tx._transform_list = deepcopy(tx._transform_list)
        new_tx._transform_index = tx._transform_index
        return new_tx
