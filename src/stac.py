from __future__ import annotations

import json
from decimal import Decimal
from typing import Union, Tuple, Iterable, cast, Optional, Callable

import shapely.geometry

from geojson.geometry import Geometry
import ciso8601

import helper

Number = Union[int, Decimal, float]
Coordinates = Iterable[Number]
BoundingBox = Tuple[Number, Number, Number, Number]


class STACQuery:
    """
    Object representing a STAC query as used by data blocks.
    """

    def __repr__(self):
        return (
            f"STACQuery(ids={self.ids}, bbox={self.bbox}, intersects={self.intersects}, "
            f"contains={self.contains}, time={self.time}, time_series={self.time_series}, "
            f"limit={self.limit})"
        )

    def __init__(
        self,
        ids: Optional[list] = None,
        bbox: Optional[BoundingBox] = None,
        intersects: Optional[Geometry] = None,
        contains: Optional[Geometry] = None,
        time: Optional[str] = None,
        limit: Optional[int] = None,
        time_series: Optional[list] = None,
        **kwargs,
    ):  # pylint: disable=too-many-arguments

        if bbox and intersects or intersects and contains or contains and bbox:
            raise ValueError(
                """Only one of the following query parameters is
                allowed at a query at any given time:
                * bbox
                * intersects
                * contains."""
            )

        self.__dict__.update(
            kwargs
        )  # At the front so we do not accidentally overwrite some members
        self.ids = ids
        self.bbox = bbox
        self.intersects = intersects
        self.contains = contains

        if not STACQuery.validate_datetime_str(time):
            raise ValueError(
                """Time string could not be validated.
                It must be one of the following formats:
                <RFC3339> (for points in time)
                <RFC3339>/<RFC3339> (for datetime ranges)"""
            )
        self.time = time

        self.limit = 1 if limit is None else limit
        if limit is not None and limit < 0:
            helper.get_logger(__name__).warning(
                "WARNING: limit parameter cannot be < 0, and has been automatically set to 1"
            )
            self.limit = 1

        if time_series is not None:
            for datestr in time_series:
                if not STACQuery.validate_datetime_str(datestr):
                    raise ValueError(
                        """Time string from time_series could not be validated.
                        It must be one of the following formats:
                        <RFC3339> (for points in time)
                        <RFC3339>/<RFC3339> (for datetime ranges)"""
                    )
        self.time_series = time_series

    def bounds(self) -> BoundingBox:
        """
        Get the bounding box of the query AOI(s) as a GeoJson Polygon object.
        """
        if self.bbox:
            return self.bbox
        elif self.intersects:
            return shapely.geometry.shape(self.intersects).bounds
        elif self.contains:
            return shapely.geometry.shape(self.contains).bounds
        else:
            raise ValueError(
                """STACQuery does not contain any of the following query parameters:
                * bbox
                * intersects
                * contains."""
            )

    def geometry(self) -> Geometry:
        if self.bbox:
            return json.loads(
                json.dumps(shapely.geometry.mapping(shapely.geometry.box(*self.bbox)))
            )
        elif self.intersects:
            return self.intersects
        elif self.contains:
            return self.contains
        else:
            raise ValueError(
                """STACQuery does not contain any of the following query parameters:
                * bbox
                * intersects
                * contains."""
            )

    def set_param_if_not_exists(self, key, value):
        if not key in self.__dict__:
            self.__dict__[key] = value

    @classmethod
    def from_json(cls, json_data: str) -> STACQuery:
        return STACQuery.from_dict(json.loads(json_data))

    @classmethod
    def from_dict(
        cls, dict_data: dict, validator: Callable[[dict], bool] = lambda x: True
    ) -> STACQuery:

        if not validator(dict_data):
            raise ValueError(
                """Input Query did not pass validation. Please refer
                to the official block documentation or block specification."""
            )

        bbox: Optional[BoundingBox] = cast(BoundingBox, dict_data.get("bbox")) if not (
            dict_data.get("bbox") is None
            or dict_data.get("bbox") == ""
            or dict_data.get("bbox") == {}
        ) else None

        intersects: Optional[Geometry] = Geometry(
            **dict_data.get("intersects")
        ) if not (
            dict_data.get("intersects") is None
            or dict_data.get("intersects") == ""
            or dict_data.get("intersects") == {}
        ) else None

        contains: Optional[Geometry] = Geometry(**dict_data.get("contains")) if not (
            dict_data.get("contains") is None
            or dict_data.get("contains") == ""
            or dict_data.get("contains") == {}
        ) else None

        time: Optional[str] = dict_data.get("time") if not dict_data.get(
            "time"
        ) == "" else None
        limit: Optional[int] = dict_data.get("limit") if not dict_data.get(
            "limit"
        ) == "" else None
        ids: Optional[list] = dict_data.get("ids") if not dict_data.get(
            "ids"
        ) == "" else None
        time_series: Optional[list] = dict_data.get("time_series") if not dict_data.get(
            "time_series"
        ) == "" else None

        known_filters = [
            "bbox",
            "intersects",
            "contains",
            "time",
            "limit",
            "geometry",
            "bounds",
            "ids",
            "time_series",
        ]

        truncated_dict_data = {
            key: dict_data[key] for key in dict_data if key not in known_filters
        }

        return STACQuery(
            ids=ids,
            bbox=bbox,
            intersects=intersects,
            contains=contains,
            time=time,
            limit=limit,
            time_series=time_series,
            **truncated_dict_data,
        )

    @staticmethod
    def validate_datetime_str(string: Optional[str]):
        try:
            if string is None:
                return True
            elif len(str(string).split("/")) == 2:
                ciso8601.parse_datetime(str(string).split("/")[0])
                ciso8601.parse_datetime(str(string).split("/")[1])
            else:
                ciso8601.parse_datetime(str(string))
        except ValueError:
            return False
        return True
