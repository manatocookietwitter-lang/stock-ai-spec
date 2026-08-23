"""Point-in-time and J-Quants V2 data-foundation APIs."""

from stock_ai.data.contracts import (
    Capability,
    CapabilityStatus,
    DatasetName,
    IngestionResult,
    ObjectKind,
    QualityReport,
    StoredObject,
    SubscriptionPlan,
    capabilities_for,
)
from stock_ai.data.jquants_v2 import (
    JQuantsCredentialError,
    JQuantsError,
    JQuantsPlanError,
    JQuantsRequestError,
    JQuantsSchemaError,
    JQuantsV2Client,
    JQuantsV2Config,
)
from stock_ai.data.pipeline import ALL_DATASETS, DEFAULT_DATASETS, JQuantsV2Ingestor
from stock_ai.data.point_in_time import (
    DataAvailabilityError,
    assert_point_in_time,
    point_in_time_view,
)
from stock_ai.data.quality import DataQualityError, require_quality, validate_rows
from stock_ai.data.storage import DuckDBCatalog, ImmutableParquetStore, StorageIntegrityError

__all__ = [
    "ALL_DATASETS",
    "DEFAULT_DATASETS",
    "Capability",
    "CapabilityStatus",
    "DataAvailabilityError",
    "DataQualityError",
    "DatasetName",
    "DuckDBCatalog",
    "ImmutableParquetStore",
    "IngestionResult",
    "JQuantsCredentialError",
    "JQuantsError",
    "JQuantsPlanError",
    "JQuantsRequestError",
    "JQuantsSchemaError",
    "JQuantsV2Client",
    "JQuantsV2Config",
    "JQuantsV2Ingestor",
    "ObjectKind",
    "QualityReport",
    "StorageIntegrityError",
    "StoredObject",
    "SubscriptionPlan",
    "assert_point_in_time",
    "capabilities_for",
    "point_in_time_view",
    "require_quality",
    "validate_rows",
]
