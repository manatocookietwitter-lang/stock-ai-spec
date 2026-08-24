"""Point-in-time and J-Quants V2 data-foundation APIs."""

from stock_ai.data.contracts import (
    BulkFileDescriptor,
    Capability,
    CapabilityStatus,
    DatasetName,
    HistoricalRevisionPolicy,
    HistorySyncResult,
    IngestionResult,
    ObjectKind,
    QualityReport,
    StoredObject,
    SubscriptionPlan,
    capabilities_for,
)
from stock_ai.data.history import JQuantsV2HistoryIngestor
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
from stock_ai.data.production import (
    ProductionDataBundle,
    build_point_in_time_universe,
    build_production_data,
)
from stock_ai.data.quality import DataQualityError, require_quality, validate_rows
from stock_ai.data.storage import DuckDBCatalog, ImmutableParquetStore, StorageIntegrityError

__all__ = [
    "ALL_DATASETS",
    "DEFAULT_DATASETS",
    "BulkFileDescriptor",
    "Capability",
    "CapabilityStatus",
    "DataAvailabilityError",
    "DataQualityError",
    "DatasetName",
    "DuckDBCatalog",
    "HistoricalRevisionPolicy",
    "HistorySyncResult",
    "ImmutableParquetStore",
    "IngestionResult",
    "JQuantsCredentialError",
    "JQuantsError",
    "JQuantsPlanError",
    "JQuantsRequestError",
    "JQuantsSchemaError",
    "JQuantsV2Client",
    "JQuantsV2Config",
    "JQuantsV2HistoryIngestor",
    "JQuantsV2Ingestor",
    "ObjectKind",
    "ProductionDataBundle",
    "QualityReport",
    "StorageIntegrityError",
    "StoredObject",
    "SubscriptionPlan",
    "assert_point_in_time",
    "build_point_in_time_universe",
    "build_production_data",
    "capabilities_for",
    "point_in_time_view",
    "require_quality",
    "validate_rows",
]
