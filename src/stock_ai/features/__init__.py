"""Point-in-time feature computation and versioned manifests."""

from stock_ai.features.catalog import (
    FEATURE_REGISTRY,
    V0_MANIFEST,
    V1_CORE_MANIFEST,
    V2_EXTENDED_MANIFEST,
)
from stock_ai.features.engine import FeatureEngine
from stock_ai.features.morning import (
    MORNING_CORE_MANIFEST,
    MORNING_FEATURE_REGISTRY,
    MORNING_MICROSTRUCTURE_MANIFEST,
    MorningFeatureOutput,
    build_morning_features,
    morning_feature_manifest,
    morning_feature_manifest_for_capabilities,
)
from stock_ai.features.registry import FeatureDefinition, FeatureRegistry, FeatureSetManifest

__all__ = [
    "FEATURE_REGISTRY",
    "MORNING_CORE_MANIFEST",
    "MORNING_FEATURE_REGISTRY",
    "MORNING_MICROSTRUCTURE_MANIFEST",
    "V0_MANIFEST",
    "V1_CORE_MANIFEST",
    "V2_EXTENDED_MANIFEST",
    "FeatureDefinition",
    "FeatureEngine",
    "FeatureRegistry",
    "FeatureSetManifest",
    "MorningFeatureOutput",
    "build_morning_features",
    "morning_feature_manifest",
    "morning_feature_manifest_for_capabilities",
]
