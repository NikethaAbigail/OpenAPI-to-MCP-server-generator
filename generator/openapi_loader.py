"""
OpenAPI Specification Loader
Loads and validates OpenAPI specs from YAML/JSON files
"""

import logging
from pathlib import Path
import json
import yaml
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class OpenAPILoadError(Exception):
    """Custom exception for OpenAPI loading errors"""
    pass


def validate_openapi_version(spec: Dict[str, Any]) -> None:
    """
    Validate that the spec is a supported OpenAPI version
    Supports: OpenAPI 3.0.x, 3.1.x
    """
    if 'openapi' not in spec:
        raise OpenAPILoadError("Missing 'openapi' version field. This doesn't appear to be an OpenAPI 3.x specification.")

    version = spec['openapi']
    if not isinstance(version, str):
        raise OpenAPILoadError(f"OpenAPI version must be a string, got {type(version).__name__}")

    # Check major version
    if not version.startswith('3.'):
        raise OpenAPILoadError(f"Unsupported OpenAPI version: {version}. Only OpenAPI 3.x is supported.")

    logger.info(f"OpenAPI version {version} detected")


def load_openapi_spec(file_path: Path, max_size_mb: int = 50) -> Dict[str, Any]:
    """
    Load an OpenAPI spec from a Path object (.yaml, .yml, .json)

    Args:
        file_path: Path to the OpenAPI specification file
        max_size_mb: Maximum file size in MB (default: 50MB)

    Returns:
        Parsed OpenAPI specification as dictionary

    Raises:
        OpenAPILoadError: If file cannot be loaded or parsed
        TypeError: If file_path is not a Path object
    """
    # Validate input type
    if not isinstance(file_path, Path):
        raise TypeError(f"file_path must be a pathlib.Path, got {type(file_path).__name__}")

    # Check file exists
    if not file_path.exists():
        raise OpenAPILoadError(f"File not found: {file_path}")

    # Check file is readable
    if not file_path.is_file():
        raise OpenAPILoadError(f"Path is not a file: {file_path}")

    # Check file size
    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    if file_size_mb > max_size_mb:
        raise OpenAPILoadError(
            f"File too large: {file_size_mb:.2f}MB (max: {max_size_mb}MB). "
            f"Large specs may cause performance issues."
        )

    suffix = file_path.suffix.lower()

    try:
        if suffix in (".yaml", ".yml"):
            with open(file_path, "r", encoding="utf-8") as f:
                spec = yaml.safe_load(f)
        elif suffix == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                spec = json.load(f)
        else:
            raise OpenAPILoadError(
                f"Unsupported file type: {suffix}. "
                f"Supported types: .yaml, .yml, .json"
            )
    except yaml.YAMLError as e:
        raise OpenAPILoadError(f"Failed to parse YAML: {str(e)}") from e
    except json.JSONDecodeError as e:
        raise OpenAPILoadError(f"Failed to parse JSON: {str(e)}") from e
    except UnicodeDecodeError as e:
        raise OpenAPILoadError(f"File encoding error: {str(e)}. File must be UTF-8 encoded.") from e
    except IOError as e:
        raise OpenAPILoadError(f"Failed to read file: {str(e)}") from e

    # Validate it's a dict (basic sanity check)
    if not isinstance(spec, dict):
        raise OpenAPILoadError(
            f"Invalid OpenAPI spec: root must be an object/dict, got {type(spec).__name__}"
        )

    # Validate OpenAPI version
    try:
        validate_openapi_version(spec)
    except OpenAPILoadError:
        raise

    # Basic structure validation
    if 'paths' not in spec or not isinstance(spec['paths'], dict):
        raise OpenAPILoadError(
            "Invalid OpenAPI spec: missing or invalid 'paths' object. "
            "An OpenAPI spec must have a 'paths' object with API endpoints."
        )

    if not spec['paths']:
        raise OpenAPILoadError(
            "OpenAPI spec has no paths/endpoints defined. "
            "At least one path must be defined to generate tools."
        )

    logger.info(f"Loaded OpenAPI spec with {len(spec['paths'])} paths from {file_path.name}")

    return spec


def get_spec_info(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract metadata/info from OpenAPI spec

    Returns:
        Dictionary with spec metadata
    """
    info = spec.get('info', {})

    return {
        'title': info.get('title', 'Unknown API'),
        'version': info.get('version', '1.0.0'),
        'description': info.get('description', ''),
        'openapi_version': spec.get('openapi', 'unknown'),
        'num_paths': len(spec.get('paths', {})),
        'num_operations': sum(
            1 for path_item in spec.get('paths', {}).values()
            for key in path_item.keys()
            if key in ['get', 'post', 'put', 'patch', 'delete', 'head', 'options']
        ),
        'servers': [s.get('url') for s in spec.get('servers', [])],
    }


if __name__ == '__main__':
    print("OpenAPI Loader module - use as import")
    print("\nUsage:")
    print("  from openapi_loader import load_openapi_spec")
    print("  spec = load_openapi_spec(Path('openapi.yaml'))")