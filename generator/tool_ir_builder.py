"""
Enhanced OpenAPI Tool IR Builder - V2.2 (Fixed)
Adds support for:
- Response schemas
- Full parameter metadata
- Composition handling (oneOf, anyOf, allOf)
- Recursive required field detection
- Per-operation security schemes

Fixes applied:
- operationId starting with a digit → prefixed with 'op_' to produce valid Python identifier
- operationId that is empty after regex strip → falls back to 'op_{method}_{index}'
- operationId deduplication now applies AFTER the digit/empty guards (correct order)
"""

import logging
import re
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


def resolve_ref_object(
    obj: Dict[str, Any],
    openapi: Dict[str, Any],
    visited: Optional[Set[str]] = None,
    max_depth: int = 25
) -> Dict[str, Any]:
    """
    Resolve a generic OpenAPI object that may contain $ref.
    Protected against circular references.
    """
    if visited is None:
        visited = set()

    if max_depth <= 0 or not isinstance(obj, dict):
        return obj

    if '$ref' not in obj:
        return obj

    ref_path = obj['$ref']
    if ref_path in visited:
        return {'type': 'object', '_circular_reference': ref_path}

    referenced = load_json_pointer(openapi, ref_path)
    if not isinstance(referenced, dict):
        return obj

    visited.add(ref_path)
    resolved = resolve_ref_object(referenced, openapi, visited.copy(), max_depth - 1)
    visited.discard(ref_path)

    # Keep local overrides if present
    local_overrides = {k: v for k, v in obj.items() if k != '$ref'}
    if local_overrides:
        merged = dict(resolved)
        merged.update(local_overrides)
        return merged
    return resolved


def merge_operation_parameters(
    path_item: Dict[str, Any],
    operation: Dict[str, Any],
    openapi: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Merge path-level and operation-level parameters per OpenAPI rules.
    Operation-level parameters override path-level ones with the same name+location.
    """
    merged: Dict[str, Dict[str, Any]] = {}

    for raw_param in path_item.get('parameters', []):
        if not isinstance(raw_param, dict):
            continue
        param = resolve_ref_object(raw_param, openapi)
        name = param.get('name')
        location = param.get('in')
        if name and location:
            merged[f"{location}:{name}"] = param

    for raw_param in operation.get('parameters', []):
        if not isinstance(raw_param, dict):
            continue
        param = resolve_ref_object(raw_param, openapi)
        name = param.get('name')
        location = param.get('in')
        if name and location:
            merged[f"{location}:{name}"] = param

    return list(merged.values())


def load_json_pointer(data: Dict[str, Any], pointer: str) -> Any:
    """
    Resolve a JSON pointer like '#/components/schemas/User'
    """
    if not pointer.startswith('#/'):
        return None

    parts = pointer[2:].split('/')
    current = data
    for part in parts:
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


def _merge_allof(schemas: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge a list of allOf sub-schemas into a single unified schema.
    Merges properties, required fields, and preserves type.
    """
    merged: Dict[str, Any] = {'type': 'object', 'properties': {}, 'required': []}
    for sub in schemas:
        if not isinstance(sub, dict):
            continue
        if 'type' in sub and 'type' not in merged:
            merged['type'] = sub['type']
        if 'properties' in sub and isinstance(sub['properties'], dict):
            merged['properties'].update(sub['properties'])
        if 'required' in sub and isinstance(sub['required'], list):
            for r in sub['required']:
                if r not in merged['required']:
                    merged['required'].append(r)
        for k, v in sub.items():
            if k not in ('properties', 'required', 'type') and k not in merged:
                merged[k] = v
    if not merged['properties']:
        del merged['properties']
    if not merged['required']:
        del merged['required']
    return merged


def _deep_resolve_schema(
    schema: Dict[str, Any],
    openapi: Dict[str, Any],
    visited: Optional[Set[str]] = None,
    max_depth: int = 25
) -> Dict[str, Any]:
    """
    Recursively resolve $ref in schema with circular reference protection.
    Supports OpenAPI 3.1 type arrays and allOf merging.
    """
    if visited is None:
        visited = set()

    if max_depth <= 0:
        return schema

    if not isinstance(schema, dict):
        return schema

    # Resolve $ref
    if '$ref' in schema:
        ref_path = schema['$ref']

        if ref_path in visited:
            return {'type': 'object', '_circular_ref': ref_path}

        referenced = load_json_pointer(openapi, ref_path)
        if referenced:
            visited.add(ref_path)
            resolved = _deep_resolve_schema(referenced, openapi, visited.copy(), max_depth - 1)
            visited.discard(ref_path)

            if schema.get('nullable') is True:
                resolved = dict(resolved)
                resolved['nullable'] = True

            # Convert OpenAPI 3.0 nullable to 3.1 type array
            if resolved.get('nullable') is True and 'type' in resolved:
                current_type = resolved['type']
                if isinstance(current_type, str):
                    resolved['type'] = [current_type, 'null']
                elif isinstance(current_type, list) and 'null' not in current_type:
                    resolved['type'] = current_type + ['null']
                resolved.pop('nullable', None)

            return resolved
        return schema

    resolved = dict(schema)

    # Convert OpenAPI 3.0 nullable → 3.1 type array
    if resolved.get('nullable') is True and 'type' in resolved:
        current_type = resolved['type']
        if isinstance(current_type, str):
            resolved['type'] = [current_type, 'null']
        elif isinstance(current_type, list) and 'null' not in current_type:
            resolved['type'] = current_type + ['null']
        resolved.pop('nullable', None)

    # Handle allOf by merging into a single schema
    if 'allOf' in resolved and isinstance(resolved['allOf'], list):
        resolved_variants = [
            _deep_resolve_schema(item, openapi, visited.copy(), max_depth - 1)
            for item in resolved['allOf'][:15]
        ]
        merged = _merge_allof(resolved_variants)
        for k, v in resolved.items():
            if k != 'allOf' and k not in merged:
                merged[k] = v
        if 'properties' in resolved and isinstance(resolved.get('properties'), dict):
            merged.setdefault('properties', {}).update(resolved['properties'])
        resolved = merged

    # Resolve in properties (for objects)
    if 'properties' in resolved and isinstance(resolved['properties'], dict):
        resolved['properties'] = {
            k: _deep_resolve_schema(v, openapi, visited.copy(), max_depth - 1)
            for k, v in list(resolved['properties'].items())[:100]
        }

    # Resolve in items (for arrays)
    if 'items' in resolved:
        resolved['items'] = _deep_resolve_schema(
            resolved['items'], openapi, visited.copy(), max_depth - 1
        )

    # Resolve in oneOf/anyOf
    for keyword in ['oneOf', 'anyOf']:
        if keyword in resolved and isinstance(resolved[keyword], list):
            resolved[keyword] = [
                _deep_resolve_schema(item, openapi, visited.copy(), max_depth - 1)
                for item in resolved[keyword][:15]
            ]

    return resolved


def extract_parameter_metadata(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract all constraint metadata from parameter schema.
    """
    return {
        'type': schema.get('type'),
        'format': schema.get('format'),
        'default': schema.get('default'),
        'example': schema.get('example'),
        'examples': schema.get('examples'),
        'enum': schema.get('enum'),
        'pattern': schema.get('pattern'),
        'minLength': schema.get('minLength'),
        'maxLength': schema.get('maxLength'),
        'minimum': schema.get('minimum'),
        'maximum': schema.get('maximum'),
        'exclusiveMinimum': schema.get('exclusiveMinimum'),
        'exclusiveMaximum': schema.get('exclusiveMaximum'),
        'multipleOf': schema.get('multipleOf'),
        'description': schema.get('description'),
        'deprecated': schema.get('deprecated', False),
        'readOnly': schema.get('readOnly', False),
        'writeOnly': schema.get('writeOnly', False),
    }


def get_required_fields(
    schema: Dict[str, Any],
    openapi: Dict[str, Any],
    visited: Optional[Set[str]] = None,
    max_depth: int = 10
) -> List[str]:
    """
    Extract required fields from schema recursively with circular reference protection.
    """
    if visited is None:
        visited = set()

    if max_depth <= 0 or not isinstance(schema, dict):
        return []

    if '$ref' in schema:
        ref_path = schema['$ref']
        if ref_path in visited:
            return []
        visited.add(ref_path)
        referenced = load_json_pointer(openapi, ref_path)
        if referenced:
            return get_required_fields(referenced, openapi, visited.copy(), max_depth - 1)
        return []

    required = []

    if schema.get('type') == 'object':
        required.extend(schema.get('required', []))

        properties = schema.get('properties', {})
        for prop_name, prop_schema in properties.items():
            if isinstance(prop_schema, dict):
                prop_required = get_required_fields(
                    prop_schema, openapi, visited.copy(), max_depth - 1
                )
                for req_field in prop_required:
                    required.append(f"{prop_name}.{req_field}")

    return list(set(required))


def extract_response_schema(
    operation: Dict[str, Any],
    openapi: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Extract response schema from operation (tries 200, 201, default).
    Supports OpenAPI 3.0 and 3.1 with circular reference protection.
    """
    responses = operation.get('responses', {})

    for chosen_code in ['200', '201', 'default']:
        if chosen_code not in responses:
            continue

        response_obj = responses[chosen_code]
        content = response_obj.get('content', {})

        if not content:
            continue

        for content_type in ['application/json', 'application/xml', 'text/plain']:
            if content_type not in content:
                continue

            content_obj = content[content_type]
            if 'schema' not in content_obj:
                continue

            try:
                resolved = _deep_resolve_schema(
                    content_obj['schema'], openapi, set(), max_depth=15
                )
                required_fields = get_required_fields(resolved, openapi, set(), max_depth=5)

                return {
                    'status_code': chosen_code,
                    'content_type': content_type,
                    'schema': resolved,
                    'required_fields': required_fields,
                }
            except RecursionError:
                return {
                    'status_code': chosen_code,
                    'content_type': content_type,
                    'schema': {'type': 'object'},
                    'required_fields': [],
                }

    if '204' in responses:
        return {
            'status_code': '204',
            'content_type': None,
            'schema': {'type': 'null'},
            'required_fields': [],
        }

    return None


def extract_operation_security(
    operation: Dict[str, Any],
    openapi: Dict[str, Any]
) -> List[Dict[str, List[str]]]:
    """
    Extract security requirements for this operation.
    Falls back to global security if not operation-specific.
    """
    if 'security' in operation:
        security = operation['security']
        if security == []:
            return []
        if security and isinstance(security, list):
            return security

    return openapi.get('security', [])


def get_composition_info(schema: Dict[str, Any], openapi: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract composition information (oneOf, anyOf, allOf, discriminator)
    with circular reference protection.
    """
    composition: Dict[str, Any] = {
        'has_composition': False,
        'composition_type': None,
        'discriminator': None,
        'variants': [],
    }

    if not isinstance(schema, dict):
        return composition

    try:
        if 'oneOf' in schema:
            composition['has_composition'] = True
            composition['composition_type'] = 'oneOf'
            composition['variants'] = [
                _deep_resolve_schema(v, openapi, set(), max_depth=3)
                for v in schema['oneOf'][:15]
            ]
        elif 'anyOf' in schema:
            composition['has_composition'] = True
            composition['composition_type'] = 'anyOf'
            composition['variants'] = [
                _deep_resolve_schema(v, openapi, set(), max_depth=3)
                for v in schema['anyOf'][:15]
            ]
        elif 'allOf' in schema:
            composition['has_composition'] = True
            composition['composition_type'] = 'allOf'
            resolved_variants = [
                _deep_resolve_schema(v, openapi, set(), max_depth=3)
                for v in schema['allOf'][:15]
            ]
            composition['variants'] = resolved_variants
            composition['merged'] = _merge_allof(resolved_variants)
    except RecursionError:
        composition['has_composition'] = True
        composition['variants'] = []

    if 'discriminator' in schema:
        composition['discriminator'] = {
            'property_name': schema['discriminator'].get('propertyName'),
            'mapping': schema['discriminator'].get('mapping', {}),
        }

    return composition


def _sanitize_operation_id(raw: str, method: str, path: str, index: int) -> str:
    """
    Convert a raw operationId string into a valid Python identifier.

    Rules:
    1. Lowercase and strip all non-alphanumeric/underscore characters.
    2. If empty after strip → fall back to 'op_{method}_{index}'.
    3. If starts with a digit → prefix with 'op_'.
    """
    sanitized = re.sub(r'[^a-z0-9_]', '', raw.lower())

    if not sanitized:
        # e.g. operationId was "@#$" or completely empty
        safe_path = re.sub(r'[^a-z0-9]', '_', path.lower()).strip('_')
        sanitized = f"op_{method.lower()}_{safe_path}" if safe_path else f"op_{method.lower()}_{index}"

    if sanitized[0].isdigit():
        sanitized = f"op_{sanitized}"

    return sanitized


def build_tool_ir(openapi_spec: Dict[str, Any], namespace: str = 'default') -> List[Dict[str, Any]]:
    """
    Build Tool IR from OpenAPI spec.

    Fixes:
    - operationId starting with digit → prefixed with 'op_'
    - operationId empty after regex strip → safe fallback name
    - Deduplication applied after sanitization
    """
    tools = []
    seen_operation_ids: Dict[str, int] = {}
    op_index = 0

    paths = openapi_spec.get('paths', {})

    for path, path_item in paths.items():
        for method_name, operation in path_item.items():
            if method_name not in ['get', 'post', 'put', 'patch', 'delete', 'head', 'options']:
                continue

            if not isinstance(operation, dict):
                continue

            try:
                op_index += 1
                raw_operation_id = operation.get('operationId', f"{method_name}_{path}")

                # Fix: sanitize → guard digit-prefix/empty → then deduplicate
                operation_id = _sanitize_operation_id(
                    raw_operation_id, method_name, path, op_index
                )

                # Deduplicate
                if operation_id in seen_operation_ids:
                    seen_operation_ids[operation_id] += 1
                    operation_id = f"{operation_id}_{seen_operation_ids[operation_id]}"
                else:
                    seen_operation_ids[operation_id] = 1

                tool: Dict[str, Any] = {
                    'name': operation_id,
                    'operation_id': operation.get('operationId', operation_id),
                    'method': method_name.upper(),
                    'path': path,
                    'namespace': namespace,
                    'description': (
                        operation.get('description') or
                        operation.get('summary') or
                        operation_id
                    ),
                    'tags': operation.get('tags', []),
                    'deprecated': operation.get('deprecated', False),
                }

                # Extract parameters
                parameters = []
                param_names_seen: Set[str] = set()

                merged_parameters = merge_operation_parameters(path_item, operation, openapi_spec)
                for param in merged_parameters:
                    param_name = param.get('name', '')
                    if not param_name or param_name in param_names_seen:
                        continue
                    param_names_seen.add(param_name)

                    param_schema = param.get('schema', {})

                    param_obj = {
                        'name': param_name,
                        'original_name': param_name,
                        'required': param.get('required', False),
                        'is_secret': param_name.lower() in ['password', 'token', 'api_key', 'secret'],
                        'in': param.get('in', 'query'),
                        'type': param_schema.get('type', 'string'),
                        'schema': param_schema,
                        'metadata': extract_parameter_metadata(param_schema),
                    }
                    parameters.append(param_obj)

                tool['parameters'] = parameters

                # Extract request body
                request_body = resolve_ref_object(
                    operation.get('requestBody', {}), openapi_spec
                )
                content = request_body.get('content', {})
                tool['request_body'] = {
                    'required': request_body.get('required', False),
                    'content': content,
                    'default_content_type': (
                        list(content.keys())[0] if content else 'application/json'
                    ),
                    'supports_files': any('multipart' in ct for ct in content),
                }

                # Resolve request body schemas
                if tool['request_body']['content']:
                    for ct, content_obj in tool['request_body']['content'].items():
                        if 'schema' in content_obj:
                            try:
                                resolved = _deep_resolve_schema(
                                    content_obj['schema'], openapi_spec, set(), max_depth=20
                                )
                                content_obj['schema'] = resolved
                                composition = get_composition_info(resolved, openapi_spec)
                                tool['request_body']['composition'] = composition
                                tool['request_body']['required_fields'] = get_required_fields(
                                    resolved, openapi_spec, set(), max_depth=5
                                )
                                tool['request_body']['metadata'] = extract_parameter_metadata(resolved)
                            except (RecursionError, Exception) as e:
                                logger.warning(
                                    f"Could not fully resolve request body for {operation_id}: {e}"
                                )
                                content_obj['schema'] = {'type': 'object'}
                                tool['request_body']['composition'] = {'has_composition': False}
                                tool['request_body']['required_fields'] = []
                                tool['request_body']['metadata'] = {}

                # Extract response schema
                try:
                    tool['response_schema'] = extract_response_schema(operation, openapi_spec)
                except (RecursionError, Exception) as e:
                    logger.warning(f"Could not extract response schema for {operation_id}: {e}")
                    tool['response_schema'] = None

                # Extract operation-specific security
                tool['security_schemes'] = extract_operation_security(operation, openapi_spec)

                tools.append(tool)

            except Exception as e:
                logger.warning(f"Skipping operation {method_name.upper()} {path}: {e}")
                continue

    return tools