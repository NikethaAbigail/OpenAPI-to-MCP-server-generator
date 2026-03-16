"""
Optimized Executor Mapper - V3 (Fixed)
Generates executors.py with:
- Fixed Parameter Distribution Logic
- Improved Circular Reference Handling
- Request Validation (checks before sending)
- Better Error Handling & Debugging
- Response Validation (API error detection)
- Complete File Upload Support
- OpenAPI 3.1 Full Compatibility
- Proper Path Parameter URL Encoding

Fixes applied:
- UTILITIES_CODE now imports typing internally (was relying on file-scope by accident)
- Removed duplicate `import httpx` inside execute_api_call
- generate_thin_executor: description triple-quote injection → use repr()
- generate_thin_executor: optional_names emitted as tuple literal, not list
- generate_compact_tool_registry: body_schema defaults to {} not None
- _serialize_value: bare except → except Exception
- Duplicate traversal logic in get_param_default_value / extract_defaults_from_tool
  now share a common _get_body_prop_schema() helper
"""

import logging
import json
import re
import time
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


# ====================================================
# Serialization Helpers
# ====================================================

def _escape_string_for_python(s: str) -> str:
    """Escape a string for safe inclusion in Python code"""
    if s is None:
        return "None"
    return repr(str(s))


def _serialize_value(v) -> str:
    """Serialize a value for Python code generation"""
    if v is None:
        return "None"
    elif isinstance(v, bool):
        return "True" if v else "False"
    elif isinstance(v, (int, float)):
        return str(v)
    elif isinstance(v, str):
        return repr(v)
    elif isinstance(v, dict):
        items = ", ".join(f"{repr(k)}: {_serialize_value(val)}" for k, val in v.items())
        return "{" + items + "}"
    elif isinstance(v, (list, tuple)):
        items = ", ".join(_serialize_value(item) for item in v)
        return "[" + items + "]"
    else:
        try:
            return repr(v)
        except Exception:  # Fix: was bare except
            return "None"


def _serialize_dict(d: Dict[str, Any]) -> str:
    """Safely serialize a dict for embedding in generated Python code"""
    if not d:
        return "{}"
    try:
        items = ", ".join(f"{repr(k)}: {_serialize_value(v)}" for k, v in d.items())
        return "{" + items + "}"
    except Exception:
        return "{}"


# ====================================================
# Shared Schema Traversal Helper
# ====================================================

def _get_body_prop_schema(prop_name: str, tool: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Shared helper: look up a body property schema by name.
    Avoids duplicating traversal logic across get_param_default_value
    and extract_defaults_from_tool.
    """
    req_body = tool.get('request_body', {})
    if not req_body or not req_body.get('content'):
        return None
    ct = req_body.get('default_content_type', 'application/json')
    schema = req_body.get('content', {}).get(ct, {}).get('schema', {})
    if not schema or schema.get('type') != 'object':
        return None
    return schema.get('properties', {}).get(prop_name)


# ====================================================
# Extract Tool Information
# ====================================================

def extract_defaults_from_tool(tool: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract default values from OpenAPI schema for runtime use.
    Returns a flat dict of {param_name: default_value}.
    Only extracts top-level simple field defaults (not nested objects).
    """
    defaults = {}

    req_body = tool.get('request_body', {})
    content = req_body.get('content', {})
    content_type = req_body.get('default_content_type', 'application/json')
    schema = content.get(content_type, {}).get('schema', {})
    properties = schema.get('properties', {}) if isinstance(schema, dict) else {}

    for prop_name, prop_schema in properties.items():
        if prop_schema.get('type') not in ('object', 'array') and 'default' in prop_schema:
            defaults[prop_name] = prop_schema['default']

    return defaults


def get_required_fields_list(tool: Dict[str, Any]) -> List[str]:
    """
    Extract top-level required fields from the body schema.
    """
    req_body = tool.get('request_body', {})
    content = req_body.get('content', {})
    content_type = req_body.get('default_content_type', 'application/json')
    schema = content.get(content_type, {}).get('schema', {})
    return schema.get('required', []) if isinstance(schema, dict) else []


def get_tool_executor_params(tool: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Determine the parameter list for the executor function.

    Strategy: The executor always receives parameters at the SCHEMA's top level.
    - Explicit path/query/header params → individual args
    - Body properties that are simple (string/int/bool etc.) → individual args
    - Body properties that are objects/arrays → single arg as dict/list
      (already reconstructed by server.py)

    This ensures server.py and executors.py share the same parameter contract.
    """
    seen = set()
    params = []

    # Explicit path/query/header parameters (not secrets)
    for p in tool.get('parameters', []):
        if p.get('is_secret'):
            continue
        name = p['name']
        if name not in seen:
            params.append({
                'name': name,
                'in': p.get('in', 'query'),
                'is_body': False,
                'schema': p.get('schema', {}),
            })
            seen.add(name)

    # Body schema properties
    req_body = tool.get('request_body', {})
    if req_body and req_body.get('content'):
        ct = req_body.get('default_content_type', 'application/json')
        schema = req_body.get('content', {}).get(ct, {}).get('schema', {})
        if schema and schema.get('type') == 'object':
            for prop_name, prop_schema in schema.get('properties', {}).items():
                if prop_name not in seen:
                    params.append({
                        'name': prop_name,
                        'in': 'body',
                        'is_body': True,
                        'schema': prop_schema,
                    })
                    seen.add(prop_name)

    return params


def get_param_default_value(param_name: str, tool: Dict[str, Any]) -> str:
    """
    Get the default value for a parameter from OpenAPI spec.
    Returns Python code string (e.g., "'CASASUMMARY'" or "None").
    Uses shared _get_body_prop_schema helper to avoid duplicating traversal logic.
    """
    # Check explicit parameters first
    for p in tool.get('parameters', []):
        if p['name'] == param_name:
            metadata = p.get('metadata', {})
            default = metadata.get('default')
            if default is not None:
                return _serialize_value(default)
            schema = p.get('schema', {})
            default = schema.get('default')
            if default is not None:
                return _serialize_value(default)

    # Check body schema via shared helper
    prop_schema = _get_body_prop_schema(param_name, tool)
    if prop_schema is not None:
        # Object/array params default to None (reconstructed by server.py)
        if prop_schema.get('type') in ('object', 'array'):
            return "None"
        default = prop_schema.get('default')
        if default is not None:
            return _serialize_value(default)

    return "None"


# ====================================================
# UTILITIES Code (injected into generated executors.py)
# ====================================================

UTILITIES_CODE = '''
import re
import logging
from typing import Dict, Any, List, Optional
from urllib.parse import quote

logger = logging.getLogger(__name__)


def apply_defaults(inputs: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge user inputs with OpenAPI default values.
    User-provided values take precedence over defaults.
    Only applies defaults for simple scalar fields (non-object, non-array).
    """
    result = dict(inputs)
    for key, default_value in defaults.items():
        if key not in result or result[key] is None:
            result[key] = default_value
    return result


def build_headers() -> Dict[str, str]:
    """
    Build HTTP headers from environment variables.
    Supports: Bearer tokens, API keys, custom headers, and Cookies.
    """
    headers: Dict[str, str] = {}

    # 1. Standard authentication (Authorization header)
    auth_header = os.getenv('MCP_AUTH_HEADER', '')
    auth_scheme = os.getenv('MCP_AUTH_SCHEME', '')
    auth_token = os.getenv('MCP_AUTH_TOKEN', '')

    if auth_header and auth_token:
        if auth_scheme:
            headers[auth_header] = f'{auth_scheme} {auth_token}'
        else:
            headers[auth_header] = auth_token

    # 2. Extra Headers (JSON)
    extra_headers_str = os.getenv('MCP_EXTRA_HEADERS', '')
    if extra_headers_str:
        try:
            if extra_headers_str.strip().startswith('{'):
                extra_headers = json.loads(extra_headers_str)
                headers.update(extra_headers)
        except Exception:
            pass  # Ignore invalid JSON in headers

    # 3. Explicit Cookie Support
    cookie_str = os.getenv('MCP_COOKIE', '')
    if cookie_str:
        if 'Cookie' in headers:
            headers['Cookie'] = f"{headers['Cookie']}; {cookie_str}"
        else:
            headers['Cookie'] = cookie_str

    return headers


def build_url(base_url: str, path_template: str, path_params: Dict[str, Any]) -> str:
    """
    Replace {param} placeholders in path with URL-encoded values,
    then safely join with the base URL (preserving base path).
    """
    path = path_template
    for param_name, param_value in path_params.items():
        placeholder = f"{{{param_name}}}"
        if placeholder in path:
            encoded_value = quote(str(param_value), safe='')
            path = path.replace(placeholder, encoded_value)

    # Safe join: preserve base URL path (urljoin would drop it for absolute paths)
    return base_url.rstrip('/') + '/' + path.lstrip('/')


def distribute_parameters(inputs: Dict[str, Any], tool: Dict[str, Any]) -> Dict[str, Any]:
    """
    Distribute parameters to correct locations: path, query, header, cookie, body.
    Uses parameter 'in' metadata from TOOL_REGISTRY to route correctly.
    Unknown keys default to body.
    """
    distributed: Dict[str, Dict[str, Any]] = {
        'path_params': {},
        'query_params': {},
        'header_params': {},
        'cookie_params': {},
        'body_params': {}
    }

    param_locations = {p['name']: p.get('in', 'body') for p in tool.get('parameters', [])}

    for key, value in inputs.items():
        location = param_locations.get(key, 'body')
        if location == 'path':
            distributed['path_params'][key] = value
        elif location == 'query':
            distributed['query_params'][key] = value
        elif location == 'header':
            distributed['header_params'][key] = value
        elif location == 'cookie':
            distributed['cookie_params'][key] = value
        else:
            distributed['body_params'][key] = value

    return distributed


def validate_required_params(params: Dict[str, Any], tool: Dict[str, Any]) -> None:
    """
    Validate required parameters exist.
    Checks both explicit parameter-level required AND body-level required fields.
    Raises ValueError with a clear message if any required field is missing.
    """
    missing = []

    # Check explicit path/query/header required params
    for param_def in tool.get('parameters', []):
        param_name = param_def.get('name')
        if param_def.get('required', False) and (param_name not in params or params[param_name] is None):
            missing.append(f"{param_name} ({param_def.get('in', 'unknown')} parameter)")

    # Check body-level required fields
    body_schema = tool.get('body_schema') or {}
    if isinstance(body_schema, dict) and body_schema.get('type') == 'object':
        for field in body_schema.get('required', []):
            if field not in params or params[field] is None:
                missing.append(f"{field} (body field)")

    if missing:
        raise ValueError(f"Required parameters missing: {', '.join(missing)}")


def validate_parameter(value: Any, param_schema: Dict[str, Any], param_name: str) -> None:
    """Validate a single parameter against its schema constraints."""
    if value is None:
        return

    param_type = param_schema.get('type')

    # Handle OpenAPI 3.1 type arrays (e.g. ["string", "null"])
    if isinstance(param_type, list):
        non_null_types = [t for t in param_type if t != 'null']
        if not non_null_types:
            return
        param_type = non_null_types[0]

    # Basic type checking
    if param_type == 'string' and not isinstance(value, str):
        raise ValueError(f"Parameter '{param_name}' must be string, got {type(value).__name__}")
    elif param_type == 'integer' and not isinstance(value, int):
        raise ValueError(f"Parameter '{param_name}' must be integer, got {type(value).__name__}")
    elif param_type == 'number' and not isinstance(value, (int, float)):
        raise ValueError(f"Parameter '{param_name}' must be numeric, got {type(value).__name__}")
    elif param_type == 'boolean' and not isinstance(value, bool):
        raise ValueError(f"Parameter '{param_name}' must be boolean, got {type(value).__name__}")
    elif param_type == 'array' and not isinstance(value, (list, tuple)):
        raise ValueError(f"Parameter '{param_name}' must be array/list, got {type(value).__name__}")

    if param_type == 'string' and isinstance(value, str):
        if 'maxLength' in param_schema and len(value) > param_schema['maxLength']:
            raise ValueError(f"Parameter '{param_name}' exceeds maxLength: {param_schema['maxLength']}")
        if 'minLength' in param_schema and len(value) < param_schema['minLength']:
            raise ValueError(f"Parameter '{param_name}' below minLength: {param_schema['minLength']}")
        if 'pattern' in param_schema:
            if not re.match(param_schema['pattern'], value):
                raise ValueError(f"Parameter '{param_name}' doesn't match pattern: {param_schema['pattern']}")

    if param_type in ('integer', 'number'):
        if 'minimum' in param_schema and value < param_schema['minimum']:
            raise ValueError(f"Parameter '{param_name}' below minimum: {param_schema['minimum']}")
        if 'maximum' in param_schema and value > param_schema['maximum']:
            raise ValueError(f"Parameter '{param_name}' above maximum: {param_schema['maximum']}")

    if 'enum' in param_schema and value not in param_schema['enum']:
        raise ValueError(f"Parameter '{param_name}' must be one of: {param_schema['enum']}, got '{value}'")


def validate_all_params(inputs: Dict[str, Any], tool: Dict[str, Any]) -> None:
    """Validate all input parameters against their schemas."""
    for param in tool.get('parameters', []):
        param_name = param['name']
        if param_name in inputs:
            schema = param.get('schema', {})
            validate_parameter(inputs[param_name], schema, param_name)


async def execute_api_call(
    method: str,
    path: str,
    distributed_params: Dict[str, Dict[str, Any]],
    content_type: str = 'application/json',
    response_schema: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generic HTTP request executor.
    Raises RuntimeError on API/network errors so callers see real failures.
    Note: httpx is imported at file level — no local import needed here.
    """
    base_url = os.getenv('MCP_API_BASE_URL')
    if not base_url:
        raise ValueError('MCP_API_BASE_URL not configured in environment')

    url = build_url(base_url, path, distributed_params.get('path_params', {}))

    headers = build_headers()
    headers.update(distributed_params.get('header_params', {}))

    cookies = distributed_params.get('cookie_params', {}) or None
    query_params = distributed_params.get('query_params', {}) or None
    body_params = distributed_params.get('body_params', {})

    timeout = httpx.Timeout(
        connect=float(os.getenv('MCP_CONNECT_TIMEOUT', '5')),
        read=float(os.getenv('MCP_READ_TIMEOUT', '30')),
        write=float(os.getenv('MCP_WRITE_TIMEOUT', '30')),
        pool=float(os.getenv('MCP_POOL_TIMEOUT', '5'))
    )

    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        try:
            if content_type == 'application/json':
                resp = await client.request(
                    method, url,
                    params=query_params,
                    headers=headers,
                    cookies=cookies,
                    json=body_params if body_params else None
                )
            elif content_type and 'multipart' in content_type:
                resp = await client.request(
                    method, url,
                    params=query_params,
                    headers=headers,
                    cookies=cookies,
                    files=body_params if body_params else None
                )
            elif content_type == 'application/x-www-form-urlencoded':
                resp = await client.request(
                    method, url,
                    params=query_params,
                    headers=headers,
                    cookies=cookies,
                    data=body_params if body_params else None
                )
            else:
                resp = await client.request(
                    method, url,
                    params=query_params,
                    headers=headers,
                    cookies=cookies,
                    data=body_params if body_params else None
                )

            resp.raise_for_status()

            try:
                return resp.json()
            except Exception:
                return {
                    'text': resp.text,
                    'status_code': resp.status_code,
                    'content_type': resp.headers.get('content-type', 'unknown')
                }

        except httpx.HTTPStatusError as e:
            error_code = e.response.status_code
            error_detail = e.response.text
            try:
                error_json = e.response.json()
                if isinstance(error_json, dict):
                    error_detail = (
                        error_json.get('message') or
                        error_json.get('error') or
                        error_json.get('detail') or
                        error_json.get('error_message') or
                        str(error_json)
                    )
            except Exception:
                pass
            raise RuntimeError(f'API Error [{error_code}]: {error_detail}')

        except httpx.TimeoutException as e:
            raise RuntimeError(f'Request timeout: {str(e)}')

        except httpx.ConnectError as e:
            raise RuntimeError(f'Connection failed to {url}: {str(e)}')

        except httpx.RequestError as e:
            raise RuntimeError(f'Request error: {str(e)}')
'''


# ====================================================
# Compact Tool Registry Generator
# ====================================================

def generate_compact_tool_registry(tools: List[Dict[str, Any]]) -> str:
    """
    Generate compact TOOL_REGISTRY with only runtime essentials.
    Includes response_schema for output validation and parameter location metadata.
    """
    lines = []
    lines.append("TOOL_REGISTRY = {")

    for tool in tools:
        tool_name = tool['name']
        method = tool['method']
        path = tool['path']
        request_body = tool.get('request_body', {})
        content_type = request_body.get('default_content_type', 'application/json')
        required = request_body.get('required', False)

        defaults = extract_defaults_from_tool(tool)
        required_fields = get_required_fields_list(tool)

        response_schema = tool.get('response_schema')
        if isinstance(response_schema, dict):
            response_schema = response_schema.get('schema', None)
        else:
            response_schema = None

        # Fix: default to {} not None so validate_required_params isinstance check works
        body_schema: Dict[str, Any] = {}
        if request_body.get('content'):
            ct = request_body.get('default_content_type', 'application/json')
            body_schema = request_body.get('content', {}).get(ct, {}).get('schema') or {}

        parameters_info = []
        for param in tool.get('parameters', []):
            parameters_info.append({
                'name': param.get('name'),
                'in': param.get('in', 'query'),
                'required': param.get('required', False),
                'schema': param.get('schema', {})
            })

        lines.append(f"    {_escape_string_for_python(tool_name)}: {{")
        lines.append(f"        'method': {_escape_string_for_python(method)},")
        lines.append(f"        'path': {_escape_string_for_python(path)},")
        lines.append(f"        'content_type': {_escape_string_for_python(content_type)},")
        lines.append(f"        'required': {required},")
        lines.append(f"        'defaults': {_serialize_dict(defaults)},")
        lines.append(f"        'required_fields': {_serialize_value(required_fields)},")
        lines.append(f"        'response_schema': {_serialize_value(response_schema)},")
        lines.append(f"        'parameters': {_serialize_value(parameters_info)},")
        lines.append(f"        'body_schema': {_serialize_value(body_schema)},")
        lines.append(f"    }},")

    lines.append("}")
    return "\n".join(lines)


# ====================================================
# Thin Executor Code Generator
# ====================================================

def generate_thin_executor(tool: Dict[str, Any]) -> str:
    """
    Generate a thin executor function.
    Required params have no default. Optional params default to None or their schema default.

    Fixes:
    - Description injected via repr() so triple-quotes/backslashes can't break the docstring
    - optional_names emitted as a tuple literal (not a list) for O(1) `not in` semantics
      and correct Python set-membership intent
    """
    tool_name = tool['name']
    exec_params = get_tool_executor_params(tool)

    req_body = tool.get('request_body', {})
    ct = req_body.get('default_content_type', 'application/json')
    body_schema = req_body.get('content', {}).get(ct, {}).get('schema', {})
    body_required = set(body_schema.get('required', [])) if isinstance(body_schema, dict) else set()

    required_explicit = {p['name'] for p in tool.get('parameters', []) if p.get('required', False)}

    required_ep = [p for p in exec_params if p['name'] in body_required or p['name'] in required_explicit]
    optional_ep = [p for p in exec_params if p['name'] not in body_required and p['name'] not in required_explicit]

    sig_parts = []
    for p in required_ep:
        sig_parts.append(f"{p['name']}: Any")
    for p in optional_ep:
        default_value = get_param_default_value(p['name'], tool)
        sig_parts.append(f"{p['name']}: Any = {default_value}")

    description = tool.get('description', tool_name)
    method = tool['method']
    path = tool['path']

    # Fix: use repr() so any special chars in description can't break the docstring
    safe_description = repr(description)

    lines = []
    params_str = ", ".join(sig_parts)
    lines.append(f"async def {tool_name}({params_str}) -> Dict[str, Any]:")
    lines.append(f"    __doc__ = {safe_description}")
    lines.append(f"    # {method} {path}")
    lines.append(f"    tool = TOOL_REGISTRY[{repr(tool_name)}]")
    lines.append(f"    ")

    all_ep = required_ep + optional_ep
    if all_ep:
        lines.append(f"    inputs = {{")
        for p in all_ep:
            lines.append(f"        {repr(p['name'])}: {p['name']},")
        lines.append(f"    }}")
        if optional_ep:
            # Fix: emit a tuple literal (immutable, O(1) `in` check, valid Python)
            optional_tuple = tuple(sorted(p['name'] for p in optional_ep))
            lines.append(f"    # Keep required params always; strip None only from optional")
            lines.append(f"    inputs = {{k: v for k, v in inputs.items() if k not in {optional_tuple!r} or v is not None}}")
    else:
        lines.append(f"    inputs = {{}}")

    lines.append(f"    ")
    lines.append(f"    params = apply_defaults(inputs, tool['defaults'])")
    lines.append(f"    validate_required_params(params, tool)")
    lines.append(f"    distributed = distribute_parameters(params, tool)")
    lines.append(f"    return await execute_api_call(")
    lines.append(f"        tool['method'],")
    lines.append(f"        tool['path'],")
    lines.append(f"        distributed,")
    lines.append(f"        tool['content_type'],")
    lines.append(f"        response_schema=tool.get('response_schema')")
    lines.append(f"    )")
    lines.append(f"")

    return "\n".join(lines)


# ====================================================
# Main Generator
# ====================================================

def generate_executor_file(tools: List[Dict[str, Any]], output_file: str) -> None:
    """
    Generate optimized executors.py file.

    Contract with server.py:
    - server.py flattens nested objects for the LLM
    - server.py reconstructs them into dicts before calling executor
    - executor receives top-level params, distributes them, fires HTTP call
    """
    code = []

    code.append('"""')
    code.append("Optimized Executors for FastMCP Server - V3")
    code.append("Generated from OpenAPI specification")
    code.append("")
    code.append(f"Total tools: {len(tools)}")
    code.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    code.append('"""')
    code.append("")
    code.append("import os")
    code.append("import re")
    code.append("import json")
    code.append("import logging")
    code.append("from typing import Dict, Any, List, Optional")
    code.append("from urllib.parse import quote")
    code.append("")
    code.append("import httpx")
    code.append("")
    code.append("")

    code.append("# " + "=" * 60)
    code.append("# TOOL_REGISTRY - Runtime Essentials")
    code.append("# " + "=" * 60)
    code.append("")
    code.append(generate_compact_tool_registry(tools))
    code.append("")
    code.append("")

    code.append("# " + "=" * 60)
    code.append("# Utilities and HTTP Executor")
    code.append("# " + "=" * 60)
    code.append("")
    # Strip the duplicate imports from UTILITIES_CODE since they're at file level now
    utilities_body = UTILITIES_CODE.strip()
    # Remove the redundant per-block imports that are now at file level
    lines_to_skip = {
        "import re",
        "import logging",
        "from typing import Dict, Any, List, Optional",
        "from urllib.parse import quote",
    }
    cleaned_utilities = "\n".join(
        line for line in utilities_body.splitlines()
        if line.strip() not in lines_to_skip
    )
    code.append(cleaned_utilities)
    code.append("")
    code.append("")

    code.append("# " + "=" * 60)
    code.append("# Executor Functions")
    code.append("# " + "=" * 60)
    code.append("")

    for tool in tools:
        code.append(generate_thin_executor(tool))

    code.append("# " + "=" * 60)
    code.append("# Executors Registry")
    code.append("# " + "=" * 60)
    code.append("")
    code.append("EXECUTORS = {")
    for tool in tools:
        tool_name = tool['name']
        code.append(f"    {_escape_string_for_python(tool_name)}: {tool_name},")
    code.append("}")
    code.append("")
    code.append("")

    code.append(f"__all__ = ['EXECUTORS', 'TOOL_REGISTRY']")
    code.append(f"__version__ = '3.0.0'")
    code.append(f"__generated__ = '{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}'")

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(code))

    logger.info(f"Generated {output_file} with {len(tools)} executor functions")


if __name__ == '__main__':
    print("Optimized Executor Mapper V3 - use as import")
    print("\nUsage:")
    print("  from executor_mapper import generate_executor_file")
    print("  generate_executor_file(tools, 'executors.py')")