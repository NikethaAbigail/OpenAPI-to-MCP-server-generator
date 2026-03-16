"""
Optimized Python SDK Code Generator - V2.3 (Fixed)
Generates simplified FastMCP server.py with:
- Thin tool wrappers
- OpenAPI defaults in function signatures
- Minimal docstrings
- Simplified error handling
- Smart Flattening with parent-prefixing for shared parameters

Fixes applied:
- Description escaping: now uses repr() instead of manual replace()
  so backslashes, triple-quotes, and other special chars cannot break generated code
- fastmcp version pin loosened to >=2.0 in generated requirements.txt
  (0.2.0 had an incompatible API for app.run())
- get_param_type_hint: non-null type extraction extracted into helper to avoid duplication
"""

import logging
import json
import time
from pathlib import Path
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


# ====================================================
# Type Resolution Helper
# ====================================================

def _resolve_openapi_type(param_type: Any) -> str:
    """
    Normalise an OpenAPI type value (string or 3.1 array) to a single non-null type string.
    Returns the first non-null type, or 'string' if all are null / empty.
    """
    if isinstance(param_type, list):
        non_null = [t for t in param_type if t != 'null']
        param_type = non_null[0] if non_null else 'string'
    return param_type or 'string'


def _type_to_hint(param_type: str) -> str:
    """Map an OpenAPI scalar type string to a Python type hint string."""
    mapping = {
        'string': 'str',
        'integer': 'int',
        'number': 'float',
        'boolean': 'bool',
        'array': 'List',
        'object': 'Dict',
    }
    return mapping.get(param_type, 'Any')


# ====================================================
# Parameter Helpers
# ====================================================

def get_param_default_value(param_name: str, tool: Dict[str, Any]) -> str:
    """
    Get the default value for a parameter from OpenAPI spec.
    Returns Python code string (e.g., "'CASASUMMARY'" or "None").
    """
    # Check explicit parameters first
    for p in tool.get('parameters', []):
        if p['name'] == param_name:
            metadata = p.get('metadata', {})
            default = metadata.get('default')
            if default is not None:
                return repr(default)
            schema = p.get('schema', {})
            default = schema.get('default')
            if default is not None:
                return repr(default)

    # Check body schema
    req_body = tool.get('request_body', {})
    if req_body and req_body.get('content'):
        ct = req_body.get('default_content_type', 'application/json')
        schema = req_body.get('content', {}).get(ct, {}).get('schema', {})
        if schema and schema.get('type') == 'object':
            properties = schema.get('properties', {})
            if param_name in properties:
                prop_schema = properties[param_name]
                if prop_schema.get('type') in ('object', 'array'):
                    return "None"
                default = prop_schema.get('default')
                if default is not None:
                    return repr(default)

    return "None"


def get_param_type_hint(
    param_name: str,
    tool: Dict[str, Any],
    schema_override: Dict = None
) -> str:
    """
    Get type hint for a parameter.
    Returns: 'str', 'int', 'bool', 'Dict', 'List', 'Any'
    Uses _resolve_openapi_type helper to avoid duplicating null-stripping logic.
    """
    if schema_override:
        raw_type = schema_override.get('type', 'string')
        return _type_to_hint(_resolve_openapi_type(raw_type))

    # Check explicit parameters
    for p in tool.get('parameters', []):
        if p['name'] == param_name:
            raw_type = p.get('schema', {}).get('type', 'string')
            return _type_to_hint(_resolve_openapi_type(raw_type))

    # Check body schema
    req_body = tool.get('request_body', {})
    if req_body and req_body.get('content'):
        ct = req_body.get('default_content_type', 'application/json')
        schema = req_body.get('content', {}).get(ct, {}).get('schema', {})
        if schema and schema.get('type') == 'object':
            properties = schema.get('properties', {})
            if param_name in properties:
                raw_type = properties[param_name].get('type', 'string')
                return _type_to_hint(_resolve_openapi_type(raw_type))

    return 'Any'


def _collect_required_flat_names(schema: Dict[str, Any]) -> set:
    """
    Walk the body schema and collect every flat parameter name that is required.

    Rules:
    - Top-level required[] simple scalars → required as-is
    - Top-level required[] objects → their nested required children are required
      (using parent_child flat names)
    - Explicit path/query params with required=True are handled separately

    Returns a set of flat parameter names that must NOT have a default value.
    """
    required_flat: set = set()

    if not isinstance(schema, dict) or schema.get('type') != 'object':
        return required_flat

    top_required = set(schema.get('required', []))
    properties = schema.get('properties', {})

    for prop_name, prop_schema in properties.items():
        if prop_schema.get('type') == 'object' and 'properties' in prop_schema:
            sub_props = prop_schema.get('properties', {})
            sub_required = set(prop_schema.get('required', []))
            parent_is_required = prop_name in top_required

            for sub_name in sub_props:
                prefixed = f"{prop_name}_{sub_name}"
                child_schema = sub_props[sub_name]
                child_has_default = (
                    'default' in child_schema
                    and child_schema.get('type') not in ('object', 'array')
                )
                if parent_is_required and sub_name in sub_required and not child_has_default:
                    required_flat.add(prefixed)
        else:
            if prop_name in top_required:
                prop_has_default = (
                    'default' in prop_schema
                    and prop_schema.get('type') not in ('object', 'array')
                )
                if not prop_has_default:
                    required_flat.add(prop_name)

    return required_flat


# ====================================================
# Tool Wrapper Generator
# ====================================================

def generate_tool_wrapper(tool: Dict[str, Any]) -> str:
    """
    Generate a smart tool wrapper that FLATTENS nested objects for the LLM
    but reconstructs them for the API.

    Flattening contract:
    - Explicit path/query/header params → individual args (passed straight to executor)
    - Body simple fields → individual args (passed straight to executor)
    - Body nested objects (type: object with properties) → flattened as parent_child args,
      reconstructed into {child: value} dict before calling executor
    - Body arrays → single arg as List (not flattened)

    Fix: description is embedded via repr() so any special characters
    (backslashes, triple-quotes, etc.) cannot produce a SyntaxError in the generated file.
    """
    tool_name = tool['name']
    description = tool.get('description', tool_name)
    method = tool['method']
    path = tool['path']

    req_body = tool.get('request_body', {})
    content = req_body.get('content', {})
    ct = req_body.get('default_content_type', 'application/json')
    schema = content.get(ct, {}).get('schema', {})

    required_flat_names = _collect_required_flat_names(schema)
    required_param_names = {
        p['name'] for p in tool.get('parameters', [])
        if p.get('required', False) and not p.get('is_secret')
    }

    flat_params = []
    seen_names: set = set()

    # Explicit path/query/header params (not secrets)
    for param in tool.get('parameters', []):
        if param.get('is_secret'):
            continue
        name = param['name']
        if name not in seen_names:
            is_req = name in required_param_names
            flat_params.append({
                'name': name,
                'type': get_param_type_hint(name, tool),
                'default': None if is_req else get_param_default_value(name, tool),
                'required': is_req,
                'source': 'param',
                'executor_key': name,
            })
            seen_names.add(name)

    # body_reconstruction: parent_name → {original_child_name: prefixed_flat_name}
    body_reconstruction: Dict[str, Dict[str, str]] = {}

    if schema.get('type') == 'object':
        properties = schema.get('properties', {})
        for prop_name, prop_schema in properties.items():
            if prop_schema.get('type') == 'object' and 'properties' in prop_schema:
                # Nested object: flatten children with parent prefix
                sub_props = prop_schema.get('properties', {})
                child_mapping: Dict[str, str] = {}

                for sub_name, sub_schema in sub_props.items():
                    prefixed_name = f"{prop_name}_{sub_name}"
                    child_mapping[sub_name] = prefixed_name

                    if prefixed_name in seen_names:
                        continue

                    is_req = prefixed_name in required_flat_names

                    if is_req:
                        default_val = None
                    elif (
                        sub_schema.get('type') not in ('object', 'array')
                        and sub_schema.get('default') is not None
                    ):
                        default_val = repr(sub_schema['default'])
                    else:
                        default_val = "None"

                    flat_params.append({
                        'name': prefixed_name,
                        'type': get_param_type_hint(sub_name, tool, schema_override=sub_schema),
                        'default': default_val,
                        'required': is_req,
                        'source': 'body_nested',
                        'executor_key': prop_name,
                    })
                    seen_names.add(prefixed_name)

                body_reconstruction[prop_name] = child_mapping

            else:
                # Simple field or array: add directly
                if prop_name not in seen_names:
                    is_req = prop_name in required_flat_names
                    default_val = None if is_req else get_param_default_value(prop_name, tool)

                    flat_params.append({
                        'name': prop_name,
                        'type': get_param_type_hint(prop_name, tool),
                        'default': default_val,
                        'required': is_req,
                        'source': (
                            'body_array' if prop_schema.get('type') == 'array'
                            else 'body_simple'
                        ),
                        'executor_key': prop_name,
                    })
                    seen_names.add(prop_name)

    # Python requires params-with-defaults AFTER params-without-defaults
    required_params = [p for p in flat_params if p['required']]
    optional_params = [p for p in flat_params if not p['required']]
    flat_params = required_params + optional_params

    # --- Build function signature ---
    lines = []
    lines.append("@app.tool(")
    lines.append(f"    name={repr(tool_name)},")
    # Fix: embed description via repr() so backslashes / triple-quotes are safe
    lines.append(f"    description={repr(description)}")
    lines.append(")")

    sig_parts = []
    for p in flat_params:
        if p['required']:
            sig_parts.append(f"{p['name']}: {p['type']}")
        else:
            sig_parts.append(f"{p['name']}: {p['type']} = {p['default']}")

    lines.append(f"async def {tool_name}(")
    if sig_parts:
        lines.append("    " + ",\n    ".join(sig_parts))
    lines.append(") -> Dict[str, Any]:")

    # Docstring — also use repr() to prevent triple-quote injection
    lines.append(f"    __doc__ = {repr(f'{description} | {method} {path}')}")

    # --- Reconstruction logic ---
    lines.append(f"    # Reconstruct nested objects from flattened parameters")
    if body_reconstruction:
        for parent, children_map in body_reconstruction.items():
            parent_required_children = {
                orig for orig, prefixed in children_map.items()
                if prefixed in required_flat_names
            }
            has_optional_children = any(
                orig not in parent_required_children for orig in children_map
            )

            lines.append(f"    {parent} = {{")
            for orig_child, prefixed_child in children_map.items():
                lines.append(f"        {repr(orig_child)}: {prefixed_child},")
            lines.append(f"    }}")

            if has_optional_children:
                optional_children = {
                    orig for orig in children_map if orig not in parent_required_children
                }
                if optional_children:
                    req_set = tuple(sorted(parent_required_children))
                    lines.append(
                        f"    {parent} = {{k: v for k, v in {parent}.items() "
                        f"if k in {req_set!r} or v is not None}}"
                    )
            lines.append("")

    # --- Build executor call args (deduplicated, ordered) ---
    executor_call_args = []
    executor_keys_seen: set = set()
    for p in flat_params:
        ekey = p['executor_key']
        if ekey not in executor_keys_seen:
            executor_keys_seen.add(ekey)
            executor_call_args.append(ekey)

    lines.append(f"    return await EXECUTORS[{repr(tool_name)}](")
    for arg in executor_call_args:
        lines.append(f"        {arg}={arg},")
    lines.append(f"    )")
    lines.append("")

    return "\n".join(lines)


# ====================================================
# MCP Server Generator
# ====================================================

def generate_mcp_server(
    tools: List[Dict[str, Any]],
    output_dir: Path,
    server_name: str = "mcp-server"
) -> None:
    """
    Generate optimized FastMCP server.py with simplified tool wrappers.

    Args:
        tools: List of tool definitions (Tool IR)
        output_dir: Directory to write generated files
        server_name: Name of the MCP server
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tools_by_namespace: Dict[str, List[Dict[str, Any]]] = {}
    for tool in tools:
        ns = tool.get('namespace') or 'default'
        if ns not in tools_by_namespace:
            tools_by_namespace[ns] = []
        tools_by_namespace[ns].append(tool)

    # ====================================================
    # server.py
    # ====================================================
    server_code = []
    server_code.append('"""')
    server_code.append(f"FastMCP Server: {server_name}")
    server_code.append("Auto-generated from OpenAPI specification")
    server_code.append("")
    server_code.append(f"Total Tools: {len(tools)}")
    server_code.append(f"Namespaces: {len(tools_by_namespace)}")
    server_code.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    server_code.append('"""')
    server_code.append("")
    server_code.append("import os")
    server_code.append("import sys")
    server_code.append("from pathlib import Path")
    server_code.append("from typing import Any, Dict, List, Optional")
    server_code.append("")
    server_code.append("from dotenv import load_dotenv")
    server_code.append("from fastmcp import FastMCP")
    server_code.append("")
    server_code.append("from executors import EXECUTORS")
    server_code.append("")
    server_code.append("# " + "=" * 60)
    server_code.append("# Configuration")
    server_code.append("# " + "=" * 60)
    server_code.append("")
    server_code.append("load_dotenv(dotenv_path=Path(__file__).parent / '.env', override=True)")
    server_code.append("")
    server_code.append(f"SERVER_NAME = {repr(server_name)}")
    server_code.append(f"app = FastMCP(SERVER_NAME)")
    server_code.append("")
    server_code.append("")
    server_code.append("# " + "=" * 60)
    server_code.append("# Tool Definitions")
    server_code.append("# " + "=" * 60)
    server_code.append("")

    for tool in tools:
        server_code.append(generate_tool_wrapper(tool))

    server_code.append("")
    server_code.append("# " + "=" * 60)
    server_code.append("# Server Entry Point")
    server_code.append("# " + "=" * 60)
    server_code.append("")
    server_code.append("if __name__ == '__main__':")
    server_code.append("    import argparse")
    server_code.append("    ")
    server_code.append("    parser = argparse.ArgumentParser(description='FastMCP Server')")
    server_code.append(
        "    parser.add_argument('--transport', choices=['stdio', 'http'], default='http',"
    )
    server_code.append(
        "                        help='Transport mode: stdio for Claude Desktop, http for direct API access')"
    )
    server_code.append(
        "    parser.add_argument('--host', default='127.0.0.1', help='HTTP host (default: 127.0.0.1)')"
    )
    server_code.append(
        "    parser.add_argument('--port', type=int, default=8000, help='HTTP port (default: 8000)')"
    )
    server_code.append("    ")
    server_code.append("    args = parser.parse_args()")
    server_code.append("    ")
    server_code.append("    if args.transport == 'stdio':")
    server_code.append("        print(f'Starting {SERVER_NAME} in STDIO mode...', file=sys.stderr)")
    server_code.append("        app.run(transport='stdio')")
    server_code.append("    else:")
    server_code.append("        print(f'Starting {SERVER_NAME} on {args.host}:{args.port}...')")
    server_code.append("        print(f'Tools endpoint: http://{args.host}:{args.port}/tools')")
    server_code.append("        print(f'Health check: http://{args.host}:{args.port}/health')")
    server_code.append("        app.run(transport='http', host=args.host, port=args.port)")

    server_file = output_dir / "server.py"
    with open(server_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(server_code))
    logger.info(f"Generated {server_file}")

    # ====================================================
    # .env
    # ====================================================
    env_lines = [
        "# " + "=" * 60,
        "# MCP Server Configuration",
        "# " + "=" * 60,
        "",
        "# API Base URL - REQUIRED",
        "MCP_API_BASE_URL=https://your-api-url.com",
        "",
        "# Authentication - Standard Bearer Token",
        "MCP_AUTH_HEADER=Authorization",
        "MCP_AUTH_SCHEME=Bearer",
        "MCP_AUTH_TOKEN=your-token-here",
        "",
        "# Authentication - API Key in Custom Header",
        "# MCP_AUTH_HEADER=X-API-Key",
        "# MCP_AUTH_TOKEN=your-api-key",
        "",
        "# Extra Headers (JSON object)",
        '# MCP_EXTRA_HEADERS={"X-Tenant-ID": "DTB", "Accept": "application/json"}',
        "",
        "# Cookie (if required)",
        "# MCP_COOKIE=session_id=abc123",
        "",
        "# HTTP Timeouts (seconds)",
        "MCP_CONNECT_TIMEOUT=5",
        "MCP_READ_TIMEOUT=30",
        "MCP_WRITE_TIMEOUT=30",
        "MCP_POOL_TIMEOUT=5",
    ]
    env_file = output_dir / ".env"
    with open(env_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(env_lines))
    logger.info(f"Generated {env_file}")

    # ====================================================
    # README.md
    # ====================================================
    readme_lines = [
        f"# {server_name}",
        "",
        "FastMCP server auto-generated from OpenAPI specification.",
        "",
        "## Overview",
        "",
        f"- **Total Tools:** {len(tools)}",
        f"- **Namespaces:** {len(tools_by_namespace)}",
        f"- **Generated:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        "",
        "### Available Tools",
        "",
    ]
    for ns in sorted(tools_by_namespace.keys()):
        ns_tools = tools_by_namespace[ns]
        readme_lines.append(f"**{ns.upper()}** ({len(ns_tools)} tools):")
        for tool in ns_tools[:10]:
            readme_lines.append(f"- `{tool['name']}` - {tool['method']} {tool['path']}")
        if len(ns_tools) > 10:
            readme_lines.append(f"- ... and {len(ns_tools) - 10} more")
        readme_lines.append("")

    readme_lines += [
        "---",
        "",
        "## Quick Start",
        "",
        "```bash",
        "pip install -r requirements.txt",
        "```",
        "",
        "Edit `.env` with your API credentials, then:",
        "",
        "```bash",
        "python server.py                          # HTTP mode",
        "python server.py --transport stdio        # STDIO mode (Claude Desktop)",
        "```",
        "",
        "## Connect with Claude Desktop",
        "",
        "Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS):",
        "",
        "```json",
        "{",
        '  "mcpServers": {',
        f'    "{server_name}": {{',
        '      "command": "python",',
        f'      "args": ["/absolute/path/to/{server_name}/server.py"]',
        "    }",
        "  }",
        "}",
        "```",
        "",
        "On Windows: `%APPDATA%\\Claude\\claude_desktop_config.json`",
    ]

    readme_file = output_dir / "README.md"
    with open(readme_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(readme_lines))
    logger.info(f"Generated {readme_file}")

    logger.info(
        f"Server generation complete: {len(tools)} tools, {len(tools_by_namespace)} namespaces"
    )


if __name__ == '__main__':
    print("Optimized SDK code generator V2.3 - use as import")
    print("\nUsage:")
    print("  from python_sdk_codegen import generate_mcp_server")
    print("  generate_mcp_server(tools, Path('output'), 'my-server')")