"""
MCP server for memvid memory operations.

Exposes OpenMemory-style tools via Model Context Protocol:
- memvid_store: Store memories with sector classification
- memvid_query: Search with sector penalties and waypoint expansion
- memvid_get/list/delete: CRUD operations
- memvid_store_fact/query_facts: Temporal knowledge graph
- memvid_reinforce: Manual salience boost
- memvid_apply_decay: Trigger decay cycle
- memvid_stats: Get statistics
"""
import asyncio
import json
import logging
import sys
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolResult,
    ListToolsResult,
    TextContent,
    Tool,
)

from .dual_memory import DualMemoryManager

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("memvid-mcp")

# Global dual memory instance
_memory: DualMemoryManager | None = None


def get_memory() -> DualMemoryManager:
    """Get or create the global dual memory manager."""
    global _memory
    if _memory is None:
        _memory = DualMemoryManager()
    return _memory


# Tool definitions
TOOLS = [
    # ==================== Core Memory Tools ====================
    Tool(
        name="memvid_store",
        description="""Store a new memory with automatic scope and sector classification.

Scopes (auto-detected or manual):
- project: Project-specific knowledge (stored in .memvid_data/)
- user: Personal preferences (stored in ~/memvid_data/)

Sectors are automatically detected based on content:
- episodic: Events, experiences ("yesterday I went...", "remember when...")
- semantic: Facts, knowledge ("Python is a language", "the capital of...")
- procedural: How-to, steps ("first install...", "click the button...")
- emotional: Feelings ("I feel happy", "love this!", "so frustrated")
- reflective: Insights ("I realized that...", "the pattern is...")

Each sector has different decay rates - emotional memories fade fastest,
reflective insights persist longest.""",
        inputSchema={
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The text content to store as a memory",
                },
                "scope": {
                    "type": "string",
                    "enum": ["auto", "project", "user"],
                    "default": "auto",
                    "description": "Memory scope: auto (semantic classification), project, or user",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional tags for categorization",
                },
                "metadata": {
                    "type": "object",
                    "description": "Optional additional metadata",
                },
                "sector": {
                    "type": "string",
                    "enum": ["episodic", "semantic", "procedural", "emotional", "reflective"],
                    "description": "Override automatic sector classification",
                },
                "user_id": {
                    "type": "string",
                    "description": "User identifier for memory isolation",
                },
            },
            "required": ["content"],
        },
    ),
    Tool(
        name="memvid_query",
        description="""Search memories from both project and user stores with intelligent merging.

Features:
- Dual-store search: Queries both project and user memories
- Project priority: Project memories get 1.2x score boost
- Deduplication: Similar results are merged
- Cross-sector penalties: Searching for facts penalizes emotional memories
- Waypoint expansion: Finds related memories via association graph
- Salience weighting: Recently accessed memories rank higher""",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query text",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "default": 10,
                    "description": "Maximum number of results to return",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter results by tags",
                },
                "sector": {
                    "type": "string",
                    "enum": ["episodic", "semantic", "procedural", "emotional", "reflective"],
                    "description": "Filter by sector",
                },
                "expand_waypoints": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to expand results via association graph",
                },
                "user_id": {
                    "type": "string",
                    "description": "User identifier for memory isolation",
                },
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="memvid_get",
        description="Get a specific memory by its ID.",
        inputSchema={
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "The memory ID to retrieve",
                },
                "user_id": {
                    "type": "string",
                    "description": "User identifier for ownership verification",
                },
            },
            "required": ["id"],
        },
    ),
    Tool(
        name="memvid_list",
        description="List stored memories sorted by salience and recency.",
        inputSchema={
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 20,
                    "description": "Maximum number of results",
                },
                "offset": {
                    "type": "integer",
                    "minimum": 0,
                    "default": 0,
                    "description": "Number of results to skip",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter by tags",
                },
                "sector": {
                    "type": "string",
                    "enum": ["episodic", "semantic", "procedural", "emotional", "reflective"],
                    "description": "Filter by sector",
                },
                "scope": {
                    "type": "string",
                    "enum": ["project", "user"],
                    "description": "Filter by scope (omit to list from both)",
                },
                "user_id": {
                    "type": "string",
                    "description": "User identifier for memory isolation",
                },
            },
        },
    ),
    Tool(
        name="memvid_delete",
        description="Delete a specific memory and its waypoint associations.",
        inputSchema={
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "The memory ID to delete",
                },
                "user_id": {
                    "type": "string",
                    "description": "User identifier for ownership verification",
                },
            },
            "required": ["id"],
        },
    ),
    Tool(
        name="memvid_delete_all",
        description="Delete all memories for a user. Use with caution.",
        inputSchema={
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": "User identifier whose memories to delete",
                },
            },
        },
    ),
    # ==================== Temporal Knowledge Graph Tools ====================
    Tool(
        name="memvid_store_fact",
        description="""Store a temporal fact with validity period.

Facts are stored as (subject, predicate, object) triples with time tracking.
When a new fact conflicts with an existing one (same subject+predicate),
the old fact is automatically closed.

Example: Store "Alice works_at Google" then later "Alice works_at Meta"
- The Google fact gets valid_to set to just before Meta fact starts
- Querying at different times returns different employers""",
        inputSchema={
            "type": "object",
            "properties": {
                "subject": {
                    "type": "string",
                    "description": "Entity the fact is about (e.g., 'Alice')",
                },
                "predicate": {
                    "type": "string",
                    "description": "Relationship type (e.g., 'works_at')",
                },
                "object": {
                    "type": "string",
                    "description": "Value/target (e.g., 'Google')",
                },
                "scope": {
                    "type": "string",
                    "enum": ["project", "user"],
                    "default": "project",
                    "description": "Which memory store to use (default: project)",
                },
                "valid_from": {
                    "type": "string",
                    "description": "When fact became true (ISO date string, default: now)",
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "default": 1.0,
                    "description": "Certainty level (decays over time)",
                },
                "metadata": {
                    "type": "object",
                    "description": "Additional data about the fact",
                },
            },
            "required": ["subject", "predicate", "object"],
        },
    ),
    Tool(
        name="memvid_query_facts",
        description="""Query facts valid at a specific point in time.

Supports point-in-time queries: "What was true on date X?"
Only returns facts where valid_from <= query_time AND (valid_to IS NULL OR valid_to >= query_time)""",
        inputSchema={
            "type": "object",
            "properties": {
                "subject": {
                    "type": "string",
                    "description": "Filter by subject",
                },
                "predicate": {
                    "type": "string",
                    "description": "Filter by predicate",
                },
                "object": {
                    "type": "string",
                    "description": "Filter by object",
                },
                "at": {
                    "type": "string",
                    "description": "Point in time to query (ISO date string, default: now)",
                },
            },
        },
    ),
    Tool(
        name="memvid_get_timeline",
        description="Get chronological history of facts for a subject.",
        inputSchema={
            "type": "object",
            "properties": {
                "subject": {
                    "type": "string",
                    "description": "Entity to get timeline for",
                },
                "predicate": {
                    "type": "string",
                    "description": "Optional predicate filter",
                },
            },
            "required": ["subject"],
        },
    ),
    # ==================== Decay & Reinforcement Tools ====================
    Tool(
        name="memvid_reinforce",
        description="""Manually reinforce a memory's salience.

Increases the memory's salience score, making it more likely to be
retrieved and slower to decay. Uses diminishing returns formula:
new_salience = salience + boost * (1 - salience)""",
        inputSchema={
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "Memory ID to reinforce",
                },
                "boost": {
                    "type": "number",
                    "minimum": 0.01,
                    "maximum": 0.5,
                    "default": 0.15,
                    "description": "Salience boost amount",
                },
            },
            "required": ["id"],
        },
    ),
    Tool(
        name="memvid_apply_decay",
        description="""Apply time-based decay to all memories.

Memories decay based on their sector:
- emotional: fastest decay (0.02/day)
- episodic: fast decay (0.015/day)
- procedural: medium decay (0.008/day)
- semantic: slow decay (0.005/day)
- reflective: almost permanent (0.001/day)

Also decays temporal fact confidence and prunes weak waypoints.""",
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    # ==================== Statistics ====================
    Tool(
        name="memvid_stats",
        description="Get comprehensive memory storage statistics.",
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
]


async def handle_tool_call(name: str, arguments: dict[str, Any]) -> CallToolResult:
    """Handle a tool call and return the result."""
    memory = get_memory()

    try:
        # ==================== Core Memory Operations ====================
        if name == "memvid_store":
            result = memory.store(
                content=arguments["content"],
                scope=arguments.get("scope", "auto"),
                user_id=arguments.get("user_id"),
                tags=arguments.get("tags"),
                metadata=arguments.get("metadata"),
                sector=arguments.get("sector"),
            )
            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps(result, indent=2))]
            )

        elif name == "memvid_query":
            results = memory.recall(
                query=arguments["query"],
                user_id=arguments.get("user_id"),
                limit=arguments.get("limit", 10),
                tags=arguments.get("tags"),
                sector=arguments.get("sector"),
                expand_waypoints=arguments.get("expand_waypoints", True),
            )
            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps(results, indent=2))]
            )

        elif name == "memvid_get":
            result = memory.get(
                memory_id=arguments["id"],
                user_id=arguments.get("user_id"),
            )
            if result is None:
                return CallToolResult(
                    content=[TextContent(type="text", text="Memory not found")]
                )
            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps(result, indent=2))]
            )

        elif name == "memvid_list":
            results = memory.list_memories(
                user_id=arguments.get("user_id"),
                limit=arguments.get("limit", 20),
                offset=arguments.get("offset", 0),
                tags=arguments.get("tags"),
                sector=arguments.get("sector"),
                scope=arguments.get("scope"),
            )
            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps(results, indent=2))]
            )

        elif name == "memvid_delete":
            success = memory.delete(
                memory_id=arguments["id"],
                user_id=arguments.get("user_id"),
            )
            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps({"deleted": success}))]
            )

        elif name == "memvid_delete_all":
            count = memory.delete_all(user_id=arguments.get("user_id"))
            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps({"deleted_count": count}))]
            )

        # ==================== Temporal Knowledge Graph ====================
        elif name == "memvid_store_fact":
            fact_id = memory.store_fact(
                subject=arguments["subject"],
                predicate=arguments["predicate"],
                obj=arguments["object"],
                scope=arguments.get("scope", "project"),
                valid_from=arguments.get("valid_from"),
                confidence=arguments.get("confidence", 1.0),
                metadata=arguments.get("metadata"),
            )
            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps({"fact_id": fact_id}))]
            )

        elif name == "memvid_query_facts":
            facts = memory.query_facts(
                subject=arguments.get("subject"),
                predicate=arguments.get("predicate"),
                obj=arguments.get("object"),
                at=arguments.get("at"),
            )
            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps(facts, indent=2))]
            )

        elif name == "memvid_get_timeline":
            timeline = memory.get_timeline(
                subject=arguments["subject"],
                predicate=arguments.get("predicate"),
            )
            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps(timeline, indent=2))]
            )

        # ==================== Decay & Reinforcement ====================
        elif name == "memvid_reinforce":
            new_salience = memory.reinforce_memory(
                memory_id=arguments["id"],
                boost=arguments.get("boost", 0.15),
            )
            if new_salience is None:
                return CallToolResult(
                    content=[TextContent(type="text", text="Memory not found")]
                )
            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps({"new_salience": new_salience}))]
            )

        elif name == "memvid_apply_decay":
            updated_count = memory.apply_decay()
            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps({"memories_updated": updated_count}))]
            )

        # ==================== Statistics ====================
        elif name == "memvid_stats":
            stats = memory.stats()
            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps(stats, indent=2))]
            )

        else:
            return CallToolResult(
                content=[TextContent(type="text", text=f"Unknown tool: {name}")]
            )

    except Exception as e:
        logger.exception(f"Error handling tool {name}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error: {str(e)}")]
        )


async def run_server():
    """Run the MCP server."""
    server = Server("memvid-mcp")

    @server.list_tools()
    async def list_tools() -> ListToolsResult:
        return ListToolsResult(tools=TOOLS)

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> CallToolResult:
        return await handle_tool_call(name, arguments)

    logger.info("Starting memvid-mcp server with cognitive memory features...")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main():
    """Entry point for the MCP server."""
    asyncio.run(run_server())


if __name__ == "__main__":
    main()
