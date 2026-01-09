---
name: memvid-core
description: This skill should be used when the user asks to "remember this", "recall", "what did I say about", "note that", "for future reference", "store memory", "my preferences", "forget", or discusses memory storage and retrieval.
version: 1.0.0
---

# Memvid Memory Skill

This skill enables natural memory storage and retrieval using a dual-memory architecture.

## Memory Scopes

### Project Memory (`.memvid_data/`)
Stores project-specific knowledge:
- Code architecture decisions: "This project uses FastAPI for APIs"
- Bug reports: "Bug in auth.py line 42"
- Technical implementation: "We use MemvidMemory class for storage"
- Project configuration: "Dependencies managed via pyproject.toml"

### User Memory (`~/memvid_data/`)
Stores personal preferences:
- Coding preferences: "I prefer pytest over unittest"
- Style preferences: "I like clean code with type hints"
- Tool preferences: "I use VS Code for development"
- General knowledge: "I always document functions with docstrings"

## Automatic Classification

The system uses semantic similarity to auto-classify memories:
- Compares content against project and user examples
- Assigns to scope with higher similarity
- Falls back to user memory if confidence < 0.65

## Commands

### memvid:store
Store a new memory with automatic scope detection.

**Trigger patterns:**
- "remember that this project uses FastAPI"
- "note that I prefer pytest"
- "for future reference: bug in line 42"

**Action:** Call `memvid_store` tool with:
```json
{
  "content": "<extracted content>",
  "scope": "auto"
}
```

**Response format:**
"Stored in [project/user] memory: [brief summary]. (ID: abc-123)"

### memvid:recall
Search both memories and return merged results.

**Trigger patterns:**
- "what did I say about testing?"
- "recall memories about authentication"
- "do you remember my coding preferences?"

**Action:** Call `memvid_query` tool with:
```json
{
  "query": "<extracted query>",
  "limit": 10
}
```

**Response format:**
Present results naturally:
"Based on your memories:
- [Project] This project uses FastAPI for REST APIs
- [User] You prefer pytest over unittest"

### memvid:forget
Delete a memory by ID.

**Trigger patterns:**
- "forget memory abc-123"
- "delete that memory about authentication"

**Action:**
1. If ID provided: Call `memvid_delete` with ID
2. If description: Call `memvid_query` first, confirm with user, then delete

### memvid:stats
Show statistics for both memory stores.

**Trigger patterns:**
- "show memory stats"
- "how many memories do I have?"

**Action:** Call `memvid_stats` tool and present:
```
Memory Statistics:
Project Memory (.memvid_data/):
  - Total: 42 memories
  - By sector: episodic (15), semantic (20), procedural (5), emotional (1), reflective (1)

User Memory (~/memvid_data/):
  - Total: 103 memories
  - By sector: semantic (60), procedural (30), reflective (13)
```

## Auto-Trigger Rules

**Strong triggers (auto-store):**
- "Remember that ..."
- "Note that ..."
- "For future reference: ..."
- "I always ..."
- "This project always ..."

**Weak triggers (ask first):**
- "I think ..."
- "Maybe ..."

## Best Practices

1. Always show classification results: Tell user if memory was stored in project or user scope
2. Natural integration: Don't say "I will call memvid_store" - just say "I'll remember that..."
3. Contextualize recalls: When presenting recalled memories, explain which scope they came from
4. Deduplicate before storing: Check if similar memory exists before storing new one

## Error Handling

- If storage fails: "I couldn't store that memory. Error: [details]"
- If no memories found: "I don't have any memories about [topic] yet."
- If ambiguous classification: "I'm not sure if this is project or personal knowledge. Storing in user memory for safety."
