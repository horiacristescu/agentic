# Simple Agent - Learning Project

> Building a minimal agent runtime from scratch to understand agent architecture best practices and improve Python engineering skills.

## 📚 Documentation Hub

**README is your control center.** Start here, navigate from here.

### Starting a New Session

1. **Check `docs/TODO.md`** - What's next? Find the current task
2. **Read `docs/PROGRESS.md`** - What happened last session? Get context
3. **Build the feature** - Write code, make decisions
4. **Update docs** - See "Where to Write" below

### What not to do
- You should never touch git, no committing, resetting, or any git operation. In a previous iteration an agent deleted 3 hours of work with a git reset command, let's avoid it from now on. This is a costly lesson.

### Where to Write What

**`README.md`** (you are here)
- Project overview and quick start
- High-level progress tracker
- Key discoveries (major insights only)
- Update: Major milestones only

**`docs/TODO.md`** 
- Roadmap with all tasks and checkboxes
- Session log entries (append after each session)
- Update: Every session (check boxes + log entry)

**`docs/PROGRESS.md`**
- Detailed session journal (what, why, how)
- Design decisions and rationale
- Update: Every session (new entry at top)

**`docs/ARCHITECTURE.md`**
- Design patterns and their rationale
- System invariants and core abstractions
- Update: When making architectural decisions

**`docs/SELF_ANALYSIS.md`**
- Learning meta - skills, struggles, growth
- Update: When reflecting on learning process

### Quick Reference

- **What's next?** → `docs/TODO.md`
- **What happened?** → `docs/PROGRESS.md`
- **Why this way?** → `docs/ARCHITECTURE.md`
- **How am I doing?** → `docs/SELF_ANALYSIS.md`

---

## 🎯 Learning Goals

**Primary Goals:**
1. **Agent Architecture**: Understand production patterns by building from scratch
2. **Engineering Python**: Move from sloppy to production-quality code

**Specific Objectives:**
- Understand the ReAct loop (Reason → Act → Observe)
- Implement tool calling with structured outputs
- Build state management and trajectory tracking
- Create an evaluation framework
- Learn modern Python patterns, typing, and tooling
- Discover why agents are hard through hands-on implementation

## 🏗️ Architecture (To Be Built)

### Core Components
- **Agent Runtime**: The ReAct loop
- **Tool System**: Function → Schema → Execution
- **State Tracker**: Conversation history and trajectory
- **Evaluator**: Test cases and scoring

## 📁 Project Structure

```
agentic/
├── README.md           # Project overview (you are here)
├── framework/          # Core agent framework
│   ├── agents.py      # Agent runtime (ReAct loop)
│   ├── messages.py    # Message types (Message, ToolCall, Result)
│   ├── tools.py       # Tool system
│   ├── llm.py         # LLM client abstraction
│   ├── observers.py   # Observer protocol
│   └── tests/         # Framework tests
├── observers/         # Observer implementations
│   ├── console_tracer.py  # Console logging with formatted output
│   └── web_debugger.py    # Interactive web-based debugger
├── tools/             # Tool implementations
│   ├── calculator_tool.py
│   └── weather_tool.py
├── agents/            # Example agents
│   ├── calculator_agent.py
│   ├── weather_agent.py
│   └── debug_example.py   # Web debugger demo
└── docs/              # Documentation
    ├── SELF_ANALYSIS.md  # Learning progress & skills
    ├── TODO.md           # Task roadmap
    ├── PROGRESS.md       # Session journal
    └── ARCHITECTURE.md   # Design decisions
```

## 🚀 How to Run

```bash
# Activate virtual environment
source activate.sh  # or: source .venv/bin/activate

# Run calculator agent
python agents/calculator_agent.py

# Run weather agent (error recovery demo)
python agents/weather_agent.py
```

## 🚀 Current Status

**Phase 1: Production Robustness** (in progress - error handling)

- Phase 0: Complete ✅
- Phase 1: ~85% (building error taxonomy and testing)
- Phases 2-6: Not started

*See `docs/TODO.md` for detailed roadmap and task breakdown*

## 💡 Key Discoveries

### Errors as Values Enable Self-Correction
The agent naturally self-corrects from tool failures without explicit retry logic. By returning errors as `Result(status=ERROR, error="...")` instead of throwing exceptions, errors flow through the message pipeline. The LLM sees the error in context, reasons about it, and retries with corrections. This "errors as values" pattern (like Rust's `Result<T,E>`) makes failure recovery an emergent property of the ReAct loop.

**Example:** When querying "temperature in Bucharest", the weather tool fails with "City not found. Use Country/City format". The agent reads this error, reasons about the required format, and retries with "Romania/Bucharest" successfully.

### LLMs Need Constant Reminders for Structured Output
Even with JSON schemas in the system prompt, LLMs frequently "forget" to return JSON when giving simple answers, responding in plain text instead. The solution: explicit rules with concrete examples at the top of the system prompt ("CRITICAL: You MUST ALWAYS respond with valid JSON...") plus robust JSON cleaning that handles trailing garbage. This reduced JSON parsing failures from ~40% to <5%.

### Observability is Critical for Agent Development
Agents are non-deterministic and fail frequently. Raw JSON dumps are unreadable. The solution: multiple viewing modes. ConsoleTracer formats tool calls as Python signatures (`calculator(operation="add", x=2, y=1)`) for quick scanning. Web debugger provides timeline view with JSON/formatted toggle plus interactive chat that continues execution. Being able to "jump in" and chat with a partially-executed agent accelerates debugging 10x.

## 🤔 Open Questions

_Track what you're struggling with or curious about_

## 📊 Results

_Evaluation results and comparisons_

