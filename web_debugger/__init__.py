"""Web-based debugger for agents with real-time SSE streaming"""

import json
import time
import webbrowser
from pathlib import Path
from threading import Timer

from flask import Flask, Response, jsonify, render_template, request, stream_with_context

from agentic.framework.messages import Message, ResultStatus


def _serialize_message(msg: Message) -> dict:
    """Serialize a Message to JSON-compatible dict"""
    msg_data = {
        "role": msg.role,
        "content": msg.content,
        "timestamp": msg.timestamp,
        "error_code": msg.error_code,
    }

    # Add tool calls if present
    if msg.tool_calls:
        msg_data["tool_calls"] = [
            {"id": tc.id, "tool": tc.tool, "args": tc.args} for tc in msg.tool_calls
        ]

    # Add tool result info
    if msg.tool_call_id:
        msg_data["tool_call_id"] = msg.tool_call_id
        msg_data["name"] = msg.name

    # Add tokens
    if msg.tokens_in or msg.tokens_out:
        msg_data["tokens"] = {
            "in": msg.tokens_in or 0,
            "out": msg.tokens_out or 0,
        }

    return msg_data


def create_app(agent):
    """Create Flask app with agent debugger routes"""
    template_dir = Path(__file__).parent / "templates"
    static_dir = Path(__file__).parent / "static"

    app = Flask(__name__, template_folder=str(template_dir), static_folder=str(static_dir))

    @app.route("/")
    def index():
        """Main debugger UI"""
        return render_template("index.html")

    @app.route("/api/messages")
    def get_messages():
        """Get all messages from agent"""
        messages = [_serialize_message(msg) for msg in agent.messages]
        return jsonify(messages)

    @app.route("/api/state")
    def get_state():
        """Get agent state"""
        return jsonify(
            {
                "turn_count": agent.turn_count,
                "tokens_used": agent.tokens_used,
                "message_count": len(agent.messages),
                "tools": [t.name for t in agent.tools],
                "max_turns": agent.max_turns,
            }
        )

    @app.route("/api/chat/stream", methods=["POST"])
    def chat_stream():
        """Stream agent messages in real-time using Server-Sent Events"""
        try:
            data = request.json or {}
            user_input = data.get("message", "")

            if not user_input:
                return jsonify({"error": "No message provided"}), 400

            def generate():
                """Generator function that yields SSE messages"""
                try:
                    # Add user message immediately and send it
                    user_msg = Message(
                        role="user",
                        content=user_input,
                        timestamp=time.time(),
                    )
                    agent.messages.append(user_msg)
                    yield f"data: {json.dumps(_serialize_message(user_msg))}\n\n"

                    # Run agent loop until finished (with max iterations for safety)
                    max_iterations = 10
                    iteration = 0

                    while iteration < max_iterations:
                        iteration += 1
                        agent.turn_count += 1

                        # Get agent response
                        agent_response = agent._get_agent_response(agent.messages)

                        if agent_response.status != ResultStatus.SUCCESS:
                            # Send error and stop
                            error_data = {
                                "error": True,
                                "message": agent_response.error or "Unknown error",
                            }
                            yield f"data: {json.dumps(error_data)}\n\n"
                            break

                        # Create assistant message from response
                        assistant_msg = Message(
                            role="assistant",
                            content=agent_response.metadata.get("raw_response", ""),
                            timestamp=time.time(),
                            tool_calls=agent_response.metadata.get("tool_calls"),
                            tokens_in=agent_response.metadata.get("tokens", 0),
                            tokens_out=0,
                        )
                        agent.messages.append(assistant_msg)

                        # Stream assistant message immediately
                        yield f"data: {json.dumps(_serialize_message(assistant_msg))}\n\n"

                        # Check if finished
                        is_finished = agent_response.metadata.get("is_finished", False)
                        tool_calls = agent_response.metadata.get("tool_calls")

                        if is_finished and not tool_calls:
                            # Task complete
                            break

                        if tool_calls:
                            # Execute all tool calls
                            tool_results = agent._execute_tools(
                                tool_calls, agent.messages, agent.turn_count
                            )

                            # Stream each tool message that was added
                            # _execute_tools now adds individual tool messages
                            num_tools = len(tool_calls)
                            for i in range(num_tools):
                                if (
                                    len(agent.messages) > 0
                                    and agent.messages[-num_tools + i].role == "tool"
                                ):
                                    tool_msg = agent.messages[-num_tools + i]
                                    yield f"data: {json.dumps(_serialize_message(tool_msg))}\n\n"

                            # Continue loop to get next response
                        else:
                            # No tool calls and not finished - should not happen but break to be safe
                            break

                    # Send completion signal
                    yield "data: [DONE]\n\n"

                except Exception as e:
                    error_data = {"error": True, "message": str(e)}
                    yield f"data: {json.dumps(error_data)}\n\n"

            return Response(
                stream_with_context(generate()),
                mimetype="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",  # Disable nginx buffering
                },
            )

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return app


def debug_agent(agent, port=8000, auto_open=False):
    """
    Launch web debugger for agent

    Usage:
        agent.run("...")
        from agentic.web_debugger import debug_agent
        debug_agent(agent)

    Args:
        agent: Agent instance to debug
        port: Port to run server on (default 8000)
        auto_open: Automatically open browser (default False)
    """
    app = create_app(agent)

    print(f"\n{'=' * 70}")
    print(f"🌐 Agent Debugger running at http://127.0.0.1:{port}")
    print(f"{'=' * 70}\n")

    if auto_open:
        Timer(1.0, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()

    # Use waitress for faster, more reliable serving
    try:
        from waitress import serve

        serve(app, host="127.0.0.1", port=port, threads=4)
    except ImportError:
        # Fallback to Flask dev server if waitress not installed
        print("⚠️  Install 'waitress' for better performance: uv add waitress")
        app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
