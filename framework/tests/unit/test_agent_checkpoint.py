"""Tests for Agent checkpoint save/load functionality."""

import json
import time
from pathlib import Path
from unittest.mock import Mock

import pytest

from agentic.framework.agents import Agent
from agentic.framework.llm import LLM
from agentic.framework.messages import Message, ResultStatus
from agentic.framework.tools import create_tool
from pydantic import BaseModel


class MockTool(BaseModel):
    """Simple mock tool for testing"""
    value: int
    
    def execute(self, dependencies: dict | None = None) -> str:
        return str(self.value * 2)


class TestCheckpointSave:
    """Test saving agent state to checkpoint"""

    def test_saves_checkpoint_file(self, tmp_path):
        """save_checkpoint creates checkpoint file"""
        agent = Agent(
            llm=Mock(spec=LLM),
            tools=[],
            max_turns=5
        )
        
        checkpoint_path = tmp_path / "checkpoint.json"
        agent.save_checkpoint(checkpoint_path)
        
        assert checkpoint_path.exists()

    def test_creates_parent_directories(self, tmp_path):
        """save_checkpoint creates parent directories if needed"""
        agent = Agent(
            llm=Mock(spec=LLM),
            tools=[],
        )
        
        checkpoint_path = tmp_path / "nested" / "dir" / "checkpoint.json"
        agent.save_checkpoint(checkpoint_path)
        
        assert checkpoint_path.exists()

    def test_saves_messages(self, tmp_path):
        """Checkpoint includes all messages"""
        agent = Agent(
            llm=Mock(spec=LLM),
            tools=[],
        )
        
        # Add some messages
        agent.messages = [
            Message(role="system", content="System prompt", timestamp=time.time()),
            Message(role="user", content="Task", timestamp=time.time()),
            Message(role="assistant", content="Response", timestamp=time.time()),
        ]
        
        checkpoint_path = tmp_path / "checkpoint.json"
        agent.save_checkpoint(checkpoint_path)
        
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
        
        assert len(checkpoint["messages"]) == 3
        assert checkpoint["messages"][0]["role"] == "system"
        assert checkpoint["messages"][1]["content"] == "Task"

    def test_saves_turn_count(self, tmp_path):
        """Checkpoint includes turn count"""
        agent = Agent(
            llm=Mock(spec=LLM),
            tools=[],
        )
        agent.turn_count = 7
        
        checkpoint_path = tmp_path / "checkpoint.json"
        agent.save_checkpoint(checkpoint_path)
        
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
        
        assert checkpoint["turn_count"] == 7

    def test_saves_tokens_used(self, tmp_path):
        """Checkpoint includes token usage"""
        agent = Agent(
            llm=Mock(spec=LLM),
            tools=[],
        )
        agent.tokens_used = 1234
        
        checkpoint_path = tmp_path / "checkpoint.json"
        agent.save_checkpoint(checkpoint_path)
        
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
        
        assert checkpoint["tokens_used"] == 1234

    def test_checkpoint_is_valid_json(self, tmp_path):
        """Checkpoint file is valid, formatted JSON"""
        agent = Agent(
            llm=Mock(spec=LLM),
            tools=[],
        )
        agent.messages = [Message(role="user", content="test", timestamp=time.time())]
        
        checkpoint_path = tmp_path / "checkpoint.json"
        agent.save_checkpoint(checkpoint_path)
        
        # Should parse without error
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
        
        assert isinstance(checkpoint, dict)
        assert "messages" in checkpoint
        assert "turn_count" in checkpoint
        assert "tokens_used" in checkpoint


class TestCheckpointLoad:
    """Test loading agent state from checkpoint"""

    def test_loads_messages(self, tmp_path):
        """load_checkpoint restores messages"""
        # Create checkpoint
        checkpoint = {
            "messages": [
                {"role": "system", "content": "sys", "timestamp": time.time()},
                {"role": "user", "content": "task", "timestamp": time.time()},
            ],
            "turn_count": 0,
            "tokens_used": 0,
        }
        
        checkpoint_path = tmp_path / "checkpoint.json"
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f)
        
        # Load into agent
        agent = Agent(llm=Mock(spec=LLM), tools=[])
        agent.load_checkpoint(checkpoint_path)
        
        assert len(agent.messages) == 2
        assert agent.messages[0].role == "system"
        assert agent.messages[1].content == "task"

    def test_loads_turn_count(self, tmp_path):
        """load_checkpoint restores turn count"""
        checkpoint = {
            "messages": [],
            "turn_count": 5,
            "tokens_used": 0,
        }
        
        checkpoint_path = tmp_path / "checkpoint.json"
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f)
        
        agent = Agent(llm=Mock(spec=LLM), tools=[])
        agent.load_checkpoint(checkpoint_path)
        
        assert agent.turn_count == 5

    def test_loads_tokens_used(self, tmp_path):
        """load_checkpoint restores token usage"""
        checkpoint = {
            "messages": [],
            "turn_count": 0,
            "tokens_used": 999,
        }
        
        checkpoint_path = tmp_path / "checkpoint.json"
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f)
        
        agent = Agent(llm=Mock(spec=LLM), tools=[])
        agent.load_checkpoint(checkpoint_path)
        
        assert agent.tokens_used == 999

    def test_accepts_string_path(self, tmp_path):
        """load_checkpoint accepts string path"""
        checkpoint = {
            "messages": [],
            "turn_count": 0,
            "tokens_used": 0,
        }
        
        checkpoint_path = tmp_path / "checkpoint.json"
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f)
        
        agent = Agent(llm=Mock(spec=LLM), tools=[])
        agent.load_checkpoint(str(checkpoint_path))  # String not Path
        
        assert agent.messages == []

    def test_raises_on_missing_file(self):
        """load_checkpoint raises FileNotFoundError for missing file"""
        agent = Agent(llm=Mock(spec=LLM), tools=[])
        
        with pytest.raises(FileNotFoundError):
            agent.load_checkpoint("nonexistent.json")


class TestCheckpointIntegration:
    """Test save/load round-trip and agent resumption"""

    def test_save_load_roundtrip(self, tmp_path):
        """Messages survive save/load roundtrip"""
        agent1 = Agent(llm=Mock(spec=LLM), tools=[])
        agent1.messages = [
            Message(role="system", content="sys", timestamp=1234567890.0),
            Message(role="user", content="task", timestamp=1234567891.0),
        ]
        agent1.turn_count = 3
        agent1.tokens_used = 500
        
        checkpoint_path = tmp_path / "checkpoint.json"
        agent1.save_checkpoint(checkpoint_path)
        
        agent2 = Agent(llm=Mock(spec=LLM), tools=[])
        agent2.load_checkpoint(checkpoint_path)
        
        assert len(agent2.messages) == 2
        assert agent2.messages[0].content == "sys"
        assert agent2.messages[1].content == "task"
        assert agent2.turn_count == 3
        assert agent2.tokens_used == 500

    def test_run_with_checkpoint_parameter(self, tmp_path):
        """Agent.run() loads checkpoint when checkpoint parameter provided"""
        # Create initial agent state
        agent1 = Agent(llm=Mock(spec=LLM), tools=[])
        agent1.messages = [
            Message(role="system", content="sys", timestamp=time.time()),
            Message(role="user", content="task", timestamp=time.time()),
        ]
        agent1.turn_count = 2
        agent1.tokens_used = 100
        
        checkpoint_path = tmp_path / "checkpoint.json"
        agent1.save_checkpoint(checkpoint_path)
        
        # Create new agent and run with checkpoint
        agent2 = Agent(llm=Mock(spec=LLM), tools=[])
        
        # Mock LLM to return finished response
        mock_llm_response = Message(
            role="assistant",
            content='{"reasoning": "test", "tool_calls": null, "result": "done", "is_finished": true}',
            timestamp=time.time(),
            tokens_in=10,
            tokens_out=20,
        )
        agent2.llm.call = Mock(return_value=mock_llm_response)
        
        # Run with checkpoint - should load state before running
        result = agent2.run("continue task", checkpoint=checkpoint_path)
        
        # Agent should have loaded the 2 messages from checkpoint
        # plus added user message and assistant response
        assert len(agent2.messages) >= 2
        assert agent2.turn_count >= 2  # At least the loaded count
        assert agent2.tokens_used >= 100  # At least the loaded count

    def test_checkpoint_parameter_sets_reset_false(self, tmp_path):
        """When checkpoint is provided, reset is automatically False"""
        checkpoint = {
            "messages": [
                {"role": "system", "content": "sys", "timestamp": time.time()},
            ],
            "turn_count": 1,
            "tokens_used": 50,
        }
        
        checkpoint_path = tmp_path / "checkpoint.json"
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f)
        
        agent = Agent(llm=Mock(spec=LLM), tools=[])
        mock_llm_response = Message(
            role="assistant",
            content='{"reasoning": "test", "tool_calls": null, "result": "done", "is_finished": true}',
            timestamp=time.time(),
            tokens_in=10,
            tokens_out=20,
        )
        agent.llm.call = Mock(return_value=mock_llm_response)
        
        # Even though reset=True by default, checkpoint should override it
        agent.run("task", checkpoint=checkpoint_path)
        
        # Should have system message from checkpoint still present
        assert any(msg.role == "system" and msg.content == "sys" for msg in agent.messages)

    def test_multiple_save_load_cycles(self, tmp_path):
        """Can save and load multiple times"""
        agent = Agent(llm=Mock(spec=LLM), tools=[])
        
        # Save checkpoint 1
        agent.messages = [Message(role="user", content="msg1", timestamp=time.time())]
        agent.turn_count = 1
        checkpoint1 = tmp_path / "checkpoint1.json"
        agent.save_checkpoint(checkpoint1)
        
        # Save checkpoint 2
        agent.messages.append(Message(role="assistant", content="msg2", timestamp=time.time()))
        agent.turn_count = 2
        checkpoint2 = tmp_path / "checkpoint2.json"
        agent.save_checkpoint(checkpoint2)
        
        # Load checkpoint 1
        agent.load_checkpoint(checkpoint1)
        assert len(agent.messages) == 1
        assert agent.turn_count == 1
        
        # Load checkpoint 2
        agent.load_checkpoint(checkpoint2)
        assert len(agent.messages) == 2
        assert agent.turn_count == 2


class TestAutoCheckpoint:
    """Test auto-save on exception"""

    def test_auto_checkpoint_saves_on_exception(self, tmp_path):
        """When auto_checkpoint is provided, state is saved on exception"""
        from agentic.framework.errors import AuthError
        
        agent = Agent(llm=Mock(spec=LLM), tools=[])
        
        # Mock LLM to raise AuthError
        agent.llm.call = Mock(side_effect=AuthError("Invalid API key"))
        
        auto_checkpoint_path = tmp_path / "crash.json"
        
        # Run should raise but save checkpoint first
        with pytest.raises(AuthError):
            agent.run("task", auto_checkpoint=auto_checkpoint_path)
        
        # Checkpoint should exist
        assert auto_checkpoint_path.exists()
        
        # Load and verify it has the state up to crash
        with open(auto_checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
        
        # Should have system and user messages (before crash)
        assert len(checkpoint["messages"]) >= 2

    def test_auto_checkpoint_not_saved_when_not_provided(self, tmp_path):
        """Without auto_checkpoint, no file is saved on exception"""
        from agentic.framework.errors import AuthError
        
        agent = Agent(llm=Mock(spec=LLM), tools=[])
        agent.llm.call = Mock(side_effect=AuthError("Invalid API key"))
        
        checkpoint_path = tmp_path / "crash.json"
        
        # Run without auto_checkpoint
        with pytest.raises(AuthError):
            agent.run("task")  # No auto_checkpoint parameter
        
        # No file should be created
        assert not checkpoint_path.exists()

    def test_auto_checkpoint_preserves_exception_traceback(self, tmp_path):
        """Exception is re-raised after checkpoint, preserving traceback"""
        from agentic.framework.errors import InvalidModelError
        
        agent = Agent(llm=Mock(spec=LLM), tools=[])
        agent.llm.call = Mock(side_effect=InvalidModelError("Bad model"))
        
        auto_checkpoint_path = tmp_path / "crash.json"
        
        # Should raise the original exception
        with pytest.raises(InvalidModelError) as exc_info:
            agent.run("task", auto_checkpoint=auto_checkpoint_path)
        
        # Exception message should be preserved
        assert "Bad model" in str(exc_info.value)

    def test_auto_checkpoint_saves_partial_progress(self, tmp_path):
        """Auto checkpoint captures partial progress before crash"""
        agent = Agent(llm=Mock(spec=LLM), tools=[])
        
        # First call succeeds, second call crashes
        call_count = [0]
        def mock_call(messages):
            call_count[0] += 1
            if call_count[0] == 1:
                # First turn: return a tool call
                return Message(
                    role="assistant",
                    content='{"reasoning": "test", "tool_calls": [{"id": "call_1", "tool": "mock", "args": {"value": 5}}], "result": null, "is_finished": false}',
                    timestamp=time.time(),
                    tokens_in=10,
                    tokens_out=20,
                )
            else:
                # Second turn: crash
                from agentic.framework.errors import TransientProviderError
                raise TransientProviderError(
                    message="Connection failed after 3 retries",
                    attempt_count=3,
                    last_error=Exception("Connection failed"),
                    error_type="APIConnectionError"
                )
        
        agent.llm.call = mock_call
        tool = create_tool(MockTool, dependencies={})
        agent.tools = [tool]
        
        auto_checkpoint_path = tmp_path / "crash.json"
        
        from agentic.framework.errors import TransientProviderError
        with pytest.raises(TransientProviderError):
            agent.run("task", auto_checkpoint=auto_checkpoint_path)
        
        # Load checkpoint
        agent2 = Agent(llm=Mock(spec=LLM), tools=[])
        agent2.load_checkpoint(auto_checkpoint_path)
        
        # Should have system, user, assistant (tool call), and tool result
        assert len(agent2.messages) >= 4
        assert agent2.turn_count >= 1  # At least one successful turn

