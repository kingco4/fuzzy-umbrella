import json

import pytest
from PIL import Image

from smolagents.agents import ToolCall
from smolagents.memory import (
    ActionStep,
    AgentMemory,
    ChatMessage,
    MemoryStep,
    MessageRole,
    PlanningStep,
    SystemPromptStep,
    TaskStep,
)
from smolagents.monitoring import Timing, TokenUsage
from smolagents.utils import AgentExecutionError, AgentMaxStepsError


class TestAgentMemory:
    def test_initialization(self):
        system_prompt = "This is a system prompt."
        memory = AgentMemory(system_prompt=system_prompt)
        assert memory.system_prompt.system_prompt == system_prompt
        assert memory.steps == []

    def test_return_all_code_actions(self):
        memory = AgentMemory(system_prompt="This is a system prompt.")
        memory.steps = [
            ActionStep(step_number=1, timing=Timing(start_time=0.0, end_time=1.0), code_action="print('Hello')"),
            ActionStep(step_number=2, timing=Timing(start_time=0.0, end_time=1.0), code_action=None),
            ActionStep(step_number=3, timing=Timing(start_time=0.0, end_time=1.0), code_action="print('World')"),
        ]  # type: ignore
        assert memory.return_full_code() == "print('Hello')\n\nprint('World')"


class TestMemoryStep:
    def test_initialization(self):
        step = MemoryStep()
        assert isinstance(step, MemoryStep)

    def test_dict(self):
        step = MemoryStep()
        assert step.dict() == {}

    def test_to_messages(self):
        step = MemoryStep()
        with pytest.raises(NotImplementedError):
            step.to_messages()


def test_action_step_dict():
    action_step = ActionStep(
        model_input_messages=[ChatMessage(role=MessageRole.USER, content="Hello")],
        tool_calls=[
            ToolCall(id="id", name="get_weather", arguments={"location": "Paris"}),
        ],
        timing=Timing(start_time=0.0, end_time=1.0),
        step_number=1,
        error=None,
        model_output_message=ChatMessage(role=MessageRole.ASSISTANT, content="Hi"),
        model_output="Hi",
        observations="This is a nice observation",
        observations_images=[Image.new("RGB", (100, 100))],
        action_output="Output",
        token_usage=TokenUsage(input_tokens=10, output_tokens=20),
    )
    action_step_dict = action_step.dict()
    # Check each key individually for better test failure messages
    assert "model_input_messages" in action_step_dict
    assert action_step_dict["model_input_messages"] == [
        {"role": MessageRole.USER, "content": "Hello", "tool_calls": None, "raw": None, "token_usage": None}
    ]

    assert "tool_calls" in action_step_dict
    assert len(action_step_dict["tool_calls"]) == 1
    assert action_step_dict["tool_calls"][0] == {
        "id": "id",
        "type": "function",
        "function": {
            "name": "get_weather",
            "arguments": {"location": "Paris"},
        },
    }

    assert "timing" in action_step_dict
    assert action_step_dict["timing"] == {"start_time": 0.0, "end_time": 1.0, "duration": 1.0}

    assert "token_usage" in action_step_dict
    assert action_step_dict["token_usage"] == {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}

    assert "step_number" in action_step_dict
    assert action_step_dict["step_number"] == 1

    assert "error" in action_step_dict
    assert action_step_dict["error"] is None

    assert "model_output_message" in action_step_dict
    assert action_step_dict["model_output_message"] == {
        "role": "assistant",
        "content": "Hi",
        "tool_calls": None,
        "raw": None,
        "token_usage": None,
    }

    assert "model_output" in action_step_dict
    assert action_step_dict["model_output"] == "Hi"

    assert "observations" in action_step_dict
    assert action_step_dict["observations"] == "This is a nice observation"

    assert "observations_images" in action_step_dict

    assert "action_output" in action_step_dict
    assert action_step_dict["action_output"] == "Output"


def test_action_step_to_messages():
    action_step = ActionStep(
        model_input_messages=[ChatMessage(role=MessageRole.USER, content="Hello")],
        tool_calls=[
            ToolCall(id="id", name="get_weather", arguments={"location": "Paris"}),
        ],
        timing=Timing(start_time=0.0, end_time=1.0),
        step_number=1,
        error=None,
        model_output_message=ChatMessage(role=MessageRole.ASSISTANT, content="Hi"),
        model_output="Hi",
        observations="This is a nice observation",
        observations_images=[Image.new("RGB", (100, 100))],
        action_output="Output",
        token_usage=TokenUsage(input_tokens=10, output_tokens=20),
    )
    messages = action_step.to_messages()
    assert len(messages) == 4
    for message in messages:
        assert isinstance(message, ChatMessage)
    assistant_message = messages[0]
    assert assistant_message.role == MessageRole.ASSISTANT
    assert len(assistant_message.content) == 1
    assert assistant_message.content[0]["type"] == "text"
    assert assistant_message.content[0]["text"] == "Hi"
    message = messages[1]
    assert message.role == MessageRole.TOOL_CALL

    assert len(message.content) == 1
    assert message.content[0]["type"] == "text"
    assert "Calling tools:" in message.content[0]["text"]

    image_message = messages[2]
    assert image_message.content[0]["type"] == "image"  # type: ignore

    observation_message = messages[3]
    assert observation_message.role == MessageRole.TOOL_RESPONSE
    assert "Observation:\nThis is a nice observation" in observation_message.content[0]["text"]


def test_action_step_to_messages_no_tool_calls_with_observations():
    action_step = ActionStep(
        model_input_messages=None,
        tool_calls=None,
        timing=Timing(start_time=0.0, end_time=1.0),
        step_number=1,
        error=None,
        model_output_message=None,
        model_output=None,
        observations="This is an observation.",
        observations_images=None,
        action_output=None,
        token_usage=TokenUsage(input_tokens=10, output_tokens=20),
    )
    messages = action_step.to_messages()
    assert len(messages) == 1
    observation_message = messages[0]
    assert observation_message.role == MessageRole.TOOL_RESPONSE
    assert "Observation:\nThis is an observation." in observation_message.content[0]["text"]


def test_planning_step_to_messages():
    planning_step = PlanningStep(
        model_input_messages=[ChatMessage(role=MessageRole.USER, content="Hello")],
        model_output_message=ChatMessage(role=MessageRole.ASSISTANT, content="Plan"),
        plan="This is a plan.",
        timing=Timing(start_time=0.0, end_time=1.0),
    )
    messages = planning_step.to_messages(summary_mode=False)
    assert len(messages) == 2
    for message in messages:
        assert isinstance(message, ChatMessage)
        assert isinstance(message.content, list)
        assert len(message.content) == 1
        for content in message.content:
            assert isinstance(content, dict)
            assert "type" in content
            assert "text" in content
    assert messages[0].role == MessageRole.ASSISTANT
    assert messages[1].role == MessageRole.USER


def test_task_step_to_messages():
    task_step = TaskStep(task="This is a task.", task_images=[Image.new("RGB", (100, 100))])
    messages = task_step.to_messages(summary_mode=False)
    assert len(messages) == 1
    for message in messages:
        assert isinstance(message, ChatMessage)
        assert message.role == MessageRole.USER
        assert isinstance(message.content, list)
        assert len(message.content) == 2
        text_content = message.content[0]
        assert isinstance(text_content, dict)
        assert "type" in text_content
        assert "text" in text_content
        for image_content in message.content[1:]:
            assert isinstance(image_content, dict)
            assert "type" in image_content
            assert "image" in image_content


def test_system_prompt_step_to_messages():
    system_prompt_step = SystemPromptStep(system_prompt="This is a system prompt.")
    messages = system_prompt_step.to_messages(summary_mode=False)
    assert len(messages) == 1
    for message in messages:
        assert isinstance(message, ChatMessage)
        assert message.role == MessageRole.SYSTEM
        assert isinstance(message.content, list)
        assert len(message.content) == 1
        for content in message.content:
            assert isinstance(content, dict)
            assert "type" in content
            assert "text" in content


def test_memory_step_json_serialization():
    """Test that memory steps can be JSON serialized without raw fields."""

    # Create a mock ChatCompletion-like object (this is what was causing the error)
    class MockChatCompletion:
        def __init__(self):
            self.id = "chatcmpl-test"
            self.choices = []

    # Create a ChatMessage with raw field containing the non-serializable object
    chat_message = ChatMessage(role=MessageRole.ASSISTANT, content="Test response", raw=MockChatCompletion())

    # Test ActionStep serialization
    action_step = ActionStep(
        step_number=1,
        timing=Timing(start_time=123456, end_time=123457),
        model_output_message=chat_message,
        model_input_messages=[chat_message],
    )

    step_dict = action_step.dict()
    json_str = json.dumps(step_dict)
    # Raw field should be present but serializable
    assert "raw" in json_str
    assert "MockChatCompletion" in json_str

    # Test PlanningStep serialization
    planning_step = PlanningStep(
        model_input_messages=[chat_message],
        model_output_message=chat_message,
        plan="Test plan",
        timing=Timing(start_time=123456, end_time=123457),
    )

    planning_dict = planning_step.dict()
    json_str = json.dumps(planning_dict)
    # Raw field should be present but serializable
    assert "raw" in json_str
    assert "MockChatCompletion" in json_str


class TestMemorySerialization:
    def test_action_step_round_trip(self):
        image = Image.new("RGB", (4, 4), color="red")
        step = ActionStep(
            step_number=3,
            timing=Timing(start_time=1.0, end_time=3.5),
            model_input_messages=[ChatMessage(role=MessageRole.USER, content="Hello")],
            tool_calls=[ToolCall(name="python_interpreter", arguments='{"not": "parsed"}', id="call_3")],
            error=AgentExecutionError.from_dict({"type": "AgentExecutionError", "message": "Division by zero"}),
            model_output_message=ChatMessage(
                role=MessageRole.ASSISTANT,
                content="Thought",
                raw=object(),
                token_usage=TokenUsage(input_tokens=10, output_tokens=20),
            ),
            model_output="Thought",
            code_action="print(1 / 0)",
            observations="Execution logs:",
            observations_images=[image],
            action_output=(1, {2, 3}),
            token_usage=TokenUsage(input_tokens=10, output_tokens=20),
            is_final_answer=True,
        )

        restored = ActionStep.from_dict(json.loads(json.dumps(step.to_dict())))

        assert restored.step_number == 3
        assert restored.timing.duration == 2.5
        assert restored.model_input_messages is None  # Not serialized
        assert restored.tool_calls == step.tool_calls
        assert isinstance(restored.error, AgentExecutionError)
        assert str(restored.error) == "Division by zero"
        assert restored.model_output_message.role == MessageRole.ASSISTANT
        assert restored.model_output_message.content == "Thought"
        assert restored.model_output_message.token_usage == TokenUsage(input_tokens=10, output_tokens=20)
        assert restored.code_action == "print(1 / 0)"
        assert restored.observations_images[0].tobytes() == image.tobytes()
        assert restored.action_output == (1, {2, 3})
        assert restored.token_usage.total_tokens == 30
        assert restored.is_final_answer is True
        # The messages given to the model are the same, except for the image object
        for message, expected_message in zip(restored.to_messages(), step.to_messages()):
            assert message.role == expected_message.role
            if message.content[0]["type"] == "text":
                assert message.content == expected_message.content

    def test_action_step_with_unserializable_output(self):
        step = ActionStep(step_number=1, timing=Timing(start_time=1.0), action_output=object)
        restored = ActionStep.from_dict(json.loads(json.dumps(step.to_dict())))
        assert restored.action_output == str(object)

    def test_agent_error_from_dict_does_not_need_logger(self):
        error = AgentMaxStepsError.from_dict({"type": "AgentMaxStepsError", "message": "Reached max steps."})
        assert isinstance(error, AgentMaxStepsError)
        assert error.dict() == {"type": "AgentMaxStepsError", "message": "Reached max steps."}

    def test_agent_memory_round_trip(self):
        memory = AgentMemory(system_prompt="You are an agent.")
        memory.steps = [
            TaskStep(task="Do it", task_images=[Image.new("RGB", (2, 2))]),
            PlanningStep(
                model_input_messages=[ChatMessage(role=MessageRole.USER, content="Plan")],
                model_output_message=ChatMessage(role=MessageRole.ASSISTANT, content="The plan"),
                plan="The plan",
                timing=Timing(start_time=1.0, end_time=2.0),
                token_usage=TokenUsage(input_tokens=1, output_tokens=2),
            ),
            ActionStep(step_number=1, timing=Timing(start_time=2.0, end_time=3.0), observations="Done"),
        ]

        restored = AgentMemory.from_dict(json.loads(json.dumps(memory.to_dict())))

        assert restored.system_prompt.system_prompt == "You are an agent."
        assert [type(step) for step in restored.steps] == [TaskStep, PlanningStep, ActionStep]
        assert restored.steps[0].task == "Do it"
        assert restored.steps[0].task_images[0].size == (2, 2)
        assert restored.steps[1].plan == "The plan"
        assert restored.steps[1].token_usage == TokenUsage(input_tokens=1, output_tokens=2)
        assert restored.steps[2].observations == "Done"

    def test_agent_memory_from_dict_unknown_step(self):
        with pytest.raises(ValueError, match="Unknown memory step class 'EvilStep'"):
            AgentMemory.from_dict({"system_prompt": "", "steps": [{"type": "EvilStep", "data": {}}]})
