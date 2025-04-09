#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import base64
import copy
import io
import json
from dataclasses import dataclass
from typing import Any, List, Optional
from loguru import logger
from PIL import Image

from pipecat.frames.frames import Frame, VisionImageRawFrame
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext
)


class BedrockLLMContext(OpenAILLMContext):
    def __init__(
        self,
        messages: Optional[List[dict]] = None,
        tools: Optional[List[dict]] = None,
        tool_choice: Optional[dict] = None,
        *,
        system: Optional[str] = None,
    ):
        super().__init__(messages=messages, tools=tools, tool_choice=tool_choice)
        self.system = system

    @staticmethod
    def upgrade_to_bedrock(obj: OpenAILLMContext) -> "BedrockLLMContext":
        logger.debug(f"Upgrading to Bedrock: {obj}")
        if isinstance(obj, OpenAILLMContext) and not isinstance(obj, BedrockLLMContext):
            obj.__class__ = BedrockLLMContext
            obj._restructure_from_openai_messages()
        else:
            obj._restructure_from_bedrock_messages()
        return obj

    @classmethod
    def from_openai_context(cls, openai_context: OpenAILLMContext):
        logger.debug("from_openai_context called")
        self = cls(
            messages=openai_context.messages,
            tools=openai_context.tools,
            tool_choice=openai_context.tool_choice,
        )
        self.set_llm_adapter(openai_context.get_llm_adapter())
        self._restructure_from_openai_messages()
        return self

    @classmethod
    def from_messages(cls, messages: List[dict]) -> "BedrockLLMContext":
        self = cls(messages=messages)
        # self._restructure_from_openai_messages()
        return self

    @classmethod
    def from_image_frame(cls, frame: VisionImageRawFrame) -> "BedrockLLMContext":
        context = cls()
        context.add_image_frame_message(
            format=frame.format, size=frame.size, image=frame.image, text=frame.text
        )
        return context

    def set_messages(self, messages: List):
        self._messages[:] = messages
        # self._restructure_from_openai_messages()

    # convert a message in Bedrock format into one or more messages in OpenAI format
    def to_standard_messages(self, obj):
        """Convert Bedrock message format to standard structured format.

        Handles text content and function calls for both user and assistant messages.

        Args:
            obj: Message in Bedrock format:
                {
                    "role": "user/assistant",
                    "content": [{"text": str} | {"toolUse": {...}} | {"toolResult": {...}}]
                }

        Returns:
            List of messages in standard format:
            [
                {
                    "role": "user/assistant/tool",
                    "content": [{"type": "text", "text": str}]
                }
            ]
        """
        role = obj.get("role")
        content = obj.get("content")
        
        if role == "assistant":
            if isinstance(content, str):
                return [{"role": role, "content": [{"type": "text", "text": content}]}]
            elif isinstance(content, list):
                text_items = []
                tool_items = []
                for item in content:
                    if "text" in item:
                        text_items.append({"type": "text", "text": item["text"]})
                    elif "toolUse" in item:
                        tool_use = item["toolUse"]
                        tool_items.append(
                            {
                                "type": "function",
                                "id": tool_use["toolUseId"],
                                "function": {
                                    "name": tool_use["name"],
                                    "arguments": json.dumps(tool_use["input"]),
                                },
                            }
                        )
                messages = []
                if text_items:
                    messages.append({"role": role, "content": text_items})
                if tool_items:
                    messages.append({"role": role, "tool_calls": tool_items})
                return messages
        elif role == "user":
            if isinstance(content, str):
                return [{"role": role, "content": [{"type": "text", "text": content}]}]
            elif isinstance(content, list):
                text_items = []
                tool_items = []
                for item in content:
                    if "text" in item:
                        text_items.append({"type": "text", "text": item["text"]})
                    elif "toolResult" in item:
                        tool_result = item["toolResult"]
                        # Extract content from toolResult
                        result_content = ""
                        if isinstance(tool_result["content"], list):
                            for content_item in tool_result["content"]:
                                if "text" in content_item:
                                    result_content = content_item["text"]
                                elif "json" in content_item:
                                    result_content = json.dumps(content_item["json"])
                        else:
                            result_content = tool_result["content"]
                            
                        tool_items.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_result["toolUseId"],
                                "content": result_content,
                            }
                        )
                messages = []
                if text_items:
                    messages.append({"role": role, "content": text_items})
                messages.extend(tool_items)
                return messages

    def from_standard_message(self, message):
        """Convert standard format message to Bedrock format.

        Handles conversion of text content, tool calls, and tool results.
        Empty text content is converted to "(empty)".

        Args:
            message: Message in standard format:
                {
                    "role": "user/assistant/tool",
                    "content": str | [{"type": "text", ...}],
                    "tool_calls": [{"id": str, "function": {"name": str, "arguments": str}}]
                }

        Returns:
            Message in Bedrock format:
            {
                "role": "user/assistant",
                "content": [
                    {"text": str} |
                    {"toolUse": {"toolUseId": str, "name": str, "input": dict}} |
                    {"toolResult": {"toolUseId": str, "content": [...], "status": str}}
                ]
            }
        """
        if message["role"] == "tool":
            # Try to parse the content as JSON if it looks like JSON
            try:
                if message["content"].strip().startswith('{') and message["content"].strip().endswith('}'):
                    content_json = json.loads(message["content"])
                    tool_result_content = [{"json": content_json}]
                else:
                    tool_result_content = [{"text": message["content"]}]
            except:
                tool_result_content = [{"text": message["content"]}]
                
            return {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": message["tool_call_id"],
                            "content": tool_result_content
                        },
                    },
                ],
            }
            
        if message.get("tool_calls"):
            tc = message["tool_calls"]
            ret = {"role": "assistant", "content": []}
            for tool_call in tc:
                function = tool_call["function"]
                arguments = json.loads(function["arguments"])
                new_tool_use = {
                    "toolUse": {
                        "toolUseId": tool_call["id"],
                        "name": function["name"],
                        "input": arguments,
                    }
                }
                ret["content"].append(new_tool_use)
            return ret
            
        # Handle text content
        content = message.get("content")
        if isinstance(content, str):
            if content == "":
                return {"role": message["role"], "content": [{"text": "(empty)"}]}
            else:
                return {"role": message["role"], "content": [{"text": content}]}
        elif isinstance(content, list):
            new_content = []
            for item in content:
                if item.get("type", "") == "text":
                    text_content = item["text"] if item["text"] != "" else "(empty)"
                    new_content.append({"text": text_content})
            return {"role": message["role"], "content": new_content}
            
        return message

    def add_image_frame_message(
        self, *, format: str, size: tuple[int, int], image: bytes, text: str = None
    ):
        buffer = io.BytesIO()
        Image.frombytes(format, size, image).save(buffer, format="JPEG")
        encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # Image should be the first content block in the message
        content = [
            {
                "type": "image",
                "format": "jpeg",
                "source": {
                    "bytes": encoded_image
                }
            }
        ]
        if text:
            content.append({"text": text})
        self.add_message({"role": "user", "content": content})

    def add_message(self, message):
        try:
            if self.messages:
                # Bedrock requires that roles alternate. If this message's role is the same as the
                # last message, we should add this message's content to the last message.
                if self.messages[-1]["role"] == message["role"]:
                    # if the last message has just a content string, convert it to a list
                    # in the proper format
                    if isinstance(self.messages[-1]["content"], str):
                        self.messages[-1]["content"] = [
                            {"text": self.messages[-1]["content"]}
                        ]
                    # if this message has just a content string, convert it to a list
                    # in the proper format
                    if isinstance(message["content"], str):
                        message["content"] = [{"text": message["content"]}]
                    # append the content of this message to the last message
                    self.messages[-1]["content"].extend(message["content"])
                else:
                    self.messages.append(message)
            else:
                self.messages.append(message)
        except Exception as e:
            logger.error(f"Error adding message: {e}")

    def _restructure_from_bedrock_messages(self):
        """Restructure messages in Bedrock format by handling system messages, 
        merging consecutive messages with the same role, and ensuring proper content formatting.
        """
        # Handle system message if present at the beginning
        logger.debug(f"_restructure_from_bedrock_messages: {self.messages}")
        if self.messages and self.messages[0]["role"] == "system":
            if len(self.messages) == 1:
                self.messages[0]["role"] = "user"
            else:
                system_content = self.messages.pop(0)["content"]
                if isinstance(system_content, str):
                    system_content = [{"text": system_content}]
                
                if self.system:
                    if isinstance(self.system, str):
                        self.system = [{"text": self.system}]
                    self.system.extend(system_content)
                else:
                    self.system = system_content

        # Ensure content is properly formatted
        for msg in self.messages:
            if isinstance(msg["content"], str):
                msg["content"] = [{"text": msg["content"]}]
            elif not msg["content"]:
                msg["content"] = [{"text": "(empty)"}]
            elif isinstance(msg["content"], list):
                for idx, item in enumerate(msg["content"]):
                    if isinstance(item, dict) and "text" in item and item["text"] == "":
                        item["text"] = "(empty)"
                    elif isinstance(item, str) and item == "":
                        msg["content"][idx] = {"text": "(empty)"}

        # Merge consecutive messages with the same role
        merged_messages = []
        for msg in self.messages:
            if merged_messages and merged_messages[-1]["role"] == msg["role"]:
                merged_messages[-1]["content"].extend(msg["content"])
            else:
                merged_messages.append(msg)
        
        self.messages.clear()
        self.messages.extend(merged_messages)

    def _restructure_from_openai_messages(self):
        logger.debug(f"_restructure_from_openai_messages: {self.messages}")
        # first, map across self._messages calling self.from_standard_message(m) to modify messages in place
        try:
            self._messages[:] = [self.from_standard_message(m) for m in self._messages]
        except Exception as e:
            logger.error(f"Error mapping messages: {e}")

        # See if we should pull the system message out of our context.messages list. (For
        # compatibility with Open AI messages format.)
        if self.messages and self.messages[0]["role"] == "system":
            if len(self.messages) == 1:
                # If we have only have a system message in the list, all we can really do
                # without introducing too much magic is change the role to "user".
                self.messages[0]["role"] = "user"
            else:
                # If we have more than one message, we'll pull the system message out of the
                # list.
                self.system = self.messages[0]["content"]
                self.messages.pop(0)

        # Merge consecutive messages with the same role.
        i = 0
        while i < len(self.messages) - 1:
            current_message = self.messages[i]
            next_message = self.messages[i + 1]
            if current_message["role"] == next_message["role"]:
                # Convert content to list of dictionaries if it's a string
                if isinstance(current_message["content"], str):
                    current_message["content"] = [
                        {"type": "text", "text": current_message["content"]}
                    ]
                if isinstance(next_message["content"], str):
                    next_message["content"] = [{"type": "text", "text": next_message["content"]}]
                # Concatenate the content
                current_message["content"].extend(next_message["content"])
                # Remove the next message from the list
                self.messages.pop(i + 1)
            else:
                i += 1

        # Avoid empty content in messages
        for message in self.messages:
            if isinstance(message["content"], str) and message["content"] == "":
                message["content"] = "(empty)"
            elif isinstance(message["content"], list) and len(message["content"]) == 0:
                message["content"] = [{"type": "text", "text": "(empty)"}]

    def get_messages_for_persistent_storage(self):
        messages = super().get_messages_for_persistent_storage()
        if self.system:
            messages.insert(0, {"role": "system", "content": self.system})
        return messages

    def get_messages_for_logging(self) -> str:
        msgs = []
        for message in self.messages:
            msg = copy.deepcopy(message)
            if "content" in msg:
                if isinstance(msg["content"], list):
                    for item in msg["content"]:
                        if item.get("image"):
                            item["source"]["bytes"] = "..."
            msgs.append(msg)
        return json.dumps(msgs)


@dataclass
class BedrockLLMContextFrame(Frame):
    """Like an LLMMessagesFrame, but with extra context specific to the OpenAI
    API. The context in this message is also mutable, and will be changed by the
    OpenAIContextAggregator frame processor.

    """

    context: BedrockLLMContext