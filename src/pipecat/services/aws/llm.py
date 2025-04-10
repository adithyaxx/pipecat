#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import base64
import copy
import io
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Union

import boto3
from botocore.config import Config
import httpx
from loguru import logger
from PIL import Image
from pydantic import BaseModel, Field

from pipecat.adapters.services.anthropic_adapter import AnthropicLLMAdapter
from pipecat.frames.frames import (
    Frame,
    FunctionCallCancelFrame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesFrame,
    LLMTextFrame,
    LLMUpdateSettingsFrame,
    UserImageRawFrame,
    VisionImageRawFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantContextAggregator,
    LLMUserContextAggregator,
)
from pipecat.processors.aggregators.bedrock_llm_context import (
    BedrockLLMContext,
    BedrockLLMContextFrame,
)
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAILLMContextFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_services import LLMService


@dataclass
class BedrockContextAggregatorPair:
    _user: "BedrockUserContextAggregator"
    _assistant: "BedrockAssistantContextAggregator"

    def user(self) -> "BedrockUserContextAggregator":
        return self._user

    def assistant(self) -> "BedrockAssistantContextAggregator":
        return self._assistant


class BedrockUserContextAggregator(LLMUserContextAggregator):
    def __init__(
        self,
        context: BedrockLLMContext,
        aggregation_timeout: float = 1.0,
        **kwargs,
    ):
        super().__init__(context=context, aggregation_timeout=aggregation_timeout, **kwargs)
    
    def get_context_frame(self) -> BedrockLLMContextFrame:
        return BedrockLLMContextFrame(context=self.context)


class BedrockAssistantContextAggregator(LLMAssistantContextAggregator):
    def __init__(
            self, 
            context: BedrockLLMContext, 
            *, 
            expect_stripped_words: bool = True, 
            **kwargs
    ):
        super().__init__(context=context, expect_stripped_words=expect_stripped_words, **kwargs)
    
    def get_context_frame(self) -> BedrockLLMContextFrame:
        return BedrockLLMContextFrame(context=self.context)
        
    async def handle_function_call_in_progress(self, frame: FunctionCallInProgressFrame):
        # Format tool use according to Bedrock API
        self._context.add_message(
            {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": frame.tool_call_id,
                            "name": frame.function_name,
                            "input": frame.arguments if frame.arguments else {}
                        }
                    }
                ],
            }
        )
        self._context.add_message(
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": frame.tool_call_id,
                            "content": [
                                {
                                    "text": "IN_PROGRESS"
                                }
                            ],
                        }
                    }
                ],
            }
        )

    async def handle_function_call_result(self, frame: FunctionCallResultFrame):
        if frame.result:
            result = json.dumps(frame.result)
            await self._update_function_call_result(frame.function_name, frame.tool_call_id, result)
        else:
            await self._update_function_call_result(
                frame.function_name, frame.tool_call_id, "COMPLETED"
            )

    async def handle_function_call_cancel(self, frame: FunctionCallCancelFrame):
        await self._update_function_call_result(
            frame.function_name, frame.tool_call_id, "CANCELLED"
        )

    async def _update_function_call_result(
        self, function_name: str, tool_call_id: str, result: Any
    ):
        for message in self._context.messages:
            if message["role"] == "user":
                for content in message["content"]:
                    if (
                        isinstance(content, dict)
                        and content.get("toolResult")
                        and content["toolResult"]["toolUseId"] == tool_call_id
                    ):
                        content["toolResult"]["content"] = [{"text": result}]

    async def handle_user_image_frame(self, frame: UserImageRawFrame):
        await self._update_function_call_result(
            frame.request.function_name, frame.request.tool_call_id, "COMPLETED"
        )
        self._context.add_image_frame_message(
            format=frame.format,
            size=frame.size,
            image=frame.image,
            text=frame.request.context,
        )


class BedrockLLMService(LLMService):
    """This class implements inference with AWS Bedrock models including Amazon Nova and Anthropic Claude.
    
    Requires AWS credentials to be configured in the environment or through boto3 configuration.
    """
    class InputParams(BaseModel):
        max_tokens: Optional[int] = Field(default_factory=lambda: 4096, ge=1)
        temperature: Optional[float] = Field(default_factory=lambda: 0.7, ge=0.0, le=1.0)
        top_p: Optional[float] = Field(default_factory=lambda: 0.999, ge=0.0, le=1.0)
        stop_sequences: Optional[List[str]] = Field(default_factory=lambda: [])
        latency: Optional[str] = Field(default_factory=lambda: "standard")
        additional_model_request_fields: Optional[Dict[str, Any]] = Field(default_factory=dict)

    def __init__(
        self,
        *,
        aws_access_key: Optional[str] = None,
        aws_secret_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        aws_region: str = "us-east-1",
        model: str,
        params: InputParams = InputParams(),
        client_config: Optional[Config] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        # Initialize the Bedrock client
        if not client_config:
            client_config = Config(
                connect_timeout=300,  # 5 minutes
                read_timeout=300,     # 5 minutes
                retries={'max_attempts': 3}
            )
        session = boto3.Session(
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            aws_session_token=aws_session_token,
            region_name=aws_region
        )
        self._client = session.client(
            service_name='bedrock-runtime',
            config=client_config
        )
        
        self.set_model_name(model)
        self._settings = {
            "max_tokens": params.max_tokens,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "latency": params.latency,
            "additional_model_request_fields": params.additional_model_request_fields if isinstance(params.additional_model_request_fields, dict) else {},
        }
        
        logger.info(f"Using AWS Bedrock model: {model}")

    def can_generate_metrics(self) -> bool:
        return True

    def create_context_aggregator(
        self,
        context: BedrockLLMContext,
        *,
        user_kwargs: Mapping[str, Any] = {},
        assistant_kwargs: Mapping[str, Any] = {},
    ) -> BedrockContextAggregatorPair:
        """Create an instance of BedrockContextAggregatorPair from an
        OpenAILLMContext. Constructor keyword arguments for both the user and
        assistant aggregators can be provided.

        Args:
            context (OpenAILLMContext): The LLM context.
            user_kwargs (Mapping[str, Any], optional): Additional keyword
                arguments for the user context aggregator constructor. Defaults
                to an empty mapping.
            assistant_kwargs (Mapping[str, Any], optional): Additional keyword
                arguments for the assistant context aggregator
                constructor. Defaults to an empty mapping.

        Returns:
            BedrockContextAggregatorPair: A pair of context aggregators, one
            for the user and one for the assistant, encapsulated in an
            BedrockContextAggregatorPair.
        """
        context.set_llm_adapter(self.get_llm_adapter())

        if isinstance(context, OpenAILLMContext) and not isinstance(context, BedrockLLMContext):
            context = BedrockLLMContext.from_openai_context(context)
                
        user = BedrockUserContextAggregator(context, **user_kwargs)
        assistant = BedrockAssistantContextAggregator(context, **assistant_kwargs)
        return BedrockContextAggregatorPair(_user=user, _assistant=assistant)

    async def _process_context(self, context: BedrockLLMContext):
        # Usage tracking
        prompt_tokens = 0
        completion_tokens = 0
        completion_tokens_estimate = 0
        cache_read_input_tokens = 0
        cache_creation_input_tokens = 0
        use_completion_tokens_estimate = False

        try:
            await self.push_frame(LLMFullResponseStartFrame())
            await self.start_processing_metrics()

            # logger.debug(
            #     f"{self}: Generating chat with Bedrock model {self.model_name} | [{context.get_messages_for_logging()}]"
            # )

            await self.start_ttfb_metrics()
            
            # Set up inference config
            inference_config = {
                "maxTokens": self._settings["max_tokens"],
                "temperature": self._settings["temperature"],
                "topP": self._settings["top_p"],
            }
            
            # Prepare request parameters
            request_params = {
                "modelId": self.model_name,
                "messages": context.messages,
                "inferenceConfig": inference_config,
                "additionalModelRequestFields": self._settings["additional_model_request_fields"]
            }
            
            # Add system message
            request_params["system"] = context.system
                
            # Add tools if present
            if context.tools:
                tool_config = {
                    "tools": context.tools
                }
                
                # Add tool_choice if specified
                if context.tool_choice:
                    if context.tool_choice == "auto":
                        tool_config["toolChoice"] = {"auto": {}}
                    elif context.tool_choice == "none":
                        # Skip adding toolChoice for "none"
                        pass
                    elif isinstance(context.tool_choice, dict) and "function" in context.tool_choice:
                        tool_config["toolChoice"] = {
                            "tool": {
                                "name": context.tool_choice["function"]["name"]
                            }
                        }
                
                request_params["toolConfig"] = tool_config
            
            # Add performance config if latency is specified
            if self._settings["latency"] in ["standard", "optimized"]:
                request_params["performanceConfig"] = {
                    "latency": self._settings["latency"]
                }
            
            logger.debug(f"Calling Bedrock model with: {request_params}")
            
            # Call Bedrock with streaming
            response = self._client.converse_stream(**request_params)
            
            await self.stop_ttfb_metrics()
            
            # Process the streaming response
            tool_use_block = None
            json_accumulator = ""
            
            for event in response["stream"]:
                # Handle text content
                if "contentBlockDelta" in event:
                    delta = event["contentBlockDelta"]["delta"]
                    if "text" in delta:
                        await self.push_frame(LLMTextFrame(delta["text"]))
                        completion_tokens_estimate += self._estimate_tokens(delta["text"])
                    elif "toolUse" in delta and "input" in delta["toolUse"]:
                        # Handle partial JSON for tool use
                        json_accumulator += delta["toolUse"]["input"]
                        completion_tokens_estimate += self._estimate_tokens(delta["toolUse"]["input"])
                
                # Handle tool use start
                elif "contentBlockStart" in event:
                    content_block_start = event["contentBlockStart"]['start']
                    if "toolUse" in content_block_start:
                        tool_use_block = {
                            "id": content_block_start["toolUse"].get("toolUseId", ""),
                            "name": content_block_start["toolUse"].get("name", "")
                        }
                        json_accumulator = ""
                
                # Handle message completion with tool use
                elif "messageStop" in event and "stopReason" in event["messageStop"]:
                    if event["messageStop"]["stopReason"] == "tool_use" and tool_use_block:
                        try:
                            arguments = json.loads(json_accumulator) if json_accumulator else {}
                            await self.call_function(
                                context=context,
                                tool_call_id=tool_use_block["id"],
                                function_name=tool_use_block["name"],
                                arguments=arguments,
                            )
                        except json.JSONDecodeError:
                            logger.error(f"Failed to parse tool arguments: {json_accumulator}")
                
                # Handle usage metrics if available
                if "metadata" in event and "usage" in event["metadata"]:
                    usage = event["metadata"]["usage"]
                    prompt_tokens += usage.get("inputTokens", 0)
                    completion_tokens += usage.get("outputTokens", 0)
                    cache_read_input_tokens += usage.get("cacheReadInputTokens", 0)
                    cache_creation_input_tokens += usage.get("cacheWriteInputTokens", 0)

        except asyncio.CancelledError:
            # If we're interrupted, we won't get a complete usage report. So set our flag to use the
            # token estimate. The reraise the exception so all the processors running in this task
            # also get cancelled.
            use_completion_tokens_estimate = True
            raise
        except httpx.TimeoutException:
            await self._call_event_handler("on_completion_timeout")
        except Exception as e:
            logger.exception(f"{self} exception: {e}")
        finally:
            await self.stop_processing_metrics()
            await self.push_frame(LLMFullResponseEndFrame())
            comp_tokens = (
                completion_tokens
                if not use_completion_tokens_estimate
                else completion_tokens_estimate
            )
            await self._report_usage_metrics(
                prompt_tokens=prompt_tokens,
                completion_tokens=comp_tokens,
                cache_read_input_tokens=cache_read_input_tokens,
                cache_creation_input_tokens=cache_creation_input_tokens
            )

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        context = None
        if isinstance(frame, OpenAILLMContextFrame):
            context = BedrockLLMContext.upgrade_to_bedrock(frame.context)
        elif isinstance(frame, LLMMessagesFrame):
            context = BedrockLLMContext.from_messages(frame.messages)
        elif isinstance(frame, VisionImageRawFrame):
            # This is only useful in very simple pipelines because it creates
            # a new context. Generally we want a context manager to catch
            # UserImageRawFrames coming through the pipeline and add them
            # to the context.
            context = BedrockLLMContext.from_image_frame(frame)
        elif isinstance(frame, LLMUpdateSettingsFrame):
            await self._update_settings(frame.settings)
        else:
            await self.push_frame(frame, direction)

        if context:
            await self._process_context(context)

    def _estimate_tokens(self, text: str) -> int:
        return int(len(re.split(r"[^\w]+", text)) * 1.3)

    async def _report_usage_metrics(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        cache_read_input_tokens: int,
        cache_creation_input_tokens: int
    ):
        if prompt_tokens or completion_tokens:
            tokens = LLMTokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                cache_read_input_tokens=cache_read_input_tokens,
                cache_creation_input_tokens=cache_creation_input_tokens
            )
            await self.start_llm_usage_metrics(tokens)
