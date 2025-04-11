#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import base64
import json
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Set, Tuple, Union

from aws_sdk_bedrock_runtime.client import BedrockRuntimeClient, InvokeModelWithBidirectionalStreamOperationInput
from aws_sdk_bedrock_runtime.models import InvokeModelWithBidirectionalStreamInputChunk, BidirectionalInputPayloadPart
from aws_sdk_bedrock_runtime.config import Config, HTTPAuthSchemeResolver, SigV4AuthScheme
from loguru import logger
from smithy_aws_core.credentials_resolvers.environment import EnvironmentCredentialsResolver

from pipecat.adapters.services.bedrock_adapter import BedrockLLMAdapter
from pipecat.frames.frames import (
    InputAudioRawFrame,
    Frame,
    FunctionCallCancelFrame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesFrame,
    LLMTextFrame,
    LLMUpdateSettingsFrame,
    OutputAudioRawFrame,
    StartFrame,
    EndFrame,
    CancelFrame
)
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.processors.aggregators.bedrock_llm_context import (
    BedrockLLMContext,
    BedrockLLMContextFrame,
)
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContextFrame
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantContextAggregator,
    LLMUserContextAggregator,
)
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.ai_services import LLMService
from pipecat.services.aws.llm import BedrockLLMContext, BedrockContextAggregatorPair

from .events import (
    SessionProperties,
    ConversationItemCreateEvent,
)
from .context import (
    BedrockRealtimeAssistantContextAggregator,
    BedrockRealtimeLLMContext,
    BedrockRealtimeUserContextAggregator,
)


@dataclass
class BedrockRealtimeContextAggregatorPair:
    _user: BedrockRealtimeUserContextAggregator
    _assistant: BedrockRealtimeAssistantContextAggregator

    def user(self) -> BedrockRealtimeUserContextAggregator:
        return self._user

    def assistant(self) -> BedrockRealtimeAssistantContextAggregator:
        return self._assistant


class BedrockRealtimeLLMService(LLMService):
    """
    Implementation of a real-time LLM service using Amazon Bedrock's Nova Sonic model.
    
    This service supports bidirectional streaming for audio input/output and text,
    as well as tool use functionality.
    """
    adapter_class = BedrockLLMAdapter
    
    def __init__(
        self,
        *,
        aws_access_key: Optional[str] = None,
        aws_secret_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        aws_region: str = "us-east-1",
        model: str = "amazon.nova-sonic-v1:0",
        session_properties: SessionProperties = SessionProperties(),
        **kwargs,
    ):
        """
        Initialize the Bedrock Realtime LLM Service.
        
        Args:
            aws_access_key: AWS access key ID
            aws_secret_key: AWS secret access key
            aws_session_token: AWS session token
            aws_region: AWS region
            model: Bedrock model ID to use
            session_properties: Configuration for the session
        """
        super().__init__(**kwargs)
        
        self.model_id = model
        self.aws_region = aws_region
        self.session_properties = session_properties
        self._context = None
        
        # AWS credentials
        self.aws_access_key = aws_access_key
        self.aws_secret_key = aws_secret_key
        self.aws_session_token = aws_session_token
        
        # Initialize client
        self._initialize_client()
        
        # Stream state
        self.stream_response = None
        self.audio_capture_task = None
        self.is_active = False
        self.prompt_name = str(uuid.uuid4())
        self.audio_content_name = str(uuid.uuid4())
        self.response_task = None
        self.audio_input_queue = asyncio.Queue()
        self.barge_in = False
        
        # Function call handling
        self._function_handlers: Dict[Optional[str], Callable] = {}
        
        # Audio settings
        self.input_sample_rate = 16000
        self.output_sample_rate = 24000
        self.channels = 1
        self.sample_size_bits = 16
        self.audio_input_started = False
        
        # Tool use tracking
        self.toolUseId = ""
        self.toolName = ""
        
        # API session state
        self._session_creation_lock = asyncio.Lock()
        self._session_creation_in_progress = False
        self._api_session_ready = False
        self._run_llm_when_api_session_ready = False
        self._messages_added_manually = {}
        
        logger.info(f"Initialized BedrockRealtimeLLMService with model {self.model_id}")
    
    def _initialize_client(self):
        """Initialize the Bedrock client."""
        config = Config(
            endpoint_uri=f"https://bedrock-runtime.{self.aws_region}.amazonaws.com",
            region=self.aws_region,
            aws_credentials_identity_resolver=EnvironmentCredentialsResolver(),
            http_auth_scheme_resolver=HTTPAuthSchemeResolver(),
            http_auth_schemes={"aws.auth#sigv4": SigV4AuthScheme()}
        )
        self.bedrock_client = BedrockRuntimeClient(config=config)
    
    def register_function(
        self, function_name: Optional[str], handler: Callable
    ) -> None:
        """
        Register a function handler for tool calls.
        
        Args:
            function_name: The name of the function to register, or None to register a default handler
            handler: The function handler
        """
        self._function_handlers[function_name] = handler
    
    def create_context_aggregator(
        self,
        context: BedrockLLMContext,
        *,
        user_kwargs: Mapping[str, Any] = {},
        assistant_kwargs: Mapping[str, Any] = {},
    ) -> BedrockContextAggregatorPair:
        """
        Create a context aggregator pair for the Bedrock Realtime LLM Service.
        
        Args:
            context: The LLM context
            user_kwargs: Additional keyword arguments for the user context aggregator
            assistant_kwargs: Additional keyword arguments for the assistant context aggregator
            
        Returns:
            A pair of context aggregators for user and assistant
        """
        context.set_llm_adapter(self.get_llm_adapter())

        if isinstance(context, OpenAILLMContext) and not isinstance(context, BedrockLLMContext):
            context = BedrockLLMContext.from_openai_context(context)
        
        user = BedrockRealtimeUserContextAggregator(context, **user_kwargs)
        assistant = BedrockRealtimeAssistantContextAggregator(context, **assistant_kwargs)
        return BedrockContextAggregatorPair(_user=user, _assistant=assistant)
    
    async def initialize_stream(self):
        try:
            await self.create_session()
            logger.info("Bedrock stream initialized successfully")
            return True
        except Exception as e:
            self.is_active = False
            logger.error(f"Failed to initialize Bedrock stream: {str(e)}")
            return False
    
    async def send_raw_event(self, event_json):
        """Send a raw event JSON to the Bedrock stream."""
        event_type = list(json.loads(event_json).get("event", {}).keys())

        if not self.stream_response:
            try:
                logger.debug(f"Stream not initialized or closed | event: {event_type[0]}")
            except Exception:
                logger.debug(f"Stream not initialized or closed | event: {event_json[:100]}...")
            return
        
        event = InvokeModelWithBidirectionalStreamInputChunk(
            value=BidirectionalInputPayloadPart(bytes_=event_json.encode('utf-8'))
        )
        
        try:
            await self.stream_response.input_stream.send(event)
            if event_type[0] == 'promptStart':
                logger.debug(f"Sent event: {event_type[0]} ({json.loads(event_json)['event'][event_type[0]].get('toolConfiguration', 'None') })")
            # elif event_type[0] != 'audioInput': # to avoid spam
            else:
                logger.debug(f"Sent event: {event_type[0]} ({json.loads(event_json)['event'][event_type[0]].get('type', 'None')})")
        except Exception as e:
            logger.error(f"Error sending event: {str(e)}")
    
    async def _process_audio_input(self):
        """Process audio input from the queue and send to Bedrock."""
        # First send audio content start event if not already started
        if not self.audio_input_started:
            await self.send_raw_event(self._create_audio_content_start_event())
            self.audio_input_started = True
        
        while self.is_active:
            try:
                # Get audio data from the queue with a timeout
                try:
                    data = await asyncio.wait_for(self.audio_input_queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    # No data available, check if we're still active and not paused
                    if not self.is_active:
                        logger.debug("No audio data available, breaking")
                        break
                    logger.debug("No audio data available, skipping")
                    continue
                
                audio_bytes = data.get('audio_bytes')
                if not audio_bytes:
                    logger.debug("No audio bytes received")
                    continue
                
                # Base64 encode the audio data
                blob = base64.b64encode(audio_bytes)
                audio_event = self._create_audio_input_event(blob.decode('utf-8'))
                
                # Send the event
                await self.send_raw_event(audio_event)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing audio: {e}")
        
        # If we exited the loop but the stream is still active, send content end
        if self.is_active and self.audio_input_started:
            try:
                await self.send_raw_event(self._create_content_end_event(content_name=self.audio_content_name))
                self.audio_input_started = False
                # Generate a new audio content name for the next turn
                self.audio_content_name = str(uuid.uuid4())
            except Exception as e:
                logger.error(f"Error ending audio content: {e}")
    
    async def _process_responses(self):
        """Process incoming responses from Bedrock."""
        try:
            while self.is_active:
                logger.debug("Listening for incoming responses from Bedrock.")
                try:
                    output = await self.stream_response.await_output()
                    result = await output[1].receive()
                    if result.value and result.value.bytes_:
                        try:
                            response_data = result.value.bytes_.decode('utf-8')
                            json_data = json.loads(response_data)
                            
                            # Handle different response types
                            if 'event' in json_data:
                                event = json_data['event']
                                
                                # Handle text output
                                if 'textOutput' in event:
                                    text_content = event['textOutput']['content']
                                    logger.debug(f"Text Output: {text_content}")
                                    role = event['textOutput']['role']
                                    
                                    # Check for barge-in
                                    if '{ "interrupted" : true }' in text_content:
                                        logger.debug("Barge-in detected. Stopping audio output.")
                                        self.barge_in = True
                                        continue
                                    
                                    # Send text frame
                                    if role == "ASSISTANT":
                                        await self.push_frame(LLMTextFrame(text_content))
                                
                                # Handle audio output
                                elif 'audioOutput' in event:
                                    audio_content = event['audioOutput']['content']
                                    audio_bytes = base64.b64decode(audio_content)
                                    
                                    # Skip audio if barge-in detected
                                    if self.barge_in:
                                        self.barge_in = False
                                        continue
                                    
                                    # Send audio frame
                                    await self.push_frame(OutputAudioRawFrame(
                                        audio=audio_bytes,
                                        sample_rate=self.output_sample_rate,
                                        num_channels=self.channels
                                    ))
                                
                                # Handle tool use
                                elif 'toolUse' in event:
                                    tool_use = event['toolUse']
                                    self.toolUseId = tool_use['toolUseId']
                                    self.toolName = tool_use['toolName']
                                    
                                    # Parse tool input
                                    try:
                                        tool_input = {}
                                        if 'input' in tool_use:
                                            tool_input = tool_use['input']
                                        elif 'content' in tool_use:
                                            tool_input = json.loads(tool_use['content'])
                                    except (json.JSONDecodeError, KeyError):
                                        tool_input = {}
                                    
                                    logger.debug(f"Tool use detected: {self.toolName}, ID: {self.toolUseId}")
                                    
                                    # Call the function
                                    await self.call_function(
                                        context=self._context,
                                        tool_call_id=self.toolUseId,
                                        function_name=self.toolName,
                                        arguments=tool_input
                                    )
                                
                                # Handle usage metrics if available
                                if 'metadata' in json_data and 'usage' in json_data['metadata']:
                                    usage = json_data['metadata']['usage']
                                    prompt_tokens = usage.get('inputTokens', 0)
                                    completion_tokens = usage.get('outputTokens', 0)
                                    
                                    # Report usage metrics
                                    tokens = LLMTokenUsage(
                                        prompt_tokens=prompt_tokens,
                                        completion_tokens=completion_tokens,
                                        total_tokens=prompt_tokens + completion_tokens
                                    )
                                    await self.start_llm_usage_metrics(tokens)
                        
                        except json.JSONDecodeError:
                            logger.error(f"Failed to parse response: {response_data}")
                
                except Exception as e:
                    logger.error(f"Error receiving response: {type(e).__name__} {e}")
                    break
                    
        except Exception as e:
            logger.error(f"Response processing error: {type(e).__name__} {e}")
        finally:
            self.is_active = False
            self._api_session_ready = False
    
    async def reset_conversation(self):
        """Reset the conversation by disconnecting and reconnecting."""
        logger.debug("Resetting conversation")
        if self._context:
            self._context.llm_needs_settings_update = True
            self._context.llm_needs_initial_messages = True
        
        logger.debug("Calling initialize_stream in reset_conversation")
        await self.close_session()
        success = await self.initialize_stream()
        
        # If successful and we have pending messages, process them
        if success and self._context and self._context.llm_needs_initial_messages:
            await self._create_response()

    async def _update_settings(self):
        settings = self.session_properties

        # Check if tools or instructions have changed
        tools_changed = False
        instructions_changed = False
        
        if self._context:
            # Check if tools have changed
            if self._context.tools and settings.tools != self._context.tools:
                settings.tools = self._context.tools
                tools_changed = True
                
            # Check if instructions have changed
            if self._context._session_instructions and settings.instructions != self._context._session_instructions:
                settings.instructions = self._context._session_instructions
                instructions_changed = True
        
        # If tools or instructions changed, we need to restart the session
        if tools_changed or instructions_changed:
            logger.debug(f"Tools or instructions changed. Restarting session. Tools changed: {tools_changed}, Instructions changed: {instructions_changed}")
            # Recreate the session
            await self.close_session()
            
            await self.initialize_stream()
        
    async def _create_response(self):
        """Create a response from the current context."""
        if not self._api_session_ready:
            self._run_llm_when_api_session_ready = True
            return
        
        if self._context.llm_needs_settings_update:
            await self._update_settings()
            self._context.llm_needs_settings_update = False
            self._context.llm_needs_initial_messages = True

        if self._context.llm_needs_initial_messages:
            messages = self._context.get_messages_for_initializing_history()
            for item in messages:
                logger.debug(f"MESSAGE: {item}")
            self._context.llm_needs_initial_messages = False

        logger.debug(f"Creating response: {self._context.get_messages_for_logging()}")

        await self.push_frame(LLMFullResponseStartFrame())
        await self.start_processing_metrics()
        await self.start_ttfb_metrics()
        
        # Process the context messages
        await self._process_context(self._context)

        # Start processing audio input if system prompt sent
        if not self.audio_input_started:
            if self.audio_capture_task:
                self.audio_capture_task.cancel()
            self.audio_capture_task = asyncio.create_task(self._process_audio_input())

    async def call_function(
        self,
        context: Optional[BedrockLLMContext],
        tool_call_id: str,
        function_name: str,
        arguments: Dict[str, Any],
    ) -> None:
        """
        Call a registered function and send the result back to Bedrock.
        
        Args:
            context: The LLM context (optional)
            tool_call_id: The tool call ID
            function_name: The function name
            arguments: The function arguments
        """
        # Find the appropriate handler
        handler = self._function_handlers.get(function_name) or self._function_handlers.get(None)
        
        if not handler:
            logger.warning(f"No handler found for function {function_name}")
            return
        
        # Create a callback to send the result back to Bedrock
        async def result_callback(result):
            # Create a unique content name for the tool result
            tool_content_name = str(uuid.uuid4())
            
            # Send tool result events
            await self.send_raw_event(self._create_tool_content_start_event(
                content_name=tool_content_name,
                tool_use_id=tool_call_id
            ))
            await self.send_raw_event(self._create_tool_result_event(
                content_name=tool_content_name,
                content=result
            ))
            await self.send_raw_event(self._create_content_end_event(
                content_name=tool_content_name
            ))
        
        # Send function call in progress frame
        await self.push_frame(
            FunctionCallInProgressFrame(
                function_name=function_name,
                tool_call_id=tool_call_id,
                arguments=arguments,
            )
        )
        
        try:
            # Call the handler
            await handler(function_name, tool_call_id, arguments, self, context, result_callback)
        except Exception as e:
            logger.error(f"Error calling function {function_name}: {e}")
            # Send error result
            await result_callback({"error": str(e)})
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """
        Process incoming frames.
        
        Args:
            frame: The frame to process
            direction: The direction of the frame
        """
        await super().process_frame(frame, direction)

        # If we have a pending LLM run, do it now
        if self._run_llm_when_api_session_ready:
            self._run_llm_when_api_session_ready = False
            await self._create_response()

        # logger.debug(f"Received {type(frame).__name__}")
        
        if isinstance(frame, InputAudioRawFrame):
            # Add audio to the queue
            # logger.debug("Adding audio to queue")
            self.audio_input_queue.put_nowait({
                'audio_bytes': frame.audio,
                'prompt_name': self.prompt_name,
                'content_name': self.audio_content_name
            })
        elif isinstance(frame, BedrockLLMContextFrame):
            context: BedrockRealtimeLLMContext = BedrockRealtimeLLMContext.upgrade_to_realtime(
                frame.context
            )
            if not self._context:
                self._context = context
            elif frame.context is not self._context:
                # If the context has changed, reset the conversation
                self._context = context
                await self.reset_conversation()
            # Run the LLM at next opportunity
            await self._create_response()
        elif isinstance(frame, LLMUpdateSettingsFrame):
            # Update settings
            self.session_properties = frame.settings
            await self._update_settings()
        else:
            await self.push_frame(frame, direction)
    
    async def _process_context(self, context: BedrockLLMContext):
        """
        Process a context by sending it to Bedrock.
        
        Args:
            context: The LLM context to process
        """
        try:
            await self.start_ttfb_metrics()
            
            # Send each message in the context
            for message in context.messages:
                content_name = str(uuid.uuid4())
                role = message["role"].upper()
                
                # Start content
                await self.send_raw_event(self._create_text_content_start_event(
                    content_name=content_name,
                    role=role
                ))
                logger.debug(f"{role}")
                
                # Send content
                if isinstance(message["content"], str):
                    await self.send_raw_event(self._create_text_input_event(
                        content_name=content_name,
                        content=message["content"]
                    ))
                elif isinstance(message["content"], list):
                    for item in message["content"]:
                        if "text" in item:
                            await self.send_raw_event(self._create_text_input_event(
                                content_name=content_name,
                                content=item["text"]
                            ))
                
                # End content
                await self.send_raw_event(self._create_content_end_event(
                    content_name=content_name
                ))
            
            # Stop TTFB metrics after we get the first response
            await self.stop_ttfb_metrics()
            
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception(f"{self} exception: {e}")
        finally:
            await self.stop_processing_metrics()
            await self.push_frame(LLMFullResponseEndFrame())

    async def create_session(self):
        """Open a new stream session."""
        try:
            self.stream_response = await self.bedrock_client.invoke_model_with_bidirectional_stream(
                    InvokeModelWithBidirectionalStreamOperationInput(model_id=self.model_id)
                )
            
            # Send opening events
            await self.send_raw_event(self._create_session_start_event())
            await self.send_raw_event(self._create_prompt_start_event())
            
            # If we have system instructions, send them
            if self.session_properties.instructions:
                system_content_name = str(uuid.uuid4())
                await self.send_raw_event(self._create_text_content_start_event(
                    content_name=system_content_name,
                    role="SYSTEM"
                ))
                await self.send_raw_event(self._create_text_input_event(
                    content_name=system_content_name,
                    content=self.session_properties.instructions
                ))
                await self.send_raw_event(self._create_content_end_event(content_name=system_content_name))

            # Start listening for responses
            self.response_task = asyncio.create_task(self._process_responses())

            # Start processing audio input
            if self.audio_capture_task:
                self.audio_capture_task.cancel()
            self.audio_capture_task = asyncio.create_task(self._process_audio_input())
            
            self.is_active = True
            self._api_session_ready = True

            return True
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            self.is_active = False
            self._api_session_ready = False
    
    async def close_session(self):
        """Close the stream properly."""
        if not self.is_active:
            logger.debug("Skipping close_session; stream not active")
            return
        
        self.is_active = False
        self._api_session_ready = False

        try:
            if self.audio_capture_task:
                self.audio_capture_task.cancel()

            if self.response_task and not self.response_task.done():
                self.response_task.cancel()

            # Send closing events
            if self.audio_input_started:
                await self.send_raw_event(self._create_content_end_event(content_name=self.audio_content_name))
                self.audio_input_started = False
            await self.send_raw_event(self._create_prompt_end_event())
            await self.send_raw_event(self._create_session_end_event())
                
            # Close the stream
            if self.stream_response:
                await self.stream_response.input_stream.close()

            logger.debug(f"Closed session successfully")
        except Exception as e:
            logger.error(f"Error closing session: {e}")
        finally:
            self.stream_response = None
        
    # Event creation methods
    def _create_session_start_event(self):
        """Create a session start event."""
        inference_config = {
            "maxTokens": 1024,
            "topP": 0.9,
            "temperature": 0.7
        }
        
        # Override with session properties if provided
        if self.session_properties.max_tokens is not None:
            inference_config["maxTokens"] = self.session_properties.max_tokens
        if self.session_properties.temperature is not None:
            inference_config["temperature"] = self.session_properties.temperature
            
        return json.dumps({
            "event": {
                "sessionStart": {
                    "inferenceConfiguration": inference_config
                }
            }
        })
    
    def _create_prompt_start_event(self):
        """Create a prompt start event with tool configuration."""
        self.prompt_name = str(uuid.uuid4())

        # Determine voice ID
        voice_id = "matthew"  # Default voice
        if self.session_properties.voice:
            voice_id = self.session_properties.voice
            
        prompt_start_event = {
            "event": {
                "promptStart": {
                    "promptName": self.prompt_name,
                    "textOutputConfiguration": {
                        "mediaType": "text/plain"
                    },
                    "audioOutputConfiguration": {
                        "mediaType": "audio/lpcm",
                        "sampleRateHertz": self.output_sample_rate,
                        "sampleSizeBits": self.sample_size_bits,
                        "channelCount": self.channels,
                        "voiceId": voice_id,
                        "encoding": "base64",
                        "audioType": "SPEECH"
                    },
                    "toolUseOutputConfiguration": {
                        "mediaType": "application/json"
                    }
                }
            }
        }
        
        # Add tool configuration if available
        if self.session_properties.tools:
            tool_config = {
                "tools": []
            }
            
            # Process each tool to ensure proper format
            for tool in self.session_properties.tools:
                if "toolSpec" not in tool:
                    # Convert from OpenAI format if needed
                    tool_spec = {
                        "name": tool.get("function", {}).get("name", ""),
                        "description": tool.get("function", {}).get("description", ""),
                        "inputSchema": {
                            "json": json.dumps(tool.get("function", {}).get("parameters", {}))
                        }
                    }
                    tool_config["tools"].append({"toolSpec": tool_spec})
                else:
                    tool_config["tools"].append(tool)
            
            prompt_start_event["event"]["promptStart"]["toolConfiguration"] = tool_config
        
        return json.dumps(prompt_start_event)
    
    def _create_audio_content_start_event(self):
        """Create an audio content start event."""
        # Configure noise reduction if specified
        audio_input_config = {
            "mediaType": "audio/lpcm",
            "sampleRateHertz": self.input_sample_rate,
            "sampleSizeBits": self.sample_size_bits,
            "channelCount": self.channels,
            "audioType": "SPEECH",
            "encoding": "base64"
        }
        
        return json.dumps({
            "event": {
                "contentStart": {
                    "promptName": self.prompt_name,
                    "contentName": self.audio_content_name,
                    "type": "AUDIO",
                    "interactive": True,
                    "role": "USER",
                    "audioInputConfiguration": audio_input_config
                }
            }
        })
    
    def _create_audio_input_event(self, base64_audio):
        """Create an audio input event."""
        return json.dumps({
            "event": {
                "audioInput": {
                    "promptName": self.prompt_name,
                    "contentName": self.audio_content_name,
                    "content": base64_audio
                }
            }
        })
    
    def _create_text_content_start_event(self, content_name, role):
        """Create a text content start event."""
        return json.dumps({
            "event": {
                "contentStart": {
                    "promptName": self.prompt_name,
                    "contentName": content_name,
                    "type": "TEXT",
                    "role": role,
                    "interactive": True,
                    "textInputConfiguration": {
                        "mediaType": "text/plain"
                    }
                }
            }
        })
    
    def _create_text_input_event(self, content_name, content):
        """Create a text input event."""
        return json.dumps({
            "event": {
                "textInput": {
                    "promptName": self.prompt_name,
                    "contentName": content_name,
                    "content": content
                }
            }
        })
    
    def _create_tool_content_start_event(self, content_name, tool_use_id):
        """Create a tool content start event."""
        return json.dumps({
            "event": {
                "contentStart": {
                    "promptName": self.prompt_name,
                    "contentName": content_name,
                    "interactive": False,
                    "type": "TOOL",
                    "role": "TOOL",
                    "toolResultInputConfiguration": {
                        "toolUseId": tool_use_id,
                        "type": "TEXT",
                        "textInputConfiguration": {
                            "mediaType": "text/plain"
                        }
                    }
                }
            }
        })
    
    def _create_tool_result_event(self, content_name, content):
        """Create a tool result event."""
        if isinstance(content, dict):
            content_json_string = json.dumps(content)
        else:
            content_json_string = content
            
        return json.dumps({
            "event": {
                "toolResult": {
                    "promptName": self.prompt_name,
                    "contentName": content_name,
                    "content": content_json_string
                }
            }
        })
    
    def _create_content_end_event(self, content_name):
        """Create a content end event."""
        return json.dumps({
            "event": {
                "contentEnd": {
                    "promptName": self.prompt_name,
                    "contentName": content_name
                }
            }
        })
    
    def _create_prompt_end_event(self):
        """Create a prompt end event."""
        return json.dumps({
            "event": {
                "promptEnd": {
                    "promptName": self.prompt_name
                }
            }
        })
    
    def _create_session_end_event(self):
        """Create a session end event."""
        return json.dumps({
            "event": {
                "sessionEnd": {}
            }
        })

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self.initialize_stream()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self.close_session()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self.close_session()