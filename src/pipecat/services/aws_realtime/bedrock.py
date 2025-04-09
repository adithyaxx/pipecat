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

from pipecat.frames.frames import (
    AudioRawFrame,
    Frame,
    FunctionCallCancelFrame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesFrame,
    LLMTextFrame,
    LLMUpdateSettingsFrame,
    OutputAudioRawFrame
)
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.processors.aggregators.bedrock_llm_context import (
    BedrockLLMContext,
    BedrockLLMContextFrame,
)
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantContextAggregator,
    LLMUserContextAggregator,
)
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.ai_services import LLMService
from pipecat.services.aws.llm import BedrockLLMContext, BedrockContextAggregatorPair

from .events import SessionProperties
from .context import (
    BedrockRealtimeAssistantContextAggregator,
    BedrockRealtimeLLMContext,
    BedrockRealtimeUserContextAggregator,
)


class BedrockRealtimeLLMService(LLMService):
    """
    Implementation of a real-time LLM service using Amazon Bedrock's Nova Sonic model.
    
    This service supports bidirectional streaming for audio input/output and text,
    as well as tool use functionality.
    """
    
    def __init__(
        self,
        *,
        aws_access_key: Optional[str] = None,
        aws_secret_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        aws_region: str = "us-east-1",
        model: str = "amazon.nova-sonic-v1:0",
        session_properties: SessionProperties = SessionProperties(),
        start_audio_paused: bool = False,
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
            start_audio_paused: Whether to start with audio paused
        """
        super().__init__(**kwargs)
        
        self.model_id = model
        self.aws_region = aws_region
        self.session_properties = session_properties
        self.audio_paused = start_audio_paused
        
        # AWS credentials
        self.aws_access_key = aws_access_key
        self.aws_secret_key = aws_secret_key
        self.aws_session_token = aws_session_token
        
        # Initialize client
        self._initialize_client()
        
        # Stream state
        self.stream_response = None
        self.is_active = False
        self.prompt_name = str(uuid.uuid4())
        self.content_name = str(uuid.uuid4())
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
        
        # Tool use tracking
        self.toolUseId = ""
        self.toolName = ""
        
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
        """Initialize the bidirectional stream with Bedrock."""
        try:
            self.stream_response = await self.bedrock_client.invoke_model_with_bidirectional_stream(
                InvokeModelWithBidirectionalStreamOperationInput(model_id=self.model_id)
            )
            self.is_active = True
            
            # Send initialization events
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
            
            # Start processing audio input if not paused
            if not self.audio_paused:
                asyncio.create_task(self._process_audio_input())
            
            logger.info("Bedrock Realtime stream initialized successfully")
            return True
        except Exception as e:
            self.is_active = False
            logger.error(f"Failed to initialize Bedrock Realtime stream: {str(e)}")
            return False
    
    async def send_raw_event(self, event_json):
        """Send a raw event JSON to the Bedrock stream."""
        if not self.stream_response or not self.is_active:
            logger.debug("Stream not initialized or closed")
            return
        
        event = InvokeModelWithBidirectionalStreamInputChunk(
            value=BidirectionalInputPayloadPart(bytes_=event_json.encode('utf-8'))
        )
        
        try:
            await self.stream_response.input_stream.send(event)
            logger.debug(f"Sent event: {event_json[:200]}...")
        except Exception as e:
            logger.error(f"Error sending event: {str(e)}")
    
    async def _process_audio_input(self):
        """Process audio input from the queue and send to Bedrock."""
        # First send audio content start event
        await self.send_raw_event(self._create_audio_content_start_event())
        
        while self.is_active and not self.audio_paused:
            try:
                # Get audio data from the queue
                data = await self.audio_input_queue.get()
                
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
    
    async def _process_responses(self):
        """Process incoming responses from Bedrock."""
        try:
            while self.is_active:
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
                                    
                                    # Send audio frame
                                    await self.push_frame(OutputAudioRawFrame(
                                        audio=audio_bytes,
                                        sample_rate=self.output_sample_rate,
                                        num_channels=self.channels
                                    ))
                                
                                # Handle tool use
                                elif 'toolUse' in event:
                                    self.toolUseId = event['toolUse']['toolUseId']
                                    self.toolName = event['toolUse']['toolName']
                                    
                                    # Parse tool input
                                    try:
                                        tool_input = json.loads(event['toolUse']['content'])
                                    except (json.JSONDecodeError, KeyError):
                                        tool_input = {}
                                    
                                    logger.debug(f"Tool use detected: {self.toolName}, ID: {self.toolUseId}")
                                    
                                    # Call the function
                                    await self.call_function(
                                        context=None,  # We don't have context here
                                        tool_call_id=self.toolUseId,
                                        function_name=self.toolName,
                                        arguments=tool_input
                                    )
                                
                                # Handle content end with tool type
                                elif 'contentEnd' in event and event.get('contentEnd', {}).get('type') == 'TOOL':
                                    logger.debug("Tool content ended")
                                
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
                
                except StopAsyncIteration:
                    # Stream has ended
                    break
                except Exception as e:
                    logger.error(f"Error receiving response: {e}")
                    break
                    
        except Exception as e:
            logger.error(f"Response processing error: {e}")
        finally:
            self.is_active = False
    
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
        
        # Initialize stream if not already active
        if not self.is_active:
            await self.initialize_stream()
        
        if isinstance(frame, AudioRawFrame) and not self.audio_paused:
            # Add audio to the queue
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
        elif isinstance(frame, LLMMessagesFrame):
            # Convert messages to context and process
            context = BedrockLLMContext.from_messages(frame.messages)
            await self._process_context(context)
        elif isinstance(frame, LLMUpdateSettingsFrame):
            # Update settings
            await self._update_settings(frame.settings)
        else:
            # Pass through other frames
            await self.push_frame(frame, direction)
    
    async def _process_context(self, context: BedrockLLMContext):
        """
        Process a context by sending it to Bedrock.
        
        Args:
            context: The LLM context to process
        """
        try:
            await self.push_frame(LLMFullResponseStartFrame())
            await self.start_processing_metrics()
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
            
            # End the prompt to get a response
            await self.send_raw_event(self._create_prompt_end_event())
            
            # Stop TTFB metrics after we get the first response
            await self.stop_ttfb_metrics()
            
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception(f"{self} exception: {e}")
        finally:
            await self.stop_processing_metrics()
            await self.push_frame(LLMFullResponseEndFrame())
    
    async def _update_settings(self, settings: Dict[str, Any]) -> None:
        """
        Update service settings.
        
        Args:
            settings: The settings to update
        """
        # Handle audio pause/unpause
        if "audio_paused" in settings:
            old_paused = self.audio_paused
            self.audio_paused = settings["audio_paused"]
            
            if old_paused and not self.audio_paused:
                # Audio was paused and now unpaused, start processing
                asyncio.create_task(self._process_audio_input())
            elif not old_paused and self.audio_paused:
                # Audio was unpaused and now paused, send content end
                await self.send_raw_event(self._create_content_end_event(content_name=self.audio_content_name))
                # Generate a new audio content name for the next turn
                self.audio_content_name = str(uuid.uuid4())
    
    async def close(self):
        """Close the stream properly."""
        if not self.is_active:
            return
        
        self.is_active = False
        if self.response_task and not self.response_task.done():
            self.response_task.cancel()
        
        # Send closing events
        await self.send_raw_event(self._create_content_end_event(content_name=self.audio_content_name))
        await self.send_raw_event(self._create_prompt_end_event())
        await self.send_raw_event(self._create_session_end_event())
        
        if self.stream_response:
            await self.stream_response.input_stream.close()
    
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
        # Convert tools to Nova Sonic format if provided
        tools_config = None
        if self.session_properties.tools:
            tools = []
            for tool in self.session_properties.tools:
                tool_spec = {
                    "toolSpec": {
                        "name": tool["function"]["name"],
                        "description": tool["function"].get("description", ""),
                        "inputSchema": {
                            "json": json.dumps(tool["function"].get("parameters", {}))
                        }
                    }
                }
                tools.append(tool_spec)
            
            tools_config = {"tools": tools}
        
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
        if tools_config:
            prompt_start_event["event"]["promptStart"]["toolConfiguration"] = tools_config
        
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
        
        # Add noise reduction if enabled
        # if (self.session_properties.input_audio_noise_reduction and 
        #     self.session_properties.input_audio_noise_reduction.enabled):
        #     audio_input_config["noiseReduction"] = {
        #         "type": self.session_properties.input_audio_noise_reduction.type
        #     }
            
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