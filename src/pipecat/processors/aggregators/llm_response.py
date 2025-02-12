#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
from abc import abstractmethod
from typing import List

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMMessagesFrame,
    LLMMessagesUpdateFrame,
    LLMSetToolsFrame,
    StartFrame,
    StartInterruptionFrame,
    TextFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAILLMContextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class BaseLLMResponseAggregator(FrameProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    @abstractmethod
    def messages(self) -> List[dict]:
        pass

    @property
    @abstractmethod
    def role(self) -> str:
        pass

    @abstractmethod
    def add_messages(self, messages):
        pass

    @abstractmethod
    def set_messages(self, messages):
        pass

    @abstractmethod
    def set_tools(self, tools):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    async def push_aggregation(self):
        pass


class LLMResponseAggregator(BaseLLMResponseAggregator):
    def __init__(
        self,
        *,
        messages: List[dict],
        role: str = "user",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._messages = messages
        self._role = role

        self._aggregation = ""

        self.reset()

    @property
    def messages(self) -> List[dict]:
        return self._messages

    @property
    def role(self) -> str:
        return self._role

    def add_messages(self, messages):
        self._messages.extend(messages)

    def set_messages(self, messages):
        self.reset()
        self._messages.clear()
        self._messages.extend(messages)

    def set_tools(self, tools):
        pass

    def reset(self):
        self._aggregation = ""

    async def push_aggregation(self):
        if len(self._aggregation) > 0:
            self._messages.append({"role": self._role, "content": self._aggregation})

            # Reset the aggregation. Reset it before pushing it down, otherwise
            # if the tasks gets cancelled we won't be able to clear things up.
            self._aggregation = ""

            frame = LLMMessagesFrame(self._messages)
            await self.push_frame(frame)


class LLMContextResponseAggregator(BaseLLMResponseAggregator):
    def __init__(self, *, context: OpenAILLMContext, role: str, **kwargs):
        super().__init__(**kwargs)
        self._context = context
        self._role = role

        self._aggregation = ""

    @property
    def messages(self) -> List[dict]:
        return self._context.get_messages()

    @property
    def role(self) -> str:
        return self._role

    @property
    def context(self):
        return self._context

    def get_context_frame(self) -> OpenAILLMContextFrame:
        return OpenAILLMContextFrame(context=self._context)

    async def push_context_frame(self):
        frame = self.get_context_frame()
        await self.push_frame(frame)

    def add_messages(self, messages):
        self._context.add_messages(messages)

    def set_messages(self, messages):
        self._context.set_messages(messages)

    def set_tools(self, tools: List):
        self._context.set_tools(tools)

    async def push_aggregation(self):
        if len(self._aggregation) > 0:
            self._context.add_message({"role": self.role, "content": self._aggregation})

            # Reset the aggregation. Reset it before pushing it down, otherwise
            # if the tasks gets cancelled we won't be able to clear things up.
            self._aggregation = ""

            frame = OpenAILLMContextFrame(self._context)
            await self.push_frame(frame)

            # Reset our accumulator state.
            self.reset()


class LLMUserContextAggregator(LLMContextResponseAggregator):
    def __init__(self, context: OpenAILLMContext, aggregation_timeout: float = 1.0, **kwargs):
        super().__init__(context=context, role="user", **kwargs)
        self._aggregation_timeout = aggregation_timeout

        self._seen_interim_results = False
        self._user_speaking = False

        self._aggregation_event = asyncio.Event()
        self._aggregation_task = None

        self.reset()

    def reset(self):
        super().reset()
        self._seen_interim_results = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            await self._start(frame)
            await self.push_frame(frame, direction)
        elif isinstance(frame, EndFrame):
            await self._stop(frame)
            await self.push_frame(frame, direction)
        elif isinstance(frame, CancelFrame):
            await self._cancel(frame)
            await self.push_frame(frame, direction)
        elif isinstance(frame, UserStartedSpeakingFrame):
            await self._handle_user_started_speaking(frame)
            await self.push_frame(frame, direction)
        elif isinstance(frame, UserStoppedSpeakingFrame):
            await self._handle_user_stopped_speaking(frame)
            await self.push_frame(frame, direction)
        elif isinstance(frame, TranscriptionFrame):
            await self._handle_transcription(frame)
        elif isinstance(frame, InterimTranscriptionFrame):
            await self._handle_interim_transcription(frame)
        elif isinstance(frame, LLMMessagesAppendFrame):
            self.add_messages(frame.messages)
        elif isinstance(frame, LLMMessagesUpdateFrame):
            self.set_messages(frame.messages)
        elif isinstance(frame, LLMSetToolsFrame):
            self.set_tools(frame.tools)
        else:
            await self.push_frame(frame, direction)

    async def _start(self, frame: StartFrame):
        self._aggregation_task = self.create_task(self._aggregation_task_handler())

    async def _stop(self, frame: EndFrame):
        if self._aggregation_task:
            await self.cancel_task(self._aggregation_task)
            self._aggregation_task = None

    async def _cancel(self, frame: CancelFrame):
        if self._aggregation_task:
            await self.cancel_task(self._aggregation_task)
            self._aggregation_task = None

    async def _handle_user_started_speaking(self, _: UserStartedSpeakingFrame):
        self._user_speaking = True

    async def _handle_user_stopped_speaking(self, _: UserStoppedSpeakingFrame):
        self._user_speaking = False
        if not self._seen_interim_results:
            await self.push_aggregation()

    async def _handle_transcription(self, frame: TranscriptionFrame):
        self._aggregation += frame.text
        # We just got our final result, so let's reset interim results.
        self._seen_interim_results = False
        # Wakeup our task.
        self._aggregation_event.set()

    async def _handle_interim_transcription(self, _: InterimTranscriptionFrame):
        self._seen_interim_results = True

    async def _aggregation_task_handler(self):
        while True:
            await self._aggregation_event.wait()
            await asyncio.sleep(self._aggregation_timeout)
            if not self._user_speaking:
                await self.push_aggregation()
            self._aggregation_event.clear()


class LLMAssistantContextAggregator(LLMContextResponseAggregator):
    def __init__(self, context: OpenAILLMContext, *, expect_stripped_words: bool = True, **kwargs):
        super().__init__(context=context, role="assistant", **kwargs)
        self._expect_stripped_words = expect_stripped_words

        self.reset()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartInterruptionFrame):
            await self.push_aggregation()
            # Reset anyways
            self.reset()
            await self.push_frame(frame, direction)
        elif isinstance(frame, LLMFullResponseStartFrame):
            await self._handle_llm_start(frame)
        elif isinstance(frame, LLMFullResponseEndFrame):
            await self._handle_llm_end(frame)
        elif isinstance(frame, TextFrame):
            await self._handle_text(frame)
        else:
            await self.push_frame(frame, direction)

    async def _handle_llm_start(self, _: LLMFullResponseStartFrame):
        self._started = True

    async def _handle_llm_end(self, _: LLMFullResponseEndFrame):
        self._started = False
        await self.push_aggregation()

    async def _handle_text(self, frame: TextFrame):
        if not self._started:
            return

        if self._expect_stripped_words:
            self._aggregation += f" {frame.text}" if self._aggregation else frame.text
        else:
            self._aggregation += frame.text


class LLMUserResponseAggregator(LLMUserContextAggregator):
    def __init__(self, messages: List[dict] = [], **kwargs):
        super().__init__(context=OpenAILLMContext(messages), **kwargs)

    async def push_aggregation(self):
        if len(self._aggregation) > 0:
            self._context.add_message({"role": self.role, "content": self._aggregation})

            # Reset the aggregation. Reset it before pushing it down, otherwise
            # if the tasks gets cancelled we won't be able to clear things up.
            self._aggregation = ""

            frame = LLMMessagesFrame(self._context.messages)
            await self.push_frame(frame)

            # Reset our accumulator state.
            self.reset()


class LLMAssistantResponseAggregator(LLMAssistantContextAggregator):
    def __init__(self, messages: List[dict], **kwargs):
        super().__init__(context=OpenAILLMContext(messages), **kwargs)

    async def push_aggregation(self):
        if len(self._aggregation) > 0:
            self._context.add_message({"role": self.role, "content": self._aggregation})

            # Reset the aggregation. Reset it before pushing it down, otherwise
            # if the tasks gets cancelled we won't be able to clear things up.
            self._aggregation = ""

            frame = LLMMessagesFrame(self._context.messages)
            await self.push_frame(frame)

            # Reset our accumulator state.
            self.reset()
