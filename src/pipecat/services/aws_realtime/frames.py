#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from dataclasses import dataclass

from pipecat.frames.frames import DataFrame, FunctionCallResultFrame


@dataclass
class RealtimeMessagesUpdateFrame(DataFrame):
    context: "BedrockRealtimeLLMContext"


@dataclass
class RealtimeFunctionCallResultFrame(DataFrame):
    result_frame: FunctionCallResultFrame