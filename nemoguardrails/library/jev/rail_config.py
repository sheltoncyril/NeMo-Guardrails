# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Configuration for the Jev (TypeSafe AI) rail."""

from typing import Dict, Optional

from pydantic import ConfigDict, field_validator

from nemoguardrails.manifests.config_schema import Field, RailConfigBaseModel, RailConfigSpec, rail_field

DEFAULT_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MODEL = "jev-latest"

DEFAULT_INPUT_QUESTIONS: Dict[str, str] = {
    "jailbreak": (
        "Is the user trying to jailbreak the assistant, override its instructions, or manipulate it into ignoring its rules?"
    ),
    "harmful_request": (
        "Is the user asking for help with something dangerous, illegal, or intended to cause serious harm to people?"
    ),
}

DEFAULT_OUTPUT_QUESTIONS: Dict[str, str] = {
    "harmful_content": (
        "Does the assistant reply contain instructions or content that could enable serious harm to people?"
    ),
    "policy_violation": ("Does the assistant reply show it has been jailbroken or is ignoring its safety rules?"),
}


class JevQuestion(RailConfigBaseModel):
    """A single Noul (yes/no) question asked of Jev about the text."""

    model_config = ConfigDict(extra="forbid")

    instructions: str = Field(description="The yes/no question to ask Jev about the text.")
    threshold: Optional[float] = Field(
        default=None,
        description="Per-question block threshold in [0, 1]. Falls back to the rail-level threshold when unset.",
    )

    @field_validator("threshold")
    @classmethod
    def _check_threshold(cls, value: Optional[float]) -> Optional[float]:
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        return value


def _default_questions(defaults: Dict[str, str]) -> Dict[str, JevQuestion]:
    return {name: JevQuestion(instructions=text) for name, text in defaults.items()}


class JevDetectionOptions(RailConfigBaseModel):
    """Per-direction (input / output) options for Jev."""

    model_config = ConfigDict(extra="forbid")

    questions: Dict[str, JevQuestion] = Field(
        default_factory=dict,
        description=(
            "Noul questions to ask about the text, keyed by an identifier. "
            "The text is blocked if any answer probability reaches its threshold."
        ),
    )


class JevDetection(RailConfigBaseModel):
    """Configuration for the Jev (TypeSafe AI) rail."""

    model_config = ConfigDict(extra="forbid")

    server_endpoint: str = Field(
        default=DEFAULT_ENDPOINT,
        description="The TypeSafe System One endpoint. The API key is read from TYPESAFE_API_KEY.",
    )
    model: str = Field(default=DEFAULT_MODEL, description="The Jev model to use. Pin a version for reproducibility.")
    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Default probability at or above which a question blocks the text.",
    )
    timeout: float = Field(default=10.0, gt=0, description="Per-request timeout in seconds.")
    fail_open: bool = Field(
        default=False,
        description="If true, allow the text when Jev cannot be reached. Default is to fail closed and block.",
    )
    input: JevDetectionOptions = Field(
        default_factory=lambda: JevDetectionOptions(questions=_default_questions(DEFAULT_INPUT_QUESTIONS)),
        description="Questions asked about the user input.",
    )
    output: JevDetectionOptions = Field(
        default_factory=lambda: JevDetectionOptions(questions=_default_questions(DEFAULT_OUTPUT_QUESTIONS)),
        description="Questions asked about the bot output.",
    )


def build_config_spec() -> RailConfigSpec:
    return RailConfigSpec(
        annotation=Optional[JevDetection],
        field_info=rail_field(
            default_factory=JevDetection,
            description="Configuration for the Jev (TypeSafe AI) rail.",
        ),
        exports={
            "JevDetection": JevDetection,
            "JevDetectionOptions": JevDetectionOptions,
            "JevQuestion": JevQuestion,
        },
    )
