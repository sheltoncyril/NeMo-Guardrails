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

from nemoguardrails.manifests import (
    ActionRef,
    Binding,
    ConfigSpecRef,
    EnvVar,
    RailActions,
    RailConfigSchema,
    RailDirection,
    RailFlows,
    RailManifest,
    RailMetadata,
    RailPrivacy,
    RailRequirements,
    RailSpec,
    RailSurface,
    ServiceRequirement,
)

JEV_CHECK = ActionRef(
    name="jev_check",
    target="nemoguardrails.library.jev.actions:jev_check",
)

RAIL = RailManifest(
    name="jev",
    metadata=RailMetadata(
        display_name="Jev (TypeSafe AI)",
        description="Screens input and output text with user-defined yes/no questions answered by TypeSafe AI's Jev model.",
        categories=("input", "output"),
        capabilities=("allow", "block", "classify", "content_safety", "detect_jailbreak"),
        tags=("third-party", "api", "typesafe", "jev"),
        docs_url="docs/configure-rails/guardrail-catalog/community/jev.mdx",
    ),
    spec=RailSpec(
        config_schema=RailConfigSchema(
            key="jev",
            spec=ConfigSpecRef(target="nemoguardrails.library.jev.rail_config:build_config_spec"),
        ),
        flows=RailFlows(flow_names=("jev check input", "jev check output")),
        actions=RailActions(refs=(JEV_CHECK,)),
        surfaces=(
            RailSurface(
                name="jev check input",
                direction=RailDirection.INPUT,
                action=JEV_CHECK,
                bindings=(Binding.literal("source", "input"), Binding.context("text", "user_message")),
            ),
            RailSurface(
                name="jev check output",
                direction=RailDirection.OUTPUT,
                action=JEV_CHECK,
                bindings=(Binding.literal("source", "output"), Binding.context("text", "bot_message")),
            ),
        ),
        requirements=RailRequirements(
            env_vars=(EnvVar(name="TYPESAFE_API_KEY", required=True),),
            services=(ServiceRequirement(name="TypeSafe AI API", required=True),),
        ),
        privacy=RailPrivacy(sends_user_text=True, sends_bot_text=True, remote_services=("TypeSafe AI API",)),
    ),
)
