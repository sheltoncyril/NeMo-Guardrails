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

"""Text assessment using Jev, the System One model from TypeSafe AI."""

import logging
import os
from typing import Dict, Optional

from nemoguardrails import RailsConfig
from nemoguardrails.actions.actions import action
from nemoguardrails.actions.rail_outcome import RailOutcome
from nemoguardrails.http import HTTPClient
from nemoguardrails.library.jev.rail_config import JevDetection
from nemoguardrails.library.jev.request import jev_noul_request

log = logging.getLogger(__name__)

VALID_SOURCES = ("input", "output")


def _get_api_key() -> Optional[str]:
    return os.environ.get("TYPESAFE_API_KEY") or None


def _failure_outcome(fail_open: bool, reason: str) -> RailOutcome:
    metadata = {"error": reason, "fail_open": fail_open}
    if fail_open:
        return RailOutcome.allow(metadata=metadata)
    return RailOutcome.block(reason=reason, metadata=metadata)


@action(is_system_action=True)
async def jev_check(
    source: str,
    text: str,
    config: RailsConfig,
    http_client: Optional[HTTPClient] = None,
    **kwargs,
) -> RailOutcome:
    """Ask Jev the configured yes/no questions about ``text`` and block on a high probability.

    Args:
        source: Either "input" or "output"; selects which configured questions to ask.
        text: The text to assess.
        config: The rails configuration object.

    Returns:
        RailOutcome.block() if any question's probability reaches its threshold, or if Jev
        cannot be reached and ``fail_open`` is false; RailOutcome.allow() otherwise. Metadata
        holds the per-question ``scores`` and the ``triggered`` question ids, never the text.

    Raises:
        ValueError: If ``source`` is invalid or TYPESAFE_API_KEY is not set, since both are
            configuration errors rather than provider failures.
    """
    if source not in VALID_SOURCES:
        raise ValueError(f"Jev can only be defined in the following flows: {list(VALID_SOURCES)}. Got '{source}'.")

    jev_config: JevDetection = getattr(config.rails.config, "jev")
    questions = getattr(jev_config, source).questions
    # Nothing to ask, or nothing to assess (e.g. an empty or tool-call-only assistant turn): skip the
    # API call and the API-key requirement rather than sending an empty state.
    if not questions or not (text or "").strip():
        return RailOutcome.allow(metadata={"scores": {}, "triggered": []})

    api_key = _get_api_key()
    if api_key is None:
        raise ValueError("TYPESAFE_API_KEY environment variable not set.")

    try:
        scores: Dict[str, float] = await jev_noul_request(
            text,
            {qid: q.instructions for qid, q in questions.items()},
            jev_config.server_endpoint,
            jev_config.model,
            api_key,
            http_client=http_client,
            timeout=jev_config.timeout,
        )
    except ValueError as err:
        log.warning("Jev assessment failed (%s); %s.", err, "allowing text" if jev_config.fail_open else "blocking text")
        return _failure_outcome(jev_config.fail_open, str(err))

    triggered = [
        qid
        for qid, score in scores.items()
        if score >= (questions[qid].threshold if questions[qid].threshold is not None else jev_config.threshold)
    ]
    metadata = {"scores": scores, "triggered": triggered}
    if triggered:
        return RailOutcome.block(reason=f"Jev flagged: {', '.join(triggered)}", metadata=metadata)
    return RailOutcome.allow(metadata=metadata)
