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

"""Client for the TypeSafe AI System One API (Jev)."""

from typing import Dict, Optional

from nemoguardrails.http import (
    HTTPClient,
    HTTPClientError,
    HTTPResponseDecodeError,
    HTTPTimeoutError,
    http_call,
)


async def jev_noul_request(
    text: str,
    questions: Dict[str, str],
    server_endpoint: str,
    model: str,
    api_key: str,
    http_client: Optional[HTTPClient] = None,
    timeout: float = 10.0,
) -> Dict[str, float]:
    """Ask Jev a batch of yes/no (Noul) questions about ``text``.

    All questions go in a single request and are answered in parallel by Jev.

    Args:
        text: The text (the Jev "state") to assess.
        questions: Mapping of question id to the yes/no question instructions.
        server_endpoint: The System One endpoint URL.
        model: The Jev model identifier.
        api_key: The TypeSafe API key, sent as a bearer token.
        http_client: Optional shared HTTP client.
        timeout: Per-request timeout in seconds.

    Returns:
        Mapping of question id to the probability (0-1) that the answer is yes.

    Raises:
        ValueError: If the call fails, times out, or the response is malformed.
            Messages never include the request text.
    """
    payload = {
        "state": text,
        "model": model,
        "questions": {qid: {"type": "noul", "instructions": instructions} for qid, instructions in questions.items()},
    }
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    try:
        response = await http_call(
            http_client,
            "POST",
            server_endpoint,
            json=payload,
            headers=headers,
            timeout=timeout,
            raise_for_status=False,
        )
    except HTTPTimeoutError as err:
        raise ValueError(f"Jev call timed out after {timeout} seconds.") from err
    except HTTPClientError as err:
        raise ValueError(f"Jev call failed: {type(err).__name__}") from err

    if response.status_code != 200:
        # Deliberately omit the response body: it can echo request content.
        raise ValueError(f"Jev call failed with status code {response.status_code}.")

    try:
        data = response.json()
    except HTTPResponseDecodeError as err:
        raise ValueError("Failed to parse Jev response as JSON.") from err

    answers = data.get("answers") if isinstance(data, dict) else None
    if not isinstance(answers, dict):
        raise ValueError("Invalid response from Jev: missing 'answers' object.")

    scores: Dict[str, float] = {}
    for qid in questions:
        answer = answers.get(qid)
        value = answer.get("noul") if isinstance(answer, dict) else None
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
            raise ValueError(f"Invalid response from Jev: bad or missing noul answer for question '{qid}'.")
        scores[qid] = float(value)
    return scores
