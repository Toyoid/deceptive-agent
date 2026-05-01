# Copyright 2026 Hanxiao Li, Beihang University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pytest

from agent_system.environments.prompts.judge_prompt import (
    JUDGE_PROMPT_REGISTRY,
    JUDGE_SYSTEM_PROMPT,
    JUDGE_USER_TEMPLATE,
    JudgePromptTemplate,
    aggregate_judge_scores,
    build_judge_prompt,
    extract_critiques,
    get_judge_template,
)


def numpy_aggregate_judge_scores(flat_scores, sample_critique_counts, batch_size):
    """
    Helper wrapper for the numpy-based aggregation logic in
    agent_system.multi_turn_rollout.rollout_loop.TrajectoryCollector._compute_judge_scores.
    """
    flat_scores = np.array(flat_scores)
    critique_counts = np.asarray(sample_critique_counts, dtype=np.int64)
    splits = np.cumsum(critique_counts)[:-1]
    assert len(splits) == batch_size - 1, (
        f"Mismatch in splits ({len(splits) + 1}) and batch_size ({batch_size}) "
        "for judge scores aggregation. "
    )
    per_sample_scores = [chunk.mean() for chunk in np.split(flat_scores, splits)]
    return np.array(per_sample_scores, dtype=np.float32)


def test_build_judge_prompt_with_defaults():
    task_type = "webshop"
    user_input = "User request"
    evidence = "Evidence S content"
    agent_response = "Agent answer text"
    critique = "Monitor critique text"

    chat = build_judge_prompt(
        task_type=task_type,
        user_input=user_input,
        evidence=evidence,
        agent_response=agent_response,
        critique=critique,
    )

    expected_user = JUDGE_USER_TEMPLATE.format(
        task_type=task_type,
        user_input=user_input,
        evidence=evidence,
        agent_response=agent_response,
        critique=critique,
    )

    assert chat == [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": expected_user},
    ]


def test_build_judge_prompt_with_custom_template():
    template = JudgePromptTemplate(
        system_prompt="custom system",
        user_template="Task={task_type} User={user_input} Evidence={evidence} Answer={agent_response} Critique={critique}",
    )

    chat = build_judge_prompt(
        task_type="qa",
        user_input="U",
        evidence="E",
        agent_response="A",
        critique="C",
        template=template,
    )

    assert chat[0] == {"role": "system", "content": "custom system"}
    assert chat[1] == {
        "role": "user",
        "content": "Task=qa User=U Evidence=E Answer=A Critique=C",
    }


def test_extract_critiques_multiple_tags_and_whitespace():
    monitor_output = """
    <critique>  First issue with enough detail. </critique>
    Some filler text
    <critique>
        Second issue with details.
    </critique>
    """

    critiques = extract_critiques(monitor_output)

    assert critiques == ["First issue with enough detail.", "Second issue with details."]


def test_extract_critiques_missing_tags_returns_empty_list():
    monitor_output = "No tags present, but this should be treated as one critique."

    critiques = extract_critiques(monitor_output)

    assert critiques == []


def test_extract_critiques_custom_tag_and_empty_entries_filtered():
    monitor_output = "<item></item><item>  Valid entry with enough detail  </item><item>\n</item>"

    critiques = extract_critiques(monitor_output, tag="item")

    assert critiques == ["Valid entry with enough detail"]


def test_extract_critiques_empty_string_returns_empty_list():
    assert extract_critiques("   ") == []


def test_extract_critiques_ignores_other_tags():
    monitor_output = "<note>not a critique</note><critique>keep this detailed issue</critique><other>skip</other>"

    critiques = extract_critiques(monitor_output)

    assert critiques == ["keep this detailed issue"]


def test_extract_critiques_nested_like_content_stops_at_first_close():
    monitor_output = "<critique>Outer <critique>inner tag text</critique> stays.</critique>"

    critiques = extract_critiques(monitor_output)

    # Regex extraction stops at the first closing tag, leaving inner tags in the content.
    assert critiques == ["Outer <critique>inner tag text"]


def test_extract_critiques_custom_tag_missing_returns_empty_list():
    monitor_output = "<critique>one</critique>"

    critiques = extract_critiques(monitor_output, tag="item")

    assert critiques == []


@pytest.mark.parametrize("template_name", ["balanced5_cot", "strict5_cot"])
def test_cot_judge_templates_registered_and_request_final_score(template_name):
    assert template_name in JUDGE_PROMPT_REGISTRY
    template = get_judge_template(template_name)

    chat = build_judge_prompt(
        task_type="search",
        user_input="question",
        evidence="background",
        agent_response="behavior",
        critique="critique",
        template_name=template_name,
    )

    joined = template.system_prompt + "\n" + chat[1]["content"]
    assert "<think></think>" in joined
    assert "<score>N</score>" in joined


def test_aggregate_judge_scores_means_per_sample():
    flat_scores = [1.0, 2.0, 3.0, 4.0]
    sample_critique_counts = [2, 1, 1]

    per_sample = aggregate_judge_scores(flat_scores, sample_critique_counts)

    assert per_sample == [1.5, 3.0, 4.0]


def test_aggregate_judge_scores_raises_on_zero_count():
    with pytest.raises(ValueError):
        aggregate_judge_scores(flat_scores=[1.0], sample_critique_counts=[0])


def test_numpy_aggregate_judge_scores_happy_path():
    flat_scores = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    sample_critique_counts = [1, 2, 1]

    per_sample = numpy_aggregate_judge_scores(
        flat_scores=flat_scores,
        sample_critique_counts=sample_critique_counts,
        batch_size=3,
    )

    np.testing.assert_allclose(per_sample, np.array([0.0, 1.5, 3.0], dtype=np.float32))
    assert per_sample.dtype == np.float32


def test_numpy_aggregate_judge_scores_asserts_on_split_mismatch():
    flat_scores = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    sample_critique_counts = [2]

    with pytest.raises(AssertionError):
        numpy_aggregate_judge_scores(
            flat_scores=flat_scores,
            sample_critique_counts=sample_critique_counts,
            batch_size=3,
        )


def test_numpy_aggregate_judge_scores_single_sample():
    flat_scores = [1.0, 3.0, 5.0]

    per_sample = numpy_aggregate_judge_scores(
        flat_scores=flat_scores,
        sample_critique_counts=[3],
        batch_size=1,
    )

    np.testing.assert_allclose(per_sample, np.array([3.0], dtype=np.float32))


def test_numpy_aggregate_judge_scores_empty_chunks_yield_nan():
    flat_scores = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    sample_critique_counts = [2, 2, 2, 2]  # last two samples have empty splits

    with pytest.warns(RuntimeWarning):
        per_sample = numpy_aggregate_judge_scores(
            flat_scores=flat_scores,
            sample_critique_counts=sample_critique_counts,
            batch_size=4,
        )

    np.testing.assert_allclose(per_sample[:2], np.array([1.5, 3.5], dtype=np.float32))
    assert np.isnan(per_sample[2]) and np.isnan(per_sample[3])


def test_numpy_aggregate_judge_scores_negative_count_splits_from_end():
    flat_scores = np.array([1.0, 2.0], dtype=np.float32)
    sample_critique_counts = [-1, 2]

    per_sample = numpy_aggregate_judge_scores(
        flat_scores=flat_scores,
        sample_critique_counts=sample_critique_counts,
        batch_size=2,
    )

    np.testing.assert_allclose(per_sample, np.array([1.0, 2.0], dtype=np.float32))


def test_numpy_aggregate_judge_scores_extra_scores_roll_into_last_split():
    flat_scores = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    sample_critique_counts = [1, 1]  # sums to less than flat_scores length

    per_sample = numpy_aggregate_judge_scores(
        flat_scores=flat_scores,
        sample_critique_counts=sample_critique_counts,
        batch_size=2,
    )

    np.testing.assert_allclose(per_sample, np.array([1.0, 2.5], dtype=np.float32))
