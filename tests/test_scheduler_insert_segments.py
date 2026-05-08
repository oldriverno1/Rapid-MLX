# SPDX-License-Identifier: Apache-2.0
"""Scheduler tests for issue #214: switch to BatchGenerator.insert_segments
when a request has a non-trivial prefix_boundary so the BatchGenerator
emits an end_of_segment=True signal at the chat-template boundary,
enabling the prefix-cache snapshot in _snapshot_promoted_prompts.
"""

from unittest.mock import MagicMock

from vllm_mlx.request import Request, RequestStatus, SamplingParams
from vllm_mlx.scheduler import Scheduler, SchedulerConfig


def _make_scheduler():
    config = SchedulerConfig(
        enable_prefix_cache=True, use_memory_aware_cache=True
    )
    scheduler = Scheduler(MagicMock(), MagicMock(), config)
    sampler_params = (0.7, 0.9, 0.0)
    scheduler._current_sampler_params = sampler_params
    scheduler.batch_generator = MagicMock()
    scheduler.batch_generator.insert.return_value = [42]
    scheduler.batch_generator.insert_segments.return_value = [42]
    return scheduler


def _enqueue(scheduler, prompt_token_ids, prefix_boundary=0, cached_tokens=0):
    req = Request(
        request_id="req-1",
        prompt="ignored",
        prompt_token_ids=prompt_token_ids,
        sampling_params=SamplingParams(max_tokens=4),
    )
    req.prefix_boundary = prefix_boundary
    req.cached_tokens = cached_tokens
    if cached_tokens == 0:
        req.remaining_tokens = None
    else:
        req.remaining_tokens = prompt_token_ids[cached_tokens:]
    req.status = RequestStatus.WAITING
    scheduler.requests[req.request_id] = req
    scheduler.waiting.append(req)
    return req


class TestScheduleWaitingSegments:
    def test_uses_insert_segments_when_boundary_inside_prompt(self):
        scheduler = _make_scheduler()
        _enqueue(
            scheduler,
            prompt_token_ids=list(range(10)),
            prefix_boundary=4,
        )

        scheduler._schedule_waiting()

        scheduler.batch_generator.insert_segments.assert_called_once()
        scheduler.batch_generator.insert.assert_not_called()
        segments_arg = scheduler.batch_generator.insert_segments.call_args.args[0]
        assert segments_arg == [[[0, 1, 2, 3], [4, 5, 6, 7, 8, 9]]]

    def test_uses_plain_insert_when_no_boundary(self):
        scheduler = _make_scheduler()
        _enqueue(
            scheduler, prompt_token_ids=[1, 2, 3], prefix_boundary=0
        )

        scheduler._schedule_waiting()

        scheduler.batch_generator.insert.assert_called_once()
        scheduler.batch_generator.insert_segments.assert_not_called()

    def test_uses_plain_insert_when_boundary_at_prompt_end(self):
        scheduler = _make_scheduler()
        _enqueue(
            scheduler,
            prompt_token_ids=[1, 2, 3, 4],
            prefix_boundary=4,
        )

        scheduler._schedule_waiting()

        scheduler.batch_generator.insert.assert_called_once()
        scheduler.batch_generator.insert_segments.assert_not_called()

    def test_partial_cache_hit_splits_remaining_at_adjusted_boundary(self):
        scheduler = _make_scheduler()
        # Full prompt is [0..9], turn 1 cached [0,1,2] (cached_tokens=3),
        # remaining_tokens=[3..9], prefix_boundary=5 -> split inside
        # remaining at index 2 -> segments = [[3,4],[5,6,7,8,9]]
        _enqueue(
            scheduler,
            prompt_token_ids=list(range(10)),
            prefix_boundary=5,
            cached_tokens=3,
        )

        scheduler._schedule_waiting()

        scheduler.batch_generator.insert_segments.assert_called_once()
        segments_arg = scheduler.batch_generator.insert_segments.call_args.args[0]
        assert segments_arg == [[[3, 4], [5, 6, 7, 8, 9]]]
