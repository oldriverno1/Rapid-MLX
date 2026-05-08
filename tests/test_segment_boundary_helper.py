# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the prompt-segment boundary split helper used by issue #214.

The helper computes whether a request's tokens-to-process should be split
into two segments at the chat-template-stable prefix_boundary so that
BatchGenerator.insert_segments emits an end_of_segment signal at that
point. Returns None when no split is needed (existing single-segment
insert path is used).
"""

from vllm_mlx.scheduler import _maybe_split_at_boundary


class TestMaybeSplitAtBoundary:
    def test_full_miss_splits_at_boundary(self):
        # tokens_to_process == prompt_token_ids (full miss); boundary inside it
        tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        segments = _maybe_split_at_boundary(
            tokens, prefix_boundary=4, cached_tokens=0
        )
        assert segments == [[1, 2, 3, 4], [5, 6, 7, 8, 9, 10]]

    def test_partial_cache_hit_adjusts_boundary(self):
        # cached_tokens=3 means tokens_to_process is the suffix [4..10]
        tokens = [4, 5, 6, 7, 8, 9, 10]
        segments = _maybe_split_at_boundary(
            tokens, prefix_boundary=5, cached_tokens=3
        )
        # effective boundary = 5 - 3 = 2 inside the suffix
        assert segments == [[4, 5], [6, 7, 8, 9, 10]]

    def test_no_split_when_boundary_zero(self):
        assert _maybe_split_at_boundary(
            [1, 2, 3], prefix_boundary=0, cached_tokens=0
        ) is None

    def test_no_split_when_boundary_already_cached(self):
        # cached_tokens >= prefix_boundary → no remaining boundary
        assert _maybe_split_at_boundary(
            [4, 5, 6], prefix_boundary=3, cached_tokens=3
        ) is None

    def test_no_split_when_boundary_at_or_past_end(self):
        assert _maybe_split_at_boundary(
            [1, 2, 3], prefix_boundary=3, cached_tokens=0
        ) is None
        assert _maybe_split_at_boundary(
            [1, 2, 3], prefix_boundary=10, cached_tokens=0
        ) is None

    def test_no_split_when_tokens_empty(self):
        assert _maybe_split_at_boundary(
            [], prefix_boundary=5, cached_tokens=0
        ) is None
