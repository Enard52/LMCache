# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the APC-covered-lookup skip (shared foundation).

Covers the correctness-critical pieces that do not require the native compute
backend:

- The covered count carried in ``request_configs`` under
  ``COVERED_CHUNKS_CONFIG_KEY`` survives ``no_worker_id_version`` and does not
  change cache identity.
- ``resolve_prefetched_obj_keys`` clamping the release range to the covered
  prefix, so a release never touches a chunk the lookup did not lock (risk 1:
  over-release would drop a concurrent prefix-sharing request's read lock).
"""

# Standard
from typing import Any, cast
from unittest.mock import MagicMock
import importlib.util
import threading

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.multiprocess.custom_types import (
    COVERED_CHUNKS_CONFIG_KEY,
    IPCCacheServerKey,
)
from lmcache.v1.multiprocess.modules.lookup import resolve_prefetched_obj_keys

CHUNK_SIZE = 16

# The connector and its tracker import vLLM at module top, so the tests that
# drive them cannot run in environments built without it (the k3 unit suite
# installs LMCache alone). The server-side tests below have no such dependency
# and must keep running there, so guard per-test rather than per-module.
requires_vllm = pytest.mark.skipif(
    importlib.util.find_spec("vllm") is None,
    reason="drives the vLLM connector, which imports vLLM at module top",
)


class _FakeRequest:
    """Duck-typed vLLM Request carrying only what the tracker reads."""

    def __init__(self, prompt_token_ids: list[int]):
        # Third Party
        from vllm.v1.request import RequestStatus

        self.request_id = "req-covered"
        self.status = RequestStatus.WAITING
        self.cache_salt = ""
        self.prompt_token_ids = list(prompt_token_ids)
        self.num_prompt_tokens = len(prompt_token_ids)
        self.all_token_ids = list(prompt_token_ids)
        self.num_computed_tokens = 0
        self.mm_features: list[Any] = []
        self.sampling_params = None


def _make_key(n_chunks: int, covered_chunks: int = 0) -> IPCCacheServerKey:
    n_tokens = n_chunks * CHUNK_SIZE
    request_configs = (
        {COVERED_CHUNKS_CONFIG_KEY: covered_chunks} if covered_chunks else None
    )
    return IPCCacheServerKey(
        model_name="m",
        world_size=1,
        worker_id=None,
        token_ids=tuple(range(n_tokens)),
        start=0,
        end=n_tokens,
        request_id="r",
        num_kv_readers=1,
        request_configs=request_configs,
    )


def _ctx_with_hashes(n_chunks: int) -> MagicMock:
    ctx = MagicMock()
    ctx.chunk_size = CHUNK_SIZE
    ctx.token_hasher.compute_chunk_hashes.return_value = [
        bytes([i]) * 4 for i in range(n_chunks)
    ]
    return ctx


def test_covered_config_absent_by_default():
    key = _make_key(4)
    assert (key.request_configs or {}).get(COVERED_CHUNKS_CONFIG_KEY, 0) == 0


def test_covered_config_survives_no_worker_id_version():
    key = _make_key(4, covered_chunks=2).no_worker_id_version()
    assert key.request_configs[COVERED_CHUNKS_CONFIG_KEY] == 2
    assert key.worker_id is None


def test_covered_config_not_part_of_cache_identity():
    # request_configs is compare=False, so two otherwise-equal keys are equal.
    a = _make_key(4, covered_chunks=0)
    b = _make_key(4, covered_chunks=3)
    assert a == b


def test_resolve_full_attention_clamps_to_covered():
    # Full attention (window -1): locked range is [covered, hit).
    ctx = _ctx_with_hashes(4)
    key = _make_key(4, covered_chunks=2)
    obj_keys = resolve_prefetched_obj_keys(
        ctx, key, hit_chunks=4, locked_gids=(), group_windows=(-1,), covered_chunks=2
    )
    # world_size 1, one group => (hit - covered) = 2 keys, for chunks 2 and 3.
    assert len(obj_keys) == 2


def test_resolve_covered_zero_releases_full_prefix():
    ctx = _ctx_with_hashes(4)
    key = _make_key(4, covered_chunks=0)
    obj_keys = resolve_prefetched_obj_keys(
        ctx, key, hit_chunks=4, locked_gids=(), group_windows=(-1,), covered_chunks=0
    )
    assert len(obj_keys) == 4


def test_resolve_covered_at_hit_releases_nothing():
    ctx = _ctx_with_hashes(4)
    key = _make_key(4, covered_chunks=4)
    obj_keys = resolve_prefetched_obj_keys(
        ctx, key, hit_chunks=4, locked_gids=(), group_windows=(-1,), covered_chunks=4
    )
    assert obj_keys == []


def test_resolve_sliding_window_clamped_by_covered():
    # Sliding window w=2, hit=4 => window [2,4); covered=3 clamps it to [3,4).
    ctx = _ctx_with_hashes(4)
    key = _make_key(4, covered_chunks=3)
    obj_keys = resolve_prefetched_obj_keys(
        ctx, key, hit_chunks=4, locked_gids=(), group_windows=(2,), covered_chunks=3
    )
    assert len(obj_keys) == 1


def test_l2_adapter_touch_keys_refreshes_recency_without_io():
    """``L2AdapterInterface.touch_keys`` marks keys accessed and moves no bytes.

    The covered prefix is skipped instead of loaded, so without this refresh L2
    would see those keys as cold and evict data a non-skipping lookup kept.
    """
    # First Party
    from lmcache.v1.distributed.eviction import L2EvictionPolicy
    from lmcache.v1.distributed.eviction_policy import LRUEvictionPolicy
    from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface

    policy = LRUEvictionPolicy()
    listener = L2EvictionPolicy(policy)
    keys = [
        ObjectKey(chunk_hash=ObjectKey.IntHash2Bytes(i), model_name="m", kv_rank=0)
        for i in range(3)
    ]
    policy.on_keys_created(keys)

    adapter = MagicMock(spec=L2AdapterInterface)
    adapter._listeners = [listener]
    # Drive the real implementation; it must reach the eviction listener and
    # perform no read/load call on the adapter.
    L2AdapterInterface.touch_keys(adapter, [keys[0]])

    adapter._notify_keys_accessed.assert_called_once_with([keys[0]])
    adapter.submit_load_task.assert_not_called()


def test_touch_cached_keys_hits_both_tiers():
    """``StorageManager.touch_cached_keys`` refreshes L1 *and* every L2 adapter."""
    # First Party
    from lmcache.v1.distributed.storage_manager import StorageManager

    sm = MagicMock(spec=StorageManager)
    l2_a, l2_b = MagicMock(), MagicMock()
    sm._adapters_lock = threading.Lock()
    sm._l2_adapters = {0: l2_a, 1: l2_b}
    sm.touch_l1_keys = MagicMock()

    keys = [ObjectKey(chunk_hash=ObjectKey.IntHash2Bytes(7), model_name="m", kv_rank=0)]
    StorageManager.touch_cached_keys(sm, keys)

    sm.touch_l1_keys.assert_called_once_with(keys)
    l2_a.touch_keys.assert_called_once_with(keys)
    l2_b.touch_keys.assert_called_once_with(keys)


def test_submit_returns_clamped_covered_and_none_when_already_pending():
    """``maybe_submit_lookup_request`` reports what the server actually got.

    The caller must key its frozen covered boundary off the return value: a
    poll that no-ops because a lookup is already in flight returns ``None``,
    and a submitted lookup returns the count after clamping to the aligned
    range -- not the count the caller passed in.
    """
    # First Party
    from lmcache.integration.vllm.vllm_multi_process_adapter import (
        LMCacheMPSchedulerAdapter,
    )

    adapter = MagicMock(spec=LMCacheMPSchedulerAdapter)
    adapter.is_healthy = True
    adapter.lmcache_tokens_per_chunk = CHUNK_SIZE
    adapter.tp_size = 1
    adapter._pending_lookups = set()
    adapter._unacked_lookups = {}
    adapter._lookup_params = {}
    adapter._server_urls = ["u"]
    adapter.req_clients = {"u": MagicMock()}
    adapter._create_key = MagicMock()
    adapter._ensure_heartbeat_started = MagicMock()

    submit = LMCacheMPSchedulerAdapter.maybe_submit_lookup_request
    # 8 chunks of tokens, but ask for 99 covered chunks -> clamped to 8.
    sent = submit(
        adapter, "req", token_ids=list(range(8 * CHUNK_SIZE)), covered_chunks=99
    )
    assert sent == 8

    # Second poll: the request is still pending, so nothing is sent.
    assert (
        submit(adapter, "req", token_ids=list(range(8 * CHUNK_SIZE)), covered_chunks=3)
        is None
    )


@requires_vllm
def test_covered_boundary_frozen_only_when_a_lookup_is_sent():
    """A no-op poll must not move ``lookup_covered_tokens``.

    ``lookup_started_at`` is cleared once a result lands while the request
    stays in ``_pending_lookups``, so a later poll re-enters the submit path.
    If that poll re-froze the boundary from the APC hit it happens to see,
    the frozen value would diverge from the one the in-flight lookup used --
    making the shrink check compare against a boundary the server never had.
    """
    # First Party
    from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPConnector

    adapter = MagicMock()
    adapter.lmcache_tokens_per_chunk = CHUNK_SIZE
    # A lookup is already in flight, so the adapter sends nothing.
    adapter.maybe_submit_lookup_request.return_value = None
    adapter.check_lookup_result.return_value = None

    connector = LMCacheMPConnector.__new__(LMCacheMPConnector)
    connector.request_trackers = {}
    connector.scheduler_adapter = adapter
    connector._reserve_last_token_for_lookup = False
    connector._hit_alignment_tokens = CHUNK_SIZE
    connector._connector_stats = MagicMock()
    connector.lazy_offload = False
    connector._skip_covered_lookup = True

    request = _FakeRequest(list(range(8 * CHUNK_SIZE)))
    tracker = connector._get_or_create_request_tracker(request)
    # The in-flight lookup was built on a 4-chunk APC hit.
    tracker.lookup_covered_tokens = 4 * CHUNK_SIZE
    tracker.lookup_started_at = None  # cleared when the previous result landed

    # Poll again while a 7-chunk APC hit is visible. Before the fix this
    # re-froze the boundary to 7 chunks even though no LOOKUP went out.
    connector.get_num_new_matched_tokens(request, 7 * CHUNK_SIZE)

    assert tracker.lookup_covered_tokens == 4 * CHUNK_SIZE


def _shrink_connector(skip_enabled: bool = True):
    """A connector wired just enough to drive ``_handle_covered_shrink``."""
    # First Party
    from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPConnector
    from lmcache.integration.vllm.lmcache_mp_metadata import LMCacheMPRequestTracker

    adapter = MagicMock()
    adapter.lmcache_tokens_per_chunk = CHUNK_SIZE
    connector = LMCacheMPConnector.__new__(LMCacheMPConnector)
    connector.scheduler_adapter = adapter
    connector._skip_covered_lookup = skip_enabled

    request = _FakeRequest(list(range(8 * CHUNK_SIZE)))
    # _FakeRequest is duck-typed, not a vllm Request subclass.
    tracker = LMCacheMPRequestTracker(cast(Any, request))
    return LMCacheMPConnector, connector, adapter, request, tracker


@requires_vllm
def test_shrink_frees_stale_locks_blocking_and_goes_sticky():
    """A shrink must release the stale locks and never skip again.

    The replacement lookup re-locks the overlap, so the stale ``[c0, ret)``
    locks have to go first or their refcount leaks -- and read locks are
    shared across prefix-sharing requests, so a leak corrupts other requests.
    The release must be the blocking variant: ``LOOKUP`` and
    ``FREE_LOOKUP_LOCKS`` share the normal thread pool, so a fire-and-forget
    release can land after the replacement's ``begin_lookup`` has already
    reset the session state.
    """
    cls, connector, adapter, request, tracker = _shrink_connector()
    adapter.check_lookup_result.return_value = 6 * CHUNK_SIZE
    tracker.lookup_covered_tokens = 4 * CHUNK_SIZE
    tracker.num_vllm_hit_tokens = 2 * CHUNK_SIZE  # APC shrank below the boundary
    tracker.num_stored_tokens = 6 * CHUNK_SIZE
    tracker.num_lmcache_hit_tokens = 6 * CHUNK_SIZE

    result = cls._handle_covered_shrink(connector, request, tracker)

    # Scheduler is told to re-poll rather than given a hit length.
    assert result == (None, True)
    # Released through the blocking variant, over the whole hit range.
    adapter.free_lookup_locks_blocking.assert_called_once()
    assert adapter.free_lookup_locks_blocking.call_args.kwargs["start"] == 0
    assert adapter.free_lookup_locks_blocking.call_args.kwargs["end"] == 6 * CHUNK_SIZE
    adapter.free_lookup_locks.assert_not_called()
    adapter.cleanup_lookup_result.assert_called_once_with(request.request_id)
    # Sticky, so a second shrink cannot recur for this request.
    assert tracker.covered_skip_disabled is True
    # Per-lookup state reset so the next poll submits a fresh full lookup.
    assert tracker.lookup_covered_tokens == 0
    assert tracker.lookup_started_at is None
    assert tracker.num_stored_tokens == 0
    assert tracker.num_lmcache_hit_tokens == 0


@requires_vllm
def test_no_shrink_when_apc_hit_still_covers_the_boundary():
    """An APC hit at or above the frozen boundary is not a shrink."""
    cls, connector, adapter, request, tracker = _shrink_connector()
    adapter.check_lookup_result.return_value = 6 * CHUNK_SIZE
    tracker.lookup_covered_tokens = 4 * CHUNK_SIZE
    tracker.num_vllm_hit_tokens = 4 * CHUNK_SIZE  # exactly at the boundary

    assert cls._handle_covered_shrink(connector, request, tracker) is None
    adapter.free_lookup_locks_blocking.assert_not_called()
    assert tracker.covered_skip_disabled is False
    assert tracker.lookup_covered_tokens == 4 * CHUNK_SIZE


@requires_vllm
def test_shrink_is_inert_when_the_skip_is_disabled():
    """With the flag off, or once sticky, the hook must do nothing.

    ``covered_skip_disabled`` is what makes the fallback fire only once; if
    the hook still acted on a disabled request it would free locks the
    replacement full lookup legitimately holds.
    """
    cls, connector, adapter, request, tracker = _shrink_connector(skip_enabled=False)
    adapter.check_lookup_result.return_value = 6 * CHUNK_SIZE
    tracker.lookup_covered_tokens = 4 * CHUNK_SIZE
    tracker.num_vllm_hit_tokens = 0  # would be a shrink if the flag were on

    assert cls._handle_covered_shrink(connector, request, tracker) is None
    adapter.free_lookup_locks_blocking.assert_not_called()

    # Same once the request has already fallen back: boundary reset to 0.
    cls2, connector2, adapter2, request2, tracker2 = _shrink_connector()
    adapter2.check_lookup_result.return_value = 6 * CHUNK_SIZE
    tracker2.covered_skip_disabled = True
    tracker2.lookup_covered_tokens = 0
    tracker2.num_vllm_hit_tokens = 0

    assert cls2._handle_covered_shrink(connector2, request2, tracker2) is None
    adapter2.free_lookup_locks_blocking.assert_not_called()


def _lookup_ctx(n_chunks: int) -> MagicMock:
    """Mock engine context sufficient to drive ``LookupModule.lookup``."""
    ctx = _ctx_with_hashes(n_chunks)
    ctx.layout_desc_registry.find.return_value = MagicMock()
    ctx.layout_desc_registry.find_attn_desc.return_value = MagicMock(
        num_chunks_in_sw=[-1], num_object_groups=1
    )
    ctx.layout_desc_registry.find_group_layout_descs.return_value = {0: MagicMock()}
    return ctx


def test_server_ignores_covered_count_unless_enabled():
    """The server honors a covered count only when ``--skip-apc-lookup`` is on.

    The flag lives on the server because the server is what skips the lock and
    the prefetch. A server built without it must ignore the count outright: if
    it locked the prefix while the engine believed it had been skipped, the
    engine would elide the release and strand shared read locks.

    The LRU touch is the observable proof that the count was honored -- it
    happens only on the skip path.
    """
    # First Party
    from lmcache.v1.multiprocess.modules.lookup import LookupModule

    key = _make_key(4, covered_chunks=2)

    off_ctx = _lookup_ctx(4)
    LookupModule(off_ctx, skip_apc_lookup=False).lookup(key, 1)
    off_ctx.storage_manager.touch_cached_keys.assert_not_called()

    on_ctx = _lookup_ctx(4)
    LookupModule(on_ctx, skip_apc_lookup=True).lookup(key, 1)
    on_ctx.storage_manager.touch_cached_keys.assert_called_once()
