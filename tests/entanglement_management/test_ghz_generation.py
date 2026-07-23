"""Tests for the GHZ entanglement generation protocol.

One test class per logical unit, similar to the existing SeQUeNCe test suite:

    TestGHZNode                 node construction and memory assignment.
    TestGHZGenerationA          protocol logic in isolation.
    TestDegreeCap               MAX_DEGREE enforcement.
    TestFusionCircuit           fusion circuit at the stabilizer level.
    TestNoiseAndDecoherence     noise injection and T1/T2 decoherence.
    TestSuccessProbability      the q^(k-1) success check in _run_ghz_fusion.
    TestNotifyBellReady         notify_bell_ready accumulation and fusion trigger.
    TestEndToEnd                full cycle via RouterNetTopo and install_ghz_eg_rules.
    TestGHZMessage              GHZMessage payload construction.
    TestGhzEgMatchFunc          ghz_eg_match_func in isolation.
    TestTruncationMemoryMapping truncation stays in sync with the memory mapping.

Running from the repository root for my reference:
    python -m pytest tests/entanglement_management/test_ghz_generation.py -v
"""

import json
import pytest
from unittest.mock import MagicMock, patch
from stim import Circuit

from sequence.entanglement_management.ghz.ghz_generation import (
    GHZGenerationA,
    GHZEntanglementGenerationA,
    GHZMessage,
    GHZMsgType,
    GHZNode,
    MAX_DEGREE,
    DEFAULT_SUCCESS_BASE,
)
from sequence.entanglement_management.ghz.ghz_rules import (
    install_ghz_eg_rules,
    ghz_eg_match_func,
)
from sequence.entanglement_management.generation.generation_base import (
    EntanglementGenerationA,
)
from sequence.kernel.timeline import Timeline
from sequence.kernel.quantum_manager.stabilizer import QuantumManagerStabilizer
from sequence.kernel.quantum_state.stabilizer import StabilizerState
from sequence.topology.router_net_topo import RouterNetTopo


# Return a Timeline backed by QuantumManagerStabilizer.
def _make_timeline(seed: int = 0) -> Timeline:
    tl = Timeline()
    tl.quantum_manager = QuantumManagerStabilizer(seed=seed)
    return tl


# Construct a GHZNode with a deterministic seed.
def _make_ghz_node(name: str, tl: Timeline, neighbors: list[str], **kwargs) -> GHZNode:
    return GHZNode(name, tl, neighbor_names=neighbors, seed=0, **kwargs)


class TestGHZNode:
    """Tests for GHZNode construction and memory assignment."""

    def test_node_creates_memory_array(self):
        # One memory slot should be allocated per neighbor.
        tl = _make_timeline()
        node = _make_ghz_node("helper", tl, ["n1", "n2", "n3"])
        memo_arrays = node.get_components_by_type("MemoryArray")
        assert len(memo_arrays) == 1
        assert len(memo_arrays[0]) == 3

    def test_node_creates_ghz_protocol(self):
        # A GHZGenerationA protocol should be attached on construction.
        tl = _make_timeline()
        node = _make_ghz_node("helper", tl, ["n1", "n2"])
        assert isinstance(node.ghz_protocol, GHZGenerationA)
        assert node.ghz_protocol in node.protocols

    def test_node_has_resource_manager(self):
        # A resource_manager should be inherited from QuantumRouter.
        tl = _make_timeline()
        node = _make_ghz_node("helper", tl, ["n1", "n2", "n3"])
        assert hasattr(node, "resource_manager")
        assert node.resource_manager is not None

    def test_get_memory_for_neighbor(self):
        # Each registered neighbor should resolve to a Memory.
        tl = _make_timeline()
        node = _make_ghz_node("helper", tl, ["alice", "bob", "charlie"])
        for neighbor in ["alice", "bob", "charlie"]:
            assert node.get_memory_for_neighbor(neighbor) is not None

    def test_get_memory_for_unknown_neighbor_returns_none(self):
        # An unregistered neighbor should return None.
        tl = _make_timeline()
        node = _make_ghz_node("helper", tl, ["alice"])
        assert node.get_memory_for_neighbor("unknown") is None

    def test_degree_cap_enforced_on_construction(self):
        # The neighbor list should be truncated to MAX_DEGREE.
        tl = _make_timeline()
        too_many = [f"n{i}" for i in range(MAX_DEGREE + 3)]
        node = _make_ghz_node("helper", tl, too_many)
        assert len(node.neighbor_names) == MAX_DEGREE
        assert len(node.get_components_by_type("MemoryArray")[0]) == MAX_DEGREE


class TestGHZGenerationA:
    """Tests for GHZGenerationA protocol logic."""

    def test_init_validates_t1_t2(self):
        # init() should raise ValueError when T2 > 2*T1.
        tl = _make_timeline()
        node = _make_ghz_node("helper", tl, ["n1"], t1_sec=0.5, t2_sec=2.0)
        with pytest.raises(ValueError):
            node.ghz_protocol.init()

    def test_start_increments_cycle_count(self):
        # start() should increment the cycle counter and reset tracking state.
        tl = _make_timeline()
        node = _make_ghz_node("helper", tl, ["n1", "n2", "n3"])
        node.send_message = MagicMock()
        tl.init()
        node.ghz_protocol.start()
        assert node.ghz_protocol._cycle_count == 1
        assert len(node.ghz_protocol._ready_neighbors) == 0

    def test_broadcast_failure_sends_to_all_neighbors(self):
        # GENERATION_FAILED should go to every neighbor.
        tl = _make_timeline()
        node = _make_ghz_node("helper", tl, ["n1", "n2", "n3"])
        node.send_message = MagicMock()
        tl.init()
        node.ghz_protocol._cycle_count = 1
        node.ghz_protocol._broadcast_failure()

        sent_dsts = [c.args[0] for c in node.send_message.call_args_list]
        assert set(sent_dsts) == {"n1", "n2", "n3"}
        for c in node.send_message.call_args_list:
            assert c.args[1].msg_type == GHZMsgType.GENERATION_FAILED

    def test_reset_cycle_state_clears_tracking(self):
        # All per-cycle tracking dictionaries should be cleared.
        tl = _make_timeline()
        node = _make_ghz_node("helper", tl, ["n1"])
        node.send_message = MagicMock()
        tl.init()

        node.ghz_protocol._ready_neighbors = {"n1"}
        node.ghz_protocol._neighbor_keys = {"n1": 5}
        node.ghz_protocol._local_keys = {"n1": 3}
        node.ghz_protocol._reset_cycle_state()

        assert len(node.ghz_protocol._ready_neighbors) == 0
        assert len(node.ghz_protocol._neighbor_keys) == 0
        assert len(node.ghz_protocol._local_keys) == 0


class TestNotifyBellReady:
    """Tests for notify_bell_ready accumulation and the fusion trigger."""

    def test_notify_accumulates_neighbors(self):
        # Notifications should accumulate until all k neighbors report in.
        tl = _make_timeline()
        node = _make_ghz_node("helper", tl, ["n1", "n2", "n3"])
        node.send_message = MagicMock()
        tl.init()

        node.ghz_protocol.notify_bell_ready("n1", local_key=0, neighbor_key=10)
        node.ghz_protocol.notify_bell_ready("n2", local_key=1, neighbor_key=11)

        assert len(node.ghz_protocol._ready_neighbors) == 2
        assert "n3" not in node.ghz_protocol._ready_neighbors

    def test_notify_ignores_unknown_neighbor(self):
        # A notification from an unknown neighbor should be ignored.
        tl = _make_timeline()
        node = _make_ghz_node("helper", tl, ["n1"])
        node.send_message = MagicMock()
        tl.init()

        node.ghz_protocol.notify_bell_ready("stranger", local_key=0, neighbor_key=99)
        assert "stranger" not in node.ghz_protocol._ready_neighbors

    def test_notify_ignores_duplicate(self):
        # A duplicate notification should not overwrite the stored key.
        tl = _make_timeline()
        node = _make_ghz_node("helper", tl, ["n1", "n2"])
        node.send_message = MagicMock()
        tl.init()

        node.ghz_protocol.notify_bell_ready("n1", local_key=0, neighbor_key=10)
        node.ghz_protocol.notify_bell_ready("n1", local_key=0, neighbor_key=99)

        assert node.ghz_protocol._neighbor_keys["n1"] == 10

    def test_notify_triggers_fusion_when_all_ready(self):
        # Fusion should fire only once all k neighbors have notified.
        tl = _make_timeline()
        node = _make_ghz_node("helper", tl, ["n1", "n2"])
        node.send_message = MagicMock()
        tl.init()

        with patch.object(node.ghz_protocol, "_run_ghz_fusion") as mock_fusion:
            node.ghz_protocol.notify_bell_ready("n1", local_key=0, neighbor_key=10)
            mock_fusion.assert_not_called()
            node.ghz_protocol.notify_bell_ready("n2", local_key=1, neighbor_key=11)
            mock_fusion.assert_called_once()


class TestDegreeCap:
    """Tests for MAX_DEGREE enforcement."""

    def test_max_degree_value(self):
        # MAX_DEGREE should be 5 per the meeting notes.
        assert MAX_DEGREE == 5

    def test_default_success_base(self):
        # DEFAULT_SUCCESS_BASE should be 0.9.
        assert DEFAULT_SUCCESS_BASE == 0.9

    def test_protocol_truncates_neighbors(self):
        # The protocol should truncate neighbor_names to MAX_DEGREE.
        tl = _make_timeline()
        node = _make_ghz_node("helper", tl, ["n0"])
        too_many = [f"n{i}" for i in range(MAX_DEGREE + 5)]
        protocol = GHZGenerationA(node, "test_proto", too_many)
        assert len(protocol.neighbor_names) == MAX_DEGREE


class TestFusionCircuit:
    """Tests for the fusion circuit run directly on QuantumManagerStabilizer."""

    def test_two_qubit_fusion_produces_shared_state(self):
        # k=2 should give a shared 2-qubit StabilizerState.
        qm = QuantumManagerStabilizer(seed=0)
        k0, k1 = qm.new(), qm.new()

        prep = Circuit()
        prep.append("H", [0])
        qm.run_circuit(prep, [k0])

        fusion = Circuit()
        fusion.append("CX", [0, 1])
        fusion.append("H", [0])
        qm.run_circuit(fusion, [k0, k1])

        state = qm.get(k0)
        assert isinstance(state, StabilizerState)
        assert len(state.keys) == 2

    def test_three_qubit_fusion_produces_shared_state(self):
        # k=3 should give a shared 3-qubit StabilizerState.
        qm = QuantumManagerStabilizer(seed=0)
        k0, k1, k2 = qm.new(), qm.new(), qm.new()

        prep = Circuit()
        prep.append("H", [0])
        qm.run_circuit(prep, [k0])

        fusion = Circuit()
        fusion.append("CX", [0, 1])
        fusion.append("CX", [0, 2])
        fusion.append("H", [0])
        qm.run_circuit(fusion, [k0, k1, k2])

        state = qm.get(k0)
        assert isinstance(state, StabilizerState)
        assert set(state.keys) == {k0, k1, k2}

    def test_five_qubit_fusion_at_degree_cap(self):
        # k=5, the degree cap, should complete without error.
        qm = QuantumManagerStabilizer(seed=0)
        keys = [qm.new() for _ in range(5)]

        prep = Circuit()
        prep.append("H", [0])
        qm.run_circuit(prep, [keys[0]])

        fusion = Circuit()
        for target_idx in range(1, 5):
            fusion.append("CX", [0, target_idx])
        fusion.append("H", [0])

        qm.run_circuit(fusion, keys)
        state = qm.get(keys[0])
        assert isinstance(state, StabilizerState)
        assert len(state.keys) == 5


class TestNoiseAndDecoherence:
    """Tests for noise injection and T1/T2 idle decoherence."""

    def test_idling_decoherence_does_not_crash(self):
        # Valid T1/T2 should run without error.
        qm = QuantumManagerStabilizer(seed=0)
        keys = [qm.new() for _ in range(3)]
        qm.apply_idling_decoherence(keys, now_ps=int(1e12), t1_sec=1.0, t2_sec=0.5)

    def test_invalid_t2_raises_in_decoherence(self):
        # T2 > 2*T1 should raise ValueError.
        qm = QuantumManagerStabilizer(seed=0)
        keys = [qm.new()]
        with pytest.raises(ValueError):
            qm.apply_idling_decoherence(keys, now_ps=int(1e12), t1_sec=0.5, t2_sec=2.0)

    def test_gate_error_injection_runs(self):
        # inject_gate_error=True should complete without error.
        qm = QuantumManagerStabilizer(seed=0, one_qubit_gate_fid=0.95, two_qubit_gate_fid=0.95)
        k0, k1 = qm.new(), qm.new()

        circuit = Circuit()
        circuit.append("H", [0])
        circuit.append("CX", [0, 1])
        qm.run_circuit(circuit, [k0, k1], inject_gate_error=True)

    def test_error_statistics_increment(self):
        # Error counters should increment when gate fidelity is 0.
        qm = QuantumManagerStabilizer(seed=42, one_qubit_gate_fid=0.0)
        k0 = qm.new()

        circuit = Circuit()
        circuit.append("H", [0])
        qm.run_circuit(circuit, [k0], inject_gate_error=True)

        stats = qm.get_error_statistics()
        assert stats["gate_1q_count"] >= 1


class TestSuccessProbability:
    """Tests for the q^(k-1) success check in _run_ghz_fusion."""

    def _setup_node_ready_for_fusion(
        self, neighbors: list[str], success_base: float = 0.9
    ) -> tuple["GHZNode", MagicMock]:
        # Return a (node, send_mock) pair with state set as if all Bell pairs are ready.
        tl = _make_timeline()
        node = _make_ghz_node("helper", tl, neighbors, success_base=success_base)
        send_mock: MagicMock = MagicMock()
        node.send_message = send_mock  # type: ignore[method-assign]
        tl.init()
        node.ghz_protocol._cycle_count = 1
        node.ghz_protocol._local_keys = {n: i for i, n in enumerate(node.neighbor_names)}
        node.ghz_protocol._neighbor_keys = {n: 100 + i for i, n in enumerate(node.neighbor_names)}
        node.ghz_protocol._ready_neighbors = set(node.neighbor_names)
        return node, send_mock

    def test_always_passes_for_k1(self):
        # For k=1, q^0 = 1.0, so the check never fails.
        node, send_mock = self._setup_node_ready_for_fusion(["n1"], success_base=0.0)
        with patch.object(node.ghz_protocol, "_random", return_value=0.9999) as mock_random:
            node.ghz_protocol._run_ghz_fusion()
        assert mock_random.call_count == 1

    def test_fails_when_random_exceeds_threshold(self):
        # Failure should broadcast when random() >= success_base^(k-1).
        neighbors = ["n1", "n2", "n3"]
        node, send_mock = self._setup_node_ready_for_fusion(neighbors, success_base=0.9)
        with patch.object(node.ghz_protocol, "_random", return_value=0.99) as mock_random:
            node.ghz_protocol._run_ghz_fusion()
        sent_types = [c.args[1].msg_type for c in send_mock.call_args_list]
        assert all(t == GHZMsgType.GENERATION_FAILED for t in sent_types)
        assert len(sent_types) == len(neighbors)

    def test_passes_when_random_below_threshold(self):
        # The check should not fail when random() < success_base^(k-1).
        neighbors = ["n1", "n2", "n3"]
        node, send_mock = self._setup_node_ready_for_fusion(neighbors, success_base=0.9)
        with patch.object(node.ghz_protocol, "_random", return_value=0.10) as mock_random:
            node.ghz_protocol._run_ghz_fusion()

    def test_zero_success_base_always_fails_for_k_gt_1(self):
        # success_base=0.0 gives q^(k-1)=0 for k>1, so fusion always fails.
        neighbors = ["n1", "n2"]
        node, send_mock = self._setup_node_ready_for_fusion(neighbors, success_base=0.0)
        with patch.object(node.ghz_protocol, "_random", return_value=0.0) as mock_random:
            node.ghz_protocol._run_ghz_fusion()
        sent_types = [c.args[1].msg_type for c in send_mock.call_args_list]
        assert GHZMsgType.GENERATION_FAILED in sent_types

    def test_unit_success_base_never_fails_check(self):
        # success_base=1.0 gives q^(k-1)=1.0 for any k, so the check never fires.
        neighbors = ["n1", "n2", "n3", "n4", "n5"]
        node, send_mock = self._setup_node_ready_for_fusion(neighbors, success_base=1.0)
        with patch.object(node.ghz_protocol, "_random", return_value=0.9999) as mock_random:
            node.ghz_protocol._run_ghz_fusion()
        assert mock_random.call_count == 1

    def test_threshold_scales_with_k(self):
        # The failure rate should increase with k under the exponential-decay model.
        for k_neighbors in [["n1"], ["n1", "n2"]]:
            node, send_mock = self._setup_node_ready_for_fusion(k_neighbors, success_base=0.9)
            k = len(k_neighbors)
            threshold = 0.9 ** (k - 1)
            draw = 0.95
            with patch.object(node.ghz_protocol, "_random", return_value=draw) as mock_random:
                node.ghz_protocol._run_ghz_fusion()
            if draw >= threshold:
                sent_types = [c.args[1].msg_type for c in send_mock.call_args_list]
                assert GHZMsgType.GENERATION_FAILED in sent_types
                assert mock_random.call_count == 1
            else:
                assert mock_random.call_count == 1


class TestEndToEnd:
    """End-to-end test loading the 4-node star topology via RouterNetTopo.

    RouterNetTopo creates standard QuantumRouter nodes from JSON, so the helper
    is a QuantumRouter here rather than a GHZNode subclass. GHZGenerationA and
    install_ghz_eg_rules are installed manually after topology load, mirroring
    the custom rule injection pattern in SeQUeNCe's Chapter 4 tutorial.
    """

    @pytest.fixture
    def star_topo_path(self, tmp_path) -> str:
        # Write star_network.json to a temp dir and return its path.
        config = {
            "stop_time": 10000000000000,
            "nodes": [
                {"name": "helper", "type": "QuantumRouter", "seed": 0, "memo_size": 3},
                {"name": "n1", "type": "QuantumRouter", "seed": 1, "memo_size": 1},
                {"name": "n2", "type": "QuantumRouter", "seed": 2, "memo_size": 1},
                {"name": "n3", "type": "QuantumRouter", "seed": 3, "memo_size": 1},
            ],
            "qconnections": [
                {"node1": "helper", "node2": "n1", "attenuation": 0.0,
                 "distance": 500, "type": "meet_in_the_middle"},
                {"node1": "helper", "node2": "n2", "attenuation": 0.0,
                 "distance": 500, "type": "meet_in_the_middle"},
                {"node1": "helper", "node2": "n3", "attenuation": 0.0,
                 "distance": 500, "type": "meet_in_the_middle"},
            ],
            "cconnections": [
                {"node1": "helper", "node2": "n1", "delay": 500000000},
                {"node1": "helper", "node2": "n2", "delay": 500000000},
                {"node1": "helper", "node2": "n3", "delay": 500000000},
                {"node1": "n1", "node2": "n2", "delay": 1000000000},
                {"node1": "n1", "node2": "n3", "delay": 1000000000},
                {"node1": "n2", "node2": "n3", "delay": 1000000000},
            ],
        }
        path = tmp_path / "star_network.json"
        path.write_text(json.dumps(config))
        return str(path)

    def test_bell_pairs_generated_on_all_links(self, star_topo_path):
        # All three helper memories should reach ENTANGLED within sim time.
        from sequence.entanglement_management.generation import (
            EntanglementGenerationA, EntanglementGenerationB,
        )
        from sequence.constants import BARRET_KOK
        from sequence.resource_management.memory_manager import MemoryInfo
        from sequence.topology.node import QuantumRouter as QR

        EntanglementGenerationA.set_global_type(BARRET_KOK)
        EntanglementGenerationB.set_global_type(BARRET_KOK)

        topo = RouterNetTopo(star_topo_path)
        tl = topo.get_timeline()

        routers = {r.name: r for r in topo.get_nodes_by_type(RouterNetTopo.QUANTUM_ROUTER)}
        helper = routers["helper"]  # type: ignore[assignment]
        neighbor_names = ["n1", "n2", "n3"]

        # Attach GHZGenerationA to the helper node.
        ghz_protocol = GHZGenerationA(
            owner=helper,  # type: ignore[arg-type]
            name="helper.GHZGenerationA",
            neighbor_names=neighbor_names,
            success_base=1.0,
            t1_sec=1.0,
            t2_sec=0.5,
        )

        # Build middle_node_map from the topology's map_to_middle_node attribute.
        map_to_middle = getattr(helper, "map_to_middle_node", {})  # type: ignore[attr-defined]
        middle_node_map = {n: map_to_middle[n] for n in neighbor_names}

        # Patch ghz_node.ghz_protocol so install_ghz_eg_rules can find it.
        helper.ghz_protocol = ghz_protocol  # type: ignore[attr-defined]

        install_ghz_eg_rules(helper, middle_node_map)  # type: ignore[arg-type]

        tl.init()
        tl.run()

        for idx, neighbor in enumerate(neighbor_names):
            info = helper.resource_manager.memory_manager[idx]  # type: ignore[attr-defined]
            assert info.state == MemoryInfo.ENTANGLED, (
                f"Memory {idx} (for {neighbor}) not ENTANGLED after simulation; "
                f"state={info.state}"
            )


class TestGHZMessage:
    """Tests for GHZMessage payload construction, isolated from the protocol
    logic that builds these messages indirectly elsewhere in this file.
    """

    def test_ghz_result_payload_round_trips(self):
        # GHZ_RESULT payload fields should round-trip exactly.
        msg = GHZMessage(
            GHZMsgType.GHZ_RESULT,
            receiver="neighbor.GHZGenerationA",
            x_correction=1,
            z_correction=0,
            cycle=3,
        )
        assert msg.msg_type == GHZMsgType.GHZ_RESULT
        assert msg.receiver == "neighbor.GHZGenerationA"
        assert msg.payload == {"x_correction": 1, "z_correction": 0, "cycle": 3}

    def test_generation_failed_payload_round_trips(self):
        # GENERATION_FAILED payload fields should round-trip exactly.
        msg = GHZMessage(
            GHZMsgType.GENERATION_FAILED,
            receiver="neighbor.GHZGenerationA",
            cycle=7,
        )
        assert msg.msg_type == GHZMsgType.GENERATION_FAILED
        assert msg.receiver == "neighbor.GHZGenerationA"
        assert msg.payload == {"cycle": 7}

    def test_payload_defaults_to_empty_dict_with_no_kwargs(self):
        # No extra kwargs should give an empty payload.
        msg = GHZMessage(GHZMsgType.GENERATION_FAILED, receiver="x.GHZGenerationA")
        assert msg.payload == {}

    def test_two_messages_have_independent_payloads(self):
        # Payload dicts on separate instances should not be aliased.
        msg1 = GHZMessage(GHZMsgType.GHZ_RESULT, receiver="a", x_correction=1)
        msg2 = GHZMessage(GHZMsgType.GHZ_RESULT, receiver="b", x_correction=0)
        msg1.payload["x_correction"] = 99
        assert msg2.payload["x_correction"] == 0

    def test_msg_type_enum_members_distinct(self):
        # GHZMsgType members should be distinct enum values.
        assert GHZMsgType.GHZ_RESULT != GHZMsgType.GENERATION_FAILED


class TestGhzEgMatchFunc:
    """Tests for ghz_eg_match_func in isolation, without the full
    RouterNetTopo and tl.run() path exercised by TestEndToEnd.
    """

    def _make_fake_ega(self, remote_node_name: str, memory, owner) -> MagicMock:
        # Mock satisfying isinstance(_, EntanglementGenerationA) plus the
        # attributes ghz_eg_match_func reads.
        fake = MagicMock(spec=EntanglementGenerationA)
        fake.remote_node_name = remote_node_name
        fake.memory = memory
        fake.owner = owner
        return fake

    def test_matches_protocol_with_correct_remote_node_and_index(self):
        # Should return the protocol matching both remote_node_name and index.
        tl = _make_timeline()
        neighbor = _make_ghz_node("neighbor", tl, ["placeholder"])
        mem_arr = neighbor.get_components_by_type("MemoryArray")[0]
        target_memory = mem_arr[0]

        matching = self._make_fake_ega("helper", target_memory, neighbor)
        non_matching_node = self._make_fake_ega("other_helper", target_memory, neighbor)

        protocols = [non_matching_node, matching]
        args = {"remote_node": "helper", "memory_index": 0}

        result = ghz_eg_match_func(protocols, args)
        assert result is matching

    def test_no_match_returns_none(self):
        # Should return None when nothing matches.
        tl = _make_timeline()
        neighbor = _make_ghz_node("neighbor", tl, ["placeholder"])
        mem_arr = neighbor.get_components_by_type("MemoryArray")[0]
        target_memory = mem_arr[0]

        non_matching = self._make_fake_ega("wrong_node", target_memory, neighbor)
        protocols = [non_matching]
        args = {"remote_node": "helper", "memory_index": 0}

        assert ghz_eg_match_func(protocols, args) is None

    def test_ignores_non_entanglement_generation_a_protocols(self):
        # Non-EntanglementGenerationA protocols should be skipped, not raise.
        not_an_ega = MagicMock()  # deliberately not spec'd as EntanglementGenerationA
        protocols = [not_an_ega]
        args = {"remote_node": "helper", "memory_index": 0}

        assert ghz_eg_match_func(protocols, args) is None

    def test_matches_correct_memory_index_among_several(self):
        # Should distinguish by memory index, not just remote_node_name.
        tl = _make_timeline()
        neighbor = _make_ghz_node("neighbor", tl, ["p1", "p2"])
        mem_arr = neighbor.get_components_by_type("MemoryArray")[0]

        proto_idx0 = self._make_fake_ega("helper", mem_arr[0], neighbor)
        proto_idx1 = self._make_fake_ega("helper", mem_arr[1], neighbor)

        protocols = [proto_idx0, proto_idx1]
        args = {"remote_node": "helper", "memory_index": 1}

        result = ghz_eg_match_func(protocols, args)
        assert result is proto_idx1


class TestTruncationMemoryMapping:
    """Tests that MAX_DEGREE truncation stays consistent with
    _neighbor_to_memory_idx and the MemoryArray, rather than only checking
    list length as the earlier degree cap tests do.
    """

    def test_truncated_neighbors_all_resolve_to_valid_memories(self):
        # Surviving neighbors should map to real Memories, gap-free.
        tl = _make_timeline()
        too_many = [f"n{i}" for i in range(MAX_DEGREE + 4)]
        node = _make_ghz_node("helper", tl, too_many)

        assert len(node._neighbor_to_memory_idx) == MAX_DEGREE
        assert set(node._neighbor_to_memory_idx.values()) == set(range(MAX_DEGREE))

        for neighbor in node.neighbor_names:
            assert node.get_memory_for_neighbor(neighbor) is not None

    def test_dropped_neighbors_are_not_in_mapping_or_retrievable(self):
        # Truncated-away neighbors should be absent from the mapping.
        tl = _make_timeline()
        too_many = [f"n{i}" for i in range(MAX_DEGREE + 4)]
        node = _make_ghz_node("helper", tl, too_many)

        dropped = too_many[MAX_DEGREE:]
        assert len(dropped) == 4
        for neighbor in dropped:
            assert neighbor not in node._neighbor_to_memory_idx
            assert node.get_memory_for_neighbor(neighbor) is None

    def test_memory_array_size_matches_truncated_neighbor_count_not_original(self):
        # MemoryArray size should reflect the post-cap count, not the request.
        tl = _make_timeline()
        requested = MAX_DEGREE + 10
        too_many = [f"n{i}" for i in range(requested)]
        node = _make_ghz_node("helper", tl, too_many)

        mem_arr = node.get_components_by_type("MemoryArray")[0]
        assert len(mem_arr) == MAX_DEGREE
        assert len(mem_arr) != requested

    def test_at_or_under_cap_no_truncation_occurs(self):
        # At or under the cap, no truncation should occur. Guards against an
        # off-by-one in the truncation slice.
        tl = _make_timeline()
        exactly_at_cap = [f"n{i}" for i in range(MAX_DEGREE)]
        node = _make_ghz_node("helper", tl, exactly_at_cap)

        assert node.neighbor_names == exactly_at_cap
        assert len(node._neighbor_to_memory_idx) == MAX_DEGREE
        for i, neighbor in enumerate(exactly_at_cap):
            assert node._neighbor_to_memory_idx[neighbor] == i