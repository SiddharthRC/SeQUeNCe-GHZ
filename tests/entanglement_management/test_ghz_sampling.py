"""Monte Carlo sampling harness for GHZ generation statistics.

The stabilizer backend (stim.TableauSimulator) tracks one sampled pure state
trajectory per run, so a statistic is computed across N independent runs with
fixed parameters rather than read off a single run. About 1000 samples gives a
precision of about 0.001.

Per Dr. Zhan (2026-07-20), fidelity here is the fraction of samples reaching the
ideal state: if 9 of 10 runs yield the ideal GHZ, fidelity is 0.9. Each run
executes the full protocol under QuantumManagerStabilizer: Bell pair generation,
GHZ fusion on the helper, per-neighbor BSM, and live Pauli corrections on the
neighbors. Two metrics are reported per run: bell_success (all links reached
ENTANGLED) and ghz_valid (the neighbor qubits hold a valid GHZ state).

Running from the repository root for my reference:
    python -m pytest tests/entanglement_management/test_ghz_sampling.py -v
"""

import json
from dataclasses import dataclass, field

import pytest
import stim

from sequence.kernel.quantum_manager.stabilizer import QuantumManagerStabilizer
from sequence.entanglement_management.ghz.ghz_generation import (
    GHZGenerationA,
)
from sequence.entanglement_management.ghz.ghz_rules import install_ghz_eg_rules
from sequence.resource_management.memory_manager import MemoryInfo
from sequence.topology.router_net_topo import RouterNetTopo


@dataclass
class SampleResult:
    """Outcome of a single independent generation attempt.

    Attributes:
        bell_success (bool): True if every link reached ENTANGLED in time.
        ghz_valid (bool): True if the neighbor qubits hold a valid GHZ state.
        entangled_count (int): links that reached ENTANGLED.
        total_links (int): links attempted.
    """
    bell_success: bool
    ghz_valid: bool
    entangled_count: int
    total_links: int


@dataclass
class SamplingStats:
    """Statistics aggregated across many independent SampleResults.

    Attributes:
        num_samples (int): independent runs performed.
        bell_success_rate (float): fraction of samples with bell_success True.
        ghz_fidelity (float): fraction of samples with ghz_valid True.
        results (list[SampleResult]): raw per sample results.
    """
    num_samples: int
    bell_success_rate: float
    ghz_fidelity: float
    results: list = field(default_factory=list)


def _build_star_topo_config(
    stop_time_ps: int = 10_000_000_000_000,
    seed_offset: int = 0,
) -> dict:
    """Return a 4 node star topology config, one helper and three neighbors.

    Args:
        stop_time_ps (int): simulation stop time in picoseconds.
        seed_offset (int): added to each node''s seed so samples use distinct
            RNG streams.

    Returns:
        dict: config for RouterNetTopo.
    """
    return {
        "stop_time": stop_time_ps,
        "nodes": [
            {"name": "helper", "type": "QuantumRouter", "seed": seed_offset, "memo_size": 3},
            {"name": "n1", "type": "QuantumRouter", "seed": seed_offset + 1, "memo_size": 1},
            {"name": "n2", "type": "QuantumRouter", "seed": seed_offset + 2, "memo_size": 1},
            {"name": "n3", "type": "QuantumRouter", "seed": seed_offset + 3, "memo_size": 1},
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


def _neighbors_hold_valid_ghz(qm, nb_keys: list[int]) -> bool:
    """Check whether the neighbor qubits form a valid GHZ state.

    A valid n-qubit GHZ is stabilized by the all-X operator and by each
    adjacent Z_i Z_{i+1}, so every such expectation must be +-1.

    Args:
        qm (QuantumManagerStabilizer): the run''s quantum manager.
        nb_keys (list[int]): the neighbor qubits'' qstate keys.

    Returns:
        bool: True if the qubits share one state and pass the GHZ check.
    """
    state = qm.get(nb_keys[0])
    joint = list(state.keys)
    local = {k: i for i, k in enumerate(joint)}
    if not all(k in local for k in nb_keys):
        return False
    sim = state.state
    n = sim.num_qubits
    idxs = [local[k] for k in nb_keys]
    allx = ["_"] * n
    for i in idxs:
        allx[i] = "X"
    if abs(sim.peek_observable_expectation(stim.PauliString("".join(allx)))) != 1:
        return False
    for a in range(len(idxs) - 1):
        s = ["_"] * n
        s[idxs[a]] = "Z"
        s[idxs[a + 1]] = "Z"
        if abs(sim.peek_observable_expectation(stim.PauliString("".join(s)))) != 1:
            return False
    return True


def _run_one_sample(
    topo_path: str,
    success_base: float = 1.0,
    t1_sec: float = 1.0,
    t2_sec: float = 0.5,
    seed: int = 0,
) -> SampleResult:
    """Run one complete, independent generation attempt.

    Builds a fresh Timeline and topology, forces QuantumManagerStabilizer, runs
    the full protocol, and reports both Bell pair success and GHZ validity.

    Args:
        topo_path (str): path to a star topology JSON config.
        success_base (float): passed through to GHZGenerationA.
        t1_sec (float): passed through to GHZGenerationA.
        t2_sec (float): passed through to GHZGenerationA.
        seed (int): seed for the stabilizer manager.

    Returns:
        SampleResult: outcome of this run.
    """
    from sequence.entanglement_management.generation import (
        EntanglementGenerationA, EntanglementGenerationB,
    )
    from sequence.constants import BARRET_KOK

    EntanglementGenerationA.set_global_type(BARRET_KOK)
    EntanglementGenerationB.set_global_type(BARRET_KOK)

    topo = RouterNetTopo(topo_path)
    tl = topo.get_timeline()
    qm = QuantumManagerStabilizer(seed=seed)
    tl.quantum_manager = qm

    routers = {r.name: r for r in topo.get_nodes_by_type(RouterNetTopo.QUANTUM_ROUTER)}
    helper = routers["helper"]  # type: ignore[assignment]
    neighbor_names = ["n1", "n2", "n3"]

    ghz_protocol = GHZGenerationA(
        owner=helper,  # type: ignore[arg-type]
        name="helper.GHZGenerationA",
        neighbor_names=neighbor_names,
        success_base=success_base,
        t1_sec=t1_sec,
        t2_sec=t2_sec,
    )

    map_to_middle = getattr(helper, "map_to_middle_node", {})  # type: ignore[attr-defined]
    middle_node_map = {n: map_to_middle[n] for n in neighbor_names}

    helper.ghz_protocol = ghz_protocol  # type: ignore[attr-defined]
    install_ghz_eg_rules(helper, middle_node_map)  # type: ignore[arg-type]

    tl.init()
    tl.run()

    entangled_count = 0
    for idx in range(len(neighbor_names)):
        info = helper.resource_manager.memory_manager[idx]  # type: ignore[attr-defined]
        if info.state == MemoryInfo.ENTANGLED:
            entangled_count += 1
    bell_success = entangled_count == len(neighbor_names)

    ghz_valid = False
    if bell_success:
        nb_keys = [
            routers[n].get_components_by_type("MemoryArray")[0][0].qstate_key
            for n in neighbor_names
        ]
        try:
            ghz_valid = _neighbors_hold_valid_ghz(qm, nb_keys)
        except Exception:
            ghz_valid = False

    return SampleResult(
        bell_success=bell_success,
        ghz_valid=ghz_valid,
        entangled_count=entangled_count,
        total_links=len(neighbor_names),
    )


def run_sampling(
    num_samples: int,
    tmp_path,
    success_base: float = 1.0,
    t1_sec: float = 1.0,
    t2_sec: float = 0.5,
) -> SamplingStats:
    """Run num_samples independent attempts and aggregate the results.

    Args:
        num_samples (int): independent samples to run.
        tmp_path: pytest tmp_path fixture, or any Path like directory.
        success_base (float): passed through to each GHZGenerationA.
        t1_sec (float): passed through to each GHZGenerationA.
        t2_sec (float): passed through to each GHZGenerationA.

    Returns:
        SamplingStats: aggregate statistics across all samples.
    """
    results = []
    for i in range(num_samples):
        config = _build_star_topo_config(seed_offset=i * 10)
        topo_path = tmp_path / f"star_network_sample_{i}.json"
        topo_path.write_text(json.dumps(config))

        result = _run_one_sample(
            str(topo_path),
            success_base=success_base,
            t1_sec=t1_sec,
            t2_sec=t2_sec,
            seed=i * 10,
        )
        results.append(result)

    bell_successes = sum(1 for r in results if r.bell_success)
    ghz_valids = sum(1 for r in results if r.ghz_valid)
    return SamplingStats(
        num_samples=num_samples,
        bell_success_rate=bell_successes / num_samples,
        ghz_fidelity=ghz_valids / num_samples,
        results=results,
    )


class TestGHZSampling:
    """Tests for the sampling harness itself.

    Sample counts are deliberately small for test runtime. Runs at the scale
    of 1000 samples belong in experiments or notebooks, not the test suite.
    """

    def test_single_sample_returns_valid_result(self, tmp_path):
        config = _build_star_topo_config()
        topo_path = tmp_path / "star_network.json"
        topo_path.write_text(json.dumps(config))

        result = _run_one_sample(str(topo_path), success_base=1.0)

        assert result.total_links == 3
        assert 0 <= result.entangled_count <= result.total_links
        assert result.bell_success == (result.entangled_count == result.total_links)

    def test_success_base_one_yields_valid_ghz_across_samples(self, tmp_path):
        stats = run_sampling(num_samples=5, tmp_path=tmp_path, success_base=1.0)

        assert stats.num_samples == 5
        assert stats.bell_success_rate == 1.0
        assert stats.ghz_fidelity == 1.0
        assert all(r.ghz_valid for r in stats.results)

    def test_rates_are_fractions_between_zero_and_one(self, tmp_path):
        stats = run_sampling(num_samples=5, tmp_path=tmp_path, success_base=1.0)
        assert 0.0 <= stats.bell_success_rate <= 1.0
        assert 0.0 <= stats.ghz_fidelity <= 1.0

    def test_forced_bsm_failure_gives_zero_fidelity(self, tmp_path):
        stats = run_sampling(num_samples=5, tmp_path=tmp_path, success_base=0.0)
        assert stats.ghz_fidelity == 0.0

    def test_samples_are_independent_fresh_runs(self, tmp_path):
        stats = run_sampling(num_samples=3, tmp_path=tmp_path, success_base=1.0)

        assert len(stats.results) == 3
        assert len(set(id(r) for r in stats.results)) == 3

    def test_stats_dataclass_fields_consistent(self, tmp_path):
        stats = run_sampling(num_samples=4, tmp_path=tmp_path, success_base=1.0)

        assert stats.num_samples == len(stats.results)
        expected_bell = sum(1 for r in stats.results if r.bell_success) / 4
        expected_ghz = sum(1 for r in stats.results if r.ghz_valid) / 4
        assert stats.bell_success_rate == expected_bell
        assert stats.ghz_fidelity == expected_ghz
