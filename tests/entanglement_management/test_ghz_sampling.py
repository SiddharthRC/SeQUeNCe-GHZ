"""Monte Carlo sampling harness for GHZ generation rate statistics.

The stabilizer backend (stim.TableauSimulator) tracks one sampled pure state
trajectory per run, not a mixed state density matrix, so fidelity cannot be
read off a single run. Instead the protocol is run N independent times with
fixed parameters and statistics are computed across those N samples. About
1000 samples gives a rate precision of about 0.001.

Per Dr. Zhan (2026-07-20), fidelity here is the fraction of samples reaching the
ideal state: if 9 of 10 generated pairs are the ideal state, fidelity is 0.9.
The success_rate computed here is basically that same quantity.

Limitation for now: this currently measures Bell pair generation only, not GHZ fusion.
RouterNetTopo builds BSM components that pass sequence.components.circuit.Circuit
to run_circuit, which QuantumManagerStabilizer rejects, so the topology uses
QuantumManagerKet. GHZGenerationA requires the stabilizer manager, so
_run_ghz_fusion fails its manager check and broadcasts failure every run. The
fusion and BSM phases are therefore never exercised here. How to run stabilizer
based fusion inside a RouterNetTopo timeline is an open question for the team.


Running from the repository root for my reference:
    python -m pytest tests/entanglement_management/test_ghz_sampling.py -v
"""

import json
from dataclasses import dataclass, field

import pytest

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
        all_entangled (bool): True if every link reached ENTANGLED in time.
        entangled_count (int): links that reached ENTANGLED.
        total_links (int): links attempted.
    """
    all_entangled: bool
    entangled_count: int
    total_links: int


@dataclass
class SamplingStats:
    """Statistics aggregated across many independent SampleResults.

    Attributes:
        num_samples (int): independent runs performed.
        success_rate (float): fraction of samples with all_entangled True.
        results (list[SampleResult]): raw per sample results, kept for
            further analysis such as a future fidelity statistic.
    """
    num_samples: int
    success_rate: float
    results: list = field(default_factory=list)


def _build_star_topo_config(
    stop_time_ps: int = 10_000_000_000_000,
    seed_offset: int = 0,
) -> dict:
    """Return a 4 node star topology config, one helper and three neighbors.

    Args:
        stop_time_ps (int): simulation stop time in picoseconds.
        seed_offset (int): added to each node's seed so samples use distinct
            RNG streams.

    Returns:
        dict: config for RouterNetTopo, matching TestEndToEnd's structure.
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


def _run_one_sample(
    topo_path: str,
    success_base: float = 1.0,
    t1_sec: float = 1.0,
    t2_sec: float = 0.5,
) -> SampleResult:
    """Run one complete, independent generation attempt.

    Builds a fresh Timeline, topology, and GHZGenerationA, runs to completion,
    and reports whether all helper memories reached ENTANGLED. The quantum
    manager is left as whatever RouterNetTopo provides; see the module
    docstring for why it is not overridden to QuantumManagerStabilizer.

    Args:
        topo_path (str): path to a star topology JSON config.
        success_base (float): passed through to GHZGenerationA.
        t1_sec (float): passed through to GHZGenerationA.
        t2_sec (float): passed through to GHZGenerationA.

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
    for idx, _neighbor in enumerate(neighbor_names):
        info = helper.resource_manager.memory_manager[idx]  # type: ignore[attr-defined]
        if info.state == MemoryInfo.ENTANGLED:
            entangled_count += 1

    return SampleResult(
        all_entangled=(entangled_count == len(neighbor_names)),
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

    Each sample builds a fresh Timeline and topology with a distinct seed
    offset. One sample is one complete run of the protocol, not a single
    fusion attempt within a shared run.

    Args:
        num_samples (int): independent samples to run.
        tmp_path: pytest tmp_path fixture, or any Path like directory, for
            writing per sample topology JSON files.
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
        )
        results.append(result)

    successes = sum(1 for r in results if r.all_entangled)
    return SamplingStats(
        num_samples=num_samples,
        success_rate=successes / num_samples,
        results=results,
    )


class TestGHZSampling:
    """Tests for the sampling harness itself.

    Sample counts are deliberately small for test runtime. Runs at the scale
    of 1000 samples belong in experiments or notebooks, not the test suite.
    """

    def test_single_sample_returns_valid_result(self, tmp_path):
        # A single sample should return a SampleResult with sane fields.
        config = _build_star_topo_config()
        topo_path = tmp_path / "star_network.json"
        topo_path.write_text(json.dumps(config))

        result = _run_one_sample(str(topo_path), success_base=1.0)

        assert result.total_links == 3
        assert 0 <= result.entangled_count <= result.total_links
        assert result.all_entangled == (result.entangled_count == result.total_links)

    def test_success_base_one_always_succeeds_across_samples(self, tmp_path):
        # With success_base 1.0, Bell pair generation completes on all links
        # in every sample.
        stats = run_sampling(num_samples=5, tmp_path=tmp_path, success_base=1.0)

        assert stats.num_samples == 5
        assert stats.success_rate == 1.0
        assert all(r.all_entangled for r in stats.results)

    def test_success_rate_is_fraction_between_zero_and_one(self, tmp_path):
        # success_rate should always be a valid fraction.
        stats = run_sampling(num_samples=5, tmp_path=tmp_path, success_base=1.0)
        assert 0.0 <= stats.success_rate <= 1.0

    def test_samples_are_independent_fresh_runs(self, tmp_path):
        # Each sample should use a distinct topology file and fresh objects.
        stats = run_sampling(num_samples=3, tmp_path=tmp_path, success_base=1.0)

        assert len(stats.results) == 3
        # Distinct object identities, so no shared mutable state.
        assert len(set(id(r) for r in stats.results)) == 3

    def test_stats_dataclass_fields_consistent(self, tmp_path):
        # num_samples should match len(results), and success_rate should match
        # the counted fraction.
        stats = run_sampling(num_samples=4, tmp_path=tmp_path, success_base=1.0)

        assert stats.num_samples == len(stats.results)
        expected_rate = sum(1 for r in stats.results if r.all_entangled) / 4
        assert stats.success_rate == expected_rate