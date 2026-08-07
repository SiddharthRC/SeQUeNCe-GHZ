"""Fusion-based GHZ state generation protocol and helper node type.

Implements the hybrid GHZ-BSM routing protocol from Chen et al., arXiv:2604.03155
(2026), using the fusion approach of Bartolucci et al., Nat. Commun. 14, 912 (2023).

Three phases on a central helper node (GHZNode):
    1. Bell state generation with each neighbor via GHZEntanglementGenerationA.
    2. GHZ fusion (deterministic): CX from a control qubit into each remaining
       qubit, then H, converting k local Bell-pair qubits into a k-qubit GHZ state.
    3. BSM on each (GHZ qubit, neighbor qubit) pair, each succeeding with a single
       probability q, with Pauli corrections sent to each neighbor.

In the hybrid GHZ-BSM protocol (Chen et al., Sec.~3), GHZ generation is
deterministic local preparation and the probabilistic success q is applied at the
BSM step, not as the q^(k-1) exponential-decay term (confirmed by Dr. Chung, Aug
2026). Degree k is capped at MAX_DEGREE=5. Idle T1/T2 decoherence is applied before
fusion. Development uses a 4-node star loaded via RouterNetTopo with
meet_in_the_middle connections.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING

from stim import Circuit

from ...entanglement_management.generation.barret_kok import BarretKokA
from ...kernel.quantum_manager.stabilizer import QuantumManagerStabilizer
from ...message import Message
from ...protocol import Protocol
from ...resource_management.memory_manager import MemoryInfo
from ...topology.node import QuantumRouter
from ...utils import log

if TYPE_CHECKING:
    from ...components.memory import Memory
    from ...kernel.timeline import Timeline


MAX_DEGREE: int = 5
DEFAULT_SUCCESS_BASE: float = 0.9

# Hardware defaults per Dr. Chung, 2026-06-30: perfect hardware to start, with
# detector efficiency 0.95 and coherence times from Table I of the
# cross-validation paper. Table I gives tau in {18, 55, inf} ms and does not
# distinguish T1 from T2, so tau is treated as T2 here.
DEFAULT_DETECTOR_EFFICIENCY: float = 0.95

COHERENCE_TIME_PRESETS_SEC: dict[str, float] = {
    "18ms": 0.018,
    "55ms": 0.055,
    "inf": 1.0e6,  # stands in for tau -> infinity, kept finite for the T1/T2 check
}
DEFAULT_T2_SEC: float = COHERENCE_TIME_PRESETS_SEC["inf"]
DEFAULT_T1_SEC: float = DEFAULT_T2_SEC  # non-limiting relative to T2


class GHZMsgType(Enum):
    """Message types between the helper node and its neighbors.

    Attributes:
        GHZ_RESULT: sent after BSM, carrying Pauli correction bits.
        GENERATION_FAILED: sent when a phase fails, so neighbors release memories.
    """
    GHZ_RESULT = auto()
    GENERATION_FAILED = auto()


class GHZMessage(Message):
    """Message type for GHZ generation, with a payload dict for protocol fields.

    Attributes:
        msg_type (GHZMsgType): type of this message.
        receiver (str): name of the destination protocol instance.
        payload (dict): protocol-specific data; keys vary by msg_type.
    """

    __slots__ = ['msg_type', 'receiver', 'protocol_type', 'payload']

    def __init__(self, msg_type: GHZMsgType, receiver: str, **kwargs):
        """Constructor for GHZMessage.

        Args:
            msg_type (GHZMsgType): type of this message.
            receiver (str): name of the destination protocol instance.
            **kwargs: additional fields stored in payload.
        """
        super().__init__(msg_type, receiver)
        self.payload = kwargs


class GHZEntanglementGenerationA(BarretKokA):
    """Barrett-Kok generation that also notifies GHZGenerationA on success.

    Subclasses BarretKokA rather than EntanglementGenerationA because the latter's
    update_memory() and received_message() are abstract stubs. Only
    _entanglement_succeed is overridden. Coupling to GHZGenerationA confirmed
    acceptable by Dr. Zhan, 2026-06-30.

    Attributes:
        ghz_protocol (GHZGenerationA): protocol to notify on the same node.
        neighbor_name (str): neighbor node this instance pairs with.
    """

    def __init__(self, owner: "GHZNode", name: str, middle: str, other: str,
                 memory: "Memory", ghz_protocol: "GHZGenerationA"):
        """Constructor for GHZEntanglementGenerationA.

        Args:
            owner (GHZNode): helper node this protocol runs on.
            name (str): unique name for this protocol instance.
            middle (str): name of the intermediate BSM node.
            other (str): name of the neighbor node.
            memory (Memory): local memory for this Bell pair leg.
            ghz_protocol (GHZGenerationA): GHZ protocol to notify on success.
        """
        super().__init__(owner, name, middle, other, memory)
        self.ghz_protocol: GHZGenerationA = ghz_protocol
        self.neighbor_name: str = other

    def _entanglement_succeed(self) -> None:
        """Do the usual Bell pair bookkeeping, then notify GHZGenerationA."""
        self.memory.entangled_memory["node_id"] = self.remote_node_name  # type: ignore[index]
        self.memory.entangled_memory["memo_id"] = self.remote_memo_id    # type: ignore[index]
        self.memory.fidelity = self.memory.raw_fidelity                  # type: ignore[assignment]
        self.update_resource_manager(self.memory, MemoryInfo.ENTANGLED)
        self.ghz_protocol.notify_bell_ready(
            neighbor=self.neighbor_name,
            local_key=self.memory.qstate_key,
            neighbor_key=self.memory.entangled_memory.get("memo_id"),
        )


class GHZGenerationA(Protocol):
    """GHZ state generation protocol on the central helper node.

    Coordinates the three-phase cycle: Bell pair accumulation from all k neighbors,
    local GHZ fusion, then BSM with classical correction dispatch. Fusion begins
    automatically once all k neighbors have notified.

    Attributes:
        owner (GHZNode): node this protocol is attached to.
        name (str): unique protocol instance name.
        neighbor_names (list[str]): neighbor node names, length <= MAX_DEGREE.
        success_base (float): single-BSM success probability q, applied per
            neighbor BSM in the hybrid protocol.
        t1_sec (float): T1 decoherence time in seconds.
        t2_sec (float): T2 decoherence time in seconds.
        _ready_neighbors (set[str]): neighbors whose Bell pairs are ready.
        _neighbor_keys (dict[str, int]): neighbor name to its memory qstate_key.
        _local_keys (dict[str, int]): neighbor name to helper's local qstate_key.
        _cycle_count (int): completed or attempted cycles, for logging.
        detector_efficiency (float): photon detector efficiency, 0.95 per Chung.
            Belongs at the heralded generation layer as a photon-loss factor, not
            in the BSM success q (confirmed by Dr. Chung, Aug 2026). Stored here;
            wired into generation once the herald-based path is implemented.

    On failure semantics: fusion failure is a single whole-measurement pass/fail
    draw, not per-qubit independent success. Confirmed by Xinan (2026-07) and
    independently by the fusion-tree protocol in Ghaderibaneh et al. (QCE 2023),
    where a failed fusion restarts link generation entirely rather than leaving a
    reduced usable state.
    """

    def __init__(self, owner: "GHZNode", name: str, neighbor_names: list[str],
                 success_base: float = DEFAULT_SUCCESS_BASE,
                 t1_sec: float = DEFAULT_T1_SEC, t2_sec: float = DEFAULT_T2_SEC,
                 detector_efficiency: float = DEFAULT_DETECTOR_EFFICIENCY):
        """Constructor for GHZGenerationA.

        Args:
            owner (GHZNode): helper node this protocol runs on.
            name (str): unique name for this protocol instance.
            neighbor_names (list[str]): neighbor names, truncated to MAX_DEGREE.
            success_base (float): single-BSM success probability q (default 0.9).
            t1_sec (float): T1 in seconds (default: perfect hardware preset).
            t2_sec (float): T2 in seconds, must satisfy 0 < t2 <= 2*t1.
            detector_efficiency (float): per-detector click probability (0.95).
                Stored only; not yet used in the success calculation.
        """
        super().__init__(owner, name)
        owner.protocols.append(self)

        if len(neighbor_names) > MAX_DEGREE:
            log.logger.warning(
                f"{name}: degree {len(neighbor_names)} exceeds MAX_DEGREE={MAX_DEGREE}. Truncating."
            )
            neighbor_names = neighbor_names[:MAX_DEGREE]

        self.neighbor_names: list[str] = list(neighbor_names)
        self.success_base: float = success_base
        self.t1_sec: float = t1_sec
        self.t2_sec: float = t2_sec
        self.detector_efficiency: float = detector_efficiency
        self._ready_neighbors: set[str] = set()
        self._neighbor_keys: dict[str, int] = {}
        self._local_keys: dict[str, int] = {}
        self._cycle_count: int = 0

    def init(self) -> None:
        """Validate the T1/T2 constraint required by apply_idling_decoherence."""
        if not (0 < self.t2_sec <= 2 * self.t1_sec):
            raise ValueError(
                f"{self.name}: must satisfy 0 < T2 <= 2*T1. "
                f"Got T1={self.t1_sec}, T2={self.t2_sec}."
            )

    def start(self) -> None:
        """Start a new cycle by resetting per-cycle state.

        Bell pair generation itself is driven by the resource manager via
        ghz_rules; this only initialises cycle state at simulation start.
        """
        self._reset_cycle_state()
        self._cycle_count += 1
        log.logger.info(
            f"{self.name}: starting cycle {self._cycle_count} "
            f"with k={len(self.neighbor_names)}."
        )

    def notify_bell_ready(self, neighbor: str, local_key: int, neighbor_key) -> None:
        """Record a ready Bell pair and trigger fusion once all k are ready.

        Args:
            neighbor (str): neighbor node whose Bell pair is ready.
            local_key (int): qstate_key of the helper's memory for this pair.
            neighbor_key: identifier of the neighbor's entangled memory.
        """
        if neighbor not in self.neighbor_names:
            log.logger.warning(
                f"{self.name}: notify_bell_ready from unknown neighbor {neighbor}. Ignoring."
            )
            return
        if neighbor in self._ready_neighbors:
            log.logger.warning(
                f"{self.name}: duplicate notify_bell_ready from {neighbor}. Ignoring."
            )
            return

        self._local_keys[neighbor] = local_key
        self._neighbor_keys[neighbor] = neighbor_key
        self._ready_neighbors.add(neighbor)
        log.logger.info(
            f"{self.name}: {len(self._ready_neighbors)}/{len(self.neighbor_names)} "
            f"Bell pairs ready."
        )

        if len(self._ready_neighbors) == len(self.neighbor_names):
            self._run_ghz_fusion()

    def received_message(self, src: str, msg: Message) -> None:
        """Route incoming GHZ-specific classical messages.

        Args:
            src (str): name of the sending node.
            msg (Message): received message.
        """
        log.logger.warning(
            f"{self.name}: unexpected message type {msg.msg_type} from {src}."
        )

    def _get_stabilizer_manager(self) -> QuantumManagerStabilizer:
        """Return the timeline's quantum manager as a QuantumManagerStabilizer.

        Raises:
            TypeError: if the active quantum manager is a different type.
        """
        qm = self.owner.timeline.quantum_manager
        if not isinstance(qm, QuantumManagerStabilizer):
            raise TypeError(
                f"{self.name}: requires QuantumManagerStabilizer, got {type(qm)}."
            )
        return qm

    def _random(self) -> float:
        """Return a uniform random float from the owner node's generator.

        Isolated as a method so tests can patch it, since numpy's Generator.random
        is a read-only C-extension attribute.

        Returns:
            float: a uniform sample in [0.0, 1.0).
        """
        return self.owner.generator.random()

    def _run_ghz_fusion(self) -> None:
        """Apply decoherence, then run the fusion circuit (deterministic).

        The circuit is CX from qubit 0 into each qubit 1..k-1, then H on qubit 0.
        In the hybrid GHZ-BSM protocol (Chen et al., arXiv:2604.03155, Sec.~3),
        GHZ state generation is deterministic local preparation inside the helper
        node; the probabilistic success q is applied per BSM in _run_bsm_phase,
        not here (confirmed by Dr. Chung, Aug 2026).
        """
        try:
            qm = self._get_stabilizer_manager()
        except TypeError as exc:
            log.logger.error(str(exc))
            self._broadcast_failure()
            return

        ordered = list(self.neighbor_names)
        local_keys = [self._local_keys[n] for n in ordered]

        try:
            qm.apply_idling_decoherence(
                local_keys, self.owner.timeline.now(), self.t1_sec, self.t2_sec
            )
        except Exception as exc:
            log.logger.error(f"{self.name}: decoherence failed: {exc}")
            self._broadcast_failure()
            return

        fusion_circuit = Circuit()
        for target_idx in range(1, len(local_keys)):
            fusion_circuit.append("CX", [0, target_idx])
        fusion_circuit.append("H", [0])

        try:
            qm.run_circuit(fusion_circuit, local_keys, inject_gate_error=True)
        except Exception as exc:
            log.logger.error(f"{self.name}: fusion circuit failed: {exc}")
            self._broadcast_failure()
            return

        self._run_bsm_phase(ordered, local_keys)

    def _run_bsm_phase(self, ordered_neighbors: list[str], local_keys: list[int]) -> None:
        """Run BSM on each (GHZ qubit, neighbor qubit) pair and send corrections.

        Per neighbor: a single-q probabilistic success check, then CX(local->
        neighbor), H(local), M(local), M(neighbor). Outcomes (m0, m1) give the Z
        and X corrections sent via GHZ_RESULT.

        Per the hybrid GHZ-BSM protocol (Chen et al., arXiv:2604.03155, Sec.~3),
        each helper BSM succeeds with a single probability q (success_base), not
        the q^(k-1) exponential-decay term (confirmed by Dr. Chung, Aug 2026).
        A failed BSM fails the whole cycle via _broadcast_failure; treating one
        neighbor's failure as a whole-cycle failure is a simplifying assumption
        for this first implementation rather than a per-link partial success.

        Args:
            ordered_neighbors (list[str]): neighbor names, same order as local_keys.
            local_keys (list[int]): helper's qstate_keys, one per neighbor.
        """
        try:
            qm = self._get_stabilizer_manager()
        except TypeError as exc:
            log.logger.error(str(exc))
            self._broadcast_failure()
            return

        bsm_circuit = Circuit()
        bsm_circuit.append("CX", [0, 1])
        bsm_circuit.append("H", [0])
        bsm_circuit.append("M", [0])
        bsm_circuit.append("M", [1])

        for i, neighbor in enumerate(ordered_neighbors):
            local_key = local_keys[i]
            neighbor_key = self._neighbor_keys[neighbor]

            if self._random() >= self.success_base:
                log.logger.info(
                    f"{self.name}: BSM failed probabilistic check for {neighbor} "
                    f"(q={self.success_base:.4f}). Broadcasting failure."
                )
                self._broadcast_failure()
                return

            try:
                results = qm.run_circuit(
                    bsm_circuit, [local_key, neighbor_key], inject_gate_error=True
                )
            except Exception as exc:
                log.logger.error(f"{self.name}: BSM failed for {neighbor}: {exc}")
                self._broadcast_failure()
                return

            m0 = results.get(local_key, 0)
            m1 = results.get(neighbor_key, 0)

            msg = GHZMessage(
                GHZMsgType.GHZ_RESULT,
                receiver=neighbor + ".GHZGenerationA",
                x_correction=m1,
                z_correction=m0,
                cycle=self._cycle_count,
            )
            self.owner.send_message(neighbor, msg)
            log.logger.info(
                f"{self.name}: GHZ_RESULT sent to {neighbor} (x={m1}, z={m0})."
            )

        log.logger.info(f"{self.name}: cycle {self._cycle_count} complete.")

    def _broadcast_failure(self) -> None:
        """Send GENERATION_FAILED to all neighbors and reset cycle state."""
        log.logger.warning(
            f"{self.name}: broadcasting failure for cycle {self._cycle_count}."
        )
        for neighbor in self.neighbor_names:
            msg = GHZMessage(
                GHZMsgType.GENERATION_FAILED,
                receiver=neighbor + ".GHZGenerationA",
                cycle=self._cycle_count,
            )
            self.owner.send_message(neighbor, msg)
        self._reset_cycle_state()

    def _reset_cycle_state(self) -> None:
        """Clear per-cycle tracking state."""
        self._ready_neighbors = set()
        self._neighbor_keys = {}
        self._local_keys = {}


class GHZNode(QuantumRouter):
    """Central helper node for fusion-based GHZ state generation.

    Subclasses QuantumRouter to inherit the ResourceManager, NetworkManager,
    MemoryArray, and channel infrastructure. The MemoryArray is sized to the
    neighbor count, one slot per neighbor.

    Attributes:
        neighbor_names (list[str]): adjacent neighbor node names.
        ghz_protocol (GHZGenerationA): the GHZ generation protocol instance.
        _neighbor_to_memory_idx (dict[str, int]): neighbor to MemoryArray index.
    """

    def __init__(self, name: str, tl: "Timeline", neighbor_names: list[str],
                 seed: int | None = None, success_base: float = DEFAULT_SUCCESS_BASE,
                 t1_sec: float = DEFAULT_T1_SEC, t2_sec: float = DEFAULT_T2_SEC,
                 detector_efficiency: float = DEFAULT_DETECTOR_EFFICIENCY,
                 component_templates: dict | None = None):
        """Constructor for GHZNode.

        Args:
            name (str): name of the node instance.
            tl (Timeline): simulation timeline.
            neighbor_names (list[str]): neighbor names, truncated to MAX_DEGREE.
            seed (int | None): seed for the node's random number generator.
            success_base (float): single-BSM success probability q (default 0.9).
            t1_sec (float): T1 in seconds (default: perfect hardware preset).
            t2_sec (float): T2 in seconds (default: same preset).
            detector_efficiency (float): per-detector click probability (0.95).
            component_templates (dict | None): optional hardware overrides passed
                to QuantumRouter.
        """
        if len(neighbor_names) > MAX_DEGREE:
            log.logger.warning(
                f"GHZNode {name}: truncating to MAX_DEGREE={MAX_DEGREE}."
            )
            neighbor_names = neighbor_names[:MAX_DEGREE]

        self.neighbor_names: list[str] = list(neighbor_names)
        k = len(self.neighbor_names)

        super().__init__(
            name, tl,
            memo_size=k,
            seed=seed,
            component_templates=component_templates or {},
        )

        self._neighbor_to_memory_idx: dict[str, int] = {
            neighbor: idx for idx, neighbor in enumerate(self.neighbor_names)
        }

        self.ghz_protocol = GHZGenerationA(
            owner=self,
            name=f"{name}.GHZGenerationA",
            neighbor_names=self.neighbor_names,
            success_base=success_base,
            t1_sec=t1_sec,
            t2_sec=t2_sec,
            detector_efficiency=detector_efficiency,
        )

    def get_memory_for_neighbor(self, neighbor: str):
        """Return the Memory assigned to the given neighbor.

        Args:
            neighbor (str): name of the neighbor node.

        Returns:
            Memory | None: assigned Memory, or None if the neighbor is unknown.
        """
        idx = self._neighbor_to_memory_idx.get(neighbor)
        if idx is None:
            log.logger.warning(
                f"GHZNode {self.name}: no memory slot for neighbor {neighbor}."
            )
            return None
        return self.get_components_by_type("MemoryArray")[0][idx]