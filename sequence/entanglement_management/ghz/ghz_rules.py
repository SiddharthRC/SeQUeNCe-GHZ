"""Rule conditions and actions for GHZ Bell pair generation.

Provides the condition and action functions a GHZNode's ResourceManager uses to
install GHZEntanglementGenerationA instances, one per neighbor, following the
pattern in SeQUeNCe's Chapter 4 tutorial. The standard eg_rule pattern is adapted
to install GHZEntanglementGenerationA, which notifies GHZGenerationA on success,
rather than the standard EntanglementGenerationA.

Usage after loading topology via RouterNetTopo:
    from sequence.entanglement_management.ghz.ghz_rules import install_ghz_eg_rules
    install_ghz_eg_rules(ghz_node, middle_node_map)
    tl.init()
    tl.run()
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from ..generation.generation_base import EntanglementGenerationA
from ...resource_management.memory_manager import MemoryInfo
from ...resource_management.rule_manager import Rule
from ...topology.node import QuantumRouter
from .ghz_generation import GHZEntanglementGenerationA

if TYPE_CHECKING:
    from ...resource_management.memory_manager import MemoryManager
    from .ghz_generation import GHZNode


def ghz_eg_rule_condition(
    memory_info: "MemoryInfo",
    manager: "MemoryManager",
    args: dict,
) -> list:
    """Select the memory slot assigned to a specific neighbor.

    Matches when the memory is RAW and its index is the slot reserved for the
    target neighbor.

    Args:
        memory_info (MemoryInfo): info for one memory slot.
        manager (MemoryManager): memory manager on the GHZNode.
        args (dict): must contain 'memory_index', the reserved MemoryArray index.

    Returns:
        list: [memory_info] if matched, else [].
    """
    if (memory_info.state == MemoryInfo.RAW
            and memory_info.index == args["memory_index"]):
        return [memory_info]
    return []


def ghz_eg_rule_action_request(
    memories_info: list["MemoryInfo"],
    args: dict,
) -> list:
    """Action for the GHZNode side of Bell pair generation.

    Creates a GHZEntanglementGenerationA for one helper-neighbor pair and sends
    a pairing request to the neighbor's resource manager.

    Args:
        memories_info (list[MemoryInfo]): single-element list from the condition.
        args (dict): must contain 'mid_name', 'other_name', 'ghz_protocol',
            'node_name', and 'memory_index'.

    Returns:
        list: [protocol, [other_name], [ghz_eg_match_func], [req_args]].
    """
    mid_name = args["mid_name"]
    other_name = args["other_name"]
    ghz_protocol = args["ghz_protocol"]
    memory = memories_info[0].memory

    protocol = GHZEntanglementGenerationA(
        owner=None,  # type: ignore[arg-type]
        name=f"GHZ_EGA.{memory.name}",
        middle=mid_name,
        other=other_name,
        memory=memory,
        ghz_protocol=ghz_protocol,
    )

    req_args = {
        "remote_node": args["node_name"],
        "memory_index": args["memory_index"],
    }
    return [protocol, [other_name], [ghz_eg_match_func], [req_args]]


def ghz_eg_rule_action_await(
    memories_info: list["MemoryInfo"],
    args: dict,
) -> list:
    """Action for the neighbor side of Bell pair generation.

    Creates a standard EntanglementGenerationA on the neighbor and waits for a
    pairing request from the GHZNode.

    Args:
        memories_info (list[MemoryInfo]): single-element list from the condition.
        args (dict): must contain 'mid_name' and 'other_name'.

    Returns:
        list: [protocol, [None], [None], [None]].
    """
    mid_name = args["mid_name"]
    other_name = args["other_name"]
    memory = memories_info[0].memory

    protocol = EntanglementGenerationA.create(
        owner=None,  # type: ignore[arg-type]
        name=f"EGA.{memory.name}",
        middle=mid_name,
        other=other_name,
        memory=memory,
    )
    return [protocol, [None], [None], [None]]


def ghz_eg_match_func(protocols: list, args: dict):
    """Pair the neighbor's EntanglementGenerationA with the GHZNode.

    Called by the remote resource manager to find an eligible protocol.

    Args:
        protocols (list): protocols on the neighbor node.
        args (dict): must contain 'remote_node' and 'memory_index'.

    Returns:
        EntanglementGenerationA | None: the matching protocol, or None.
    """
    remote_node = args["remote_node"]
    memory_index = args["memory_index"]

    for protocol in protocols:
        if not isinstance(protocol, EntanglementGenerationA):
            continue
        mem_arr = protocol.owner.get_components_by_type("MemoryArray")[0]
        if (protocol.remote_node_name == remote_node
                and mem_arr.memories.index(protocol.memory) == memory_index):
            return protocol
    return None


def install_ghz_eg_rules(ghz_node: "GHZNode", middle_node_map: dict[str, str]) -> None:
    """Install Bell pair generation rules on the GHZNode and each neighbor.

    Per neighbor: a request rule on the GHZNode creating a
    GHZEntanglementGenerationA, and an await rule on the neighbor creating a
    standard EntanglementGenerationA. Both use ghz_eg_rule_condition.

    The neighbor list is read from ghz_node.ghz_protocol rather than from
    ghz_node directly, so this works with any node carrying a ghz_protocol
    attribute, including RouterNetTopo-loaded QuantumRouters in tests that
    attach GHZGenerationA manually.

    Args:
        ghz_node (GHZNode): the central helper node.
        middle_node_map (dict[str, str]): neighbor name to the intermediate BSM
            node name, e.g. {"n1": "bsm_helper_n1", ...}.

    Side Effects:
        Loads rules into ghz_node.resource_manager and each neighbor's.
    """
    tl = ghz_node.timeline
    ghz_protocol = ghz_node.ghz_protocol

    for idx, neighbor_name in enumerate(ghz_protocol.neighbor_names):
        mid_name = middle_node_map[neighbor_name]
        neighbor_node = cast(QuantumRouter, tl.get_entity_by_name(neighbor_name))

        # Request side, on the GHZNode.
        action_args_request = {
            "mid_name": mid_name,
            "other_name": neighbor_name,
            "ghz_protocol": ghz_protocol,
            "node_name": ghz_node.name,
            "memory_index": 0,
        }
        condition_args_helper = {"memory_index": idx}

        rule_request = Rule(
            priority=10,
            action=ghz_eg_rule_action_request,  # type: ignore[arg-type]
            condition=ghz_eg_rule_condition,
            action_args=action_args_request,
            condition_args=condition_args_helper,
        )
        ghz_node.resource_manager.load(rule_request)

        # Await side, on the neighbor. Memory index 0 by convention.
        neighbor_memory_index = 0
        action_args_await = {
            "mid_name": mid_name,
            "other_name": ghz_node.name,
        }
        condition_args_neighbor = {"memory_index": neighbor_memory_index}

        rule_await = Rule(
            priority=10,
            action=ghz_eg_rule_action_await,  # type: ignore[arg-type]
            condition=ghz_eg_rule_condition,
            action_args=action_args_await,
            condition_args=condition_args_neighbor,
        )
        neighbor_node.resource_manager.load(rule_await)