import pdb
import yaml
from pathlib import Path
import numpy as np
import re
import json
from transforms3d.quaternions import qmult, axangle2quat
from transforms3d.euler import euler2quat


class Rotation:
    def __init__(self, quat):
        self.quat = quat

    def __mul__(self, other):
        assert isinstance(other, Rotation)
        return Rotation(qmult(self.quat, other.quat))

    def __rmul__(self, other):
        assert isinstance(other, Rotation)
        return Rotation(qmult(other.quat, self.quat))

    def to_quat(self):
        return self.quat


def quat(q):
    assert len(q) == 4
    q = np.array(q)
    q = q / np.linalg.norm(q)
    return Rotation(q)


def euler(xyz):
    assert len(xyz) == 3
    return Rotation(euler2quat(*xyz))


def angleAxis(angle, axis):
    assert len(axis) == 3
    return Rotation(axangle2quat(axis, angle))


def random_angle_axis(angle_low, angle_high, axis, rng: np.random.RandomState):
    return angleAxis(rng.uniform(angle_low, angle_high), axis)


def iter_config_dict(config, func):
    if isinstance(config, list):
        for elem in config:
            iter_config_dict(elem, func)
    elif isinstance(config, dict):
        func(config)
        for k, v in config.items():
            iter_config_dict(v, func)
    else:
        pass


def preprocess(filename):
    path = Path(filename).resolve()
    with path.open("r") as f:
        raw_yaml = yaml.load(f, Loader=yaml.SafeLoader)

    # normalize the name of files
    def resolve_files(config):
        for key in config:
            if key == "file" or key.endswith("_file"):
                assert isinstance(config[key], str)
                config[key] = str(path.parent.joinpath(config[key]).resolve())

    iter_config_dict(raw_yaml, resolve_files)

    # load includes
    includes = []

    def find_include(d):
        if "_include" in d:
            includes.append(d)

    iter_config_dict(raw_yaml, find_include)

    for config in includes[::-1]:
        loaded_yaml = preprocess(
            str(path.parent.joinpath(config["_include"]).resolve())
        )
        assert isinstance(loaded_yaml, dict)
        for key in loaded_yaml:
            assert key not in config
            config[key] = loaded_yaml[key]
        del config["_include"]
        if "_override" in config:
            for key in config["_override"]:
                assert key in loaded_yaml
                config[key] = config["_override"][key]
            del config["_override"]

    return raw_yaml


def parse_exp(exp: str):
    segs = []
    pattern = re.compile("(\\$[a-zA-Z_][a-zA-Z_0-9]*)")
    start = 0
    while True:
        result = pattern.search(exp, pos=start)
        if result is None:
            segs.append(exp[start:])
            break
        first, last = result.span()
        segs.append(exp[start:first])
        segs.append(exp[first:last])
        start = last
    segs = [s for s in segs if s]
    return segs


def eval_expression(node: str, rng: np.random.RandomState, scope: dict):
    # now eval("true") would be valid
    true = True
    false = False

    def Uniform(low, high):
        return rng.uniform(low, high)

    def RandomAngleAxis(angle_low, angle_high, axis):
        return random_angle_axis(angle_low, angle_high, axis, rng)

    if "$" in node and not (node.startswith("eval(") and node.endswith(")")):
        exp = node
    elif node.startswith("eval(") and node.endswith(")"):
        exp = node[5:-1]
    elif node.startswith("Uniform") or node.startswith("RandomAngleAxis"):
        exp = node
    else:
        return node
    exps = parse_exp(exp)
    if len(exps) == 1:
        if exps[0][0] == "$":
            # single variable exp
            return scope[exp]
        # no variable exp
        return eval(exp)
    # multi-variable exp
    new_exps = []
    for term in exps:
        if term[0] == "$":
            term = json.dumps(scope[term[0]])
        new_exps.append(term)
    exp = "".join(new_exps)
    return eval(exp)


def eval_scoped_variables(node, rng: np.random.RandomState, scope):
    if isinstance(node, dict):
        local_scope = {}
        for key in list(node.keys()):
            if key.startswith("$"):
                local_scope[key] = eval_scoped_variables(node[key], rng, scope)
                del node[key]
        # update local_scope
        for key in scope:
            if key not in local_scope:
                local_scope[key] = scope[key]
        for key in node:
            node[key] = eval_scoped_variables(node[key], rng, local_scope)
        return node
    if isinstance(node, list):
        for i in range(len(node)):
            node[i] = eval_scoped_variables(node[i], rng, scope)
        return node
    if isinstance(node, str):
        return eval_expression(node, rng, scope)
    return node


def process_variables(config, rng: np.random.RandomState):
    return eval_scoped_variables(config, rng, {})


def resolve_variants(node, rng: np.random.RandomState, variant_config, output_config):
    if isinstance(node, dict):
        if "_variants" in node:
            var = node["_variants"]
            global_id = var["global_id"]
            if var["type"] == "options":
                option_dict = var["options"]
                keys = list(option_dict.keys())
                if global_id in variant_config:
                    idx = keys.index(variant_config[global_id])
                else:
                    idx = rng.choice(len(keys))
                key = keys[idx]
                resolved_value = option_dict[key]
                output_config[global_id] = {"type": "options", "key": key, "index": idx}
            else:
                raise NotImplementedError
            if isinstance(resolved_value, dict):
                node.update(resolved_value)
                del node["_variants"]
            else:
                assert (
                    len(node) == 1
                ), "parent node of scalar variant should not have other keys"
                return resolved_value
        for key in node:
            node[key] = resolve_variants(node[key], rng, variant_config, output_config)
        return node
    if isinstance(node, list):
        for i in range(len(node)):
            node[i] = resolve_variants(node[i], rng, variant_config, output_config)
        return node
    return node


def process_variants(node, rng: np.random.RandomState, variant_config):
    output_config = {}
    node = resolve_variants(node, rng, variant_config, output_config)
    return node, output_config


def test():
    rng = np.random.RandomState(0)
    yaml1 = preprocess("../assets/config_files/pick_floating-v1.yml")
    yaml2 = process_variables(yaml1, rng)
    yaml3, config = process_variants(yaml2, rng, {})
    print(yaml3, config)
