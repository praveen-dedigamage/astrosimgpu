#!/usr/bin/env python3
"""Dump the NEST defaults this simulator was written against.

Run this in an environment that has NEST installed and diff the output against
config/use_case.json. Any difference is a parameter that has changed upstream
since the C++ defaults were fixed, and is worth reconciling before trusting a
quantitative comparison between the two simulators.

    python scripts/dump_nest_defaults.py > nest_defaults.json
"""

import json
import sys

MODELS = ["astrocyte_lr_1994", "aeif_cond_alpha_astro", "tsodyks_synapse"]

# Values NEST reports that are bookkeeping rather than model parameters.
SKIP = {
    "archiver_length", "available", "beta_Ca", "capacity", "element_type",
    "elementsize", "frozen", "global_id", "instantiations", "local",
    "model", "node_uses_wfr", "recordables", "requires_symmetric",
    "sizeof", "supports_precise_spikes", "synapse_model", "synapse_modelid",
    "tau_Ca", "thread", "thread_local_id", "type_id", "vp",
    "has_delay", "num_connections", "receptor_type", "property_object",
}


def jsonable(value):
    """Reduce a NEST parameter to something json can serialise."""
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    return str(value)


def main():
    try:
        import nest
    except ImportError:
        sys.exit(
            "NEST is not importable in this environment.\n"
            "This script is only useful where NEST is installed; it exists so\n"
            "the C++ defaults can be checked against the reference simulator."
        )

    nest.set_verbosity("M_ERROR")
    out = {"nest_version": nest.__version__}

    for model in MODELS:
        try:
            defaults = nest.GetDefaults(model)
        except Exception as exc:  # noqa: BLE001 - report and continue
            out[model] = {"error": str(exc)}
            continue
        out[model] = {
            key: jsonable(value)
            for key, value in sorted(defaults.items())
            if key not in SKIP and not key.startswith("t_spike")
        }

    json.dump(out, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
