"""A terminal summary of the statistics of a clean or a dirty dataset.

The statistics themselves are computed by ``datagen.generator.stats`` (clean data) and
``datagen.injector.stats`` (errors and dirty data).
"""
from __future__ import annotations


def summary(stats) -> str:
    """A few lines for the terminal."""
    rows = ", ".join(f"{t}: {n:,}" for t, n in stats["rows"].items())
    cpp = stats["children_per_parent"]
    violations = stats["violations"]
    lines = [
        f"rows                 {rows}",
        f"children per parent  mean {cpp['mean']:.2f}, min {cpp['min']}, max {cpp['max']}, "
        f"0: {cpp['share_0']:.1%}, 1: {cpp['share_1']:.1%}, 2+: {cpp['share_2_or_more']:.1%}",
    ]
    if "errors" in stats:
        err = stats["errors"]
        lines += [
            f"violations           clean {violations['clean']}, dirty {violations['dirty']}",
            f"errors               {err['errors']:,} of {err['cells']:,} eligible cells ({err['rate']:.2%})",
        ]
        for table, r in err["by_table"].items():
            lines.append(f"  {table:<19}{r['errors']:,} of {r['cells']:,} ({r['rate']:.2%})")
        lines.append(f"  detectable         {err['detectable']:,} ({err['detectable_share']:.1%})")
        if err["shortfalls"]:
            lines.append(f"  shortfalls         {len(err['shortfalls'])} scopes, e.g. {err['shortfalls'][0]}")
    else:
        lines.append(f"violations           clean {violations['clean']} (no errors injected)")
    worst = max(stats["marginals"].items(), key=lambda kv: kv[1]["tvd"])
    lines.append(f"largest marginal gap {worst[0]} (TVD {worst[1]['tvd']:.3f} from the target)")
    return "\n".join(lines)
