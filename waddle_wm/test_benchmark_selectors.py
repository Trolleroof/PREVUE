"""Offline check for honest pool-prefix coverage reporting."""

from waddle_wm.benchmark_selectors import prefix_coverage
from waddle_wm.pools import PREFIXES


def main():
    pools = {
        "wider": {"pool_id": "wider", "scene": {"observation_id": "obs-wider"},
                  "candidates": [{}] * 40},
        "short": {"pool_id": "short", "scene": {"observation_id": "obs-short"},
                  "candidates": [{}] * 20},
    }
    scenes = [{"pool_id": pool_id, "prefix": size}
              for pool_id, pool in pools.items()
              for size in PREFIXES if size <= len(pool["candidates"])]
    coverage = prefix_coverage({"scenes": scenes}, pools)

    assert list(coverage["by_prefix"]) == [str(size) for size in PREFIXES]
    assert coverage["by_prefix"]["32"]["pools"] == 1
    assert coverage["by_prefix"]["64"]["pools"] == 0
    assert len(coverage["by_prefix"]["64"]["short_of_prefix"]) == 2
    assert coverage["balanced"] is False
    print("prefix coverage ok")


if __name__ == "__main__":
    main()
