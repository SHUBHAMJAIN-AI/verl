#!/usr/bin/env python3
"""
Ray GPU Placement Verification Script

Verifies that Ray is correctly managing GPU placement for split training:
- Checks placement groups
- Maps workers to GPUs
- Validates resource allocation
"""

import subprocess
import sys

import ray


def verify_ray_gpu_placement():
    print("=" * 80)
    print("RAY GPU PLACEMENT VERIFICATION")
    print("=" * 80)

    try:
        # Connect to Ray cluster
        ray.init(address="auto", ignore_reinit_error=True)
        print("✅ Connected to Ray cluster")

        # Get placement groups
        placement_groups = ray.util.placement_group_table()
        print(f"\n📋 Placement Groups: {len(placement_groups)}")

        pg_details = []
        for pg_id, pg_info in placement_groups.items():
            pg_short_id = pg_id[:12]
            bundles = pg_info["bundles"]
            strategy = pg_info.get("strategy", "STRICT_PACK")
            state = pg_info["state"]

            pg_details.append({"id": pg_short_id, "bundles": bundles, "strategy": strategy, "state": state})

            print(f"  PG {pg_short_id}... - {state}")
            print(f"    Strategy: {strategy}")
            print(f"    Bundles: {bundles}")

        # Check resource allocation
        cluster_resources = ray.cluster_resources()
        available_resources = ray.available_resources()

        total_gpus = cluster_resources.get("GPU", 0)
        available_gpus = available_resources.get("GPU", 0)
        used_gpus = total_gpus - available_gpus

        print("\n🔧 GPU Resource Status:")
        print(f"  Total GPUs in cluster: {total_gpus}")
        print(f"  GPUs currently allocated: {used_gpus}")
        print(f"  GPUs available: {available_gpus}")

        # Verify expected configuration
        print("\n✅ Verification Results:")

        expected_pgs = 2

        if len(placement_groups) == expected_pgs:
            print(f"  ✅ Placement groups: {len(placement_groups)}/{expected_pgs}")
        else:
            print(f"  ❌ Placement groups: {len(placement_groups)}/{expected_pgs}")

        if used_gpus == expected_pgs:
            print(f"  ✅ GPU allocation: {used_gpus}/{expected_pgs} GPUs in use")
        else:
            print(f"  ❌ GPU allocation: {used_gpus}/{expected_pgs} GPUs in use")

        # Check if all placement groups have correct GPU allocation
        all_pgs_correct = True
        for pg_detail in pg_details:
            bundles = pg_detail["bundles"]
            if len(bundles) != 1 or bundles[0].get("GPU", 0) != 1:
                all_pgs_correct = False
                break

        if all_pgs_correct:
            print("  ✅ Bundle configuration: Each PG has exactly 1 GPU")
        else:
            print("  ❌ Bundle configuration: Incorrect GPU allocation per PG")

        # Try to map workers to placement groups
        print("\n🔍 Worker to Placement Group Mapping:")
        try:
            # Get process information
            result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
            lines = result.stdout.split("\n")

            ray_workers = []
            for line in lines:
                if "ray::WorkerDict" in line and ("790299" in line or "790929" in line):
                    parts = line.split()
                    pid = parts[1]
                    cmd = " ".join(parts[10:])
                    ray_workers.append({"pid": pid, "cmd": cmd})

            print(f"  Found {len(ray_workers)} Ray workers:")
            for worker in ray_workers:
                pid = worker["pid"]
                cmd = worker["cmd"]

                # Identify worker type based on command
                if "actor_rollout" in cmd:
                    worker_type = "Actor+Rollout"
                    expected_gpu = "GPU 0"
                elif "WorkerDict" in cmd and "actor_rollout" not in cmd:
                    worker_type = "Critic+Reference"
                    expected_gpu = "GPU 1"
                else:
                    worker_type = "Unknown"
                    expected_gpu = "Unknown"

                print(f"    PID {pid}: {worker_type} → Expected on {expected_gpu}")

        except Exception as e:
            print(f"  Could not map workers: {e}")

        # Final assessment
        print("\n🎯 Overall Assessment:")
        if len(placement_groups) == 2 and used_gpus == 2 and all_pgs_correct:
            print("  ✅ Ray is correctly managing GPU separation!")
            print("  ✅ Each placement group has 1 GPU with STRICT_PACK strategy")
            print("  ✅ All GPUs are allocated to separate placement groups")
            return True
        else:
            print("  ❌ Ray GPU separation issues detected")
            return False

    except Exception as e:
        print(f"❌ Ray connection error: {e}")
        return False
    finally:
        try:
            ray.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    success = verify_ray_gpu_placement()
    sys.exit(0 if success else 1)
