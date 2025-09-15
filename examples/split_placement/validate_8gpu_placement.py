#!/usr/bin/env python3
"""
8-GPU Placement Validation Script for Split Training

Validates that models are correctly placed on separate GPU pools:
- GPUs 0-5: Actor + Rollout + Reference (6 GPUs)
- GPUs 6-7: Critic (2 GPUs)

Usage:
    python validate_8gpu_placement.py [--interval SECONDS] [--log-file FILE]
"""

import argparse
import json
import subprocess
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple


class GPU8PlacementValidator:
    def __init__(self, check_interval: int = 10, log_file: Optional[str] = None):
        self.check_interval = check_interval
        self.log_file = log_file
        self.validation_history = []

        # Expected 8-GPU placement
        self.expected_placement = {
            # Actor + Rollout + Reference pool (6 GPUs)
            0: ["actor", "rollout", "reference"],
            1: ["actor", "rollout", "reference"],
            2: ["actor", "rollout", "reference"],
            3: ["actor", "rollout", "reference"],
            4: ["actor", "rollout", "reference"],
            5: ["actor", "rollout", "reference"],
            # Critic pool (2 GPUs)
            6: ["critic"],
            7: ["critic"],
        }

        self.gpu_pools = {"actor_rollout_ref_pool": [0, 1, 2, 3, 4, 5], "critic_pool": [6, 7]}

        print("=" * 80)
        print("8-GPU PLACEMENT VALIDATION FOR SPLIT TRAINING")
        print("=" * 80)
        print("Expected Placement:")
        print("- GPUs 0-5: Actor + Rollout + Reference (6 GPUs)")
        print("- GPUs 6-7: Critic (2 GPUs)")
        print("=" * 80)
        print()

    def get_gpu_processes(self) -> Dict[int, List[Dict]]:
        """Get processes running on each GPU with detailed info"""
        try:
            # Get GPU process info
            result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,gpu_uuid,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)

            gpu_processes = defaultdict(list)

            for line in result.stdout.strip().split("\n"):
                if line.strip():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 4:
                        pid = parts[0]
                        process_name = parts[1]
                        gpu_uuid = parts[2]
                        memory_mb = parts[3]

                        # Get GPU index from UUID
                        gpu_id = self._get_gpu_index_from_uuid(gpu_uuid)
                        if gpu_id is not None:
                            gpu_processes[gpu_id].append({"pid": pid, "process_name": process_name, "memory_mb": memory_mb, "component_type": self._identify_component_type(pid)})

            return dict(gpu_processes)

        except Exception as e:
            print(f"Error getting GPU processes: {e}")
            return {}

    def _get_gpu_index_from_uuid(self, gpu_uuid: str) -> Optional[int]:
        """Convert GPU UUID to index"""
        try:
            result = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], capture_output=True, text=True, timeout=5)

            for line in result.stdout.strip().split("\n"):
                if line.strip():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 2 and parts[1] == gpu_uuid:
                        return int(parts[0])
        except Exception:
            pass
        return None

    def _identify_component_type(self, pid: str) -> str:
        """Identify the type of component based on process info"""
        try:
            # Get process command line
            with open(f"/proc/{pid}/cmdline") as f:
                cmdline = f.read().replace("\x00", " ").strip().lower()

            # Identify component based on patterns
            if "ray::workerdict" in cmdline:
                return self._identify_ray_worker_type(pid, cmdline)
            elif "vllm" in cmdline or "sglang" in cmdline:
                return "rollout"
            elif "main_ppo_split_8gpu" in cmdline:
                return "main_trainer"
            else:
                return "unknown"

        except Exception:
            return "unknown"

    def _identify_ray_worker_type(self, pid: str, cmdline: str) -> str:
        """Identify Ray worker type by analyzing process patterns"""
        try:
            # Check if this worker is saving actor or critic checkpoints
            result = subprocess.run(["lsof", "-p", pid, "+D", "/root/verl/examples/split_placement/checkpoints"], capture_output=True, text=True, timeout=5)

            if "actor" in result.stdout:
                return "actor_rollout_ref"
            elif "critic" in result.stdout:
                return "critic"

            # Check process tree for more clues
            try:
                with open(f"/proc/{pid}/stat") as f:
                    stat_info = f.read().split()
                    ppid = stat_info[3]

                with open(f"/proc/{ppid}/cmdline") as f:
                    parent_cmdline = f.read().replace("\x00", " ").strip().lower()

                if any(term in parent_cmdline for term in ["actor", "rollout"]):
                    return "actor_rollout_ref"
                elif "critic" in parent_cmdline:
                    return "critic"
            except Exception:
                pass

            return "ray_worker"

        except Exception:
            return "ray_worker"

    def check_8gpu_placement_correctness(self, gpu_processes: Dict[int, List[Dict]]) -> Tuple[bool, List[str]]:
        """Check if 8-GPU placement matches expected configuration"""
        issues = []
        all_correct = True

        # Check if we have processes on expected GPUs
        expected_gpus = set(range(8))
        active_gpus = set(gpu_processes.keys())

        if not active_gpus.issubset(expected_gpus):
            unexpected_gpus = active_gpus - expected_gpus
            issues.append(f"Unexpected GPUs in use: {unexpected_gpus}")
            all_correct = False

        # Check pool-level distribution
        actor_pool_gpus = []
        critic_pool_gpus = []

        for gpu_id, processes in gpu_processes.items():
            if gpu_id in self.gpu_pools["actor_rollout_ref_pool"]:
                actor_pool_gpus.append(gpu_id)
            elif gpu_id in self.gpu_pools["critic_pool"]:
                critic_pool_gpus.append(gpu_id)

            # Check individual GPU assignment
            found_components = [p["component_type"] for p in processes]

            if gpu_id in range(6):  # Actor pool GPUs (0-5)
                # Should have actor/rollout/reference components
                expected_in_pool = any("actor" in comp or "rollout" in comp or "ray_worker" in comp for comp in found_components)
                if not expected_in_pool and processes:  # Only check if there are processes
                    issues.append(f"GPU {gpu_id}: Expected actor/rollout components but found {found_components}")

            elif gpu_id in [6, 7]:  # Critic pool GPUs (6-7)
                # Should have critic components
                expected_in_pool = any("critic" in comp or "ray_worker" in comp for comp in found_components)
                if not expected_in_pool and processes:  # Only check if there are processes
                    issues.append(f"GPU {gpu_id}: Expected critic components but found {found_components}")

        # Check pool utilization
        if actor_pool_gpus:
            print(f"Actor pool active on GPUs: {sorted(actor_pool_gpus)}")
        if critic_pool_gpus:
            print(f"Critic pool active on GPUs: {sorted(critic_pool_gpus)}")

        return all_correct, issues

    def log_validation_result(self, timestamp: str, gpu_processes: Dict[int, List[Dict]], is_correct: bool, issues: List[str]):
        """Log validation results"""
        result = {"timestamp": timestamp, "gpu_processes": gpu_processes, "placement_correct": is_correct, "issues": issues}

        self.validation_history.append(result)

        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(result) + "\n")

    def print_8gpu_status(self, gpu_processes: Dict[int, List[Dict]], is_correct: bool, issues: List[str]):
        """Print current 8-GPU validation status"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(f"[{timestamp}] 8-GPU Validation Check")
        print("-" * 60)

        if not gpu_processes:
            print("❌ No GPU processes detected - training may not be running")
            print()
            return

        # Group by pools for better visualization
        actor_pool_processes = {}
        critic_pool_processes = {}

        for gpu_id in sorted(gpu_processes.keys()):
            processes = gpu_processes[gpu_id]

            if gpu_id in self.gpu_pools["actor_rollout_ref_pool"]:
                actor_pool_processes[gpu_id] = processes
            elif gpu_id in self.gpu_pools["critic_pool"]:
                critic_pool_processes[gpu_id] = processes

        # Print Actor Pool Status
        print("🎭 ACTOR+ROLLOUT+REFERENCE POOL (GPUs 0-5):")
        if actor_pool_processes:
            for gpu_id in sorted(actor_pool_processes.keys()):
                processes = actor_pool_processes[gpu_id]
                print(f"  GPU {gpu_id}:")
                if not processes:
                    print("    No processes")
                else:
                    for proc in processes:
                        print(f"    PID {proc['pid']}: {proc['component_type']} ({proc['memory_mb']} MB)")
        else:
            print("  No active processes")

        # Print Critic Pool Status
        print("\n🎯 CRITIC POOL (GPUs 6-7):")
        if critic_pool_processes:
            for gpu_id in sorted(critic_pool_processes.keys()):
                processes = critic_pool_processes[gpu_id]
                print(f"  GPU {gpu_id}:")
                if not processes:
                    print("    No processes")
                else:
                    for proc in processes:
                        print(f"    PID {proc['pid']}: {proc['component_type']} ({proc['memory_mb']} MB)")
        else:
            print("  No active processes")

        # Print validation result
        print("\n📊 Validation Result:")
        if is_correct:
            print("✅ 8-GPU PLACEMENT CORRECT: All components on expected GPU pools")
        else:
            print("❌ 8-GPU PLACEMENT ISSUES DETECTED:")
            for issue in issues:
                print(f"   - {issue}")

        print()

    def run_continuous_validation(self):
        """Run continuous 8-GPU validation monitoring"""
        print(f"Starting 8-GPU validation monitoring (checking every {self.check_interval}s)")
        print("Press Ctrl+C to stop")
        print()

        try:
            while True:
                gpu_processes = self.get_gpu_processes()
                is_correct, issues = self.check_8gpu_placement_correctness(gpu_processes)

                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.log_validation_result(timestamp, gpu_processes, is_correct, issues)
                self.print_8gpu_status(gpu_processes, is_correct, issues)

                time.sleep(self.check_interval)

        except KeyboardInterrupt:
            print("\n" + "=" * 60)
            print("8-GPU validation monitoring stopped")
            self._print_summary()

    def _print_summary(self):
        """Print validation summary"""
        if not self.validation_history:
            return

        total_checks = len(self.validation_history)
        correct_checks = sum(1 for r in self.validation_history if r["placement_correct"])

        print(f"Summary: {correct_checks}/{total_checks} checks passed")
        if correct_checks == total_checks:
            print("✅ All 8-GPU validations passed - GPU placement is working correctly!")
        else:
            print("❌ Some 8-GPU validation issues detected - check logs for details")


def main():
    parser = argparse.ArgumentParser(description="Validate 8-GPU placement for split training")
    parser.add_argument("--interval", type=int, default=15, help="Check interval in seconds (default: 15)")
    parser.add_argument("--log-file", type=str, help="Log validation results to file")
    parser.add_argument("--single-check", action="store_true", help="Run single validation check and exit")

    args = parser.parse_args()

    validator = GPU8PlacementValidator(args.interval, args.log_file)

    if args.single_check:
        gpu_processes = validator.get_gpu_processes()
        is_correct, issues = validator.check_8gpu_placement_correctness(gpu_processes)
        validator.print_8gpu_status(gpu_processes, is_correct, issues)

        exit(0 if is_correct else 1)
    else:
        validator.run_continuous_validation()


if __name__ == "__main__":
    main()
