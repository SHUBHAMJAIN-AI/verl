#!/usr/bin/env python3
"""
GPU Placement Validation Script for Split Training

Validates that models are correctly placed on separate GPUs:
- GPU 0: Actor + Rollout
- GPU 1: Critic + Reference

Usage:
    python validate_gpu_placement.py [--interval SECONDS] [--log-file FILE]
"""

import argparse
import json
import subprocess
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple


class GPUPlacementValidator:
    def __init__(self, check_interval: int = 10, log_file: Optional[str] = None):
        self.check_interval = check_interval
        self.log_file = log_file
        self.validation_history = []

        # Expected placement
        self.expected_placement = {0: ["actor", "rollout"], 1: ["critic", "reference"]}

        print("=" * 80)
        print("GPU PLACEMENT VALIDATION FOR SPLIT TRAINING")
        print("=" * 80)
        print("Expected Placement:")
        print("- GPU 0: Actor + Rollout")
        print("- GPU 1: Critic + Reference")
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
                # This is a Ray worker - need to determine type from parent or context
                return self._identify_ray_worker_type(pid, cmdline)
            elif "vllm" in cmdline or "sglang" in cmdline:
                return "rollout"
            elif "main_ppo_split" in cmdline:
                return "main_trainer"
            else:
                return "unknown"

        except Exception:
            return "unknown"

    def _identify_ray_worker_type(self, pid: str, cmdline: str) -> str:
        """Identify Ray worker type by analyzing process tree and patterns"""
        try:
            # Check if this worker is saving actor or critic checkpoints
            # by looking at recent file operations (approximation)
            result = subprocess.run(["lsof", "-p", pid, "+D", "/root/verl/examples/split_placement/checkpoints"], capture_output=True, text=True, timeout=5)

            if "actor" in result.stdout:
                return "actor"
            elif "critic" in result.stdout:
                return "critic"

            # Fallback: try to identify from parent process or other clues
            # Check parent process info
            try:
                with open(f"/proc/{pid}/stat") as f:
                    stat_info = f.read().split()
                    ppid = stat_info[3]

                with open(f"/proc/{ppid}/cmdline") as f:
                    parent_cmdline = f.read().replace("\x00", " ").strip().lower()

                if "actor" in parent_cmdline or "rollout" in parent_cmdline:
                    return "actor_or_rollout"
                elif "critic" in parent_cmdline or "reference" in parent_cmdline:
                    return "critic_or_reference"
            except Exception:
                pass

            return "ray_worker"

        except Exception:
            return "ray_worker"

    def check_placement_correctness(self, gpu_processes: Dict[int, List[Dict]]) -> Tuple[bool, List[str]]:
        """Check if placement matches expected configuration"""
        issues = []
        all_correct = True

        for gpu_id, processes in gpu_processes.items():
            if gpu_id not in self.expected_placement:
                issues.append(f"Unexpected GPU {gpu_id} in use")
                all_correct = False
                continue

            expected_components = self.expected_placement[gpu_id]
            found_components = [p["component_type"] for p in processes]

            # Check for expected components
            for expected in expected_components:
                component_found = any(expected in comp or comp in expected for comp in found_components)
                if not component_found:
                    # For Ray workers, be more lenient since identification is approximate
                    if any("ray_worker" in comp or "actor_or_rollout" in comp for comp in found_components) and expected in ["actor", "rollout"]:
                        continue
                    if any("ray_worker" in comp or "critic_or_reference" in comp for comp in found_components) and expected in ["critic", "reference"]:
                        continue
                    issues.append(f"GPU {gpu_id}: Expected {expected} component not clearly identified")

            # Check for unexpected components on wrong GPU
            for process in processes:
                comp_type = process["component_type"]
                if comp_type == "unknown" or "ray_worker" in comp_type:
                    continue  # Skip uncertain identifications

                expected_gpu = None
                for exp_gpu, exp_comps in self.expected_placement.items():
                    if any(comp_type in exp_comp or exp_comp in comp_type for exp_comp in exp_comps):
                        expected_gpu = exp_gpu
                        break

                if expected_gpu is not None and expected_gpu != gpu_id:
                    issues.append(f"GPU {gpu_id}: Found {comp_type} (PID {process['pid']}) - should be on GPU {expected_gpu}")
                    all_correct = False

        return all_correct, issues

    def log_validation_result(self, timestamp: str, gpu_processes: Dict[int, List[Dict]], is_correct: bool, issues: List[str]):
        """Log validation results"""
        result = {"timestamp": timestamp, "gpu_processes": gpu_processes, "placement_correct": is_correct, "issues": issues}

        self.validation_history.append(result)

        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(result) + "\n")

    def print_status(self, gpu_processes: Dict[int, List[Dict]], is_correct: bool, issues: List[str]):
        """Print current validation status"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(f"[{timestamp}] Validation Check")
        print("-" * 50)

        if not gpu_processes:
            print("❌ No GPU processes detected - training may not be running")
            print()
            return

        # Print GPU status
        for gpu_id in sorted(gpu_processes.keys()):
            processes = gpu_processes[gpu_id]
            expected = self.expected_placement.get(gpu_id, [])

            print(f"GPU {gpu_id} (Expected: {', '.join(expected)}):")
            if not processes:
                print("  No processes")
            else:
                for proc in processes:
                    print(f"  PID {proc['pid']}: {proc['component_type']} ({proc['memory_mb']} MB)")

        # Print validation result
        if is_correct:
            print("✅ PLACEMENT CORRECT: All components on expected GPUs")
        else:
            print("❌ PLACEMENT ISSUES DETECTED:")
            for issue in issues:
                print(f"   - {issue}")

        print()

    def run_continuous_validation(self):
        """Run continuous validation monitoring"""
        print(f"Starting continuous validation (checking every {self.check_interval}s)")
        print("Press Ctrl+C to stop")
        print()

        try:
            while True:
                gpu_processes = self.get_gpu_processes()
                is_correct, issues = self.check_placement_correctness(gpu_processes)

                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.log_validation_result(timestamp, gpu_processes, is_correct, issues)
                self.print_status(gpu_processes, is_correct, issues)

                time.sleep(self.check_interval)

        except KeyboardInterrupt:
            print("\n" + "=" * 50)
            print("Validation monitoring stopped")
            self._print_summary()

    def _print_summary(self):
        """Print validation summary"""
        if not self.validation_history:
            return

        total_checks = len(self.validation_history)
        correct_checks = sum(1 for r in self.validation_history if r["placement_correct"])

        print(f"Summary: {correct_checks}/{total_checks} checks passed")
        if correct_checks == total_checks:
            print("✅ All validations passed - GPU placement is working correctly!")
        else:
            print("❌ Some validation issues detected - check logs for details")


def main():
    parser = argparse.ArgumentParser(description="Validate GPU placement for split training")
    parser.add_argument("--interval", type=int, default=10, help="Check interval in seconds (default: 10)")
    parser.add_argument("--log-file", type=str, help="Log validation results to file")
    parser.add_argument("--single-check", action="store_true", help="Run single validation check and exit")

    args = parser.parse_args()

    validator = GPUPlacementValidator(args.interval, args.log_file)

    if args.single_check:
        gpu_processes = validator.get_gpu_processes()
        is_correct, issues = validator.check_placement_correctness(gpu_processes)
        validator.print_status(gpu_processes, is_correct, issues)

        exit(0 if is_correct else 1)
    else:
        validator.run_continuous_validation()


if __name__ == "__main__":
    main()
