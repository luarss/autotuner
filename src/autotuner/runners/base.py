"""Base runner implementations for AutoTuner."""

import os
import subprocess
from pathlib import Path
from typing import Any

from autotuner.core.interfaces import Runner


class BaseRunner(Runner):
    """Base runner implementation for command execution."""

    def __init__(self):
        self.execution_logs: list[str] = []
        self.execution_context: dict[str, Any] = {}

    def execute(self, command: str, **kwargs) -> dict[str, Any]:
        """Execute a command and return results."""
        try:
            # Set up environment
            env = os.environ.copy()
            env.update(kwargs.get("env", {}))

            # Execute command
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                env=env,
                cwd=kwargs.get("cwd"),
                timeout=kwargs.get("timeout"),
            )

            # Log execution
            self.execution_logs.append(f"Command: {command}")
            self.execution_logs.append(f"Return code: {result.returncode}")
            self.execution_logs.append(f"Stdout: {result.stdout}")
            if result.stderr:
                self.execution_logs.append(f"Stderr: {result.stderr}")

            return {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "command": command,
                "success": result.returncode == 0,
            }

        except subprocess.TimeoutExpired:
            error_msg = f"Command timed out: {command}"
            self.execution_logs.append(error_msg)
            return {
                "returncode": -1,
                "stdout": "",
                "stderr": error_msg,
                "command": command,
                "success": False,
                "timeout": True,
            }
        except Exception as e:
            error_msg = f"Execution failed: {str(e)}"
            self.execution_logs.append(error_msg)
            return {
                "returncode": -1,
                "stdout": "",
                "stderr": error_msg,
                "command": command,
                "success": False,
                "error": str(e),
            }

    def setup_execution_context(self, context: dict[str, Any]) -> None:
        """Set up the execution context."""
        self.execution_context.update(context)

    def get_execution_logs(self) -> list[str]:
        """Get execution logs."""
        return self.execution_logs.copy()

    def cleanup(self) -> None:
        """Clean up resources after execution."""
        self.execution_logs.clear()


class OpenROADRunner(BaseRunner):
    """Specialized runner for OpenROAD flow execution."""

    def __init__(self, orfs_flow_dir: str = None):
        super().__init__()
        self.orfs_flow_dir = Path(orfs_flow_dir) if orfs_flow_dir else None

    def execute(self, command: str, **kwargs) -> dict[str, Any]:
        """Execute OpenROAD flow command."""
        # Set up OpenROAD specific environment
        if self.orfs_flow_dir:
            kwargs.setdefault("cwd", str(self.orfs_flow_dir))

        # Add OpenROAD specific environment variables
        env = kwargs.get("env", {})
        if "OPENROAD_FLOW_DIR" not in env and self.orfs_flow_dir:
            env["OPENROAD_FLOW_DIR"] = str(self.orfs_flow_dir)
        kwargs["env"] = env

        return super().execute(command, **kwargs)

    def setup_execution_context(self, context: dict[str, Any]) -> None:
        """Set up OpenROAD execution context."""
        super().setup_execution_context(context)

        # Set OpenROAD specific context
        if "orfs_flow_dir" in context:
            self.orfs_flow_dir = Path(context["orfs_flow_dir"])

    def execute_flow(self, design: str, platform: str, variant: str = None, **kwargs) -> dict[str, Any]:
        """Execute OpenROAD flow for a specific design and platform."""
        command_parts = ["make"]

        if variant:
            command_parts.append(f"FLOW_VARIANT={variant}")

        command_parts.extend([f"DESIGN_CONFIG=designs/{platform}/{design}/config.mk", f"PLATFORM={platform}", design])

        command = " ".join(command_parts)
        return self.execute(command, **kwargs)
