"""Run HELM-based evaluations with local inputs in an offline Docker worker."""

import os
import shlex
from pathlib import Path

from magnet.process_node import MagnetProcessNode


class OfflineHelmProcessNode(MagnetProcessNode):
    """Mount code and inputs read-only; allow writes only to the job directory.

    MAGNET's default worker forwards inference credentials and uses host
    networking. This evaluator only needs saved responses and local weights.
    """

    container_runtime_env = ()
    container_capture_env = ("PYTHONPATH",)

    def _container_command_prefix(self):
        if str(self.container_docker_args or "").strip():
            raise ValueError("Offline HELM jobs do not accept extra Docker arguments")

        command = [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
        ]
        for mount in self._container_mount_paths():
            path = Path(mount).resolve()
            command.extend(["-v", f"{path}:{path}:ro"])

        # The more specific job mount overrides the read-only output root.
        job_directory = Path(self.final_node_dpath).resolve()
        command.extend(
            [
                "-v",
                f"{job_directory}:{job_directory}:rw",
                "-e",
                "HOME=/tmp",
                "-e",
                "TRANSFORMERS_OFFLINE=1",
                "-e",
                "HF_HUB_OFFLINE=1",
            ]
        )
        pythonpath = self._container_resolved_env().get("PYTHONPATH")
        if pythonpath:
            command.extend(["-e", f"PYTHONPATH={pythonpath}"])

        # Expand PWD in the generated invocation script, not in the controller.
        return (
            shlex.join(command) + ' -w "$PWD" ' + shlex.quote(str(self.container_image))
        )
