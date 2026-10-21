from __future__ import annotations

import hashlib
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from deepsolo.candidates import (
    DEEPSOLO_SOURCE_REVISION,
    GITHUB_COMMIT_ARCHIVE_SOURCE_TYPE,
    verify_source_identity,
)


class DeepSoloSourceIdentityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.source = self.root / "DeepSolo"
        self.source.mkdir()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_git_checkout_valid(self) -> None:
        with patch(
            "deepsolo.candidates.subprocess.run",
            return_value=subprocess.CompletedProcess(
                [], 0, stdout=DEEPSOLO_SOURCE_REVISION + "\n", stderr=""
            ),
        ) as run:
            identity = verify_source_identity(self.source)
        run.assert_called_once_with(
            ["git", "-C", str(self.source.resolve()), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(identity["source_type"], "git_checkout")
        self.assertEqual(identity["source_revision"], DEEPSOLO_SOURCE_REVISION)
        self.assertIsNone(identity["source_archive_sha256"])

    def test_git_checkout_wrong_commit_fails(self) -> None:
        with patch(
            "deepsolo.candidates.subprocess.run",
            return_value=subprocess.CompletedProcess(
                [], 0, stdout="0" * 40 + "\n", stderr=""
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "source revision mismatch"):
                verify_source_identity(self.source)

    def _archive_inputs(self) -> tuple:
        archive = self.root / "source.tar.gz"
        archive.write_bytes(b"official archive fixture")
        archive_sha256 = hashlib.sha256(archive.read_bytes()).hexdigest()
        marker = self.root / "source-commit.txt"
        marker.write_text(DEEPSOLO_SOURCE_REVISION + "\n", encoding="ascii")
        return archive, archive_sha256, marker

    def test_github_archive_valid(self) -> None:
        archive, archive_sha256, marker = self._archive_inputs()
        with patch(
            "deepsolo.candidates.QUALIFIED_SOURCE_ARCHIVE_SHA256",
            archive_sha256,
        ):
            identity = verify_source_identity(
                self.source,
                source_type=GITHUB_COMMIT_ARCHIVE_SOURCE_TYPE,
                declared_revision=DEEPSOLO_SOURCE_REVISION,
                source_commit_marker=marker,
                source_archive=archive,
                source_archive_sha256=archive_sha256,
            )
        self.assertEqual(identity["source_type"], "github_commit_archive")
        self.assertEqual(identity["source_revision"], DEEPSOLO_SOURCE_REVISION)
        self.assertEqual(identity["source_archive_sha256"], archive_sha256)

    def test_github_archive_wrong_marker_commit_fails(self) -> None:
        archive, archive_sha256, marker = self._archive_inputs()
        marker.write_text("0" * 40 + "\n", encoding="ascii")
        with patch(
            "deepsolo.candidates.QUALIFIED_SOURCE_ARCHIVE_SHA256",
            archive_sha256,
        ):
            with self.assertRaisesRegex(RuntimeError, "commit marker"):
                verify_source_identity(
                    self.source,
                    source_type=GITHUB_COMMIT_ARCHIVE_SOURCE_TYPE,
                    declared_revision=DEEPSOLO_SOURCE_REVISION,
                    source_commit_marker=marker,
                    source_archive=archive,
                    source_archive_sha256=archive_sha256,
                )

    def test_github_archive_wrong_sha256_fails(self) -> None:
        archive, _, marker = self._archive_inputs()
        with patch(
            "deepsolo.candidates.QUALIFIED_SOURCE_ARCHIVE_SHA256",
            "0" * 64,
        ):
            with self.assertRaisesRegex(RuntimeError, "archive SHA256 mismatch"):
                verify_source_identity(
                    self.source,
                    source_type=GITHUB_COMMIT_ARCHIVE_SOURCE_TYPE,
                    declared_revision=DEEPSOLO_SOURCE_REVISION,
                    source_commit_marker=marker,
                    source_archive=archive,
                    source_archive_sha256="0" * 64,
                )

    def test_source_without_git_or_archive_metadata_fails(self) -> None:
        error = subprocess.CalledProcessError(128, ["git", "rev-parse", "HEAD"])
        with patch("deepsolo.candidates.subprocess.run", side_effect=error):
            with self.assertRaisesRegex(RuntimeError, "git checkout revision"):
                verify_source_identity(self.source)


if __name__ == "__main__":
    unittest.main()
