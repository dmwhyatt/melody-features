#!/usr/bin/env python3
"""
Script to run all tests in the project.
"""

import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_tests():
    """Run all tests in the project."""

    # Get the project root directory (parent of tests/)
    project_root = Path(__file__).parent.parent

    # Build the pytest command
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/",  # Run pytest tests
        "--doctest-modules",
        "src/",  # Run doctests in src/
        "-v",  # Verbose output
        "--tb=short",  # Short traceback format
        "--strict-markers",  # Strict marker checking
        "--disable-warnings",  # Disable warnings for cleaner output
    ]

    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)

    try:
        # Run the tests from project root
        result = subprocess.run(
            cmd,
            cwd=project_root,
            capture_output=False,  # Show output in real-time
            text=True,
        )

        print("-" * 60)

        if result.returncode == 0:
            logger.info("All tests passed successfully!")
            return True
        else:
            logger.info("Some tests failed")
            return False

    except Exception as e:
        logger.error(f"Error running tests: {e}")
        return False


def run_specific_test_suite(suite_name):
    """Run a specific test suite."""

    # Get the project root directory (parent of tests/)
    project_root = Path(__file__).parent.parent

    suites = {
        "pytest": ["tests/", "-v"],
        "doctest": ["--doctest-modules", "src/", "-v"],
        "corpus": ["tests/test_corpus_import.py", "-v"],
    }

    if suite_name not in suites:
        logger.error(f"Unknown test suite: {suite_name}")
        logger.error(f"Available suites: {', '.join(suites.keys())}")
        return False

    logger.info(f"Running {suite_name} tests...")

    cmd = [sys.executable, "-m", "pytest"] + suites[suite_name]

    try:
        result = subprocess.run(cmd, cwd=project_root, capture_output=False, text=True)

        return result.returncode == 0

    except Exception as e:
        logger.error(f"Error running {suite_name} tests: {e}")
        return False


def main():
    """Main function to handle command line arguments."""

    if len(sys.argv) > 1:
        suite = sys.argv[1].lower()
        if suite in ["pytest", "doctest", "corpus"]:
            success = run_specific_test_suite(suite)
        else:
            logger.error(f"Unknown test suite: {suite}")
            logger.error("Usage: python tests/run_tests.py [pytest|doctest|corpus]")
            logger.error("       python tests/run_tests.py (runs all tests)")
            return 1
    else:
        success = run_tests()

    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
