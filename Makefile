.PHONY: test-report

test-report:
	pytest tests/test_nemo_guard.py -v > test_results.txt 2>&1
