golden:
	python -m tests.golden.generate_golden_files

test:
	pytest -q
