.PHONY: format
format:
	@ruff format
	@ruff check --fix

.PHONY: check
check:
	@ruff check
	@pre-commit run --all-files
