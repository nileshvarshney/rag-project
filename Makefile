VENV    := .venv
PIP     := $(VENV)/bin/pip
COMPILE := $(VENV)/bin/pip-compile
SYNC    := $(VENV)/bin/pip-sync

# ── Lock-file management ──────────────────────────────────────────────────────

.PHONY: compile
compile:                        ## Re-pin all lock files from *.in sources
	$(COMPILE) requirements/base.in -o requirements/base.txt
	$(COMPILE) requirements/dev.in  -o requirements/dev.txt

.PHONY: sync
sync: requirements/base.txt    ## Install/remove packages to exactly match base.txt
	$(SYNC) requirements/base.txt

.PHONY: sync-dev
sync-dev: requirements/dev.txt ## Install/remove packages to exactly match dev.txt (includes base)
	$(SYNC) requirements/dev.txt

# ── Convenience targets ───────────────────────────────────────────────────────

.PHONY: test
test:                           ## Run the test suite
	$(VENV)/bin/pytest

.PHONY: ingest
ingest:                         ## Run incremental document ingest
	$(VENV)/bin/python ingest.py

.PHONY: serve
serve:                          ## Start the FastAPI server
	$(VENV)/bin/python server.py

.PHONY: help
help:                           ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*##"}; {printf "  %-12s %s\n", $$1, $$2}'
