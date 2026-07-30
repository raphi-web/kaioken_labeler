# One entry point for the things the README used to call out by hand.
#
# `make help` prints the targets; `make` with no argument does the same.
#
# Test runners:
#   make test        backend (pytest) + frontend (vitest)
#   make test-py     backend only
#   make test-js     frontend only
#
# Build / lint:
#   make build       vite build (frontend/dist/, which backend/main.py serves)
#   make lint        oxlint on the frontend
#
# Run the app:
#   make run         backend/main.py with the repo's project
#   make run-smoke   vite dev server, browse to /smoke.html
#
# Maintenance:
#   make clean       remove build artifacts and __pycache__ trees

.PHONY: help test test-py test-js build lint run run-smoke clean

help:                              ## show this list
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

test: test-py test-js               ## run all tests

test-py:                            ## backend tests (pytest)
	python3 -m pytest backend/

test-js:                            ## frontend tests (vitest)
	cd frontend && npm test

build:                              ## build the frontend (frontend/dist/)
	cd frontend && npm run build

lint:                               ## oxlint on the frontend
	cd frontend && npm run lint

run:                                ## launch the labeler
	python3 backend/main.py

run-smoke:                          ## vite dev server, browse to /smoke.html
	cd frontend && npm run dev

clean:                              ## remove build artifacts
	rm -rf frontend/dist
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name .pytest_cache -prune -exec rm -rf {} +