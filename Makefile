PYODIDE_VERSION ?= 0.29.3

.PHONY: pyodide-wheel pyodide-smoke

# Build the Pyodide (wasm32) wheel into dist-pyodide/
pyodide-wheel:
	PYODIDE_VERSION=$(PYODIDE_VERSION) ./scripts/build_pyodide.sh

# Run the render_frame smoke test natively and under Pyodide in Node; outputs must match
pyodide-smoke:
	npm install --no-save pyodide@$(PYODIDE_VERSION)
	python scripts/smoke_render_frame.py > dist-pyodide/smoke-native.json
	node scripts/pyodide_smoke.mjs dist-pyodide/pyayay-*.whl > dist-pyodide/smoke-pyodide.json
	diff dist-pyodide/smoke-native.json dist-pyodide/smoke-pyodide.json
	cat dist-pyodide/smoke-pyodide.json
