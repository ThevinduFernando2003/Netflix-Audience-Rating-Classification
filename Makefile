PYTHON := .venv/Scripts/python.exe

.PHONY: setup profile design-target tune train evaluate improve test

setup:
	"C:\Users\User\AppData\Roaming\uv\python\cpython-3.11.15-windows-x86_64-none\python.exe" -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

profile:
	$(PYTHON) scripts/profile_data.py

design-target:
	$(PYTHON) scripts/design_target.py

tune:
	$(PYTHON) scripts/tune.py

train:
	$(PYTHON) scripts/train.py

evaluate:
	$(PYTHON) scripts/evaluate_test.py

improve:
	$(PYTHON) scripts/improve_cv.py

test:
	$(PYTHON) -m pytest
