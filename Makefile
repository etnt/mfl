
VENV_NAME := venv
PYTHON := python3
PIP := $(VENV_NAME)/bin/pip
REQUIREMENTS := requirements.txt


# Default target
.PHONY: all
all: venv

# Create virtual environment and install requirements
.PHONY: venv
venv: $(VENV_NAME)/bin/activate

$(VENV_NAME)/bin/activate: $(REQUIREMENTS)
	$(PYTHON) -m venv $(VENV_NAME)
	$(PIP) install -r $(REQUIREMENTS)
	touch $(VENV_NAME)/bin/activate

# Clean up
.PHONY: clean
clean:
	rm -rf $(VENV_NAME)

# Install a new package and add it to requirements.txt
.PHONY: add
add:
	@read -p "Enter package name: " package; \
	$(PIP) install $$package && $(PIP) freeze | grep -i $$package >> $(REQUIREMENTS)

.PHONY: install-requirements
install-requirements: 
	$(PIP) install -r $(REQUIREMENTS)

.PHONY: test
test: 
	./venv/bin/$(PYTHON) -m unittest ./test/test_ast_interpreter.py -v
	./venv/bin/$(PYTHON) -m unittest ./test/test_parser.py -v
