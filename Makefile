
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
test: test_ast test_type test_ply_parser test_secd test_ski test_llvm test_transform

.PHONY: test_ast
test_ast:
	(cd test; ../venv/bin/$(PYTHON) -m unittest test_ast_interpreter.py -v )

.PHONY: test_parser
test_parser:
	(cd test; ../venv/bin/$(PYTHON) -m unittest test_parser.py -v )

.PHONY: test_ply_parser
test_ply_parser:
	(cd test; ../venv/bin/$(PYTHON) -m unittest test_ply_parser.py -v )

.PHONY: test_secd
test_secd:
	(cd test; ../venv/bin/$(PYTHON) -m unittest test_secd.py -v )

.PHONY: test_type
test_type:
	(cd test; ../venv/bin/$(PYTHON) -m unittest test_type_checker.py -v )

.PHONY: test_ski
test_ski:
	(cd test; ../venv/bin/$(PYTHON) -m unittest test_ski.py -v )
	(cd test; ../venv/bin/$(PYTHON) -m unittest test_ski_io.py -v )

.PHONY: test_llvm
test_llvm:
	(cd test; ../venv/bin/$(PYTHON) -m unittest test_llvm.py -v )

.PHONY: test_transform
test_transform:
	(cd test; ../venv/bin/$(PYTHON) -m unittest test_transform.py -v )
