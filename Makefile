# --- Source files

pygment_targets := $(patsubst %.cpp,%.tex,$(wildcard code/*.cpp))

NO_COLOR=\x1b[0m
OK_COLOR=\x1b[32;01m
ERROR_COLOR=\x1b[31;01m
WARN_COLOR=\x1b[33;01m

# --- Main targets


all: $(pygment_targets)

clean:
	@rm -f $(pygment_targets)

graphs: $(pygment_targets) 

.PHONY: all clean slides

# --- Syntax highlighting for code (needs pygmentize)
# http://pygments.org/
$(pygment_targets): %.tex: %.cpp
	@echo "$(OK_COLOR)[x] Syntax highlighting for $<$(NO_COLOR)"
	@pygmentize -P mathescape=True -f latex "$<" > $@




