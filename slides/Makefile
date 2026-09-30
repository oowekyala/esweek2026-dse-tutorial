PDF = dse-tutorial.pdf

all: $(PDF)

$(PDF): dse-tutorial.tex $(wildcard figures/*.pdf)
	latexmk -pdf -interaction=nonstopmode -halt-on-error dse-tutorial.tex

clean:
	latexmk -C dse-tutorial.tex

.PHONY: all clean
