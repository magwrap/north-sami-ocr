TEX = latexmk
TEXFLAGS = -pdf -interaction=nonstopmode -halt-on-error

all: main.pdf

main.pdf: main.tex
	$(TEX) $(TEXFLAGS) main.tex

watch:
	$(TEX) $(TEXFLAGS) -pvc main.tex

clean:
	$(TEX) -C main.tex
	rm -f main.{aux,log,nav,out,snm,toc,vrb,fdb_latexmk,fls,synctex.gz}

.PHONY: all watch clean
