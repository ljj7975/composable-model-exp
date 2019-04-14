#!/bin/sh
rm *.log
rm *.glg
rm *.glg-abr
rm *.aux
rm *.lof
rm *.lot
rm *.out
rm *.blg
rm *.glo
rm *.glo-abr
rm *.gls
rm *.gls-abr
rm *.ist
rm *.nomenclature-glg
rm *.nomenclature-glo
rm *.nomenclature-gls
rm *.bbl
rm *.symbols-glg
rm *.symbols-glo
rm *.symbols-gls
rm *.toc

pdflatex project
bibtex project
pdflatex project
pdflatex project
if [ "$(uname)" == "Darwin" ]; then
    # Do something under Mac OS X platform
    open project.pdf
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    # Do something under GNU/Linux platform
    xdg-open project.pdf
fi
