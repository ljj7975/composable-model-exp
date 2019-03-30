#!/bin/sh
pdflatex project
bibtex project
pdflatex project
pdflatex project
xdg-open project.pdf