all: pdf html

html: src/report.qmd
	quarto render src/report.qmd --to html

pdf: src/report.qmd
	quarto render src/report.qmd --to pdf