import fitz
import numpy as np  # import package PyMuPDF


# open input PDF
doc = fitz.open("test.pdf")


# load desired page (0-based page number)
# page = doc[0]
for page in doc:

    # search for "whale", results in a list of rectangles
    rects = page.search_for("meta")

    # mark all occurrences in one go
    # page.add_highlight_annot(rects)

    # define two colors to choose from
    colors = (fitz.pdfcolor["pink"], fitz.pdfcolor["green"])
    print(colors)

    # give each occurrence a different color
    for i, rect in enumerate(rects):
        # color = colors[i % 2]  # select the color for this occurrence
        annot = page.add_highlight_annot(rect)  # highlight it
        # annot.set_colors(stroke=color)  # change default color
        a = np.random.rand()
        annot.set_opacity(a)  # set opacity
        annot.update()  # update annotation

# save the document with these changes
doc.save("output.pdf")
