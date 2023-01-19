from traitlets.config import Config
import nbformat as nbf
from nbconvert.exporters import PDFExporter
from nbconvert.preprocessors import TagRemovePreprocessor
from pathlib import Path
import sys

# Removes components of Jupyter-Notebook cells based on cell tags.
# Based on https://nbconvert.readthedocs.io/en/latest/removing_cells.html.

def main():

    path_of_jupyter_notebook_as_string = sys.argv[1]
    path_of_jupyter_notebook = Path(path_of_jupyter_notebook_as_string)
    path_of_parent_directory = path_of_jupyter_notebook.parent
    absolute_path_of_parent_directory = path_of_parent_directory.resolve()
    name_of_jupyter_notebook = path_of_jupyter_notebook.stem

    # Setup config
    c = Config()

    # Configure tag removal - be sure to tag your cells to remove  using the
    # words remove_cell to remove cells. You can also modify the code to use
    # a different tag word
    c.TagRemovePreprocessor.remove_cell_tags = ("remove_cell",)
    c.TagRemovePreprocessor.remove_all_outputs_tags = ('remove_output',)
    c.TagRemovePreprocessor.remove_input_tags = ('remove_input',)
    c.TagRemovePreprocessor.enabled = True

    # Configure and run out exporter
    c.PDFExporter.preprocessors = ["nbconvert.preprocessors.TagRemovePreprocessor"]

    exporter = PDFExporter(config=c)
    exporter.register_preprocessor(TagRemovePreprocessor(config=c),True)

    # Configure and run our exporter - returns a tuple - first element with pdf,
    # second with notebook metadata
    output = PDFExporter(config=c).from_filename(path_of_jupyter_notebook_as_string)

    # Write to output pdf file
    with open(str(absolute_path_of_parent_directory) + "/" + name_of_jupyter_notebook + ".pdf",  "wb") as f:
        f.write(output[0])

if __name__ == "__main__":
    main()