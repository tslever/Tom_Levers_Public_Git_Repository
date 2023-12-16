# Removes components of Jupyter-Notebook cells based on cell tags.
# Based on https://nbconvert.readthedocs.io/en/latest/removing_cells.html.
# Usage:
# python exporttopdf.py JuPyteR_Notebook.ipynb
# from tomleverspythonpackage.exporttopdf import export_to_pdf; export_to_pdf('JuPyteR_Notebook.ipynb')

import argparse
from traitlets.config import Config
from pathlib import Path
from nbconvert.exporters import PDFExporter
from nbconvert.preprocessors import TagRemovePreprocessor

def export_to_pdf(path_of_jupyter_notebook_as_string):

    path_of_jupyter_notebook_as_string
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

def parse_arguments():
    dictionary_of_arguments = {}
    parser = argparse.ArgumentParser(prog = 'Export JuPyteR Notebook To PDF', description = 'This program exports a JuPyteR notebook to a PDF.')
    parser.add_argument('path_of_jupyter_notebook_as_string', help = 'path of JuPyteR notebook as string')
    args = parser.parse_args()
    path_of_jupyter_notebook_as_string = args.path_of_jupyter_notebook_as_string
    print(f'path of JuPyteR notebook as string: {path_of_jupyter_notebook_as_string}')
    dictionary_of_arguments['path_of_jupyter_notebook_as_string'] = path_of_jupyter_notebook_as_string
    return dictionary_of_arguments

if __name__ == "__main__":
    dictionary_of_arguments = parse_arguments()
    export_to_pdf(dictionary_of_arguments['path_of_jupyter_notebook_as_string'])