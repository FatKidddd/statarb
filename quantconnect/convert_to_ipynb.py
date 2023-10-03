import nbformat as nbf

# Function to split the script into cells based on a specific string
def split_script_into_cells(script):
    cells = []
    lines = script.split('\n')
    cell = []
    for line in lines:
        if line.strip().startswith("#cell"):
            if cell:
                cells.append(cell)
            cell = []
        else:
            cell.append(line)
    if cell:
        cells.append(cell)
    return cells

# Read the Python script file
with open('input.py', 'r') as script_file:
    script_content = script_file.read()

# Split the script into cells
script_cells = split_script_into_cells(script_content)

# Create a new Jupyter Notebook
nb = nbf.v4.new_notebook()

# Add cells to the notebook
for cell_num, script_cell in enumerate(script_cells, 1):
    source_code = '\n'.join(script_cell)
    code_cell = nbf.v4.new_code_cell(source_code)
    nb.cells.append(code_cell)

# Save the notebook as an IPython Notebook (.ipynb) file
nbf.write(nb, 'output.ipynb')