import dearpygui.dearpygui as dpg
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

# Create DearPyGui context
dpg.create_context()

# Default number of rows and columns
row_num = 3
col_num = 3

def solve_and_plot_linear_system(sender, app_data, user_data):
    """
    Solves the linear system Ax = 0 and Ax = b, and plots the solutions.
    """
    global row_num, col_num
    matrix = []
    b_vector = []

    # Retrieve matrix and vector values from input
    for r in range(row_num):
        row = []
        for c in range(col_num):
            value = dpg.get_value(f"r{r + 1}c{c + 1}")
            row.append(value)
        matrix.append(row)
        b_value = dpg.get_value(f"b{r + 1}")
        b_vector.append(b_value)

    A = np.array(matrix, dtype=float)
    b = np.array(b_vector, dtype=float)

    # Solve Ax = 0
    null_space = linalg.null_space(A)
    if null_space.size > 0:
        x_null = null_space[:, 0]  # Take the first basis vector of the null space
        null_space_description = f"Basis vectors of the null space: {null_space}"
    else:
        x_null = "No Non-Trivial Solution"
        null_space_description = "The null space is trivial, so the only solution is the zero vector."

    # Solve Ax = b
    rank_A = np.linalg.matrix_rank(A)
    augmented_matrix = np.hstack([A, b.reshape(-1, 1)])
    rank_augmented = np.linalg.matrix_rank(augmented_matrix)
    if rank_A == rank_augmented:
        if rank_A == col_num:
            x_particular = np.linalg.solve(A, b)
        else:
            x_particular = "Infinitely Many Solutions"
    else:
        x_particular = "No Solution"

    # Display results
    dpg.set_value("result_null", f"Solution for Ax = 0: {x_null}\n{null_space_description}")
    dpg.set_value("result_particular", f"Solution for Ax = b: {x_particular}")

    # Plot solutions
    fig = plt.figure(figsize=(12, 5))

    if col_num <= 3:
        if col_num == 3:
            # Plot Ax = 0 in 3D
            ax1 = fig.add_subplot(121, projection='3d')
            if isinstance(x_null, np.ndarray):
                t = np.linspace(-10, 10, 100)
                x = x_null[0] * t
                y = x_null[1] * t
                z = x_null[2] * t
                ax1.plot(x, y, z)
            elif null_space.size > 0:
                t = np.linspace(-10, 10, 100)
                for i in range(null_space.shape[1]):
                    x = null_space[0, i] * t
                    y = null_space[1, i] * t
                    z = null_space[2, i] * t
                    ax1.plot(x, y, z)
            ax1.set_title('Solution for Ax = 0')
            ax1.set_xlabel('x')
            ax1.set_ylabel('y')
            ax1.set_zlabel('z')

            # Plot Ax = b in 3D
            ax2 = fig.add_subplot(122, projection='3d')
            if isinstance(x_particular, np.ndarray):
                ax2.scatter(x_particular[0], x_particular[1], x_particular[2])
            elif x_particular == "Infinitely Many Solutions" and null_space.size > 0:
                t = np.linspace(-10, 10, 100)
                for i in range(null_space.shape[1]):
                    x_homogeneous = null_space[:, i]
                    x = x_homogeneous[0] * t + np.linalg.lstsq(A, b, rcond=None)[0][0]
                    y = x_homogeneous[1] * t + np.linalg.lstsq(A, b, rcond=None)[0][1]
                    z = x_homogeneous[2] * t + np.linalg.lstsq(A, b, rcond=None)[0][2]
                    ax2.plot(x, y, z)
            ax2.set_title('Solution for Ax = b')
            ax2.set_xlabel('x')
            ax2.set_ylabel('y')
            ax2.set_zlabel('z')

        elif col_num == 2:
            # Plot Ax = 0 in 2D
            ax1 = fig.add_subplot(121)
            if isinstance(x_null, np.ndarray):
                t = np.linspace(-10, 10, 100)
                x = x_null[0] * t
                y = x_null[1] * t
                ax1.plot(x, y)
            elif null_space.size > 0:
                t = np.linspace(-10, 10, 100)
                for i in range(null_space.shape[1]):
                    x = null_space[0, i] * t
                    y = null_space[1, i] * t
                    ax1.plot(x, y)
            ax1.set_title('Solution for Ax = 0')
            ax1.set_xlabel('x')
            ax1.set_ylabel('y')

            # Plot Ax = b in 2D
            ax2 = fig.add_subplot(122)
            if isinstance(x_particular, np.ndarray):
                ax2.scatter(x_particular[0], x_particular[1])
            elif x_particular == "Infinitely Many Solutions" and null_space.size > 0:
                t = np.linspace(-10, 10, 100)
                for i in range(null_space.shape[1]):
                    x_homogeneous = null_space[:, i]
                    x = x_homogeneous[0] * t + np.linalg.lstsq(A, b, rcond=None)[0][0]
                    y = x_homogeneous[1] * t + np.linalg.lstsq(A, b, rcond=None)[0][1]
                    ax2.plot(x, y)
            ax2.set_title('Solution for Ax = b')
            ax2.set_xlabel('x')
            ax2.set_ylabel('y')

        elif col_num == 1:
            # Plot Ax = 0 in 1D
            ax1 = fig.add_subplot(121)
            if isinstance(x_null, np.ndarray):
                t = np.linspace(-10, 10, 100)
                x = x_null[0] * t
                ax1.plot(t, x)
            elif null_space.size > 0:
                t = np.linspace(-10, 10, 100)
                for i in range(null_space.shape[1]):
                    x = null_space[0, i] * t
                    ax1.plot(t, x)
            ax1.set_title('Solution for Ax = 0')
            ax1.set_xlabel('t')
            ax1.set_ylabel('x')

            # Plot Ax = b in 1D
            ax2 = fig.add_subplot(122)
            if isinstance(x_particular, np.ndarray):
                ax2.scatter([0], x_particular[0])
            elif x_particular == "Infinitely Many Solutions" and null_space.size > 0:
                t = np.linspace(-10, 10, 100)
                for i in range(null_space.shape[1]):
                    x_homogeneous = null_space[:, i]
                    x = x_homogeneous[0] * t + np.linalg.lstsq(A, b, rcond=None)[0][0]
                    ax2.plot(t, x)
            ax2.set_title('Solution for Ax = b')
            ax2.set_xlabel('t')
            ax2.set_ylabel('x')

    else:
        # Handle cases where the matrix has more than 3 columns
        plt.text(0.5, 0.5, 'Matrix dimensions are greater than 3x3. Visualization is limited to 1D, 2D, or 3D projections.',
                 horizontalalignment='center', verticalalignment='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def source_dtype():
    """
    Returns the data type for the matrix input fields.
    """
    return 'float'

def matrix_table(sender, app_data, user_data):
    """
    Creates the input table for the matrix and vector.
    """
    global row_num, col_num
    row_num = int(dpg.get_value("rows"))
    col_num = int(dpg.get_value("columns"))

    # Delete existing matrix table if it exists
    if dpg.does_item_exist("Matrix"):
        dpg.delete_item("Matrix", children_only=True)
        dpg.delete_item("Matrix")

    # Create new matrix table
    with dpg.table(
            header_row=True,
            row_background=True,
            borders_innerH=True,
            borders_innerV=True,
            borders_outerH=True,
            borders_outerV=True,
            parent="pri",
            tag="Matrix"
    ):
        # Add columns for matrix A and vector b
        for i in range(col_num + 1):  # +1 for b column
            dpg.add_table_column(label=f'{"b" if i == col_num else f"C{i + 1}"}')
        for R in range(row_num):
            with dpg.table_row():
                for C in range(col_num):
                    dpg.add_input_float(tag=f"r{R + 1}c{C + 1}", label=f'R{R + 1}', source=source_dtype())
                dpg.add_input_float(tag=f"b{R + 1}", label=f'b{R + 1}', source=source_dtype())

# Main window
with dpg.window(tag='pri'):
    # Input for rows and columns
    dpg.add_text("Enter number of rows and columns:")
    dpg.add_input_int(tag="rows", label="Number of Rows", default_value=3)
    dpg.add_input_int(tag="columns", label="Number of Columns", default_value=3)

    col_num = dpg.get_value("columns")
    row_num = dpg.get_value("rows")

    # Button to create the matrix input table
    dpg.add_button(label="Create Matrix", callback=matrix_table)

    # Button to solve the linear system and plot results
    dpg.add_button(label="Solve and Plot", callback=solve_and_plot_linear_system)

    # Text fields to display results
    dpg.add_text("", tag="result_null")
    dpg.add_text("", tag="result_particular")

# Create and show the viewport
dpg.create_viewport(title='Linear Algebra App (R^3)', width=800, height=600)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window('pri', True)
dpg.start_dearpygui()

# Destroy the DearPyGui context when done
dpg.destroy_context()
