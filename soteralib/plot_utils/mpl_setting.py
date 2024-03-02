import matplotlib.pyplot as plt

# color settings
# https://www.yutaka-note.com/entry/matplotlib_color_cycle
# https://qiita.com/saka1_p/items/bb4206c6349eb61c073c
# https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_prop_cycle.html

mycolortab = ['#0072B2', '#E69F00', '#009E73', '#D55E00','#CC79A7', '#56B4E9' , '#F0E442', '#000000']
config = {'axes.prop_cycle': plt.cycler("color", mycolortab)}
plt.rcParams.update(config)

def set_plot_cycle(linestyle=False, marker=False):
    """
    Set the color, linestyle, and marker cycle for Matplotlib plots.

    Parameters:
    - linestyle (bool): If True, sets the linestyle cycle using predefined line styles.
    - marker (bool): If True, sets the marker cycle using predefined marker styles.

    Returns:
    - None

    Example:
    >>> set_plot_cycle(linestyle=True, marker=True)
    """
    my_color_cycle = ['#0072B2', '#E69F00', '#009E73', '#D55E00',
                      '#CC79A7', '#56B4E9' , '#F0E442', '#000000']
    my_linestyle_cycle = ['-', '--', '-.', ':', 
                          (0, (5,1,1,1,1,1)), 
                          (0, (5,1,1,1,1,1,1,1)), 
                          (0, (5,1,1,1,1,1,1,1,1,1)), 
                           '-']
    my_marker_cycle = ['o', 'v', 's', '*',
                       'p', '^', '<', '>']
    prop_cycle = plt.cycler(color=my_color_cycle) 
    if linestyle:
        prop_cycle += plt.cycler(linestyle=my_linestyle_cycle)
    if marker:
        prop_cycle += plt.cycler(marker=my_marker_cycle)
    config = {'axes.prop_cycle': prop_cycle}
    plt.rcParams.update(config)
    return

