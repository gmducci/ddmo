import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap, LogNorm

dark_blue = [0./255., 108./255., 102./255.]
light_green = [198./255., 211./255., 37./255.]
light_blue = [0./255., 177./255., 234./255.]
orange = [239./255., 124./255., 0./255.]

fhicmap =  LinearSegmentedColormap.from_list('fhicmap', [light_blue, 'white', orange], 512)
fhicmap2 =  LinearSegmentedColormap.from_list('fhicmap2', [dark_blue, light_green], 512)

def set_poster_params():

    black       = [0., 0., 0.]
    dark_green  = [0./255., 108./255., 102./255.]
    dark_blue   = [41./255., 72./255., 93./255.]
    light_green = [198./255., 211./255., 37./255.]
    light_blue  = [0./255., 177./255., 234./255.]
    orange      = [239./255., 124./255., 0./255.]
    red         = [192./255., 0., 0.]
    blue        = [0., 112./255., 192./255.]
    green       = [0., 176./255., 82./255.]

    ratio = 28./12.
    textcolor = black

    preamblelist =  [
                        r'\usepackage{helvet}',
                        r'\usepackage{amsmath}',
                        #r'\usepackage{mathptmx}', #If you want arial, comment out
                        r'\usepackage[EULERGREEK]{sansmath}',
                        r'\sansmath',
                        r'\usepackage{siunitx}',
                        r'\sisetup{detect-all}',
                        r'\sisetup{detect-family = true}',
                    ]
    rcdict =    {
        "axes.edgecolor":       textcolor,
        "axes.labelcolor":      textcolor,
        "axes.titlepad":        6.0 * ratio,
        "axes.labelpad":        4.0 * ratio,
        'figure.figsize':       (5.2 *ratio, 3.9 * ratio),
        'font.family':          'serif',
        #'font.family':          'sans-serif',
        'font.sans-serif':      ['helvet'],
        #'font.sans-serif':      ['Arial'],
        'font.size':            12 * ratio,
        'lines.linewidth':      3.,
        'lines.markersize':     5 * ratio,
        'mathtext.fontset':     'cm', #'stixsans',
        "text.color":           textcolor,
        'text.latex.preamble':  '\n'.join(preamblelist),
        'text.usetex':          True,
        "xtick.color":          textcolor,
        'xtick.direction':      'in',
        'xtick.major.size':     3.5 * ratio,
        'xtick.major.pad':      3.5 * ratio,
        'xtick.minor.size':     2.0 * ratio,
        'xtick.minor.pad':      3.4 * ratio,
        'xtick.minor.visible':  True,
        'xtick.top':            True,
        "ytick.color":          textcolor,
        'ytick.direction':      'in',
        'ytick.major.size':     3.5 * ratio,
        'ytick.major.pad':      3.5 * ratio,
        'ytick.minor.size':     2.0 * ratio,
        'ytick.minor.pad':      3.4 * ratio,
        'ytick.minor.visible':  True,
        'ytick.right':          True,
        'axes.prop_cycle':      mpl.cycler(color=[
            dark_blue,
            light_green,
            light_blue,
            orange,
            dark_green,
            red,
            green,
            blue
            ]),
        'image.cmap':           "coolwarm",
    }
    plt.rcParams.update(rcdict)
    return None

def get_fhicmap():
    light_blue  = [0./255., 177./255., 234./255.]
    orange      = [239./255., 124./255., 0./255.]

    fhicmap =  LinearSegmentedColormap.from_list(
        'fhicmap',  # Identifier for the colormap
        [light_blue, 'white', orange], # Sequence of colors from 0. to 1.
        512 # Number of unique colors in colormap
    )
    return fhicmap


def get_fhicmap2():
    dark_green  = [0./255., 108./255., 102./255.]
    light_green = [198./255., 211./255., 37./255.]

    fhicmap2 =  LinearSegmentedColormap.from_list(
        'fhicmap2',
        [dark_green, light_green],
        512
    )
    return fhicmap2
