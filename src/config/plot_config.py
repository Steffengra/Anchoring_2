
import matplotlib.pyplot as plt


class Config:
    def __init__(self):
        self.bar_plot_args: dict = {
            'height': 0.7,  # default: 0.8
            'error_kw': {  # for error bars
                'elinewidth': 1,
                'capsize': 2,
                'capthick': 1.4,
            },
        }

        # Styling
        # for font sizes, see table below
        plt.rc('axes', labelsize=8.33)  # fontsize of the axes labels in pt, default: 10
        plt.rc('xtick', labelsize=8.33)  # fontsize of the x tick labels in pt, default: 10
        plt.rc('ytick', labelsize=8.33)  # fontsize of the y tick labels in pt, default: 10
        plt.rc('legend', fontsize=8.33)  # fontsize of the legend in pt, default: 10

        # latex \pt         8   8.5     9   9.5    10  10.5    11  11.5    12
        #               -----------------------------------------------------
        # \tiny          4.00  4.25  4.50  4.75  5.00  5.25  5.50  5.75  6.00
        # \scriptsize    5.33  5.67  6.00  6.33  6.67  7.00  7.33  7.67  8.00
        # \footnotesize  6.67  7.08  7.50  7.92  8.33  8.75  9.17  9.58 10.00
        # \small         7.30  7.76  8.21  8.67  9.13  9.58 10.04 10.49 10.95
        # \normalsize    8.00  8.50  9.00  9.50 10.00 10.50 11.00 11.50 12.00
        # \large         9.60 10.20 10.80 11.40 12.00 12.60 13.20 13.80 14.40
        # \Large        11.52 12.24 12.96 13.68 14.40 15.12 15.84 16.56 17.28
        # \LARGE        13.82 14.69 15.55 16.42 17.28 18.14 19.01 19.87 20.74
        # \huge         16.59 17.63 18.67 19.70 20.74 21.78 22.81 23.85 24.89
        # \Huge         19.90 21.15 22.39 23.64 24.88 26.12 27.37 28.61 29.86
        # \HUGE         24.05 25.55 27.05 28.56 30.06 31.56 33.07 34.57 36.07

        plt.rc('grid',
               color='gainsboro',
               linewidth=0.5 * plt.rcParams['grid.linewidth']  # linewidth of grid lines, default: 0.8
        )
        plt.rc('axes.spines', top=False, right=False)  # Visibility of axis spines
        plt.rc('xtick.minor', visible=True)  # set minor x axis ticks visible
        plt.rc('ytick.minor', visible=True)  # set minor y axis ticks visible
        plt.rc('font', family='serif')  # label fonts, ['serif', >'sans-serif', 'monospace']

        # plt.rc('pgf', rcfonts=False)  # set up fonts from rcparams, default: True
        plt.rc('text', usetex=True)  # use inline math for ticks, default: False


        # color palettes
        self.cp2: dict[str: str] = {  # colorblindness palette
            'magenta': '#d01b88',
            'blue': '#254796',
            'green': '#307b3b',
            'gold': '#caa023',
            'white': '#ffffff',
            'black': '#000000',
        }

        self.cp3: dict[str: str] = {  # uni branding
            'red1': '#9d2246',
            'red2': '#d50c2f',
            'red3': '#f39ca9',
            'blue1': '#00326d',
            'blue2': '#0068b4',
            'blue3': '#89b4e1',
            'purple1': '#3b296a',
            'purple2': '#8681b1',
            'purple3': '#c7c1e1',
            'peach1': '#d45b65',
            'peach2': '#f4a198',
            'peach3': '#fbdad2',
            'orange1': '#f7a600',
            'orange2': '#fece43',
            'orange3': '#ffe7b6',
            'green1': '#008878',
            'green2': '#8acbb7',
            'green3': '#d6ebe1',
            'yellow1': '#dedc00',
            'yellow2': '#f6e945',
            'yellow3': '#fff8bd',
            'white': '#ffffff',
            'black': '#000000',
        }

        self.cp4 = {  # pastel ice cream
            'brown': '#D3BBA7',
            'grey': '#C5CDD3',
            'red': '#ECC0C1',
            'orange': '#FFDCCC',
            'vanilla': '#FFF9EC',
            'mint': '#d8e2dc',
            'blue': '#C6DEF1',
            'white': '#ffffff',
            'black': '#000000',
        }

        self.cp5 = {  # pastel rainbow
            'brown': '#E2CFC4',
            'orange': '#F7D9C4',
            'yellow': '#FAEDCB',
            'green': '#C9E4DE',
            'blue': '#C6DEF1',
            'purple': '#DBCDF0',
            'pink': '#F2C6DE',
            'red': '#F9C6C9',
            'light_grey': '#E2E2DF',
            'dark_grey': '#D2D2CF',
            'white': '#ffffff',
            'black': '#000000',
        }

        # self.textwidth = 5.80910486111  # for masters thesis, in latex inches
        self.textwidth = 3.50  # ieee 2 column paper textwidth, in inches

    @staticmethod
    def save_figures(
            plot_name,
            padding,
    ) -> None:
        plt.savefig(f'plots/pgf/{plot_name}.pgf',
                    bbox_inches='tight', pad_inches=padding, transparent=True)
        plt.savefig(f'plots/pdf/{plot_name}.pdf',
                    bbox_inches='tight', pad_inches=padding, dpi=800, transparent=True)
        plt.savefig(f'plots/eps/{plot_name}.eps',
                    bbox_inches='tight', pad_inches=padding)

    @staticmethod
    def plot_color_palette(
            color_palette: dict
    ) -> None:
        list_of_colors = list(color_palette.keys())

        plt.figure()
        for ii in range(len(list_of_colors)):
            plt.barh(ii, 10, color=color_palette[list_of_colors[ii]])

        plt.yticks(range(len(list_of_colors)), list_of_colors)
        plt.grid(alpha=.25)
        plt.show()


def pt_to_inches(
        pt: float
) -> float:
    return 0.01389 * pt
