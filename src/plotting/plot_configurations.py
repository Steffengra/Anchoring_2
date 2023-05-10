

class PlotContentConfigs:

    def __init__(self):
        color_other = 'white'
        color_gem = 'green1'
        color_ewc = 'orange1'
        color_augmented = 'white'

        self.contents_baseline = {
            'plot_title': 'Baseline',
            'plot_title_alt': 'Naive - No adjustments',
            'stem': 'baseline',
            'affix': '0.0001_base',
            'color': color_other,
        }

        self.contents_critical = {
            'plot_title': 'Prio. Only',
            'plot_title_alt': 'Minimize Prio. Timeouts',
            'stem': '512_experiences',
            'affix': '1.0_critical',
            'color': color_other,
        }

        self.contents_random = {
            'plot_title': 'Random',
            'stem': 'random',
            'affix': 'random_scheduler',
            'color': color_other,
        }

        self.contents_gem512 = {
            'plot_title': 'GEM $ K = 2^{9} $',
            'plot_title_alt': 'GEM 1',
            'stem': '512_experiences',
            'affix': '0.0001_gem',
            'color': color_gem,
        }
        self.contents_gem8138 = {
            'plot_title': 'GEM $ K = 2^{13} $',
            'plot_title_alt': 'GEM 2',
            'stem': '8138_experiences',
            'affix': '0.0001_gem',
            'color': color_gem,
        }
        self.contents_gem32768 = {
            'plot_title': 'GEM $ K = 2^{15} $',
            'plot_title_alt': 'GEM 3',
            'stem': '32768_experiences',
            'affix': '0.0001_gem',
            'color': color_gem,
        }
        self.contents_gem65536 = {
            'plot_title': 'GEM $ K = 2^{16} $',
            'plot_title_alt': 'GEM 4',
            'stem': '65536_experiences',
            'affix': '0.0001_gem',
            'color': color_gem,
        }

        self.contents_gem512cont = {
            'plot_title': '+GEM $ K = 2^{9} $',
            'plot_title_alt': '+GEM 1',
            'stem': '512_experiences',
            'affix': '0.0_twice',
            'color': color_gem,
        }
        self.contents_gem8138cont = {
            'plot_title': '+GEM $ K = 2^{13} $',
            'plot_title_alt': '+GEM 2',
            'stem': '8138_experiences',
            'affix': '0.0_twice',
            'color': color_gem,
        }
        self.contents_gem32768cont = {
            'plot_title': '+GEM $ K = 2^{15} $',
            'plot_title_alt': '+GEM 3',
            'stem': '32768_experiences',
            'affix': '0.0_twice',
            'color': color_gem,
        }
        self.contents_gem65536cont = {
            'plot_title': '+GEM $ K = 2^{16} $',
            'plot_title_alt': 'Continue on new data: GEM 4',
            'stem': '65536_experiences',
            'affix': '0.0_twice',
            'color': color_gem,
        }

        self.contents_ewce5 = {
            'plot_title': 'EWC $ \eta = 1e5 $',
            'plot_title_alt': 'EWC 1',
            'stem': 'anchoring_1e5',
            'affix': '0.0001_crit_events_anchored_1.0_crit_events_base_pretrained',
            'color': color_ewc,
        }
        self.contents_ewce6 = {
            'plot_title': 'EWC $ \eta = 1e6 $',
            'plot_title_alt': 'EWC 2',
            'stem': 'anchoring_1e6',
            'affix': '0.0001_crit_events_anchored_1.0_crit_events_base_pretrained',
            'color': color_ewc,
        }
        self.contents_ewce7 = {
            'plot_title': 'EWC $ \eta = 1e7 $',
            'plot_title_alt': 'EWC 3',
            'stem': 'anchoring_1e7',
            'affix': '0.0001_crit_events_anchored_1.0_crit_events_base_pretrained',
            'color': color_ewc,
        }

        self.contents_ewce5cont = {
            'plot_title': '+EWC $ \eta = 1e5 $',
            'plot_title_alt': '+EWC 1',
            'stem': 'anchoring_1e5',
            'affix': 'continued_0.0_crit_events_anchored_1.0',
            'color': color_ewc,
        }
        self.contents_ewce6cont = {
            'plot_title': '+EWC $ \eta = 1e6 $',
            'plot_title_alt': '+EWC 2',
            'stem': 'anchoring_1e6',
            'affix': 'continued_0.0_crit_events_anchored_1.0',
            'color': color_ewc,
        }
        self.contents_ewce7cont = {
            'plot_title': '+EWC $ \eta = 1e7 $',
            'plot_title_alt': 'Continue on new data: EWC 3',
            'stem': 'anchoring_1e7',
            'affix': 'continued_0.0_crit_events_anchored_1.0',
            'color': color_ewc,
        }

        self.contents_aug20 = {
            'plot_title': 'Aug. 20',
            'plot_title_alt': 'SotA - Change sample distribution',
            'stem': 'augmented',
            'affix': '0.2_crit_events_base',
            'color': color_augmented,
        }
        self.contents_aug50 = {
            'plot_title': 'Aug. 50',
            'stem': 'augmented',
            'affix': '0.5_crit_events_base',
            'color': color_augmented,
        }

        self.contents_aug20cont = {
            'plot_title': '+Aug. 20',
            'plot_title_alt': 'Continue on new data: SotA',
            'stem': 'augmented',
            'affix': 'continued_0.0_crit_events_pretrained_twenty',
            'color': color_augmented,
        }
