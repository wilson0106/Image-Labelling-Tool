#!/usr/bin/env python3
import json
import os
import queue
import re
import sys
from pathlib import Path
import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox, ttk

try:
    from PIL import Image, ImageTk
except Exception:
    Image = None
    ImageTk = None

from platform_utils import application_dir, resource_dir
from project_settings import DEFAULT_PIXEL_SIZE_UM_PER_PX, format_pixel_size, parse_pixel_size
from ui_resources import APP_TEXT as T, COLORS
from ui_contrast import enforce_readable_tree, readable_text_on
from dashboard_widgets import GradientBar, RoundedPanel, ScrollableFrame, TrainingPlotCard
from dashboard_execution_mixin import ILTProcessMixin
from mascot_games_mixin import MascotGamesMixin
from training_dataset import DATASET_MODE_INVALID
from training_dataset_ui_mixin import TrainingDatasetUIMixin
from training_recipe import DEFAULT_RECIPE, PRESETS, RECIPE_KEYS, recipe_from_config

try:
    from aux import defaults as ILT_DEFAULTS
except Exception:
    ILT_DEFAULTS = {
        'BATCH_MAX_SIZE': 8000,
        'img_mode': 'auto',
        'base_tsne_id': 'batch_0001',
        'inner_folder': 'samples',
        'pixel_size': DEFAULT_PIXEL_SIZE_UM_PER_PX,
    }


class ILTApp(MascotGamesMixin, TrainingDatasetUIMixin, ILTProcessMixin, tk.Tk):
    def __init__(self):
        super().__init__()
        self.src_dir = Path(__file__).resolve().parent
        self.project_root = application_dir()
        self.input_dir = self.project_root / 'input'
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = self.project_root / 'output'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        external_assets = self.project_root / 'assets' / 'mascots'
        bundled_assets = resource_dir() / 'assets' / 'mascots'
        # An empty external folder must not shadow mascots bundled by PyInstaller.
        # This matters on Windows, where the distribution creates assets/mascots
        # even when the actual PNGs live in the packaged resources.
        candidates=(external_assets,bundled_assets)
        self.assets_dir=next(
            (d for d in candidates if d.is_dir() and any(d.glob('*.png'))),
            external_assets
        )

        external_branding = self.project_root / 'assets' / 'branding'
        bundled_branding = resource_dir() / 'assets' / 'branding'
        branding_candidates = (external_branding, bundled_branding)
        self.branding_dir = next((d for d in branding_candidates if d.is_dir()), external_branding)
        self._brand_images = {}
        self._brand_icons = {}
        self._brand_page_icons = {}

        self.lang = 'pt'
        self.font_scale = 1.0
        self._font_base = {}
        self._mascot_photo = None
        self._mascot_current = None
        self._mascot_click_job = None
        self._memory_win = None
        self._memory_levels = (3, 4, 5, 6)
        self._hunt_win = None
        self._hunt_round_job = None
        self._last_mascot_game = None
        self.current_task = 'training'
        self.callbacks = []
        self.proc = None
        self.proc_queue = queue.Queue()
        self.proc_task = None
        self.start_time = None
        self.last_output_path = None
        self.temp_train_json = None
        self.temp_train_dataset = None
        self._train_dataset_refresh_job = None
        self._train_class_dataset_key = None
        self._train_dataset_mode = DATASET_MODE_INVALID
        self._train_class_counts = {}
        self._train_class_totals = {}
        self._train_class_map_vars = {}
        self._train_class_use_vars = {}
        self._train_class_allocations = {}
        self._train_class_percentages = {}
        self._train_class_split_vars = {}
        self._train_class_unused_vars = {}
        self._train_class_allocation_errors = {}
        self._train_allocation_sync = False
        self._train_distribution_mode_current = 'percent'
        self._train_distribution_default_cache = {
            'percent': {'train': '70', 'valid': '15', 'test': '15'},
            'number': {'train': '0', 'valid': '0', 'test': '0'},
        }
        self._train_class_rows_host = None
        self._train_model_tree = None
        self._train_model_summary_var = None
        self._train_dataset_detection_var = None
        self._train_distribution_unit_var = None
        self.last_split_manifest_path = None
        self.class_monitor_info = {}
        self._stopping = False
        self.run_class_clustering_after = True
        self.clustering_total = 0
        self.clustering_seen = 0

        self.title('ILT Data Extraction & IAT')
        self.geometry('1540x900')
        self.minsize(1280, 760)
        self.configure(bg=COLORS['bg'])
        self.protocol('WM_DELETE_WINDOW', self.on_close)
        self._style()
        self._init_vars()
        self._load_branding_assets()
        self._apply_window_icon()
        self._build_shell()
        # Callbacks registered after this point belong to the current page and
        # are discarded when the page is rebuilt (see show_task).
        self._shell_callback_count = len(self.callbacks)
        self.show_task('training')
        self.after(100, self._poll_process_queue)
        self.after(500, self._tick)

    def _style(self):
        style = ttk.Style(self)
        style.theme_use('clam')
        style.configure('Dark.TCombobox', fieldbackground=COLORS['entry'], background=COLORS['entry'], foreground=COLORS['text'], arrowcolor=COLORS['text'], bordercolor=COLORS['border'])
        style.map('Dark.TCombobox', fieldbackground=[('readonly', COLORS['entry'])], foreground=[('readonly', COLORS['text'])])
        style.configure('Blue.Horizontal.TProgressbar', troughcolor='#202024', background=COLORS['blue'], bordercolor='#202024', lightcolor=COLORS['blue'], darkcolor=COLORS['blue'])
        style.configure('Green.Horizontal.TProgressbar', troughcolor='#202024', background=COLORS['green'], bordercolor='#202024', lightcolor=COLORS['green'], darkcolor=COLORS['green'])
        style.configure(
            'Dark.Treeview',
            background=COLORS['card2'], fieldbackground=COLORS['card2'],
            foreground=COLORS['text'], bordercolor=COLORS['border'],
            rowheight=24, font=('Sans', 8)
        )
        style.map('Dark.Treeview', background=[('selected', COLORS['nav_active'])], foreground=[('selected', '#ffffff')])
        style.configure(
            'Dark.Treeview.Heading',
            background=COLORS['entry'], foreground='#d4d4d8',
            bordercolor=COLORS['border'], relief='flat', font=('Sans', 8, 'bold')
        )
        style.map('Dark.Treeview.Heading', background=[('active', COLORS['border'])])

    def _load_train_defaults(self):
        defaults = {
            'lr': 0.00001, 'lossName': 'focal', 'focalGamma': 2,
            'trainFolder': 'train', 'validFolder': 'valid', 'testFolder': 'test',
            'readModel': 'T', 'numEpochs': 500, 'validMetric': 'valid_macro',
            'patience': 10, 'numClasses': -1, 'numWorkers': -1, 'seed': 0,
            'batchSize': 30,
            'pixelSize': float(ILT_DEFAULTS.get('pixel_size', DEFAULT_PIXEL_SIZE_UM_PER_PX)),
            **DEFAULT_RECIPE,
        }
        p = self.input_dir / 'train_params.json'
        if p.exists():
            try:
                defaults.update(json.loads(p.read_text()))
            except Exception:
                pass
        return defaults

    def _init_vars(self):
        cfg = self._load_train_defaults()
        output_default = str(self.output_dir)
        self.v = {
            'train_dataset': tk.StringVar(),
            'train_output': tk.StringVar(value=output_default),
            'train_project': tk.StringVar(value='meu_projeto_treino'),
            'numEpochs': tk.StringVar(value=str(cfg['numEpochs'])),
            'batchSize': tk.StringVar(value=str(cfg.get('batchSize', 30))),
            'lr': tk.StringVar(value=str(cfg['lr'])),
            'lossName': tk.StringVar(value=str(cfg['lossName'])),
            'focalGamma': tk.StringVar(value=str(cfg['focalGamma'])),
            'patience': tk.StringVar(value=str(cfg['patience'])),
            'validMetric': tk.StringVar(value=str(cfg['validMetric'])),
            'seed': tk.StringVar(value=str(cfg['seed'])),
            'numClasses': tk.StringVar(value=str(cfg['numClasses'])),
            'trainFolder': tk.StringVar(value=str(cfg['trainFolder'])),
            'validFolder': tk.StringVar(value=str(cfg['validFolder'])),
            'testFolder': tk.StringVar(value=str(cfg['testFolder'])),
            'numWorkers': tk.StringVar(value=str(cfg['numWorkers'])),
            'readModel': tk.StringVar(value=str(cfg['readModel'])),
            'pixelSize': tk.StringVar(value=format_pixel_size(cfg.get('pixelSize', DEFAULT_PIXEL_SIZE_UM_PER_PX))),
            'class_input': tk.StringVar(),
            'class_output_parent': tk.StringVar(value=output_default),
            'class_project': tk.StringVar(value='meu_projeto_classificacao'),
            'model_folder': tk.StringVar(),
            'model_file': tk.StringVar(),
            'labels_file': tk.StringVar(),
            'max_batch': tk.StringVar(value=str(ILT_DEFAULTS.get('BATCH_MAX_SIZE', 8000))),
            'img_mode': tk.StringVar(value=str(ILT_DEFAULTS.get('img_mode', 'auto'))),
            'class_pixel_size': tk.StringVar(value=format_pixel_size(ILT_DEFAULTS.get('pixel_size', DEFAULT_PIXEL_SIZE_UM_PER_PX))),
            'run_class_clustering': tk.BooleanVar(value=True),
            'audit_source_labels': tk.BooleanVar(value=False),
        }
        # Training recipe (training_recipe.py); 'augment' is kept as 'T'/'F'
        # like 'readModel' so both interfaces can use a two-value selector.
        for key in RECIPE_KEYS:
            self.v[key] = tk.StringVar(value=self._recipe_var_text(key, cfg.get(key, DEFAULT_RECIPE[key])))
        self._train_distribution_mode_var = tk.StringVar(value='percent')
        self._train_distribution_default_vars = {
            'train': tk.StringVar(value='70'),
            'valid': tk.StringVar(value='15'),
            'test': tk.StringVar(value='15'),
        }
        self._train_distribution_mode_var.trace_add(
            'write', self._on_train_distribution_mode_changed
        )
        for key in ('train_dataset', 'trainFolder', 'validFolder', 'testFolder'):
            self.v[key].trace_add('write', self._schedule_train_dataset_refresh)

    def tr(self, key):
        return T[self.lang].get(key, key)

    def register_callback(self, callback):
        self.callbacks.append(callback)

    def _text(self, parent, key, **kwargs):
        var = tk.StringVar(value=self.tr(key))
        label = tk.Label(parent, textvariable=var, **kwargs)
        self.register_callback(lambda v=var, k=key: v.set(self.tr(k)))
        return label

    def _button(self, parent, key, command, kind='secondary', **kwargs):
        var = tk.StringVar(value=self.tr(key))
        bg = COLORS['green'] if kind == 'primary' else COLORS['card2']
        if kind == 'danger': bg = '#5b2630'
        # Text colour follows the background: white on the bright primary
        # green was only 2.1:1; dark ink reaches 9:1 (WCAG AA needs 4.5:1).
        fg = readable_text_on(bg)
        active_bg = COLORS['green2'] if kind == 'primary' else COLORS['border']
        b = tk.Button(parent, textvariable=var, command=command, bg=bg, fg=fg,
                      activebackground=active_bg,
                      activeforeground=readable_text_on(active_bg),
                      disabledforeground=COLORS['disabled'] if kind != 'primary' else '#2f5a43',
                      relief='flat', bd=0, padx=14, pady=8,
                      cursor='hand2', highlightthickness=0, font=('Sans', 10, 'bold' if kind == 'primary' else 'normal'), **kwargs)
        self.register_callback(lambda v=var, k=key: v.set(self.tr(k)))
        return b

    def _build_shell(self):
        self.sidebar = tk.Frame(self, bg=COLORS['sidebar'], width=230)
        self.sidebar.pack(side='left', fill='y')
        self.sidebar.pack_propagate(False)

        brand = tk.Frame(self.sidebar, bg=COLORS['sidebar'])
        brand.pack(fill='x', padx=14, pady=(16, 18))
        if self._brand_images.get('global_sidebar') is not None:
            tk.Label(brand, image=self._brand_images['global_sidebar'], bg=COLORS['sidebar']).pack(side='left')
        else:
            tk.Label(brand, text='◈', bg=COLORS['sidebar'], fg=COLORS['blue'], font=('Sans', 20, 'bold')).pack(side='left')
        brand_text = tk.Frame(brand, bg=COLORS['sidebar'])
        brand_text.pack(side='left', padx=8)
        tk.Label(brand_text, text='ILT Data Extraction', bg=COLORS['sidebar'], fg=COLORS['text'], font=('Sans', 11, 'bold')).pack(anchor='w')
        tk.Label(brand_text, text='Treino • Classificação • IAT', bg=COLORS['sidebar'], fg=COLORS['muted'], font=('Sans', 7)).pack(anchor='w')

        self._text(self.sidebar, 'tasks', bg=COLORS['sidebar'], fg=COLORS['muted'], font=('Sans', 8, 'bold')).pack(anchor='w', padx=16, pady=(0, 6))
        self.nav_train = self._nav_button('training', 'training')
        self.nav_class = self._nav_button('classification', 'classification')
        self.nav_iat = self._nav_button('iat', 'iat')
        self.iat_subnav_frame = tk.Frame(self.sidebar, bg=COLORS['sidebar'])
        self.iat_subnav_buttons = {}
        for tab, key in [('scatter','iat_scatter'),('gallery','iat_gallery'),('temporal','iat_temporal'),('clustering','iat_clustering'),('model_cycle','iat_model_cycle'),('segmentation','iat_segmentation'),('export','iat_export'),('logs','iat_logs')]:
            var = tk.StringVar(value=self.tr(key))
            btn = tk.Button(self.iat_subnav_frame, textvariable=var, command=lambda t=tab: self._show_iat_tab(t), anchor='w',
                            bg=COLORS['sidebar'], fg=COLORS['muted'], activebackground='#4a2779', activeforeground='#fff',
                            relief='flat', bd=0, highlightthickness=0, padx=25, pady=6, cursor='hand2', font=('Sans', 8))
            btn.pack(fill='x', padx=8, pady=1)
            self.iat_subnav_buttons[tab] = btn
            self.register_callback(lambda v=var, k=key: v.set(self.tr(k)))

        self._text(self.sidebar, 'system', bg=COLORS['sidebar'], fg=COLORS['muted'], font=('Sans', 8, 'bold')).pack(anchor='w', padx=16, pady=(26, 6))
        self._button(self.sidebar, 'old_logs', self.open_old_log).pack(fill='x', padx=10, pady=2)
        self._button(self.sidebar, 'settings', self.show_settings).pack(fill='x', padx=10, pady=2)
        self._button(self.sidebar, 'about', self.show_about).pack(fill='x', padx=10, pady=2)
        self._build_mascot()

        self.system_box = tk.Frame(self.sidebar, bg=COLORS['card2'], bd=0, highlightthickness=0)
        self.system_box.pack(side='bottom', fill='x', padx=12, pady=14)
        self.system_status_var = tk.StringVar(value=self.tr('ready'))
        tk.Label(self.system_box, text='●', bg=COLORS['card2'], fg=COLORS['green']).pack(side='left', padx=(10, 5), pady=10)
        tk.Label(self.system_box, textvariable=self.system_status_var, bg=COLORS['card2'], fg=COLORS['text'], font=('Sans', 9, 'bold')).pack(side='left', pady=10)

        self.main_area = tk.Frame(self, bg=COLORS['bg'])
        self.main_area.pack(side='left', fill='both', expand=True)

        top = tk.Frame(self.main_area, bg=COLORS['panel'], height=44)
        top.pack(fill='x')
        top.pack_propagate(False)
        top_brand = tk.Frame(top, bg=COLORS['panel'])
        top_brand.pack(side='left', padx=12)
        if self._brand_images.get('global_top') is not None:
            tk.Label(top_brand, image=self._brand_images['global_top'], bg=COLORS['panel']).pack(side='left', padx=(0, 8))
        tk.Label(top_brand, text='ILT Data Extraction & IAT', bg=COLORS['panel'], fg=COLORS['text'], font=('Sans', 11, 'bold')).pack(side='left')

        lang = tk.Frame(top, bg=COLORS['panel'])
        lang.pack(side='right', padx=18)
        tk.Label(lang, text='PT', bg=COLORS['panel'], fg=COLORS['muted'], font=('Sans', 9, 'bold')).pack(side='left', padx=(0, 6))
        self.lang_canvas = tk.Canvas(lang, width=48, height=24, bg=COLORS['panel'], highlightthickness=0, cursor='hand2')
        self.lang_canvas.pack(side='left')
        self.lang_canvas.bind('<Button-1>', lambda e: self.toggle_language())
        tk.Label(lang, text='EN', bg=COLORS['panel'], fg=COLORS['muted'], font=('Sans', 9, 'bold')).pack(side='left', padx=(6, 0))
        self._draw_language_switch()
        self._build_font_controls(top)

        GradientBar(self.main_area, COLORS['grad_a'], COLORS['grad_b'], height=2).pack(fill='x')

        self.page_host = tk.Frame(self.main_area, bg=COLORS['bg'])
        self.page_host.pack(fill='both', expand=True)

    def _nav_button(self, task, key):
        var = tk.StringVar(value=self.tr(key))
        icon = self._brand_icons.get(task)
        b = tk.Button(
            self.sidebar,
            textvariable=var,
            image=icon,
            compound='left' if icon is not None else 'none',
            command=lambda: self.show_task(task),
            anchor='w',
            bg=COLORS['sidebar'],
            fg=COLORS['text'],
            activebackground=COLORS['nav_active'],
            activeforeground='#fff',
            relief='flat',
            bd=0,
            highlightthickness=0,
            padx=14,
            pady=10,
            cursor='hand2',
            font=('Sans', 10),
        )
        b.pack(fill='x', padx=8, pady=2)
        self.register_callback(lambda v=var, k=key: v.set(self.tr(k)))
        return b


    def _load_branding_assets(self):
        """Load the visual identity without making the GUI depend on the assets."""
        if Image is None or ImageTk is None or not self.branding_dir.is_dir():
            return

        icon_files = {
            'global': self.branding_dir / 'icon_global.png',
            'training': self.branding_dir / 'icon_training.png',
            'classification': self.branding_dir / 'icon_classification.png',
            'iat': self.branding_dir / 'icon_iat.png',
        }

        for key, path in icon_files.items():
            if not path.is_file():
                continue
            try:
                source = Image.open(path).convert('RGBA')

                nav = source.copy()
                nav.thumbnail((24, 24), Image.Resampling.LANCZOS)
                self._brand_icons[key] = ImageTk.PhotoImage(nav)

                page = source.copy()
                page.thumbnail((32, 32), Image.Resampling.LANCZOS)
                self._brand_page_icons[key] = ImageTk.PhotoImage(page)

                if key == 'global':
                    side = source.copy()
                    side.thumbnail((34, 34), Image.Resampling.LANCZOS)
                    self._brand_images['global_sidebar'] = ImageTk.PhotoImage(side)

                    top = source.copy()
                    top.thumbnail((26, 26), Image.Resampling.LANCZOS)
                    self._brand_images['global_top'] = ImageTk.PhotoImage(top)
            except Exception:
                # Branding must never prevent the scientific application from opening.
                pass

        # Keep the complete logo lockups loaded for places that can use a larger mark
        # (e.g. About / future splash screens) while the compact UI uses simplified icons.
        for key in ('global', 'training', 'classification', 'iat'):
            path = self.branding_dir / f'logo_{key}.png'
            if not path.is_file():
                continue
            try:
                logo = Image.open(path).convert('RGBA')
                logo.thumbnail((420, 105), Image.Resampling.LANCZOS)
                self._brand_images[f'logo_{key}'] = ImageTk.PhotoImage(logo)
            except Exception:
                pass

    def _apply_window_icon(self):
        """Apply the global icon to Tk and, on Windows, to the native window."""
        if Image is None or ImageTk is None or not self.branding_dir.is_dir():
            return

        ico = self.branding_dir / 'app_icon.ico'
        png = self.branding_dir / 'icon_global.png'

        if sys.platform.startswith('win') and ico.is_file():
            try:
                self.iconbitmap(default=str(ico))
            except Exception:
                pass

        if png.is_file():
            try:
                source = Image.open(png).convert('RGBA')
                source.thumbnail((64, 64), Image.Resampling.LANCZOS)
                self._window_icon_photo = ImageTk.PhotoImage(source)
                self.iconphoto(True, self._window_icon_photo)
            except Exception:
                pass


    def _disabled_nav(self, key):
        frame = tk.Frame(self.sidebar, bg=COLORS['sidebar'])
        frame.pack(fill='x', padx=8, pady=2)
        label = self._text(frame, key, bg=COLORS['sidebar'], fg=COLORS['disabled'], font=('Sans', 10))
        label.pack(side='left', padx=14, pady=9)
        soon = self._text(frame, 'coming_soon', bg=COLORS['sidebar'], fg=COLORS['disabled'], font=('Sans', 7))
        soon.pack(side='right', padx=8)

    def _draw_language_switch(self):
        c = self.lang_canvas
        c.delete('all')
        c.create_rectangle(7, 4, 41, 20, fill=COLORS['blue'], outline=COLORS['blue'])
        c.create_oval(1, 4, 17, 20, fill=COLORS['blue'], outline=COLORS['blue'])
        c.create_oval(31, 4, 47, 20, fill=COLORS['blue'], outline=COLORS['blue'])
        x = 9 if self.lang == 'pt' else 31
        c.create_oval(x-6, 5, x+8, 19, fill='#ffffff', outline='#ffffff')


    def _build_font_controls(self, parent):
        box = tk.Frame(parent, bg=COLORS['panel'])
        box.pack(side='right', padx=(4, 10))
        self._text(box, 'text_size', bg=COLORS['panel'], fg=COLORS['muted'], font=('Sans', 8)).pack(side='left', padx=(0, 6))
        minus = tk.Button(box, text='A−', command=lambda: self.change_font_size(-0.1), bg=COLORS['card2'], fg=COLORS['text'],
                          activebackground=COLORS['border'], activeforeground='#fff', relief='flat', bd=0,
                          padx=8, pady=3, cursor='hand2', highlightthickness=0, font=('Sans', 9, 'bold'))
        minus.pack(side='left')
        self.font_scale_var = tk.StringVar(value='100%')
        tk.Label(box, textvariable=self.font_scale_var, bg=COLORS['panel'], fg=COLORS['text'], width=5,
                 font=('Sans', 8, 'bold')).pack(side='left', padx=4)
        plus = tk.Button(box, text='A+', command=lambda: self.change_font_size(0.1), bg=COLORS['card2'], fg=COLORS['text'],
                         activebackground=COLORS['border'], activeforeground='#fff', relief='flat', bd=0,
                         padx=8, pady=3, cursor='hand2', highlightthickness=0, font=('Sans', 9, 'bold'))
        plus.pack(side='left')

    def change_font_size(self, delta):
        self.font_scale = max(0.8, min(1.4, round(self.font_scale + delta, 1)))
        if hasattr(self, 'font_scale_var'):
            self.font_scale_var.set(f'{int(round(self.font_scale * 100))}%')
        self._apply_font_scale()

    def _apply_font_scale(self):
        if self.font_scale == 1.0 and not self._font_base:
            # Nothing has ever been rescaled: widgets already use base sizes.
            return
        seen = set()

        def walk(widget):
            try:
                current = widget.cget('font')
            except Exception:
                current = None
            if current:
                key = str(widget)
                seen.add(key)
                if key not in self._font_base:
                    try:
                        f = tkfont.Font(font=current)
                        a = f.actual()
                        size = abs(int(a.get('size', 10))) or 10
                        self._font_base[key] = (a.get('family', 'Sans'), size, a.get('weight', 'normal'), a.get('slant', 'roman'))
                    except Exception:
                        self._font_base[key] = ('Sans', 9, 'normal', 'roman')
                family, size, weight, slant = self._font_base[key]
                new_size = max(7, int(round(size * self.font_scale)))
                try:
                    widget.configure(font=(family, new_size, weight, slant))
                except Exception:
                    pass
            for child in widget.winfo_children():
                walk(child)
        walk(self)
        # Forget destroyed widgets (pages are rebuilt on every navigation).
        self._font_base = {k: v for k, v in self._font_base.items() if k in seen}
        try:
            ttk.Style(self).configure('Dark.TCombobox', font=('Sans', max(8, int(round(9 * self.font_scale)))))
        except Exception:
            pass
        for chart_name in ('loss_chart', 'validation_chart'):
            chart = getattr(self, chart_name, None)
            if chart is not None:
                try:
                    if chart.winfo_exists():
                        chart._draw()
                except Exception:
                    pass

    def _build_mascot(self):
        self.mascot_box = tk.Frame(self.sidebar, bg=COLORS['sidebar'])
        self.mascot_box.pack(fill='x', padx=20, pady=(18, 8))
        self.mascot_label = tk.Label(self.mascot_box, bg=COLORS['sidebar'], bd=0, cursor='hand2')
        self.mascot_label.pack()
        # Delay the single-click action slightly so a double click does not
        # change the mascot immediately before opening the memory game.
        self.mascot_label.bind('<Button-1>', self._on_mascot_single_click)
        self.mascot_label.bind('<Double-Button-1>', self._on_mascot_double_click)
        self._show_random_mascot()


    # -------- mascot memory game --------


    # -------- mascot "Where's Waldo"-style hunt --------


    def toggle_language(self):
        self.lang = 'en' if self.lang == 'pt' else 'pt'
        self._draw_language_switch()
        for cb in list(self.callbacks):
            try: cb()
            except Exception: pass
        self._refresh_dynamic_texts()

    def show_task(self, task):
        if self.proc is not None:
            return
        if self.current_task == 'iat' and hasattr(self, 'iat_workspace') and self.iat_workspace.winfo_exists() and getattr(self.iat_workspace, '_op_running', False):
            return
        self.current_task = task
        for w in self.page_host.winfo_children():
            w.destroy()
        # Drop language callbacks of the destroyed page; they referenced dead
        # widgets and accumulated on every navigation.
        del self.callbacks[getattr(self, '_shell_callback_count', len(self.callbacks)):]
        self.nav_train.configure(bg=COLORS['nav_active'] if task == 'training' else COLORS['sidebar'])
        self.nav_class.configure(bg=COLORS['nav_active'] if task == 'classification' else COLORS['sidebar'])
        self.nav_iat.configure(bg=COLORS['nav_active'] if task == 'iat' else COLORS['sidebar'])
        if task == 'iat':
            # Performance-only lazy import: the IAT stack pulls in NumPy/Pandas/
            # Matplotlib and optional scientific packages.  Loading it here keeps
            # Treino/Classificação startup fast while preserving IAT behaviour.
            from iat_workspace import IATWorkspace
            self.iat_subnav_frame.pack(fill='x', after=self.nav_iat, padx=0, pady=(0, 4))
            self.iat_workspace = IATWorkspace(self.page_host, self)
            self.iat_workspace.pack(fill='both', expand=True)
            self.set_iat_subnav_active('scatter')
        else:
            self.iat_subnav_frame.pack_forget()
            if task == 'training':
                self._build_training_page()
            else:
                self._build_classification_page()
        self.after_idle(self._apply_font_scale)
        self.after_idle(self._enforce_contrast)

    def _enforce_contrast(self):
        try:
            enforce_readable_tree(self)
        except Exception:
            pass

    def _show_iat_tab(self, tab):
        if self.current_task == 'iat' and hasattr(self, 'iat_workspace') and getattr(self.iat_workspace, '_op_running', False):
            origin=getattr(self.iat_workspace,'_operation_origin_tab','scatter')
            if tab not in {'logs', origin}:
                return
        if self.current_task != 'iat':
            self.show_task('iat')
        if hasattr(self, 'iat_workspace') and self.iat_workspace.winfo_exists():
            self.iat_workspace.show_tab(tab)

    def set_iat_subnav_active(self, tab):
        for name, btn in getattr(self, 'iat_subnav_buttons', {}).items():
            btn.configure(bg='#4a2779' if name == tab else COLORS['sidebar'],
                          fg=COLORS['text'] if name == tab else COLORS['muted'])

    def _page_layout(self, title_key, subtitle_key):
        outer = tk.Frame(self.page_host, bg=COLORS['bg'])
        outer.pack(fill='both', expand=True, padx=14, pady=14)
        left = tk.Frame(outer, bg=COLORS['panel'], bd=0, highlightthickness=0)
        left.pack(side='left', fill='both', expand=True, padx=(0, 7))
        right = tk.Frame(outer, bg=COLORS['panel'], width=500, bd=0, highlightthickness=0)
        right.pack(side='right', fill='y', padx=(7, 0))
        right.pack_propagate(False)

        head = tk.Frame(left, bg=COLORS['panel'])
        head.pack(fill='x', padx=20, pady=(18, 10))
        head_row = tk.Frame(head, bg=COLORS['panel'])
        head_row.pack(anchor='w')
        task_map = {'training_title': 'training', 'classification_title': 'classification'}
        page_icon = self._brand_page_icons.get(task_map.get(title_key, ''))
        if page_icon is not None:
            tk.Label(head_row, image=page_icon, bg=COLORS['panel']).pack(side='left', padx=(0, 8))
        self._text(head_row, title_key, bg=COLORS['panel'], fg=COLORS['text'], font=('Sans', 16, 'bold')).pack(side='left')
        self._text(head, subtitle_key, bg=COLORS['panel'], fg=COLORS['muted'], font=('Sans', 9)).pack(anchor='w', pady=(3, 0))

        body = ScrollableFrame(left, COLORS['panel'])
        body.pack(fill='both', expand=True, padx=8, pady=(0, 8))
        return body.inner, right

    def _card(self, parent, title_key, column=0, row=0, columnspan=1, sticky='nsew'):
        rounded = RoundedPanel(parent, bg=COLORS['card'], border=COLORS['border'], radius=16, pad=1)
        rounded.grid(row=row, column=column, columnspan=columnspan, sticky=sticky, padx=7, pady=7)
        card = rounded.inner
        title_row = tk.Frame(card, bg=COLORS['card'])
        title_row.pack(fill='x', padx=14, pady=(11, 6))
        accent = tk.Canvas(title_row, width=5, height=18, bg=COLORS['card'], highlightthickness=0)
        accent.pack(side='left', padx=(0,8))
        accent.create_rectangle(1, 1, 4, 17, fill=COLORS['blue'], outline=COLORS['blue'])
        self._text(title_row, title_key, bg=COLORS['card'], fg='#ffffff', font=('Sans', 10, 'bold')).pack(side='left')
        content = tk.Frame(card, bg=COLORS['card'])
        content.pack(fill='both', expand=True, padx=14, pady=(0, 14))
        return content

    def _field(self, parent, label_key, var, row, col=0, width=18, combo=None, browse=None, readonly=False):
        holder = tk.Frame(parent, bg=COLORS['card'])
        holder.grid(row=row, column=col, sticky='ew', padx=5, pady=5)
        parent.grid_columnconfigure(col, weight=1)
        self._text(holder, label_key, bg=COLORS['card'], fg=COLORS['muted'], font=('Sans', 8)).pack(anchor='w', pady=(0, 4))
        line = tk.Frame(holder, bg=COLORS['card'])
        line.pack(fill='x')
        if combo:
            widget = ttk.Combobox(line, textvariable=var, values=combo, state='readonly', style='Dark.TCombobox', width=width)
        else:
            widget = tk.Entry(line, textvariable=var, bg=COLORS['entry'], fg=COLORS['text'], insertbackground=COLORS['text'],
                              relief='flat', highlightbackground=COLORS['border'], highlightthickness=0, font=('Sans', 9))
            if readonly:
                widget.configure(state='readonly', readonlybackground=COLORS['entry'])
        widget.pack(side='left', fill='x', expand=True, ipady=5)
        if browse:
            b = self._button(line, 'browse', browse)
            b.pack(side='left', padx=(6, 0))
        return widget

    def _build_training_page(self):
        body, right = self._page_layout('training_title', 'training_subtitle')
        body.grid_columnconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=1)

        c1 = self._card(body, 'dataset', 0, 0)
        self._field(c1, 'dataset_root', self.v['train_dataset'], 0, browse=self.pick_train_dataset)
        structure_tip = tk.StringVar(value=self.tr('dataset_structure_help'))
        structure = tk.Label(
            c1, textvariable=structure_tip,
            bg=COLORS['card'], fg=COLORS['muted'], justify='left',
            anchor='w', font=('Monospace', 8)
        )
        structure.grid(row=1, column=0, sticky='w', padx=5, pady=(6, 2))
        self.register_callback(
            lambda v=structure_tip: v.set(self.tr('dataset_structure_help'))
        )
        self._train_dataset_detection_var = tk.StringVar(
            value=self.tr('dataset_detection_wait')
        )
        self._train_dataset_detection_label = tk.Label(
            c1, textvariable=self._train_dataset_detection_var,
            bg=COLORS['card'], fg=COLORS['muted'], justify='left',
            anchor='w', wraplength=620, font=('Sans', 8, 'bold')
        )
        self._train_dataset_detection_label.grid(
            row=2, column=0, sticky='ew', padx=5, pady=(5, 0)
        )
        self.register_callback(self._refresh_train_dataset_detection_text)

        c2 = self._card(body, 'output', 1, 0)
        self._field(c2, 'output_folder', self.v['train_output'], 0, browse=lambda: self.pick_dir(self.v['train_output']))
        self._field(c2, 'project_name', self.v['train_project'], 1)

        # Dataset classes: selection, per-class split allocation, grouping/renaming.
        cclasses = self._card(body, 'dataset_classes', 0, 1, 2)
        classes_tip = tk.StringVar(value=self.tr('dataset_classes_tip'))
        tk.Label(
            cclasses, textvariable=classes_tip, bg=COLORS['card'], fg=COLORS['muted'],
            justify='left', anchor='w', wraplength=1050, font=('Sans', 8)
        ).pack(fill='x', padx=5, pady=(0, 6))
        self.register_callback(lambda v=classes_tip: v.set(self.tr('dataset_classes_tip')))

        class_toolbar = tk.Frame(cclasses, bg=COLORS['card'])
        class_toolbar.pack(fill='x', padx=5, pady=(0, 6))
        self._button(class_toolbar, 'restore_class_names', self._reset_training_class_names).pack(side='left')
        self._button(class_toolbar, 'refresh_dataset_classes', self._refresh_training_dataset_classes).pack(side='right')

        distribution = tk.Frame(cclasses, bg=COLORS['card2'])
        distribution.pack(fill='x', padx=5, pady=(0, 7))

        mode_box = tk.Frame(distribution, bg=COLORS['card2'])
        mode_box.pack(side='left', padx=(8, 14), pady=7)
        self._text(
            mode_box, 'distribution_by', bg=COLORS['card2'], fg=COLORS['muted'],
            font=('Sans', 8, 'bold')
        ).pack(side='left', padx=(0, 8))
        for value, key in (('percent', 'distribution_percent'), ('number', 'distribution_number')):
            label_var = tk.StringVar(value=self.tr(key))
            rb = tk.Radiobutton(
                mode_box, textvariable=label_var,
                variable=self._train_distribution_mode_var, value=value,
                bg=COLORS['card2'], fg=COLORS['text'],
                activebackground=COLORS['card2'], activeforeground=COLORS['text'],
                selectcolor=COLORS['entry'], highlightthickness=0, bd=0,
                font=('Sans', 8), cursor='hand2'
            )
            rb.pack(side='left', padx=(0, 8))
            self.register_callback(lambda v=label_var, k=key: v.set(self.tr(k)))

        defaults = tk.Frame(distribution, bg=COLORS['card2'])
        defaults.pack(side='right', padx=(8, 8), pady=5)
        self._text(
            defaults, 'default_distribution', bg=COLORS['card2'],
            fg=COLORS['muted'], font=('Sans', 8)
        ).pack(side='left', padx=(0, 7))
        self._train_distribution_default_entries = {}
        for split, key in (
            ('train', 'train_short'), ('valid', 'valid_short'), ('test', 'test_short')
        ):
            self._text(
                defaults, key, bg=COLORS['card2'], fg=COLORS['muted'],
                font=('Sans', 7)
            ).pack(side='left', padx=(4, 2))
            ent = tk.Entry(
                defaults, textvariable=self._train_distribution_default_vars[split],
                width=5, justify='center', bg=COLORS['entry'], fg=COLORS['text'],
                insertbackground=COLORS['text'], relief='flat',
                highlightbackground=COLORS['border'], highlightthickness=1,
                font=('Monospace', 8)
            )
            ent.pack(side='left', ipady=2)
            self._train_distribution_default_entries[split] = ent

        self._train_distribution_unit_var = tk.StringVar(value='%')
        tk.Label(
            defaults, textvariable=self._train_distribution_unit_var,
            bg=COLORS['card2'], fg=COLORS['cyan'], width=3, font=('Sans', 8, 'bold')
        ).pack(side='left', padx=(4, 5))
        self.register_callback(
            lambda: self._train_distribution_unit_var.set(
                '%' if self._train_distribution_mode_var.get() == 'percent'
                else self.tr('images_unit')
            )
        )
        self._button(
            defaults, 'apply_distribution_all', self._apply_training_distribution_to_all
        ).pack(side='left', padx=(4, 0))

        header = tk.Frame(cclasses, bg=COLORS['entry'])
        header.pack(fill='x', padx=5)
        headers = [
            ('use_in_training', 5, 'center', False),
            ('original_class', 17, 'w', True),
            ('total_images', 7, 'center', False),
            ('train_short', 7, 'center', False),
            ('valid_short', 7, 'center', False),
            ('test_short', 7, 'center', False),
            ('unused_images', 7, 'center', False),
            ('final_class', 18, 'w', True),
        ]
        for key, width, anchor, stretch in headers:
            var = tk.StringVar(value=self.tr(key))
            tk.Label(
                header, textvariable=var, bg=COLORS['entry'], fg='#d4d4d8',
                width=width, anchor=anchor, font=('Sans', 7, 'bold'), padx=2, pady=5
            ).pack(side='left', fill='x' if stretch else 'none', expand=stretch)
            self.register_callback(lambda v=var, k=key: v.set(self.tr(k)))

        class_wrap = tk.Frame(
            cclasses, bg=COLORS['card2'], highlightbackground=COLORS['border'],
            highlightthickness=1, height=245
        )
        class_wrap.pack(fill='both', expand=True, padx=5, pady=(0, 3))
        class_wrap.pack_propagate(False)
        class_canvas = tk.Canvas(class_wrap, bg=COLORS['card2'], highlightthickness=0, bd=0)
        class_scroll = ttk.Scrollbar(class_wrap, orient='vertical', command=class_canvas.yview)
        class_rows = tk.Frame(class_canvas, bg=COLORS['card2'])
        class_window = class_canvas.create_window((0, 0), window=class_rows, anchor='nw')
        class_rows.bind('<Configure>', lambda _e: class_canvas.configure(scrollregion=class_canvas.bbox('all')))
        class_canvas.bind('<Configure>', lambda e: class_canvas.itemconfigure(class_window, width=e.width))
        class_canvas.configure(yscrollcommand=class_scroll.set)
        class_canvas.pack(side='left', fill='both', expand=True)
        class_scroll.pack(side='right', fill='y')
        self._train_class_rows_host = class_rows

        # Final class preview: this is exactly what train.py will see.
        cpreview = self._card(body, 'model_classes_preview', 0, 2, 2)
        preview_tip = tk.StringVar(value=self.tr('model_classes_preview_tip'))
        tk.Label(
            cpreview, textvariable=preview_tip, bg=COLORS['card'], fg=COLORS['muted'],
            justify='left', anchor='w', wraplength=1050, font=('Sans', 8)
        ).pack(fill='x', padx=5, pady=(0, 7))
        self.register_callback(lambda v=preview_tip: v.set(self.tr('model_classes_preview_tip')))

        preview_box = tk.Frame(cpreview, bg=COLORS['card2'], highlightbackground=COLORS['border'], highlightthickness=1)
        preview_box.pack(fill='both', expand=True, padx=5)
        cols = ('class', 'total', 'train', 'valid', 'test', 'sources')
        tree = ttk.Treeview(preview_box, columns=cols, show='headings', height=8, style='Dark.Treeview')
        preview_cols = [
            ('class', 'model_class', 190, 'w', True),
            ('total', 'total_images', 70, 'center', False),
            ('train', 'train_short', 70, 'center', False),
            ('valid', 'valid_short', 70, 'center', False),
            ('test', 'test_short', 70, 'center', False),
            ('sources', 'source_classes', 80, 'center', False),
        ]
        for col, key, width, anchor, stretch in preview_cols:
            tree.heading(col, text=self.tr(key))
            tree.column(col, width=width, anchor=anchor, stretch=stretch)
            self.register_callback(lambda t=tree, c=col, k=key: t.heading(c, text=self.tr(k)))
        sy = ttk.Scrollbar(preview_box, orient='vertical', command=tree.yview)
        tree.configure(yscrollcommand=sy.set)
        tree.pack(side='left', fill='both', expand=True)
        sy.pack(side='right', fill='y')
        self._train_model_tree = tree

        self._train_model_summary_var = tk.StringVar(value=self.tr('dataset_classes_wait'))
        self._train_model_summary_label = tk.Label(
            cpreview, textvariable=self._train_model_summary_var, bg=COLORS['card'],
            fg=COLORS['cyan'], justify='left', anchor='w', wraplength=1050,
            font=('Sans', 8, 'bold')
        )
        self._train_model_summary_label.pack(fill='x', padx=5, pady=(7, 2))

        c3 = self._card(body, 'train_parameters', 0, 3, 2)
        fields = [
            ('epochs', 'numEpochs', None), ('batch_size', 'batchSize', None), ('learning_rate', 'lr', None),
            ('loss', 'lossName', ['focal', 'ce']), ('focal_gamma', 'focalGamma', None), ('patience', 'patience', None),
            ('validation_metric', 'validMetric', ['valid_loss', 'valid_acc', 'valid_macro']), ('seed', 'seed', None),
            ('pixel_size', 'pixelSize', None),
        ]
        for i, (label, key, combo) in enumerate(fields):
            self._field(c3, label, self.v[key], i // 3, i % 3, combo=combo)
        px_train_tip = tk.StringVar(value=self.tr('pixel_size_train_tip'))
        tk.Label(
            c3, textvariable=px_train_tip, bg=COLORS['card'], fg=COLORS['muted'],
            justify='left', anchor='w', wraplength=480, font=('Sans', 7)
        ).grid(row=3, column=1, columnspan=2, sticky='ew', padx=5, pady=(17, 5))
        self.register_callback(lambda v=px_train_tip: v.set(self.tr('pixel_size_train_tip')))

        cr = self._card(body, 'train_recipe', 0, 4, 2)
        recipe_fields = [
            ('input_normalize', 'inputNormalize', ['imagenet', 'none']), ('input_resize', 'inputResize', ['pad', 'stretch']),
            ('augment', 'augment', ['T', 'F']), ('optimizer', 'optimizer', ['adamw', 'adam']),
            ('weight_decay', 'weightDecay', None), ('scheduler', 'scheduler', ['plateau', 'cosine', 'none']),
            ('warmup_epochs', 'warmupEpochs', None), ('head_lr_mult', 'headLrMult', None),
            ('label_smoothing', 'labelSmoothing', None), ('stochastic_depth', 'stochasticDepth', None),
        ]
        for i, (label, key, combo) in enumerate(recipe_fields):
            self._field(cr, label, self.v[key], i // 3, i % 3, combo=combo)
        recipe_row = tk.Frame(cr, bg=COLORS['card'])
        recipe_row.grid(row=4, column=0, columnspan=3, sticky='ew', padx=5, pady=(6, 2))
        self._button(recipe_row, 'recipe_recommended', lambda: self.apply_training_recipe('recommended')).pack(side='left')
        self._button(recipe_row, 'recipe_legacy', lambda: self.apply_training_recipe('legacy')).pack(side='left', padx=8)
        recipe_tip = tk.StringVar(value=self.tr('train_recipe_tip'))
        tk.Label(
            cr, textvariable=recipe_tip, bg=COLORS['card'], fg=COLORS['muted'],
            justify='left', anchor='w', wraplength=790, font=('Sans', 7)
        ).grid(row=5, column=0, columnspan=3, sticky='ew', padx=5, pady=(2, 4))
        self.register_callback(lambda v=recipe_tip: v.set(self.tr('train_recipe_tip')))

        c4 = self._card(body, 'advanced', 0, 5, 2)
        advanced = [
            ('train_folder', 'trainFolder', None), ('valid_folder', 'validFolder', None), ('test_folder', 'testFolder', None),
            ('workers', 'numWorkers', None), ('resume', 'readModel', ['T', 'F'])
        ]
        for i, (label, key, combo) in enumerate(advanced):
            self._field(c4, label, self.v[key], i // 3, i % 3, combo=combo)
        amp_holder = tk.Frame(c4, bg=COLORS['card'])
        amp_holder.grid(row=1, column=2, sticky='ew', padx=5, pady=5)
        self._text(amp_holder, 'amp', bg=COLORS['card2'], fg=COLORS['muted'], font=('Sans', 8)).pack(anchor='w')
        self._text(amp_holder, 'amp_fixed', bg=COLORS['card'], fg=COLORS['disabled'], font=('Sans', 8)).pack(anchor='w', pady=(4, 0))

        c5 = self._card(body, 'description', 0, 6, 2)
        self.train_desc_var = tk.StringVar(value=self.tr('train_desc'))
        tk.Label(
            c5, textvariable=self.train_desc_var, bg=COLORS['card'], fg=COLORS['text'],
            justify='left', anchor='nw', wraplength=790, font=('Sans', 8)
        ).pack(fill='x')
        self.register_callback(lambda: self.train_desc_var.set(self.tr('train_desc')))

        actions = tk.Frame(body, bg=COLORS['panel'])
        actions.grid(row=7, column=0, columnspan=2, sticky='ew', padx=6, pady=(4, 16))
        self._button(actions, 'save_config', self.save_train_config).pack(side='left')
        self._button(actions, 'load_config', self.load_train_config).pack(side='left', padx=8)
        self._button(actions, 'training_results', self.open_training_results).pack(side='left')
        self.start_button = self._button(actions, 'start_training', self.start_training, kind='primary')
        self.start_button.pack(side='right')

        self._build_execution_panel(right, 'training')
        self.register_callback(self._refresh_training_model_summary)
        self._refresh_training_dataset_classes()

    def _build_classification_page(self):
        body, right = self._page_layout('classification_title', 'classification_subtitle')
        body.grid_columnconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=1)

        c1 = self._card(body, 'dataset', 0, 0)
        self._field(c1, 'dataset_root', self.v['class_input'], 0, browse=lambda: self.pick_dir(self.v['class_input']))
        info = tk.Label(c1, text='dataset/\n  ├─ classe_A/  imagens\n  ├─ classe_B/  imagens\n  └─ ...', bg=COLORS['card'], fg=COLORS['muted'], justify='left', font=('Monospace',8))
        info.grid(row=1, column=0, sticky='w', padx=5, pady=(5,0))

        c2 = self._card(body, 'output', 1, 0)
        self._field(c2, 'output_folder', self.v['class_output_parent'], 0, browse=lambda: self.pick_dir(self.v['class_output_parent']))
        self._field(c2, 'project_name', self.v['class_project'], 1)

        c3 = self._card(body, 'model_folder', 0, 1)
        self._field(c3, 'model_folder_label', self.v['model_folder'], 0, browse=self.pick_model_folder)
        self.model_status_var = tk.StringVar(value=self.tr('detected_wait'))
        self.model_status_label = tk.Label(c3, textvariable=self.model_status_var, bg=COLORS['card'], fg=COLORS['muted'], font=('Sans',9,'bold'))
        self.model_status_label.grid(row=1, column=0, sticky='w', padx=5, pady=(6, 3))
        self._field(c3, 'model_file', self.v['model_file'], 2, readonly=True)
        self._field(c3, 'labels_file', self.v['labels_file'], 3, readonly=True)
        self.register_callback(self._refresh_model_status_language)

        c4 = self._card(body, 'classification_options', 1, 1)
        self._field(c4, 'max_batch', self.v['max_batch'], 0)
        self._field(c4, 'image_mode', self.v['img_mode'], 1, combo=['auto','ln','hardlink','cp'])
        self._field(c4, 'pixel_size', self.v['class_pixel_size'], 2)
        px_class_tip=tk.StringVar(value=self.tr('pixel_size_class_tip'))
        tk.Label(c4,textvariable=px_class_tip,bg=COLORS['card'],fg=COLORS['muted'],
                 wraplength=360,justify='left',anchor='w',font=('Sans',7)).grid(
                     row=3,column=0,sticky='ew',padx=5,pady=(0,5))
        self.register_callback(lambda v=px_class_tip:v.set(self.tr('pixel_size_class_tip')))
        mode_tip=tk.StringVar(value=self.tr('image_mode_tip'))
        tk.Label(c4,textvariable=mode_tip,bg=COLORS['card'],fg=COLORS['muted'],
                 wraplength=360,justify='left',anchor='w',font=('Sans',7)).grid(
                     row=4,column=0,sticky='ew',padx=5,pady=(0,5))
        self.register_callback(lambda v=mode_tip:v.set(self.tr('image_mode_tip')))
        emb = tk.Frame(c4, bg=COLORS['card'])
        emb.grid(row=5, column=0, sticky='ew', padx=5, pady=7)
        self._text(emb, 'embeddings', bg=COLORS['card'], fg=COLORS['muted'], font=('Sans',8)).pack(anchor='w')
        self._text(emb, 'always_enabled', bg=COLORS['card'], fg=COLORS['green'], font=('Sans',9,'bold')).pack(anchor='w', pady=(4,0))

        audit_row = tk.Frame(c4, bg=COLORS['card'])
        audit_row.grid(row=6, column=0, sticky='ew', padx=5, pady=(5,5))
        audit_text=tk.StringVar(value=self.tr('audit_source_labels'))
        audit_cb=tk.Checkbutton(
            audit_row,textvariable=audit_text,variable=self.v['audit_source_labels'],
            bg=COLORS['card'],fg=COLORS['text'],activebackground=COLORS['card'],
            activeforeground=COLORS['text'],selectcolor=COLORS['entry'],
            highlightthickness=0,bd=0,font=('Sans',9,'bold'),cursor='hand2'
        )
        audit_cb.pack(anchor='w')
        audit_tip=tk.StringVar(value=self.tr('audit_source_tip'))
        tk.Label(audit_row,textvariable=audit_tip,bg=COLORS['card'],fg=COLORS['muted'],
                 justify='left',anchor='w',wraplength=360,font=('Sans',7)).pack(anchor='w',padx=(22,0),pady=(2,0))
        self.register_callback(lambda v=audit_text:v.set(self.tr('audit_source_labels')))
        self.register_callback(lambda v=audit_tip:v.set(self.tr('audit_source_tip')))

        cluster_row = tk.Frame(c4, bg=COLORS['card'])
        cluster_row.grid(row=7, column=0, sticky='ew', padx=5, pady=(5,7))
        cluster_text = tk.StringVar(value=self.tr('run_class_clustering'))
        cb = tk.Checkbutton(cluster_row, textvariable=cluster_text, variable=self.v['run_class_clustering'],
                            bg=COLORS['card'], fg=COLORS['text'], activebackground=COLORS['card'],
                            activeforeground=COLORS['text'], selectcolor=COLORS['entry'],
                            highlightthickness=0, bd=0, font=('Sans',9,'bold'), cursor='hand2')
        cb.pack(anchor='w')
        note = tk.StringVar(value=self.tr('cluster_note'))
        tk.Label(cluster_row, textvariable=note, bg=COLORS['card'], fg=COLORS['muted'],
                 justify='left', anchor='w', wraplength=360, font=('Sans',8)).pack(anchor='w', padx=(22,0), pady=(2,0))
        self.register_callback(lambda v=cluster_text: v.set(self.tr('run_class_clustering')))
        self.register_callback(lambda v=note: v.set(self.tr('cluster_note')))

        c5 = self._card(body, 'description', 0, 2, 2)
        self.class_desc_var = tk.StringVar(value=self.tr('class_desc'))
        tk.Label(c5, textvariable=self.class_desc_var, bg=COLORS['card'], fg=COLORS['text'], justify='left', anchor='nw', wraplength=790, font=('Sans',8)).pack(fill='x')
        self.register_callback(lambda: self.class_desc_var.set(self.tr('class_desc')))

        actions = tk.Frame(body, bg=COLORS['panel'])
        actions.grid(row=3, column=0, columnspan=2, sticky='ew', padx=6, pady=(4,16))
        self.start_button = self._button(actions, 'start_classification', self.start_classification, kind='primary')
        self.start_button.pack(side='right')

        self._build_execution_panel(right, 'classification')
        if self.v['model_folder'].get():
            self.detect_model_pair(self.v['model_folder'].get(), fill_pixel_size=True)

    def _build_execution_panel(self, right, task):
        top = tk.Frame(right, bg=COLORS['panel'])
        top.pack(fill='x', padx=14, pady=(14,8))
        self._text(top, 'execution', bg=COLORS['panel'], fg=COLORS['text'], font=('Sans',12,'bold')).pack(side='left')
        self.run_status_var = tk.StringVar(value=self.tr('ready'))
        self.run_status_label = tk.Label(top, textvariable=self.run_status_var, bg=COLORS['card'], fg=COLORS['green'], padx=9, pady=3, font=('Sans',8,'bold'))
        self.run_status_label.pack(side='left', padx=8)
        self.elapsed_var = tk.StringVar(value='00:00:00')
        tk.Label(top, textvariable=self.elapsed_var, bg=COLORS['panel'], fg=COLORS['muted'], font=('Monospace',9)).pack(side='right')

        self._build_stage_bar(right, task)

        ph = tk.Frame(right, bg=COLORS['panel'])
        ph.pack(fill='x', padx=14, pady=(4, 8))
        self._text(ph, 'general_progress', bg=COLORS['panel'], fg=COLORS['text'], font=('Sans',9,'bold')).pack(side='left')
        self.progress_pct_var = tk.StringVar(value='0%')
        tk.Label(ph, textvariable=self.progress_pct_var, bg=COLORS['panel'], fg=COLORS['muted'], font=('Sans',9)).pack(side='right')
        self.progress = ttk.Progressbar(right, maximum=100, style='Blue.Horizontal.TProgressbar')
        self.progress.pack(fill='x', padx=14, pady=(0, 12))

        self.metric_container = tk.Frame(right, bg=COLORS['panel'])
        self.metric_container.pack(fill='x', padx=10, pady=(0,8))
        if task == 'training':
            self.loss_chart = TrainingPlotCard(
                self.metric_container, self, 'loss_history',
                [('train_loss', 'train_loss', COLORS['cyan']),
                 ('valid_loss', 'valid_loss', COLORS['amber'])],
                percent=False)
            self.loss_chart.pack(fill='x', padx=4, pady=(2, 5))
            self.validation_chart = TrainingPlotCard(
                self.metric_container, self, 'validation_history',
                [('valid_acc', 'valid_acc', COLORS['green']),
                 ('valid_macro', 'valid_macro', COLORS['purple'])],
                percent=True)
            self.validation_chart.pack(fill='x', padx=4, pady=(0, 4))
            self._current_train_epoch = 0
        else:
            self.class_metric_vars = {}
            for i, key in enumerate(['current_batch','batches','images_processed','total_images']):
                metric_round = RoundedPanel(self.metric_container, bg=COLORS['card2'], border=COLORS['border'], radius=12, pad=1)
                metric_round.grid(row=i//2, column=i%2, padx=4, pady=4, sticky='nsew')
                self.metric_container.grid_columnconfigure(i%2, weight=1)
                f = metric_round.inner
                title = tk.StringVar(value=self.tr(key))
                value = tk.StringVar(value='—')
                self.class_metric_vars[key] = value
                tk.Label(f, textvariable=title, bg=COLORS['card2'], fg=COLORS['muted'], font=('Sans',8)).pack(anchor='w', padx=9, pady=(7,2))
                tk.Label(f, textvariable=value, bg=COLORS['card2'], fg=COLORS['text'], font=('Sans',12,'bold')).pack(anchor='w', padx=9, pady=(0,8))
                self.register_callback(lambda v=title, k=key: v.set(self.tr(k)))

        log_head = tk.Frame(right, bg=COLORS['panel'])
        log_head.pack(fill='x', padx=14, pady=(4,4))
        self._text(log_head, 'logs', bg=COLORS['panel'], fg=COLORS['text'], font=('Sans',9,'bold')).pack(side='left')
        self._button(log_head, 'clear', self.clear_log).pack(side='right')

        self.live_progress_var = tk.StringVar(value='')
        self.live_progress_label = tk.Label(right, textvariable=self.live_progress_var, bg=COLORS['panel'],
                                            fg=COLORS['cyan'], font=('Monospace',8), anchor='w')
        self.live_progress_label.pack(fill='x', padx=14, pady=(0,4))

        log_round = RoundedPanel(right, bg=COLORS['log_bg'], border=COLORS['border'], radius=14, pad=1)
        log_round.pack(fill='both', expand=True, padx=14, pady=(0,10))
        log_box = log_round.inner
        self.log_text = tk.Text(log_box, bg=COLORS['log_bg'], fg='#e8e8ea', insertbackground='#fff', relief='flat',
                                wrap='word', font=('Monospace',9), height=20, padx=11, pady=9, bd=0,
                                selectbackground=COLORS['nav_active'], highlightthickness=0)
        self.log_text.pack(fill='both', expand=True, padx=3, pady=3)
        self._configure_log_tags()
        self.log_text.configure(state='disabled')

        actions = tk.Frame(right, bg=COLORS['panel'])
        actions.pack(fill='x', padx=14, pady=(0,14))
        self.stop_button = self._button(actions, 'stop', self.stop_process, kind='danger')
        self.stop_button.pack(side='left')
        self.stop_button.configure(state='disabled')
        self._button(actions, 'open_output', self.open_output).pack(side='right')

    def _build_stage_bar(self, parent, task):
        if task == 'classification':
            self.stage_keys = ['stage_loading','stage_features','stage_tsne','stage_results','stage_clustering']
        else:
            self.stage_keys = ['train_stage_prepare','train_stage_epochs','train_stage_test','train_stage_save']
        self.stage_labels = []
        self.stage_task = task
        frame = tk.Frame(parent, bg=COLORS['panel'])
        frame.pack(fill='x', padx=10, pady=(4,8))
        for i, key in enumerate(self.stage_keys):
            cell = tk.Frame(frame, bg=COLORS['panel'])
            cell.pack(side='left', fill='x', expand=True)
            dot = tk.Label(cell, text=str(i+1), bg=COLORS['card'], fg=COLORS['muted'], width=2, height=1, font=('Sans',8,'bold'))
            dot.pack()
            var = tk.StringVar(value=self.tr(key))
            lab = tk.Label(cell, textvariable=var, bg=COLORS['panel'], fg=COLORS['muted'], font=('Sans',7), wraplength=76)
            lab.pack(pady=(3,0))
            self.stage_labels.append((dot, lab, var, key))
            self.register_callback(lambda v=var, k=key: v.set(self.tr(k)))
        self.stage_index = -1

    def _set_stage(self, index):
        if not hasattr(self, 'stage_labels'):
            return
        self.stage_index = index
        for i, (dot, lab, _, _) in enumerate(self.stage_labels):
            if i < index:
                dot.configure(bg=COLORS['green'], fg='#06120a', text='✓')
                lab.configure(fg=COLORS['green'])
            elif i == index:
                dot.configure(bg=COLORS['blue'], fg=readable_text_on(COLORS['blue']), text=str(i+1))
                lab.configure(fg=COLORS['text'])
            else:
                dot.configure(bg=COLORS['card'], fg=COLORS['muted'], text=str(i+1))
                lab.configure(fg=COLORS['muted'])

    def _complete_stages(self):
        if not hasattr(self, 'stage_labels'):
            return
        for dot, lab, _, _ in self.stage_labels:
            dot.configure(bg=COLORS['green'], fg='#06120a', text='✓')
            lab.configure(fg=COLORS['green'])
        self.stage_index = len(self.stage_labels)

    def _skip_last_stage(self):
        if not hasattr(self, 'stage_labels') or not self.stage_labels:
            return
        for i, (dot, lab, _, _) in enumerate(self.stage_labels):
            if i < len(self.stage_labels)-1:
                dot.configure(bg=COLORS['green'], fg='#06120a', text='✓')
                lab.configure(fg=COLORS['green'])
            else:
                dot.configure(bg=COLORS['card'], fg=COLORS['disabled'], text='—')
                lab.configure(fg=COLORS['disabled'])

    def _refresh_dynamic_texts(self):
        if hasattr(self, 'run_status_var'):
            if self.proc is not None:
                self.run_status_var.set(self.tr('running'))
            elif getattr(self, '_last_result', None) == 'completed':
                self.run_status_var.set(self.tr('completed'))
            else:
                self.run_status_var.set(self.tr('ready'))
        self.system_status_var.set(self.tr('running') if self.proc is not None else self.tr('ready'))
        self._refresh_model_status_language()

    def _refresh_model_status_language(self):
        if not hasattr(self, 'model_status_var'):
            return
        label = getattr(self, 'model_status_label', None)
        label_alive = False
        if label is not None:
            try:
                label_alive = bool(label.winfo_exists())
            except Exception:
                label_alive = False
        if self.v['model_file'].get() and self.v['labels_file'].get():
            self.model_status_var.set('✓ ' + self.tr('detected_ok'))
            if label_alive:
                label.configure(fg=COLORS['green'])
        elif self.v['model_folder'].get():
            self.model_status_var.set('! ' + self.tr('detected_error'))
            if label_alive:
                label.configure(fg=COLORS['amber'])
        else:
            self.model_status_var.set(self.tr('detected_wait'))
            if label_alive:
                label.configure(fg=COLORS['muted'])

    def pick_dir(self, var):
        initial = var.get() or str(self.project_root)
        p = filedialog.askdirectory(initialdir=initial)
        if p:
            var.set(p)

    def pick_train_dataset(self):
        initial=self.v['train_dataset'].get() or str(self.project_root)
        p=filedialog.askdirectory(initialdir=initial)
        if not p:
            return
        self.v['train_dataset'].set(p)
        # Datasets exported by IAT carry their calibration in dataset_config.json.
        # Reuse it as a convenience default; the field remains editable.
        cfg_path=Path(p)/'dataset_config.json'
        if cfg_path.is_file():
            try:
                cfg=json.loads(cfg_path.read_text(encoding='utf-8'))
                raw=cfg.get('pixel_size_um_per_px')
                if raw is not None:
                    self.v['pixelSize'].set(format_pixel_size(raw))
            except Exception:
                pass
        self._refresh_training_dataset_classes()

    def pick_model_folder(self):
        initial = self.v['model_folder'].get() or str(Path.home())
        p = filedialog.askdirectory(initialdir=initial)
        if p:
            self.v['model_folder'].set(p)
            self.detect_model_pair(p, fill_pixel_size=True)

    def detect_model_pair(self, folder, fill_pixel_size=False):
        """Find the model_/labels_ pair in *folder*.

        ``fill_pixel_size`` (only when the user picks the folder) copies the
        training calibration into the classification field as a convenience,
        and only while that field still holds the default value: the pixel
        size belongs to the images being classified, so a value typed by the
        user is never replaced (starting a classification re-checks the pair
        with ``fill_pixel_size=False``).
        """
        self.v['model_file'].set('')
        self.v['labels_file'].set('')
        path = Path(folder)
        if not path.is_dir():
            self._refresh_model_status_language()
            return None
        models = [p for p in path.iterdir() if p.is_file() and p.name.startswith('model_') and not p.name.endswith('_last_epoch')]
        labels = [p for p in path.iterdir() if p.is_file() and p.name.startswith('labels_') and p.suffix.lower() == '.json']
        model_by_suffix = {p.name[len('model_'):]: p for p in models}
        label_by_suffix = {p.stem[len('labels_'):]: p for p in labels}
        matches = sorted(set(model_by_suffix) & set(label_by_suffix))
        if len(matches) == 1:
            suffix = matches[0]
            self.v['model_file'].set(str(model_by_suffix[suffix]))
            self.v['labels_file'].set(str(label_by_suffix[suffix]))
            # Reuse calibration metadata from training when available.  This is
            # only a convenience default; the user can always edit it before
            # running classification.
            cfg_path=path/f'training_config_{suffix}.json'
            if fill_pixel_size and cfg_path.is_file() and self._class_pixel_size_is_default():
                try:
                    cfg=json.loads(cfg_path.read_text(encoding='utf-8'))
                    self.v['class_pixel_size'].set(
                        format_pixel_size(cfg.get('pixelSize', self.v['class_pixel_size'].get()))
                    )
                except Exception:
                    pass
        # Never guess a model/labels pair from file counts alone.  The suffix
        # must match exactly so class indices cannot be paired with the wrong
        # semantic label map.
        self._refresh_model_status_language()
        return self.v['model_file'].get(), self.v['labels_file'].get()

    def _class_pixel_size_is_default(self):
        raw = str(self.v['class_pixel_size'].get()).strip()
        if not raw:
            return True
        try:
            return abs(parse_pixel_size(raw) - DEFAULT_PIXEL_SIZE_UM_PER_PX) < 1e-9
        except ValueError:
            return False

    @staticmethod
    def _recipe_var_text(key, value):
        if key == 'augment':
            if isinstance(value, str):
                return 'T' if value.strip().lower() in {'1', 'true', 't', 'yes', 'y', 'on'} else 'F'
            return 'T' if bool(value) else 'F'
        return str(value)

    def apply_training_recipe(self, name):
        """Fill the recipe fields with a preset ('recommended' or 'legacy')."""
        preset = PRESETS.get(name, DEFAULT_RECIPE)
        for key in RECIPE_KEYS:
            if key in self.v:
                self.v[key].set(self._recipe_var_text(key, preset[key]))

    def _train_recipe_config(self):
        return recipe_from_config({key: self.v[key].get() for key in RECIPE_KEYS if key in self.v})

    def _train_config(self):
        try:
            return {
                'lr': float(self.v['lr'].get()),
                'lossName': self.v['lossName'].get(),
                'focalGamma': float(self.v['focalGamma'].get()),
                'trainFolder': self.v['trainFolder'].get().strip(),
                'validFolder': self.v['validFolder'].get().strip(),
                'testFolder': self.v['testFolder'].get().strip(),
                'readModel': self.v['readModel'].get(),
                'numEpochs': int(self.v['numEpochs'].get()),
                'validMetric': self.v['validMetric'].get(),
                'patience': int(self.v['patience'].get()),
                # Macro metrics are always averaged over the real classes
                # (train.py); the old "macro divisor" field was removed.
                'numClasses': -1,
                'numWorkers': int(self.v['numWorkers'].get()),
                'seed': int(self.v['seed'].get()),
                # 'auto' lets train.py pick the batch from the free VRAM.
                'batchSize': ('auto' if self.v['batchSize'].get().strip().lower() == 'auto'
                              else int(self.v['batchSize'].get())),
                'pixelSize': parse_pixel_size(self.v['pixelSize'].get()),
                **self._train_recipe_config(),
            }
        except ValueError as e:
            raise ValueError(self.tr('invalid_input')) from e

    def save_train_config(self):
        try:
            cfg = self._train_config()
        except ValueError as e:
            messagebox.showerror(self.tr('error'), str(e)); return
        p = filedialog.asksaveasfilename(initialdir=str(self.input_dir), initialfile='train_params.json', defaultextension='.json', filetypes=[('JSON','*.json')])
        if p:
            Path(p).write_text(json.dumps(cfg, indent=2) + '\n')
            messagebox.showinfo(self.tr('success'), self.tr('config_saved'))

    def load_train_config(self):
        p = filedialog.askopenfilename(initialdir=str(self.input_dir), filetypes=[('JSON','*.json')])
        if not p: return
        try:
            cfg = json.loads(Path(p).read_text())
            for key in ['lr','lossName','focalGamma','trainFolder','validFolder','testFolder','readModel','numEpochs','validMetric','patience','numClasses','numWorkers','seed','batchSize','pixelSize',*RECIPE_KEYS]:
                if key in cfg:
                    self.v[key].set(self._recipe_var_text(key, cfg[key]) if key in RECIPE_KEYS else str(cfg[key]))
            messagebox.showinfo(self.tr('success'), self.tr('config_loaded'))
        except Exception as e:
            messagebox.showerror(self.tr('error'), str(e))


    _ansi_re = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')


    def show_settings(self):
        messagebox.showinfo(self.tr('settings'), self.tr('settings_text').format(python=sys.executable, src=self.src_dir, root=self.project_root))

    def show_about(self):
        logo = self._brand_images.get('logo_global')
        if logo is None:
            messagebox.showinfo(self.tr('about'), self.tr('about_text'))
            return

        win = tk.Toplevel(self)
        win.title(self.tr('about'))
        win.configure(bg=COLORS['panel'])
        win.resizable(False, False)
        try:
            if getattr(self, '_window_icon_photo', None) is not None:
                win.iconphoto(True, self._window_icon_photo)
        except Exception:
            pass

        card = tk.Frame(win, bg=COLORS['panel'])
        card.pack(fill='both', expand=True, padx=24, pady=22)
        tk.Label(card, image=logo, bg=COLORS['panel']).pack(pady=(0, 14))
        tk.Label(
            card,
            text=self.tr('about_text'),
            bg=COLORS['panel'],
            fg=COLORS['text'],
            justify='left',
            wraplength=520,
            font=('Sans', 10),
        ).pack(anchor='w')
        self._button(card, 'close', win.destroy).pack(anchor='e', pady=(18, 0)) if 'close' in T[self.lang] else tk.Button(
            card, text='OK', command=win.destroy, bg=COLORS['card2'], fg=COLORS['text'],
            relief='flat', bd=0, padx=18, pady=7
        ).pack(anchor='e', pady=(18,0))
        win.transient(self)
        win.grab_set()
        win.update_idletasks()
        x = self.winfo_rootx() + max(20, (self.winfo_width() - win.winfo_reqwidth()) // 2)
        y = self.winfo_rooty() + max(20, (self.winfo_height() - win.winfo_reqheight()) // 2)
        win.geometry(f'+{x}+{y}')

    def on_close(self):
        if self.proc is not None:
            if not messagebox.askyesno(self.tr('stop'), self.tr('confirm_stop')):
                return
            try: self.proc.terminate()
            except Exception: pass
        self.destroy()


def main():
    # The IAT integrated cycle classifies batches inside this process.
    from hw_resources import raise_open_file_limit
    raise_open_file_limit()
    app = ILTApp()
    # Optional start page (used by the web interface to open the full IAT).
    start = os.environ.get('ILT_START_TASK', '').strip().lower()
    if start in ('classification', 'iat'):
        app.show_task(start)
    app.mainloop()


def _run_frozen_worker():
    """Route background jobs when the application is packaged as one EXE."""
    if len(sys.argv) < 3 or sys.argv[1] != '--ilt-worker':
        return False
    role=sys.argv[2]
    args=sys.argv[3:]

    # PyInstaller --windowed sets sys.stdout/sys.stderr to None on Windows.
    # Popen still supplies a real redirected STDOUT handle to worker children;
    # reconnect Python to that handle so the GUI can parse progress normally.
    if os.name=='nt' and sys.stdout is None:
        try:
            import ctypes, msvcrt
            handle=ctypes.windll.kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
            fd=msvcrt.open_osfhandle(handle,os.O_WRONLY)
            sys.stdout=os.fdopen(fd,'w',buffering=1,encoding='utf-8',errors='replace',closefd=False)
            sys.stderr=sys.stdout
        except Exception:
            pass
    try:
        if hasattr(sys.stdout,'reconfigure'):
            sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    if role=='classification':
        from main import main as worker_main
        worker_main(['main.py',*args])
    elif role=='training':
        from train import main as worker_main
        worker_main(['train.py',*args])
    elif role=='clustering':
        import class_clustering
        old_argv=sys.argv
        try:
            sys.argv=['class_clustering.py',*args]
            class_clustering.main()
        finally:
            sys.argv=old_argv
    else:
        raise SystemExit(f'Unknown ILT worker: {role}')
    return True


def _run_self_test():
    """Import the runtime stack without opening Tk; used by platform build scripts."""
    if '--ilt-self-test' not in sys.argv:
        return False
    checks = [
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('opencv-python', 'cv2'),
        ('matplotlib', 'matplotlib'),
        ('scipy', 'scipy'),
        ('scikit-learn', 'sklearn'),
        ('torch', 'torch'),
        ('torchvision', 'torchvision'),
        ('openTSNE', 'openTSNE'),
        ('Pillow', 'PIL'),
        ('psutil', 'psutil'),
        ('tqdm', 'tqdm'),
        ('UMAP', 'umap'),
        ('PaCMAP', 'pacmap'),
        ('ILT clustering core', 'ilt_clustering'),
        ('ILT contracts', 'ilt_contracts'),
    ]
    failures = []
    try:
        from sklearn.cluster import HDBSCAN as _ILT_HDBSCAN  # noqa: F401
    except Exception as exc:
        failures.append(f'sklearn HDBSCAN: {exc}')
    for label, module_name in checks:
        try:
            __import__(module_name)
        except Exception as exc:
            failures.append(f'{label}: {exc}')
    if failures:
        try:
            Path(application_dir(), 'ILT_SELF_TEST_FAILED.txt').write_text(
                'Falhas no autoteste do executável:\n' + '\n'.join(failures),
                encoding='utf-8'
            )
        except Exception:
            pass
        raise SystemExit(5)
    raise SystemExit(0)


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    _run_self_test()
    if not _run_frozen_worker():
        main()
